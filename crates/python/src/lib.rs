//! Python environment binding for online policy evaluation.
//!
//! The extension owns one authoritative [`Game`] per [`Env`]. A reset starts a
//! fresh seeded game and each accepted step advances the same game until it
//! returns the next pending decision or the terminal settlement. The encoder
//! snapshots only player-visible features into independent float32 NumPy
//! arrays; Python never borrows authoritative game memory.
//!
//! The crate is built as the `mahjong_eval._native` extension by the online
//! package and is deliberately outside the root Cargo workspace, so ordinary
//! core, collector, and neural builds do not resolve or link Python.

#![forbid(unsafe_code)]

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use sichuan_mahjong_neural::collection::{
    build_info, environment_info, seed_split as rust_seed_split, split_bucket as rust_split_bucket,
    stage_id, Seed,
};
use sichuan_mahjong_neural::encoding::{
    canonicalize, encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH,
};
use sichuan_mahjong_sim::policy::{HeuristicCache, HeuristicPolicy, Policy, DEFAULT_BUDGET_BYTES};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::Game;

mod env;
mod snapshot;

use env::{limit_error, make_decision_id, Env};
use snapshot::{action_dict, set_terminal_fields, Decision, Terminal};

create_exception!(mahjong_eval._native, MahjongEvalError, PyException);
create_exception!(
    mahjong_eval._native,
    EnvironmentStateError,
    MahjongEvalError
);
create_exception!(mahjong_eval._native, StaleDecisionError, MahjongEvalError);
create_exception!(mahjong_eval._native, CandidateIndexError, MahjongEvalError);
create_exception!(mahjong_eval._native, DecisionLimitError, MahjongEvalError);
create_exception!(mahjong_eval._native, GameRuleError, MahjongEvalError);

/// Decision identity is `(reset generation << 32) | sequence`.
///
/// The generation changes on every reset and the first pending decision of a
/// game uses sequence 1. A request carrying any other identity is rejected
/// before the authoritative game is touched.
const DECISION_ID_FIRST_SEQUENCE: u32 = 1;

/// Explicit heuristic-cache context for native teacher choices.
///
/// The cache stores derived masked suit tables. It never changes a teacher
/// action; one cache may be shared by several environment handles or native
/// replay calls to reuse those tables.
#[pyclass(module = "mahjong_eval._native", frozen)]
struct TeacherCache {
    inner: HeuristicCache,
}

#[pymethods]
impl TeacherCache {
    /// Creates a context with a byte budget. Zero disables caching.
    #[new]
    #[pyo3(signature = (budget_bytes = DEFAULT_BUDGET_BYTES))]
    fn new(budget_bytes: usize) -> PyResult<Self> {
        HeuristicCache::new(budget_bytes)
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Configured byte budget; zero means disabled.
    #[getter]
    fn budget_bytes(&self) -> usize {
        self.inner.status().budget_bytes
    }

    /// Snapshot of availability, allocation, and lookup counters.
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let status = self.inner.status();
        let dict = PyDict::new(py);
        dict.set_item("state", cache_state_name(status.state))?;
        dict.set_item("budget_bytes", status.budget_bytes)?;
        dict.set_item("context_overhead_bytes", status.context_overhead_bytes)?;
        dict.set_item("allocated_bytes", status.allocated_bytes)?;
        dict.set_item("capacity_entries", status.capacity_entries)?;
        dict.set_item("occupied_entries", status.occupied_entries)?;
        dict.set_item("evictions", status.evictions)?;
        dict.set_item("hits", status.hits)?;
        dict.set_item("misses", status.misses)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        let status = self.inner.status();
        format!(
            "TeacherCache(state={}, budget_bytes={}, hits={}, misses={})",
            cache_state_name(status.state),
            status.budget_bytes,
            status.hits,
            status.misses
        )
    }
}

fn cache_state_name(state: sichuan_mahjong_sim::CacheState) -> &'static str {
    match state {
        sichuan_mahjong_sim::CacheState::Disabled => "disabled",
        sichuan_mahjong_sim::CacheState::Unallocated => "unallocated",
        sichuan_mahjong_sim::CacheState::Ready => "ready",
        sichuan_mahjong_sim::CacheState::AllocationFailed => "allocation_failed",
    }
}

/// Replays one seeded game with the native heuristic teacher on every seat.
///
/// The returned mapping is JSON-compatible: it records each semantic action
/// (kind and wire parameters) with its canonical local index, plus the accepted
/// count and terminal result. Python teacher drivers use the same `Game`,
/// teacher, encoder, and optional cache, so their traces can be compared
/// directly with this native execution.
#[pyfunction]
#[pyo3(signature = (seed, cache = None, decision_limit = DEFAULT_DECISION_LIMIT))]
fn native_teacher_replay(
    py: Python<'_>,
    seed: u64,
    cache: Option<Py<TeacherCache>>,
    decision_limit: u64,
) -> PyResult<Py<PyAny>> {
    let cache = match cache {
        Some(cache) => cache.borrow(py).inner.clone(),
        None => HeuristicCache::default(),
    };
    let mut teachers: [HeuristicPolicy; 4] =
        std::array::from_fn(|_| HeuristicPolicy::with_cache(cache.clone()));
    let mut game = Game::new(seed);
    let trace = PyList::empty(py);
    let mut accepted = 0u64;

    while let Some(decision) = game.next_decision() {
        let sequence = u32::try_from(accepted + 1).unwrap_or(u32::MAX);
        if accepted >= decision_limit {
            return Err(limit_error(
                seed,
                decision_limit,
                accepted,
                Some(make_decision_id(1, sequence)),
            ));
        }
        let seat = decision.seat;
        let action = canonicalize(teachers[seat.index()].choose(&decision));
        let encoded = encode(&decision);
        let index = encoded.target_index(&action).map_err(|error| {
            GameRuleError::new_err(format!(
                "native teacher action {action:?} is not a candidate at decision \
                 {accepted} (seed {seed}): {error}"
            ))
        })?;
        let entry = PyDict::new(py);
        entry.set_item("decision_id", make_decision_id(1, sequence))?;
        entry.set_item("actor_seat", decision.seat.index())?;
        entry.set_item("stage", stage_id(decision.observation.stage))?;
        entry.set_item("candidate_count", encoded.candidates.len())?;
        entry.set_item("teacher_index", index)?;
        entry.set_item("action", action_dict(py, encoded.candidates[index].action)?)?;
        trace.append(entry)?;

        game.submit(seat, encoded.candidates[index].action)
            .map_err(|error| {
                GameRuleError::new_err(format!(
                    "native teacher candidate {index} rejected at decision {accepted} \
                     (seed {seed}): {error}"
                ))
            })?;
        accepted += 1;
    }

    let result = game
        .result()
        .ok_or_else(|| EnvironmentStateError::new_err("terminal game has no result"))?;
    let root = PyDict::new(py);
    root.set_item("format", "mahjong-eval-native-teacher-replay-v1")?;
    root.set_item("seed", seed)?;
    root.set_item("decision_limit", decision_limit)?;
    root.set_item("trace", trace)?;
    set_terminal_fields(&root, accepted, result)?;
    Ok(root.into_any().unbind())
}

/// Frozen seed split bucket in `0..100`, matching the Rust collection API.
#[pyfunction]
fn split_bucket(seed: u64) -> u8 {
    rust_split_bucket(Seed::from_u64(seed))
}

/// Frozen seed split name: `train`, `validation`, or `test`.
#[pyfunction]
fn seed_split(seed: u64) -> &'static str {
    rust_seed_split(Seed::from_u64(seed)).as_str()
}

/// Compile-time source identity for the executing extension.
///
/// `extension` covers this binding crate plus the simulator core and neural
/// encoder source trees. `neural` is that crate's own collection build
/// provenance, retained separately so it cannot be mistaken for the complete
/// extension-scoped identity. The content hash is computed from build-time
/// files, not from the caller's current directory.
#[pyfunction]
fn compiled_source(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let root = PyDict::new(py);
    root.set_item("format", "mahjong-eval-compiled-source-v1")?;
    root.set_item("extension", extension_build_dict(py)?)?;
    root.set_item("neural", neural_build_dict(py)?)?;
    root.set_item("environment", environment_dict(py)?)?;
    root.set_item("encoding_version", ENCODING_VERSION)?;
    Ok(root.into_any().unbind())
}

fn optional_env(value: Option<&'static str>) -> Option<&'static str> {
    value.filter(|value| !value.is_empty())
}

fn extension_build_dict(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("package_version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item("git_commit", optional_env(option_env!("EXT_GIT_COMMIT")))?;
    dict.set_item(
        "git_available",
        option_env!("EXT_GIT_AVAILABLE") == Some("1"),
    )?;
    dict.set_item("git_dirty", option_env!("EXT_GIT_DIRTY") != Some("0"))?;
    dict.set_item(
        "source_sha256",
        option_env!("EXT_SOURCE_SHA256").unwrap_or(""),
    )?;
    dict.set_item(
        "source_identity",
        option_env!("EXT_SOURCE_IDENTITY").unwrap_or(""),
    )?;
    let scope: Vec<&str> = option_env!("EXT_SOURCE_SCOPE")
        .unwrap_or("")
        .split('+')
        .filter(|value| !value.is_empty())
        .collect();
    dict.set_item("source_scope", scope)?;
    let files: Vec<&str> = option_env!("EXT_SOURCE_FILES")
        .unwrap_or("")
        .split(';')
        .filter(|value| !value.is_empty())
        .collect();
    dict.set_item("source_files", files)?;
    dict.set_item(
        "source_complete",
        option_env!("EXT_SOURCE_COMPLETE") == Some("1"),
    )?;
    dict.set_item(
        "rustc_version",
        optional_env(option_env!("EXT_RUSTC_VERSION")),
    )?;
    dict.set_item(
        "profile",
        option_env!("EXT_BUILD_PROFILE").unwrap_or("unknown"),
    )?;
    dict.set_item(
        "opt_level",
        option_env!("EXT_BUILD_OPT_LEVEL").unwrap_or("unknown"),
    )?;
    dict.set_item(
        "target",
        option_env!("EXT_BUILD_TARGET").unwrap_or("unknown"),
    )?;
    dict.set_item("debug_assertions", cfg!(debug_assertions))?;
    Ok(dict.unbind())
}

fn neural_build_dict(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = build_info();
    let dict = PyDict::new(py);
    dict.set_item("package_version", info.package_version)?;
    dict.set_item("git_commit", info.git_commit)?;
    dict.set_item("git_available", info.git_available)?;
    dict.set_item("git_dirty", info.git_dirty)?;
    dict.set_item("rustc_version", info.rustc_version)?;
    dict.set_item("profile", info.profile)?;
    dict.set_item("opt_level", info.opt_level)?;
    dict.set_item("target", info.target)?;
    dict.set_item("debug_assertions", info.debug_assertions)?;
    Ok(dict.unbind())
}

fn environment_dict(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = environment_info();
    let dict = PyDict::new(py);
    dict.set_item("os", info.os)?;
    dict.set_item("arch", info.arch)?;
    dict.set_item("family", info.family)?;
    dict.set_item("available_parallelism", info.available_parallelism)?;
    dict.set_item("pointer_width", info.pointer_width)?;
    Ok(dict.unbind())
}

/// Native module loaded as `mahjong_eval._native`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Env>()?;
    module.add_class::<Decision>()?;
    module.add_class::<Terminal>()?;
    module.add_class::<TeacherCache>()?;
    module.add_function(wrap_pyfunction!(native_teacher_replay, module)?)?;
    module.add_function(wrap_pyfunction!(split_bucket, module)?)?;
    module.add_function(wrap_pyfunction!(seed_split, module)?)?;
    module.add_function(wrap_pyfunction!(compiled_source, module)?)?;
    module.add(
        "MahjongEvalError",
        module.py().get_type::<MahjongEvalError>(),
    )?;
    module.add(
        "EnvironmentStateError",
        module.py().get_type::<EnvironmentStateError>(),
    )?;
    module.add(
        "StaleDecisionError",
        module.py().get_type::<StaleDecisionError>(),
    )?;
    module.add(
        "CandidateIndexError",
        module.py().get_type::<CandidateIndexError>(),
    )?;
    module.add(
        "DecisionLimitError",
        module.py().get_type::<DecisionLimitError>(),
    )?;
    module.add("GameRuleError", module.py().get_type::<GameRuleError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("ENCODING_VERSION", ENCODING_VERSION)?;
    module.add("STATE_WIDTH", STATE_WIDTH)?;
    module.add("CANDIDATE_WIDTH", ACTION_WIDTH)?;
    module.add("DEFAULT_DECISION_LIMIT", DEFAULT_DECISION_LIMIT)?;
    module.add("DEFAULT_TEACHER_CACHE_BYTES", DEFAULT_BUDGET_BYTES)?;
    Ok(())
}
