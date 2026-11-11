//! Python environment binding for online policy evaluation.
//!
//! The extension owns one authoritative [`Game`] per [`Env`]. `reset` starts a
//! fresh seeded game, encodes its first pending decision with the frozen v1
//! encoder, and copies the visible features into independent float32 NumPy
//! arrays. Python never borrows authoritative game memory.
//!
//! The crate is built as the `mahjong_eval._native` extension by the online
//! package and is deliberately outside the root Cargo workspace, so ordinary
//! core, collector, and neural builds do not resolve or link Python.

#![forbid(unsafe_code)]

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use sichuan_mahjong_neural::encoding::{encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::{Decision as GameDecision, DecisionStage, Game};

/// A decision identity is `(reset generation << 32) | sequence`. The
/// generation changes on every [`Env::reset`] and the first pending decision
/// of a reset uses sequence 1. T2 extends the sequence for later decisions;
/// including the generation here already lets stale requests be rejected.
const DECISION_ID_FIRST_SEQUENCE: u32 = 1;

/// One pending decision with an owned, player-visible numeric snapshot.
///
/// `state` is a C-contiguous `float32` array of width [`STATE_WIDTH`].
/// `candidate_features` is a C-contiguous `float32` array with one row per
/// canonical candidate and width [`CANDIDATE_WIDTH`]. Both arrays own their
/// data independently of the game; mutating them cannot change the game.
#[pyclass(module = "mahjong_eval._native", frozen)]
struct Decision {
    /// Reset generation and decision sequence, packed into one nonnegative ID.
    #[pyo3(get)]
    decision_id: u64,
    /// Deciding seat in `0..=3`.
    #[pyo3(get)]
    actor_seat: u8,
    /// Frozen v1 stage code in `0..=4`.
    #[pyo3(get)]
    stage: u8,
    /// Number of canonical candidate rows.
    #[pyo3(get)]
    candidate_count: usize,
    state: Py<PyArray1<f32>>,
    candidate_features: Py<PyArray2<f32>>,
}

#[pymethods]
impl Decision {
    /// Encoded state of shape `(STATE_WIDTH,)`, dtype `float32`.
    #[getter]
    fn state(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.state.clone_ref(py)
    }

    /// Candidate features of shape `(candidate_count, CANDIDATE_WIDTH)`,
    /// dtype `float32`.
    #[getter]
    fn candidate_features(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.candidate_features.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Decision(decision_id={}, actor_seat={}, stage={}, candidate_count={})",
            self.decision_id, self.actor_seat, self.stage, self.candidate_count
        )
    }
}

/// One environment handle owning one game.
///
/// `reset(seed)` returns the first pending decision. Transactional `step` and
/// terminal results are added by T2; this slice exposes only the first-decision
/// contract used to verify the build and encoding handoff.
#[pyclass(module = "mahjong_eval._native")]
struct Env {
    game: Option<Game>,
    decision_limit: u64,
    generation: u32,
}

#[pymethods]
impl Env {
    /// Creates a handle. `decision_limit` is retained for T2's per-game safety
    /// bound and defaults to the Rust runner's limit.
    #[new]
    #[pyo3(signature = (decision_limit = DEFAULT_DECISION_LIMIT))]
    fn new(decision_limit: u64) -> Self {
        Self {
            game: None,
            decision_limit,
            generation: 0,
        }
    }

    /// Per-game decision safety bound retained for the full lifecycle.
    #[getter]
    fn decision_limit(&self) -> u64 {
        self.decision_limit
    }

    /// Starts a fresh seeded game and returns its first pending decision.
    ///
    /// A previous game owned by this handle is discarded. The reset generation
    /// advances so decision identities from an earlier game never validate
    /// against the new one.
    #[pyo3(signature = (seed))]
    fn reset(&mut self, py: Python<'_>, seed: u64) -> PyResult<Decision> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("environment reset generation exhausted"))?;

        let mut game = Game::new(seed);
        let first = {
            let pending = game.next_decision().ok_or_else(|| {
                PyRuntimeError::new_err("fresh game produced no opening decision")
            })?;
            snapshot(
                py,
                &pending,
                decision_id(generation, DECISION_ID_FIRST_SEQUENCE),
            )?
        };

        // Replace (and thereby drop) any prior game only after the new opening
        // snapshot succeeds, keeping a failed reset from discarding old state.
        let previous = self.game.replace(game);
        drop(previous);
        self.generation = generation;
        Ok(first)
    }

    fn __repr__(&self) -> String {
        format!(
            "Env(started={}, decision_limit={})",
            self.game.is_some(),
            self.decision_limit
        )
    }
}

fn decision_id(generation: u32, sequence: u32) -> u64 {
    (u64::from(generation) << 32) | u64::from(sequence)
}

/// Frozen v1 wire stage code. The encoding fixture compares this value with
/// the encoder's state features, so an order change cannot pass silently.
fn stage_code(stage: DecisionStage) -> u8 {
    match stage {
        DecisionStage::Exchange => 0,
        DecisionStage::MissingSuit => 1,
        DecisionStage::Turn => 2,
        DecisionStage::DiscardResponse => 3,
        DecisionStage::AddedKongResponse => 4,
    }
}

fn snapshot(py: Python<'_>, decision: &GameDecision<'_>, decision_id: u64) -> PyResult<Decision> {
    let encoded = encode(decision);
    let state = state_array(py, &encoded.state.flatten());

    let candidate_count = encoded.candidates.len();
    let mut features = Vec::with_capacity(candidate_count * ACTION_WIDTH);
    for candidate in &encoded.candidates {
        features.extend_from_slice(&candidate.features.flatten());
    }
    let candidate_features = candidate_array(py, candidate_count, &features);

    Ok(Decision {
        decision_id,
        actor_seat: decision.seat.index() as u8,
        stage: stage_code(decision.observation.stage),
        candidate_count,
        state,
        candidate_features,
    })
}

/// Copies state values into a NumPy-owned array. The Python array owns its
/// allocation (`OWNDATA`) and never references game or encoder memory.
fn state_array(py: Python<'_>, values: &[f32]) -> Py<PyArray1<f32>> {
    let array = PyArray1::<f32>::zeros(py, [values.len()], false);
    {
        let mut writable = array.readwrite();
        writable
            .as_slice_mut()
            .expect("new state array is contiguous")
            .copy_from_slice(values);
    }
    array.unbind()
}

/// Copies canonical candidate rows into a NumPy-owned C-contiguous array.
fn candidate_array(py: Python<'_>, rows: usize, values: &[f32]) -> Py<PyArray2<f32>> {
    let array = PyArray2::<f32>::zeros(py, [rows, ACTION_WIDTH], false);
    {
        let mut writable = array.readwrite();
        writable
            .as_slice_mut()
            .expect("new candidate array is contiguous")
            .copy_from_slice(values);
    }
    array.unbind()
}

/// Native module loaded as `mahjong_eval._native`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Env>()?;
    module.add_class::<Decision>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("ENCODING_VERSION", ENCODING_VERSION)?;
    module.add("STATE_WIDTH", STATE_WIDTH)?;
    module.add("CANDIDATE_WIDTH", ACTION_WIDTH)?;
    module.add("DEFAULT_DECISION_LIMIT", DEFAULT_DECISION_LIMIT)?;
    Ok(())
}
