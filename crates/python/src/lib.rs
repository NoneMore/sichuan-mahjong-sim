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

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use sichuan_mahjong_neural::encoding::{encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::{
    Action, Decision as GameDecision, DecisionStage, Game, GameResult, Seat,
};

create_exception!(_native, MahjongEvalError, PyException);
create_exception!(_native, EnvironmentStateError, MahjongEvalError);
create_exception!(_native, StaleDecisionError, MahjongEvalError);
create_exception!(_native, CandidateIndexError, MahjongEvalError);
create_exception!(_native, DecisionLimitError, MahjongEvalError);
create_exception!(_native, GameRuleError, MahjongEvalError);

/// Decision identity is `(reset generation << 32) | sequence`.
///
/// The generation changes on every reset and the first pending decision of a
/// game uses sequence 1. A request carrying any other identity is rejected
/// before the authoritative game is touched.
const DECISION_ID_FIRST_SEQUENCE: u32 = 1;

/// One pending decision with an owned, player-visible numeric snapshot and the
/// canonical semantic actions used to resolve local indices.
struct PendingDecision {
    decision_id: u64,
    actor_seat: u8,
    stage: u8,
    state: Vec<f32>,
    candidate_features: Vec<f32>,
    candidate_actions: Vec<Action>,
}

impl PendingDecision {
    fn snapshot(decision: &GameDecision<'_>, decision_id: u64) -> PyResult<Self> {
        let encoded = encode(decision);
        let mut candidate_features = Vec::with_capacity(encoded.candidates.len() * ACTION_WIDTH);
        let mut candidate_actions = Vec::with_capacity(encoded.candidates.len());
        for candidate in &encoded.candidates {
            candidate_features.extend_from_slice(&candidate.features.flatten());
            candidate_actions.push(candidate.action);
        }
        Ok(Self {
            decision_id,
            actor_seat: decision.seat.index() as u8,
            stage: stage_code(decision.observation.stage),
            state: encoded.state.flatten().to_vec(),
            candidate_features,
            candidate_actions,
        })
    }

    fn to_python(&self, py: Python<'_>) -> PyResult<Decision> {
        Ok(Decision {
            decision_id: self.decision_id,
            actor_seat: self.actor_seat,
            stage: self.stage,
            candidate_count: self.candidate_actions.len(),
            state: state_array(py, &self.state),
            candidate_features: candidate_array(
                py,
                self.candidate_actions.len(),
                &self.candidate_features,
            ),
        })
    }
}

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

/// Terminal settlement of one environment game.
#[pyclass(module = "mahjong_eval._native", frozen)]
struct Terminal {
    /// Seed of the game that ended.
    #[pyo3(get)]
    seed: u64,
    /// Number of accepted decisions, including forced single-candidate choices.
    #[pyo3(get)]
    decisions: u64,
    /// Why the game ended: `three_winners` or `wall_exhausted`.
    #[pyo3(get)]
    end_reason: &'static str,
    scores: [i32; 4],
    winners: [bool; 4],
}

#[pymethods]
impl Terminal {
    /// Net score by seat, summing to zero.
    #[getter]
    fn scores(&self) -> (i32, i32, i32, i32) {
        (
            self.scores[0],
            self.scores[1],
            self.scores[2],
            self.scores[3],
        )
    }

    /// Whether each seat won during the game.
    #[getter]
    fn winners(&self) -> (bool, bool, bool, bool) {
        (
            self.winners[0],
            self.winners[1],
            self.winners[2],
            self.winners[3],
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Terminal(seed={}, decisions={}, end_reason={}, scores={:?}, winners={:?})",
            self.seed, self.decisions, self.end_reason, self.scores, self.winners
        )
    }
}

/// One environment handle owning one game.
///
/// `reset(seed)` returns the first pending decision. `step(decision_id, index)`
/// commits exactly one cached canonical action and returns either the next
/// pending decision or a [`Terminal`]. Every rejection is raised before
/// authoritative game state changes.
#[pyclass(module = "mahjong_eval._native")]
struct Env {
    game: Option<Game>,
    decision_limit: u64,
    generation: u32,
    seed: u64,
    sequence: u32,
    accepted: u64,
    pending: Option<PendingDecision>,
}

#[pymethods]
impl Env {
    /// Creates a handle with a per-game decision safety bound.
    #[new]
    #[pyo3(signature = (decision_limit = DEFAULT_DECISION_LIMIT))]
    fn new(decision_limit: u64) -> Self {
        Self {
            game: None,
            decision_limit,
            generation: 0,
            seed: 0,
            sequence: 0,
            accepted: 0,
            pending: None,
        }
    }

    /// Per-game decision safety bound, counting accepted decisions.
    #[getter]
    fn decision_limit(&self) -> u64 {
        self.decision_limit
    }

    /// Starts a fresh seeded game, discarding any previous game, and returns its
    /// first pending decision.
    ///
    /// The reset generation advances so every identity from an earlier game
    /// becomes stale. A failed reset leaves the previous game untouched.
    #[pyo3(signature = (seed))]
    fn reset(&mut self, py: Python<'_>, seed: u64) -> PyResult<Decision> {
        if self.decision_limit == 0 {
            return Err(limit_error(seed, self.decision_limit, 0, None));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("environment reset generation exhausted"))?;

        let mut game = Game::new(seed);
        let pending = {
            let first = game.next_decision().ok_or_else(|| {
                PyRuntimeError::new_err("fresh game produced no opening decision")
            })?;
            PendingDecision::snapshot(
                &first,
                make_decision_id(generation, DECISION_ID_FIRST_SEQUENCE),
            )?
        };
        let decision = pending.to_python(py)?;

        // Replace (and thereby drop) any prior game only after the new opening
        // snapshot succeeds, keeping a failed reset from discarding old state.
        let previous = self.game.replace(game);
        drop(previous);
        self.generation = generation;
        self.seed = seed;
        self.sequence = DECISION_ID_FIRST_SEQUENCE;
        self.accepted = 0;
        self.pending = Some(pending);
        Ok(decision)
    }

    /// Commits one cached canonical action.
    ///
    /// `decision_id` must identify the current pending decision and
    /// `candidate_index` must address one of its canonical candidates. A stale
    /// identity or an out-of-range index is rejected without changing the game.
    #[allow(clippy::needless_pass_by_value)]
    fn step(
        &mut self,
        py: Python<'_>,
        decision_id: u64,
        candidate_index: usize,
    ) -> PyResult<Py<PyAny>> {
        let Some(game) = self.game.as_ref() else {
            return Err(EnvironmentStateError::new_err(
                "environment has no active game; call reset(seed) first",
            ));
        };
        if game.result().is_some() {
            return Err(EnvironmentStateError::new_err(
                "game is terminal; call reset(seed) to start another game",
            ));
        }
        let Some(pending) = self.pending.as_ref() else {
            return Err(EnvironmentStateError::new_err(
                "game has no pending decision",
            ));
        };
        if decision_id != pending.decision_id {
            return Err(StaleDecisionError::new_err(format!(
                "stale decision id {decision_id}: current pending decision is {} \
                 (seed {}, accepted {})",
                pending.decision_id, self.seed, self.accepted
            )));
        }
        if candidate_index >= pending.candidate_actions.len() {
            return Err(CandidateIndexError::new_err(format!(
                "candidate index {candidate_index} is out of range for {} candidates \
                 (decision {decision_id}, seed {}, accepted {})",
                pending.candidate_actions.len(),
                self.seed,
                self.accepted
            )));
        }
        if self.accepted >= self.decision_limit {
            return Err(limit_error(
                self.seed,
                self.decision_limit,
                self.accepted,
                Some(decision_id),
            ));
        }

        let action = pending.candidate_actions[candidate_index];
        let seat =
            Seat::new(usize::from(pending.actor_seat)).expect("snapshotted seat index is in range");
        let game = self.game.as_mut().expect("checked above");
        game.submit(seat, action).map_err(|error| {
            GameRuleError::new_err(format!(
                "game rejected canonical candidate {candidate_index} for decision \
                 {decision_id} (seed {}, accepted {}): {error}",
                self.seed, self.accepted
            ))
        })?;

        self.accepted += 1;
        self.sequence = self.sequence.checked_add(1).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "environment decision sequence exhausted (seed {}, accepted {})",
                self.seed, self.accepted
            ))
        })?;

        let next = self.game.as_mut().expect("game is present").next_decision();
        let Some(next) = next else {
            self.pending = None;
            let result = *self
                .game
                .as_ref()
                .and_then(Game::result)
                .ok_or_else(|| EnvironmentStateError::new_err("terminal game has no result"))?;
            return terminal_object(py, self.seed, self.accepted, &result);
        };

        let pending =
            PendingDecision::snapshot(&next, make_decision_id(self.generation, self.sequence))?;
        let decision = pending.to_python(py)?;
        self.pending = Some(pending);
        Ok(Py::new(py, decision)?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "Env(started={}, decision_limit={}, seed={}, accepted={})",
            self.game.is_some(),
            self.decision_limit,
            self.seed,
            self.accepted
        )
    }
}

/// Builds a terminal result object owned by Python.
fn terminal_object(
    py: Python<'_>,
    seed: u64,
    decisions: u64,
    result: &GameResult,
) -> PyResult<Py<PyAny>> {
    let terminal = Terminal {
        seed,
        decisions,
        end_reason: match result.end_reason {
            sichuan_mahjong_sim::EndReason::ThreeWinners => "three_winners",
            sichuan_mahjong_sim::EndReason::WallExhausted => "wall_exhausted",
        },
        scores: result.scores,
        winners: result.winners,
    };
    Ok(Py::new(py, terminal)?.into_any())
}

fn limit_error(seed: u64, limit: u64, accepted: u64, next_id: Option<u64>) -> PyErr {
    let next = match next_id {
        Some(id) => format!(" before decision {id}"),
        None => String::new(),
    };
    DecisionLimitError::new_err(format!(
        "decision limit {limit} reached after {accepted} accepted decisions \
         (seed {seed}){next}"
    ))
}

fn make_decision_id(generation: u32, sequence: u32) -> u64 {
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
    module.add_class::<Terminal>()?;
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
    Ok(())
}
