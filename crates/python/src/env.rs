use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use sichuan_mahjong_neural::encoding::{canonicalize, encode};
use sichuan_mahjong_sim::policy::{HeuristicCache, HeuristicPolicy, Policy};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::{Game, Seat};

use crate::snapshot::{terminal_object, Decision, PendingDecision};
use crate::{
    CandidateIndexError, DecisionLimitError, EnvironmentStateError, GameRuleError,
    StaleDecisionError, TeacherCache, DECISION_ID_FIRST_SEQUENCE,
};

/// One environment handle owning one game.
///
/// `reset(seed)` returns the first pending decision. `step(decision_id, index)`
/// commits exactly one cached canonical action and returns either the next
/// pending decision or a [`Terminal`](crate::snapshot::Terminal). Every rejection is raised
/// before authoritative game state changes.
#[pyclass(module = "mahjong_eval._native")]
pub(super) struct Env {
    game: Option<Game>,
    decision_limit: u64,
    generation: u32,
    seed: u64,
    sequence: u32,
    accepted: u64,
    pending: Option<PendingDecision>,
    teacher_cache: HeuristicCache,
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
            teacher_cache: HeuristicCache::default(),
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

    /// Returns the teacher's canonical candidate index for the current pending
    /// decision without advancing the game.
    ///
    /// The teacher chooses on the original Rust `Decision`; the encoder's
    /// semantic target lookup maps that action to the canonical local index.
    /// `cache` is optional; without one the handle's own derived cache is used.
    #[pyo3(signature = (cache = None))]
    fn teacher_index(
        &mut self,
        py: Python<'_>,
        cache: Option<Py<TeacherCache>>,
    ) -> PyResult<usize> {
        let cache = match cache {
            Some(cache) => cache.borrow(py).inner.clone(),
            None => self.teacher_cache.clone(),
        };
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
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| EnvironmentStateError::new_err("game has no pending decision"))?;

        let mut teacher = HeuristicPolicy::with_cache(cache);
        let game = self.game.as_mut().expect("game is present");
        let decision = game
            .next_decision()
            .ok_or_else(|| EnvironmentStateError::new_err("game has no pending decision"))?;
        let action = canonicalize(teacher.choose(&decision));
        let encoded = encode(&decision);
        let index = encoded.target_index(&action).map_err(|error| {
            GameRuleError::new_err(format!(
                "teacher action {action:?} is not a candidate of decision {} \
                 (seed {}, accepted {}): {error}",
                pending.decision_id, self.seed, self.accepted
            ))
        })?;
        if encoded.candidates[index].action != pending.candidate_actions[index] {
            return Err(PyRuntimeError::new_err(format!(
                "teacher target lookup disagrees with cached canonical action at index \
                 {index} (decision {}, seed {})",
                pending.decision_id, self.seed
            )));
        }
        Ok(index)
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

pub(super) fn limit_error(seed: u64, limit: u64, accepted: u64, next_id: Option<u64>) -> PyErr {
    let next = match next_id {
        Some(id) => format!(" before decision {id}"),
        None => String::new(),
    };
    DecisionLimitError::new_err(format!(
        "decision limit {limit} reached after {accepted} accepted decisions \
         (seed {seed}){next}"
    ))
}

pub(super) fn make_decision_id(generation: u32, sequence: u32) -> u64 {
    (u64::from(generation) << 32) | u64::from(sequence)
}
