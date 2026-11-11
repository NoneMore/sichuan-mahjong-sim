use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sichuan_mahjong_neural::collection::{
    action_kind_id, encode_action, stage_id, ACTION_KIND_NAMES,
};
use sichuan_mahjong_neural::encoding::{encode, ACTION_WIDTH};
use sichuan_mahjong_sim::{Action, Decision as GameDecision, EndReason, GameResult};

/// One pending decision with an owned, player-visible numeric snapshot and the
/// canonical semantic actions used to resolve local indices.
pub(super) struct PendingDecision {
    pub(super) decision_id: u64,
    pub(super) actor_seat: u8,
    stage: u8,
    state: Vec<f32>,
    candidate_features: Vec<f32>,
    pub(super) candidate_actions: Vec<Action>,
}

impl PendingDecision {
    pub(super) fn snapshot(decision: &GameDecision<'_>, decision_id: u64) -> PyResult<Self> {
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
            stage: stage_id(decision.observation.stage),
            state: encoded.state.flatten().to_vec(),
            candidate_features,
            candidate_actions,
        })
    }

    pub(super) fn to_python(&self, py: Python<'_>) -> PyResult<Decision> {
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
/// `state` is a C-contiguous `float32` array of width `STATE_WIDTH`.
/// `candidate_features` is a C-contiguous `float32` array with one row per
/// canonical candidate and width [`CANDIDATE_WIDTH`]. Both arrays own their
/// data independently of the game; mutating them cannot change the game.
#[pyclass(module = "mahjong_eval._native", frozen)]
pub(super) struct Decision {
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
pub(super) struct Terminal {
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

/// Builds a terminal result object owned by Python.
pub(super) fn terminal_object(
    py: Python<'_>,
    seed: u64,
    decisions: u64,
    result: &GameResult,
) -> PyResult<Py<PyAny>> {
    let terminal = Terminal {
        seed,
        decisions,
        end_reason: end_reason_name(result.end_reason),
        scores: result.scores,
        winners: result.winners,
    };
    Ok(Py::new(py, terminal)?.into_any())
}

fn end_reason_name(reason: EndReason) -> &'static str {
    match reason {
        EndReason::ThreeWinners => "three_winners",
        EndReason::WallExhausted => "wall_exhausted",
    }
}

pub(super) fn action_dict(py: Python<'_>, action: Action) -> PyResult<Py<PyDict>> {
    let kind_id = action_kind_id(&action);
    let wire: Vec<i64> = encode_action(action).into_iter().map(i64::from).collect();
    let dict = PyDict::new(py);
    dict.set_item("kind", ACTION_KIND_NAMES[usize::from(kind_id)])?;
    dict.set_item("kind_id", kind_id)?;
    dict.set_item("wire", wire)?;
    Ok(dict.unbind())
}

pub(super) fn set_terminal_fields(
    dict: &Bound<'_, PyDict>,
    decisions: u64,
    result: &GameResult,
) -> PyResult<()> {
    dict.set_item("decisions", decisions)?;
    dict.set_item(
        "scores",
        (
            result.scores[0],
            result.scores[1],
            result.scores[2],
            result.scores[3],
        ),
    )?;
    dict.set_item(
        "winners",
        (
            result.winners[0],
            result.winners[1],
            result.winners[2],
            result.winners[3],
        ),
    )?;
    dict.set_item("end_reason", end_reason_name(result.end_reason))?;
    Ok(())
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
