//! Fallible experiment driver over the public `next_decision` / `submit`
//! surface.
//!
//! The driver owns orchestration and failure reporting, not legality, state
//! transitions, or scoring: those stay exclusively authoritative in [`Game`].
//! Every game starts from `Game::new(seed)` and fresh choosers. A pending
//! decision is read before the decision limit is checked, so a game that ends
//! exactly on its last allowed submission is complete rather than a limit
//! failure.

use std::fmt;

use sichuan_mahjong_sim::{
    Action, Decision, DecisionStage, EndReason, FirstLegalPolicy, Game, GameError, Policy, Seat,
    PLAYER_COUNT,
};

use crate::adapter;
use crate::worker::{self, Rpc, RpcError};

/// The six fixed A-seat placements. Each pair is covered once, so every seat
/// is an A seat in exactly three of the six games per seed.
pub const PLACEMENTS: [[u8; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Seat-policy mode of one experiment run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// A seats use the Python heuristic, B seats use `FirstLegalPolicy`.
    Mixed,
    /// All four seats use `FirstLegalPolicy` (harness symmetry check).
    FirstOnly,
    /// All four seats use the Python heuristic (worker end-to-end check).
    HeuristicOnly,
}

impl Mode {
    /// Parses the frozen CLI spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mixed" => Some(Mode::Mixed),
            "first-only" => Some(Mode::FirstOnly),
            "heuristic-only" => Some(Mode::HeuristicOnly),
            _ => None,
        }
    }

    /// CLI spelling, also used as the mode token in [`trace_hash`].
    pub const fn token(self) -> &'static str {
        match self {
            Mode::Mixed => "mixed",
            Mode::FirstOnly => "first-only",
            Mode::HeuristicOnly => "heuristic-only",
        }
    }

    /// Whether this mode needs a Python worker at all.
    pub const fn uses_heuristic(self) -> bool {
        !matches!(self, Mode::FirstOnly)
    }
}

/// Which chooser serves one seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// Native [`FirstLegalPolicy`].
    FirstLegal,
    /// Imported Python heuristic through the worker.
    Heuristic,
}

/// Validated mode plus A-seat pair for one game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeatPlan {
    mode: Mode,
    a_seats: [u8; 2],
}

impl SeatPlan {
    /// Validates the mode and the A-seat pair.
    pub fn new(mode: Mode, a_seats: [u8; 2]) -> Result<Self, String> {
        for &seat in &a_seats {
            if seat as usize >= PLAYER_COUNT {
                return Err(format!("A seat {seat} is outside 0..{PLAYER_COUNT}"));
            }
        }
        if a_seats[0] == a_seats[1] {
            return Err(format!("A seats must be distinct, got {a_seats:?}"));
        }
        Ok(Self { mode, a_seats })
    }

    /// Mode of this plan.
    pub const fn mode(self) -> Mode {
        self.mode
    }

    /// A-seat pair of this plan.
    pub const fn a_seats(self) -> [u8; 2] {
        self.a_seats
    }

    /// Chooser slot for `seat`.
    pub const fn slot(self, seat: Seat) -> Slot {
        let is_a = seat.index() as u8 == self.a_seats[0] || seat.index() as u8 == self.a_seats[1];
        match self.mode {
            Mode::FirstOnly => Slot::FirstLegal,
            Mode::HeuristicOnly => Slot::Heuristic,
            Mode::Mixed if is_a => Slot::Heuristic,
            Mode::Mixed => Slot::FirstLegal,
        }
    }
}

/// One action in decision order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionStep {
    /// Seat that acted.
    pub seat: u8,
    /// Submitted action.
    pub action: Action,
}

/// One decision chooser. All choosers are stateless across games: a fresh
/// `FirstLegalPolicy` or a fresh heuristic request is used for every game.
pub trait Chooser {
    /// Chooses an action for the pending decision.
    fn choose(&mut self, decision: &Decision<'_>) -> Result<Action, ChoiceError>;
}

/// Chooser failures. None of them triggers a fallback action.
#[derive(Debug)]
pub enum ChoiceError {
    /// Adapter could not build the request or mapping.
    Adapter(adapter::AdapterError),
    /// Worker transport or response protocol failure.
    Worker(RpcError),
    /// The worker returned an ID that is not in the request.
    UnknownId {
        /// Returned ID.
        id: String,
        /// Number of candidates that were offered.
        candidate_count: usize,
    },
    /// Request serialization failed.
    Serialize(serde_json::Error),
}

impl fmt::Display for ChoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChoiceError::Adapter(error) => write!(f, "adapter error: {error}"),
            ChoiceError::Worker(error) => write!(f, "{error}"),
            ChoiceError::UnknownId {
                id,
                candidate_count,
            } => write!(
                f,
                "worker returned unknown candidate id '{id}' (offered {candidate_count} candidates)"
            ),
            ChoiceError::Serialize(error) => {
                write!(f, "request serialization error: {error}")
            }
        }
    }
}

impl std::error::Error for ChoiceError {}

impl From<adapter::AdapterError> for ChoiceError {
    fn from(error: adapter::AdapterError) -> Self {
        ChoiceError::Adapter(error)
    }
}

impl From<RpcError> for ChoiceError {
    fn from(error: RpcError) -> Self {
        ChoiceError::Worker(error)
    }
}

impl From<serde_json::Error> for ChoiceError {
    fn from(error: serde_json::Error) -> Self {
        ChoiceError::Serialize(error)
    }
}

/// Native baseline chooser.
pub struct FirstLegalChooser {
    policy: FirstLegalPolicy,
}

impl FirstLegalChooser {
    /// Creates a fresh, stateless baseline chooser.
    pub fn new() -> Self {
        Self {
            policy: FirstLegalPolicy::new(),
        }
    }
}

impl Default for FirstLegalChooser {
    fn default() -> Self {
        Self::new()
    }
}

impl Chooser for FirstLegalChooser {
    fn choose(&mut self, decision: &Decision<'_>) -> Result<Action, ChoiceError> {
        Ok(self.policy.choose(decision))
    }
}

/// Chooser that sends the adapter request to the Python worker and maps the
/// returned candidate ID back to the original action.
pub struct HeuristicChooser<'a> {
    rpc: &'a mut dyn Rpc,
}

impl<'a> HeuristicChooser<'a> {
    /// Wraps one long-lived worker connection.
    pub fn new(rpc: &'a mut dyn Rpc) -> Self {
        Self { rpc }
    }
}

impl Chooser for HeuristicChooser<'_> {
    fn choose(&mut self, decision: &Decision<'_>) -> Result<Action, ChoiceError> {
        let candidates = adapter::build(decision)?;
        let request = serde_json::to_string(&candidates.request)?;
        let response = self.rpc.request(&request)?;
        let id = worker::parse_response(&response)?;
        candidates.action_for(&id).ok_or(ChoiceError::UnknownId {
            id,
            candidate_count: candidates.len(),
        })
    }
}

/// Choosers available to one game.
pub struct SeatChoosers<'a> {
    /// Native baseline chooser, used by B seats and `first-only` runs.
    pub first: &'a mut dyn Chooser,
    /// Heuristic chooser, used by A seats in `mixed` and every seat in
    /// `heuristic-only` runs.
    pub heuristic: Option<&'a mut dyn Chooser>,
}

/// Pending-decision context attached to every failure record.
///
/// It contains only the deciding player's own decision input: the deciding
/// seat, the decision stage, the candidate count, and the exact minimal adapter
/// request (`hand`, `missing_suit`, `candidates`) when one can be built. No
/// opponent concealed hand, wall order, or staged private choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorContext {
    /// Zero-based index of the pending decision within the game.
    pub decision_index: u64,
    /// Deciding seat.
    pub seat: u8,
    /// Decision stage name (`exchange`, `missing_suit`, `turn`, ...).
    pub stage: &'static str,
    /// Number of legal candidates the adapter exposes.
    pub candidate_count: usize,
    /// Exact adapter request for this decision, when the adapter can build it.
    pub request: Option<adapter::DecisionRequest>,
}

impl ErrorContext {
    fn for_decision(decision: &Decision<'_>, decision_index: u64) -> Self {
        let request = adapter::build(decision)
            .ok()
            .map(|candidates| candidates.request);
        let candidate_count = request.as_ref().map_or_else(
            || adapter::candidate_count(decision),
            |request| request.candidates.len(),
        );
        Self {
            decision_index,
            seat: decision.seat.index() as u8,
            stage: stage_name(decision.observation.stage),
            candidate_count,
            request,
        }
    }
}

/// Terminal facts plus the deterministic action trace of one completed game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayedGame {
    /// Game seed.
    pub seed: u64,
    /// A-seat placement.
    pub a_seats: [u8; 2],
    /// Terminal net score by seat.
    pub scores: [i64; 4],
    /// `three_winners` or `wall_exhausted`.
    pub end_reason: &'static str,
    /// Number of submitted policy decisions.
    pub decisions: u64,
    /// Submitted actions in order.
    pub trace: Vec<ActionStep>,
    /// Deterministic identity of mode, seed, placement, and action trace.
    pub trace_hash: u64,
}

/// A failed game drive.
#[derive(Debug)]
pub enum PlayError {
    /// The decision safety bound was reached while a choice was still pending.
    DecisionLimit {
        /// Configured bound.
        limit: u64,
        /// Pending decision context.
        context: Box<ErrorContext>,
    },
    /// Choosing an action failed (adapter, worker, or unknown ID).
    Choice {
        /// Failure message.
        message: String,
        /// Pending decision context.
        context: Box<ErrorContext>,
    },
    /// The game rejected a chosen action.
    Illegal {
        /// Seat that submitted.
        seat: Seat,
        /// Rejected action.
        action: Action,
        /// Authoritative rejection from the game.
        source: GameError,
        /// Pending decision context.
        context: Box<ErrorContext>,
    },
    /// No pending decision and no terminal result.
    MissingResult {
        /// Number of decisions submitted before the game went silent.
        decisions: u64,
    },
}

impl PlayError {
    /// Pending decision context, when the failure happened at a decision.
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            PlayError::DecisionLimit { context, .. }
            | PlayError::Choice { context, .. }
            | PlayError::Illegal { context, .. } => Some(context.as_ref()),
            PlayError::MissingResult { .. } => None,
        }
    }

    /// Number of decisions submitted before the failure.
    pub fn decision_index(&self) -> u64 {
        match self {
            PlayError::DecisionLimit { context, .. }
            | PlayError::Choice { context, .. }
            | PlayError::Illegal { context, .. } => context.decision_index,
            PlayError::MissingResult { decisions } => *decisions,
        }
    }

    /// Human-readable failure message for the failure record.
    pub fn message(&self) -> String {
        match self {
            PlayError::DecisionLimit { limit, context } => format!(
                "decision limit {limit} reached with a pending {} decision for seat {} (candidates: {})",
                context.stage, context.seat, context.candidate_count
            ),
            PlayError::Choice { message, .. } => message.clone(),
            PlayError::Illegal {
                seat,
                action,
                source,
                ..
            } => format!("game rejected action {action:?} from seat {seat}: {source}"),
            PlayError::MissingResult { decisions } => format!(
                "game reported neither a pending decision nor a terminal result after {decisions} decisions"
            ),
        }
    }
}

impl fmt::Display for PlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for PlayError {}

/// Plays one fresh game from `seed` with the given seat plan.
pub fn play_game(
    seed: u64,
    plan: SeatPlan,
    choosers: &mut SeatChoosers<'_>,
    decision_limit: u64,
) -> Result<PlayedGame, PlayError> {
    let mut game = Game::new(seed);
    let mut trace: Vec<ActionStep> = Vec::new();
    let mut decisions = 0u64;

    loop {
        let (seat, action, context) = {
            // Read the pending decision first: a game that ended exactly on the
            // last allowed submission is complete, not limit exhaustion.
            let Some(decision) = game.next_decision() else {
                break;
            };
            // Built once per decision so every failure record carries the exact
            // request the adapter exposes for that pending decision.
            let context = ErrorContext::for_decision(&decision, decisions);
            if decisions >= decision_limit {
                return Err(PlayError::DecisionLimit {
                    limit: decision_limit,
                    context: Box::new(context),
                });
            }
            let chooser = match plan.slot(decision.seat) {
                Slot::FirstLegal => &mut *choosers.first,
                Slot::Heuristic => {
                    choosers
                        .heuristic
                        .as_deref_mut()
                        .ok_or_else(|| PlayError::Choice {
                            message: "heuristic seat has no worker chooser".to_owned(),
                            context: Box::new(context.clone()),
                        })?
                }
            };
            let action = chooser
                .choose(&decision)
                .map_err(|error| PlayError::Choice {
                    message: error.to_string(),
                    context: Box::new(context.clone()),
                })?;
            (decision.seat, action, context)
        };

        game.submit(seat, action)
            .map_err(|source| PlayError::Illegal {
                seat,
                action,
                source,
                context: Box::new(context),
            })?;
        trace.push(ActionStep {
            seat: seat.index() as u8,
            action,
        });
        decisions += 1;
    }

    let result = game
        .result()
        .ok_or(PlayError::MissingResult { decisions })?;
    Ok(PlayedGame {
        seed,
        a_seats: plan.a_seats(),
        scores: std::array::from_fn(|index| result.scores[index] as i64),
        end_reason: end_reason_name(result.end_reason),
        decisions,
        trace_hash: trace_hash(plan.mode(), seed, plan.a_seats(), &trace),
        trace,
    })
}

/// Public stage name used in failure records.
pub const fn stage_name(stage: DecisionStage) -> &'static str {
    match stage {
        DecisionStage::Exchange => "exchange",
        DecisionStage::MissingSuit => "missing_suit",
        DecisionStage::Turn => "turn",
        DecisionStage::DiscardResponse => "discard_response",
        DecisionStage::AddedKongResponse => "added_kong_response",
    }
}

/// Snake-case end reason used in per-game records.
pub const fn end_reason_name(reason: EndReason) -> &'static str {
    match reason {
        EndReason::ThreeWinners => "three_winners",
        EndReason::WallExhausted => "wall_exhausted",
    }
}

/// Deterministic identity of one game's action trace.
///
/// Timing never contributes. The hash is FNV-1a 64 over the following
/// unambiguous byte sequence:
///
/// 1. the version tag `policy_ab.trace.v1`;
/// 2. the mode token length and bytes;
/// 3. the seed as 8 little-endian bytes;
/// 4. the two A-seat indices;
/// 5. for every decision in order: the seat index, an action tag, and the
///    action parameters. Exchange tiles are sorted first, so the exchange
///    triple is treated as the multiset the game actually applies.
///
/// Action tags: 1 exchange, 2 choose missing, 3 discard, 4 peng, 5 open kong,
/// 6 concealed kong, 7 added kong, 8 Hu, 9 pass. Tile parameters are encoded
/// tile indices; the missing-suit parameter is a suit index.
pub fn trace_hash(mode: Mode, seed: u64, a_seats: [u8; 2], trace: &[ActionStep]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv(hash, b"policy_ab.trace.v1");
    let token = mode.token().as_bytes();
    hash = fnv(hash, &[token.len() as u8]);
    hash = fnv(hash, token);
    hash = fnv(hash, &seed.to_le_bytes());
    hash = fnv(hash, &a_seats);
    for step in trace {
        hash = fnv(hash, &[step.seat]);
        hash = hash_action(hash, step.action);
    }
    hash
}

/// 16-character lowercase hex form of [`trace_hash`].
pub fn trace_hash_hex(hash: u64) -> String {
    format!("{hash:016x}")
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_action(mut hash: u64, action: Action) -> u64 {
    match action {
        Action::Exchange(tiles) => {
            let mut sorted = tiles;
            sorted.sort();
            hash = fnv(hash, &[1]);
            for tile in sorted {
                hash = fnv(hash, &[tile.index()]);
            }
        }
        Action::ChooseMissing(suit) => hash = fnv(hash, &[2, suit.index() as u8]),
        Action::Discard(tile) => hash = fnv(hash, &[3, tile.index()]),
        Action::Peng => hash = fnv(hash, &[4]),
        Action::OpenKong => hash = fnv(hash, &[5]),
        Action::ConcealedKong(tile) => hash = fnv(hash, &[6, tile.index()]),
        Action::AddedKong(tile) => hash = fnv(hash, &[7, tile.index()]),
        Action::Hu => hash = fnv(hash, &[8]),
        Action::Pass => hash = fnv(hash, &[9]),
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use sichuan_mahjong_sim::{Suit, Tile};

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    struct ScriptedRpc {
        handler: Box<dyn FnMut(&str) -> Result<String, RpcError>>,
        seen: Vec<String>,
    }

    impl ScriptedRpc {
        fn new(handler: impl FnMut(&str) -> Result<String, RpcError> + 'static) -> Self {
            Self {
                handler: Box::new(handler),
                seen: Vec::new(),
            }
        }

        fn echo_first_candidate() -> Self {
            Self::new(|line| {
                let request: serde_json::Value = serde_json::from_str(line).unwrap();
                let id = request["candidates"][0]["id"].as_str().unwrap();
                Ok(format!(r#"{{"id":"{id}"}}"#))
            })
        }
    }

    impl Rpc for ScriptedRpc {
        fn request(&mut self, line: &str) -> Result<String, RpcError> {
            self.seen.push(line.to_owned());
            (self.handler)(line)
        }
    }

    fn first_seat_plan(a_seats: [u8; 2]) -> SeatPlan {
        SeatPlan::new(Mode::FirstOnly, a_seats).unwrap()
    }

    fn first_only(seed: u64, a_seats: [u8; 2], decision_limit: u64) -> PlayedGame {
        let mut first = FirstLegalChooser::new();
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: None,
        };
        play_game(
            seed,
            first_seat_plan(a_seats),
            &mut choosers,
            decision_limit,
        )
        .unwrap()
    }

    fn heuristic_only(seed: u64, rpc: &mut ScriptedRpc, decision_limit: u64) -> PlayedGame {
        let mut first = FirstLegalChooser::new();
        let mut chooser = HeuristicChooser::new(rpc);
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: Some(&mut chooser),
        };
        play_game(
            seed,
            SeatPlan::new(Mode::HeuristicOnly, [0, 1]).unwrap(),
            &mut choosers,
            decision_limit,
        )
        .unwrap()
    }

    #[test]
    fn first_only_placements_are_symmetric_and_replayable() {
        for seed in [3u64, 11] {
            let mut scores_by_placement = Vec::new();
            for a_seats in PLACEMENTS {
                let first_run = first_only(seed, a_seats, 10_000);
                let second_run = first_only(seed, a_seats, 10_000);
                assert_eq!(first_run.trace, second_run.trace);
                assert_eq!(first_run.trace_hash, second_run.trace_hash);
                assert_eq!(first_run.scores, second_run.scores);
                assert_eq!(first_run.end_reason, second_run.end_reason);
                assert_eq!(first_run.decisions, second_run.decisions);
                assert_eq!(first_run.scores.iter().sum::<i64>(), 0);
                assert!(!first_run.trace.is_empty());
                scores_by_placement.push(first_run.scores);
            }
            // All four seats share one policy, so the six placements replay the
            // same deal and the same actions.
            for scores in &scores_by_placement {
                assert_eq!(*scores, scores_by_placement[0]);
            }
            let mut contrasts = Vec::new();
            for (placement, scores) in PLACEMENTS.iter().zip(&scores_by_placement) {
                let a =
                    (scores[placement[0] as usize] + scores[placement[1] as usize]) as f64 / 2.0;
                let b_sum: i64 = (0..PLAYER_COUNT)
                    .filter(|seat| !placement.contains(&(*seat as u8)))
                    .map(|seat| scores[seat])
                    .sum();
                contrasts.push(a - b_sum as f64 / 2.0);
            }
            let mean = contrasts.iter().sum::<f64>() / contrasts.len() as f64;
            assert!(mean.abs() < 1e-9, "placement mean contrast {mean}");
        }
    }

    #[test]
    fn trace_hash_is_action_sensitive_and_ignores_timing() {
        let exchange =
            Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)]);
        let permuted =
            Action::Exchange([tile(Suit::Wan, 3), tile(Suit::Wan, 1), tile(Suit::Wan, 2)]);
        let base = trace_hash(
            Mode::FirstOnly,
            7,
            [0, 1],
            &[ActionStep {
                seat: 0,
                action: exchange,
            }],
        );
        assert_eq!(
            base,
            trace_hash(
                Mode::FirstOnly,
                7,
                [0, 1],
                &[ActionStep {
                    seat: 0,
                    action: permuted
                }]
            ),
            "exchange order is a multiset, not an identity input"
        );
        assert_ne!(
            base,
            trace_hash(
                Mode::FirstOnly,
                7,
                [0, 1],
                &[ActionStep {
                    seat: 0,
                    action: Action::Discard(tile(Suit::Wan, 1))
                }]
            )
        );
        assert_ne!(
            base,
            trace_hash(
                Mode::FirstOnly,
                8,
                [0, 1],
                &[ActionStep {
                    seat: 0,
                    action: exchange
                }]
            )
        );
        assert_ne!(
            base,
            trace_hash(
                Mode::Mixed,
                7,
                [0, 1],
                &[ActionStep {
                    seat: 0,
                    action: exchange
                }]
            )
        );
        assert_ne!(
            base,
            trace_hash(
                Mode::FirstOnly,
                7,
                [1, 0],
                &[ActionStep {
                    seat: 0,
                    action: exchange
                }]
            )
        );
        assert_eq!(trace_hash_hex(0xab), "00000000000000ab");
    }

    #[test]
    fn decision_limit_requires_a_pending_choice() {
        let plan = first_seat_plan([0, 1]);
        let mut first = FirstLegalChooser::new();
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: None,
        };
        let error = play_game(5, plan, &mut choosers, 3).unwrap_err();
        match error {
            PlayError::DecisionLimit { limit, context } => {
                assert_eq!(limit, 3);
                assert_eq!(context.decision_index, 3);
                assert_eq!(context.seat, 3);
                assert_eq!(context.stage, "exchange");
                assert!(context.candidate_count > 0);
                let request = context.request.as_ref().expect("adapter request");
                assert_eq!(request.candidates.len(), context.candidate_count);
                assert_eq!(request.hand.len(), 13, "seat 3 is not the dealer");
                assert_eq!(request.missing_suit, None);
            }
            other => panic!("expected a decision-limit failure, got {other:?}"),
        }

        // Ending exactly on the last allowed submission is a complete game.
        let complete = first_only(5, [0, 1], 10_000);
        let exact = first_only(5, [0, 1], complete.decisions);
        assert_eq!(exact, complete);
        let mut first = FirstLegalChooser::new();
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: None,
        };
        let short = play_game(5, plan, &mut choosers, complete.decisions - 1).unwrap_err();
        assert!(matches!(short, PlayError::DecisionLimit { .. }));
    }

    #[test]
    fn illegal_action_keeps_its_pending_context() {
        struct AlwaysPass;
        impl Chooser for AlwaysPass {
            fn choose(&mut self, _decision: &Decision<'_>) -> Result<Action, ChoiceError> {
                Ok(Action::Pass)
            }
        }

        let mut first = AlwaysPass;
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: None,
        };
        let error = play_game(1, first_seat_plan([0, 1]), &mut choosers, 100).unwrap_err();
        match error {
            PlayError::Illegal {
                seat,
                action,
                context,
                ..
            } => {
                assert_eq!(seat.index(), 0);
                assert_eq!(action, Action::Pass);
                assert_eq!(context.decision_index, 0);
                assert_eq!(context.stage, "exchange");
                assert!(context.candidate_count > 0);
                assert!(context.request.is_some());
            }
            other => panic!("expected an illegal-action failure, got {other:?}"),
        }
    }

    #[test]
    fn seat_plans_assign_the_documented_slots() {
        let mixed = SeatPlan::new(Mode::Mixed, [1, 3]).unwrap();
        let slots: Vec<Slot> = (0..PLAYER_COUNT)
            .map(|index| mixed.slot(Seat::new(index).unwrap()))
            .collect();
        assert_eq!(
            slots,
            vec![
                Slot::FirstLegal,
                Slot::Heuristic,
                Slot::FirstLegal,
                Slot::Heuristic
            ]
        );
        let heuristic = SeatPlan::new(Mode::HeuristicOnly, [0, 1]).unwrap();
        for index in 0..PLAYER_COUNT {
            assert_eq!(heuristic.slot(Seat::new(index).unwrap()), Slot::Heuristic);
        }
        let first = SeatPlan::new(Mode::FirstOnly, [0, 1]).unwrap();
        for index in 0..PLAYER_COUNT {
            assert_eq!(first.slot(Seat::new(index).unwrap()), Slot::FirstLegal);
        }
        assert_eq!(Mode::Mixed.token(), "mixed");
        assert_eq!(Mode::FirstOnly.token(), "first-only");
        assert_eq!(Mode::HeuristicOnly.token(), "heuristic-only");
        assert_eq!(Mode::parse("first-only"), Some(Mode::FirstOnly));
        assert_eq!(Mode::parse("heuristic_only"), None);

        assert!(SeatPlan::new(Mode::Mixed, [0, 0]).is_err());
        assert!(SeatPlan::new(Mode::Mixed, [0, 4]).is_err());
    }

    #[test]
    fn heuristic_chooser_round_trips_through_the_worker_boundary() {
        let mut rpc = ScriptedRpc::echo_first_candidate();
        let game = heuristic_only(21, &mut rpc, 10_000);
        assert!(game.decisions >= 9, "exchange and missing-suit phases run");
        assert_eq!(game.scores.iter().sum::<i64>(), 0);
        assert_eq!(rpc.seen.len() as u64, game.decisions);
        let parsed: serde_json::Value = serde_json::from_str(&rpc.seen[0]).unwrap();
        let hand = parsed["hand"].as_array().unwrap();
        assert_eq!(
            hand.len(),
            14,
            "the dealer opens the exchange with 14 tiles"
        );
        assert!(parsed["missing_suit"].is_null());
        assert!(!parsed["candidates"].as_array().unwrap().is_empty());
    }

    #[test]
    fn heuristic_mode_is_replayable_through_the_worker_boundary() {
        let first = heuristic_only(21, &mut ScriptedRpc::echo_first_candidate(), 10_000);
        let second = heuristic_only(21, &mut ScriptedRpc::echo_first_candidate(), 10_000);
        assert_eq!(first.trace, second.trace);
        assert_eq!(first.trace_hash, second.trace_hash);
        assert_eq!(first.scores, second.scores);
    }

    #[test]
    fn unknown_worker_id_fails_the_game_with_context() {
        let mut rpc = ScriptedRpc::new(|_| Ok(r#"{"id":"not-a-candidate"}"#.to_owned()));
        let mut first = FirstLegalChooser::new();
        let mut chooser = HeuristicChooser::new(&mut rpc);
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: Some(&mut chooser),
        };
        let error = play_game(
            4,
            SeatPlan::new(Mode::HeuristicOnly, [0, 1]).unwrap(),
            &mut choosers,
            100,
        )
        .unwrap_err();
        let message = error.message();
        assert!(message.contains("not-a-candidate"), "message: {message}");
        let context = error.context().cloned().expect("decision context");
        let request = context.request.as_ref().expect("adapter request");
        assert_eq!(request.candidates.len(), context.candidate_count);
        assert_eq!(context.decision_index, 0);
        assert_eq!(context.stage, "exchange");
        assert!(context.candidate_count > 0);
    }

    #[test]
    fn protocol_failures_do_not_fall_back_to_a_baseline_action() {
        let mut rpc = ScriptedRpc::new(|_| Ok("[]".to_owned()));
        let mut first = FirstLegalChooser::new();
        let mut chooser = HeuristicChooser::new(&mut rpc);
        let mut choosers = SeatChoosers {
            first: &mut first,
            heuristic: Some(&mut chooser),
        };
        let error = play_game(
            4,
            SeatPlan::new(Mode::HeuristicOnly, [0, 1]).unwrap(),
            &mut choosers,
            100,
        )
        .unwrap_err();
        assert!(
            error.message().contains("protocol error"),
            "{}",
            error.message()
        );
        assert_eq!(error.context().map(|context| context.seat), Some(0));
    }

    #[test]
    fn every_placement_pair_is_present_once() {
        let pairs: HashSet<[u8; 2]> = PLACEMENTS.iter().copied().collect();
        assert_eq!(pairs.len(), PLACEMENTS.len());
        for pair in PLACEMENTS {
            assert!(pair[0] < pair[1]);
        }
        for seat in 0..PLAYER_COUNT as u8 {
            assert_eq!(
                PLACEMENTS
                    .iter()
                    .filter(|candidate| candidate.contains(&seat))
                    .count(),
                3,
                "every seat must be an A seat three times"
            );
        }
    }
}
