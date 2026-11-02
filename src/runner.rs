//! Drives games with four policies and reports outcome and timing.
//!
//! The runner understands no mahjong rules. It only calls
//! [`Game::next_decision`], hands the borrowed decision to the pending seat's
//! [`Policy`], and submits the chosen action. Policy time and rule-driving time
//! are measured separately. A rejected action is returned as an error instead
//! of being replaced or retried.

use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

use crate::game::{EndReason, Game, GameError};
use crate::policy::{random_seat_seed, FirstLegalPolicy, Policy, RandomPolicy};
use crate::{Action, Seat, PLAYER_COUNT};

/// Default safety bound on policy decisions in one drive call.
///
/// A legal baseline game needs far fewer than this. The bound turns a broken
/// policy/game interaction into an error instead of an unbounded loop. It is
/// counted per call (`run_game`, each game of `run_batch`, or the seeded
/// `run_new`) and never accumulated across resumed calls.
pub const DEFAULT_DECISION_LIMIT: u64 = 10_000;

/// Owned finish reason used by execution reports.
///
/// This mirrors the game's terminal reason without requiring the report to own
/// or clone the authoritative game result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Only one active player remained.
    ThreeWinners,
    /// The wall was exhausted.
    WallExhausted,
}

/// One policy choice in the order it was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionRecord {
    /// Seat that made the choice.
    pub seat: Seat,
    /// Action returned by the policy.
    pub action: Action,
}

/// Result of driving one game to termination.
#[derive(Debug)]
pub struct RunReport {
    /// Terminal net score by seat.
    pub scores: [i32; 4],
    /// Seats that won during the game.
    pub winners: [bool; 4],
    /// Why the game ended.
    pub end_reason: FinishReason,
    /// Number of policy decisions made.
    pub decisions: u64,
    /// Time spent inside policy `choose` calls.
    pub policy_time: Duration,
    /// Time spent in `next_decision` plus `submit`.
    pub rules_time: Duration,
    /// Decisions in order. Empty for batch runs.
    pub trace: Vec<DecisionRecord>,
}

/// Aggregated result of a reproducible batch.
#[derive(Debug)]
pub struct BatchReport {
    /// Number of completed games.
    pub games: u64,
    /// Total policy decisions across the batch.
    pub decisions: u64,
    /// Time spent inside policy `choose` calls.
    pub policy_time: Duration,
    /// Time spent in `next_decision` plus `submit`.
    pub rules_time: Duration,
    /// Wall-clock time for the whole batch, including game creation.
    pub total_time: Duration,
    /// Summed terminal net scores by seat.
    pub score_totals: [i64; 4],
    /// Number of games each seat won.
    pub winners: [u64; 4],
    /// Games ended because only one active player remained.
    pub three_winners: u64,
    /// Games ended by wall exhaustion.
    pub wall_exhausted: u64,
}

impl BatchReport {
    /// Completed games per second over the measured batch wall time.
    pub fn games_per_sec(&self) -> f64 {
        let seconds = self.total_time.as_secs_f64();
        if seconds > 0.0 {
            self.games as f64 / seconds
        } else {
            0.0
        }
    }

    /// Policy decisions per second over the measured batch wall time.
    pub fn decisions_per_sec(&self) -> f64 {
        let seconds = self.total_time.as_secs_f64();
        if seconds > 0.0 {
            self.decisions as f64 / seconds
        } else {
            0.0
        }
    }

    /// Mean net score per game by seat.
    pub fn mean_scores(&self) -> [f64; 4] {
        if self.games == 0 {
            return [0.0; 4];
        }
        let games = self.games as f64;
        [
            self.score_totals[0] as f64 / games,
            self.score_totals[1] as f64 / games,
            self.score_totals[2] as f64 / games,
            self.score_totals[3] as f64 / games,
        ]
    }

    /// Sum of all terminal net scores (zero for a zero-sum rule set).
    pub fn score_sum(&self) -> i64 {
        self.score_totals.iter().sum()
    }
}

/// Errors that stop a run.
#[derive(Debug)]
pub enum RunnerError {
    /// The policy returned an action the game rejected.
    IllegalAction {
        /// Seat whose policy chose the action.
        seat: Seat,
        /// Rejected action.
        action: Action,
        /// Authoritative rejection from the game.
        source: GameError,
    },
    /// The configured decision bound was reached before termination.
    ///
    /// The game already contains every action accepted before the bound. The
    /// caller may raise the limit and call `run_game` again to continue.
    DecisionLimit {
        /// Configured bound.
        limit: u64,
    },
    /// The game reported no pending decision but also no terminal result.
    MissingResult,
    /// A batch was requested with zero games.
    EmptyBatch,
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::IllegalAction {
                seat,
                action,
                source,
            } => write!(
                f,
                "policy for {seat} returned illegal action {action:?}: {source}"
            ),
            RunnerError::DecisionLimit { limit } => {
                write!(f, "game reached the {limit}-decision safety limit")
            }
            RunnerError::MissingResult => {
                f.write_str("game has no pending decision but no terminal result")
            }
            RunnerError::EmptyBatch => f.write_str("a batch requires at least one game"),
        }
    }
}

impl Error for RunnerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RunnerError::IllegalAction { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Drives games with one policy per seat.
pub struct Runner {
    policies: [Box<dyn Policy>; PLAYER_COUNT],
    decision_limit: u64,
}

impl Runner {
    /// Creates a runner from four seat policies.
    pub fn new(policies: [Box<dyn Policy>; PLAYER_COUNT]) -> Self {
        Self {
            policies,
            decision_limit: DEFAULT_DECISION_LIMIT,
        }
    }

    /// Creates a runner using [`FirstLegalPolicy`] for every seat.
    pub fn first_legal() -> Self {
        Self::new(std::array::from_fn(|_| {
            Box::new(FirstLegalPolicy::new()) as Box<dyn Policy>
        }))
    }

    /// Creates a runner using independently seeded [`RandomPolicy`] streams.
    ///
    /// Seat seeds are derived from `seed` with
    /// [`random_seat_seed`], so the same seed reproduces the same batch. This
    /// always starts every stream at its seeded position; it does not restore
    /// a mid-game policy checkpoint. To resume a random run, clone the
    /// policies held by the checkpoint and pass them to [`Runner::new`].
    pub fn random(seed: u64) -> Self {
        Self::new(std::array::from_fn(|index| {
            let seat = Seat::new(index).expect("seat index is in range");
            Box::new(RandomPolicy::new(random_seat_seed(seed, seat))) as Box<dyn Policy>
        }))
    }

    /// Sets the decision safety limit applied to each subsequent drive call
    /// (builder form).
    ///
    /// The limit is per call, not cumulative: continuing a game that stopped
    /// at the limit starts a fresh count. A later `run_game` may therefore
    /// reach the same limit again unless it is raised first.
    pub fn with_decision_limit(mut self, limit: u64) -> Self {
        self.decision_limit = limit;
        self
    }

    /// Changes the decision safety limit applied to each subsequent drive
    /// call.
    ///
    /// See [`Runner::with_decision_limit`] for the per-call counting rule.
    pub fn set_decision_limit(&mut self, limit: u64) {
        self.decision_limit = limit;
    }

    /// Creates a new game from `seed` and drives it to termination.
    pub fn run_new(&mut self, seed: u64) -> Result<RunReport, RunnerError> {
        let mut game = Game::new(seed);
        self.run_game(&mut game)
    }

    /// Drives an existing (possibly partially played or cloned) game.
    ///
    /// The runner does not reset or inspect the game; it continues from the
    /// current pending decision and returns once the game reports a result.
    /// The decision limit counts only the decisions made by this call, so a
    /// game interrupted by [`RunnerError::DecisionLimit`] keeps its accepted
    /// actions and can be driven again.
    ///
    /// Only the game state is supplied here. Continuation is exact only if
    /// every stateful policy also continues from its checkpoint state; a
    /// rebuilt `Runner::random(seed)` restarts its streams instead.
    pub fn run_game(&mut self, game: &mut Game) -> Result<RunReport, RunnerError> {
        self.drive(game, true)
    }

    /// Runs `games` seeded games, game `i` using seed
    /// `first_seed.wrapping_add(i)`.
    ///
    /// Policy RNG streams continue across games, matching repeated calls to
    /// [`Runner::run_new`] on the same runner. Pass `games == 0` is rejected.
    pub fn run_batch(&mut self, first_seed: u64, games: u64) -> Result<BatchReport, RunnerError> {
        if games == 0 {
            return Err(RunnerError::EmptyBatch);
        }

        let started = Instant::now();
        let mut report = BatchReport {
            games,
            decisions: 0,
            policy_time: Duration::ZERO,
            rules_time: Duration::ZERO,
            total_time: Duration::ZERO,
            score_totals: [0; 4],
            winners: [0; 4],
            three_winners: 0,
            wall_exhausted: 0,
        };

        for index in 0..games {
            let seed = first_seed.wrapping_add(index);
            let mut game = Game::new(seed);
            // Traces are only useful for single-game replay; skip recording
            // them to keep batch measurement allocation-light.
            let run = self.drive(&mut game, false)?;

            report.decisions += run.decisions;
            report.policy_time += run.policy_time;
            report.rules_time += run.rules_time;
            for seat in Seat::ALL {
                let seat_index = seat.index();
                report.score_totals[seat_index] += run.scores[seat_index] as i64;
                if run.winners[seat_index] {
                    report.winners[seat_index] += 1;
                }
            }
            match run.end_reason {
                FinishReason::ThreeWinners => report.three_winners += 1,
                FinishReason::WallExhausted => report.wall_exhausted += 1,
            }
        }

        report.total_time = started.elapsed();
        Ok(report)
    }

    fn drive(&mut self, game: &mut Game, record_trace: bool) -> Result<RunReport, RunnerError> {
        let mut decisions = 0u64;
        let mut policy_time = Duration::ZERO;
        let mut rules_time = Duration::ZERO;
        let mut trace = Vec::new();

        loop {
            let started = Instant::now();
            let pending = game.next_decision();
            rules_time += started.elapsed();

            let (seat, action) = match pending {
                Some(decision) => {
                    if decisions >= self.decision_limit {
                        return Err(RunnerError::DecisionLimit {
                            limit: self.decision_limit,
                        });
                    }
                    let seat = decision.seat;
                    let policy = &mut self.policies[seat.index()];
                    let started = Instant::now();
                    let action = policy.choose(&decision);
                    policy_time += started.elapsed();
                    (seat, action)
                }
                None => break,
            };

            let started = Instant::now();
            let submitted = game.submit(seat, action);
            rules_time += started.elapsed();

            if let Err(source) = submitted {
                return Err(RunnerError::IllegalAction {
                    seat,
                    action,
                    source,
                });
            }

            decisions += 1;
            if record_trace {
                trace.push(DecisionRecord { seat, action });
            }
        }

        let result = game.result().ok_or(RunnerError::MissingResult)?;
        Ok(RunReport {
            scores: result.scores,
            winners: result.winners,
            end_reason: match result.end_reason {
                EndReason::ThreeWinners => FinishReason::ThreeWinners,
                EndReason::WallExhausted => FinishReason::WallExhausted,
            },
            decisions,
            policy_time,
            rules_time,
            trace,
        })
    }
}
