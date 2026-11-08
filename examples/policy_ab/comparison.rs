//! Per-game execution and the placement loop for one comparison plan.
//!
//! Every game builds fresh seat policies and a fresh [`Runner`], so no policy
//! stream continues between games or depends on loop order. Only an explicitly
//! supplied heuristic cache context outlives a game.

use std::io::Write;

use super::cache_tool::SharedCache;
use super::records::{format_record, validate_record, SeedIndex};
use super::{seat, Plan, Record, Style, A_PAIRS};
use sichuan_mahjong_sim::policy::{random_seat_seed, Policy};
use sichuan_mahjong_sim::runner::{RunReport, Runner};
use sichuan_mahjong_sim::{Seat, PLAYER_COUNT};

/// Plays every planned placement in fixed order and writes each record as its
/// game finishes.
///
/// Only the persisted rows are produced here; the summary is later recomputed
/// from the written file, which is the sole statistics input.
#[cfg(test)]
pub(super) fn run_plan(plan: &Plan, out: &mut dyn Write) -> Result<(), String> {
    run_plan_with_cache(plan, out, None)
}

/// Plays the plan with an optional shared heuristic context.
///
/// The context is used only by heuristic seats and is shared across every
/// placement and game of the run; all other styles build their normal policies.
pub(super) fn run_plan_with_cache(
    plan: &Plan,
    out: &mut dyn Write,
    cache: Option<&SharedCache>,
) -> Result<(), String> {
    let index = SeedIndex::new(plan);
    for &seed in &plan.seeds {
        for pair in A_PAIRS {
            let report = play_with_cache(
                seed,
                pair,
                plan.style_a,
                plan.style_b,
                plan.policy_seed,
                plan.decision_limit,
                cache,
            )
            .map_err(|error| {
                format!(
                    "game failed for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;

            let record = Record {
                seed,
                pair,
                scores: report.scores,
                end_reason: report.end_reason,
                decisions: report.decisions,
            };
            validate_record(&index, &record).map_err(|error| {
                format!(
                    "seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
            writeln!(out, "{}", format_record(&record)).map_err(|error| {
                format!(
                    "failed to write record for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
            out.flush().map_err(|error| {
                format!(
                    "failed to flush record for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
        }
    }
    Ok(())
}

/// Plays one fresh game with fresh seat policies and a fresh runner.
///
/// Every seat's policy stream starts from
/// `random_seat_seed(seed ^ policy_seed, seat)`. A deterministic policy ignores
/// the stream, and a random policy starts exactly there, so results do not
/// depend on earlier games or loop order.
#[cfg(test)]
pub(super) fn play_with(
    seed: u64,
    pair: [Seat; 2],
    style_a: Style,
    style_b: Style,
    policy_seed: u64,
    decision_limit: u64,
) -> Result<RunReport, String> {
    play_with_cache(
        seed,
        pair,
        style_a,
        style_b,
        policy_seed,
        decision_limit,
        None,
    )
}

/// Plays one game with fresh seat policies and an optional shared context.
pub(super) fn play_with_cache(
    seed: u64,
    pair: [Seat; 2],
    style_a: Style,
    style_b: Style,
    policy_seed: u64,
    decision_limit: u64,
    cache: Option<&SharedCache>,
) -> Result<RunReport, String> {
    let mut a_seats = [false; PLAYER_COUNT];
    for seat in pair {
        a_seats[seat.index()] = true;
    }

    let policies: [Box<dyn Policy>; PLAYER_COUNT] = std::array::from_fn(|index| {
        let seat = seat(index);
        let stream_seed = random_seat_seed(seed ^ policy_seed, seat);
        if a_seats[index] {
            style_a.build(stream_seed, cache)
        } else {
            style_b.build(stream_seed, cache)
        }
    });

    Runner::new(policies)
        .with_decision_limit(decision_limit)
        .run_new(seed)
        .map_err(|error| error.to_string())
}
