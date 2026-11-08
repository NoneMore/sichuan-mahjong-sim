//! Measurement report text for one benchmark run.
//!
//! Units belong to field names and every line is a single stable `key: value`,
//! so the report can be consumed without prose parsing. Nothing here plays a
//! game or prints; the caller owns the output stream.

use super::cache_tool::snapshot_lines;
use super::{Config, Lifecycle, Phases};
use sichuan_mahjong_sim::runner::{BatchReport, DEFAULT_DECISION_LIMIT};
use sichuan_mahjong_sim::PLAYER_COUNT;

/// Workload protocol identifier reported with every measurement.
const PROTOCOL: &str = "heuristic-benchmark-v2";

/// Policy used in all four seats.
const POLICY_NAME: &str = "heuristic";

/// Rate per measured batch wall second, unavailable when the measured
/// duration is zero. Both games and decisions use this same denominator.
fn rate_per_second(count: u64, batch: &BatchReport) -> Option<f64> {
    let seconds = batch.total_time.as_secs_f64();
    if seconds > 0.0 {
        Some(count as f64 / seconds)
    } else {
        None
    }
}

/// Mean policy time per decision in microseconds over every decision kind, or
/// unavailable when the measured batch recorded no decisions.
fn mean_policy_time_us(batch: &BatchReport) -> Option<f64> {
    if batch.decisions == 0 {
        None
    } else {
        Some(batch.policy_time.as_secs_f64() * 1_000_000.0 / batch.decisions as f64)
    }
}

/// Where the measured context obtained its table allocation.
///
/// `warmup` means the inherited warm context was already ready (or had
/// already failed) when measurement began. `measured` means the measured batch
/// observed the transition from unallocated to ready or failed and therefore
/// includes the lazy initialization attempt. `none` means the context never
/// allocated; `disabled` means no table allocation is possible.
pub(super) fn cache_init_phase(config: &Config, phases: &Phases) -> &'static str {
    let before = phases.measured_cache_before;
    let after = phases.measured_cache_after;
    if before.is_disabled() && after.is_disabled() {
        return "disabled";
    }
    if before.is_allocated() || before.is_allocation_failed() {
        return if config.lifecycle == Lifecycle::Warm {
            "warmup"
        } else {
            "pre-measured"
        };
    }
    if after.state == "ready" || after.is_allocation_failed() {
        return "measured";
    }
    "none"
}

/// Timing boundary that contains the measured context's initialization.
pub(super) fn cache_init_timing_boundary(phase: &str) -> &'static str {
    match phase {
        "warmup" => "excluded-from-measured-total-time",
        "measured" => "included-in-measured-total-time",
        _ => "not-applicable",
    }
}

/// Formats one measured batch as stable, single-value "key: value" lines.
///
/// Units belong to field names, rather than natural-language value suffixes.
/// The protocol identifier distinguishes this workload and its timing and cache
/// lifecycle semantics. No prose or extra lines are needed to consume the
/// report.
pub(super) fn render_report(config: &Config, phases: &Phases) -> String {
    let measured = &phases.measured;
    let mut text = format!(
        "protocol: {PROTOCOL}\n\
         policy: {POLICY_NAME}\n\
         seats: {PLAYER_COUNT}\n\
         seat_order: [0, 1, 2, 3]\n\
         cache_scope: four-seat-shared-context\n\
         cache_lifecycle: {}\n\
         cache_budget_source: {}\n\
         cache_budget_bytes: {}\n\
         cache_warmup_scope: {}\n\
         warmup_games: {}\n",
        config.lifecycle.name(),
        config.cache_budget_source.name(),
        config.cache_budget_bytes,
        phases.warmup.as_ref().map_or("none", |warmup| warmup.scope),
        config.warmup
    );

    match &phases.warmup {
        Some(warmup) => {
            text.push_str(&format!(
                "warmup_first_seed: {}\n\
                 warmup_completed_games: {}\n\
                 warmup_decisions: {}\n\
                 warmup_total_time_seconds: {:.6}\n\
                 warmup_policy_time_seconds: {:.6}\n\
                 warmup_rules_time_seconds: {:.6}\n",
                config.warmup_start(),
                warmup.batch.games,
                warmup.batch.decisions,
                warmup.batch.total_time.as_secs_f64(),
                warmup.batch.policy_time.as_secs_f64(),
                warmup.batch.rules_time.as_secs_f64()
            ));
            text.push_str(&snapshot_lines("warmup_cache_before", warmup.cache_before));
            text.push_str(&snapshot_lines("warmup_cache_after", warmup.cache_after));
        }
        None => {
            text.push_str(
                "warmup_first_seed: none\n\
                 warmup_completed_games: none\n\
                 warmup_decisions: none\n\
                 warmup_total_time_seconds: none\n\
                 warmup_policy_time_seconds: none\n\
                 warmup_rules_time_seconds: none\n",
            );
            for prefix in ["warmup_cache_before", "warmup_cache_after"] {
                text.push_str(&format!(
                    "{prefix}_state: none\n\
                     {prefix}_budget_bytes: none\n\
                     {prefix}_context_overhead_bytes: none\n\
                     {prefix}_allocated_bytes: none\n\
                     {prefix}_capacity_entries: none\n\
                     {prefix}_occupied_entries: none\n\
                     {prefix}_evictions: none\n\
                     {prefix}_hits: none\n\
                     {prefix}_misses: none\n"
                ));
            }
        }
    }

    text.push_str(&format!(
        "seed: {}\n\
         games: {}\n\
         decision_limit_per_game: {DEFAULT_DECISION_LIMIT}\n",
        config.seed, config.games
    ));
    text.push_str(&snapshot_lines(
        "cache_before_measured",
        phases.measured_cache_before,
    ));
    text.push_str(&format!(
        "completed_games: {}\n\
         decisions: {}\n\
         total_time_seconds: {:.6}\n\
         policy_time_seconds: {:.6}\n\
         rules_time_seconds: {:.6}\n",
        measured.games,
        measured.decisions,
        measured.total_time.as_secs_f64(),
        measured.policy_time.as_secs_f64(),
        measured.rules_time.as_secs_f64()
    ));
    text.push_str(&snapshot_lines(
        "cache_after_measured",
        phases.measured_cache_after,
    ));
    text.push_str(&format!(
        "cache_at_capacity_after_measured: {}\n",
        if phases.measured_cache_after.is_at_capacity() {
            "yes"
        } else {
            "no"
        }
    ));
    let init_phase = cache_init_phase(config, phases);
    text.push_str(&format!(
        "cache_init_phase: {init_phase}\n\
         cache_init_timing_boundary: {}\n\
         cache_warning_count: {}\n",
        cache_init_timing_boundary(init_phase),
        phases.warnings.len()
    ));

    match rate_per_second(measured.games, measured) {
        Some(rate) => text.push_str(&format!("games_per_sec: {rate:.3}\n")),
        None => text.push_str("games_per_sec: unavailable\n"),
    }
    match rate_per_second(measured.decisions, measured) {
        Some(rate) => text.push_str(&format!("decisions_per_sec: {rate:.3}\n")),
        None => text.push_str("decisions_per_sec: unavailable\n"),
    }
    match mean_policy_time_us(measured) {
        Some(mean) => text.push_str(&format!("mean_policy_us_per_decision: {mean:.3}\n")),
        None => text.push_str("mean_policy_us_per_decision: unavailable\n"),
    }

    text.push_str(&format!(
        "three_winners: {}\n\
         wall_exhausted: {}\n\
         score_totals: {:?}\n\
         score_sum: {}\n\
         winners: {:?}\n",
        measured.three_winners,
        measured.wall_exhausted,
        measured.score_totals,
        measured.score_sum(),
        measured.winners
    ));
    text
}
