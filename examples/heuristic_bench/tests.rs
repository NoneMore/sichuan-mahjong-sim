//! Shared fixtures and module wiring for the heuristic benchmark tests.
//!
//! Configuration and snapshot builders are shared across themes; report,
//! sampling, seed-planning, and failure helpers live with their own cases.

use super::cli::*;
use super::phases::*;
use super::report::*;
use super::*;
use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;

fn config(games: u64, seed: u64, warmup: u64) -> Config {
    validate(
        games,
        seed,
        Lifecycle::Cold,
        warmup,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault,
    )
    .expect("fixture configuration must be valid")
}

fn warm_config(games: u64, seed: u64, warmup: u64) -> Config {
    validate(
        games,
        seed,
        Lifecycle::Warm,
        warmup,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault,
    )
    .expect("fixture warm configuration must be valid")
}

fn config_with_budget(
    games: u64,
    seed: u64,
    warmup: u64,
    lifecycle: Lifecycle,
    cache_budget_bytes: usize,
) -> Config {
    validate(
        games,
        seed,
        lifecycle,
        warmup,
        cache_budget_bytes,
        BudgetSource::Explicit,
    )
    .expect("fixture budget configuration must be valid")
}

fn snapshot(state: &'static str, allocated_bytes: usize, entries: usize) -> CacheSnapshot {
    CacheSnapshot {
        state,
        budget_bytes: DEFAULT_BUDGET_BYTES,
        context_overhead_bytes: 256,
        allocated_bytes,
        capacity_entries: entries,
        occupied_entries: entries,
        evictions: 0,
        hits: 0,
        misses: 0,
    }
}

#[path = "tests/cli.rs"]
mod cli;
#[path = "tests/errors.rs"]
mod errors;
#[path = "tests/lifecycle.rs"]
mod lifecycle;
#[path = "tests/report.rs"]
mod report;
