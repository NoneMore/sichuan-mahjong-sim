//! Shared fixtures and module wiring for the policy A/B tool tests.
//!
//! Plan and aggregation fixtures are shared across themes; output-directory
//! setup lives with the artifact cases.

use super::cli::*;
use super::comparison::*;
use super::records::*;
use super::report::*;
use super::*;

const LIMIT: u64 = 10_000;

fn plan(seeds: Vec<u64>, style_a: Style, style_b: Style) -> Plan {
    Plan {
        seeds,
        policy_seed: 7,
        decision_limit: LIMIT,
        style_a,
        style_b,
        cache_budget: CacheBudget::CandidateDefault,
    }
}

fn plan_with_cache(seeds: Vec<u64>, style_a: Style, style_b: Style, bytes: usize) -> Plan {
    Plan {
        cache_budget: CacheBudget::Explicit(bytes),
        ..plan(seeds, style_a, style_b)
    }
}

fn default_plan(seeds: Vec<u64>) -> Plan {
    plan(seeds, Style::First, Style::Random)
}

/// Synthesizes a zero-sum record where the two A players score `x` each and
/// the two B players score `-x` each, so `a = x`, `b = -x`, `d = 2x`.
fn fixture_record(seed: u64, pair_index: usize, x: i32) -> Record {
    let pair = A_PAIRS[pair_index];
    let mut scores = [0i32; PLAYER_COUNT];
    for seat in Seat::ALL {
        scores[seat.index()] = if pair.contains(&seat) { x } else { -x };
    }
    Record {
        seed,
        pair,
        scores,
        end_reason: FinishReason::ThreeWinners,
        decisions: 1,
    }
}

fn fixture_records(plan: &Plan, x_values: &[&[i32; 6]]) -> Vec<Record> {
    let mut records = Vec::new();
    for (seed, values) in plan.seeds.iter().zip(x_values) {
        for (index, value) in values.iter().enumerate() {
            records.push(fixture_record(*seed, index, *value));
        }
    }
    records
}

fn approx(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-9, "expected {right}, got {left}");
}

#[path = "tests/artifacts.rs"]
mod artifacts;
#[path = "tests/cli.rs"]
mod cli;
#[path = "tests/execution.rs"]
mod execution;
#[path = "tests/records.rs"]
mod records;
