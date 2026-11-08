//! Cache lifecycle, phase execution, and shared-context coverage.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::cache_tool::CacheWarnings;
use sichuan_mahjong_sim::policy::HeuristicPolicy;

/// The non-time fields that must be reproducible for a fixed workload.
type NonTime = (u64, u64, [i64; 4], [u64; 4], u64, u64);

fn non_time(batch: &BatchReport) -> NonTime {
    (
        batch.games,
        batch.decisions,
        batch.score_totals,
        batch.winners,
        batch.three_winners,
        batch.wall_exhausted,
    )
}

/// Runs the real heuristic workload.
fn sample(config: &Config) -> Phases {
    measure(config, heuristic_runner).expect("heuristic batch must complete")
}

#[test]
fn cold_warmup_uses_a_separate_context_and_measurement_starts_unallocated() {
    let workload = config(2, 21, 2);
    let phases = sample(&workload);
    let warmup = phases
        .warmup
        .as_ref()
        .expect("cold warmup must be retained");
    assert_eq!(warmup.scope, "separate-context");
    // The measured context is created after the warmup context is dropped, so
    // it cannot inherit any tables or reservation state.
    assert_eq!(phases.measured_cache_before.state, "unallocated");
    assert_eq!(phases.measured_cache_before.allocated_bytes, 0);
    assert_eq!(phases.measured_cache_before.capacity_entries, 0);
    assert_eq!(phases.measured_cache_before.occupied_entries, 0);
    // Two games are enough to exercise cross-decision analysis; if the core
    // still reports unallocated here, the report must say no initialization
    // happened rather than claiming one.
    assert_ne!(phases.measured_cache_after.state, "disabled");
    let phase = cache_init_phase(&workload, &phases);
    assert!(
        matches!(phase, "measured" | "none"),
        "unexpected cold init phase {phase}"
    );
    if phases.measured_cache_after.state != "unallocated" {
        assert_eq!(phase, "measured");
        assert_eq!(
            cache_init_timing_boundary(phase),
            "included-in-measured-total-time"
        );
    }
}

#[test]
fn warm_measurement_inherits_the_warmup_context() {
    let workload = warm_config(1, 21, 2);
    let phases = sample(&workload);
    let warmup = phases
        .warmup
        .as_ref()
        .expect("warm warmup must be retained");
    assert_eq!(warmup.scope, "inherited-context");
    assert_eq!(phases.measured_cache_before, warmup.cache_after);
    assert_ne!(phases.measured_cache_before.state, "disabled");
    if phases.measured_cache_before.state != "unallocated" {
        assert_eq!(cache_init_phase(&workload, &phases), "warmup");
        assert_eq!(
            cache_init_timing_boundary("warmup"),
            "excluded-from-measured-total-time"
        );
    }
}

#[test]
fn budget_zero_disables_allocation_and_reports_disabled() {
    let workload = config_with_budget(1, 31, 0, Lifecycle::Cold, 0);
    let phases = sample(&workload);
    assert_eq!(phases.measured_cache_before.state, "disabled");
    assert_eq!(phases.measured_cache_after.state, "disabled");
    for snapshot in [phases.measured_cache_before, phases.measured_cache_after] {
        assert_eq!(snapshot.budget_bytes, 0);
        assert_eq!(snapshot.allocated_bytes, 0);
        assert_eq!(snapshot.capacity_entries, 0);
        assert_eq!(snapshot.occupied_entries, 0);
    }
    assert_eq!(cache_init_phase(&workload, &phases), "disabled");
    assert_eq!(cache_init_timing_boundary("disabled"), "not-applicable");
    assert!(phases.warnings.is_empty());
}

#[test]
fn enabled_and_disabled_caches_agree_on_non_time_results() {
    let enabled = sample(&config(2, 300, 0));
    let disabled = sample(&config_with_budget(2, 300, 0, Lifecycle::Cold, 0));
    assert_eq!(non_time(&enabled.measured), non_time(&disabled.measured));
}

#[test]
fn cold_and_warm_lifecycles_agree_on_non_time_results() {
    let cold = sample(&config(2, 300, 0));
    let warmed = sample(&warm_config(2, 300, 2));
    assert_eq!(non_time(&cold.measured), non_time(&warmed.measured));
}

#[test]
fn phases_build_their_own_runner_and_reuse_policies_within_a_batch() {
    let policy_count = Arc::new(AtomicUsize::new(0));

    for (lifecycle, warmup, expected_runners, expected_policies) in [
        (Lifecycle::Cold, 0, 1, 4),
        (Lifecycle::Cold, 2, 2, 8),
        (Lifecycle::Warm, 2, 1, 4),
    ] {
        let workload = validate(
            2,
            21,
            lifecycle,
            warmup,
            DEFAULT_BUDGET_BYTES,
            BudgetSource::CandidateDefault,
        )
        .unwrap();
        let mut factory_calls = 0;
        let policies = Arc::clone(&policy_count);
        let before = policies.load(Ordering::SeqCst);
        let phases = measure(&workload, |cache| {
            factory_calls += 1;
            Runner::new(cache.group().map(|policy| {
                policies.fetch_add(1, Ordering::SeqCst);
                Box::new(policy) as Box<dyn Policy>
            }))
        })
        .expect("both phases must complete");

        assert_eq!(phases.measured.games, 2);
        assert_eq!(
            factory_calls,
            expected_runners,
            "lifecycle {} warmup {warmup}",
            lifecycle.name()
        );
        assert_eq!(
            policies.load(Ordering::SeqCst) - before,
            expected_policies,
            "lifecycle {} warmup {warmup}",
            lifecycle.name()
        );
    }
}

#[test]
fn warmup_is_skipped_without_an_empty_batch_when_disabled() {
    let mut factory_calls = 0;
    let phases = measure(&config(1, 33, 0), |cache| {
        factory_calls += 1;
        heuristic_runner(cache)
    })
    .expect("a zero-warmup workload must run only the measured phase");
    assert_eq!(factory_calls, 1);
    assert!(phases.warmup.is_none());
    assert_eq!(phases.measured.games, 1);
}

#[test]
fn warmup_results_do_not_enter_the_measured_report() {
    let cold = sample(&config(2, 300, 0));
    let warmed_up_process = sample(&config(2, 300, 2));
    let inherited = sample(&warm_config(2, 300, 2));
    assert_eq!(
        non_time(&cold.measured),
        non_time(&warmed_up_process.measured)
    );
    assert_eq!(non_time(&cold.measured), non_time(&inherited.measured));
}

#[test]
fn repeated_runs_produce_identical_non_time_results() {
    let workload = config(2, 17, 0);
    assert_eq!(
        non_time(&sample(&workload).measured),
        non_time(&sample(&workload).measured)
    );
}

#[test]
fn a_small_measured_batch_completes_every_game_with_zero_sum_scores() {
    let phases = sample(&config(3, 99, 1));
    assert_eq!(phases.measured.games, 3);
    assert!(phases.measured.decisions > 0);
    assert_eq!(
        phases.measured.three_winners + phases.measured.wall_exhausted,
        3
    );
    assert_eq!(phases.measured.score_sum(), 0);
}

#[test]
fn allocation_failure_warns_once_per_context_lifetime() {
    let failed = CacheSnapshot {
        state: "allocation-failed",
        budget_bytes: DEFAULT_BUDGET_BYTES,
        context_overhead_bytes: 256,
        allocated_bytes: 0,
        capacity_entries: 0,
        occupied_entries: 0,
        evictions: 0,
        hits: 0,
        misses: 0,
    };
    let mut warner = CacheWarnings::default();
    let first = warner.observe(failed, "benchmark measured");
    assert!(first.is_some(), "a failed context must warn");
    assert!(first.unwrap().contains("benchmark measured"));
    assert!(
        warner.observe(failed, "benchmark measured").is_none(),
        "the same context must not warn twice"
    );

    let mut healthy = CacheWarnings::default();
    assert_eq!(
        healthy.observe(snapshot("ready", 1, 1), "benchmark measured"),
        None
    );

    // A cold run owns a separate warmup context and a measured context; each
    // lifetime gets its own single warning.
    let mut warmup = CacheWarnings::default();
    let mut measured = CacheWarnings::default();
    assert!(warmup.observe(failed, "benchmark warmup").is_some());
    assert!(warmup.observe(failed, "benchmark warmup").is_none());
    assert!(measured.observe(failed, "benchmark measured").is_some());
    assert!(measured.observe(failed, "benchmark measured").is_none());
}

#[test]
fn library_shared_group_binds_one_default_context() {
    let group = HeuristicPolicy::shared_group();
    let cache = group[0].cache().clone();
    let initial = cache.status();
    assert_eq!(initial.budget_bytes, DEFAULT_BUDGET_BYTES);
    assert_eq!(initial.allocated_bytes, 0);

    let mut runner = Runner::new(group.map(|policy| Box::new(policy) as Box<dyn Policy>));
    let report = runner.run_new(7).expect("shared-group game runs");
    assert!(report.decisions > 0);

    let status = cache.status();
    assert!(status.allocated_bytes > 0);
    assert!(status.allocated_bytes <= status.budget_bytes);
    assert!(status.hits + status.misses > 0);
}
