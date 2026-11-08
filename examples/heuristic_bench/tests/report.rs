//! Measurement report rendering coverage.

use super::*;
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;

/// Batch aggregate with hand-computable report values:
/// `4 games / 0.5 s = 8 games/s` and `2000 us / 400 decisions = 5 us`.
fn fixture_batch() -> BatchReport {
    BatchReport {
        games: 4,
        decisions: 400,
        policy_time: std::time::Duration::from_micros(2_000),
        rules_time: std::time::Duration::from_micros(1_000),
        total_time: std::time::Duration::from_millis(500),
        score_totals: [12, -4, -5, -3],
        winners: [3, 3, 2, 1],
        three_winners: 3,
        wall_exhausted: 1,
    }
}

fn fixture_phases() -> Phases {
    Phases {
        warmup: Some(WarmupPhase {
            batch: fixture_batch(),
            cache_before: snapshot("unallocated", 0, 0),
            cache_after: snapshot("ready", 1_000, 10),
            scope: "separate-context",
        }),
        measured: fixture_batch(),
        measured_cache_before: snapshot("unallocated", 0, 0),
        measured_cache_after: snapshot("ready", 2_000, 20),
        warnings: Vec::new(),
    }
}

#[test]
fn report_states_workload_units_cache_lifecycle_and_game_results() {
    let workload = config(4, 1234, 2);
    let text = render_report(&workload, &fixture_phases());
    let expected = format!(
        "protocol: heuristic-benchmark-v2\n\
         policy: heuristic\n\
         seats: 4\n\
         seat_order: [0, 1, 2, 3]\n\
         cache_scope: four-seat-shared-context\n\
         cache_lifecycle: cold\n\
         cache_budget_source: candidate-default\n\
         cache_budget_bytes: {DEFAULT_BUDGET_BYTES}\n\
         cache_warmup_scope: separate-context\n\
         warmup_games: 2\n\
         warmup_first_seed: 1232\n\
         warmup_completed_games: 4\n\
         warmup_decisions: 400\n\
         warmup_total_time_seconds: 0.500000\n\
         warmup_policy_time_seconds: 0.002000\n\
         warmup_rules_time_seconds: 0.001000\n\
         warmup_cache_before_state: unallocated\n\
         warmup_cache_before_budget_bytes: {DEFAULT_BUDGET_BYTES}\n\
         warmup_cache_before_context_overhead_bytes: 256\n\
         warmup_cache_before_allocated_bytes: 0\n\
         warmup_cache_before_capacity_entries: 0\n\
         warmup_cache_before_occupied_entries: 0\n\
         warmup_cache_before_evictions: 0\n\
         warmup_cache_before_hits: 0\n\
         warmup_cache_before_misses: 0\n\
         warmup_cache_after_state: ready\n\
         warmup_cache_after_budget_bytes: {DEFAULT_BUDGET_BYTES}\n\
         warmup_cache_after_context_overhead_bytes: 256\n\
         warmup_cache_after_allocated_bytes: 1000\n\
         warmup_cache_after_capacity_entries: 10\n\
         warmup_cache_after_occupied_entries: 10\n\
         warmup_cache_after_evictions: 0\n\
         warmup_cache_after_hits: 0\n\
         warmup_cache_after_misses: 0\n\
         seed: 1234\n\
         games: 4\n\
         decision_limit_per_game: {DEFAULT_DECISION_LIMIT}\n\
         cache_before_measured_state: unallocated\n\
         cache_before_measured_budget_bytes: {DEFAULT_BUDGET_BYTES}\n\
         cache_before_measured_context_overhead_bytes: 256\n\
         cache_before_measured_allocated_bytes: 0\n\
         cache_before_measured_capacity_entries: 0\n\
         cache_before_measured_occupied_entries: 0\n\
         cache_before_measured_evictions: 0\n\
         cache_before_measured_hits: 0\n\
         cache_before_measured_misses: 0\n\
         completed_games: 4\n\
         decisions: 400\n\
         total_time_seconds: 0.500000\n\
         policy_time_seconds: 0.002000\n\
         rules_time_seconds: 0.001000\n\
         cache_after_measured_state: ready\n\
         cache_after_measured_budget_bytes: {DEFAULT_BUDGET_BYTES}\n\
         cache_after_measured_context_overhead_bytes: 256\n\
         cache_after_measured_allocated_bytes: 2000\n\
         cache_after_measured_capacity_entries: 20\n\
         cache_after_measured_occupied_entries: 20\n\
         cache_after_measured_evictions: 0\n\
         cache_after_measured_hits: 0\n\
         cache_after_measured_misses: 0\n\
         cache_at_capacity_after_measured: yes\n\
         cache_init_phase: measured\n\
         cache_init_timing_boundary: included-in-measured-total-time\n\
         cache_warning_count: 0\n\
         games_per_sec: 8.000\n\
         decisions_per_sec: 800.000\n\
         mean_policy_us_per_decision: 5.000\n\
         three_winners: 3\n\
         wall_exhausted: 1\n\
         score_totals: [12, -4, -5, -3]\n\
         score_sum: 0\n\
         winners: [3, 3, 2, 1]\n"
    );
    assert_eq!(text, expected);
    for line in text.lines() {
        let (key, value) = line.split_once(": ").expect("every report line has a key");
        assert!(!key.contains(' '), "{line}");
        assert!(!value.is_empty(), "{line}");
        assert!(
            !value.contains(" ("),
            "value should not contain prose: {line}"
        );
    }
}

#[test]
fn warm_report_describes_the_inherited_initialization_boundary() {
    let workload = warm_config(2, 9, 1);
    let phases = Phases {
        warmup: Some(WarmupPhase {
            batch: fixture_batch(),
            cache_before: snapshot("unallocated", 0, 0),
            cache_after: snapshot("ready", 1_000, 10),
            scope: "inherited-context",
        }),
        measured: fixture_batch(),
        measured_cache_before: snapshot("ready", 1_000, 10),
        measured_cache_after: snapshot("ready", 2_000, 20),
        warnings: Vec::new(),
    };
    let text = render_report(&workload, &phases);
    assert!(text.contains("cache_lifecycle: warm\n"), "{text}");
    assert!(
        text.contains("cache_warmup_scope: inherited-context\n"),
        "{text}"
    );
    assert!(text.contains("cache_init_phase: warmup\n"), "{text}");
    assert!(
        text.contains("cache_init_timing_boundary: excluded-from-measured-total-time\n"),
        "{text}"
    );
    assert!(
        text.contains("cache_before_measured_state: ready\n"),
        "{text}"
    );
}

#[test]
fn disabled_and_unallocated_contexts_report_explicit_boundaries() {
    let workload = config_with_budget(1, 5, 0, Lifecycle::Cold, 0);
    let phases = Phases {
        warmup: None,
        measured: fixture_batch(),
        measured_cache_before: CacheSnapshot {
            state: "disabled",
            budget_bytes: 0,
            context_overhead_bytes: 256,
            allocated_bytes: 0,
            capacity_entries: 0,
            occupied_entries: 0,
            evictions: 0,
            hits: 0,
            misses: 0,
        },
        measured_cache_after: CacheSnapshot {
            state: "disabled",
            budget_bytes: 0,
            context_overhead_bytes: 256,
            allocated_bytes: 0,
            capacity_entries: 0,
            occupied_entries: 0,
            evictions: 0,
            hits: 0,
            misses: 0,
        },
        warnings: Vec::new(),
    };
    let text = render_report(&workload, &phases);
    assert!(text.contains("cache_budget_source: explicit\n"), "{text}");
    assert!(text.contains("cache_budget_bytes: 0\n"), "{text}");
    assert!(text.contains("cache_warmup_scope: none\n"), "{text}");
    assert!(text.contains("cache_init_phase: disabled\n"), "{text}");
    assert!(
        text.contains("cache_init_timing_boundary: not-applicable\n"),
        "{text}"
    );
    assert!(text.contains("warmup_first_seed: none\n"), "{text}");
    assert!(text.contains("warmup_cache_before_state: none\n"), "{text}");
}

#[test]
fn unavailable_rates_and_zero_decisions_render_without_nan() {
    let mut batch = fixture_batch();
    batch.decisions = 0;
    batch.policy_time = std::time::Duration::ZERO;
    batch.total_time = std::time::Duration::ZERO;
    let phases = Phases {
        warmup: None,
        measured: batch,
        measured_cache_before: snapshot("unallocated", 0, 0),
        measured_cache_after: snapshot("unallocated", 0, 0),
        warnings: Vec::new(),
    };
    let text = render_report(&config(1, 0, 0), &phases);

    for field in [
        "games_per_sec: unavailable\n",
        "decisions_per_sec: unavailable\n",
        "mean_policy_us_per_decision: unavailable\n",
    ] {
        assert!(text.contains(field), "{text}");
    }
    assert!(!text.contains("NaN"), "{text}");
    assert!(!text.contains("inf"), "{text}");
}

#[test]
fn rates_use_completed_games_and_handle_zero_decisions_with_nonzero_time() {
    let mut batch = fixture_batch();
    batch.decisions = 0;
    let phases = Phases {
        warmup: None,
        measured: batch,
        measured_cache_before: snapshot("unallocated", 0, 0),
        measured_cache_after: snapshot("unallocated", 0, 0),
        warnings: Vec::new(),
    };
    let text = render_report(&config(5, 0, 0), &phases);
    assert!(text.contains("games: 5\n"));
    assert!(text.contains("completed_games: 4\n"));
    assert!(text.contains("games_per_sec: 8.000\n"));
    assert!(text.contains("decisions_per_sec: 0.000\n"));
    assert!(text.contains("mean_policy_us_per_decision: unavailable\n"));
}

#[test]
fn a_successful_run_writes_the_measured_report() {
    let mut buffer = Vec::new();
    execute(&config(1, 5, 0), heuristic_runner, &mut buffer).unwrap();
    let text = String::from_utf8(buffer).expect("report is UTF-8");
    assert!(text.contains("protocol: heuristic-benchmark-v2"), "{text}");
    assert!(text.contains("games: 1\n"), "{text}");
    assert!(
        text.contains("cache_scope: four-seat-shared-context\n"),
        "{text}"
    );
    assert!(text.contains("cache_budget_bytes: "), "{text}");
    assert!(
        text.contains("cache_after_measured_context_overhead_bytes: "),
        "{text}"
    );
    assert!(text.contains("games_per_sec: "), "{text}");
    assert!(text.contains("score_sum: 0\n"), "{text}");
}
