use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::cache_tool::{CacheSnapshot, CacheWarnings, SharedCache};
use super::*;
use sichuan_mahjong_sim::policy::{HeuristicPolicy, DEFAULT_BUDGET_BYTES};

struct RejectingPolicy;

impl Policy for RejectingPolicy {
    fn choose(
        &mut self,
        _decision: &sichuan_mahjong_sim::Decision<'_>,
    ) -> sichuan_mahjong_sim::Action {
        // Pass is never a legal exchange, so the runner must reject it.
        sichuan_mahjong_sim::Action::Pass
    }
}

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

/// Runner whose policies always return an illegal action.
fn rejecting_runner(_cache: &SharedCache) -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(RejectingPolicy) as Box<dyn Policy>
    }))
}

/// Measured then warmup seed sequences, in play order.
fn seed_ranges(config: &Config) -> (Vec<u64>, Vec<u64>) {
    let measured = (0..config.games).map(|i| config.measure_seed(i)).collect();
    let warmup = (0..config.warmup).map(|i| config.warmup_seed(i)).collect();
    (measured, warmup)
}

#[test]
fn help_is_standalone_and_ignores_later_arguments() {
    for arguments in [vec!["-h"], vec!["--help"], vec!["--help", "--games", "0"]] {
        let command = parse_args(arguments.iter().map(|value| (*value).to_owned())).unwrap();
        assert_eq!(command, Command::Help, "{arguments:?}");
    }
}

#[test]
fn games_and_seed_are_required_and_defaults_are_explicit() {
    let parsed = parse_args(
        ["--games", "2", "--seed", "3"]
            .iter()
            .map(|value| (*value).to_owned()),
    )
    .unwrap();
    assert_eq!(
        parsed,
        Command::Run(Config {
            games: 2,
            seed: 3,
            warmup: 0,
            lifecycle: Lifecycle::Cold,
            cache_budget_bytes: DEFAULT_BUDGET_BYTES,
            cache_budget_source: BudgetSource::CandidateDefault,
        })
    );

    for missing in ["--games", "--seed"] {
        let others: Vec<&str> = [("--games", "2"), ("--seed", "3"), ("--warmup", "0")]
            .into_iter()
            .filter(|(option, _)| *option != missing)
            .flat_map(|(option, value)| [option, value])
            .collect();
        let error = parse_args(others.iter().map(|value| (*value).to_owned())).unwrap_err();
        assert!(error.contains(missing), "{error}");
        assert!(error.contains("required"), "{error}");
    }
}

#[test]
fn missing_duplicate_and_unknown_options_are_rejected() {
    for arguments in [
        vec!["--games"],
        vec!["--games", "1", "--seed"],
        vec!["--games", "1", "--seed", "2", "--warmup"],
        vec![
            "--games", "1", "--games", "2", "--seed", "2", "--warmup", "0",
        ],
        vec!["--games", "1", "--seed", "2", "--warmup", "0", "--extra"],
        vec!["games", "1"],
    ] {
        let error = parse_args(arguments.iter().map(|value| (*value).to_owned())).unwrap_err();
        assert!(!error.is_empty(), "{arguments:?}");
    }

    let duplicate = parse_args(
        [
            "--games", "1", "--games", "2", "--seed", "0", "--warmup", "0",
        ]
        .iter()
        .map(|value| (*value).to_owned()),
    )
    .unwrap_err();
    assert_eq!(duplicate, "duplicate --games");

    let missing_value =
        parse_args(["--games"].iter().map(|value| (*value).to_owned())).unwrap_err();
    assert_eq!(missing_value, "--games requires a value");

    let unknown = parse_args(["--extra"].iter().map(|value| (*value).to_owned())).unwrap_err();
    assert_eq!(unknown, "unexpected argument '--extra'");
}

#[test]
fn numeric_options_accept_only_decimal_unsigned_64_bit_values() {
    let accepted = ["0", "7", "007", "18446744073709551615"];
    for value in accepted {
        assert_eq!(
            parse_decimal_u64("--seed", value).unwrap(),
            value.parse::<u64>().unwrap()
        );
    }

    for value in [
        "",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1_000",
        "0x10",
        "1.0",
        "1e3",
        "18446744073709551616",
    ] {
        let error = parse_decimal_u64("--games", value).unwrap_err();
        assert!(error.contains("--games"), "{error}");
        assert!(error.contains(value), "{error}");
    }
}

#[test]
fn lifecycle_and_budget_parse_with_explicit_defaults() {
    let parsed = parse_args(
        [
            "--games",
            "2",
            "--seed",
            "3",
            "--lifecycle",
            "warm",
            "--warmup",
            "4",
            "--cache-budget-bytes",
            "0",
        ]
        .iter()
        .map(|value| (*value).to_owned()),
    )
    .unwrap();
    let Command::Run(config) = parsed else {
        panic!("expected a run command");
    };
    assert_eq!(config.lifecycle, Lifecycle::Warm);
    assert_eq!(config.warmup, 4);
    assert_eq!(config.cache_budget_bytes, 0);
    assert_eq!(config.cache_budget_source, BudgetSource::Explicit);

    let invalid = |extra: &[&str]| {
        let mut arguments = vec!["--games", "1", "--seed", "0"];
        arguments.extend_from_slice(extra);
        parse_args(arguments.iter().map(|value| (*value).to_owned())).unwrap_err()
    };
    assert!(invalid(&["--lifecycle", "hot"]).contains("invalid --lifecycle"));
    assert!(
        invalid(&["--lifecycle", "cold", "--lifecycle", "warm"]).contains("duplicate --lifecycle")
    );
    assert!(invalid(&["--cache-budget-bytes", "-1"]).contains("invalid"));
    assert!(
        invalid(&["--cache-budget-bytes", "1", "--cache-budget-bytes", "2"])
            .contains("duplicate --cache-budget-bytes")
    );
    assert_eq!(invalid(&["--lifecycle"]), "--lifecycle requires a value");
    assert_eq!(
        invalid(&["--cache-budget-bytes"]),
        "--cache-budget-bytes requires a value"
    );
}

#[test]
fn warm_requires_a_warmup_and_cold_allows_none() {
    let error = parse_args(
        ["--games", "1", "--seed", "0", "--lifecycle", "warm"]
            .iter()
            .map(|value| (*value).to_owned()),
    )
    .unwrap_err();
    assert!(error.contains("requires --warmup"), "{error}");

    assert!(validate(
        1,
        0,
        Lifecycle::Warm,
        0,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_err());
    assert!(validate(
        1,
        0,
        Lifecycle::Warm,
        1,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_ok());
    assert!(validate(
        1,
        0,
        Lifecycle::Cold,
        0,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_ok());
}

#[test]
fn zero_games_is_rejected_before_play() {
    let error = parse_args(
        ["--games", "0", "--seed", "0", "--warmup", "0"]
            .iter()
            .map(|value| (*value).to_owned()),
    )
    .unwrap_err();
    assert!(error.contains("--games must be at least 1"), "{error}");
}

#[test]
fn seed_plan_wraps_backwards_into_warmup_and_forwards_past_the_maximum() {
    let wrapping = config(2, 0, 2);
    assert_eq!(wrapping.warmup_start(), u64::MAX - 1);
    assert_eq!(
        seed_ranges(&wrapping),
        (vec![0, 1], vec![u64::MAX - 1, u64::MAX])
    );

    let forward = config(2, u64::MAX, 0);
    assert_eq!(seed_ranges(&forward), (vec![u64::MAX, 0], Vec::new()));

    let plain = config(3, 5, 4);
    assert_eq!(seed_ranges(&plain), (vec![5, 6, 7], vec![1, 2, 3, 4]));

    let no_warmup = config(2, 9, 0);
    assert!(!no_warmup.warmup_enabled());
    assert_eq!(seed_ranges(&no_warmup), (vec![9, 10], Vec::new()));
}

#[test]
fn ranges_stay_adjacent_and_disjoint_and_warmup_never_moves_the_measured_seeds() {
    for (games, seed) in [(1, 0), (2, 0), (3, 5), (2, u64::MAX), (4, 100)] {
        let reference = seed_ranges(&config(games, seed, 0)).0;
        for warmup in 0..=3 {
            let workload = config(games, seed, warmup);
            let (measured, warmup_seeds) = seed_ranges(&workload);
            assert_eq!(
                measured, reference,
                "games {games} seed {seed} warmup {warmup}"
            );

            for warmup_seed in &warmup_seeds {
                assert!(
                    !measured.contains(warmup_seed),
                    "seed {warmup_seed} is in both ranges"
                );
            }
            // Warmup seeds are the W predecessors of the measured start.
            for (index, warmup_seed) in warmup_seeds.iter().enumerate() {
                assert_eq!(
                    *warmup_seed,
                    workload
                        .measure_seed(0)
                        .wrapping_sub(warmup as u64 - index as u64)
                );
            }
        }
    }
}

#[test]
fn seed_span_boundary_accepts_two_to_the_sixty_fourth_and_rejects_more() {
    // N + W == 2^64 is the full ring; the ranges meet but never overlap.
    let boundary = config(u64::MAX, 123, 1);
    assert_eq!(boundary.warmup_start(), 122);
    assert_eq!(boundary.measure_seed(u64::MAX - 1), 121);
    assert_eq!(boundary.measure_seed(0), 123);

    assert!(validate(
        u64::MAX,
        123,
        Lifecycle::Cold,
        2,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_err());
    assert!(validate(
        2,
        0,
        Lifecycle::Cold,
        u64::MAX,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_err());
    assert!(validate(
        u64::MAX,
        u64::MAX,
        Lifecycle::Cold,
        u64::MAX,
        DEFAULT_BUDGET_BYTES,
        BudgetSource::CandidateDefault
    )
    .is_err());

    // Large accepted workloads are validated arithmetically, never played.
    let large = config(1_000_000_000_000, u64::MAX - 3, 4);
    assert_eq!(large.measure_seed(3), u64::MAX);
    assert_eq!(large.warmup_seed(3), u64::MAX - 4);
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
fn cache_config_errors_are_reported_before_play_and_write_no_report() {
    // A one-byte non-zero budget is too small for any table; the library owns
    // the exact minimum and reports a configuration error at creation.
    let workload = config_with_budget(1, 5, 0, Lifecycle::Cold, 1);
    let error = measure(&workload, heuristic_runner).unwrap_err();
    assert!(
        matches!(error, BenchError::CacheConfig(_)),
        "expected a cache configuration error: {error}"
    );
    assert!(
        error.to_string().contains("invalid cache budget"),
        "{error}"
    );
    assert!(error.source().is_some());

    let mut buffer = Vec::new();
    let error = execute(&workload, heuristic_runner, &mut buffer).unwrap_err();
    assert!(matches!(error, BenchError::CacheConfig(_)));
    assert!(buffer.is_empty(), "no report may be written on failure");
}

#[test]
fn warmup_and_measured_failures_are_reported_by_stage() {
    let error = measure(&config(1, 5, 1), rejecting_runner).unwrap_err();
    assert_eq!(
        error.to_string().split(" failed:").next(),
        Some("warmup phase")
    );
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Warmup,
            ..
        }
    ));

    let error = measure(&config(1, 5, 0), rejecting_runner).unwrap_err();
    assert_eq!(
        error.to_string().split(" failed:").next(),
        Some("measured phase")
    );
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Measure,
            ..
        }
    ));

    // A healthy cold warmup does not hide a measured-phase failure.
    let mut factory_calls = 0;
    let error = measure(&config(1, 5, 1), |cache| {
        factory_calls += 1;
        if factory_calls == 1 {
            heuristic_runner(cache)
        } else {
            rejecting_runner(cache)
        }
    })
    .unwrap_err();
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Measure,
            ..
        }
    ));
}

#[test]
fn a_failed_phase_writes_no_report() {
    for warmup in [0, 1] {
        let mut buffer = Vec::new();
        let error = execute(&config(1, 5, warmup), rejecting_runner, &mut buffer).unwrap_err();
        assert!(buffer.is_empty(), "no summary may be written on failure");
        assert!(matches!(error, BenchError::Phase { .. }));
    }
}

#[test]
fn decision_limit_exhaustion_invalidates_each_phase_without_a_report() {
    for (warmup, expected_stage) in [(0, Stage::Measure), (1, Stage::Warmup)] {
        let mut buffer = Vec::new();
        let error = execute(
            &config(1, 5, warmup),
            |cache| heuristic_runner(cache).with_decision_limit(0),
            &mut buffer,
        )
        .unwrap_err();
        assert!(buffer.is_empty(), "no summary may be written on failure");
        assert!(matches!(
            error,
            BenchError::Phase {
                stage,
                source: RunnerError::DecisionLimit { .. },
            } if stage == expected_stage
        ));
        assert!(
            error.source().is_some(),
            "underlying error must be retained"
        );
    }
}

struct FailingWriter {
    write_error: bool,
}

impl Write for FailingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.write_error {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "sink closed"))
        } else {
            Ok(buffer.len())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "sink closed"))
    }
}

#[test]
fn report_write_and_flush_failures_are_output_errors() {
    for write_error in [true, false] {
        let error = execute(
            &config(1, 5, 0),
            heuristic_runner,
            &mut FailingWriter { write_error },
        )
        .unwrap_err();
        assert!(
            matches!(error, BenchError::Output(_)),
            "write_error {write_error}: {error}"
        );
        assert!(error.to_string().contains("sink closed"), "{error}");
    }
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
