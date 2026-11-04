use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;

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
    validate(games, seed, warmup).expect("fixture configuration must be valid")
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
fn sample(config: &Config) -> BatchReport {
    measure(config, heuristic_runner).expect("heuristic batch must complete")
}

/// Runner whose policies always return an illegal action.
fn rejecting_runner() -> Runner {
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
fn every_workload_option_is_required() {
    let complete = ["--games", "2", "--seed", "3", "--warmup", "0"];
    let parsed = parse_args(complete.iter().map(|value| (*value).to_owned())).unwrap();
    assert_eq!(
        parsed,
        Command::Run(Config {
            games: 2,
            seed: 3,
            warmup: 0,
        })
    );

    for missing in 0..3 {
        let arguments: Vec<String> = complete
            .iter()
            .enumerate()
            .filter(|(index, _)| index / 2 != missing)
            .map(|(_, value)| (*value).to_owned())
            .collect();
        let error = parse_args(arguments).unwrap_err();
        let expected = ["--games", "--seed", "--warmup"][missing];
        assert!(error.contains(expected), "{error}");
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

    assert!(validate(u64::MAX, 123, 2).is_err());
    assert!(validate(2, 0, u64::MAX).is_err());
    assert!(validate(u64::MAX, u64::MAX, u64::MAX).is_err());

    // Large accepted workloads are validated arithmetically, never played.
    let large = config(1_000_000_000_000, u64::MAX - 3, 4);
    assert_eq!(large.measure_seed(3), u64::MAX);
    assert_eq!(large.warmup_seed(3), u64::MAX - 4);
}

#[test]
fn phases_build_their_own_runner_and_reuse_policies_within_a_batch() {
    let runners = Arc::new(AtomicUsize::new(0));
    let policies = Arc::new(AtomicUsize::new(0));
    let mut factory_calls = 0;

    let batch = measure(&config(2, 21, 2), || {
        factory_calls += 1;
        runners.fetch_add(1, Ordering::SeqCst);
        let policies = Arc::clone(&policies);
        Runner::new(std::array::from_fn(|_| {
            policies.fetch_add(1, Ordering::SeqCst);
            Box::new(HeuristicPolicy::new()) as Box<dyn Policy>
        }))
    })
    .expect("both phases must complete");

    assert_eq!(batch.games, 2);
    // One runner per phase: warmup and measured, never rebuilt per game.
    assert_eq!(factory_calls, 2);
    assert_eq!(runners.load(Ordering::SeqCst), 2);
    // Four seat policies per phase, reused across that phase's two games.
    assert_eq!(policies.load(Ordering::SeqCst), 2 * 4);
}

#[test]
fn warmup_is_skipped_without_an_empty_batch_when_disabled() {
    let mut factory_calls = 0;
    let batch = measure(&config(1, 33, 0), || {
        factory_calls += 1;
        heuristic_runner()
    })
    .expect("a zero-warmup workload must run only the measured phase");
    assert_eq!(factory_calls, 1);
    assert_eq!(batch.games, 1);
}

#[test]
fn warmup_results_do_not_enter_the_measured_report() {
    let cold = sample(&config(2, 300, 0));
    let warmed = sample(&config(2, 300, 2));
    assert_eq!(non_time(&cold), non_time(&warmed));
}

#[test]
fn repeated_runs_produce_identical_non_time_results() {
    let workload = config(2, 17, 0);
    assert_eq!(non_time(&sample(&workload)), non_time(&sample(&workload)));
}

#[test]
fn a_small_measured_batch_completes_every_game_with_zero_sum_scores() {
    let batch = sample(&config(3, 99, 1));
    assert_eq!(batch.games, 3);
    assert!(batch.decisions > 0);
    assert_eq!(batch.three_winners + batch.wall_exhausted, 3);
    assert_eq!(batch.score_sum(), 0);
}

#[test]
fn report_states_workload_units_and_game_results() {
    let workload = config(4, 1234, 2);
    let text = render_report(&workload, &fixture_batch());
    let expected = format!(
        "protocol: heuristic-benchmark-v1\n\
         policy: heuristic\n\
         seats: 4\n\
         seat_order: [0, 1, 2, 3]\n\
         seed: 1234\n\
         games: 4\n\
         warmup_games: 2\n\
         warmup_first_seed: 1232\n\
         decision_limit_per_game: {DEFAULT_DECISION_LIMIT}\n\
         completed_games: 4\n\
         decisions: 400\n\
         total_time_seconds: 0.500000\n\
         policy_time_seconds: 0.002000\n\
         rules_time_seconds: 0.001000\n\
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
fn report_renders_wrapping_seed_ranges() {
    let backwards = render_report(&config(2, 0, 2), &fixture_batch());
    assert!(backwards.contains("seed: 0\n"));
    assert!(backwards.contains("warmup_games: 2\n"));
    assert!(backwards.contains("warmup_first_seed: 18446744073709551614\n"));

    let forwards = render_report(&config(2, u64::MAX, 0), &fixture_batch());
    assert!(forwards.contains("seed: 18446744073709551615\n"));
    assert!(forwards.contains("warmup_first_seed: none\n"));
}

#[test]
fn disabled_warmup_is_identifiable_in_the_report() {
    let text = render_report(&config(1, 5, 0), &fixture_batch());
    assert!(text.contains("warmup_games: 0\n"), "{text}");
    assert!(text.contains("warmup_first_seed: none\n"), "{text}");
}

#[test]
fn zero_time_and_zero_decisions_report_unavailable_instead_of_nan() {
    let mut batch = fixture_batch();
    batch.decisions = 0;
    batch.policy_time = std::time::Duration::ZERO;
    batch.total_time = std::time::Duration::ZERO;
    let text = render_report(&config(1, 0, 0), &batch);

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
    let text = render_report(&config(5, 0, 0), &batch);
    assert!(text.contains("games: 5\n"));
    assert!(text.contains("completed_games: 4\n"));
    assert!(text.contains("games_per_sec: 8.000\n"));
    assert!(text.contains("decisions_per_sec: 0.000\n"));
    assert!(text.contains("mean_policy_us_per_decision: unavailable\n"));
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

    // A healthy warmup does not hide a measured-phase failure.
    let mut factory_calls = 0;
    let error = measure(&config(1, 5, 1), || {
        factory_calls += 1;
        if factory_calls == 1 {
            heuristic_runner()
        } else {
            rejecting_runner()
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
            || heuristic_runner().with_decision_limit(0),
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

    // Also preserve the measured stage after an entirely healthy warmup.
    let mut factory_calls = 0;
    let mut buffer = Vec::new();
    let error = execute(
        &config(1, 5, 1),
        || {
            factory_calls += 1;
            if factory_calls == 1 {
                heuristic_runner()
            } else {
                heuristic_runner().with_decision_limit(0)
            }
        },
        &mut buffer,
    )
    .unwrap_err();
    assert_eq!(factory_calls, 2);
    assert!(buffer.is_empty());
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Measure,
            source: RunnerError::DecisionLimit { .. },
        }
    ));
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
fn a_successful_run_writes_the_measured_report() {
    let mut buffer = Vec::new();
    execute(&config(1, 5, 0), heuristic_runner, &mut buffer).unwrap();
    let text = String::from_utf8(buffer).expect("report is UTF-8");
    assert!(text.contains("protocol: heuristic-benchmark-v1"), "{text}");
    assert!(text.contains("games: 1\n"), "{text}");
    assert!(text.contains("games_per_sec: "), "{text}");
    assert!(text.contains("score_sum: 0\n"), "{text}");
}
