//! Command line, workload validation, and seed-plan coverage.

use super::*;

/// Measured then warmup seed sequences, in play order.
fn seed_ranges(config: &Config) -> (Vec<u64>, Vec<u64>) {
    let measured = (0..config.games).map(|i| config.measure_seed(i)).collect();
    let warmup = (0..config.warmup).map(|i| config.warmup_seed(i)).collect();
    (measured, warmup)
}

#[test]
fn help_is_standalone_and_ignores_later_arguments() {
    for arguments in [vec!["-h"], vec!["--help"], vec!["--help", "--games", "0"]] {
        let command = parse_args(owned_args(&arguments)).unwrap();
        assert_eq!(command, Command::Help, "{arguments:?}");
    }
}

#[test]
fn games_and_seed_are_required_and_defaults_are_explicit() {
    let parsed = parse_args(owned_args(&["--games", "2", "--seed", "3"])).unwrap();
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
        let error = parse_args(owned_args(&others)).unwrap_err();
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
        let error = parse_args(owned_args(&arguments)).unwrap_err();
        assert!(!error.is_empty(), "{arguments:?}");
    }

    let duplicate = parse_args(owned_args(&[
        "--games", "1", "--games", "2", "--seed", "0", "--warmup", "0",
    ]))
    .unwrap_err();
    assert_eq!(duplicate, "duplicate --games");

    let missing_value = parse_args(owned_args(&["--games"])).unwrap_err();
    assert_eq!(missing_value, "--games requires a value");

    let unknown = parse_args(owned_args(&["--extra"])).unwrap_err();
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
    let parsed = parse_args(owned_args(&[
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
    ]))
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
        parse_args(owned_args(&arguments)).unwrap_err()
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
    let error = parse_args(owned_args(&[
        "--games",
        "1",
        "--seed",
        "0",
        "--lifecycle",
        "warm",
    ]))
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
    let error = parse_args(owned_args(&[
        "--games", "0", "--seed", "0", "--warmup", "0",
    ]))
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
