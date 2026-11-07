use super::*;
use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;

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

fn temp_dir(name: &str) -> PathBuf {
    let path = env::temp_dir().join(format!("policy_ab_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

#[test]
fn arguments_require_all_inputs_and_a_distinct_seed_list() {
    let parsed = parse_args(
        [
            "--seeds",
            "1,2",
            "--policy-seed",
            "3",
            "--decision-limit",
            "50",
            "--out",
            "run",
        ]
        .iter()
        .map(|value| (*value).to_owned()),
    )
    .unwrap();
    match parsed {
        Command::Run(options) => {
            assert_eq!(options.plan.seeds, vec![1, 2]);
            assert_eq!(options.plan.policy_seed, 3);
            assert_eq!(options.plan.decision_limit, 50);
            assert_eq!(options.plan.style_a, Style::First);
            assert_eq!(options.plan.style_b, Style::Random);
            assert_eq!(options.out, PathBuf::from("run"));
        }
        Command::Help => panic!("expected a run command"),
    }

    let args = |values: &[&str]| {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
    };
    assert!(matches!(
        parse_args(args(&["--help", "--seeds", "0"])).unwrap(),
        Command::Help
    ));
    assert!(parse_args(args(&[])).is_err());
    assert!(parse_args(args(&["--seeds", "1"])).is_err());
    assert!(parse_args(args(&["--seeds", "1,,2"])).is_err());
    assert!(parse_args(args(&["--seeds", ""])).is_err());
    assert!(parse_args(args(&["--seeds", "1,1"])).is_err());
    assert!(parse_args(args(&["--seeds", "abc"])).is_err());
    assert!(parse_args(args(&["--seeds", "1", "--decision-limit", "0"])).is_err());
    assert!(parse_args(args(&["--seeds", "1", "--policy-seed"])).is_err());
    assert!(parse_args(args(&["--seeds", "1", "--seeds", "2"])).is_err());
    assert!(parse_args(args(&["--seeds", "1", "extra"])).is_err());
}

#[test]
fn help_short_circuits_later_arguments() {
    let parsed = parse_args(
        ["-h", "--seeds", "not-a-seed"]
            .iter()
            .map(|value| (*value).to_owned()),
    )
    .unwrap();
    assert!(matches!(parsed, Command::Help));
}

#[test]
fn policy_selection_defaults_to_first_versus_random() {
    let args = |values: &[&str]| {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
    };
    let base = [
        "--seeds",
        "1",
        "--policy-seed",
        "3",
        "--decision-limit",
        "50",
        "--out",
        "run",
    ];
    let defaults = parse_args(args(&base)).unwrap();
    let Command::Run(defaults) = defaults else {
        panic!("expected a run command");
    };
    assert_eq!(defaults.plan.style_a, Style::First);
    assert_eq!(defaults.plan.style_b, Style::Random);

    // Selection is independent of argument order.
    let mut explicit = base.to_vec();
    explicit.extend_from_slice(&["--a", "heuristic", "--b", "first"]);
    let Command::Run(options) = parse_args(args(&explicit)).unwrap() else {
        panic!("expected a run command");
    };
    assert_eq!(options.plan.style_a, Style::Heuristic);
    assert_eq!(options.plan.style_b, Style::First);
    assert_eq!(options.plan.seeds, vec![1]);
}

#[test]
fn every_policy_style_parses_and_round_trips_through_its_name() {
    for style in [
        Style::First,
        Style::Random,
        Style::Heuristic,
        Style::MissingOnly,
    ] {
        assert_eq!(parse_style(style.name()).unwrap(), style);
    }
    assert_eq!(Style::First.name(), "first");
    assert_eq!(Style::Heuristic.name(), "heuristic");
    assert_eq!(Style::MissingOnly.name(), "missing-only");

    let args = |values: &[&str]| {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
    };
    for style in ["first", "random", "heuristic", "missing-only"] {
        let parsed = parse_args(args(&[
            "--seeds",
            "1",
            "--policy-seed",
            "0",
            "--decision-limit",
            "10",
            "--out",
            "run",
            "--a",
            style,
            "--b",
            style,
        ]));
        assert!(parsed.is_ok(), "--a/--b {style} should parse: {parsed:?}");
    }

    assert!(parse_style("greedy").is_err());
    assert!(parse_style("").is_err());
    assert_eq!(parse_style("Heuristic").unwrap(), Style::Heuristic);
    assert_eq!(parse_style("MISSING-ONLY").unwrap(), Style::MissingOnly);
    assert!(parse_style("missing_only").is_err());
    // Supply every required run input so these cannot pass merely because
    // unrelated required options are missing.
    let invalid = |extra: &[&str]| {
        let mut arguments = args(&[
            "--seeds",
            "1",
            "--policy-seed",
            "0",
            "--decision-limit",
            "10",
            "--out",
            "run",
        ]);
        arguments.extend(args(extra));
        parse_args(arguments).unwrap_err()
    };
    assert_eq!(invalid(&["--a"]), "--a requires a value");
    assert_eq!(invalid(&["--b"]), "--b requires a value");
    for option in ["--a", "--b"] {
        assert!(invalid(&[option, "greedy"]).contains("invalid policy style"));
        assert_eq!(
            invalid(&[option, "first", option, "random"]),
            format!("duplicate {option}")
        );
    }
}

#[test]
fn run_record_names_the_selected_policies() {
    let plan = plan(vec![1], Style::Heuristic, Style::First);
    let metadata = Metadata {
        revision: "abc".to_owned(),
        dirty: Some(false),
        toolchain: "rustc test".to_owned(),
        warnings: Vec::new(),
    };
    let text = run_record_text(&plan, &["policy_ab".to_owned()], &metadata, None);
    assert!(text.contains("policy A: heuristic\n"), "{text}");
    assert!(text.contains("policy B: first\n"), "{text}");
    assert!(!text.contains("policy A: first"), "{text}");

    let summary = summary_text(
        &plan,
        &Summary {
            seeds: 1,
            games: A_PAIRS.len(),
            a_mean: 0.0,
            b_mean: 0.0,
            delta: 0.0,
            se: None,
            per_seed: vec![SeedContrast {
                seed: 1,
                contrast: 0.0,
            }],
        },
    );
    assert!(summary.contains("policy A: heuristic"), "{summary}");
    assert!(summary.contains("policy B: first"), "{summary}");
}

#[test]
fn small_corpus_produces_six_placements_per_seed_and_recomputes_the_summary() {
    let plan = default_plan(vec![1, 2]);
    let mut output = Vec::new();
    run_plan(&plan, &mut output).expect("seed corpus runs");

    let records =
        parse_records(&plan, &String::from_utf8(output).unwrap()).expect("persisted records parse");
    assert_eq!(records.len(), 2 * A_PAIRS.len());
    for seed in &plan.seeds {
        assert_eq!(
            records.iter().filter(|record| record.seed == *seed).count(),
            A_PAIRS.len()
        );
    }
    for record in &records {
        assert_eq!(
            record
                .scores
                .iter()
                .map(|score| i64::from(*score))
                .sum::<i64>(),
            0
        );
        assert!(record.decisions > 0);
    }

    // The summary is computed from the persisted records alone.
    let summary = summarize(&plan, &records).expect("summary recomputes");
    assert_eq!(summary.seeds, 2);
    assert_eq!(summary.games, 2 * A_PAIRS.len());
}

#[test]
fn seed_parsing_preserves_input_order_and_full_u64_values() {
    assert_eq!(
        parse_seeds(" 18446744073709551615 , 0 ,7, 42 ").unwrap(),
        vec![u64::MAX, 0, 7, 42]
    );

    // A duplicate stays detectable when it appears late in a long list.
    let late = format!("1,2,3,4,5,6,7,8,9,{},{} ", u64::MAX, u64::MAX);
    let error = parse_seeds(&late).unwrap_err();
    assert!(
        error.contains(&format!("duplicate game seed {}", u64::MAX)),
        "{error}"
    );
}

#[test]
fn extreme_u64_seeds_round_trip_through_records() {
    let plan = default_plan(vec![u64::MAX, 0]);
    let record = fixture_record(u64::MAX, 0, 5);
    let text = format!("{}\n", format_record(&record));
    assert_eq!(parse_records(&plan, &text).unwrap(), vec![record]);
}

#[test]
fn unsorted_records_map_to_the_requested_seed_order() {
    // Descending seeds with the u64 extremes: neither sorted nor grouped.
    let seeds = vec![u64::MAX, 9_000_000_000_000_000_000, 3, 0];
    let plan = default_plan(seeds.clone());
    let mut records = Vec::new();
    for (seed_index, seed) in seeds.iter().enumerate() {
        for pair_index in 0..A_PAIRS.len() {
            records.push(fixture_record(*seed, pair_index, seed_index as i32 + 1));
        }
    }
    // Group by placement instead of seed so blocks arrive interleaved.
    records.sort_by_key(|record| pair_index(record.pair).expect("fixture placement"));

    let summary = summarize(&plan, &records).unwrap();
    assert_eq!(summary.games, seeds.len() * A_PAIRS.len());
    for (index, entry) in summary.per_seed.iter().enumerate() {
        assert_eq!(entry.seed, seeds[index]);
        approx(entry.contrast, 2.0 * (index as f64 + 1.0));
    }
    approx(summary.delta, 2.0 * (1.0 + 2.0 + 3.0 + 4.0) / 4.0);
}

#[test]
fn large_fixture_aggregates_in_the_requested_seed_order() {
    let seeds: Vec<u64> = (0..256u64)
        .map(|index| u64::MAX - index * 1_000_003)
        .collect();
    let plan = default_plan(seeds.clone());
    let mut records = Vec::new();
    for (seed_index, seed) in seeds.iter().enumerate() {
        let x = (seed_index as i32 % 5) + 1;
        for pair_index in 0..A_PAIRS.len() {
            records.push(fixture_record(*seed, pair_index, x));
        }
    }
    records.sort_by_key(|record| (record.scores[0], record.seed));

    let summary = summarize(&plan, &records).unwrap();
    assert_eq!(summary.seeds, seeds.len());
    assert_eq!(summary.games, seeds.len() * A_PAIRS.len());
    for (index, entry) in summary.per_seed.iter().enumerate() {
        assert_eq!(entry.seed, seeds[index]);
        approx(entry.contrast, 2.0 * (((index as i32 % 5) + 1) as f64));
    }
}

#[test]
fn identical_inputs_reproduce_scores_and_decision_traces() {
    let first = play_with(21, A_PAIRS[4], Style::First, Style::Random, 99, LIMIT).unwrap();
    let second = play_with(21, A_PAIRS[4], Style::First, Style::Random, 99, LIMIT).unwrap();
    assert_eq!(first.scores, second.scores);
    assert_eq!(first.decisions, second.decisions);
    assert_eq!(first.end_reason, second.end_reason);
    assert_eq!(first.trace, second.trace);
    assert!(!first.trace.is_empty());
}

#[test]
fn placement_order_does_not_change_results() {
    let mut forward = Vec::new();
    for pair in A_PAIRS {
        forward.push((
            pair,
            play_with(11, pair, Style::First, Style::Random, 3, LIMIT).unwrap(),
        ));
    }

    let mut backward = Vec::new();
    for pair in A_PAIRS.into_iter().rev() {
        backward.push((
            pair,
            play_with(11, pair, Style::First, Style::Random, 3, LIMIT).unwrap(),
        ));
    }

    for (pair, report) in forward {
        let (_, reordered) = backward
            .iter()
            .find(|(candidate, _)| *candidate == pair)
            .expect("every placement is played");
        assert_eq!(report.scores, reordered.scores);
        assert_eq!(report.decisions, reordered.decisions);
        assert_eq!(report.end_reason, reordered.end_reason);
        assert_eq!(report.trace, reordered.trace);
    }
}

#[test]
fn first_legal_against_itself_has_zero_seed_contrasts() {
    let plan = plan(vec![1, 2], Style::First, Style::First);
    let mut output = Vec::new();
    run_plan(&plan, &mut output).expect("self-comparison runs");
    let records = parse_records(&plan, &String::from_utf8(output).unwrap())
        .expect("self-comparison records parse");
    let summary = summarize(&plan, &records).unwrap();

    for entry in &summary.per_seed {
        approx(entry.contrast, 0.0);
    }
    approx(summary.delta, 0.0);
    approx(summary.a_mean, 0.0);
    approx(summary.b_mean, 0.0);

    // Balanced labeling: every seat is an A seat three times.
    for seat in Seat::ALL {
        let count = A_PAIRS.iter().filter(|pair| pair.contains(&seat)).count();
        assert_eq!(count, 3);
    }
}

#[test]
fn hand_calculated_block_means_and_standard_error() {
    let mut plan = default_plan(vec![10, 20]);
    plan.decision_limit = 10;
    let records = fixture_records(&plan, &[&[1, 2, 3, 4, 5, 6], &[2, 2, 2, 2, 2, 2]]);
    let summary = summarize(&plan, &records).unwrap();

    // Seed 10 contrasts are 2x: mean 7; seed 20 contrasts are all 4.
    assert_eq!(summary.per_seed.len(), 2);
    approx(summary.per_seed[0].contrast, 7.0);
    approx(summary.per_seed[1].contrast, 4.0);
    approx(summary.a_mean, 2.75);
    approx(summary.b_mean, -2.75);
    approx(summary.delta, 5.5);
    approx(summary.delta, summary.a_mean - summary.b_mean);
    approx(summary.se.unwrap(), 1.5);
    assert_eq!(summary.games, 12);

    // One seed block: SE is unavailable, not zero.
    let one_seed = default_plan(vec![10]);
    let records = fixture_records(&one_seed, &[&[1, 2, 3, 4, 5, 6]]);
    let summary = summarize(&one_seed, &records).unwrap();
    approx(summary.delta, 7.0);
    assert_eq!(summary.se, None);
}

#[test]
fn record_validation_rejects_invalid_or_unplanned_records() {
    let plan = default_plan(vec![5]);
    let valid = fixture_record(5, 0, 3);
    assert!(
        summarize(&plan, &[valid; 6]).is_err(),
        "duplicates are rejected"
    );

    let mut records = fixture_records(&plan, &[&[1, 2, 3, 4, 5, 6]]);
    assert!(summarize(&plan, &records).is_ok());
    assert_eq!(summarize(&plan, &records).unwrap().se, None);

    let mut unexpected_seed = records.clone();
    unexpected_seed[0].seed = 6;
    assert!(summarize(&plan, &unexpected_seed).is_err());

    let mut unexpected_pair = records.clone();
    unexpected_pair[0].pair = [A_PAIRS[0][1], A_PAIRS[0][0]];
    assert!(summarize(&plan, &unexpected_pair).is_err());

    let mut nonzero_sum = records.clone();
    nonzero_sum[0].scores = [1, 1, 1, 1];
    assert!(summarize(&plan, &nonzero_sum).is_err());

    let mut zero_decisions = records.clone();
    zero_decisions[0].decisions = 0;
    assert!(summarize(&plan, &zero_decisions).is_err());

    let mut over_limit = records.clone();
    over_limit[0].decisions = plan.decision_limit + 1;
    assert!(summarize(&plan, &over_limit).is_err());

    records.pop();
    assert!(
        summarize(&plan, &records).is_err(),
        "missing records are rejected"
    );
}

#[test]
fn duplicate_coverage_is_rejected_even_with_the_expected_count() {
    let plan = default_plan(vec![5]);
    let mut records = fixture_records(&plan, &[&[1, 2, 3, 4, 5, 6]]);
    records[5] = records[0];
    let error = summarize(&plan, &records).unwrap_err();
    assert!(error.contains("duplicate record"), "{error}");
}

#[test]
fn malformed_record_text_is_rejected() {
    let plan = default_plan(vec![1]);
    let header =
        "# seed\ta_seat0\ta_seat1\tscore0\tscore1\tscore2\tscore3\tend_reason\tdecisions\n";
    let valid = "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\t12\n";
    let parsed = parse_records(&plan, &format!("{header}{valid}")).unwrap();
    assert_eq!(parsed.len(), 1);

    let strict = Plan {
        decision_limit: 10,
        ..default_plan(vec![1])
    };
    let cases = [
        "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\n",
        "abc\t0\t1\t3\t3\t-3\t-3\tthree_winners\t12\n",
        "1\t0\t1\t3000000000\t3\t-3\t-3\tthree_winners\t12\n",
        "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\t18446744073709551616\n",
        "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\t-1\n",
        "1\t0\t1\t3\t3\t-3\t-3\tdraw\t12\n",
        "1\t0\t9\t3\t3\t-3\t-3\tthree_winners\t12\n",
        "2\t0\t1\t3\t3\t-3\t-3\tthree_winners\t12\n",
        "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\t0\n",
        "1\t0\t1\t3\t3\t-3\t-3\tthree_winners\t11\n",
    ];
    for line in cases {
        assert!(
            parse_records(&strict, line).is_err(),
            "line should be rejected: {line}"
        );
    }
}

#[test]
fn missing_only_orientations_run_and_record_the_selected_policies() {
    // The two experiment orientations requested for the ablation: the
    // ablation against the baseline it delegates to, and against the
    // heuristic whose missing-suit rule it borrows.
    for (style_a, style_b) in [
        (Style::First, Style::MissingOnly),
        (Style::MissingOnly, Style::Heuristic),
    ] {
        let plan = plan(vec![7], style_a, style_b);
        let mut output = Vec::new();
        run_plan(&plan, &mut output).expect("missing-only corpus runs");

        let records = parse_records(&plan, &String::from_utf8(output).unwrap())
            .expect("missing-only records parse");
        assert_eq!(records.len(), A_PAIRS.len());
        for record in &records {
            assert_eq!(
                record
                    .scores
                    .iter()
                    .map(|score| i64::from(*score))
                    .sum::<i64>(),
                0
            );
            assert!(record.decisions > 0);
            assert!(record.decisions <= plan.decision_limit);
        }

        let metadata = Metadata {
            revision: "abc".to_owned(),
            dirty: Some(false),
            toolchain: "rustc test".to_owned(),
            warnings: Vec::new(),
        };
        let text = run_record_text(&plan, &["policy_ab".to_owned()], &metadata, None);
        assert!(
            text.contains(&format!("policy A: {}\n", style_a.name())),
            "{text}"
        );
        assert!(
            text.contains(&format!("policy B: {}\n", style_b.name())),
            "{text}"
        );
    }
}

#[test]
fn heuristic_games_are_legal_and_cover_all_six_placements() {
    let plan = plan(vec![7], Style::Heuristic, Style::First);
    assert_eq!(A_PAIRS.len(), 6);
    let mut output = Vec::new();
    run_plan(&plan, &mut output).expect("heuristic corpus runs");

    let records =
        parse_records(&plan, &String::from_utf8(output).unwrap()).expect("heuristic records parse");
    assert_eq!(records.len(), A_PAIRS.len());
    for record in &records {
        assert_eq!(
            record
                .scores
                .iter()
                .map(|score| i64::from(*score))
                .sum::<i64>(),
            0
        );
        assert!(record.decisions > 0);
        assert!(record.decisions <= plan.decision_limit);
    }
    let summary = summarize(&plan, &records).expect("heuristic summary recomputes");
    assert_eq!(summary.games, A_PAIRS.len());
    assert_eq!(summary.seeds, 1);
}

#[test]
fn heuristic_decisions_replay_from_the_same_seed() {
    let first = play_with(31, A_PAIRS[3], Style::Heuristic, Style::Random, 5, LIMIT)
        .expect("heuristic game runs");
    let second = play_with(31, A_PAIRS[3], Style::Heuristic, Style::Random, 5, LIMIT)
        .expect("heuristic game replays");
    assert_eq!(first.scores, second.scores);
    assert_eq!(first.decisions, second.decisions);
    assert_eq!(first.end_reason, second.end_reason);
    assert_eq!(first.trace, second.trace);
    assert!(!first.trace.is_empty());
}

#[test]
fn heuristic_against_itself_has_zero_seed_contrasts() {
    // All four seats use the same deterministic policy, so the balanced A/B
    // labels must cancel exactly over the six placements.
    let plan = plan(vec![4], Style::Heuristic, Style::Heuristic);
    let mut output = Vec::new();
    run_plan(&plan, &mut output).expect("heuristic self-comparison runs");
    let records = parse_records(&plan, &String::from_utf8(output).unwrap())
        .expect("self-comparison records parse");
    let summary = summarize(&plan, &records).unwrap();
    for entry in &summary.per_seed {
        approx(entry.contrast, 0.0);
    }
    approx(summary.delta, 0.0);
    approx(summary.a_mean, 0.0);
    approx(summary.b_mean, 0.0);
}

#[test]
fn rejected_actions_stop_the_run_reported_with_seed_and_placement() {
    let plan = plan(vec![1], Style::Rejecting, Style::Random);
    let mut output = Vec::new();
    let error = run_plan(&plan, &mut output).unwrap_err();
    assert!(error.contains("seed 1 placement [0, 1]"), "{error}");
    assert!(error.contains("illegal action"), "{error}");
}

#[test]
fn decision_limit_exhaustion_is_an_error_not_a_score() {
    let error = play_with(1, A_PAIRS[0], Style::First, Style::First, 0, 1).unwrap_err();
    assert!(error.contains("1-decision safety limit"), "{error}");
}

#[test]
fn output_errors_stop_the_run() {
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("disk full"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("disk full"))
        }
    }

    let plan = default_plan(vec![1]);
    let error = run_plan(&plan, &mut FailingWriter).unwrap_err();
    assert!(error.contains("failed to write record"), "{error}");
}

#[test]
fn execute_writes_artifacts_and_recomputes_the_summary_from_records() {
    let out = temp_dir("execute");
    let plan = default_plan(vec![3]);
    let invocation = vec![
        "target/release/examples/policy_ab".to_owned(),
        "--out".to_owned(),
        "cargo test execute".to_owned(),
    ];
    let summary = execute(&plan, &out, &invocation).expect("run succeeds");

    let run_record = fs::read_to_string(out.join("run.txt")).unwrap();
    assert!(run_record.contains("simulator revision:"));
    assert!(run_record.contains("policy stream derivation: random_seat_seed"));
    assert!(run_record.contains("heuristic cache: not applicable (no heuristic seat)"));
    assert!(
        !run_record.contains("heuristic cache scope:"),
        "{run_record}"
    );
    assert!(!out.join("cache.txt").exists());
    assert!(run_record.contains(
        "invocation argv (debug-quoted, one element per argument): \
         [\"target/release/examples/policy_ab\", \"--out\", \"cargo test execute\"]"
    ));

    let persisted =
        parse_records(&plan, &fs::read_to_string(out.join("games.tsv")).unwrap()).unwrap();
    assert_eq!(persisted.len(), A_PAIRS.len());
    assert_eq!(summarize(&plan, &persisted).unwrap(), summary);
    assert!(out.join("summary.txt").is_file());

    let error = execute(&plan, &out, &invocation).unwrap_err();
    assert!(error.contains("refusing to overwrite"), "{error}");

    let _ = fs::remove_dir_all(&out);
}

#[test]
fn invocation_arguments_keep_their_boundaries_in_the_run_record() {
    let plan = default_plan(vec![1]);
    let metadata = Metadata {
        revision: "abc".to_owned(),
        dirty: Some(false),
        toolchain: "rustc test".to_owned(),
        warnings: Vec::new(),
    };
    let invocation = vec![
        "target/release/examples/policy_ab".to_owned(),
        "--out".to_owned(),
        "a run dir".to_owned(),
        "he said \"hi\"".to_owned(),
    ];
    let text = run_record_text(&plan, &invocation, &metadata, None);
    assert!(
        text.contains(
            "invocation argv (debug-quoted, one element per argument): \
             [\"target/release/examples/policy_ab\", \"--out\", \"a run dir\", \
             \"he said \\\"hi\\\"\"]"
        ),
        "{text}"
    );
}

#[test]
fn summary_marks_se_unavailable_for_one_seed_block() {
    let mut plan = default_plan(vec![10]);
    plan.decision_limit = 10;
    let records = fixture_records(&plan, &[&[1, 2, 3, 4, 5, 6]]);
    let summary = summarize(&plan, &records).unwrap();
    let text = summary_text(&plan, &summary);
    assert!(
        text.contains("SE: unavailable (fewer than 2 seed blocks)"),
        "{text}"
    );
    assert!(!text.contains("SE: 0"), "{text}");
}

#[test]
fn unknown_metadata_is_rendered_with_its_warning() {
    let plan = default_plan(vec![1]);
    let metadata = Metadata {
        revision: "unknown".to_owned(),
        dirty: None,
        toolchain: "unknown".to_owned(),
        warnings: vec!["simulator revision unavailable; recorded as unknown".to_owned()],
    };
    let text = run_record_text(&plan, &["policy_ab".to_owned()], &metadata, None);
    assert!(text.contains("simulator revision: unknown"), "{text}");
    assert!(text.contains("working tree dirty: unknown"), "{text}");
    assert!(text.contains("toolchain: unknown"), "{text}");
    assert!(
        text.contains("warning: simulator revision unavailable; recorded as unknown"),
        "{text}"
    );
}

#[test]
fn metadata_commands_are_anchored_to_the_source_tree() {
    let source_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let metadata = collect_metadata(source_dir);
    if command_stdout("rustc", &["--version"], source_dir).is_some() {
        assert_ne!(metadata.toolchain, "unknown");
    }
    if command_stdout("git", &["rev-parse", "HEAD"], source_dir).is_some() {
        assert_ne!(metadata.revision, "unknown");
        assert!(metadata.dirty.is_some());
    }

    // A missing source tree degrades to unknown with warnings instead of
    // silently reporting metadata for the caller's working directory.
    let missing = temp_dir("missing-source-tree");
    let metadata = collect_metadata(&missing);
    assert_eq!(metadata.revision, "unknown");
    assert_eq!(metadata.dirty, None);
    assert_eq!(metadata.toolchain, "unknown");
    assert_eq!(metadata.warnings.len(), 3, "{:?}", metadata.warnings);
}

#[test]
fn execute_failures_keep_artifacts_without_a_summary() {
    let mut limited = plan(vec![1], Style::First, Style::First);
    limited.decision_limit = 1;
    let cases = [
        (
            "rejecting",
            plan(vec![1], Style::Rejecting, Style::Random),
            "illegal action",
        ),
        ("limited", limited, "1-decision safety limit"),
    ];

    for (name, plan, expected) in cases {
        let out = temp_dir(name);
        let error = execute(&plan, &out, &["policy_ab".to_owned()]).unwrap_err();
        assert!(error.contains(expected), "{name}: {error}");
        assert!(out.join("run.txt").is_file(), "{name} retains run.txt");
        assert!(out.join("games.tsv").is_file(), "{name} retains games.tsv");
        assert!(
            !out.join("summary.txt").exists(),
            "{name} must not write summary.txt"
        );
        let _ = fs::remove_dir_all(&out);
    }
}

#[test]
fn unavailable_metadata_is_recorded_as_unknown_with_a_warning() {
    let mut warnings = Vec::new();
    assert_eq!(
        metadata_value(None, "simulator revision", &mut warnings),
        "unknown"
    );
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("simulator revision"));
    assert_eq!(
        metadata_value(Some("abc\n".to_owned()), "toolchain", &mut warnings),
        "abc"
    );
    assert_eq!(warnings.len(), 1);
}

#[test]
fn cache_budget_option_parses_and_requires_a_heuristic_seat() {
    let args = |values: &[&str]| {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
    };
    let base = [
        "--seeds",
        "1",
        "--policy-seed",
        "3",
        "--decision-limit",
        "50",
        "--out",
        "run",
        "--a",
        "heuristic",
        "--b",
        "first",
    ];

    let Command::Run(defaults) = parse_args(args(&base)).unwrap() else {
        panic!("expected a run command");
    };
    assert_eq!(defaults.plan.cache_budget, CacheBudget::CandidateDefault);
    assert!(defaults.plan.uses_heuristic());

    let mut explicit = base.to_vec();
    explicit.extend_from_slice(&["--cache-budget-bytes", "0"]);
    let Command::Run(options) = parse_args(args(&explicit)).unwrap() else {
        panic!("expected a run command");
    };
    assert_eq!(options.plan.cache_budget, CacheBudget::Explicit(0));

    let invalid = |extra: &[&str]| {
        let mut values = base.to_vec();
        values.extend_from_slice(extra);
        parse_args(args(&values)).unwrap_err()
    };
    assert!(invalid(&["--cache-budget-bytes", "abc"]).contains("invalid"));
    assert_eq!(
        invalid(&["--cache-budget-bytes"]),
        "--cache-budget-bytes requires a value"
    );
    assert_eq!(
        invalid(&["--cache-budget-bytes", "1", "--cache-budget-bytes", "2",]),
        "duplicate --cache-budget-bytes"
    );

    // No heuristic seat means the option is a configuration mistake, not an
    // ignored setting.
    let no_heuristic = [
        "--seeds",
        "1",
        "--policy-seed",
        "0",
        "--decision-limit",
        "10",
        "--out",
        "run",
        "--cache-budget-bytes",
        "100",
    ];
    let error = parse_args(args(&no_heuristic)).unwrap_err();
    assert!(
        error.contains("requires at least one heuristic seat"),
        "{error}"
    );

    // The option is accepted when either arm is heuristic, including 0.
    for styles in [
        ["heuristic", "first"],
        ["first", "heuristic"],
        ["heuristic", "heuristic"],
    ] {
        let values = [
            "--seeds",
            "1",
            "--policy-seed",
            "0",
            "--decision-limit",
            "10",
            "--out",
            "run",
            "--a",
            styles[0],
            "--b",
            styles[1],
            "--cache-budget-bytes",
            "0",
        ];
        assert!(parse_args(args(&values)).is_ok(), "{styles:?}");
    }
}

#[test]
fn byte_values_are_strict_decimal_usize_counts() {
    assert_eq!(parse_usize("--cache-budget-bytes", "0").unwrap(), 0);
    assert_eq!(
        parse_usize("--cache-budget-bytes", "27787264").unwrap(),
        27_787_264
    );
    for value in [
        "",
        "+1",
        "-1",
        "1_000",
        "0x10",
        "1.5",
        "18446744073709551616",
    ] {
        assert!(
            parse_usize("--cache-budget-bytes", value).is_err(),
            "{value}"
        );
    }
}

#[test]
fn a_shared_heuristic_context_preserves_actions_across_games_and_placements() {
    let cache = SharedCache::new(DEFAULT_BUDGET_BYTES).expect("candidate cache context");
    let shared = play_with_cache(
        31,
        A_PAIRS[3],
        Style::Heuristic,
        Style::First,
        5,
        LIMIT,
        Some(&cache),
    )
    .expect("shared-context game runs");
    let fresh = play_with(31, A_PAIRS[3], Style::Heuristic, Style::First, 5, LIMIT)
        .expect("fresh-context game runs");

    assert_eq!(shared.scores, fresh.scores);
    assert_eq!(shared.decisions, fresh.decisions);
    assert_eq!(shared.end_reason, fresh.end_reason);
    assert_eq!(shared.trace, fresh.trace);
    assert!(!shared.trace.is_empty());
    assert_ne!(cache.snapshot().state, "disabled");
}

#[test]
fn run_plan_with_cache_shares_one_context_over_the_whole_run() {
    let plan = plan_with_cache(
        vec![7, 8],
        Style::Heuristic,
        Style::Heuristic,
        DEFAULT_BUDGET_BYTES,
    );
    let cache = SharedCache::new(plan.cache_budget.bytes()).expect("cache context");
    let mut output = Vec::new();
    run_plan_with_cache(&plan, &mut output, Some(&cache)).expect("shared corpus runs");

    let records = parse_records(&plan, &String::from_utf8(output).unwrap())
        .expect("shared-context records parse");
    assert_eq!(records.len(), 2 * A_PAIRS.len());
    let summary = summarize(&plan, &records).expect("shared-context summary recomputes");
    for entry in &summary.per_seed {
        approx(entry.contrast, 0.0);
    }
    assert_ne!(cache.snapshot().state, "disabled");
}

#[test]
fn execute_persists_the_explicit_cache_scope_and_allocation() {
    let out = temp_dir("shared-cache-artifacts");
    let plan = plan_with_cache(
        vec![3],
        Style::Heuristic,
        Style::First,
        DEFAULT_BUDGET_BYTES,
    );
    let summary = execute(&plan, &out, &["policy_ab".to_owned()]).expect("run succeeds");

    let run_record = fs::read_to_string(out.join("run.txt")).unwrap();
    assert!(
        run_record.contains("heuristic cache scope: shared by all heuristic seats and all games"),
        "{run_record}"
    );
    assert!(
        run_record.contains("heuristic cache budget source: explicit"),
        "{run_record}"
    );
    assert!(
        run_record.contains(&format!(
            "heuristic cache budget bytes: {DEFAULT_BUDGET_BYTES}"
        )),
        "{run_record}"
    );
    assert!(
        run_record.contains("heuristic cache initial state: unallocated"),
        "{run_record}"
    );
    assert!(
        run_record.contains("heuristic cache context overhead bytes: "),
        "{run_record}"
    );

    let cache_record = fs::read_to_string(out.join("cache.txt")).unwrap();
    assert!(
        cache_record.contains("scope: all heuristic seats in all games of this run"),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("initial state: unallocated"),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("initial context overhead bytes: "),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("final context overhead bytes: "),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("final state: ready"),
        "{cache_record}"
    );
    assert!(cache_record.contains("warning count: 0"), "{cache_record}");
    assert!(
        cache_record.contains("final allocated bytes: "),
        "{cache_record}"
    );
    assert!(summary.games > 0);

    let _ = fs::remove_dir_all(&out);
}

#[test]
fn execute_persists_a_disabled_cache_without_table_allocation() {
    let out = temp_dir("disabled-cache-artifacts");
    let plan = plan_with_cache(vec![3], Style::Heuristic, Style::First, 0);
    execute(&plan, &out, &["policy_ab".to_owned()]).expect("disabled-cache run succeeds");

    let run_record = fs::read_to_string(out.join("run.txt")).unwrap();
    assert!(
        run_record.contains("heuristic cache budget bytes: 0"),
        "{run_record}"
    );
    assert!(
        run_record.contains("heuristic cache initial state: disabled"),
        "{run_record}"
    );

    let cache_record = fs::read_to_string(out.join("cache.txt")).unwrap();
    assert!(
        cache_record.contains("final state: disabled"),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("final allocated bytes: 0"),
        "{cache_record}"
    );
    assert!(
        cache_record.contains("final capacity entries: 0"),
        "{cache_record}"
    );

    let _ = fs::remove_dir_all(&out);
}

#[test]
fn invalid_cache_budgets_fail_before_creating_the_output_directory() {
    let out = temp_dir("invalid-cache-budget");
    let plan = plan_with_cache(vec![1], Style::Heuristic, Style::First, 1);
    let error = execute(&plan, &out, &["policy_ab".to_owned()]).unwrap_err();
    assert!(error.contains("invalid heuristic cache budget"), "{error}");
    assert!(!out.exists(), "no output directory may be created");
}

#[test]
fn cache_record_states_one_allocation_warning_per_context() {
    let plan = plan_with_cache(vec![1], Style::Heuristic, Style::First, 4096);
    let initial = CacheSnapshot {
        state: "unallocated",
        budget_bytes: 4096,
        context_overhead_bytes: 256,
        allocated_bytes: 0,
        capacity_entries: 0,
        occupied_entries: 0,
        evictions: 0,
        hits: 0,
        misses: 0,
    };
    let final_snapshot = CacheSnapshot {
        state: "allocation-failed",
        budget_bytes: 4096,
        context_overhead_bytes: 256,
        allocated_bytes: 0,
        capacity_entries: 0,
        occupied_entries: 0,
        evictions: 0,
        hits: 0,
        misses: 0,
    };
    let warnings = vec![
        "policy_ab heuristic cache: heuristic cache allocation failed; caching is disabled \
         for this context and the run continues"
            .to_owned(),
    ];
    let text = cache_record_text(&plan, Some(initial), final_snapshot, &warnings);
    assert!(text.contains("final state: allocation-failed"), "{text}");
    assert!(text.contains("warning count: 1"), "{text}");
    assert!(text.matches("warning: ").count() == 1, "{text}");
    assert!(text.contains("budget source: explicit"), "{text}");
}
