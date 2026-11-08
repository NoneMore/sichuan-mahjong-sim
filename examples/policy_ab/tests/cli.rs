//! Command-line parsing and plan-selection coverage for the A/B tool.

use super::*;

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
