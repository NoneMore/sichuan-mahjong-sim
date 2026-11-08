//! Artifact, report, metadata, and failure-path coverage.

use super::*;
use crate::cache_tool::CacheSnapshot;

fn temp_dir(name: &str) -> PathBuf {
    let path = env::temp_dir().join(format!("policy_ab_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
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
