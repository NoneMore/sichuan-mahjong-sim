//! Game execution and shared heuristic-context coverage.

use super::*;

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
