//! Persisted-record codec, validation, and summary statistics coverage.

use super::*;

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
