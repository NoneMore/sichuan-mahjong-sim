//! Bounded needed-mask DP checks for the minimal `g > d` candidate.
//!
//! This module is deliberately separate from the long-term complete-mask
//! evidence in `one_pass_tests.rs` and `suit.rs`: those continue to check the
//! original complete routes and concrete-target oracles without depending on
//! any bounded entry point. Everything here specifically pins the candidate:
//!
//! - every per-suit entry with cost `<= d` is preserved exactly, and every
//!   entry above `d` is normalized to `(INF, 0)`, for the concrete-target
//!   reference and for the complete table;
//! - a complete hand query and the bounded hand query agree with the
//!   independent global oracle, including seven-pairs-only and tied routes;
//! - the strict cutoff keeps a prefix at `g == d` that finishes with zero-cost
//!   transitions, while never exposing a partial entry above `d`;
//! - bounded suit tables are cached separately: the complete cache is read
//!   first, never written by a bounded build, and a wider cutoff rebuilds an
//!   entry while a narrower query may reuse it.
//!
//! Fixtures are the existing physical boundary/sampled positions plus small
//! hand-constructed tables. The oracle is the independent concrete-target
//! enumeration from `oracle.rs`, not the candidate DP.

use super::super::test_fixtures::{boundary_states, hand_from_counts, sampled_states, tile};
use super::super::{suit_table_mask_bounded, suit_table_with_mask, DistanceEvaluator, INF};
use super::super::test_support::targets::reference_needed_table as reference_suit_table;
use super::oracle::reference_hand_mask;
use crate::tile::{Suit, TILE_KINDS};

fn grouped(groups: &[(Suit, &[u8])]) -> [u8; TILE_KINDS] {
    let mut counts = [0u8; TILE_KINDS];
    for &(suit, ranks) in groups {
        for &rank in ranks {
            counts[tile(suit, rank).index() as usize] += 1;
        }
    }
    counts
}

fn assert_bounded_suit_table(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize, cutoff: u8) {
    let expected = reference_suit_table(hand, capacity, target_sets);
    let actual = suit_table_mask_bounded(hand, capacity, target_sets, cutoff);
    for sets in 0..5 {
        for pair in 0..=1 {
            let expected_entry = expected[sets][pair];
            let actual_entry = actual[sets][pair];
            if expected_entry.0 <= cutoff {
                assert_eq!(
                    actual_entry, expected_entry,
                    "bounded table lost an entry at or below the cutoff: \
                     hand {hand:?}, capacity {capacity:?}, target_sets {target_sets}, \
                     cutoff {cutoff}, [{sets}][{pair}]"
                );
            } else {
                assert_eq!(
                    actual_entry,
                    (INF, 0),
                    "bounded table exposed an entry above the cutoff: \
                     hand {hand:?}, capacity {capacity:?}, target_sets {target_sets}, \
                     cutoff {cutoff}, [{sets}][{pair}]"
                );
            }
        }
    }
}

fn bounded_suit_fixtures() -> Vec<(&'static str, [u8; 9], [u8; 9])> {
    vec![
        ("empty-full", [0u8; 9], [4u8; 9]),
        ("one-per-rank", [1u8; 9], [4u8; 9]),
        ("tenpai-like", [3, 1, 1, 1, 1, 1, 1, 1, 3], [4u8; 9]),
        ("low-pairs", [2, 2, 2, 2, 2, 0, 0, 0, 0], [4u8; 9]),
        (
            "blocked-middle",
            [1, 2, 3, 3, 2, 1, 0, 0, 0],
            [4, 4, 4, 4, 0, 4, 4, 4, 4],
        ),
        (
            "pung-rank-five",
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [4, 4, 4, 4, 1, 4, 4, 4, 4],
        ),
        (
            "kong-both-ends",
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 4, 4, 4, 4, 4, 4, 4, 0],
        ),
        ("over-capacity", [4, 4, 4, 0, 0, 0, 0, 0, 2], [5u8; 9]),
    ]
}

#[test]
fn bounded_suit_tables_preserve_the_exact_prefix_for_every_fixture() {
    let cutoffs = [0u8, 1, 2, 3, 4, 5, 8, 13, 14];
    for (name, hand, capacity) in bounded_suit_fixtures() {
        for target_sets in 0..=4 {
            let complete = suit_table_with_mask(&hand, &capacity, target_sets);
            let expected = reference_suit_table(&hand, &capacity, target_sets);
            assert_eq!(
                complete, expected,
                "{name}: the complete needed table must stay exact, \
                 target_sets {target_sets}"
            );
            for cutoff in cutoffs {
                assert_bounded_suit_table(&hand, &capacity, target_sets, cutoff);
            }
        }
    }
}

/// A prefix already at the exact cutoff must survive a zero-cost suffix.
///
/// The four-set target `123,123,456,789` needs the second 1W at rank 1; every
/// later rank is covered by the hand, so the whole path to a cost-one table
/// entry accumulates its only cost at the first rank. A `g >= d` cutoff would
/// reject that prefix and lose the entry; the required strict `g > d` cutoff
/// keeps it and its needed bit.
#[test]
fn strict_cutoff_keeps_an_equal_cost_prefix_with_a_zero_cost_suffix() {
    let hand = [1u8, 2, 2, 1, 1, 1, 1, 1, 2];
    let capacity = [4u8; 9];
    let expected = reference_suit_table(&hand, &capacity, 4);
    assert_eq!(
        expected[4][0].0, 1,
        "test premise: the prefix reaches the exact cutoff at rank 1"
    );
    assert_ne!(
        expected[4][0].1 & 1,
        0,
        "test premise: the cost is the needed first rank"
    );
    assert_bounded_suit_table(&hand, &capacity, 4, expected[4][0].0);
}

#[test]
fn bounded_hand_queries_match_complete_and_concrete_oracles() {
    let mut states = boundary_states();
    states.extend(sampled_states());
    let mut meld_counts = [0usize; 5];
    let mut missing_counts = [0usize; 3];
    let mut zero_deficits = 0usize;

    for state in &states {
        let expected = reference_hand_mask(&state.hand, &state.melds, state.missing);
        let mut bounded =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        assert_eq!(
            bounded.deficit_with_mask_bounded(&state.hand, expected.0),
            expected,
            "{}: bounded global result differs from the concrete oracle",
            state.label
        );
        // A warm repeat is served from the shared hand mask cache and must
        // keep the same complete value.
        assert_eq!(
            bounded.deficit_with_mask_bounded(&state.hand, expected.0),
            expected,
            "{}: warm bounded result changed",
            state.label
        );

        let mut complete =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        assert_eq!(
            complete.deficit_with_mask(&state.hand),
            expected,
            "{}: complete route differs from the concrete oracle",
            state.label
        );

        if expected.0 == 0 {
            zero_deficits += 1;
        }
        meld_counts[state.melds.len()] += 1;
        missing_counts[state.missing.index()] += 1;
    }

    for (meld_count, count) in meld_counts.iter().enumerate() {
        assert!(*count > 0, "no bounded fixture with {meld_count} melds");
    }
    for (missing, count) in missing_counts.iter().enumerate() {
        assert!(*count > 0, "no bounded fixture with missing suit {missing}");
    }
    assert_eq!(
        zero_deficits, 0,
        "the physical sweep is post-discard; zero-distance hands are covered \
         explicitly below"
    );
}

#[test]
fn zero_distance_hands_keep_the_empty_mask_under_a_zero_cutoff() {
    let standard = hand_from_counts(&grouped(&[(
        Suit::Wan,
        &[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9],
    )]));
    let seven_pairs = hand_from_counts(&grouped(&[(
        Suit::Wan,
        &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    )]));

    for (label, hand) in [("standard", standard), ("seven-pairs", seven_pairs)] {
        let expected = reference_hand_mask(&hand, &[], Suit::Tiao);
        assert_eq!(expected, (0, 0), "{label}: test premise");
        let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
        assert_eq!(
            evaluator.deficit_with_mask_bounded(&hand, 0),
            (0, 0),
            "{label}: a zero cutoff keeps only the exact zero-cost target"
        );
    }
}

#[test]
fn complete_and_bounded_queries_interleave_on_one_evaluator() {
    for state in boundary_states().iter().take(8) {
        let expected = reference_hand_mask(&state.hand, &state.melds, state.missing);

        let mut complete_first =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        assert_eq!(complete_first.deficit_with_mask(&state.hand), expected);
        assert_eq!(
            complete_first.deficit_with_mask_bounded(&state.hand, expected.0),
            expected,
            "{}: bounded after complete",
            state.label
        );

        let mut bounded_first =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        assert_eq!(
            bounded_first.deficit_with_mask_bounded(&state.hand, expected.0),
            expected
        );
        assert_eq!(
            bounded_first.deficit_with_mask(&state.hand),
            expected,
            "{}: complete after bounded",
            state.label
        );
        assert_eq!(
            bounded_first.deficit_with_mask(&state.hand),
            expected,
            "{}: repeated complete after bounded",
            state.label
        );
    }
}

/// Two all-Tong tenpai hands that share a zero Wan vector.
fn tenpai_a() -> [u8; TILE_KINDS] {
    grouped(&[(Suit::Tong, &[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9])])
}

fn tenpai_b() -> [u8; TILE_KINDS] {
    grouped(&[(Suit::Tong, &[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 8])])
}

/// A sparse hand with the same zero Wan vector: almost every held tile is in
/// the missing Tiao suit, so an optimal target must use Wan sets whose per-suit
/// costs are far above one.
fn sparse_shared_wan() -> [u8; TILE_KINDS] {
    grouped(&[
        (Suit::Tong, &[1]),
        (Suit::Tiao, &[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    ])
}

fn bounded_wan_cutoff(evaluator: &DistanceEvaluator<'_>) -> Option<u8> {
    evaluator
        .bounded_needed_cache
        .as_ref()?
        .suit
        .get(&(Suit::Wan.index() as u8, [0u8; 9]))
        .map(|entry| entry.cutoff)
}

#[test]
fn bounded_builds_never_pollute_the_complete_suit_cache() {
    let narrow = hand_from_counts(&tenpai_a());
    let wide = hand_from_counts(&sparse_shared_wan());
    let narrow_ref = reference_hand_mask(&narrow, &[], Suit::Tiao);
    let wide_ref = reference_hand_mask(&wide, &[], Suit::Tiao);
    assert_eq!(narrow_ref.0, 1, "test premise: the narrow hand is tenpai");
    assert!(
        wide_ref.0 > narrow_ref.0,
        "test premise: the sparse hand has a wider exact distance"
    );

    let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    assert_eq!(
        evaluator.deficit_with_mask_bounded(&narrow, narrow_ref.0),
        narrow_ref
    );
    let key = (Suit::Wan.index() as u8, [0u8; 9]);
    assert_eq!(bounded_wan_cutoff(&evaluator), Some(narrow_ref.0));
    assert!(
        !evaluator.suit_mask_cache.contains_key(&key),
        "a bounded build must not write the complete suit cache"
    );

    // The complete query for a different hand sharing the Wan vector must
    // rebuild the genuine complete table, not reuse a truncated cutoff-one one.
    assert_eq!(evaluator.deficit_with_mask(&wide), wide_ref);
    let full_wan = reference_suit_table(&[0u8; 9], &[4u8; 9], 4);
    assert_eq!(
        evaluator.suit_mask_cache.get(&key),
        Some(&full_wan),
        "the complete cache must hold the full Wan table after the wide query"
    );
}

#[test]
fn bounded_suit_cache_reuses_narrow_queries_and_rebuilds_wider_ones() {
    let narrow_a = hand_from_counts(&tenpai_a());
    let narrow_b = hand_from_counts(&tenpai_b());
    let wide = hand_from_counts(&sparse_shared_wan());
    let narrow_a_ref = reference_hand_mask(&narrow_a, &[], Suit::Tiao);
    let narrow_b_ref = reference_hand_mask(&narrow_b, &[], Suit::Tiao);
    let wide_ref = reference_hand_mask(&wide, &[], Suit::Tiao);
    assert_eq!(narrow_a_ref.0, 1);
    assert_eq!(narrow_b_ref.0, 1);

    // Wide first, then two different narrow hands: both narrow queries reuse
    // the wider cached suit table, and the cache entry stays wide.
    let mut wide_first = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    assert_eq!(
        wide_first.deficit_with_mask_bounded(&wide, wide_ref.0),
        wide_ref
    );
    assert_eq!(bounded_wan_cutoff(&wide_first), Some(wide_ref.0));
    assert_eq!(
        wide_first.deficit_with_mask_bounded(&narrow_a, 1),
        narrow_a_ref
    );
    assert_eq!(
        bounded_wan_cutoff(&wide_first),
        Some(wide_ref.0),
        "a narrower query must reuse, not shrink, the cached cutoff"
    );
    assert_eq!(
        wide_first.deficit_with_mask_bounded(&narrow_b, 1),
        narrow_b_ref
    );
    assert_eq!(bounded_wan_cutoff(&wide_first), Some(wide_ref.0));

    // Narrow first, then wide: the wider query must rebuild the shared suit.
    let mut narrow_first = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    assert_eq!(
        narrow_first.deficit_with_mask_bounded(&narrow_a, 1),
        narrow_a_ref
    );
    assert_eq!(bounded_wan_cutoff(&narrow_first), Some(1));
    assert_eq!(
        narrow_first.deficit_with_mask_bounded(&wide, wide_ref.0),
        wide_ref
    );
    assert_eq!(
        bounded_wan_cutoff(&narrow_first),
        Some(wide_ref.0),
        "a wider query must rebuild the shared suit at its own cutoff"
    );
}

#[test]
fn bounded_results_are_stable_under_cold_warm_and_reversed_order() {
    let hands = [
        hand_from_counts(&tenpai_a()),
        hand_from_counts(&tenpai_b()),
        hand_from_counts(&sparse_shared_wan()),
    ];
    let expected: Vec<(u8, u32)> = hands
        .iter()
        .map(|hand| reference_hand_mask(hand, &[], Suit::Tiao))
        .collect();

    // Forward, repeated, then reversed on one evaluator.
    let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    for (hand, &reference) in hands.iter().zip(expected.iter()) {
        assert_eq!(
            evaluator.deficit_with_mask_bounded(hand, reference.0),
            reference
        );
    }
    for (hand, &reference) in hands.iter().zip(expected.iter()) {
        assert_eq!(
            evaluator.deficit_with_mask_bounded(hand, reference.0),
            reference,
            "warm repeat"
        );
    }
    for (hand, &reference) in hands.iter().zip(expected.iter()).rev() {
        assert_eq!(
            evaluator.deficit_with_mask_bounded(hand, reference.0),
            reference,
            "reversed order"
        );
    }

    // A different fixed context never leaks into another evaluator.
    let mut other_missing = DistanceEvaluator::new(&[], Suit::Tong).expect("no melds");
    let other = expected[0];
    let other_reference = reference_hand_mask(&hands[0], &[], Suit::Tong);
    assert_ne!(
        other_reference, other,
        "test premise: changing the missing suit changes the analysis"
    );
    assert_eq!(
        other_missing.deficit_with_mask_bounded(&hands[0], other_reference.0),
        other_reference
    );
    for (hand, &reference) in hands.iter().zip(expected.iter()) {
        assert_eq!(
            evaluator.deficit_with_mask_bounded(hand, reference.0),
            reference,
            "the reused Tiao evaluator must stay exact"
        );
    }
}

/// Per-suit bounded prefixes for the physical boundary/sample states.
///
/// The hand-level test above already checks the global result; this checks the
/// intermediate tables used by the bounded combine for every suit, including
/// own-meld and missing-suit capacity shapes.
#[test]
fn bounded_physical_suit_tables_preserve_the_exact_prefix() {
    let mut states = boundary_states();
    states.extend(sampled_states());
    for state in &states {
        let (cutoff, _) = reference_hand_mask(&state.hand, &state.melds, state.missing);
        let capacity = super::super::test_fixtures::capacity_by_suit(&state.melds, state.missing);
        for suit in Suit::ALL {
            let mut local = [0u8; 9];
            let start = suit.index() * 9;
            local.copy_from_slice(&state.hand.counts()[start..start + 9]);
            assert_bounded_suit_table(
                &local,
                &capacity[suit.index()],
                4 - state.melds.len(),
                cutoff,
            );
        }
    }
}

/// A complete table entry above the cutoff must never leak a mask bit.
#[test]
fn unreachable_and_above_cutoff_entries_are_normalized() {
    let zero = [0u8; 9];
    let full = [4u8; 9];
    let table = suit_table_mask_bounded(&zero, &zero, 4, 8);
    assert_eq!(table[0][0], (0, 0));
    for (sets, row) in table.iter().enumerate().skip(1) {
        for &entry in row {
            assert_eq!(
                entry,
                (INF, 0),
                "zero capacity cannot host {sets} sets above the cutoff"
            );
        }
    }

    // A cutoff above the largest possible finite cost keeps the whole table.
    let complete = suit_table_with_mask(&zero, &full, 4);
    assert_eq!(suit_table_mask_bounded(&zero, &full, 4, 14), complete);
    let above = suit_table_mask_bounded(&zero, &full, 4, 3);
    for (sets, row) in above.iter().enumerate() {
        for (pair, &(cost, mask)) in row.iter().enumerate() {
            if cost != INF {
                assert!(cost <= 3, "sets {sets}, pair {pair}");
                assert!(mask > 0 || (sets == 0 && pair == 0));
            }
        }
    }
}

#[test]
fn complete_suit_tables_serve_bounded_without_allocating_the_bounded_cache() {
    for state in boundary_states().iter().take(6) {
        let expected = reference_hand_mask(&state.hand, &state.melds, state.missing);
        let mut evaluator =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        assert_eq!(evaluator.deficit_with_mask(&state.hand), expected);
        // Force the bounded hand path while keeping the complete suit tables.
        evaluator.hand_mask_cache.clear();
        assert!(evaluator.bounded_needed_cache.is_none());
        assert_eq!(
            evaluator.deficit_with_mask_bounded(&state.hand, expected.0),
            expected,
            "{}: complete suit tables must serve the bounded combine",
            state.label
        );
        assert!(
            evaluator.bounded_needed_cache.is_none(),
            "{}: serving a bounded query from complete tables must not allocate \
             the bounded cache",
            state.label
        );
    }
}
