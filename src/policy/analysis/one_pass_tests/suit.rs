//! Complete-table checks for `suit_table_with_mask`.
//!
//! Every `[sets][pair]` entry is compared, not just the terminal distance, and
//! both the cost and the local rank mask are checked. The oracle enumerates
//! concrete per-suit target multisets; the production plain `suit_table` is
//! also cross-checked so a mask-table transition bug that changes only cost is
//! caught. Capacity patterns include physically reachable own-meld shapes and
//! deliberately non-physical `2`, `3`, and `5` capacities to stress the bounds.

use super::super::test_support::targets::{reference_needed_table as reference_suit_table, RANKS};
use super::super::test_support::Lcg;
use super::super::{suit_table, suit_table_with_mask, INF};

fn assert_suit_table(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
    context: &str,
) {
    let expected = reference_suit_table(hand, capacity, target_sets);
    let actual = suit_table_with_mask(hand, capacity, target_sets);
    let plain = suit_table(hand, capacity, target_sets);

    for sets in 0..5 {
        for pair in 0..=1 {
            let (cost, mask) = actual[sets][pair];
            assert_eq!(
                (cost, mask),
                expected[sets][pair],
                "{context}: concrete-target mismatch at [sets {sets}][pair {pair}], \
                 hand {hand:?}, capacity {capacity:?}, target_sets {target_sets}"
            );
            assert_eq!(
                cost, plain[sets][pair],
                "{context}: plain and masked costs must agree at \
                 [sets {sets}][pair {pair}], hand {hand:?}, capacity {capacity:?}"
            );
            assert!(
                mask < 1u32 << RANKS,
                "{context}: suit masks are local to nine ranks, got {mask:#x}"
            );
            if cost == INF {
                assert_eq!(
                    mask, 0,
                    "{context}: an unreachable entry must not carry a mask bit"
                );
            }
        }
    }
}

fn capacity_patterns() -> Vec<(&'static str, [u8; RANKS])> {
    vec![
        ("zero", [0u8; RANKS]),
        ("full", [4u8; RANKS]),
        ("one", [1u8; RANKS]),
        ("pung-rank-5", [4, 4, 4, 4, 1, 4, 4, 4, 4]),
        ("kong-rank-1", [0, 4, 4, 4, 4, 4, 4, 4, 4]),
        ("kong-rank-9", [4, 4, 4, 4, 4, 4, 4, 4, 0]),
        ("kong-both-ends", [0, 4, 4, 4, 4, 4, 4, 4, 0]),
        ("kong-middle", [4, 4, 4, 4, 0, 4, 4, 4, 4]),
        ("beyond-four", [5u8; RANKS]),
    ]
}

fn representative_hands() -> Vec<(&'static str, [u8; RANKS])> {
    vec![
        ("empty", [0u8; RANKS]),
        ("123456789", [1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("1112345678999", [3, 1, 1, 1, 1, 1, 1, 1, 3]),
        ("1122334455", [2, 2, 2, 2, 2, 0, 0, 0, 0]),
        ("two-rank-fives", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
        ("loose-ends", [4, 0, 0, 0, 0, 0, 0, 0, 4]),
        ("middle-kong", [0, 0, 0, 0, 4, 0, 0, 0, 0]),
        ("over-capacity", [4, 4, 4, 0, 0, 0, 0, 0, 2]),
    ]
}

#[test]
fn representative_tables_match_concrete_targets() {
    for (hand_name, hand) in representative_hands() {
        for (capacity_name, capacity) in capacity_patterns() {
            for target_sets in 0..=4 {
                assert_suit_table(
                    &hand,
                    &capacity,
                    target_sets,
                    &format!("hand {hand_name}, capacity {capacity_name}"),
                );
            }
        }
    }
}

/// Deterministic capacity sweep (LCG seed `0x0000_5EED_0F0F_0002`, 40 cases).
///
/// Held counts and capacities are drawn independently from `0..=4`; the hand is
/// trimmed to at most fourteen tiles. Capacities `2` and `3` cannot come from
/// a single own peng/kong but still exercise the bounds. The raw seed keeps the
/// historical even-state stream exactly.
fn generated_capacity_cases() -> Vec<([u8; RANKS], [u8; RANKS])> {
    let mut rng = Lcg::raw(0x0000_5EED_0F0F_0002);
    let mut cases = Vec::new();
    for _ in 0..40 {
        let mut hand = [0u8; RANKS];
        let mut capacity = [0u8; RANKS];
        for (held, cap) in hand.iter_mut().zip(capacity.iter_mut()) {
            *held = rng.below(5) as u8;
            *cap = rng.below(5) as u8;
        }
        let mut total: usize = hand.iter().map(|&held| held as usize).sum();
        for held in hand.iter_mut().rev() {
            while total > 14 && *held > 0 {
                *held -= 1;
                total -= 1;
            }
        }
        cases.push((hand, capacity));
    }
    cases
}

#[test]
fn generated_tables_match_concrete_targets() {
    for (hand, capacity) in generated_capacity_cases() {
        for target_sets in 0..=4 {
            assert_suit_table(&hand, &capacity, target_sets, "generated capacity case");
        }
    }
}

#[test]
fn empty_and_full_tables_pin_the_complete_shape() {
    let empty = [0u8; RANKS];
    let full = [4u8; RANKS];

    // Set budget one: the cheapest one-set targets use three copies of one
    // rank; their union over every triplet and sequence marks all nine ranks.
    let one_set = suit_table_with_mask(&empty, &full, 1);
    assert_eq!(one_set[0][0], (0, 0));
    assert_eq!(one_set[0][1], (2, 0x1FF), "a pair may sit at any rank");
    assert_eq!(one_set[1][0], (3, 0x1FF));
    assert_eq!(one_set[1][1], (5, 0x1FF));
    for entry in &one_set[2..] {
        assert_eq!(*entry, [(INF, 0), (INF, 0)]);
    }

    // Full set budget: every reachable row costs three copies per set plus two
    // for the pair, and the union over all cheapest targets marks every rank.
    let four_sets = suit_table_with_mask(&empty, &full, 4);
    for (sets, row) in four_sets.iter().enumerate() {
        for (pair, &(cost, mask)) in row.iter().enumerate() {
            let expected_mask = if sets == 0 && pair == 0 { 0 } else { 0x1FF };
            assert_eq!(
                (cost, mask),
                (3 * sets as u8 + 2 * pair as u8, expected_mask),
                "sets {sets}, pair {pair}"
            );
        }
    }
    assert_suit_table(&empty, &full, 4, "empty hand, full capacity");

    // Zero set budget still exposes the pair-only rows and nothing above it.
    let pair_only = suit_table_with_mask(&empty, &full, 0);
    assert_eq!(pair_only[0][0], (0, 0));
    assert_eq!(pair_only[0][1], (2, 0x1FF));
    for entry in &pair_only[1..] {
        assert_eq!(*entry, [(INF, 0), (INF, 0)]);
    }
}

#[test]
fn zero_capacity_cannot_host_any_set_or_pair() {
    let empty = [0u8; RANKS];
    let zero = [0u8; RANKS];
    for target_sets in 0..=4 {
        let table = suit_table_with_mask(&empty, &zero, target_sets);
        assert_eq!(table[0][0], (0, 0), "target_sets {target_sets}");
        assert_eq!(
            table[0][1],
            (INF, 0),
            "a zero-capacity suit cannot place a pair"
        );
        for (sets, row) in table.iter().enumerate().skip(1) {
            assert_eq!(
                *row,
                [(INF, 0), (INF, 0)],
                "zero capacity cannot hold {sets} sets"
            );
        }
        assert_suit_table(&empty, &zero, target_sets, "zero capacity");
    }
}

#[test]
fn capacity_modes_bound_sets_pairs_and_masks() {
    let empty = [0u8; RANKS];

    // One concealed copy per rank: sequences fit, triplets and pairs do not.
    let one = [1u8; RANKS];
    let table = suit_table_with_mask(&empty, &one, 4);
    assert_eq!(table[1][0], (3, 0x1FF));
    assert_eq!(table[2][0], (6, 0x1FF));
    assert_eq!(table[3][0], (9, 0x1FF));
    assert_eq!(table[4][0], (INF, 0), "nine ranks cannot host four sets");
    for (sets, row) in table.iter().enumerate().take(4) {
        assert_eq!(
            row[1],
            (INF, 0),
            "row {sets} cannot place a pair with one copy per rank"
        );
    }
    assert_suit_table(&empty, &one, 4, "one copy per rank");

    // Five copies on one rank: one triplet and the pair may coexist there.
    let five_on_first = [5u8, 0, 0, 0, 0, 0, 0, 0, 0];
    let table = suit_table_with_mask(&empty, &five_on_first, 2);
    assert_eq!(table[0][1], (2, 0b1));
    assert_eq!(table[1][0], (3, 0b1));
    assert_eq!(
        table[1][1],
        (5, 0b1),
        "triplet plus pair share the five-copy rank"
    );
    assert_eq!(
        table[2][0],
        (INF, 0),
        "two triplets need six copies on the only rank"
    );
    assert_eq!(table[2][1], (INF, 0));

    // A zero-capacity middle rank blocks pairs and crossing sequences there.
    let blocked_middle = [4, 4, 4, 4, 0, 4, 4, 4, 4];
    let table = suit_table_with_mask(&empty, &blocked_middle, 1);
    assert_eq!(table[0][1], (2, 0x1EF), "no pair bit at the blocked rank");
    assert_eq!(table[1][0], (3, 0x1EF));
    assert_eq!(table[1][1], (5, 0x1EF));
    assert_suit_table(&empty, &blocked_middle, 1, "blocked middle rank");

    // A kong on rank 1 leaves ranks 2..=9 for pairs, triplets, and sequences.
    let kong_low = [0, 4, 4, 4, 4, 4, 4, 4, 4];
    let table = suit_table_with_mask(&empty, &kong_low, 1);
    assert_eq!(table[0][1], (2, 0x1FE));
    assert_eq!(table[1][0], (3, 0x1FE));
    assert_suit_table(&empty, &kong_low, 1, "kong on rank 1");

    // A kong on ranks 7..=9 leaves only ranks 1..=6 reachable.
    let kong_high = [4, 4, 4, 4, 4, 4, 0, 0, 0];
    let table = suit_table_with_mask(&empty, &kong_high, 1);
    assert_eq!(table[0][1], (2, 0x3F));
    assert_eq!(table[1][0], (3, 0x3F));
    assert_suit_table(&empty, &kong_high, 1, "kong on ranks 7..=9");
}

/// The mask records a *need*, not every rank a cheapest target uses.
///
/// With two held rank-5 copies and a pung capacity of one, every cheapest
/// one-set target is a sequence through rank 5, but rank 5 needs no draw and
/// therefore never appears in the mask.
#[test]
fn mask_marks_needed_ranks_not_used_ranks() {
    let hand = [0, 0, 0, 0, 2, 0, 0, 0, 0];
    let capacity = [4, 4, 4, 4, 1, 4, 4, 4, 4];
    let table = suit_table_with_mask(&hand, &capacity, 1);
    assert_eq!(
        table[1][0],
        (2, 0x6C),
        "sequences 234/345/456 need ranks 2,3,5,6 only"
    );
    assert_eq!(table[0][1], (2, 0x1EF));
    assert_eq!(table[1][1], (4, 0x1EF));
    assert_suit_table(&hand, &capacity, 1, "needed versus used ranks");
}
