//! Independent checks for the closed-form seven-pairs deficit.
//!
//! The reference walks seven-pairs targets directly: each non-missing kind
//! contributes 0, 2, or 4 target copies (supplying 0, 1, or 2 pair units) and
//! the cheapest way to reach exactly seven units is kept. It never reasons
//! about held pair units or odd kinds, so it is independent of the production
//! `K + max(0, K - O)` identity.
//!
//! The class sweep covers every held-count category of the 18 non-missing
//! kinds with at most 14 held tiles, classified by how many kinds hold 0, 1,
//! 2, 3, or 4 copies (241 classes). Standard-target checks that isolate the
//! seven-pairs gate use a one-suit triplet/sequence enumeration.

use super::*;
use crate::action::Meld;
use crate::hand::Hand;
use crate::seat::Seat;
use crate::tile::{Suit, Tile};

const UNITS: usize = 7;

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

fn hand_from_counts(counts: &[u8; TILE_KINDS]) -> Hand {
    let mut hand = Hand::new();
    for kind in Tile::ALL {
        for _ in 0..counts[kind.index() as usize] {
            hand.add(kind).expect("at most four copies per kind");
        }
    }
    hand
}

/// Cheapest seven-pairs target found by enumerating per-kind target copies.
fn enumerated_seven_pairs_deficit(counts: &[u8; TILE_KINDS], missing: Suit) -> u8 {
    let mut best = [INF; UNITS + 1];
    best[0] = 0;
    for kind in Tile::ALL {
        if kind.suit() == missing {
            continue;
        }
        let held = counts[kind.index() as usize];
        let mut next = [INF; UNITS + 1];
        for units in 0..=UNITS {
            let current = best[units];
            if current == INF {
                continue;
            }
            for copies in [0u8, 2, 4] {
                let added = (copies / 2) as usize;
                if units + added > UNITS {
                    continue;
                }
                let extra = copies.saturating_sub(held);
                let total = current.saturating_add(extra);
                let slot = &mut next[units + added];
                if total < *slot {
                    *slot = total;
                }
            }
        }
        best = next;
    }
    best[UNITS]
}

/// One-suit target enumeration used only by the declared-meld gating fixture
/// below. For that specific seven-pairs hand, three sets plus a pair have
/// minimum cost one in-suit; a target using the empty other suit costs at least
/// two for its pair or three for a set, so it cannot improve that minimum.
/// This is not a general reduction for arbitrary hands or capacities.
fn one_suit_standard_deficit(hand: &[u8; 9], target_sets: usize) -> u8 {
    let mut sets: Vec<[u8; 9]> = Vec::new();
    for rank in 0..9 {
        let mut triplet = [0u8; 9];
        triplet[rank] = 3;
        sets.push(triplet);
    }
    for rank in 0..7 {
        let mut sequence = [0u8; 9];
        sequence[rank] = 1;
        sequence[rank + 1] = 1;
        sequence[rank + 2] = 1;
        sets.push(sequence);
    }

    fn search(
        start: usize,
        remaining: usize,
        sets: &[[u8; 9]],
        hand: &[u8; 9],
        target: &mut [u8; 9],
        best: &mut u8,
    ) {
        if remaining == 0 {
            for pair in 0..9 {
                let mut deficit = 0u8;
                for rank in 0..9 {
                    let mut need = target[rank];
                    if rank == pair {
                        need += 2;
                    }
                    if need > COPIES_PER_TILE {
                        deficit = INF;
                        break;
                    }
                    deficit += need.saturating_sub(hand[rank]);
                }
                *best = (*best).min(deficit);
            }
            return;
        }

        for index in start..sets.len() {
            let set = &sets[index];
            if (0..9).any(|rank| target[rank] + set[rank] > COPIES_PER_TILE) {
                continue;
            }
            for rank in 0..9 {
                target[rank] += set[rank];
            }
            search(index, remaining - 1, sets, hand, target, best);
            for rank in 0..9 {
                target[rank] -= set[rank];
            }
        }
    }

    let mut best = INF;
    let mut target = [0u8; 9];
    search(0, target_sets, &sets, hand, &mut target, &mut best);
    best
}

fn suit_slice(hand: &Hand, suit: Suit) -> [u8; 9] {
    let mut slice = [0u8; 9];
    let start = suit.index() * 9;
    slice.copy_from_slice(&hand.counts()[start..start + 9]);
    slice
}

/// Builds the canonical held-count vector for one of the 241 classes.
///
/// The class is described by how many non-missing kinds hold 1, 2, 3, or 4
/// copies; `reverse` assigns those kinds from the opposite end of the eligible
/// list so both suit and rank placement orders are exercised.
fn class_counts(counts_by_copies: [usize; 4], missing: Suit, reverse: bool) -> [u8; TILE_KINDS] {
    let eligible: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .collect();
    let mut counts = [0u8; TILE_KINDS];
    let mut assigned = 0usize;
    let assignments = [
        (4u8, counts_by_copies[3]),
        (3, counts_by_copies[2]),
        (2, counts_by_copies[1]),
        (1, counts_by_copies[0]),
    ];
    for (copies, kinds) in assignments {
        for _ in 0..kinds {
            let chosen = if reverse {
                eligible[eligible.len() - 1 - assigned]
            } else {
                eligible[assigned]
            };
            counts[chosen.index() as usize] = copies;
            assigned += 1;
        }
    }
    counts
}

#[test]
fn closed_form_matches_target_enumeration_for_every_held_count_class() {
    let missing = Suit::Tiao;
    let mut classes = 0usize;

    for singles in 0..=18usize {
        for pairs in 0..=7usize {
            for triples in 0..=4usize {
                for quads in 0..=3usize {
                    let kinds = singles + pairs + triples + quads;
                    let total = singles + 2 * pairs + 3 * triples + 4 * quads;
                    if kinds > 18 || total > 14 {
                        continue;
                    }
                    classes += 1;

                    for reverse in [false, true] {
                        let counts =
                            class_counts([singles, pairs, triples, quads], missing, reverse);
                        let hand = hand_from_counts(&counts);
                        let expected = enumerated_seven_pairs_deficit(&counts, missing);
                        let evaluator = DistanceEvaluator::new(&[], missing)
                            .expect("an empty meld set is legal");
                        assert_eq!(
                            evaluator.seven_pairs_deficit(&counts),
                            expected,
                            "class 1x{singles} 2x{pairs} 3x{triples} 4x{quads}, \
                             reverse {reverse}, hand {hand:?}"
                        );
                    }
                }
            }
        }
    }

    assert_eq!(classes, 241, "held-count classes with at most 14 tiles");
}

#[test]
fn quads_triples_and_pair_units_follow_the_closed_form() {
    let missing = Suit::Tiao;
    let evaluator = DistanceEvaluator::new(&[], missing).expect("an empty meld set is legal");

    // A quad is two free pair units: 1111 + four pairs + one single.
    let mut counts = [0u8; TILE_KINDS];
    counts[tile(Suit::Wan, 1).index() as usize] = 4;
    for rank in 2..=5 {
        counts[tile(Suit::Wan, rank).index() as usize] = 2;
    }
    counts[tile(Suit::Wan, 6).index() as usize] = 1;
    assert_eq!(enumerated_seven_pairs_deficit(&counts, missing), 1);
    assert_eq!(evaluator.seven_pairs_deficit(&counts), 1);

    // A held triple is one free unit plus one odd kind: 111 + five pairs.
    let mut triple = [0u8; TILE_KINDS];
    triple[tile(Suit::Wan, 1).index() as usize] = 3;
    for rank in 2..=6 {
        triple[tile(Suit::Wan, rank).index() as usize] = 2;
    }
    assert_eq!(enumerated_seven_pairs_deficit(&triple, missing), 1);
    assert_eq!(evaluator.seven_pairs_deficit(&triple), 1);

    // No odd kinds left to complete: six pairs need a fresh kind, two copies.
    let mut even = [0u8; TILE_KINDS];
    for rank in 1..=6 {
        even[tile(Suit::Wan, rank).index() as usize] = 2;
    }
    assert_eq!(enumerated_seven_pairs_deficit(&even, missing), 2);
    assert_eq!(evaluator.seven_pairs_deficit(&even), 2);

    // Fourteen distinct singletons: seven extra pairs at one copy each.
    let mut singles = [0u8; TILE_KINDS];
    for kind in Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .take(14)
    {
        singles[kind.index() as usize] = 1;
    }
    assert_eq!(enumerated_seven_pairs_deficit(&singles, missing), 7);
    assert_eq!(evaluator.seven_pairs_deficit(&singles), 7);

    // Four quads plus two pairs already exceed seven units.
    let mut saturated = [0u8; TILE_KINDS];
    for rank in 1..=4 {
        saturated[tile(Suit::Wan, rank).index() as usize] = 4;
    }
    for rank in 5..=6 {
        saturated[tile(Suit::Wan, rank).index() as usize] = 2;
    }
    assert_eq!(enumerated_seven_pairs_deficit(&saturated, missing), 0);
    assert_eq!(evaluator.seven_pairs_deficit(&saturated), 0);
}

#[test]
fn pair_units_beyond_seven_do_not_underflow() {
    let missing = Suit::Tiao;

    // Eight pairs: P = 8 > 7, so the saturating subtraction must stay at zero.
    let mut eight_pairs = [0u8; TILE_KINDS];
    for rank in 1..=8 {
        eight_pairs[tile(Suit::Tong, rank).index() as usize] = 2;
    }
    let hand = hand_from_counts(&eight_pairs);
    assert_eq!(hand.total(), 16);
    let mut evaluator = DistanceEvaluator::new(&[], missing).expect("an empty meld set is legal");
    assert_eq!(evaluator.seven_pairs_deficit(&eight_pairs), 0);
    assert_eq!(evaluator.deficit(&hand), 0);
    assert_eq!(
        enumerated_seven_pairs_deficit(&eight_pairs, missing),
        0,
        "seven of the eight pairs are a legal target"
    );

    // Four quads and a pair: P = 9 > 7 as well.
    let mut many_units = [0u8; TILE_KINDS];
    for rank in 1..=4 {
        many_units[tile(Suit::Wan, rank).index() as usize] = 4;
    }
    many_units[tile(Suit::Wan, 9).index() as usize] = 2;
    assert_eq!(evaluator.seven_pairs_deficit(&many_units), 0);
    assert_eq!(enumerated_seven_pairs_deficit(&many_units, missing), 0);
}

#[test]
fn missing_suit_kinds_are_excluded_from_the_pair_units() {
    let missing = Suit::Tiao;

    // Ten missing-suit tiles would be five free units if they counted.
    let mut counts = [0u8; TILE_KINDS];
    for rank in 1..=5 {
        counts[tile(Suit::Tiao, rank).index() as usize] = 2;
    }
    counts[tile(Suit::Wan, 1).index() as usize] = 1;

    let evaluator = DistanceEvaluator::new(&[], missing).expect("an empty meld set is legal");
    assert_eq!(evaluator.seven_pairs_deficit(&counts), 13);
    assert_eq!(enumerated_seven_pairs_deficit(&counts, missing), 13);

    // Repeating the same held counts under a different missing suit keeps the
    // formula and the enumeration aligned.
    let other_missing = Suit::Tong;
    let other = DistanceEvaluator::new(&[], other_missing).expect("an empty meld set is legal");
    let expected = enumerated_seven_pairs_deficit(&counts, other_missing);
    assert_eq!(other.seven_pairs_deficit(&counts), expected);
    assert_eq!(expected, 3, "ten Tiao tiles are now five free pair units");
}

#[test]
fn declared_melds_disable_the_seven_pairs_route() {
    let missing = Suit::Tiao;

    // Seven distinct Wan pairs at ranks 1, 2, 4, 5, 7, 8, 9. The only set they
    // contain is 789, available twice, so three sets plus a pair always needs
    // at least one drawn tile even though seven pairs is already complete.
    let seven_pairs_counts = [2u8, 2, 0, 2, 2, 0, 2, 2, 2];
    let mut counts = [0u8; TILE_KINDS];
    for (index, &copies) in seven_pairs_counts.iter().enumerate() {
        counts[index] = copies;
    }
    let hand = hand_from_counts(&counts);
    assert_eq!(hand.total(), 14);

    let closed = DistanceEvaluator::new(&[], missing).expect("an empty meld set is legal");
    assert_eq!(closed.seven_pairs_deficit(&counts), 0);
    assert_eq!(
        DistanceEvaluator::new(&[], missing)
            .expect("an empty meld set is legal")
            .deficit(&hand),
        0
    );

    // Both a peng and a concealed kong are declared melds, so both disable the
    // seven-pairs route and leave three concealed sets plus one pair.
    let expected = one_suit_standard_deficit(&suit_slice(&hand, Suit::Wan), 3);
    assert!(expected > 0);
    let peng = [Meld::peng(tile(Suit::Tong, 9), Seat::DEALER)];
    let kong = [Meld::concealed_kong(tile(Suit::Tong, 9))];
    for melds in [peng, kong] {
        let mut open = DistanceEvaluator::new(&melds, missing).expect("one meld is legal");
        assert_eq!(
            open.deficit(&hand),
            expected,
            "declared meld {melds:?} must fall back to the standard-only route"
        );
    }
}
