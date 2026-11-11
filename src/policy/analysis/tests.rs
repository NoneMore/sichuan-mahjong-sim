//! Independent checks for the exact target-deficit analysis.
//!
//! The oracle below enumerates standard winning concealed parts by choosing
//! `4 - meld_count` sets (triplets and sequences) plus one pair, and seven-pairs
//! targets by choosing 0/2/4 copies of each kind. It shares no decomposition
//! code with the production per-suit dynamic program. Cases are bounded to two
//! non-missing suits so the enumeration stays small; the declared missing suit
//! blocks the third suit in both implementations.

use super::*;
use crate::action::Meld;
use crate::hand::Hand;
use crate::seat::Seat;
use crate::tile::{Suit, Tile};

use super::test_fixtures::capacity_by_kind;
use super::test_support::targets::{add_counts, subtract_counts, target_deficit};
use super::test_support::{hand, suit_hand, tile, Lcg};

#[allow(clippy::too_many_arguments)]
fn standard_search(
    start: usize,
    remaining: usize,
    sets: &[[u8; TILE_KINDS]],
    pairs: &[[u8; TILE_KINDS]],
    cap: &[u8; TILE_KINDS],
    hand: &Hand,
    counts: &mut [u8; TILE_KINDS],
    best: &mut u8,
) {
    if remaining == 0 {
        for pair in pairs {
            let mut total = *counts;
            let mut legal = true;
            for index in 0..TILE_KINDS {
                total[index] += pair[index];
                if total[index] > cap[index] {
                    legal = false;
                    break;
                }
            }
            if legal {
                *best = (*best).min(target_deficit(hand.counts(), &total));
            }
        }
        return;
    }

    for index in start..sets.len() {
        let set = &sets[index];
        let legal = (0..TILE_KINDS).all(|kind| counts[kind] + set[kind] <= cap[kind]);
        if !legal {
            continue;
        }
        add_counts(counts, set);
        standard_search(index, remaining - 1, sets, pairs, cap, hand, counts, best);
        subtract_counts(counts, set);
    }
}

fn seven_pairs_search(
    position: usize,
    units: usize,
    kinds: &[Tile],
    cap: &[u8; TILE_KINDS],
    hand: &Hand,
    counts: &mut [u8; TILE_KINDS],
    best: &mut u8,
) {
    if units == 0 {
        *best = (*best).min(target_deficit(hand.counts(), counts));
        return;
    }
    if position == kinds.len() {
        return;
    }

    let kind = kinds[position];
    let index = kind.index() as usize;
    for copies in [0u8, 2, 4] {
        if copies > cap[index] || copies as usize / 2 > units {
            continue;
        }
        counts[index] = copies;
        seven_pairs_search(
            position + 1,
            units - copies as usize / 2,
            kinds,
            cap,
            hand,
            counts,
            best,
        );
    }
    counts[index] = 0;
}

fn oracle(hand: &Hand, melds: &[Meld], missing: Suit) -> u8 {
    assert!(melds.len() <= 4);
    let cap = capacity_by_kind(melds, missing);
    let target_sets = 4 - melds.len();
    let mut best = INF;

    let mut sets = Vec::new();
    for kind in Tile::ALL {
        if kind.suit() == missing {
            continue;
        }
        let mut triplet = [0u8; TILE_KINDS];
        triplet[kind.index() as usize] = 3;
        sets.push(triplet);
    }
    for suit in Suit::ALL {
        if suit == missing {
            continue;
        }
        for rank in 1..=7 {
            let mut sequence = [0u8; TILE_KINDS];
            for offset in 0..3 {
                sequence[tile(suit, rank + offset).index() as usize] = 1;
            }
            sets.push(sequence);
        }
    }

    let mut pairs = Vec::new();
    for kind in Tile::ALL {
        if kind.suit() == missing || cap[kind.index() as usize] < 2 {
            continue;
        }
        let mut pair = [0u8; TILE_KINDS];
        pair[kind.index() as usize] = 2;
        pairs.push(pair);
    }

    let mut counts = [0u8; TILE_KINDS];
    standard_search(
        0,
        target_sets,
        &sets,
        &pairs,
        &cap,
        hand,
        &mut counts,
        &mut best,
    );

    if melds.is_empty() {
        let kinds: Vec<Tile> = Tile::ALL
            .into_iter()
            .filter(|kind| kind.suit() != missing)
            .collect();
        let mut counts = [0u8; TILE_KINDS];
        seven_pairs_search(0, 7, &kinds, &cap, hand, &mut counts, &mut best);
    }

    best
}

fn assert_matches_oracle(hand: &Hand, melds: &[Meld], missing: Suit) {
    let expected = oracle(hand, melds, missing);
    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid meld count");
    assert_eq!(
        evaluator.deficit(hand),
        expected,
        "hand {hand:?}, melds {melds:?}, missing {missing}"
    );
}

#[test]
fn standard_targets_match_the_independent_oracle() {
    let missing = Suit::Tiao;

    // Tenpai: 1122334567899 waits on 3W, 6W, 9W.
    assert_matches_oracle(
        &hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 6),
            tile(Suit::Wan, 7),
            tile(Suit::Wan, 8),
            tile(Suit::Wan, 9),
            tile(Suit::Wan, 9),
        ]),
        &[],
        missing,
    );

    // Scattered hand with a larger exact distance.
    assert_matches_oracle(
        &hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 7),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 8),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 6),
            tile(Suit::Wan, 9),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 7),
            tile(Suit::Tong, 9),
        ]),
        &[],
        missing,
    );

    // Open hand: one declared peng, ten concealed tiles.
    assert_matches_oracle(
        &hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 6),
            tile(Suit::Wan, 7),
            tile(Suit::Wan, 8),
            tile(Suit::Wan, 9),
            tile(Suit::Wan, 9),
        ]),
        &[Meld::peng(tile(Suit::Tong, 5), Seat::DEALER)],
        missing,
    );

    // Two declared melds, seven concealed tiles.
    assert_matches_oracle(
        &hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 6),
            tile(Suit::Wan, 9),
        ]),
        &[
            Meld::peng(tile(Suit::Tong, 5), Seat::DEALER),
            Meld::peng(tile(Suit::Tong, 6), Seat::DEALER),
        ],
        missing,
    );
}

#[test]
fn seven_pairs_targets_match_the_oracle_and_quads_count_twice() {
    let missing = Suit::Tiao;

    // Six pairs plus a singleton: tenpai on 6T.
    let six_pairs = suit_hand(&[1, 1], &[2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7], &[]);
    let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
    assert_eq!(evaluator.deficit(&six_pairs), 1);
    assert_matches_oracle(&six_pairs, &[], missing);

    // A quad supplies two pair units: 1111 + four pairs + singleton.
    let quad = suit_hand(&[1, 1, 1, 1], &[2, 2, 3, 3, 4, 4, 5, 5, 6], &[]);
    let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
    assert_eq!(evaluator.deficit(&quad), 1);
    assert_matches_oracle(&quad, &[], missing);

    // Complete seven pairs has zero deficit.
    let complete = suit_hand(&[1, 1], &[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], &[]);
    let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
    assert_eq!(evaluator.deficit(&complete), 0);
    assert_eq!(
        eval::win_shape(&complete, 0),
        Some(crate::eval::WinShape::SevenPairs),
        "shape is otherwise a standard win, so the policy must prefer seven pairs"
    );

    // Any declared meld disables the seven-pairs route.
    let melds = [Meld::peng(tile(Suit::Tong, 9), Seat::DEALER)];
    let open_hand = suit_hand(&[1, 1], &[2, 2, 3, 3, 4, 4, 5, 5], &[]);
    assert_matches_oracle(&open_hand, &melds, missing);
}

#[test]
fn own_meld_copies_restrict_target_capacity() {
    let missing = Suit::Tiao;
    // Without the kong this hand wins on 3W; the kong owns all four 3W copies.
    let concealed = suit_hand(&[1, 2, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let melds = [Meld::concealed_kong(tile(Suit::Wan, 3))];
    let mut evaluator = DistanceEvaluator::new(&melds, missing).unwrap();
    assert!(
        evaluator.deficit(&concealed) > 1,
        "the 3W wait must not survive a own-meld four-copy exhaustion"
    );
    assert!(!eval::waiting_tiles(&concealed, &melds).contains(&tile(Suit::Wan, 3)));
    assert_matches_oracle(&concealed, &melds, missing);
}

#[test]
fn zero_shanten_agrees_with_structural_waits() {
    let missing = Suit::Tiao;
    for seed in 0..32u64 {
        let sampled = sample_hand(seed, 13, missing);
        let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
        let deficit = evaluator.deficit(&sampled);
        let waits: Vec<Tile> = eval::waiting_tiles(&sampled, &[])
            .into_iter()
            .filter(|tile| tile.suit() != missing)
            .collect();
        assert!(
            deficit >= 1,
            "a 13-tile hand cannot have a negative shanten"
        );
        assert_eq!(
            deficit == 1,
            !waits.is_empty(),
            "hand {sampled:?} deficit {deficit}, waits {waits:?}"
        );
    }
}

#[test]
fn publicly_exhausted_waits_keep_tenpai_with_zero_ukeire() {
    let missing = Suit::Tiao;
    let ready = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
    let deficit = evaluator.deficit(&ready);
    assert_eq!(deficit, 1);

    let mut public = [0u8; TILE_KINDS];
    let waits = [tile(Suit::Wan, 3), tile(Suit::Wan, 6), tile(Suit::Wan, 9)];
    for wait in waits {
        public[wait.index() as usize] = COPIES_PER_TILE - ready.count(wait);
    }

    assert_eq!(
        ukeire(&mut evaluator, &ready, &public, deficit),
        0,
        "exhausted waits keep tenpai but contribute no unseen copies"
    );
}

#[test]
fn ukeire_sums_unseen_improving_copies() {
    let missing = Suit::Tiao;
    let ready = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let mut evaluator = DistanceEvaluator::new(&[], missing).unwrap();
    let deficit = evaluator.deficit(&ready);

    // Full availability: 3W (2 unseen) + 6W (3) + 9W (2).
    assert_eq!(
        ukeire(&mut evaluator, &ready, &[0u8; TILE_KINDS], deficit),
        7
    );

    // One public 6W leaves two unseen copies.
    let mut public = [0u8; TILE_KINDS];
    public[tile(Suit::Wan, 6).index() as usize] = 1;
    assert_eq!(ukeire(&mut evaluator, &ready, &public, deficit), 6);
}

fn sample_hand(seed: u64, len: usize, missing: Suit) -> Hand {
    let allowed: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|tile| tile.suit() != missing)
        .collect();
    let mut rng = Lcg::raw(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut hand = Hand::new();
    while hand.total() < len {
        let tile = allowed[rng.below(allowed.len())];
        let _ = hand.add(tile);
    }
    hand
}

#[test]
fn draw_normalization_matches_exhaustive_discard_reference() {
    let missing = Suit::Tiao;

    // Small post-discard hands keep the exhaustive discard reference cheap.
    // m = 3 leaves four concealed tiles; m = 4 leaves one.
    let open_melds = [
        Meld::peng(tile(Suit::Tong, 5), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 6), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 7), Seat::DEALER),
    ];
    let one_tile_melds = [
        Meld::peng(tile(Suit::Tong, 5), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 6), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 7), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 8), Seat::DEALER),
    ];
    let two_melds = [
        Meld::peng(tile(Suit::Tong, 5), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 6), Seat::DEALER),
    ];

    let hands = [
        (
            hand(&[
                tile(Suit::Wan, 1),
                tile(Suit::Wan, 4),
                tile(Suit::Wan, 7),
                tile(Suit::Tong, 2),
                tile(Suit::Tong, 5),
                tile(Suit::Tong, 8),
                tile(Suit::Wan, 9),
            ]),
            two_melds.as_slice(),
        ),
        (
            hand(&[
                tile(Suit::Wan, 1),
                tile(Suit::Wan, 1),
                tile(Suit::Wan, 2),
                tile(Suit::Wan, 2),
            ]),
            open_melds.as_slice(),
        ),
        (
            hand(&[
                tile(Suit::Wan, 1),
                tile(Suit::Wan, 2),
                tile(Suit::Wan, 3),
                tile(Suit::Wan, 9),
            ]),
            open_melds.as_slice(),
        ),
        (
            hand(&[
                tile(Suit::Tong, 1),
                tile(Suit::Tong, 1),
                tile(Suit::Tong, 1),
                tile(Suit::Wan, 9),
            ]),
            open_melds.as_slice(),
        ),
        (hand(&[tile(Suit::Wan, 9)]), one_tile_melds.as_slice()),
        (hand(&[tile(Suit::Wan, 1)]), one_tile_melds.as_slice()),
    ];

    let mut normalized_non_winning_cases = 0usize;
    for (concealed, melds) in hands {
        let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid melds");
        for draw in Tile::ALL {
            if draw.suit() == missing {
                continue;
            }
            let mut drawn = concealed;
            if drawn.add(draw).is_err() {
                continue;
            }

            let drawn_deficit = oracle(&drawn, melds, missing);
            assert_eq!(evaluator.deficit(&drawn), drawn_deficit);

            if crate::rules::can_hu(&drawn, melds, missing) {
                assert_eq!(drawn_deficit, 0, "a drawn win has zero target deficit");
                continue;
            }

            normalized_non_winning_cases += 1;
            // Best legal subsequent discard. With the missing suit cleared,
            // every concealed tile is legal to discard here.
            let mut best_after_discard = INF;
            for discard in drawn.distinct_tiles() {
                let mut after = drawn;
                after.remove(discard).expect("drawn tile is held");
                let after_deficit = oracle(&after, melds, missing);
                assert_eq!(evaluator.deficit(&after), after_deficit);
                best_after_discard = best_after_discard.min(after_deficit);
            }
            assert_eq!(
                best_after_discard, drawn_deficit,
                "hand {concealed:?}, drawn {draw}, melds {melds:?}"
            );
        }
    }
    assert!(
        normalized_non_winning_cases > 0,
        "the test must exercise the non-winning draw normalization"
    );
}

#[test]
fn cache_fill_order_does_not_change_results() {
    let missing = Suit::Tiao;
    let hands = [
        hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 6),
            tile(Suit::Wan, 7),
            tile(Suit::Wan, 8),
            tile(Suit::Wan, 9),
            tile(Suit::Wan, 9),
        ]),
        hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 7),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 7),
            tile(Suit::Wan, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Wan, 5),
            tile(Suit::Tong, 5),
            tile(Suit::Wan, 8),
            tile(Suit::Tong, 8),
            tile(Suit::Wan, 9),
        ]),
        hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Tong, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Tong, 3),
            tile(Suit::Tong, 3),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 5),
            tile(Suit::Tong, 5),
            tile(Suit::Tong, 6),
            tile(Suit::Tong, 7),
            tile(Suit::Tong, 7),
        ]),
    ];

    let mut cold = DistanceEvaluator::new(&[], missing).expect("valid melds");
    let cold_distances: Vec<u8> = hands.iter().map(|hand| cold.deficit(hand)).collect();
    let cold_ukeire: Vec<u32> = hands
        .iter()
        .zip(cold_distances.iter())
        .map(|(hand, &deficit)| ukeire(&mut cold, hand, &[0u8; TILE_KINDS], deficit))
        .collect();

    // Populate the cache in reverse order, then recompute the original order.
    let mut warm = DistanceEvaluator::new(&[], missing).expect("valid melds");
    for hand in hands.iter().rev() {
        warm.deficit(hand);
    }
    for hand in hands.iter() {
        warm.deficit(hand);
    }
    let warm_distances: Vec<u8> = hands.iter().map(|hand| warm.deficit(hand)).collect();
    let warm_ukeire: Vec<u32> = hands
        .iter()
        .zip(warm_distances.iter())
        .map(|(hand, &deficit)| ukeire(&mut warm, hand, &[0u8; TILE_KINDS], deficit))
        .collect();

    assert_eq!(cold_distances, warm_distances);
    assert_eq!(cold_ukeire, warm_ukeire);

    // A fresh evaluator per hand must agree with both filling orders and with
    // repeated queries after hypothetical draws have populated the caches.
    for ((hand, &distance), &available) in hands
        .iter()
        .zip(cold_distances.iter())
        .zip(cold_ukeire.iter())
    {
        let mut fresh = DistanceEvaluator::new(&[], missing).expect("valid melds");
        let fresh_distance = fresh.deficit(hand);
        assert_eq!(fresh_distance, distance);
        assert_eq!(
            ukeire(&mut fresh, hand, &[0u8; TILE_KINDS], fresh_distance),
            available
        );
        assert_eq!(cold.deficit(hand), distance);
        assert_eq!(
            ukeire(&mut cold, hand, &[0u8; TILE_KINDS], distance),
            available
        );
    }
}

#[test]
fn known_open_waits_and_own_meld_caps() {
    let missing = Suit::Tiao;

    // Two declared melds leave 1234567W, which waits on 7W for two sets + pair.
    let two_melds = [
        Meld::peng(tile(Suit::Tong, 5), Seat::DEALER),
        Meld::peng(tile(Suit::Tong, 6), Seat::DEALER),
    ];
    let open = suit_hand(&[1, 2, 3, 4, 5, 6, 7], &[], &[]);
    let mut evaluator = DistanceEvaluator::new(&two_melds, missing).expect("valid melds");
    assert_eq!(evaluator.deficit(&open), 1);
    let waits: Vec<Tile> = eval::waiting_tiles(&open, &two_melds)
        .into_iter()
        .filter(|tile| tile.suit() != missing)
        .collect();
    assert!(waits.contains(&tile(Suit::Wan, 7)));

    // One declared meld: the 10-tile concealed part waits on the pair tile.
    let one_meld = [Meld::peng(tile(Suit::Tong, 5), Seat::DEALER)];
    let one_meld_hand = suit_hand(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let mut evaluator = DistanceEvaluator::new(&one_meld, missing).expect("valid melds");
    assert_eq!(evaluator.deficit(&one_meld_hand), 1);
    let waits: Vec<Tile> = eval::waiting_tiles(&one_meld_hand, &one_meld)
        .into_iter()
        .filter(|tile| tile.suit() != missing)
        .collect();
    assert!(waits.contains(&tile(Suit::Wan, 9)));

    // A concealed kong owns all four 3W copies, so 3W cannot be a target.
    let kong = [Meld::concealed_kong(tile(Suit::Wan, 3))];
    let capped = suit_hand(&[1, 2, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let mut evaluator = DistanceEvaluator::new(&kong, missing).expect("valid melds");
    assert!(evaluator.deficit(&capped) > 1);
    assert!(!eval::waiting_tiles(&capped, &kong).contains(&tile(Suit::Wan, 3)));
}
