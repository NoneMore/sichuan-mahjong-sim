//! Seven-pairs route evidence: concrete-target DFS fixtures and hand-built
//! seven-pairs wins.

use super::super::test_fixtures::{hand_from_counts, tile};
use super::super::DistanceEvaluator;
use super::bit;
use super::super::test_support::targets::merge_route;
use super::oracle::{
    reference_hand_mask, reference_seven_pairs_mask, reference_seven_pairs_mask_direct,
    reference_standard_mask,
};
use crate::rules;
use crate::tile::{Suit, Tile, TILE_KINDS};

#[test]
fn seven_pairs_route_covers_zeros_triples_quads_and_missing() {
    // Bounded direct-DFS fixtures. Each hand's seven-pairs route enumerates all
    // 241,128 concrete 0/2/4-copy targets over the 18 non-missing kinds; this
    // is a concrete-target oracle, not the production closed form or the
    // per-kind DP used for the randomized sweep.
    let cases: [(&str, Suit, [u8; TILE_KINDS]); 7] = [
        // Uncleared at the production size: P=0, O=6, K=7, so O<K.
        // Zero-held kinds must also appear in the complete optimal union.
        (
            "uncleared-cost-two",
            Suit::Tiao,
            grouped(&[
                (Suit::Wan, &[1, 2, 3, 4, 5, 6]),
                (Suit::Tiao, &[1, 1, 1, 1, 2, 2, 2]),
            ]),
        ),
        // Complete seven pairs: both routes are complete and no bit is needed.
        (
            "complete",
            Suit::Tiao,
            grouped(&[(Suit::Wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])]),
        ),
        // Six pairs plus a singleton (13 tiles): one pair unit is missing and
        // the standard route ties with a different additional wait.
        (
            "six-pairs",
            Suit::Tiao,
            grouped(&[(Suit::Wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7])]),
        ),
        // A quad is two pair units; only the singleton pair is missing and the
        // standard route ties with several extra waits.
        (
            "quad",
            Suit::Tiao,
            grouped(&[(Suit::Wan, &[1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])]),
        ),
        // A held triple is one unit plus one odd kind; only 1W completes.
        (
            "triple",
            Suit::Tong,
            grouped(&[(Suit::Wan, &[1, 1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 9, 9])]),
        ),
        // Unheld kinds (Wan ranks 6..=9 are zero here) still leave a
        // three-way pair-unit tie among the Tong odd kinds.
        (
            "zero-kinds",
            Suit::Tiao,
            grouped(&[
                (Suit::Wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
                (Suit::Tong, &[1, 4, 7]),
            ]),
        ),
        // Five pairs, three odd kinds under a missing Wan suit: three
        // alternative pair units, and the whole pairing must ignore Wan.
        (
            "missing-held",
            Suit::Wan,
            grouped(&[
                (Suit::Tong, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
                (Suit::Tiao, &[1, 4, 7]),
            ]),
        ),
    ];

    for (label, missing, counts) in cases {
        let hand = hand_from_counts(&counts);
        let mut evaluator = DistanceEvaluator::new(&[], missing).expect("no melds");
        let (deficit, mask) = evaluator.deficit_with_mask(&hand);
        let reference = reference_hand_mask(&hand, &[], missing);
        assert_eq!((deficit, mask), reference, "{label}: definition reference");
        assert_eq!(mask >> TILE_KINDS, 0, "{label}: mask must stay in 27 bits");

        // Concrete seven-pairs DFS: every complete target is scored without
        // merging partial assignments, and the exact visit count is pinned.
        let (seven_direct, visited) = reference_seven_pairs_mask_direct(&counts, missing);
        assert_eq!(
            visited, 241_128,
            "{label}: every concrete seven-pairs target must be visited"
        );
        let seven_dp = reference_seven_pairs_mask(&counts, missing);
        assert_eq!(
            seven_direct, seven_dp,
            "{label}: concrete seven-pairs DFS and per-kind DP disagree"
        );
        let standard = reference_standard_mask(&hand, &[], missing);
        assert_eq!(
            reference,
            merge_route(standard, seven_direct),
            "{label}: route merge must equal the concrete-route union"
        );

        match label {
            "uncleared-cost-two" => {
                let non_missing_mask = (1u32 << 18) - 1;
                assert_eq!(hand.total(), 13);
                assert_eq!(seven_direct, (8, non_missing_mask));
                assert_eq!((deficit, mask), (8, non_missing_mask));
                assert!(!rules::can_hu(&hand, &[], missing));
            }
            "complete" => {
                assert_eq!(deficit, 0);
                assert_eq!(mask, 0);
                assert_eq!(seven_direct, (0, 0));
                assert_eq!(standard, (0, 0));
                assert!(rules::can_hu(&hand, &[], missing));
            }
            "six-pairs" => {
                assert_eq!(
                    (deficit, mask),
                    (1, bit(tile(Suit::Wan, 3)) | bit(tile(Suit::Wan, 6))),
                    "the standard route adds the 3W wait to the seven-pairs 6W wait"
                );
                assert_eq!(seven_direct, (1, bit(tile(Suit::Wan, 6))));
                assert_eq!(standard.0, seven_direct.0, "the routes must tie");
            }
            "quad" => {
                assert_eq!(
                    (deficit, mask),
                    (
                        1,
                        bit(tile(Suit::Wan, 2))
                            | bit(tile(Suit::Wan, 3))
                            | bit(tile(Suit::Wan, 5))
                            | bit(tile(Suit::Wan, 6))
                    ),
                    "the standard route adds four waits to the seven-pairs 6W wait"
                );
                assert_eq!(seven_direct, (1, bit(tile(Suit::Wan, 6))));
                assert_eq!(standard.0, seven_direct.0, "the routes must tie");
            }
            "triple" => {
                assert_eq!((deficit, mask), (1, bit(tile(Suit::Wan, 1))));
                assert_eq!(seven_direct, (1, bit(tile(Suit::Wan, 1))));
                assert!(
                    standard.0 > seven_direct.0,
                    "the standard route must be strictly worse"
                );
            }
            "zero-kinds" => {
                assert_eq!(
                    (deficit, mask),
                    (
                        2,
                        bit(tile(Suit::Tong, 1))
                            | bit(tile(Suit::Tong, 4))
                            | bit(tile(Suit::Tong, 7))
                    )
                );
                assert_eq!(seven_direct.0, 2);
                assert!(
                    standard.0 > seven_direct.0,
                    "the standard route must be strictly worse"
                );
                for rank in [6u8, 7, 8, 9] {
                    assert_eq!(hand.count(tile(Suit::Wan, rank)), 0);
                }
            }
            "missing-held" => {
                assert_eq!(
                    (deficit, mask),
                    (
                        2,
                        bit(tile(Suit::Tiao, 1))
                            | bit(tile(Suit::Tiao, 4))
                            | bit(tile(Suit::Tiao, 7))
                    )
                );
                assert_eq!(seven_direct.0, 2);
                assert!(
                    standard.0 > seven_direct.0,
                    "the standard route must be strictly worse"
                );
                for kind in Tile::ALL
                    .into_iter()
                    .filter(|kind| kind.suit() == Suit::Wan)
                {
                    assert_eq!(mask & bit(kind), 0, "missing-suit bit {kind}");
                }
            }
            other => panic!("unexpected seven-pairs case {other}"),
        }
    }
}

fn grouped(groups: &[(Suit, &[u8])]) -> [u8; TILE_KINDS] {
    let mut counts = [0u8; TILE_KINDS];
    for &(suit, ranks) in groups {
        for &rank in ranks {
            counts[tile(suit, rank).index() as usize] += 1;
        }
    }
    counts
}

#[test]
fn hand_built_seven_pairs_wins_keep_their_distance() {
    // A 14-tile seven-pairs hand has zero deficit. A 16-tile hand with eight
    // pairs also has zero deficit because surplus copies are free, but it is
    // not a winning hand: the rules require exactly 14 concealed tiles when the
    // meld count is zero. The two facts are checked separately rather than
    // relaxing either rule to fit the other.
    let complete = hand_from_counts(&grouped(&[(
        Suit::Wan,
        &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    )]));
    let extra_pairs = hand_from_counts(&grouped(&[(
        Suit::Wan,
        &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    )]));
    for hand in [complete, extra_pairs] {
        let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
        let (deficit, mask) = evaluator.deficit_with_mask(&hand);
        assert_eq!(
            (deficit, mask),
            (0, 0),
            "no draw can improve a zero deficit"
        );
        assert_eq!(
            reference_hand_mask(&hand, &[], Suit::Tiao),
            (deficit, mask),
            "reference must agree for the seven-pairs hand"
        );
    }
    assert!(rules::can_hu(&complete, &[], Suit::Tiao));
    assert!(
        !rules::can_hu(&extra_pairs, &[], Suit::Tiao),
        "can_hu requires the fixed 14-tile size; a zero deficit on a 16-tile \
         hand does not make it a win"
    );
}
