//! Physical-fixture evidence: boundary and sampled positions plus the
//! supported analysis domain.

use super::super::test_fixtures::{
    assert_physical, boundary_states, capacity_by_kind, meld_copies, sampled_states, tile, State,
};
use super::super::DistanceEvaluator;
use super::oracle::{reference_seven_pairs_mask, reference_standard_mask};
use super::{bit, check_state, CheckDepth};
use crate::action::Meld;
use crate::seat::Seat;
use crate::tile::{Suit, Tile, COPIES_PER_TILE};

/// Literal expectations for the hand-picked boundary fixtures.
fn assert_boundary_literals(state: &State, deficit: u8, mask: u32) {
    let wan = Suit::Wan;
    let tong = Suit::Tong;
    match state.label {
        "standard-tenpai" => {
            assert_eq!(deficit, 1);
            assert_eq!(
                mask,
                bit(tile(wan, 3)) | bit(tile(wan, 6)) | bit(tile(wan, 9))
            );
        }
        "scattered" => {
            assert_eq!(deficit, 2);
            assert_eq!(
                mask,
                bit(tile(tong, 1)) | bit(tile(tong, 4)) | bit(tile(tong, 8))
            );
        }
        "route-tie-overlap" | "route-tie-disjoint" => {
            let standard = reference_standard_mask(&state.hand, &state.melds, state.missing);
            let seven = reference_seven_pairs_mask(state.hand.counts(), state.missing);
            assert_eq!(
                standard.0, seven.0,
                "{}: the fixture must tie the two routes",
                state.label
            );
            assert_ne!(
                standard.1, seven.1,
                "{}: the fixture must need a route-mask union",
                state.label
            );
            assert_eq!(mask, standard.1 | seven.1, "{}", state.label);
            if state.label == "route-tie-overlap" {
                assert_eq!(deficit, 1);
                assert_eq!(
                    mask,
                    bit(tile(wan, 3)) | bit(tile(wan, 6)),
                    "seven pairs waits on 6W, standard also on 3W"
                );
            } else {
                assert_eq!(deficit, 1);
                assert_eq!(
                    mask,
                    bit(tile(wan, 7)) | bit(tile(wan, 8)),
                    "seven pairs waits on 8W, standard on 7W"
                );
            }
        }
        "seven-pairs-internal-tie" => {
            let seven = reference_seven_pairs_mask(state.hand.counts(), state.missing);
            let standard = reference_standard_mask(&state.hand, &state.melds, state.missing);
            assert_eq!(deficit, 2);
            assert_eq!(seven.0, 2, "seven pairs is the optimal route");
            assert_eq!(mask, seven.1);
            assert_eq!(
                mask,
                bit(tile(tong, 1)) | bit(tile(tong, 4)) | bit(tile(tong, 7)),
                "every odd kind supplies one pair unit"
            );
            assert!(
                standard.0 > seven.0,
                "the standard route must be strictly worse for this fixture"
            );
        }
        "seven-pairs-triple" => {
            let seven = reference_seven_pairs_mask(state.hand.counts(), state.missing);
            let standard = reference_standard_mask(&state.hand, &state.melds, state.missing);
            assert_eq!(deficit, 1);
            assert_eq!(mask, bit(tile(wan, 1)), "only the quadded kind completes");
            assert_eq!(mask, seven.1);
            assert!(standard.0 > seven.0);
        }
        "uncleared-missing-tile" => {
            assert_eq!(deficit, 2);
            assert_eq!(mask, (1u32 << 18) - 1, "all non-missing kinds help");
            for rank in 1..=9 {
                assert_eq!(mask & bit(tile(Suit::Tiao, rank)), 0);
            }
        }
        "one-peng" => {
            assert_eq!(deficit, 1);
            assert_eq!(
                mask,
                bit(tile(wan, 3)) | bit(tile(wan, 6)) | bit(tile(wan, 9))
            );
        }
        "peng-capacity" => {
            assert_eq!(
                (deficit, mask),
                (1, bit(tile(wan, 3)) | bit(tile(wan, 6)) | bit(tile(wan, 9)))
            );
            assert_eq!(
                mask & bit(tile(wan, 5)),
                0,
                "the own peng leaves one concealed copy, already held"
            );
            assert_eq!(
                capacity_by_kind(&state.melds, state.missing)[tile(wan, 5).index() as usize],
                1
            );
        }
        "concealed-kong-capacity" => {
            assert_eq!(deficit, 2);
            assert_eq!(
                mask,
                bit(tile(wan, 1)) | bit(tile(wan, 2)) | bit(tile(wan, 6)) | bit(tile(wan, 9))
            );
            assert_eq!(
                mask & bit(tile(wan, 3)),
                0,
                "the own kong exhausts every 3W target copy"
            );
            assert_eq!(
                capacity_by_kind(&state.melds, state.missing)[tile(wan, 3).index() as usize],
                0
            );
        }
        "added-kong-capacity" => {
            assert_eq!(
                mask & bit(tile(wan, 8)),
                0,
                "the own added kong exhausts every 8W target copy"
            );
        }
        "two-peng" => {
            assert_eq!(deficit, 1);
            assert_eq!(mask, bit(tile(wan, 9)));
        }
        "three-meld" => {
            assert_eq!(deficit, 1);
            assert_eq!(mask, bit(tile(wan, 1)) | bit(tile(wan, 2)));
        }
        "four-meld" => {
            assert_eq!(deficit, 1);
            assert_eq!(mask, bit(tile(wan, 3)));
        }
        "four-meld-missing-wan" => {
            assert_eq!(deficit, 1);
            assert_eq!(
                mask,
                bit(tile(tong, 3)),
                "the four melds leave only the pair wait"
            );
        }
        other => panic!("fixture {other} lacks a literal expectation"),
    }
}

#[test]
fn boundary_states_match_both_references() {
    let fixtures = boundary_states();
    assert_eq!(fixtures.len(), 15, "fixture list size changed");
    let mut meld_counts = [0usize; 5];
    let mut missing_counts = [0usize; 3];
    for state in &fixtures {
        let (deficit, mask) = check_state(state, CheckDepth::Full);
        assert_boundary_literals(state, deficit, mask);
        meld_counts[state.melds.len()] += 1;
        missing_counts[state.missing.index()] += 1;
    }
    for (meld_count, count) in meld_counts.iter().enumerate() {
        assert!(*count > 0, "no fixture with {meld_count} melds");
    }
    assert!(
        missing_counts.iter().filter(|&&count| count > 0).count() >= 2,
        "fixtures must cover more than one missing suit"
    );
}

#[test]
fn sampled_physical_states_match_both_references() {
    let samples = sampled_states();
    assert_eq!(samples.len(), 75, "sample sweep size changed");
    let mut meld_counts = [0usize; 5];
    let mut missing_counts = [0usize; 3];
    let mut nonzero_masks = 0usize;
    let mut distance_two_plus = 0usize;
    let mut capped_kinds = 0usize;
    for state in &samples {
        let (deficit, mask) = check_state(state, CheckDepth::Sampled);
        assert!(deficit >= 1, "a post-discard hand cannot have zero deficit");
        if mask != 0 {
            nonzero_masks += 1;
        }
        if deficit >= 2 {
            distance_two_plus += 1;
        }
        let capacity = capacity_by_kind(&state.melds, state.missing);
        for kind in Tile::ALL {
            if state.hand.count(kind) > 0
                && state.hand.count(kind) == capacity[kind.index() as usize]
            {
                capped_kinds += 1;
            }
        }
        meld_counts[state.melds.len()] += 1;
        missing_counts[state.missing.index()] += 1;
    }
    for (meld_count, count) in meld_counts.iter().enumerate() {
        assert_eq!(
            *count, 15,
            "meld count {meld_count} needs 15 sampled states"
        );
    }
    for (missing, count) in missing_counts.iter().enumerate() {
        assert_eq!(*count, 25, "missing suit {missing} needs 25 sampled states");
    }
    assert!(nonzero_masks > 0, "the sweep must include improving kinds");
    assert!(
        distance_two_plus > 0,
        "the sweep must include non-tenpai distances"
    );
    assert!(
        capped_kinds > 0,
        "the sweep must include kinds already at their target capacity"
    );
}

#[test]
fn declared_sizes_and_capacities_keep_the_supported_domain() {
    assert!(DistanceEvaluator::new(&[], Suit::Wan).is_some());
    let tiles = [
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
    ];
    for meld_count in 1..=5 {
        let melds: Vec<Meld> = tiles[..meld_count]
            .iter()
            .map(|&kind| Meld::peng(kind, Seat::DEALER))
            .collect();
        assert_eq!(
            DistanceEvaluator::new(&melds, Suit::Tiao).is_some(),
            meld_count <= 4,
            "the legal meld-count domain is 0..=4"
        );
    }

    for state in boundary_states() {
        assert_physical(&state);
        let capacity = capacity_by_kind(&state.melds, state.missing);
        for rank in 1..=9 {
            assert_eq!(
                capacity[tile(state.missing, rank).index() as usize],
                0,
                "{}: the missing suit has no target capacity",
                state.label
            );
        }
        let copies = meld_copies(&state.melds);
        for kind in Tile::ALL {
            let index = kind.index() as usize;
            assert!(state.public[index] <= COPIES_PER_TILE);
            assert!(copies[index] + state.hand.count(kind) <= COPIES_PER_TILE);
        }
    }
}
