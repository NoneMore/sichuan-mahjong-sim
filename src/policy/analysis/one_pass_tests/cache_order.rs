//! Cache-lifetime evidence: plain and masked caches stay separate, and a
//! single evaluator stays context-independent across query reordering.

use super::super::test_fixtures::{boundary_states, meld_copies, State};
use super::super::{one_pass_reference as baseline, DistanceEvaluator};
use super::oracle::reference_hand_mask;
use crate::rules;
use crate::tile::{Suit, Tile, COPIES_PER_TILE};

#[test]
fn mask_cache_is_separate_from_distance_cache() {
    for state in boundary_states().iter().take(8) {
        let expected = reference_hand_mask(&state.hand, &state.melds, state.missing);

        // Plain distance first: the hand cache hit must not answer the mask.
        let mut distance_first =
            DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        assert_eq!(distance_first.deficit(&state.hand), expected.0);
        assert_eq!(
            distance_first.deficit_with_mask(&state.hand),
            expected,
            "{}: a plain-distance cache hit must not suppress the mask",
            state.label
        );

        // Mask first: the plain query must still be exact.
        let mut mask_first = DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        assert_eq!(mask_first.deficit_with_mask(&state.hand), expected);
        assert_eq!(mask_first.deficit(&state.hand), expected.0);

        // Populate the plain hand cache with every legal drawn hand first, then
        // query the mask for the original and for a winning drawn hand.
        let mut interleaved = DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        let mut frozen =
            baseline::DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        let mut winning = None;
        let copies = meld_copies(&state.melds);
        for kind in Tile::ALL {
            let index = kind.index() as usize;
            if kind.suit() == state.missing
                || state.hand.count(kind) + copies[index] >= COPIES_PER_TILE
            {
                continue;
            }
            let mut drawn = state.hand;
            drawn.add(kind).expect("a physically available draw");
            assert_eq!(
                interleaved.deficit(&drawn),
                frozen.deficit(&drawn),
                "{}: drawn distance mismatch",
                state.label
            );
            if rules::can_hu(&drawn, &state.melds, state.missing) {
                winning = Some(drawn);
            }
        }
        assert_eq!(
            interleaved.deficit_with_mask(&state.hand),
            expected,
            "{}: a target-size distance entry must not answer the mask query",
            state.label
        );
        if let Some(drawn) = winning {
            assert_eq!(interleaved.deficit(&drawn), 0);
            assert_eq!(
                interleaved.deficit_with_mask(&drawn),
                (0, 0),
                "{}: a winning hand's optimal target is itself",
                state.label
            );
        }
    }
}

#[test]
fn mask_queries_are_order_and_evaluator_independent() {
    let fixtures: Vec<State> = boundary_states()
        .into_iter()
        .filter(|state| state.melds.is_empty() && state.missing == Suit::Tiao)
        .collect();
    assert!(fixtures.len() >= 4, "need several same-context hands");

    let expected: Vec<(u8, u32)> = fixtures
        .iter()
        .map(|state| reference_hand_mask(&state.hand, &state.melds, state.missing))
        .collect();

    // Fresh evaluator per hand, mask first.
    for (state, &reference) in fixtures.iter().zip(expected.iter()) {
        let mut evaluator = DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        assert_eq!(evaluator.deficit_with_mask(&state.hand), reference);
    }

    // One reused evaluator in forward order, then reversed, then repeated.
    let mut reused = DistanceEvaluator::new(&[], Suit::Tiao).expect("melds");
    for (state, &reference) in fixtures.iter().zip(expected.iter()) {
        assert_eq!(reused.deficit_with_mask(&state.hand), reference);
    }
    for (state, &reference) in fixtures.iter().zip(expected.iter()).rev() {
        assert_eq!(reused.deficit_with_mask(&state.hand), reference);
        assert_eq!(reused.deficit(&state.hand), reference.0);
    }
    for (state, &reference) in fixtures.iter().zip(expected.iter()) {
        assert_eq!(reused.deficit_with_mask(&state.hand), reference);
    }

    // A different fixed context does not leak into another evaluator.
    let other_missing = Suit::Tong;
    let other = fixtures
        .iter()
        .find(|state| state.label == "route-tie-overlap")
        .expect("same fixture is reusable under a different missing suit");
    let other_reference = reference_hand_mask(&other.hand, &[], other_missing);
    let mut other_evaluator = DistanceEvaluator::new(&[], other_missing).expect("melds");
    assert_eq!(
        other_evaluator.deficit_with_mask(&other.hand),
        other_reference
    );
    for (state, &reference) in fixtures.iter().zip(expected.iter()) {
        assert_eq!(reused.deficit_with_mask(&state.hand), reference);
    }
}
