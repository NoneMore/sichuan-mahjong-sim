//! Independent differential evidence for the one-pass ukeire contract.
//!
//! Two independent references back every assertion:
//!
//! - [`baseline`] is the frozen `dc6ff5c` distance and per-tile `ukeire`
//!   implementation, with its own evaluator, caches, and per-tile loop. It
//!   never calls the masked analysis.
//! - [`oracle`] enumerates concrete winning targets (per-suit set multisets +
//!   pair, and per-kind seven-pairs copies) and returns the exact distance plus
//!   the union of needed kinds over all optimal targets. For two or more melds
//!   it also runs a second, global concrete-target enumeration.
//!
//! The checked properties are the spec's exact behavior contract: the complete
//! `(deficit, mask)` pair for `deficit_with_mask`, every `[sets][pair]` entry of
//! `suit_table_with_mask`, the per-tile improvement equivalence
//! `bit(t) <=> D(H+t) < D(H)`, `can_hu(H+t) <=> D(H+t) == 0` at legal sizes,
//! the `unseen` per-kind contributions and ukeire sum, publicly exhausted mask
//! bits, and fresh/reused cache interleavings. `mask_marks_needed_ranks_not_used_ranks`
//! in `suit.rs` pins that the mask encodes needed deficit, not merely kinds a
//! cheapest target happens to use, so "unseen > 0" is never the only evidence.
//!
//! Scope and honesty notes:
//!
//! - Fixtures are hand-picked boundary positions; [`states::sampled_states`]
//!   uses a documented fixed seed and count. The random sweep is statistical
//!   coverage of the combined parameter space, not exhaustive verification of
//!   every possible tile pattern.
//! - The direct global enumeration is used for `melds.len() >= 2`, where it is
//!   small; the combined concrete enumeration covers all meld counts.
//! - Existing decision-level coverage in `src/policy/heuristic/tests/calls.rs`
//!   (peng/kong gates, hypothetical meld positions, owner-consistent public
//!   adjustment) is retained and not duplicated here. This module is the new
//!   analysis-level differential evidence; it does not claim to re-measure
//!   those policy gates.
//! - Trace-level baseline/candidate policy comparison, benchmark timings, and
//!   the formal proof are separate deliverables.

mod oracle;
mod states;
mod suit;

use super::one_pass_reference as baseline;
use super::{ukeire, DistanceEvaluator};

use self::oracle::{
    merge_route, reference_hand_mask, reference_hand_mask_direct, reference_seven_pairs_mask,
    reference_seven_pairs_mask_direct, reference_standard_mask,
};
use self::states::{
    assert_physical, boundary_states, capacity_by_kind, hand_from_counts, meld_copies, own_public,
    sampled_states, tile, State,
};
use crate::action::Meld;
use crate::hand::Hand;
use crate::rules;
use crate::seat::Seat;
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

/// How much reference work a differential check performs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckDepth {
    /// Boundary fixtures: also compare reference distances for every drawn hand.
    Full,
    /// Sampled sweep: still checks drawn production distances, the direct-win
    /// equivalence, and the frozen ukeire, but skips the reference distance for
    /// each drawn hand to keep the sweep cheap.
    Sampled,
}

fn bit(kind: Tile) -> u32 {
    1u32 << kind.index()
}

fn unseen_copies(hand: &Hand, public: &[u8; TILE_KINDS], kind: Tile) -> u32 {
    COPIES_PER_TILE
        .saturating_sub(hand.count(kind))
        .saturating_sub(public[kind.index() as usize]) as u32
}

/// Spec-level ukeire: unseen copies of every masked, non-publicly-exhausted
/// improving kind. Missing-suit bits are asserted absent by the callers.
fn spec_ukeire(hand: &Hand, public: &[u8; TILE_KINDS], mask: u32) -> u32 {
    let mut total = 0u32;
    for kind in Tile::ALL {
        if mask & bit(kind) != 0 {
            total += unseen_copies(hand, public, kind);
        }
    }
    total
}

/// Differential check of one physical position against both references.
///
/// Returns the tested `(deficit, mask)` so boundary tests can pin literals.
fn check_state(state: &State, depth: CheckDepth) -> (u8, u32) {
    assert_physical(state);
    let melds = &state.melds;
    let mut evaluator = DistanceEvaluator::new(melds, state.missing).expect("at most four melds");

    let (deficit, mask) = evaluator.deficit_with_mask(&state.hand);
    let (expected_deficit, expected_mask) = reference_hand_mask(&state.hand, melds, state.missing);
    assert_eq!(
        (deficit, mask),
        (expected_deficit, expected_mask),
        "concrete-target reference mismatch for {}: hand {:?}, melds {:?}, \
         missing {}, public {:?}",
        state.label,
        state.hand,
        state.melds,
        state.missing,
        state.public
    );

    // The global enumeration is only affordable up to two target sets.
    if melds.len() >= 2 {
        let direct = reference_hand_mask_direct(&state.hand, melds, state.missing);
        assert_eq!(
            direct,
            (expected_deficit, expected_mask),
            "the per-suit and global references disagree for {}",
            state.label
        );
    }

    assert_eq!(
        mask >> TILE_KINDS,
        0,
        "{}: the mask must use only the low {} bits, got {mask:#x}",
        state.label,
        TILE_KINDS
    );
    assert_eq!(
        evaluator.deficit(&state.hand),
        deficit,
        "{}: masked and plain distances differ",
        state.label
    );
    let mut frozen =
        baseline::DistanceEvaluator::new(melds, state.missing).expect("at most four melds");
    assert_eq!(
        frozen.deficit(&state.hand),
        deficit,
        "{}: frozen dc6ff5c distance differs",
        state.label
    );

    // Structural mask preconditions: never the missing suit, never more target
    // copies than the own-meld capacity allows.
    let capacity = capacity_by_kind(melds, state.missing);
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        let marked = mask & bit(kind) != 0;
        if kind.suit() == state.missing {
            assert!(!marked, "{}: missing-suit bit {kind} is set", state.label);
        }
        if state.hand.count(kind) >= capacity[index] {
            assert!(
                !marked,
                "{}: {kind} holds {} copies with target capacity {}, so no \
                 optimal target can need more of it",
                state.label,
                state.hand.count(kind),
                capacity[index]
            );
        }
    }

    // Per-tile improvement and direct-win equivalence at the legal draw size.
    // The equivalence is only stated for physically available draws: a kind
    // already owning four copies across the hand and own melds cannot be drawn
    // in a legal position. The capacity precondition above already proves its
    // mask bit is clear, so skipping the drawn check loses no evidence.
    let target_size = 14 - 3 * melds.len();
    let copies = meld_copies(melds);
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        if state.hand.count(kind) + copies[index] >= COPIES_PER_TILE {
            // The public ownership includes every own meld copy, so the frozen
            // and production ukeire loops never attempt this illegal draw.
            assert_eq!(
                unseen_copies(&state.hand, &state.public, kind),
                0,
                "{}: a physically exhausted kind has no unseen copy",
                state.label
            );
            continue;
        }
        let mut drawn = state.hand;
        drawn.add(kind).expect("a physically available draw");
        assert_eq!(
            drawn.total(),
            target_size,
            "{}: a draw must reach the target size",
            state.label
        );
        let (drawn_deficit, drawn_mask) = evaluator.deficit_with_mask(&drawn);
        assert_eq!(
            drawn_mask >> TILE_KINDS,
            0,
            "{}: drawn mask high bits",
            state.label
        );
        if depth == CheckDepth::Full {
            let (reference_drawn, _) = reference_hand_mask(&drawn, melds, state.missing);
            assert_eq!(
                drawn_deficit, reference_drawn,
                "{}: reference distance for the drawn hand differs",
                state.label
            );
        }
        if kind.suit() == state.missing {
            assert_eq!(
                drawn_deficit, deficit,
                "{}: drawing a missing-suit tile cannot change the deficit",
                state.label
            );
        } else {
            let marked = mask & bit(kind) != 0;
            assert_eq!(
                marked,
                drawn_deficit < deficit,
                "{}: bit {kind} must mean exactly D(H+t) < D(H) \
                 (D={deficit}, D(H+t)={drawn_deficit})",
                state.label
            );
        }
        assert_eq!(
            rules::can_hu(&drawn, melds, state.missing),
            drawn_deficit == 0,
            "{}: can_hu and zero deficit must agree for the drawn hand {:?}",
            state.label,
            drawn
        );
    }

    // Per-kind unseen contributions, the frozen per-tile ukeire, and the
    // production ukeire total must agree.
    let expected_ukeire = spec_ukeire(&state.hand, &state.public, mask);
    let production_ukeire = ukeire(&mut evaluator, &state.hand, &state.public, deficit);
    assert_eq!(
        production_ukeire, expected_ukeire,
        "{}: production ukeire differs from the mask/unseen formula",
        state.label
    );
    let frozen_ukeire = baseline::ukeire(&mut frozen, &state.hand, &state.public, deficit);
    assert_eq!(
        frozen_ukeire, expected_ukeire,
        "{}: frozen per-tile ukeire differs from the mask/unseen formula",
        state.label
    );

    assert_eq!(
        evaluator.deficit_with_mask(&state.hand),
        (deficit, mask),
        "{}: a repeated masked query must be stable",
        state.label
    );
    (deficit, mask)
}

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
fn improving_mask_matches_draw_distance_and_direct_wins() {
    let mut direct_wins = 0usize;
    let mut non_winning_draws = 0usize;
    let mut improving_draws = 0usize;
    let fixtures = boundary_states();
    for state in &fixtures {
        let mut evaluator =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        let (deficit, mask) = evaluator.deficit_with_mask(&state.hand);
        let target_size = 14 - 3 * state.melds.len();
        let copies = meld_copies(&state.melds);
        for kind in Tile::ALL {
            let index = kind.index() as usize;
            // Only physically available draws are in the equivalence's domain.
            if state.hand.count(kind) + copies[index] >= COPIES_PER_TILE {
                continue;
            }
            let mut drawn = state.hand;
            drawn.add(kind).expect("a physically available draw");
            assert_eq!(drawn.total(), target_size);
            let (drawn_deficit, _) = evaluator.deficit_with_mask(&drawn);
            let can_hu = rules::can_hu(&drawn, &state.melds, state.missing);
            assert_eq!(
                can_hu,
                drawn_deficit == 0,
                "{}: can_hu must equal zero deficit for the legal drawn size, \
                 drawn hand {:?}",
                state.label,
                drawn
            );
            if can_hu {
                direct_wins += 1;
            } else {
                non_winning_draws += 1;
            }
            if kind.suit() != state.missing {
                let marked = mask & bit(kind) != 0;
                assert_eq!(
                    marked,
                    drawn_deficit < deficit,
                    "{}: bit {kind} must mean exactly D(H+t) < D(H)",
                    state.label
                );
                if marked {
                    improving_draws += 1;
                }
            }
        }
    }
    assert!(direct_wins > 0, "no directly winning draw was exercised");
    assert!(non_winning_draws > 0, "no non-winning draw was exercised");
    assert!(improving_draws > 0, "no improving draw was exercised");
}

#[test]
fn unseen_and_ukeire_follow_public_ownership() {
    let fixtures = boundary_states();
    let tenpai = fixtures
        .iter()
        .find(|state| state.label == "standard-tenpai")
        .expect("tenpai fixture");
    let mut evaluator = DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    let (deficit, mask) = evaluator.deficit_with_mask(&tenpai.hand);
    assert_eq!((deficit, mask), (1, 0x124));
    let waits = [tile(Suit::Wan, 3), tile(Suit::Wan, 6), tile(Suit::Wan, 9)];

    // Fully available: 3W has two unseen copies, 6W three, 9W two.
    let empty = own_public(tenpai);
    assert_eq!(empty, [0u8; TILE_KINDS]);
    for (kind, expected) in [(waits[0], 2), (waits[1], 3), (waits[2], 2)] {
        assert_eq!(unseen_copies(&tenpai.hand, &empty, kind), expected);
    }
    assert_eq!(spec_ukeire(&tenpai.hand, &empty, mask), 7);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &empty, deficit),
        7
    );

    // Per-kind production contributions: exhaust every other kind so the
    // production total is exactly this improving kind's unseen copies.
    for &kind in &waits {
        let mut only = empty;
        for other in Tile::ALL {
            if other != kind {
                only[other.index() as usize] = COPIES_PER_TILE - tenpai.hand.count(other);
            }
        }
        let contribution = unseen_copies(&tenpai.hand, &only, kind);
        assert!(contribution > 0, "the wait {kind} must have unseen copies");
        assert_eq!(
            ukeire(&mut evaluator, &tenpai.hand, &only, deficit),
            contribution,
            "production ukeire must count each improving kind independently"
        );
        let mut frozen =
            baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
        assert_eq!(
            baseline::ukeire(&mut frozen, &tenpai.hand, &only, deficit),
            contribution,
            "frozen ukeire must count the same per-kind contribution"
        );
    }

    // One public 6W leaves two unseen copies of that wait.
    let mut partial = empty;
    partial[waits[1].index() as usize] = 1;
    assert_eq!(unseen_copies(&tenpai.hand, &partial, waits[1]), 2);
    assert_eq!(spec_ukeire(&tenpai.hand, &partial, mask), 6);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &partial, deficit),
        6
    );

    // A non-improving public copy must not change availability.
    let mut irrelevant = empty;
    irrelevant[tile(Suit::Wan, 1).index() as usize] = 1;
    assert_eq!(spec_ukeire(&tenpai.hand, &irrelevant, mask), 7);

    // Public exhaustion leaves the structural mask unchanged and contributes
    // zero unseen copies for every improving kind.
    let mut exhausted = empty;
    for kind in waits {
        exhausted[kind.index() as usize] = COPIES_PER_TILE - tenpai.hand.count(kind);
        assert_eq!(unseen_copies(&tenpai.hand, &exhausted, kind), 0);
    }
    let (exhausted_deficit, exhausted_mask) = evaluator.deficit_with_mask(&tenpai.hand);
    assert_eq!((exhausted_deficit, exhausted_mask), (deficit, mask));
    assert_eq!(
        exhausted_mask, 0x124,
        "all structural bits survive exhaustion"
    );
    assert_eq!(spec_ukeire(&tenpai.hand, &exhausted, mask), 0);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &exhausted, deficit),
        0
    );

    // Own meld copies are public: the peng fixture still totals eight unseen
    // improving copies, and the capped 5W contributes nothing.
    for label in ["one-peng", "peng-capacity"] {
        let state = fixtures
            .iter()
            .find(|state| state.label == label)
            .expect("open-hand fixture");
        let (deficit, mask) = check_state(state, CheckDepth::Full);
        assert_eq!((deficit, mask), (1, 0x124));
        assert_eq!(spec_ukeire(&state.hand, &state.public, mask), 8);
        let mut frozen =
            baseline::DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        assert_eq!(
            baseline::ukeire(&mut frozen, &state.hand, &state.public, deficit),
            8
        );
    }
    let capped = fixtures
        .iter()
        .find(|state| state.label == "peng-capacity")
        .expect("peng fixture");
    assert_eq!(
        unseen_copies(&capped.hand, &capped.public, tile(Suit::Wan, 5)),
        0,
        "own meld copy plus the held copy exhaust the physical copies"
    );

    // The kong fixture's structural distance-two mask still sums only unseen
    // copies: 3 copies each for 1W/2W/6W and one for 9W.
    let kong = fixtures
        .iter()
        .find(|state| state.label == "concealed-kong-capacity")
        .expect("kong fixture");
    let (deficit, mask) = check_state(kong, CheckDepth::Full);
    assert_eq!(deficit, 2);
    assert_eq!(spec_ukeire(&kong.hand, &kong.public, mask), 10);
}

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
