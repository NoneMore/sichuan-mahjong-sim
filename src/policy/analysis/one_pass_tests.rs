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
//! The checks are grouped by subject: `oracle.rs` holds the concrete-target
//! reference, `suit.rs` and `bounded_needed.rs` check the per-suit kernels,
//! `physical.rs` the physical fixtures and the supported domain, `draw.rs` the
//! draw-level equivalence, `cache_order.rs` the cache lifetime, and
//! `seven_pairs.rs` the seven-pairs route. The shared fixtures live in
//! `analysis::test_fixtures`.
//!
//! Scope and honesty notes:
//!
//! - Fixtures are hand-picked boundary positions; [`super::test_fixtures::sampled_states`]
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

mod bounded_needed;
mod cache_order;
mod draw;
mod oracle;
mod physical;
mod seven_pairs;
mod suit;

use super::one_pass_reference as baseline;
use super::test_fixtures::{assert_physical, capacity_by_kind, meld_copies, State};
use super::{ukeire, DistanceEvaluator};

use self::oracle::{reference_hand_mask, reference_hand_mask_direct};
use crate::hand::Hand;
use crate::rules;
use crate::tile::{Tile, COPIES_PER_TILE, TILE_KINDS};

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
    // The evaluator above already holds the complete mask. Also exercise the
    // production bounded calculation cold, rather than only its shared hit.
    let mut cold = DistanceEvaluator::new(melds, state.missing).expect("at most four melds");
    assert_eq!(
        ukeire(&mut cold, &state.hand, &state.public, deficit),
        expected_ukeire,
        "{}: cold bounded ukeire differs from the mask/unseen formula",
        state.label
    );
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
