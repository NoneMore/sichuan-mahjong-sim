//! Differential checks for the one-pass candidate-deficit ranking.
//!
//! The reference is the pre-change path: remove one candidate tile and ask for
//! the ordinary exact deficit of the remaining hand. It never asks for the new
//! surplus analysis. Candidate collection, the minimum-deficit filter, the
//! minimum-only ukeire stage, the fan tie-break, and the peng/kong gates stay
//! shared with production; only the candidate distance comes from a frozen
//! test-only ranking built on ordinary deficits.
//!
//! Every helper is test-only. Production deliberately has no instrumentation
//! for the "once per original hand" analysis count, so the tests compare
//! observable deficits, ukeire, selected tiles, and progress instead.
//! Only candidate deficits have an independent ordinary-distance reference;
//! ukeire, fan, and unchanged call helpers are intentionally reused to check
//! ranking/gate wiring, not to re-prove those helpers' semantics.
//!
//! The frozen reference lives in `reference.rs`; the checks are grouped into
//! `candidates.rs` (candidate distances), `selection.rs` (ranking and order),
//! `decisions.rs` (kong and peng gates), and `sampled.rs` (fixed-seed sweep).

#[path = "one_pass_discard/candidates.rs"]
mod candidates;
#[path = "one_pass_discard/decisions.rs"]
mod decisions;
#[path = "one_pass_discard/reference.rs"]
mod reference;
#[path = "one_pass_discard/sampled.rs"]
mod sampled;
#[path = "one_pass_discard/selection.rs"]
mod selection;

use super::*;

use self::reference::{reference_candidates, reference_ranking};

/// Compares the production and removed-tile rankings for one context and
/// candidate list.
///
/// The full ranking must agree on the selected tile and `Progress`. Every
/// distinct candidate is then ranked alone, which exposes its exact deficit
/// (`shanten + 1`) and ukeire without adding production hooks; invalid
/// candidates must be skipped by both paths.
fn assert_rank_differential(
    label: &str,
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    candidates: &[Tile],
    with_fan: bool,
) {
    let actual = rank_discards(
        hand,
        melds,
        missing,
        public,
        candidates,
        with_fan,
        &uncached(),
    );
    let mut reference = reference_candidates(hand, melds, missing, public, candidates)
        .expect("test contexts use a legal meld count");
    let expected = reference_ranking(&mut reference, melds, missing, with_fan);
    assert_eq!(actual, expected, "{label}: full candidate ranking");

    let mut unique = Vec::new();
    for &candidate in candidates {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    for candidate in unique {
        match reference.iter().find(|scored| scored.tile == candidate) {
            Some(scored) => {
                let (tile, progress) = rank_discards(
                    hand,
                    melds,
                    missing,
                    public,
                    &[candidate],
                    false,
                    &uncached(),
                )
                .expect("a held candidate is scored");
                assert_eq!(tile, candidate, "{label}: single candidate {candidate}");
                assert_eq!(
                    progress.shanten,
                    scored.deficit as i32 - 1,
                    "{label}: deficit after removing {candidate}"
                );
                assert_eq!(
                    progress.ukeire, scored.ukeire,
                    "{label}: ukeire after removing {candidate}"
                );
            }
            None => assert_eq!(
                rank_discards(
                    hand,
                    melds,
                    missing,
                    public,
                    &[candidate],
                    false,
                    &uncached()
                ),
                None,
                "{label}: invalid candidate {candidate} must be skipped"
            ),
        }
    }
}

/// Exact production deficit for one held candidate, recovered from a
/// single-candidate ranking where that candidate is necessarily minimal.
fn single_deficit(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    candidate: Tile,
) -> u8 {
    let (tile, progress) = rank_discards(
        hand,
        melds,
        missing,
        public,
        &[candidate],
        false,
        &uncached(),
    )
    .unwrap_or_else(|| panic!("candidate {candidate} must be held"));
    assert_eq!(tile, candidate);
    u8::try_from(progress.shanten + 1).expect("deficit fits in u8")
}

fn plain_deficit(hand: &Hand, melds: &[Meld], missing: Suit) -> u8 {
    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid meld count");
    evaluator.deficit(hand)
}

fn all_candidates(hand: &Hand) -> Vec<Tile> {
    hand.distinct_tiles().collect()
}

fn assert_physical(hand: &Hand, melds: &[Meld], missing: Suit, public: &[u8; TILE_KINDS]) {
    assert!(melds.len() <= 4, "at most four declared melds");
    assert_eq!(
        hand.total(),
        14 - 3 * melds.len(),
        "ranking input holds the target-size concealed hand"
    );
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        assert!(
            hand.count(kind) + public[index] <= crate::tile::COPIES_PER_TILE,
            "{kind} exceeds four physical copies"
        );
    }
    for meld in melds {
        let index = meld.tile().index() as usize;
        assert_ne!(
            meld.tile().suit(),
            missing,
            "an own meld outside the missing suit"
        );
        assert!(
            public[index] >= meld.tile_count(),
            "public ownership includes the own meld copies"
        );
        assert!(
            hand.count(meld.tile()) + meld.tile_count() <= crate::tile::COPIES_PER_TILE,
            "concealed and meld copies stay within the physical deck"
        );
    }
}
