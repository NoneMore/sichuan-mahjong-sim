//! Frozen removed-tile reference path.
//!
//! Candidate distance is the ordinary exact deficit after removing the tile,
//! exactly as the pre-change ranking computed it; this path never asks for the
//! surplus analysis. The ranking built on those scores is the old selection
//! logic: minimum deficit, minimum-only ukeire, then the tenpai-only fan
//! tie-break in candidate order.

use super::super::*;

/// One scored candidate from the frozen removed-tile reference path.
pub(super) struct ReferenceCandidate {
    pub(super) tile: Tile,
    pub(super) hand: Hand,
    pub(super) deficit: u8,
    pub(super) ukeire: u32,
    pub(super) fan: u8,
}

/// Scores every removable candidate by removing it and using the ordinary
/// exact deficit, exactly as the pre-change ranking did.
///
/// Ukeire is computed for every retained candidate so single-candidate
/// differentials can compare it. The reference ranking below still consults
/// ukeire only for minimum-deficit candidates, so the selected tile and
/// progress match the old path; the extra values are test-only.
pub(super) fn reference_candidates(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public_before_discard: &[u8; TILE_KINDS],
    candidates: &[Tile],
) -> Option<Vec<ReferenceCandidate>> {
    let mut evaluator = DistanceEvaluator::new(melds, missing)?;
    let mut scored = Vec::with_capacity(candidates.len());
    for &tile in candidates {
        let mut after = *hand;
        if after.remove(tile).is_err() {
            continue;
        }
        let deficit = evaluator.deficit(&after);
        let mut public = *public_before_discard;
        let index = tile.index() as usize;
        public[index] = public[index].saturating_add(1);
        let ukeire = analysis::ukeire(&mut evaluator, &after, &public, deficit);
        scored.push(ReferenceCandidate {
            tile,
            hand: after,
            deficit,
            ukeire,
            fan: 0,
        });
    }
    Some(scored)
}

/// Frozen ranking over already scored reference candidates.
///
/// This is the old selection logic: minimum deficit, minimum-only ukeire, then
/// the tenpai-only fan tie-break in candidate order. Empty scores select
/// nothing.
pub(super) fn reference_ranking(
    scored: &mut [ReferenceCandidate],
    melds: &[Meld],
    missing: Suit,
    with_fan: bool,
) -> Option<(Tile, Progress)> {
    if scored.is_empty() {
        return None;
    }
    let best_deficit = scored
        .iter()
        .map(|candidate| candidate.deficit)
        .min()
        .expect("a scored discard exists");
    let mut best_ukeire = 0u32;
    for candidate in scored.iter().filter(|c| c.deficit == best_deficit) {
        best_ukeire = best_ukeire.max(candidate.ukeire);
    }

    let tenpai = best_deficit == 1;
    let finalists: Vec<usize> = (0..scored.len())
        .filter(|&index| {
            scored[index].deficit == best_deficit && scored[index].ukeire == best_ukeire
        })
        .collect();
    let best_index = if with_fan && tenpai && finalists.len() > 1 {
        let mut best = finalists[0];
        for &index in &finalists {
            scored[index].fan = analysis::max_discard_fan(&scored[index].hand, melds, missing);
            if scored[index].fan > scored[best].fan {
                best = index;
            }
        }
        best
    } else {
        finalists[0]
    };

    Some((
        scored[best_index].tile,
        Progress {
            shanten: best_deficit as i32 - 1,
            ukeire: best_ukeire,
        },
    ))
}

/// Frozen old path for one ranking call.
pub(super) fn reference_rank_discards(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public_before_discard: &[u8; TILE_KINDS],
    candidates: &[Tile],
    with_fan: bool,
) -> Option<(Tile, Progress)> {
    let mut scored = reference_candidates(hand, melds, missing, public_before_discard, candidates)?;
    reference_ranking(&mut scored, melds, missing, with_fan)
}
