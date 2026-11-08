//! Standard-target analysis over the three suits.
//!
//! Each suit chooses triplets and sequences while tracking the two pending
//! sequence tails; a second pass combines the per-suit tables and places the
//! single pair. The plain, complete-masked, and bounded-masked routes combine
//! costs identically, so every route reports the same distance; only the
//! accumulated mask bits and the cutoff filtering differ.

use std::collections::HashMap;

use crate::tile::{Suit, TILE_KINDS};

use super::super::cache::HeuristicCache;
use super::suit_dp::{masked_suit_table, suit_table};
use super::{BoundedNeededCache, BoundedSuitTable, DistanceEvaluator, MaskTable, INF};

impl<'a> DistanceEvaluator<'a> {
    pub(super) fn standard_deficit(&mut self, counts: &[u8; TILE_KINDS]) -> u8 {
        let mut combine = [[INF; 2]; 5];
        combine[0][0] = 0;

        for suit in Suit::ALL {
            let mut local = [0u8; 9];
            let start = suit.index() * 9;
            local.copy_from_slice(&counts[start..start + 9]);

            let table = match self.suit_cache.get(&(suit.index() as u8, local)) {
                Some(table) => *table,
                None => {
                    let table =
                        suit_table(&local, &self.capacities[suit.index()], self.target_sets);
                    self.suit_cache.insert((suit.index() as u8, local), table);
                    table
                }
            };

            let mut next = [[INF; 2]; 5];
            for sets in 0..=self.target_sets {
                for pair in 0..=1 {
                    let current = combine[sets][pair];
                    if current == INF {
                        continue;
                    }
                    for suit_sets in 0..=(self.target_sets - sets) {
                        for suit_pair in 0..=1 {
                            if pair + suit_pair > 1 {
                                continue;
                            }
                            let cost = table[suit_sets][suit_pair];
                            if cost == INF {
                                continue;
                            }
                            let total = current.saturating_add(cost);
                            let slot = &mut next[sets + suit_sets][pair + suit_pair];
                            if total < *slot {
                                *slot = total;
                            }
                        }
                    }
                }
            }
            combine = next;
        }

        combine[self.target_sets][1]
    }

    /// Masked counterpart of [`Self::standard_deficit`].
    ///
    /// Costs are combined exactly as in the plain analysis. Masks are shifted
    /// from suit-local rank bits into the global tile index and merged when the
    /// combined costs tie, because both suit prefixes are then part of some
    /// cheapest target; a strictly cheaper combination replaces the mask
    /// instead of unioning it.
    #[allow(dead_code)]
    pub(super) fn standard_deficit_with_mask(&mut self, counts: &[u8; TILE_KINDS]) -> (u8, u32) {
        standard_masked::<false>(
            &mut self.suit_mask_cache,
            &self.capacities,
            self.target_sets,
            counts,
            self.shared,
        )
    }

    /// Bounded counterpart of [`Self::standard_deficit_with_mask`].
    ///
    /// The bounded suit cache stays `None` while complete suit tables can
    /// serve the query; it is allocated only when an actual per-suit build is
    /// needed.
    pub(super) fn standard_deficit_with_mask_bounded(
        &mut self,
        counts: &[u8; TILE_KINDS],
        cutoff: u8,
    ) -> (u8, u32) {
        let Self {
            suit_mask_cache,
            bounded_needed_cache,
            capacities,
            target_sets,
            shared,
            ..
        } = self;
        standard_masked_bounded(
            suit_mask_cache,
            bounded_needed_cache,
            capacities,
            *target_sets,
            counts,
            cutoff,
            *shared,
        )
    }
}

/// Shared masked standard analysis for both mask meanings.
///
/// `SURPLUS` selects the per-suit mask predicate: `false` is the needed mask
/// used by [`super::ukeire`] (target count above the held count), `true` is the
/// surplus mask used by [`DistanceEvaluator::deficit_with_surplus`] (target
/// count below the held count). Costs are identical to
/// [`DistanceEvaluator::standard_deficit`] in both variants; only the
/// accumulated bits differ. The suit table is computed once per exact local
/// hand and cached in the caller's cache, which belongs to one evaluator; an
/// optional cross-decision context may answer the same key across evaluators.
///
/// Suit-local masks are shifted to global tile indices and unioned on equal
/// cost, because both prefixes are then part of some cheapest completion; a
/// strictly cheaper combination replaces the mask instead of unioning it.
pub(super) fn standard_masked<const SURPLUS: bool>(
    suit_cache: &mut HashMap<(u8, [u8; 9]), MaskTable>,
    capacities: &[[u8; 9]; 3],
    target_sets: usize,
    counts: &[u8; TILE_KINDS],
    shared: Option<&HeuristicCache>,
) -> (u8, u32) {
    let mut combine = [[(INF, 0u32); 2]; 5];
    combine[0][0] = (0, 0);

    for suit in Suit::ALL {
        let mut local = [0u8; 9];
        let start = suit.index() * 9;
        local.copy_from_slice(&counts[start..start + 9]);

        let table = match suit_cache.get(&(suit.index() as u8, local)) {
            Some(table) => *table,
            None => {
                let table = masked_suit_table::<SURPLUS, false>(
                    &local,
                    &capacities[suit.index()],
                    target_sets,
                    INF,
                    shared,
                );
                suit_cache.insert((suit.index() as u8, local), table);
                table
            }
        };

        let shift = (suit.index() * 9) as u32;
        let mut next = [[(INF, 0u32); 2]; 5];
        for sets in 0..=target_sets {
            for pair in 0..=1 {
                let (current, current_mask) = combine[sets][pair];
                if current == INF {
                    continue;
                }
                for suit_sets in 0..=(target_sets - sets) {
                    for suit_pair in 0..=1 {
                        if pair + suit_pair > 1 {
                            continue;
                        }
                        let (cost, suit_mask) = table[suit_sets][suit_pair];
                        if cost == INF {
                            continue;
                        }
                        let total = current.saturating_add(cost);
                        let mask = current_mask | (suit_mask << shift);
                        let slot = &mut next[sets + suit_sets][pair + suit_pair];
                        if total < slot.0 {
                            *slot = (total, mask);
                        } else if total == slot.0 {
                            slot.1 |= mask;
                        }
                    }
                }
            }
        }
        combine = next;
    }

    combine[target_sets][1]
}

/// Bounded counterpart of [`standard_masked`] for the needed mask.
///
/// `cutoff` is an exact full-hand deficit. A per-suit table entry with cost
/// above it can never participate in an optimal full target, so it is treated
/// as unreachable. The complete suit cache is consulted first and serves any
/// cutoff; otherwise the bounded cache is reused only when its built cutoff is
/// at least as wide as this query, and a wider query replaces the entry. A
/// bounded build is never written into the complete cache.
///
/// A cross-decision miss computes the truncated table and stores it under the
/// exact bound, so a table built for one cutoff never answers another one. A
/// cross-decision hit fills the local bounded cache with the retrieved table;
/// any local entry it replaces is narrower, because a wider local table is
/// reused before the shared lookup. Query-time filtering happens on the
/// returned copy and never rewrites a stored entry.
fn standard_masked_bounded(
    suit_cache: &mut HashMap<(u8, [u8; 9]), MaskTable>,
    bounded_cache: &mut Option<Box<BoundedNeededCache>>,
    capacities: &[[u8; 9]; 3],
    target_sets: usize,
    counts: &[u8; TILE_KINDS],
    cutoff: u8,
    shared: Option<&HeuristicCache>,
) -> (u8, u32) {
    debug_assert!(
        cutoff < INF,
        "a finite exact deficit is required for a bounded query"
    );

    let mut combine = [[(INF, 0u32); 2]; 5];
    combine[0][0] = (0, 0);

    for suit in Suit::ALL {
        let mut local = [0u8; 9];
        let start = suit.index() * 9;
        local.copy_from_slice(&counts[start..start + 9]);
        let key = (suit.index() as u8, local);

        let mut table = match suit_cache.get(&key) {
            Some(table) => *table,
            None => match bounded_cache
                .as_ref()
                .and_then(|cache| cache.suit.get(&key))
            {
                Some(entry) if entry.cutoff >= cutoff => entry.table,
                _ => {
                    let table = masked_suit_table::<false, true>(
                        &local,
                        &capacities[suit.index()],
                        target_sets,
                        cutoff,
                        shared,
                    );
                    bounded_cache
                        .get_or_insert_with(Box::default)
                        .suit
                        .insert(key, BoundedSuitTable { cutoff, table });
                    table
                }
            },
        };

        // A complete table or a table built for a wider cutoff can carry
        // finite entries above this query's cutoff. They are exact for their
        // own cost, but this query must treat them as unreachable rather than
        // letting a too-expensive suit participate.
        for row in &mut table {
            for entry in row {
                if entry.0 > cutoff {
                    *entry = (INF, 0);
                }
            }
        }

        let shift = (suit.index() * 9) as u32;
        let mut next = [[(INF, 0u32); 2]; 5];
        for sets in 0..=target_sets {
            for pair in 0..=1 {
                let (current, current_mask) = combine[sets][pair];
                if current == INF {
                    continue;
                }
                for suit_sets in 0..=(target_sets - sets) {
                    for suit_pair in 0..=1 {
                        if pair + suit_pair > 1 {
                            continue;
                        }
                        let (cost, suit_mask) = table[suit_sets][suit_pair];
                        if cost == INF {
                            continue;
                        }
                        let total = current.saturating_add(cost);
                        let mask = current_mask | (suit_mask << shift);
                        let slot = &mut next[sets + suit_sets][pair + suit_pair];
                        if total < slot.0 {
                            *slot = (total, mask);
                        } else if total == slot.0 {
                            slot.1 |= mask;
                        }
                    }
                }
            }
        }
        combine = next;
    }

    let result = combine[target_sets][1];
    // A standard route above the exact full-hand distance is not optimal and
    // may have been truncated internally; normalize it instead of exposing a
    // partial finite result. The caller merges the exact seven-pairs route.
    if result.0 > cutoff {
        (INF, 0)
    } else {
        result
    }
}
