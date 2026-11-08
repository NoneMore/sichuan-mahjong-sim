//! Seven-pairs analysis.
//!
//! A seven-pairs target is only legal with no declared melds, where every
//! non-missing kind keeps the full four-copy capacity. The plain distance and
//! the needed mask use a closed form over held pair units and odd kinds; the
//! surplus query uses a small pair-unit DP because it must union the omitted
//! kinds over every optimal target.

use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

use super::{DistanceEvaluator, INF};

impl<'a> DistanceEvaluator<'a> {
    /// Exact `D_7(hand)` for a seven-pairs target.
    ///
    /// Only called with no declared melds, so every non-missing kind can hold
    /// the full four target copies this formula assumes. With `t` ranging over
    /// the 18 non-missing kinds:
    ///
    /// ```text
    /// P = sum_t floor(H[t] / 2)   free pair units, a quad being two of them
    /// O = sum_t (H[t] mod 2)      odd kinds, including held triples
    /// K = max(0, 7 - P)           pair units still to fill
    /// D_7 = K + max(0, K - O)
    /// ```
    ///
    /// The `P` units are target pairs fully covered by held tiles. Each of the
    /// `O` odd kinds supplies one free tile towards an extra pair, so the first
    /// `min(K, O)` missing units cost one copy each. Every remaining unit needs
    /// two copies. Surplus units beyond seven are free, hence
    /// the saturating subtraction that keeps `P > 7` from underflowing.
    pub(super) fn seven_pairs_deficit(&self, counts: &[u8; TILE_KINDS]) -> u8 {
        debug_assert!(
            self.melds.is_empty(),
            "declared melds disable the seven-pairs route"
        );

        let (free_units, odd_kinds) = self.seven_pairs_units(counts);
        seven_pairs_cost(free_units, odd_kinds)
    }

    /// Held pair units `P` and odd kinds `O` over the non-missing suits.
    ///
    /// `P = sum_t floor(H[t] / 2)` counts target pair units already fully
    /// covered by held tiles, a quad being two of them. `O = sum_t (H[t] mod 2)`
    /// counts kinds where one more copy upgrades the held tiles to a target
    /// pair, so it is the number of extra units available at cost one.
    fn seven_pairs_units(&self, counts: &[u8; TILE_KINDS]) -> (u8, u8) {
        let mut free_units = 0u8;
        let mut odd_kinds = 0u8;
        for tile in Tile::ALL {
            if tile.suit() == self.missing {
                continue;
            }
            let held = counts[tile.index() as usize];
            free_units += held / 2;
            odd_kinds += held % 2;
        }
        (free_units, odd_kinds)
    }

    /// Masked counterpart of [`Self::seven_pairs_deficit`].
    ///
    /// The optimal seven-pairs targets are characterized by the cost of their
    /// pair units, so the needed-kind union follows from the same `P`/`O`
    /// quantities (see `specs/archive/one-pass-ukeire/proof.md`):
    ///
    /// - zero deficit: every optimal target has cost zero, so no target count
    ///   can exceed the held count and the mask is empty;
    /// - `0 < K <= O` with `K = 7 - P`: an optimal target extends exactly `K`
    ///   distinct odd kinds by one copy, hence every odd kind is needed and no
    ///   other kind can be;
    /// - `O < K`: every odd kind is extended and the remaining `K - O` units
    ///   cost two each from any non-missing kind holding fewer than four
    ///   copies, so every such kind is needed.
    pub(super) fn seven_pairs_deficit_with_mask(&self, counts: &[u8; TILE_KINDS]) -> (u8, u32) {
        debug_assert!(
            self.melds.is_empty(),
            "declared melds disable the seven-pairs route"
        );

        let (free_units, odd_kinds) = self.seven_pairs_units(counts);
        let deficit = seven_pairs_cost(free_units, odd_kinds);
        if deficit == 0 {
            return (0, 0);
        }

        let missing_units = 7u8.saturating_sub(free_units);
        let every_held_kind = missing_units > odd_kinds;
        let mut mask = 0u32;
        for tile in Tile::ALL {
            if tile.suit() == self.missing {
                continue;
            }
            let held = counts[tile.index() as usize];
            let needed = if every_held_kind {
                held < COPIES_PER_TILE
            } else {
                held % 2 == 1
            };
            if needed {
                mask |= 1u32 << tile.index();
            }
        }
        (deficit, mask)
    }
}

/// Exact seven-pairs cost from the `P`/`O` quantities documented on
/// [`DistanceEvaluator::seven_pairs_deficit`].
///
/// Shared by the plain and masked routes so the two can never disagree on the
/// distance.
fn seven_pairs_cost(free_units: u8, odd_kinds: u8) -> u8 {
    let missing_units = 7u8.saturating_sub(free_units);
    missing_units + missing_units.saturating_sub(odd_kinds)
}

/// Exact seven-pairs distance together with the union of surplus kinds over
/// every optimal seven-pairs target.
///
/// A seven-pairs target is a choice of seven pair units over the 18 non-missing
/// kinds, each kind contributing zero, one, or two units (`0`, `2`, or `4`
/// target copies). The DP walks the kinds in encoding order and keeps, for every
/// unit count, the cheapest cost and the surplus union over all partial targets
/// attaining it. Each kind is scored against its held count exactly like the
/// standard suit DP: a target count above the held count costs the difference;
/// a target count below it costs nothing and sets the surplus bit. A strictly
/// cheaper arrival replaces `(cost, mask)` and an
/// equal-cost arrival unions the mask, so the final entry is exact over every
/// optimal seven-pairs target.
///
/// Missing-suit kinds are walked with units fixed at zero instead of being
/// skipped: a held copy can never appear in a target, so it is surplus and its
/// bit must join the union. The distance agrees with [`seven_pairs_cost`].
pub(super) fn seven_pairs_deficit_surplus(missing: Suit, counts: &[u8; TILE_KINDS]) -> (u8, u32) {
    let mut dp = [(INF, 0u32); 8];
    dp[0] = (0, 0);

    for kind in Tile::ALL {
        let held = counts[kind.index() as usize];
        let max_units = if kind.suit() == missing { 0 } else { 2 };
        let mut next = [(INF, 0u32); 8];
        for (units, &(cost, mask)) in dp.iter().enumerate() {
            if cost == INF {
                continue;
            }
            for target_units in 0..=max_units {
                let total_units = units + target_units;
                if total_units > 7 {
                    continue;
                }
                let target_count = (2 * target_units) as u8;
                let extra = target_count.saturating_sub(held);
                let new_cost = cost.saturating_add(extra);
                let new_mask = if target_count < held {
                    mask | (1u32 << kind.index())
                } else {
                    mask
                };
                let slot = &mut next[total_units];
                if new_cost < slot.0 {
                    *slot = (new_cost, new_mask);
                } else if new_cost == slot.0 {
                    slot.1 |= new_mask;
                }
            }
        }
        dp = next;
    }

    dp[7]
}
