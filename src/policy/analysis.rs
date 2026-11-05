//! Private exact hand analysis for [`super::HeuristicPolicy`].
//!
//! The central quantity is the *target deficit* from the policy specification:
//! for a post-discard concealed hand `H` (with the declared melds fixed), the
//! minimum number of copies that must be drawn to reach any legal winning
//! target `T`:
//!
//! ```text
//! D(H) = min over legal targets T of sum_t max(0, T[t] - H[t])
//! S(H) = D(H) - 1
//! ```
//!
//! A target has `14 - 3 * meld_count` concealed tiles and may not contain the
//! missing suit or exceed four physical copies together with the player's own
//! melds. Surplus copies in `H` are discarded for free, which is why the sum
//! only counts positive deficits.
//!
//! Standard targets use an exact counted dynamic program:
//!
//! - For a standard target, each suit independently chooses triplets and
//!   sequences while tracking the two pending sequence tails. A second pass
//!   combines the relevant suits and places the single pair.
//! - A seven-pairs target is only legal with no declared melds, where every
//!   non-missing kind keeps the full four-copy capacity. It is therefore
//!   evaluated with a closed form over held pair units and odd kinds rather
//!   than a per-kind target DP; see `seven_pairs_deficit` below.
//!
//! No heuristic shanten estimate is used. The only memoization is keyed by
//! exact counted hands (and exact per-suit count vectors); it is local to one
//! policy decision and can never change an action.

use std::collections::HashMap;

use crate::action::Meld;
use crate::eval;
use crate::hand::Hand;
use crate::rules::{self, WinMethod};
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

const INF: u8 = u8::MAX;

/// Per-suit result table indexed by `[sets][pair_used]`.
type SuitTable = [[u8; 2]; 5];

/// Exact target-deficit evaluator for one fixed meld set and missing suit.
///
/// Creating a new evaluator per meld set keeps the four-copy capacity and the
/// seven-pairs availability fixed for its whole cache lifetime.
pub(crate) struct DistanceEvaluator<'a> {
    melds: &'a [Meld],
    missing: Suit,
    target_sets: usize,
    /// Per-suit, per-rank maximum concealed copies allowed in a target.
    capacities: [[u8; 9]; 3],
    suit_cache: HashMap<(u8, [u8; 9]), SuitTable>,
    hand_cache: HashMap<[u8; TILE_KINDS], u8>,
}

impl<'a> DistanceEvaluator<'a> {
    /// Builds an evaluator, or `None` for an impossible meld count.
    pub(crate) fn new(melds: &'a [Meld], missing: Suit) -> Option<Self> {
        if melds.len() > 4 {
            return None;
        }

        let mut capacities = [[COPIES_PER_TILE; 9]; 3];
        // A winning target never contains the missing suit, even when the
        // missing-suit tiles have already been discarded from the hand.
        capacities[missing.index()].fill(0);
        for meld in melds {
            let suit = meld.tile().suit_index();
            let rank = meld.tile().rank_index();
            capacities[suit][rank] = capacities[suit][rank].saturating_sub(meld.tile_count());
        }

        Some(Self {
            melds,
            missing,
            target_sets: 4 - melds.len(),
            capacities,
            suit_cache: HashMap::new(),
            hand_cache: HashMap::new(),
        })
    }

    /// The missing suit fixed for this evaluator.
    pub(crate) const fn missing(&self) -> Suit {
        self.missing
    }

    /// The declared melds fixed for this evaluator.
    pub(crate) const fn melds(&self) -> &'a [Meld] {
        self.melds
    }

    /// Exact `D(hand)` for the fixed target size.
    ///
    /// A hand equal to a legal target has deficit zero; a post-discard hand of
    /// one fewer tile has deficit at least one, so its shanten is `deficit - 1`.
    /// Zero deficit identifies the input as a target only when the input and
    /// target sizes agree and the declared melds are legal.
    pub(crate) fn deficit(&mut self, hand: &Hand) -> u8 {
        let key = *hand.counts();
        if let Some(&cached) = self.hand_cache.get(&key) {
            return cached;
        }

        let standard = self.standard_deficit(&key);
        // Declared melds complete standard sets and disable seven pairs.
        let seven_pairs = if self.melds.is_empty() {
            self.seven_pairs_deficit(&key)
        } else {
            INF
        };
        let result = standard.min(seven_pairs);
        debug_assert!(
            result != INF,
            "a legal meld count always admits a standard target"
        );
        self.hand_cache.insert(key, result);
        result
    }

    fn standard_deficit(&mut self, counts: &[u8; TILE_KINDS]) -> u8 {
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
    fn seven_pairs_deficit(&self, counts: &[u8; TILE_KINDS]) -> u8 {
        debug_assert!(
            self.melds.is_empty(),
            "declared melds disable the seven-pairs route"
        );

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

        // At most 18 kinds x 4 held copies, so `free_units` fits a byte and the
        // result is at most 14.
        let missing_units = 7u8.saturating_sub(free_units);
        missing_units + missing_units.saturating_sub(odd_kinds)
    }
}

/// Computes the exact per-suit cost table for a standard target.
///
/// At rank `r`, `a` and `b` count sequences started at `r - 2` and `r - 1`;
/// `sequences` starts new sequences at `r`, and `triplets` chooses triplets.
/// The pair may be placed at any rank. A transition is rejected when the resulting per-rank
/// target count exceeds the four-copy capacity or the total set budget.
fn suit_table(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize) -> SuitTable {
    // Flat state: (((a * 5 + b) * 5 + sets) * 2 + pair_used).
    const STRIDE_A: usize = 5;
    const STRIDE_B: usize = 5;
    const STRIDE_SETS: usize = 5;
    const STATES: usize = STRIDE_A * STRIDE_B * STRIDE_SETS * 2;

    let index = |a: usize, b: usize, sets: usize, pair: usize| -> usize {
        (((a * STRIDE_B + b) * STRIDE_SETS + sets) * 2) + pair
    };

    let mut dp = [INF; STATES];
    let mut active = [0u16; STATES];
    let mut active_len = 0usize;
    let start = index(0, 0, 0, 0);
    dp[start] = 0;
    active[0] = start as u16;
    active_len += 1;

    for rank in 0..9 {
        let mut next = [INF; STATES];
        let mut next_active = [0u16; STATES];
        let mut next_len = 0usize;
        let max_new_sequences = if rank <= 6 { target_sets } else { 0 };

        for &active_state in &active[..active_len] {
            let state = active_state as usize;
            let current = dp[state];
            let pair_used = state & 1;
            let rest = state >> 1;
            let sets = rest % STRIDE_SETS;
            let rest = rest / STRIDE_SETS;
            let b = rest % STRIDE_B;
            let a = rest / STRIDE_B;

            let set_budget = target_sets - sets;
            let max_sequences = set_budget.min(max_new_sequences);
            for sequences in 0..=max_sequences {
                for triplets in 0..=(set_budget - sequences) {
                    for pair in 0..=1 {
                        if pair == 1 && pair_used == 1 {
                            continue;
                        }
                        let count = 3 * triplets + sequences + a + b + 2 * pair;
                        if count > capacity[rank] as usize {
                            continue;
                        }
                        let extra = (count as u8).saturating_sub(hand[rank]);
                        let cost = current.saturating_add(extra);
                        let next_state =
                            index(b, sequences, sets + triplets + sequences, pair_used + pair);
                        if cost < next[next_state] {
                            if next[next_state] == INF {
                                next_active[next_len] = next_state as u16;
                                next_len += 1;
                            }
                            next[next_state] = cost;
                        }
                    }
                }
            }
        }

        dp = next;
        active = next_active;
        active_len = next_len;
        if active_len == 0 {
            break;
        }
    }

    let mut table = [[INF; 2]; 5];
    for sets in 0..=target_sets {
        for pair in 0..=1 {
            table[sets][pair] = dp[index(0, 0, sets, pair)];
        }
    }
    table
}

/// Counts unseen copies of each tile kind that improve a post-discard hand.
///
/// `deficit` is `D(H)` for the post-discard hand. The normalization proof used
/// here is:
///
/// - `H` has one fewer tile than a target.
/// - After drawing `t`, the drawn hand and every target have the same size.
/// - If the drawn hand is not winning, every optimal target has at least one
///   surplus kind, so discarding a surplus tile leaves that target's deficit
///   unchanged. Conversely, discarding a needed kind can only raise it. Hence
///   the best legal post-draw discard has deficit `D(H + t)` exactly.
/// - A winning drawn hand has deficit zero and is a direct win.
///
/// Therefore `t` improves `H` iff `D(H + t) < D(H)`. The drawn hand is never
/// compared against `H` through a raw 14-tile/13-tile value: both sides are
/// exact target deficits for the same target size.
pub(crate) fn ukeire(
    evaluator: &mut DistanceEvaluator<'_>,
    hand: &Hand,
    public: &[u8; TILE_KINDS],
    deficit: u8,
) -> u32 {
    let mut total = 0u32;
    for tile in Tile::ALL {
        if tile.suit() == evaluator.missing() {
            continue;
        }
        let index = tile.index() as usize;
        let seen = hand.count(tile) as i32 + public[index] as i32;
        let unseen = COPIES_PER_TILE as i32 - seen;
        if unseen <= 0 {
            continue;
        }

        let mut drawn = *hand;
        if drawn.add(tile).is_err() {
            continue;
        }

        // Recognize a direct win before the structural comparison. The
        // deficit of a winning drawn hand is also zero, so the two paths agree.
        if rules::can_hu(&drawn, evaluator.melds(), evaluator.missing()) {
            total += unseen as u32;
            continue;
        }

        if evaluator.deficit(&drawn) < deficit {
            total += unseen as u32;
        }
    }
    total
}

/// Maximum hypothetical plain discard-win fan for a legal ready hand.
///
/// This matches the big-call quantity: legal structural waits only, publicly
/// exhausted waits included, no event bonus. The missing suit is filtered
/// because a winning hand may not contain it.
pub(crate) fn max_discard_fan(hand: &Hand, melds: &[Meld], missing: Suit) -> u8 {
    let mut best = 0u8;
    for tile in eval::waiting_tiles(hand, melds) {
        if tile.suit() == missing {
            continue;
        }
        let mut winning = *hand;
        if winning.add(tile).is_err() {
            continue;
        }
        if !rules::can_hu(&winning, melds, missing) {
            continue;
        }
        if let Some(fan) = rules::win_fan(&winning, melds, WinMethod::Discard) {
            best = best.max(fan.total);
        }
    }
    best
}

#[cfg(test)]
mod optimization_b_tests;
#[cfg(test)]
mod tests;
