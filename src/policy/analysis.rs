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
//! [`DistanceEvaluator::deficit_with_mask`] additionally propagates, for every
//! DP state, the union of tile kinds that the cheapest partial targets reaching
//! that state still need, including kinds entirely absent from the hand.
//! The resulting 27-bit mask is exactly the set of
//! draws that lower the distance, so [`ukeire`] no longer reruns the distance
//! analysis once per tile kind. The masked analysis keeps its own per-suit and
//! full-hand caches: a plain-distance cache hit never answers a mask query.
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

/// Per-suit masked result table indexed by `[sets][pair_used]`.
///
/// Each entry holds the cheapest per-suit cost and the union of needed ranks
/// over every partial target attaining that cost. Bits are local to the suit:
/// rank `r` is `1 << r`, so bits 9.. are always zero. Unreachable entries are
/// `(INF, 0)`; an entry with cost `INF` never carries a mask bit.
type MaskTable = [[(u8, u32); 2]; 5];

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
    /// Masked counterparts of the two plain-distance caches. They are kept
    /// separate so that a plain hit never suppresses the mask analysis.
    suit_mask_cache: HashMap<(u8, [u8; 9]), MaskTable>,
    hand_mask_cache: HashMap<[u8; TILE_KINDS], (u8, u32)>,
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
            suit_mask_cache: HashMap::new(),
            hand_mask_cache: HashMap::new(),
        })
    }

    /// The missing suit fixed for this evaluator.
    pub(crate) const fn missing(&self) -> Suit {
        self.missing
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

    /// Exact `deficit` together with the union of draws that lower it.
    ///
    /// Bit `t` of the mask is set exactly when some optimal target `T` of this
    /// hand needs kind `t`, i.e. `T[t] > hand[t]`. The mask is structural: it
    /// ignores `public`/`unseen` ownership and may therefore hold a bit for a
    /// kind whose four copies are already publicly exhausted.
    ///
    /// The returned distance is the same value as [`Self::deficit`] for this
    /// hand and context; only the cached state differs. The two routes are
    /// merged by taking the lower distance, and an exact tie unions the masks
    /// because both target families are then optimal.
    ///
    /// For a hand one tile below the target size (the post-discard and
    /// post-declaration shapes [`ukeire`] analyzes), a legal added copy of kind
    /// `t` strictly lowers this distance if and only if bit `t` is set. See
    /// `specs/archive/one-pass-ukeire/proof.md`.
    pub(crate) fn deficit_with_mask(&mut self, hand: &Hand) -> (u8, u32) {
        let key = *hand.counts();
        if let Some(&cached) = self.hand_mask_cache.get(&key) {
            return cached;
        }

        let (standard, standard_mask) = self.standard_deficit_with_mask(&key);
        // Declared melds complete standard sets and disable seven pairs.
        let (seven_pairs, seven_pairs_mask) = if self.melds.is_empty() {
            self.seven_pairs_deficit_with_mask(&key)
        } else {
            (INF, 0)
        };
        let result = if seven_pairs < standard {
            (seven_pairs, seven_pairs_mask)
        } else if seven_pairs == standard {
            (standard, standard_mask | seven_pairs_mask)
        } else {
            (standard, standard_mask)
        };
        debug_assert!(
            result.0 != INF,
            "a legal meld count always admits a standard target"
        );
        debug_assert_eq!(
            result.0,
            self.deficit(hand),
            "the masked analysis must keep the exact plain deficit"
        );
        self.hand_mask_cache.insert(key, result);
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

    /// Masked counterpart of [`Self::standard_deficit`].
    ///
    /// Costs are combined exactly as in the plain analysis. Masks are shifted
    /// from suit-local rank bits into the global tile index and merged when the
    /// combined costs tie, because both suit prefixes are then part of some
    /// cheapest target; a strictly cheaper combination replaces the mask
    /// instead of unioning it.
    fn standard_deficit_with_mask(&mut self, counts: &[u8; TILE_KINDS]) -> (u8, u32) {
        let mut combine = [[(INF, 0u32); 2]; 5];
        combine[0][0] = (0, 0);

        for suit in Suit::ALL {
            let mut local = [0u8; 9];
            let start = suit.index() * 9;
            local.copy_from_slice(&counts[start..start + 9]);

            let table = match self.suit_mask_cache.get(&(suit.index() as u8, local)) {
                Some(table) => *table,
                None => {
                    let table = suit_table_with_mask(
                        &local,
                        &self.capacities[suit.index()],
                        self.target_sets,
                    );
                    self.suit_mask_cache
                        .insert((suit.index() as u8, local), table);
                    table
                }
            };

            let shift = (suit.index() * 9) as u32;
            let mut next = [[(INF, 0u32); 2]; 5];
            for sets in 0..=self.target_sets {
                for pair in 0..=1 {
                    let (current, current_mask) = combine[sets][pair];
                    if current == INF {
                        continue;
                    }
                    for suit_sets in 0..=(self.target_sets - sets) {
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
    fn seven_pairs_deficit_with_mask(&self, counts: &[u8; TILE_KINDS]) -> (u8, u32) {
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

/// Computes the exact per-suit cost table for a standard target.
///
/// At rank `r`, `a` and `b` count sequences started at `r - 2` and `r - 1`;
/// `sequences` starts new sequences at `r`, and `triplets` chooses triplets.
/// The pair may be placed at any rank. Capacity is charged before enumerating
/// the candidates that add tiles at this rank: the pending tails `a + b` are
/// mandatory target tiles, so a state that cannot pay them is dropped, and the
/// leftover capacity bounds the new sequences and triplets. The pair is still
/// enumerated in the innermost loop and only rejected when the resulting
/// per-rank count exceeds that rank's capacity, so the traversal order and
/// every legal transition are unchanged.
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
        let rank_capacity = capacity[rank] as usize;

        for &active_state in &active[..active_len] {
            let state = active_state as usize;
            let current = dp[state];
            let pair_used = state & 1;
            let rest = state >> 1;
            let sets = rest % STRIDE_SETS;
            let rest = rest / STRIDE_SETS;
            let b = rest % STRIDE_B;
            let a = rest / STRIDE_B;

            // Tails carried from earlier ranks are mandatory tiles here.
            let reserved = a + b;
            if reserved > rank_capacity {
                continue;
            }
            let free = rank_capacity - reserved;
            let set_budget = target_sets - sets;
            let max_sequences = set_budget.min(max_new_sequences).min(free);
            for sequences in 0..=max_sequences {
                let after_sequences = free - sequences;
                for triplets in 0..=(set_budget - sequences) {
                    // The pair only raises the count, so once the triplet count
                    // alone overflows the free capacity no pair choice can fit
                    // and the increasing triplet count cannot recover.
                    if 3 * triplets > after_sequences {
                        break;
                    }
                    for pair in 0..=1 {
                        if pair == 1 && pair_used == 1 {
                            continue;
                        }
                        let count = 3 * triplets + sequences + reserved + 2 * pair;
                        if count > rank_capacity {
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

/// Computes the exact masked per-suit table for a standard target.
///
/// This mirrors [`suit_table`] transition for transition, including the A2
/// capacity bounds and the `pair` placement, so the costs are identical. Each
/// state additionally keeps the union of needed local ranks over every partial
/// target attaining its cheapest cost: a rank whose target count exceeds the
/// held count contributes its bit, a strictly cheaper arrival replaces
/// `(cost, mask)`, and an equal-cost arrival only unions the mask because both
/// prefixes are then part of some cheapest completion. Unreachable states stay
/// `(INF, 0)`: `INF` entries are never written or ORed because every written
/// cost is finite: only reachable active states are expanded, and their
/// target costs are too small to saturate at `INF`.
fn suit_table_with_mask(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize) -> MaskTable {
    // Flat state: (((a * 5 + b) * 5 + sets) * 2 + pair_used).
    const STRIDE_A: usize = 5;
    const STRIDE_B: usize = 5;
    const STRIDE_SETS: usize = 5;
    const STATES: usize = STRIDE_A * STRIDE_B * STRIDE_SETS * 2;

    let index = |a: usize, b: usize, sets: usize, pair: usize| -> usize {
        (((a * STRIDE_B + b) * STRIDE_SETS + sets) * 2) + pair
    };

    let mut dp = [(INF, 0u32); STATES];
    let mut active = [0u16; STATES];
    let mut active_len = 0usize;
    let start = index(0, 0, 0, 0);
    dp[start] = (0, 0);
    active[0] = start as u16;
    active_len += 1;

    for rank in 0..9 {
        let mut next = [(INF, 0u32); STATES];
        let mut next_active = [0u16; STATES];
        let mut next_len = 0usize;
        let max_new_sequences = if rank <= 6 { target_sets } else { 0 };
        let rank_capacity = capacity[rank] as usize;

        for &active_state in &active[..active_len] {
            let state = active_state as usize;
            let (current, current_mask) = dp[state];
            let pair_used = state & 1;
            let rest = state >> 1;
            let sets = rest % STRIDE_SETS;
            let rest = rest / STRIDE_SETS;
            let b = rest % STRIDE_B;
            let a = rest / STRIDE_B;

            // Tails carried from earlier ranks are mandatory tiles here.
            let reserved = a + b;
            if reserved > rank_capacity {
                continue;
            }
            let free = rank_capacity - reserved;
            let set_budget = target_sets - sets;
            let max_sequences = set_budget.min(max_new_sequences).min(free);
            for sequences in 0..=max_sequences {
                let after_sequences = free - sequences;
                for triplets in 0..=(set_budget - sequences) {
                    if 3 * triplets > after_sequences {
                        break;
                    }
                    for pair in 0..=1 {
                        if pair == 1 && pair_used == 1 {
                            continue;
                        }
                        let count = 3 * triplets + sequences + reserved + 2 * pair;
                        if count > rank_capacity {
                            continue;
                        }
                        let needed = count as u8 > hand[rank];
                        let extra = (count as u8).saturating_sub(hand[rank]);
                        let cost = current.saturating_add(extra);
                        let mask = if needed {
                            current_mask | (1u32 << rank)
                        } else {
                            current_mask
                        };
                        let next_state =
                            index(b, sequences, sets + triplets + sequences, pair_used + pair);
                        let slot = &mut next[next_state];
                        if cost < slot.0 {
                            if slot.0 == INF {
                                next_active[next_len] = next_state as u16;
                                next_len += 1;
                            }
                            *slot = (cost, mask);
                        } else if cost == slot.0 {
                            slot.1 |= mask;
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

    let mut table = [[(INF, 0u32); 2]; 5];
    for sets in 0..=target_sets {
        for pair in 0..=1 {
            table[sets][pair] = dp[index(0, 0, sets, pair)];
        }
    }
    table
}

/// Counts unseen copies of each tile kind that improve a post-discard hand.
///
/// `hand` is the equivalent hand one tile below the target size (post-discard
/// or post-declaration, before the replacement draw), and `deficit` must be
/// its exact distance `D(hand)` under the same evaluator context.
///
/// The structural criterion is read from [`DistanceEvaluator::deficit_with_mask`]:
/// kind `t` improves the hand exactly when some optimal target needs it, i.e.
/// when bit `t` is set. Justification, with the full proof in
/// `specs/archive/one-pass-ukeire/proof.md`:
///
/// - For a fixed target, adding a held copy only lowers its deficit for the
///   kinds the target still needs, so `D(hand + t)` is `D(hand)` or
///   `D(hand) - 1` and the latter holds exactly for the mask bits.
/// - A drawn winning hand has deficit zero, which is below the post-discard
///   distance (a shorter hand can never contain a whole target), so direct
///   wins are covered by the same bit instead of a separate check.
/// - The drawn hand is never compared through a raw 14-tile/13-tile value:
///   both sides are exact target deficits for the same fixed target size.
///
/// `public` only limits how many copies remain; a structurally improving kind
/// whose four copies are publicly exhausted keeps its mask bit but contributes
/// zero to the returned count.
pub(crate) fn ukeire(
    evaluator: &mut DistanceEvaluator<'_>,
    hand: &Hand,
    public: &[u8; TILE_KINDS],
    deficit: u8,
) -> u32 {
    let (structural_deficit, mask) = evaluator.deficit_with_mask(hand);
    debug_assert_eq!(
        structural_deficit, deficit,
        "ukeire must use the exact distance of the same hand and context"
    );
    ukeire_from_mask(evaluator.missing(), hand, public, mask)
}

/// Sums unseen copies for the structurally improving kinds in `mask`.
///
/// Missing-suit kinds are skipped defensively even though no legal target can
/// contain them, so a mask produced for a legal context never wastes a bit on
/// them.
fn ukeire_from_mask(missing: Suit, hand: &Hand, public: &[u8; TILE_KINDS], mask: u32) -> u32 {
    let mut total = 0u32;
    for tile in Tile::ALL {
        if tile.suit() == missing || mask & (1u32 << tile.index()) == 0 {
            continue;
        }
        let index = tile.index() as usize;
        let unseen = COPIES_PER_TILE as i32 - hand.count(tile) as i32 - public[index] as i32;
        if unseen > 0 {
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
mod one_pass_reference;
#[cfg(test)]
mod one_pass_tests;
#[cfg(test)]
mod optimization_a2_tests;
#[cfg(test)]
mod optimization_b_tests;
#[cfg(test)]
mod tests;
