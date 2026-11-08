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
//!   evaluated with a closed form over held pair units and odd kinds for plain
//!   distance and needed masks; the surplus query uses a small pair-unit DP.
//!
//! Needed-mask analysis additionally propagates, for every DP state, the union
//! of tile kinds that the cheapest partial targets reaching
//! that state still need, including kinds entirely absent from the hand.
//! The resulting 27-bit mask is exactly the set of
//! draws that lower the distance, so [`ukeire`] no longer reruns the distance
//! analysis once per tile kind. The masked analysis keeps its own per-suit and
//! full-hand caches: a plain-distance cache hit never answers a mask query.
//! [`ukeire`] uses [`DistanceEvaluator::deficit_with_mask_bounded`] with its
//! caller-known exact distance; the complete `deficit_with_mask` route remains
//! available without a cutoff.
//!
//! [`DistanceEvaluator::deficit_with_surplus`] is the dual query for an
//! original hand: it returns the same deficit together with the union of kinds
//! that some optimal target holds fewer of than the hand, so one analysis
//! answers every candidate discard as `D(H - t) = D(H)` or `D(H) + 1` without a
//! per-candidate distance solve. It is not the complement of the needed mask
//! and keeps its own caches, allocated only on the first surplus query.
//!
//! No heuristic shanten estimate is used. The decision-local memoization is
//! keyed by exact counted hands (and exact per-suit count vectors). An optional
//! [`HeuristicCache`] may additionally hold exact masked per-suit tables across
//! decisions; it is keyed by the complete local input of the kernel, is
//! consulted only after the local caches miss, and can never change an action.
//!
//! This module owns the evaluator, its caches, and the query surface. The
//! counted dynamic programs live in the child modules [`suit_dp`] (per-suit
//! kernels), [`standard`] (per-suit combination), and [`seven_pairs`].

use std::collections::HashMap;

use crate::action::Meld;
use crate::eval;
use crate::hand::Hand;
use crate::rules::{self, WinMethod};
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

use super::cache::HeuristicCache;

use self::seven_pairs::seven_pairs_deficit_surplus;
use self::standard::standard_masked;

mod seven_pairs;
mod standard;
mod suit_dp;

/// Test-facing per-suit kernels, re-exported on the historical module path so
/// the analysis tests keep addressing them as `analysis::suit_table` and
/// friends. Production reaches them through [`standard`] only.
#[cfg(test)]
use self::suit_dp::{masked_suit_table, suit_table, suit_table_mask_bounded, suit_table_with_mask};

const INF: u8 = u8::MAX;

/// Per-suit result table indexed by `[sets][pair_used]`.
type SuitTable = [[u8; 2]; 5];

/// Per-suit masked result table indexed by `[sets][pair_used]`.
///
/// Each entry holds the cheapest per-suit cost and the union of marked ranks
/// (needed or surplus, according to the query) over every partial target
/// attaining that cost. Bits are local to the suit:
/// rank `r` is `1 << r`, so bits 9.. are always zero. Unreachable entries are
/// `(INF, 0)`; an entry with cost `INF` never carries a mask bit.
pub(crate) type MaskTable = [[(u8, u32); 2]; 5];

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
    /// Optional cross-decision context for exact masked suit tables. It is
    /// read only after the decision-local caches miss.
    shared: Option<&'a HeuristicCache>,
    suit_cache: HashMap<(u8, [u8; 9]), SuitTable>,
    hand_cache: HashMap<[u8; TILE_KINDS], u8>,
    /// Masked counterparts of the two plain-distance caches. They are kept
    /// separate so that a plain hit never suppresses the mask analysis.
    suit_mask_cache: HashMap<(u8, [u8; 9]), MaskTable>,
    hand_mask_cache: HashMap<[u8; TILE_KINDS], (u8, u32)>,
    /// Bounded needed-mask suit tables, `None` until the first bounded query
    /// that actually needs a per-suit build. Ordinary distance, complete
    /// needed masks, and surplus queries never allocate or consult it.
    bounded_needed_cache: Option<Box<BoundedNeededCache>>,
    /// Surplus-analysis caches, `None` until the first surplus query so an
    /// evaluator used for ordinary distance or needed masks pays neither the
    /// analysis nor an allocation for them.
    surplus_cache: Option<Box<SurplusCache>>,
}

/// Lazily allocated caches for [`DistanceEvaluator::deficit_with_surplus`].
///
/// The keys are the same exact counted hands and per-suit vectors as the other
/// caches, so the cache lifetime (one evaluator, one decision) and the meld,
/// missing-suit, and capacity isolation are unchanged.
#[derive(Default)]
struct SurplusCache {
    suit: HashMap<(u8, [u8; 9]), MaskTable>,
    hand: HashMap<[u8; TILE_KINDS], (u8, u32)>,
}

/// One lazily built bounded-needed suit table.
///
/// `cutoff` is the exact full-hand distance the table was built for. Every
/// entry with cost `<= cutoff` is exact, including its mask; every other entry
/// is `(INF, 0)`. A cached table can therefore answer any query whose cutoff
/// is no larger than `cutoff` after filtering its table rows, but a wider
/// query must rebuild it.
struct BoundedSuitTable {
    cutoff: u8,
    table: MaskTable,
}

/// Lazily allocated cache for bounded needed-mask suit tables.
///
/// Kept separate from the complete [`DistanceEvaluator::suit_mask_cache`] so a
/// truncated table can never answer a complete query or a wider cutoff. The
/// complete table is always read first and may serve any bounded query; a
/// bounded build never writes the complete map.
#[derive(Default)]
struct BoundedNeededCache {
    suit: HashMap<(u8, [u8; 9]), BoundedSuitTable>,
}

impl<'a> DistanceEvaluator<'a> {
    /// Builds an evaluator, or `None` for an impossible meld count.
    ///
    /// This evaluator uses its decision-local caches only. Production binds an
    /// explicit cross-decision context with [`Self::with_cache`]; this
    /// uncached form backs the differential tests.
    #[cfg(test)]
    pub(crate) fn new(melds: &'a [Meld], missing: Suit) -> Option<Self> {
        Self::build(melds, missing, None)
    }

    /// Builds an evaluator that may reuse masked suit tables from `cache`.
    ///
    /// The local caches are still consulted first; a cross-decision hit fills
    /// the corresponding local cache with the retrieved table.
    pub(crate) fn with_cache(
        melds: &'a [Meld],
        missing: Suit,
        cache: &'a HeuristicCache,
    ) -> Option<Self> {
        Self::build(melds, missing, Some(cache))
    }

    fn build(melds: &'a [Meld], missing: Suit, shared: Option<&'a HeuristicCache>) -> Option<Self> {
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
            shared,
            suit_cache: HashMap::new(),
            hand_cache: HashMap::new(),
            suit_mask_cache: HashMap::new(),
            hand_mask_cache: HashMap::new(),
            bounded_needed_cache: None,
            surplus_cache: None,
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
    ///
    /// Production ukeire uses [`Self::deficit_with_mask_bounded`]. This
    /// complete route stays available for differential tests and for callers
    /// that do not already know the exact deficit; it is currently unreferenced
    /// outside tests, so the dead-code lint is explicitly suppressed.
    #[allow(dead_code)]
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

    /// Exact needed `(deficit, mask)` using a caller-known exact deficit as a
    /// strict upper cutoff.
    ///
    /// `exact_deficit` must be `D_C(hand)` from this evaluator's context, as
    /// supplied by every [`ukeire`] caller. All transition costs are
    /// non-negative, so no target with total cost at most `exact_deficit` can
    /// pass through a partial state whose accumulated cost exceeds it. A
    /// strict `g > d` cutoff therefore preserves every per-suit entry with
    /// cost `<= d` and its entire optimal mask; ties at `g == d` are kept
    /// because they can still finish at cost `d` and contribute mask bits.
    ///
    /// The standard route may be truncated above `d`; the seven-pairs route is
    /// evaluated exactly and merged as usual, so a strictly cheaper seven-pairs
    /// target never becomes unreachable. Because every optimal full-hand
    /// target has all its per-suit costs at most `d`, the merged result is the
    /// complete global `(D, M)` and is cached in the shared hand mask cache.
    ///
    /// Bounded per-suit tables live in a separate cache and are only reused for
    /// a query whose cutoff they already cover; the complete suit cache is
    /// consulted first and is never written by this path.
    pub(crate) fn deficit_with_mask_bounded(
        &mut self,
        hand: &Hand,
        exact_deficit: u8,
    ) -> (u8, u32) {
        let key = *hand.counts();
        if let Some(&cached) = self.hand_mask_cache.get(&key) {
            return cached;
        }

        let (standard, standard_mask) =
            self.standard_deficit_with_mask_bounded(&key, exact_deficit);
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
        debug_assert_eq!(
            result.0, exact_deficit,
            "bounded needed analysis must preserve the caller's exact deficit"
        );
        debug_assert_eq!(
            result.0,
            self.deficit(hand),
            "the bounded analysis must keep the exact plain deficit"
        );
        self.hand_mask_cache.insert(key, result);
        result
    }

    /// Exact `deficit` of an original hand together with its surplus mask.
    ///
    /// Intended for the original `14 - 3 * melds.len()` tile hand a policy ranks
    /// discards from. Bit `t` is set exactly when some optimal target `T`
    /// attaining the returned deficit holds fewer `t` than the hand, i.e.
    /// `T[t] < hand[t]`. Removing one held copy of kind `t` therefore leaves the
    /// deficit unchanged exactly for the surplus bits:
    ///
    /// ```text
    /// D(hand - t) = D(hand)      when bit t is set
    ///             = D(hand) + 1  otherwise
    /// ```
    ///
    /// The identity also holds for other counted sizes, but the intended use is
    /// the original hand; a post-discard hand one tile below the target size has
    /// its own exact [`Self::deficit`].
    ///
    /// The surplus mask is not the complement of the needed mask from
    /// [`Self::deficit_with_mask`]: different optimal targets may make the same
    /// kind both needed and surplus, and a held missing-suit copy is surplus
    /// because no target may contain it. Equally cheap standard and seven-pairs
    /// targets union their masks; a strictly cheaper route replaces the other
    /// one, and a strictly cheaper target always replaces the mask instead of
    /// unioning it.
    pub(crate) fn deficit_with_surplus(&mut self, hand: &Hand) -> (u8, u32) {
        let key = *hand.counts();
        if let Some(cached) = self
            .surplus_cache
            .as_ref()
            .and_then(|cache| cache.hand.get(&key))
        {
            return *cached;
        }

        let melds_empty = self.melds.is_empty();
        let missing = self.missing;
        let target_sets = self.target_sets;
        let capacities = self.capacities;
        let shared = self.shared;
        let cache = self.surplus_cache.get_or_insert_with(Box::default);

        let (standard, standard_mask) =
            standard_masked::<true>(&mut cache.suit, &capacities, target_sets, &key, shared);
        // Declared melds complete standard sets and disable seven pairs.
        let (seven_pairs, seven_pairs_mask) = if melds_empty {
            seven_pairs_deficit_surplus(missing, &key)
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
        cache.hand.insert(key, result);
        debug_assert_eq!(
            result.0,
            self.deficit(hand),
            "the surplus analysis must keep the exact plain deficit"
        );
        result
    }
}

/// Counts unseen copies of each tile kind that improve a post-discard hand.
///
/// `hand` is the equivalent hand one tile below the target size (post-discard
/// or post-declaration, before the replacement draw), and `deficit` must be
/// its exact distance `D(hand)` under the same evaluator context.
///
/// The structural criterion is read from
/// [`DistanceEvaluator::deficit_with_mask_bounded`] using the supplied distance:
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
    let (structural_deficit, mask) = evaluator.deficit_with_mask_bounded(hand, deficit);
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
mod cache_tests;
#[cfg(test)]
mod discard_surplus_tests;
#[cfg(test)]
mod one_pass_reference;
#[cfg(test)]
mod one_pass_tests;
#[cfg(test)]
mod optimization_a2_tests;
#[cfg(test)]
mod optimization_b_tests;
#[cfg(test)]
mod test_fixtures;
#[cfg(test)]
mod tests;
