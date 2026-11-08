//! Per-suit dynamic programs for the counted standard targets.
//!
//! Both kernels charge target tiles against the hand's per-rank counts; the
//! masked kernel additionally carries, for every state, the union of marked
//! ranks over every partial target attaining its cheapest cost. The plain and
//! masked variants share the transition order, so they can never disagree on a
//! cost. Caching belongs to the caller: the evaluator caches the plain table,
//! and [`masked_suit_table`] is the single gate for masked tables, including
//! the optional cross-decision context.

use super::super::cache::{HeuristicCache, MaskKind, COMPLETE_BOUND};

use super::{MaskTable, SuitTable, INF};

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
pub(super) fn suit_table(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize) -> SuitTable {
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

/// Returns one masked per-suit table from the cross-decision cache or the DP.
///
/// `SURPLUS` selects the mask meaning and `BOUNDED` the truncated needed table
/// for the caller's exact full-hand cutoff. The cross-decision key carries the
/// kind and the exact bound, so a stored table can only answer the same query.
/// A miss computes the table with the unchanged kernel and stores a copy; the
/// returned table is owned by the caller, so the later cutoff filtering of the
/// analysis never rewrites a stored entry.
pub(super) fn masked_suit_table<const SURPLUS: bool, const BOUNDED: bool>(
    hand: &[u8; 9],
    capacity: &[u8; 9],
    target_sets: usize,
    cutoff: u8,
    shared: Option<&HeuristicCache>,
) -> MaskTable {
    let kind = if SURPLUS {
        MaskKind::Surplus
    } else {
        MaskKind::Needed
    };
    let bound = if BOUNDED { cutoff } else { COMPLETE_BOUND };
    if let Some(cache) = shared {
        if let Some(table) = cache.lookup_suit(hand, capacity, target_sets, kind, bound) {
            return table;
        }
    }
    let table = suit_table_masked_kernel::<SURPLUS, BOUNDED>(hand, capacity, target_sets, cutoff);
    if let Some(cache) = shared {
        cache.insert_suit(hand, capacity, target_sets, kind, bound, &table);
    }
    table
}

/// Computes the exact masked per-suit table for a standard target, marking
/// needed ranks (target count above the held count).
///
/// This is the test-facing name used by the complete-table checks; it goes
/// through the same cache gate as production with no cross-decision context.
#[cfg(test)]
pub(super) fn suit_table_with_mask(
    hand: &[u8; 9],
    capacity: &[u8; 9],
    target_sets: usize,
) -> MaskTable {
    masked_suit_table::<false, false>(hand, capacity, target_sets, INF, None)
}

/// Bounded needed-mask per-suit table for an exact full-hand cutoff.
///
/// Every entry with cost `<= cutoff` is exact; every other entry is `(INF, 0)`.
/// Strictly greater successors are pruned, so a prefix at cost exactly `cutoff`
/// remains available for zero-cost completion and its mask bits survive.
///
/// This is the test-facing bounded build; production goes through
/// [`masked_suit_table`] so the cross-decision context can answer first.
#[cfg(test)]
pub(super) fn suit_table_mask_bounded(
    hand: &[u8; 9],
    capacity: &[u8; 9],
    target_sets: usize,
    cutoff: u8,
) -> MaskTable {
    debug_assert!(
        cutoff < INF,
        "a finite exact deficit is required for a bounded query"
    );
    suit_table_masked_kernel::<false, true>(hand, capacity, target_sets, cutoff)
}

/// Shared masked per-suit DP for the needed and surplus masks.
///
/// This mirrors [`suit_table`] transition for transition, including the A2
/// capacity bounds and the `pair` placement, so the costs are identical. Each
/// state additionally keeps the union of marked local ranks over every partial
/// target attaining its cheapest cost. `SURPLUS` selects the predicate:
///
/// - `false` marks `count > hand[rank]`, the ranks a target still needs;
/// - `true` marks `count < hand[rank]`, the ranks a target holds fewer of.
///
/// A strictly cheaper arrival replaces `(cost, mask)`, and an equal-cost
/// arrival only unions the mask because both prefixes are then part of some
/// cheapest completion. Unreachable states stay `(INF, 0)`: `INF` entries are
/// never written or ORed because every written cost is finite: only reachable
/// active states are expanded, and their target costs are too small to saturate
/// at `INF`.
///
/// `BOUNDED` switches on the strict successor cutoff for a caller-known exact
/// full-hand deficit. In the unbounded instantiation the cutoff comparison is a
/// compile-time false and disappears, so the complete needed and surplus paths
/// keep their original runtime behavior. Bounded queries are needed-only;
/// surplus is never bounded.
fn suit_table_masked_kernel<const SURPLUS: bool, const BOUNDED: bool>(
    hand: &[u8; 9],
    capacity: &[u8; 9],
    target_sets: usize,
    cutoff: u8,
) -> MaskTable {
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
                        let count = count as u8;
                        let marked = if SURPLUS {
                            count < hand[rank]
                        } else {
                            count > hand[rank]
                        };
                        let extra = count.saturating_sub(hand[rank]);
                        let cost = current.saturating_add(extra);
                        // A successor above the exact full-hand distance can
                        // never belong to an optimal target. Keep `== cutoff`:
                        // it may still finish at zero further cost.
                        if BOUNDED && cost > cutoff {
                            continue;
                        }
                        let mask = if marked {
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
            let entry = dp[index(0, 0, sets, pair)];
            table[sets][pair] = if BOUNDED && entry.0 > cutoff {
                (INF, 0)
            } else {
                entry
            };
        }
    }
    table
}
