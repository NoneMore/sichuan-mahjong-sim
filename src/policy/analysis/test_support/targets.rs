//! Concrete per-suit target enumeration shared by the analysis references.
//!
//! A target is a multiset of nine triplets and seven sequences plus an optional
//! pair. The walkers visit every legal concrete multiset directly; they never
//! call the production DP. Callers choose the projection they need:
//!
//! - [`reference_cost_table`] keeps only the cheapest cost (the A2 enumeration
//!   baseline);
//! - [`reference_needed_table`] keeps the union of needed ranks over cheapest
//!   targets (the one-pass mask oracle);
//! - [`reference_mask_tables`] keeps both the needed and the surplus unions (the
//!   cross-decision cache oracle).
//!
//! `[sets][pair]` rows above the requested set budget stay `(INF, 0)`, and an
//! unreachable entry never carries a mask bit. Equal-cost targets union their
//! masks; a cheaper target replaces them.

use super::super::{MaskTable, SuitTable, INF};

/// Nine ranks in one suit.
pub(crate) const RANKS: usize = 9;

/// Nine triplets followed by seven sequences.
const SET_CANDIDATES: usize = RANKS + 7;

/// Nine triplets followed by seven sequences, in a fixed order.
pub(crate) fn suit_set_candidates() -> [[u8; RANKS]; SET_CANDIDATES] {
    let mut sets = [[0u8; RANKS]; SET_CANDIDATES];
    for (rank, set) in sets.iter_mut().take(RANKS).enumerate() {
        set[rank] = 3;
    }
    for start in 0..=6 {
        sets[RANKS + start][start..start + 3].fill(1);
    }
    sets
}

/// Empty per-suit mask table with unreachable entries normalized.
pub(crate) fn empty_mask_table() -> MaskTable {
    [[(INF, 0u32); 2]; 5]
}

/// Empty per-suit cost table with unreachable entries normalized.
pub(crate) fn empty_cost_table() -> SuitTable {
    [[INF; 2]; 5]
}

/// Deficit of one concrete target against the held counts.
pub(crate) fn target_deficit<const N: usize>(held: &[u8; N], target: &[u8; N]) -> u8 {
    let mut cost = 0u8;
    for rank in 0..N {
        cost = cost.saturating_add(target[rank].saturating_sub(held[rank]));
    }
    cost
}

/// Deficit plus the needed and surplus rank masks of one concrete target.
///
/// The masks use the same bit positions as the target array: local ranks for a
/// suit target and global kinds for a full-hand target.
pub(crate) fn target_score<const N: usize>(held: &[u8; N], target: &[u8; N]) -> (u8, u32, u32) {
    let mut cost = 0u8;
    let mut needed = 0u32;
    let mut surplus = 0u32;
    for rank in 0..N {
        let want = target[rank];
        let have = held[rank];
        cost = cost.saturating_add(want.saturating_sub(have));
        if want > have {
            needed |= 1 << rank;
        }
        if want < have {
            surplus |= 1 << rank;
        }
    }
    (cost, needed, surplus)
}

/// Records a cheaper mask entry or unions an equal-cost one.
pub(crate) fn merge_entry(slot: &mut (u8, u32), cost: u8, mask: u32) {
    if cost < slot.0 {
        *slot = (cost, mask);
    } else if cost == slot.0 {
        slot.1 |= mask;
    }
}

/// Records the cheaper cost and keeps the slot unchanged on a tie.
pub(crate) fn merge_cost(slot: &mut u8, cost: u8) {
    if cost < *slot {
        *slot = cost;
    }
}

/// Merges two route results: cheaper wins, equal costs union their masks.
pub(crate) fn merge_route(best: (u8, u32), candidate: (u8, u32)) -> (u8, u32) {
    match candidate.0.cmp(&best.0) {
        std::cmp::Ordering::Less => candidate,
        std::cmp::Ordering::Equal => (best.0, best.1 | candidate.1),
        std::cmp::Ordering::Greater => best,
    }
}

/// Adds one counted set to a target or hand vector.
pub(crate) fn add_counts<const N: usize>(counts: &mut [u8; N], add: &[u8; N]) {
    for (count, add) in counts.iter_mut().zip(add) {
        *count += *add;
    }
}

/// Removes one counted set from a target or hand vector.
pub(crate) fn subtract_counts<const N: usize>(counts: &mut [u8; N], remove: &[u8; N]) {
    for (count, remove) in counts.iter_mut().zip(remove) {
        *count -= *remove;
    }
}

/// Visits every paired variant of a concrete target.
///
/// The pair may sit on any rank with two free target copies. The no-pair
/// variant is the caller's original target and is not visited here.
pub(crate) fn for_each_paired_target(
    target: &[u8; RANKS],
    capacity: &[u8; RANKS],
    visit: &mut impl FnMut(&[u8; RANKS]),
) -> usize {
    let mut count = 0;
    for rank in 0..RANKS {
        if target[rank] + 2 > capacity[rank] {
            continue;
        }
        let mut paired = *target;
        paired[rank] += 2;
        visit(&paired);
        count += 1;
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn walk_targets(
    start: usize,
    placed: usize,
    target_sets: usize,
    sets: &[[u8; RANKS]; SET_CANDIDATES],
    capacity: &[u8; RANKS],
    target: &mut [u8; RANKS],
    visit_partials: bool,
    visit: &mut impl FnMut(&[u8; RANKS], usize),
) -> usize {
    let mut visited = 0;
    if visit_partials {
        visit(target, placed);
        visited += 1;
    }
    if placed == target_sets {
        if !visit_partials {
            visit(target, placed);
            visited += 1;
        }
        return visited;
    }
    for index in start..sets.len() {
        let candidate = &sets[index];
        if (0..RANKS).any(|rank| target[rank] + candidate[rank] > capacity[rank]) {
            continue;
        }
        add_counts(target, candidate);
        visited += walk_targets(
            index,
            placed + 1,
            target_sets,
            sets,
            capacity,
            target,
            visit_partials,
            visit,
        );
        subtract_counts(target, candidate);
    }
    visited
}

/// Visits every concrete multiset of at most `target_sets` sets.
///
/// The visitor also receives the number of sets already placed, so callers can
/// fill every `[sets]` row in one pass. Returns the number of target visits.
pub(crate) fn for_each_suit_target_up_to(
    target_sets: usize,
    capacity: &[u8; RANKS],
    visit: &mut impl FnMut(&[u8; RANKS], usize),
) -> usize {
    let sets = suit_set_candidates();
    let mut target = [0u8; RANKS];
    walk_targets(0, 0, target_sets, &sets, capacity, &mut target, true, visit)
}

/// Visits every concrete multiset of exactly `target_sets` sets.
///
/// Returns the number of target visits.
pub(crate) fn for_each_suit_target_exactly(
    target_sets: usize,
    capacity: &[u8; RANKS],
    visit: &mut impl FnMut(&[u8; RANKS]),
) -> usize {
    let sets = suit_set_candidates();
    let mut target = [0u8; RANKS];
    walk_targets(
        0,
        0,
        target_sets,
        &sets,
        capacity,
        &mut target,
        false,
        &mut |target, _| visit(target),
    )
}

/// Cheapest per-suit cost table by concrete-target enumeration.
pub(crate) fn reference_cost_table(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
) -> SuitTable {
    let mut table = empty_cost_table();
    for (chosen, row) in table.iter_mut().enumerate().take(target_sets + 1) {
        for_each_suit_target_exactly(chosen, capacity, &mut |target| {
            merge_cost(&mut row[0], target_deficit(hand, target));
            for_each_paired_target(target, capacity, &mut |paired| {
                merge_cost(&mut row[1], target_deficit(hand, paired));
            });
        });
    }
    table
}

/// Cheapest per-suit needed table by concrete-target enumeration.
pub(crate) fn reference_needed_table(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
) -> MaskTable {
    let mut table = empty_mask_table();
    for_each_suit_target_up_to(target_sets, capacity, &mut |target, placed| {
        let (cost, needed, _) = target_score(hand, target);
        merge_entry(&mut table[placed][0], cost, needed);
        for_each_paired_target(target, capacity, &mut |paired| {
            let (cost, needed, _) = target_score(hand, paired);
            merge_entry(&mut table[placed][1], cost, needed);
        });
    });
    table
}

/// Cheapest per-suit needed and surplus tables by concrete-target enumeration.
pub(crate) fn reference_mask_tables(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
) -> (MaskTable, MaskTable) {
    let mut needed = empty_mask_table();
    let mut surplus = empty_mask_table();
    for_each_suit_target_up_to(target_sets, capacity, &mut |target, placed| {
        let (cost, needed_mask, surplus_mask) = target_score(hand, target);
        merge_entry(&mut needed[placed][0], cost, needed_mask);
        merge_entry(&mut surplus[placed][0], cost, surplus_mask);
        for_each_paired_target(target, capacity, &mut |paired| {
            let (cost, needed_mask, surplus_mask) = target_score(hand, paired);
            merge_entry(&mut needed[placed][1], cost, needed_mask);
            merge_entry(&mut surplus[placed][1], cost, surplus_mask);
        });
    });
    (needed, surplus)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The up-to walker visits exactly the rows the exact walkers visit once.
    ///
    /// This pins the shared enumeration contract the cost, needed, and
    /// needed/surplus projections all rely on: one visit per legal multiset,
    /// grouped by its set count, with no duplicate or missing row.
    #[test]
    fn suit_walkers_visit_each_legal_multiset_once() {
        let capacity = [4u8; RANKS];
        let exact: Vec<usize> = (0..=4)
            .map(|sets| for_each_suit_target_exactly(sets, &capacity, &mut |_| {}))
            .collect();
        let mut up_to = vec![0usize; exact.len()];
        let total = for_each_suit_target_up_to(4, &capacity, &mut |_, placed| {
            up_to[placed] += 1;
        });
        assert_eq!(up_to, exact, "up-to and exact rows must agree");
        assert_eq!(total, exact.iter().sum::<usize>());

        // Zero capacity still visits the empty target exactly once and admits
        // no pair, so unreachable rows stay unreachable.
        let zero = [0u8; RANKS];
        assert_eq!(
            for_each_suit_target_up_to(4, &zero, &mut |_, _| {}),
            1,
            "zero capacity has only the empty multiset"
        );
    }
}
