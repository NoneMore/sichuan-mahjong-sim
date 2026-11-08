//! Independent definition oracle for the one-pass mask contract.
//!
//! The oracle never reuses a production transition. A standard target is a
//! union of per-suit targets, each a concrete multiset of triplets and
//! sequences plus an optional pair, so [`reference_suit_table`] enumerates
//! those multisets directly and keeps, per `[sets][pair]`, the cheapest cost
//! and the union of needed ranks over every cheapest target. The full-hand
//! reference combines those concrete per-suit families over every
//! `(sets, pair)` allocation. For fixed per-suit `(sets, pair)` counts, a
//! globally cheapest target must use a per-suit cheapest target for those
//! counts; otherwise replacing that suit part would lower the total. Hence the
//! union of the per-suit masks for the globally cheapest allocations is
//! exactly the union over all globally optimal targets.
//!
//! A seven-pairs target is enumerated kind by kind: each non-missing kind
//! contributes `0`, `2`, or `4` copies, the target must reach seven pair units,
//! and equal-cost partial targets union their masks. This is the per-kind
//! definition, not the production closed form.
//!
//! [`reference_hand_mask_direct`] additionally enumerates standard targets
//! globally (all sets across all suits, then the pair) with no per-suit
//! decomposition. It is used for meld counts `2..=4`, where the global
//! enumeration is small, and it cross-checks the combined reference and the
//! tested analysis on those shapes.
//!
//! Rows above `target_sets` and entries without a legal target stay
//! `(INF, 0)`: a cost can never be `INF`, so unreachable states cannot leak a
//! mask bit.

use super::super::test_fixtures::{capacity_by_kind, capacity_by_suit, suit_counts};
use super::super::INF;
use crate::action::Meld;
use crate::hand::Hand;
use crate::tile::{Suit, Tile, TILE_KINDS};

pub(super) const RANKS: usize = 9;

/// Per-suit reference table indexed by `[sets][pair]`.
pub(super) type RefMaskTable = [[(u8, u32); 2]; 5];

fn merge_entry(slot: &mut (u8, u32), cost: u8, mask: u32) {
    if cost < slot.0 {
        *slot = (cost, mask);
    } else if cost == slot.0 {
        slot.1 |= mask;
    }
}

pub(super) fn merge_route(best: (u8, u32), candidate: (u8, u32)) -> (u8, u32) {
    match candidate.0.cmp(&best.0) {
        std::cmp::Ordering::Less => candidate,
        std::cmp::Ordering::Equal => (best.0, best.1 | candidate.1),
        std::cmp::Ordering::Greater => best,
    }
}

/// Deficit and local needed-rank mask of one concrete per-suit target.
fn suit_target_need(hand: &[u8; RANKS], target: &[u8; RANKS]) -> (u8, u32) {
    let mut cost = 0u8;
    let mut mask = 0u32;
    for rank in 0..RANKS {
        let held = hand[rank];
        let need = target[rank];
        cost = cost.saturating_add(need.saturating_sub(held));
        if need > held {
            mask |= 1 << rank;
        }
    }
    (cost, mask)
}

/// Nine triplets followed by seven sequences, in a fixed order.
fn suit_set_candidates() -> Vec<[u8; RANKS]> {
    let mut sets = Vec::new();
    for rank in 0..RANKS {
        let mut triplet = [0u8; RANKS];
        triplet[rank] = 3;
        sets.push(triplet);
    }
    for start in 0..=6 {
        let mut sequence = [0u8; RANKS];
        sequence[start..start + 3].fill(1);
        sets.push(sequence);
    }
    sets
}

/// Visits every concrete per-suit set multiset with `placed` sets in `target`.
///
/// `start` keeps the multiset order non-decreasing so each multiset is visited
/// once. Capacity is checked against the concrete target counts, which is the
/// definition of a legal per-suit target.
fn walk_suit_targets(
    start: usize,
    placed: usize,
    target_sets: usize,
    sets: &[[u8; RANKS]],
    capacity: &[u8; RANKS],
    target: &mut [u8; RANKS],
    visit: &mut impl FnMut(&[u8; RANKS], usize),
) {
    visit(target, placed);
    if placed == target_sets {
        return;
    }
    for index in start..sets.len() {
        let candidate = &sets[index];
        if (0..RANKS).any(|rank| target[rank] + candidate[rank] > capacity[rank]) {
            continue;
        }
        for rank in 0..RANKS {
            target[rank] += candidate[rank];
        }
        walk_suit_targets(
            index,
            placed + 1,
            target_sets,
            sets,
            capacity,
            target,
            visit,
        );
        for rank in 0..RANKS {
            target[rank] -= candidate[rank];
        }
    }
}

/// Complete reference table for one suit: cheapest cost and union of local
/// masks over every cheapest concrete target.
pub(super) fn reference_suit_table(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
) -> RefMaskTable {
    debug_assert!(
        target_sets <= 4,
        "production tables only request 0..=4 sets"
    );
    let sets = suit_set_candidates();
    let mut table = [[(INF, 0u32); 2]; 5];
    let mut target = [0u8; RANKS];
    walk_suit_targets(
        0,
        0,
        target_sets,
        &sets,
        capacity,
        &mut target,
        &mut |target, placed| {
            let (cost, mask) = suit_target_need(hand, target);
            merge_entry(&mut table[placed][0], cost, mask);
            for rank in 0..RANKS {
                if target[rank] + 2 > capacity[rank] {
                    continue;
                }
                let mut paired = *target;
                paired[rank] += 2;
                let (cost, mask) = suit_target_need(hand, &paired);
                merge_entry(&mut table[placed][1], cost, mask);
            }
        },
    );
    table
}

/// Standard-route distance and global mask from the combined per-suit tables.
pub(super) fn reference_standard_mask(hand: &Hand, melds: &[Meld], missing: Suit) -> (u8, u32) {
    let target_sets = 4 - melds.len();
    let capacity = capacity_by_suit(melds, missing);
    let mut tables = [[[(INF, 0u32); 2]; 5]; 3];
    for suit in Suit::ALL {
        tables[suit.index()] = reference_suit_table(
            &suit_counts(hand, suit),
            &capacity[suit.index()],
            target_sets,
        );
    }

    let mut best = (INF, 0u32);
    for sets0 in 0..=target_sets {
        for pair0 in 0..=1 {
            let (cost0, mask0) = tables[0][sets0][pair0];
            if cost0 == INF {
                continue;
            }
            for sets1 in 0..=(target_sets - sets0) {
                for pair1 in 0..=(1 - pair0) {
                    let (cost1, mask1) = tables[1][sets1][pair1];
                    if cost1 == INF {
                        continue;
                    }
                    let sets2 = target_sets - sets0 - sets1;
                    let pair2 = 1 - pair0 - pair1;
                    let (cost2, mask2) = tables[2][sets2][pair2];
                    if cost2 == INF {
                        continue;
                    }
                    let cost = cost0.saturating_add(cost1).saturating_add(cost2);
                    let mask = mask0 | (mask1 << 9) | (mask2 << 18);
                    best = merge_route(best, (cost, mask));
                }
            }
        }
    }
    best
}

/// Seven-pairs distance and mask by per-kind target enumeration.
pub(super) fn reference_seven_pairs_mask(counts: &[u8; TILE_KINDS], missing: Suit) -> (u8, u32) {
    const UNITS: usize = 7;
    let mut best = [(INF, 0u32); UNITS + 1];
    best[0] = (0, 0);

    for kind in Tile::ALL {
        if kind.suit() == missing {
            continue;
        }
        let held = counts[kind.index() as usize];
        let mut next = [(INF, 0u32); UNITS + 1];
        for units in 0..=UNITS {
            let (cost, mask) = best[units];
            if cost == INF {
                continue;
            }
            for copies in [0u8, 2, 4] {
                let added = copies as usize / 2;
                if units + added > UNITS {
                    continue;
                }
                let next_cost = cost.saturating_add(copies.saturating_sub(held));
                let bit = if copies > held {
                    1u32 << kind.index()
                } else {
                    0
                };
                merge_entry(&mut next[units + added], next_cost, mask | bit);
            }
        }
        best = next;
    }
    best[UNITS]
}

/// Seven-pairs distance and mask by a direct depth-first enumeration of
/// concrete targets.
///
/// Each of the 18 non-missing kinds takes `0`, `2`, or `4` target copies, the
/// recursion stops exactly at seven pair units, and every complete target is
/// scored independently: partial assignments are never merged. There are
/// exactly 241,128 complete seven-pairs targets for a full 18-kind context, so
/// the returned visit count is a reproducible witness of exhaustive coverage.
/// The production closed form and [`reference_seven_pairs_mask`] reuse no part
/// of this traversal.
pub(super) fn reference_seven_pairs_mask_direct(
    counts: &[u8; TILE_KINDS],
    missing: Suit,
) -> ((u8, u32), usize) {
    let kinds: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .collect();
    let mut target = [0u8; TILE_KINDS];
    let mut best = (INF, 0u32);
    let mut visited = 0usize;

    fn walk(
        position: usize,
        units: usize,
        kinds: &[Tile],
        counts: &[u8; TILE_KINDS],
        target: &mut [u8; TILE_KINDS],
        best: &mut (u8, u32),
        visited: &mut usize,
    ) {
        if units == 7 {
            let mut cost = 0u8;
            let mut mask = 0u32;
            for kind in Tile::ALL {
                let index = kind.index() as usize;
                let held = counts[index];
                let need = target[index];
                cost = cost.saturating_add(need.saturating_sub(held));
                if need > held {
                    mask |= 1u32 << kind.index();
                }
            }
            *visited += 1;
            *best = merge_route(*best, (cost, mask));
            return;
        }
        if position == kinds.len() {
            return;
        }
        // The remaining kinds can add at most two pair units each.
        if units + 2 * (kinds.len() - position) < 7 {
            return;
        }
        let index = kinds[position].index() as usize;
        for copies in [0u8, 2, 4] {
            let added = copies as usize / 2;
            if units + added > 7 {
                continue;
            }
            target[index] = copies;
            walk(
                position + 1,
                units + added,
                kinds,
                counts,
                target,
                best,
                visited,
            );
            target[index] = 0;
        }
    }

    walk(0, 0, &kinds, counts, &mut target, &mut best, &mut visited);
    (best, visited)
}

/// Full-hand distance and optimal-target mask from the combined reference.
pub(super) fn reference_hand_mask(hand: &Hand, melds: &[Meld], missing: Suit) -> (u8, u32) {
    let standard = reference_standard_mask(hand, melds, missing);
    if melds.is_empty() {
        let seven_pairs = reference_seven_pairs_mask(hand.counts(), missing);
        merge_route(standard, seven_pairs)
    } else {
        standard
    }
}

/// Triplets for non-missing kinds followed by sequences for non-missing suits.
fn kind_set_candidates(missing: Suit) -> Vec<[u8; TILE_KINDS]> {
    let mut sets = Vec::new();
    for kind in Tile::ALL {
        if kind.suit() == missing {
            continue;
        }
        let mut triplet = [0u8; TILE_KINDS];
        triplet[kind.index() as usize] = 3;
        sets.push(triplet);
    }
    for suit in Suit::ALL {
        if suit == missing {
            continue;
        }
        for start in 0..=6 {
            let mut sequence = [0u8; TILE_KINDS];
            for offset in 0..3 {
                let rank = start + offset;
                let index = suit.index() * 9 + rank;
                sequence[index] = 1;
            }
            sets.push(sequence);
        }
    }
    sets
}

fn kind_target_need(hand: &Hand, target: &[u8; TILE_KINDS]) -> (u8, u32) {
    let counts = hand.counts();
    let mut cost = 0u8;
    let mut mask = 0u32;
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        let held = counts[index];
        let need = target[index];
        cost = cost.saturating_add(need.saturating_sub(held));
        if need > held {
            mask |= 1u32 << kind.index();
        }
    }
    (cost, mask)
}

#[allow(clippy::too_many_arguments)]
fn walk_kind_targets(
    start: usize,
    placed: usize,
    target_sets: usize,
    sets: &[[u8; TILE_KINDS]],
    capacity: &[u8; TILE_KINDS],
    target: &mut [u8; TILE_KINDS],
    visit: &mut impl FnMut(&[u8; TILE_KINDS], usize),
) {
    visit(target, placed);
    if placed == target_sets {
        return;
    }
    for index in start..sets.len() {
        let candidate = &sets[index];
        if (0..TILE_KINDS).any(|kind| target[kind] + candidate[kind] > capacity[kind]) {
            continue;
        }
        for kind in 0..TILE_KINDS {
            target[kind] += candidate[kind];
        }
        walk_kind_targets(
            index,
            placed + 1,
            target_sets,
            sets,
            capacity,
            target,
            visit,
        );
        for kind in 0..TILE_KINDS {
            target[kind] -= candidate[kind];
        }
    }
}

/// Standard-route mask by global concrete target enumeration.
///
/// This shares no per-suit table or transition with the production DP or with
/// [`reference_standard_mask`]. It is affordable for `target_sets <= 2`
/// (`melds.len() >= 2`) and is used there as a second definition oracle.
pub(super) fn reference_standard_mask_direct(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
) -> (u8, u32) {
    let target_sets = 4 - melds.len();
    debug_assert!(
        target_sets <= 2,
        "global enumeration is only used for small target sets"
    );
    let capacity = capacity_by_kind(melds, missing);
    let sets = kind_set_candidates(missing);
    let mut best = (INF, 0u32);
    let mut target = [0u8; TILE_KINDS];
    walk_kind_targets(
        0,
        0,
        target_sets,
        &sets,
        &capacity,
        &mut target,
        &mut |target, placed| {
            if placed != target_sets {
                return;
            }
            for kind in Tile::ALL {
                let index = kind.index() as usize;
                if target[index] + 2 > capacity[index] {
                    continue;
                }
                let mut paired = *target;
                paired[index] += 2;
                let (cost, mask) = kind_target_need(hand, &paired);
                merge_entry(&mut best, cost, mask);
            }
        },
    );
    best
}

/// Full-hand mask by global enumeration plus the seven-pairs enumeration.
pub(super) fn reference_hand_mask_direct(hand: &Hand, melds: &[Meld], missing: Suit) -> (u8, u32) {
    let standard = reference_standard_mask_direct(hand, melds, missing);
    if melds.is_empty() {
        merge_route(standard, reference_seven_pairs_mask(hand.counts(), missing))
    } else {
        standard
    }
}
