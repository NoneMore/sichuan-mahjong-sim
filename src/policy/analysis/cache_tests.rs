//! Differential evidence for the cross-decision masked-suit cache.
//!
//! Two independent references back the assertions:
//!
//! - an uncached call of the unchanged DP kernel through the same cache gate
//!   (`shared: None`), which pins that the cache wiring changes no table;
//! - a concrete-target oracle that enumerates every per-suit multiset of
//!   triplets and sequences plus an optional pair and scores it directly,
//!   without a DP transition. It returns both the needed and the surplus mask,
//!   so a cache round trip cannot silently swap the two kind-specific tables.
//!
//! Coverage: the complete `5 x 2` table for every checked key (hand,
//! capacity, target sets, kind, bound) is compared entry by entry; bounded
//! tables are compared with the exact prefix of the complete table, including
//! equal-cost entries and narrow-then-wide and wide-then-narrow query orders;
//! a one-group context forces collisions and evictions; and evaluator-level
//! runs pin local-first lookup with cross-decision fill for `deficit`,
//! complete and bounded needed masks, surplus masks, and `ukeire`.
//!
//! The oracle enumerates all concrete targets, so the fixtures stay small: a
//! documented hand-picked grid, not an exhaustive sweep of every tile pattern.

use super::{masked_suit_table, ukeire, DistanceEvaluator, MaskTable, INF};
use crate::action::Meld;
use crate::hand::Hand;
use crate::policy::cache::{
    CacheState, HeuristicCache, MaskKind, COMPLETE_BOUND, MINIMUM_BUDGET_BYTES,
};
use crate::seat::Seat;
use crate::tile::{Suit, Tile, TILE_KINDS};

const RANKS: usize = 9;

/// Nine triplets followed by seven sequences, in a fixed order.
fn set_candidates() -> Vec<[u8; RANKS]> {
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

/// Visits every concrete multiset of at most `target_sets` sets.
fn walk_targets(
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
        walk_targets(
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

fn merge(slot: &mut (u8, u32), cost: u8, mask: u32) {
    if cost < slot.0 {
        *slot = (cost, mask);
    } else if cost == slot.0 {
        slot.1 |= mask;
    }
}

/// Definition-level cost, needed mask, and surplus mask of one target.
fn score_target(hand: &[u8; RANKS], target: &[u8; RANKS]) -> (u8, u32, u32) {
    let mut cost = 0u8;
    let mut needed = 0u32;
    let mut surplus = 0u32;
    for rank in 0..RANKS {
        let held = hand[rank];
        let want = target[rank];
        cost = cost.saturating_add(want.saturating_sub(held));
        if want > held {
            needed |= 1 << rank;
        }
        if want < held {
            surplus |= 1 << rank;
        }
    }
    (cost, needed, surplus)
}

/// Concrete-target `(needed, surplus)` tables for one suit.
fn reference_suit_masks(
    hand: &[u8; RANKS],
    capacity: &[u8; RANKS],
    target_sets: usize,
) -> (MaskTable, MaskTable) {
    let sets = set_candidates();
    let mut needed = [[(INF, 0u32); 2]; 5];
    let mut surplus = [[(INF, 0u32); 2]; 5];
    let mut target = [0u8; RANKS];
    walk_targets(
        0,
        0,
        target_sets,
        &sets,
        capacity,
        &mut target,
        &mut |target, placed| {
            for pair_rank in 0..=RANKS {
                let mut counted = *target;
                let mut pair = 0;
                if pair_rank < RANKS {
                    if counted[pair_rank] + 2 > capacity[pair_rank] {
                        continue;
                    }
                    counted[pair_rank] += 2;
                    pair = 1;
                }
                let (cost, needed_mask, surplus_mask) = score_target(hand, &counted);
                merge(&mut needed[placed][pair], cost, needed_mask);
                merge(&mut surplus[placed][pair], cost, surplus_mask);
            }
        },
    );
    (needed, surplus)
}

/// One per-suit cache key with a readable label.
struct SuitCase {
    label: &'static str,
    hand: [u8; RANKS],
    capacity: [u8; RANKS],
    target_sets: usize,
}

fn suit_cases() -> Vec<SuitCase> {
    let full = [4u8; RANKS];
    vec![
        SuitCase {
            label: "empty hand, full capacity",
            hand: [0; RANKS],
            capacity: full,
            target_sets: 4,
        },
        SuitCase {
            label: "zero capacity is unreachable",
            hand: [1, 2, 3, 0, 0, 0, 0, 0, 0],
            capacity: [0; RANKS],
            target_sets: 4,
        },
        SuitCase {
            label: "pair only",
            hand: [2, 0, 0, 0, 0, 0, 0, 0, 0],
            capacity: full,
            target_sets: 0,
        },
        SuitCase {
            label: "single set",
            hand: [1, 2, 3, 0, 0, 0, 0, 0, 0],
            capacity: full,
            target_sets: 1,
        },
        SuitCase {
            label: "four sets",
            hand: [1, 1, 1, 2, 2, 2, 3, 3, 3],
            capacity: full,
            target_sets: 4,
        },
        SuitCase {
            label: "held meld zeroes one rank",
            hand: [4, 1, 2, 3, 4, 0, 0, 0, 0],
            capacity: [0, 4, 4, 4, 4, 4, 4, 4, 4],
            target_sets: 3,
        },
        SuitCase {
            label: "several equal-cost branches",
            hand: [1, 1, 2, 2, 3, 3, 4, 4, 5],
            capacity: full,
            target_sets: 3,
        },
        SuitCase {
            label: "edge sequences",
            hand: [1, 1, 2, 2, 3, 7, 8, 9, 9],
            capacity: full,
            target_sets: 4,
        },
        SuitCase {
            label: "deliberately non-physical capacity",
            hand: [5, 3, 2, 1, 0, 0, 0, 0, 0],
            capacity: [5, 3, 2, 4, 4, 4, 4, 4, 4],
            target_sets: 2,
        },
    ]
}

fn assert_well_formed(table: &MaskTable, context: &str) {
    for (sets, row) in table.iter().enumerate() {
        for (pair, &(cost, mask)) in row.iter().enumerate() {
            assert!(
                mask < 1u32 << RANKS,
                "{context}: suit mask is local to nine ranks at [{sets}][{pair}]: {mask:#x}"
            );
            if cost == INF {
                assert_eq!(
                    mask, 0,
                    "{context}: unreachable entry carries a mask at [{sets}][{pair}]"
                );
            }
        }
    }
}

/// Runs the production cache gate for one key and runtime-selected kind/bound.
fn lookup(cache: &HeuristicCache, case: &SuitCase, kind: MaskKind, bound: u8) -> MaskTable {
    match (kind, bound) {
        (MaskKind::Needed, COMPLETE_BOUND) => masked_suit_table::<false, false>(
            &case.hand,
            &case.capacity,
            case.target_sets,
            INF,
            Some(cache),
        ),
        (MaskKind::Surplus, COMPLETE_BOUND) => masked_suit_table::<true, false>(
            &case.hand,
            &case.capacity,
            case.target_sets,
            INF,
            Some(cache),
        ),
        (MaskKind::Needed, cutoff) => masked_suit_table::<false, true>(
            &case.hand,
            &case.capacity,
            case.target_sets,
            cutoff,
            Some(cache),
        ),
        (MaskKind::Surplus, _) => panic!("surplus tables are never bounded"),
    }
}

#[test]
fn cached_tables_match_the_uncached_dp_and_the_concrete_oracle() {
    let cache = HeuristicCache::new(64 * MINIMUM_BUDGET_BYTES).expect("valid budget");
    for case in suit_cases() {
        let (expected_needed, expected_surplus) =
            reference_suit_masks(&case.hand, &case.capacity, case.target_sets);
        assert_well_formed(&expected_needed, case.label);
        assert_well_formed(&expected_surplus, case.label);
        for (sets, row) in expected_needed.iter().enumerate() {
            for (pair, entry) in row.iter().enumerate() {
                assert_eq!(
                    entry.0, expected_surplus[sets][pair].0,
                    "{}: needed and surplus costs must agree",
                    case.label
                );
            }
        }

        let uncached_needed = masked_suit_table::<false, false>(
            &case.hand,
            &case.capacity,
            case.target_sets,
            INF,
            None,
        );
        let uncached_surplus = masked_suit_table::<true, false>(
            &case.hand,
            &case.capacity,
            case.target_sets,
            INF,
            None,
        );
        for (kind, actual, expected) in [
            ("needed", uncached_needed, expected_needed),
            ("surplus", uncached_surplus, expected_surplus),
        ] {
            for (sets, row) in expected.iter().enumerate() {
                for (pair, &entry) in row.iter().enumerate() {
                    assert_eq!(
                        actual[sets][pair], entry,
                        "{} {kind}: DP and concrete target disagree at [{sets}][{pair}], \
                         hand {:?}, capacity {:?}, target_sets {}",
                        case.label, case.hand, case.capacity, case.target_sets
                    );
                }
            }
        }

        // The first cached query builds and stores; the second must return the
        // stored copy. Both must equal the unchanged DP exactly.
        for (kind, uncached, label) in [
            (MaskKind::Needed, uncached_needed, "needed"),
            (MaskKind::Surplus, uncached_surplus, "surplus"),
        ] {
            let misses_before = cache.status().misses;
            let built = lookup(&cache, &case, kind, COMPLETE_BOUND);
            assert_eq!(
                cache.status().misses,
                misses_before + 1,
                "{} {label}: the first query is a miss",
                case.label
            );
            let hits_before = cache.status().hits;
            let replayed = lookup(&cache, &case, kind, COMPLETE_BOUND);
            assert_eq!(
                cache.status().hits,
                hits_before + 1,
                "{} {label}: the second query hits",
                case.label
            );
            for (sets, row) in uncached.iter().enumerate() {
                for (pair, &entry) in row.iter().enumerate() {
                    assert_eq!(
                        built[sets][pair], entry,
                        "{} {label}: built table differs at [{sets}][{pair}]",
                        case.label
                    );
                    assert_eq!(
                        replayed[sets][pair], entry,
                        "{} {label}: replayed table differs at [{sets}][{pair}]",
                        case.label
                    );
                }
            }
        }
    }
}

#[test]
fn bounded_tables_are_exact_prefixes_in_either_query_order() {
    let cases = suit_cases();
    // Narrow then wide must not let a narrower stored table answer a wider
    // query, and wide then narrow must not do the reverse.
    for ascending in [true, false] {
        let cache = HeuristicCache::new(64 * MINIMUM_BUDGET_BYTES).expect("valid budget");
        let order: Vec<u8> = if ascending {
            (0..=8).collect()
        } else {
            (0..=8).rev().collect()
        };
        for case in &cases {
            let complete = masked_suit_table::<false, false>(
                &case.hand,
                &case.capacity,
                case.target_sets,
                INF,
                None,
            );
            for &cutoff in &order {
                let built = lookup(&cache, case, MaskKind::Needed, cutoff);
                for (sets, row) in complete.iter().enumerate() {
                    for (pair, &(cost, mask)) in row.iter().enumerate() {
                        let expected = if cost <= cutoff {
                            (cost, mask)
                        } else {
                            (INF, 0)
                        };
                        assert_eq!(
                            built[sets][pair], expected,
                            "{}: cutoff {cutoff} at [{sets}][{pair}] (ascending {ascending})",
                            case.label
                        );
                    }
                }
            }
            // A second pass over the same bounds must hit the stored entries
            // and reproduce every table exactly.
            for &cutoff in &order {
                let hits_before = cache.status().hits;
                let replayed = lookup(&cache, case, MaskKind::Needed, cutoff);
                assert_eq!(
                    cache.status().hits,
                    hits_before + 1,
                    "{}: cutoff {cutoff} must be an exact-key hit",
                    case.label
                );
                let expected = masked_suit_table::<false, true>(
                    &case.hand,
                    &case.capacity,
                    case.target_sets,
                    cutoff,
                    None,
                );
                assert_eq!(replayed, expected, "{}: cutoff {cutoff} replay", case.label);
            }
        }
    }
}

#[test]
fn a_full_single_group_stays_exact_under_collisions_and_eviction() {
    // One group: every key collides, and only four entries can stay resident.
    let cache = HeuristicCache::new(MINIMUM_BUDGET_BYTES).expect("one group");
    let cases = suit_cases();
    let expected: Vec<MaskTable> = cases
        .iter()
        .map(|case| {
            masked_suit_table::<false, false>(
                &case.hand,
                &case.capacity,
                case.target_sets,
                INF,
                None,
            )
        })
        .collect();

    for (case, expected) in cases.iter().zip(&expected) {
        let table = lookup(&cache, case, MaskKind::Needed, COMPLETE_BOUND);
        assert_eq!(
            &table, expected,
            "{}: exact table despite collision",
            case.label
        );
    }
    let status = cache.status();
    assert_eq!(status.state, CacheState::Ready);
    assert_eq!(status.occupied_entries, 4, "one group holds four entries");
    assert_eq!(
        status.evictions,
        cases.len() as u64 - 4,
        "every further distinct key replaces one resident entry"
    );
    assert_eq!(status.capacity_entries, 4);

    // Re-querying every key after all evictions must still return exact
    // tables, whether the query hits or rebuilds.
    for (case, expected) in cases.iter().zip(&expected) {
        let table = lookup(&cache, case, MaskKind::Needed, COMPLETE_BOUND);
        assert_eq!(&table, expected, "{}: exact after eviction", case.label);
    }
}

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

fn hand_with(wan: &[u8], tong: &[u8], tiao: &[u8]) -> Hand {
    let mut hand = Hand::new();
    for (suit, ranks) in [(Suit::Wan, wan), (Suit::Tong, tong), (Suit::Tiao, tiao)] {
        for &rank in ranks {
            hand.add(tile(suit, rank)).expect("at most four copies");
        }
    }
    hand
}

/// One analysis fixture: a full concealed hand, fixed melds, and a miss suit.
struct AnalysisState {
    label: &'static str,
    hand: Hand,
    melds: Vec<Meld>,
    missing: Suit,
}

fn analysis_states() -> Vec<AnalysisState> {
    let two_melds = vec![
        Meld::peng(tile(Suit::Tong, 5), Seat::new(1).expect("seat")),
        Meld::open_kong(tile(Suit::Tiao, 2), Seat::new(2).expect("seat")),
    ];
    vec![
        AnalysisState {
            label: "no meld, two-suit shape",
            hand: hand_with(&[1, 1, 2, 2, 3, 3, 4, 5, 6], &[1, 7, 8, 9, 9], &[]),
            melds: Vec::new(),
            missing: Suit::Tiao,
        },
        AnalysisState {
            label: "one meld",
            hand: hand_with(&[1, 1, 2, 3, 4, 5, 6, 9], &[7, 8, 9], &[]),
            melds: vec![Meld::peng(tile(Suit::Tong, 5), Seat::new(1).expect("seat"))],
            missing: Suit::Tiao,
        },
        AnalysisState {
            label: "seven pairs shape",
            hand: hand_with(&[1, 1, 3, 3, 5, 5, 7, 9], &[2, 2, 4, 4, 6, 8], &[]),
            melds: Vec::new(),
            missing: Suit::Tiao,
        },
        AnalysisState {
            label: "two melds",
            hand: hand_with(&[], &[1, 2, 3, 4], &[1, 3, 4, 5]),
            melds: two_melds,
            missing: Suit::Wan,
        },
        AnalysisState {
            label: "missing suit held",
            hand: hand_with(&[1, 2, 3, 4, 5], &[1, 2, 3, 4], &[3, 4, 5, 6, 7]),
            melds: Vec::new(),
            missing: Suit::Tong,
        },
    ]
}

/// Every query the analysis exposes, in a fixed order.
///
/// The public vector is all zeros: it only limits unseen copies, and this
/// digest targets the structural tables, which are independent of public
/// ownership.
fn analysis_digest(evaluator: &mut DistanceEvaluator<'_>, state: &AnalysisState) -> Vec<(u8, u32)> {
    let public = [0u8; TILE_KINDS];
    let mut digest = Vec::new();
    digest.push(evaluator.deficit_with_surplus(&state.hand));
    digest.push(evaluator.deficit_with_mask(&state.hand));

    let mut candidates: Vec<Tile> = state.hand.distinct_tiles().collect();
    candidates.sort_by_key(|tile| tile.index());
    for tile in candidates {
        let mut after = state.hand;
        after.remove(tile).expect("candidate is held");
        let deficit = evaluator.deficit(&after);
        digest.push((deficit, 0));
        digest.push(evaluator.deficit_with_mask_bounded(&after, deficit));
        let unseen = ukeire(evaluator, &after, &public, deficit);
        digest.push((deficit, unseen));
    }
    digest
}

/// Runs every fixture for `rounds` rounds, keeping each evaluator alive across
/// rounds so a repeated query sees the same decision-local caches.
fn digest_rounds(
    states: &[AnalysisState],
    cache: Option<&HeuristicCache>,
    rounds: usize,
) -> Vec<Vec<Vec<(u8, u32)>>> {
    let mut results = vec![Vec::new(); rounds];
    for state in states {
        let mut evaluator = match cache {
            Some(cache) => DistanceEvaluator::with_cache(&state.melds, state.missing, cache),
            None => DistanceEvaluator::new(&state.melds, state.missing),
        }
        .expect("at most four melds");
        for round in results.iter_mut() {
            round.push(analysis_digest(&mut evaluator, state));
        }
    }
    results
}

/// One uncached digest per fixture.
fn digest_run(states: &[AnalysisState], cache: Option<&HeuristicCache>) -> Vec<Vec<(u8, u32)>> {
    digest_rounds(states, cache, 1)
        .pop()
        .expect("one round was requested")
}

#[test]
fn evaluator_queries_reuse_the_cache_without_changing_results() {
    let states = analysis_states();
    let uncached = digest_run(&states, None);

    let cache = HeuristicCache::new(4 * MINIMUM_BUDGET_BYTES).expect("valid budget");
    assert_eq!(cache.status().state, CacheState::Unallocated);
    let rounds = digest_rounds(&states, Some(&cache), 2);
    let after_first = cache.status();
    for (index, state) in states.iter().enumerate() {
        assert_eq!(
            rounds[0][index], uncached[index],
            "{}: cached queries must match the uncached DP",
            state.label
        );
        assert_eq!(
            rounds[1][index], uncached[index],
            "{}: a local-hit replay must match the uncached DP",
            state.label
        );
    }
    assert_eq!(after_first.state, CacheState::Ready);
    assert!(after_first.misses > 0, "the first run builds tables");
    assert!(after_first.occupied_entries > 0);

    // Repeating the same queries in the same evaluator is answered by the
    // decision-local caches: the shared context must not be consulted again
    // (local first).
    assert_eq!(
        cache.status().hits,
        after_first.hits,
        "a local hit must not reach the shared context"
    );
    assert_eq!(
        cache.status().misses,
        after_first.misses,
        "a local hit must not reach the shared context"
    );

    // A fresh evaluator set on the same context starts with empty local caches
    // and reuses the stored tables across decisions.
    let third = digest_run(&states, Some(&cache));
    assert_eq!(third, uncached);
    assert!(
        cache.status().hits > after_first.hits,
        "a new evaluator must reuse stored tables"
    );
}

#[test]
fn disabled_and_failed_caches_keep_the_uncached_results() {
    let states = analysis_states();
    let uncached = digest_run(&states, None);

    let disabled = HeuristicCache::new(0).expect("zero disables");
    assert_eq!(digest_run(&states, Some(&disabled)), uncached);
    let status = disabled.status();
    assert_eq!(status.state, CacheState::Disabled);
    assert_eq!(status.allocated_bytes, 0);
    assert_eq!(status.occupied_entries, 0);
    assert_eq!(status.hits, 0);

    let failed = HeuristicCache::new_with_allocation_failure(4 * MINIMUM_BUDGET_BYTES);
    assert_eq!(digest_run(&states, Some(&failed)), uncached);
    let status = failed.status();
    assert_eq!(status.state, CacheState::AllocationFailed);
    assert_eq!(status.allocated_bytes, 0);
    assert_eq!(status.hits, 0);
    assert!(status.misses > 0, "failed lookups are attempts, not hits");

    // The original computation continues and never retries the allocation.
    let misses_after_first_run = status.misses;
    assert_eq!(digest_run(&states, Some(&failed)), uncached);
    let status = failed.status();
    assert_eq!(status.state, CacheState::AllocationFailed);
    assert_eq!(status.allocated_bytes, 0);
    assert!(
        status.misses > misses_after_first_run,
        "later lookups still attempt the cache gate and fail closed"
    );
}
