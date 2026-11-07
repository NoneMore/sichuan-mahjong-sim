//! Unit tests for the cross-decision cache core.
//!
//! These cover the byte layout and accounting, the budget lifecycle, sharing,
//! exact key matching, group-local replacement, and the one-shot allocation
//! failure. Analysis-level cache correctness (cached versus uncached and
//! concrete-target tables) lives in `analysis::cache_tests`; public
//! policy/runner behavior lives in `tests/heuristic_policy.rs`.

use super::*;

fn key_with(hand_rank: u8) -> SuitKey {
    let mut hand = [0u8; 9];
    hand[0] = hand_rank;
    SuitKey::new(&hand, &[4u8; 9], 4, MaskKind::Needed, COMPLETE_BOUND)
}

/// Distinct, self-describing table so a wrong entry is visible.
fn table_with(seed: u8) -> MaskTable {
    let mut table = [[(0u8, 0u32); 2]; 5];
    for (sets, row) in table.iter_mut().enumerate() {
        for (pair, entry) in row.iter_mut().enumerate() {
            let slot = (sets * 2 + pair) as u8;
            *entry = (seed.wrapping_add(slot), 1u32 << (seed as u32 % 9));
        }
    }
    table
}

fn kind_of(key: &SuitKey) -> MaskKind {
    match key.kind {
        0 => MaskKind::Needed,
        _ => MaskKind::Surplus,
    }
}

fn lookup(cache: &HeuristicCache, key: SuitKey) -> Option<MaskTable> {
    cache.lookup_suit(
        &key.hand,
        &key.capacity,
        key.target_sets as usize,
        kind_of(&key),
        key.bound,
    )
}

fn insert(cache: &HeuristicCache, key: SuitKey, table: &MaskTable) {
    cache.insert_suit(
        &key.hand,
        &key.capacity,
        key.target_sets as usize,
        kind_of(&key),
        key.bound,
        table,
    );
}

#[test]
fn layout_accounting_covers_every_group_byte() {
    // These concrete sizes are the byte budget's unit. If a layout change
    // moves them, the cache accounting must be reviewed with it.
    assert_eq!(
        std::mem::size_of::<SuitKey>(),
        21,
        "key: hand + capacity + metadata"
    );
    assert_eq!(ENTRY_BYTES, 72, "entry: key + costs + masks with alignment");
    assert_eq!(
        GROUP_BYTES, 292,
        "group: four entries plus replacement cursor"
    );
    assert_eq!(
        GROUP_BYTES,
        GROUP_WAYS * ENTRY_BYTES + 4,
        "cursor plus 3 padding bytes"
    );
    assert_eq!(GROUP_WAYS, 4);
    assert_eq!(TABLE_SLOTS, 5 * 2);
}

/// The context overhead is the shared block request, not a table or a handle.
#[test]
fn context_overhead_accounts_the_shared_block() {
    let counters = 2 * std::mem::size_of::<usize>();
    let payload = std::mem::size_of::<CacheInner>();
    let align = std::mem::align_of::<CacheInner>();
    #[allow(clippy::manual_div_ceil)]
    let padded_counters = ((counters + align - 1) / align) * align;

    assert_eq!(CONTEXT_OVERHEAD_BYTES, padded_counters + payload);
    assert!(
        CONTEXT_OVERHEAD_BYTES >= counters + payload,
        "the counters and payload are both charged"
    );
    assert!(
        CONTEXT_OVERHEAD_BYTES < counters + payload + align,
        "only the payload alignment padding is charged"
    );
    // The table metadata is inline in the context: the vector header must be
    // part of the payload, so no second heap object describes the table.
    assert!(
        payload >= std::mem::size_of::<Store>(),
        "the store must be inline in the shared context"
    );
}

#[test]
fn context_overhead_is_constant_across_states_and_budgets() {
    let disabled = HeuristicCache::new(0).expect("zero disables");
    let unallocated = HeuristicCache::new(4 * GROUP_BYTES).expect("valid budget");
    let ready = HeuristicCache::new(2 * GROUP_BYTES).expect("valid budget");
    let failed = HeuristicCache::new_with_allocation_failure(2 * GROUP_BYTES);
    assert_eq!(lookup(&ready, key_with(11)), None);
    assert_eq!(lookup(&failed, key_with(11)), None);

    for (label, cache, expected_state) in [
        ("disabled", &disabled, CacheState::Disabled),
        ("unallocated", &unallocated, CacheState::Unallocated),
        ("ready", &ready, CacheState::Ready),
        ("failed", &failed, CacheState::AllocationFailed),
    ] {
        let status = cache.status();
        assert_eq!(status.state, expected_state, "{label}: state");
        assert_eq!(
            status.context_overhead_bytes, CONTEXT_OVERHEAD_BYTES,
            "{label}: the context overhead is constant"
        );
    }
    // The table buffer is reported separately and stays inside the budget.
    let status = ready.status();
    assert!(status.allocated_bytes > 0);
    assert!(status.allocated_bytes <= status.budget_bytes);
    assert_eq!(failed.status().allocated_bytes, 0);
    assert_eq!(disabled.status().allocated_bytes, 0);
}

#[test]
fn store_rejects_a_capacity_that_would_exceed_the_budget() {
    // A one-group budget cannot cover a two-group reservation, so the setup
    // must degrade instead of reporting an over-budget table.
    assert!(Store::try_new(2, GROUP_BYTES).is_none());
    assert!(Store::try_new(0, GROUP_BYTES).is_none());

    let store = Store::try_new(1, GROUP_BYTES).expect("one group fits the budget");
    assert_eq!(store.capacity_bytes(), GROUP_BYTES);
    assert_eq!(store.capacity_entries(), GROUP_WAYS);
    assert_eq!(
        store.groups.len(),
        store.groups.capacity(),
        "every reserved group is usable"
    );
}

#[test]
fn zero_budget_disables_the_context_without_allocating() {
    let cache = HeuristicCache::new(0).expect("zero is the disabled budget");
    assert_eq!(
        cache.status(),
        CacheStatus {
            state: CacheState::Disabled,
            budget_bytes: 0,
            context_overhead_bytes: CONTEXT_OVERHEAD_BYTES,
            allocated_bytes: 0,
            capacity_entries: 0,
            occupied_entries: 0,
            evictions: 0,
            hits: 0,
            misses: 0,
        }
    );

    let key = key_with(1);
    insert(&cache, key, &table_with(1));
    assert_eq!(lookup(&cache, key), None);
    let status = cache.status();
    assert_eq!(status.state, CacheState::Disabled);
    assert_eq!(status.allocated_bytes, 0);
    assert_eq!(status.occupied_entries, 0);
    assert_eq!(
        status.hits + status.misses,
        0,
        "a disabled lookup is not counted"
    );
}

#[test]
fn invalid_budgets_are_rejected_with_a_readable_error() {
    match HeuristicCache::new(MINIMUM_BUDGET_BYTES - 1) {
        Err(CacheConfigError::BudgetTooSmall {
            budget_bytes,
            minimum_bytes,
        }) => {
            assert_eq!(budget_bytes, MINIMUM_BUDGET_BYTES - 1);
            assert_eq!(minimum_bytes, MINIMUM_BUDGET_BYTES);
        }
        other => panic!("expected a too-small budget error, got {other:?}"),
    }
    assert!(
        HeuristicCache::new(MINIMUM_BUDGET_BYTES).is_ok(),
        "the minimum budget allocates one full group"
    );
    match HeuristicCache::new(usize::MAX) {
        Err(CacheConfigError::BudgetTooLarge {
            budget_bytes,
            maximum_bytes,
        }) => {
            assert_eq!(budget_bytes, usize::MAX);
            assert_eq!(maximum_bytes, MAXIMUM_BUDGET_BYTES);
        }
        other => panic!("expected a too-large budget error, got {other:?}"),
    }

    let error = HeuristicCache::new(3).expect_err("three bytes cannot hold a group");
    let message = error.to_string();
    assert!(message.contains("minimum"), "readable message: {message}");
    let dynamic: &dyn std::error::Error = &error;
    assert_eq!(dynamic.to_string(), message);
}

#[test]
fn table_is_lazily_allocated_once_and_within_budget() {
    let cache = HeuristicCache::new(4 * GROUP_BYTES).expect("valid budget");
    assert_eq!(cache.status().state, CacheState::Unallocated);
    assert_eq!(cache.status().capacity_entries, 0);
    assert_eq!(cache.status().allocated_bytes, 0);

    let key = key_with(2);
    assert_eq!(lookup(&cache, key), None);
    let status = cache.status();
    assert_eq!(status.state, CacheState::Ready);
    assert_eq!(status.allocated_bytes, 4 * GROUP_BYTES);
    assert_eq!(status.capacity_entries, 4 * GROUP_WAYS);
    assert_eq!(status.misses, 1);
    assert_eq!(status.hits, 0);
    assert_eq!(status.occupied_entries, 0);

    let table = table_with(7);
    insert(&cache, key, &table);
    assert_eq!(lookup(&cache, key), Some(table));
    let status = cache.status();
    assert_eq!(
        status.allocated_bytes,
        4 * GROUP_BYTES,
        "allocation stays fixed"
    );
    assert_eq!(status.occupied_entries, 1);
    assert_eq!(status.hits, 1);
    assert_eq!(status.misses, 1);
}

#[test]
fn budget_is_an_upper_bound_and_the_remainder_is_smaller_than_a_group() {
    for budget in [
        GROUP_BYTES,
        GROUP_BYTES + 1,
        GROUP_BYTES + 1000,
        1 << 20,
        53 * 1024 * 1024,
    ] {
        let cache = HeuristicCache::new(budget).expect("valid budget");
        assert_eq!(lookup(&cache, key_with(3)), None);
        let status = cache.status();
        assert_eq!(status.state, CacheState::Ready, "budget {budget}");
        assert_eq!(status.budget_bytes, budget);
        assert_eq!(
            status.context_overhead_bytes, CONTEXT_OVERHEAD_BYTES,
            "budget {budget}: the context overhead is independent of the budget"
        );
        assert!(
            status.allocated_bytes <= budget,
            "budget {budget}: allocated {} bytes",
            status.allocated_bytes
        );
        assert!(
            budget - status.allocated_bytes < GROUP_BYTES,
            "budget {budget}: unaccounted remainder {}",
            budget - status.allocated_bytes
        );
        assert_eq!(status.allocated_bytes % GROUP_BYTES, 0);
        assert_eq!(
            status.capacity_entries,
            GROUP_WAYS * (status.allocated_bytes / GROUP_BYTES)
        );
        assert_eq!(status.occupied_entries, 0);
    }
}

#[test]
fn every_key_dimension_is_matched_exactly() {
    let cache = HeuristicCache::new(4 * GROUP_BYTES).expect("valid budget");
    let base = key_with(1);
    let table = table_with(5);
    // Allocate through the first lookup, then store the base key.
    assert_eq!(lookup(&cache, base), None);
    insert(&cache, base, &table);
    let before = cache.status();
    assert_eq!(lookup(&cache, base), Some(table), "the owned key must hit");
    assert_eq!(cache.status().hits, before.hits + 1);

    let mut variants = Vec::new();
    let mut hand = base.hand;
    hand[0] += 1;
    variants.push(SuitKey { hand, ..base });
    let mut capacity = base.capacity;
    capacity[0] -= 1;
    variants.push(SuitKey { capacity, ..base });
    variants.push(SuitKey {
        target_sets: 3,
        ..base
    });
    variants.push(SuitKey { kind: 1, ..base });
    variants.push(SuitKey { bound: 5, ..base });
    variants.push(SuitKey {
        hand: {
            let mut changed = base.hand;
            changed[8] = 4;
            changed
        },
        ..base
    });

    let hits = cache.status().hits;
    for variant in variants {
        assert_ne!(variant, base);
        assert_eq!(
            lookup(&cache, variant),
            None,
            "a different key must not alias a stored table: {variant:?}"
        );
        assert_eq!(
            cache.status().hits,
            hits,
            "a different key is a miss: {variant:?}"
        );
    }
    assert_eq!(
        lookup(&cache, base),
        Some(table),
        "variant probes must not disturb the stored entry"
    );
}

#[test]
fn replacement_prefers_vacant_slots_then_rotates_one_group() {
    let cache = HeuristicCache::new(GROUP_BYTES).expect("one group");
    let keys: Vec<SuitKey> = (0..4).map(|rank| key_with(rank + 1)).collect();
    let tables: Vec<MaskTable> = (0..4).map(|rank| table_with(rank + 1)).collect();

    for (key, table) in keys.iter().zip(&tables) {
        // Allocate first so inserts are not dropped.
        assert_eq!(lookup(&cache, *key), None);
        insert(&cache, *key, table);
    }
    let status = cache.status();
    assert_eq!(status.state, CacheState::Ready);
    assert_eq!(status.occupied_entries, GROUP_WAYS);
    assert_eq!(
        status.evictions, 0,
        "vacant slots are used before replacing"
    );

    for (key, table) in keys.iter().zip(&tables) {
        assert_eq!(lookup(&cache, *key), Some(*table));
    }

    // The fifth key finds a full group and replaces the cursor slot, which is
    // slot 0 because vacant-slot fills never advance the cursor.
    let extra_key = key_with(9);
    let extra_table = table_with(9);
    insert(&cache, extra_key, &extra_table);
    let status = cache.status();
    assert_eq!(status.evictions, 1);
    assert_eq!(status.occupied_entries, GROUP_WAYS);
    assert_eq!(
        lookup(&cache, keys[0]),
        None,
        "the cursor slot was replaced"
    );
    for (key, table) in keys.iter().skip(1).zip(tables.iter().skip(1)) {
        assert_eq!(lookup(&cache, *key), Some(*table));
    }
    assert_eq!(lookup(&cache, extra_key), Some(extra_table));

    // The cursor advanced, so the next replacement evicts the second entry.
    let next_key = key_with(10);
    insert(&cache, next_key, &table_with(10));
    assert_eq!(cache.status().evictions, 2);
    assert_eq!(lookup(&cache, keys[0]), None);
    assert_eq!(lookup(&cache, keys[1]), None);
    assert_eq!(lookup(&cache, keys[2]), Some(tables[2]));
}

#[test]
fn clone_shares_the_table_and_the_degradation_state() {
    let cache = HeuristicCache::new(GROUP_BYTES).expect("valid budget");
    let clone = cache.clone();
    assert!(cache.shares_context_with(&clone));
    assert_eq!(cache.status().state, CacheState::Unallocated);

    let key = key_with(4);
    assert_eq!(
        lookup(&clone, key),
        None,
        "the clone drives the one allocation"
    );
    assert_eq!(cache.status().state, CacheState::Ready, "state is shared");
    assert_eq!(cache.status().misses, 1);

    let table = table_with(4);
    insert(&cache, key, &table);
    assert_eq!(lookup(&clone, key), Some(table), "entries are shared");
    assert_eq!(cache.status().hits, 1);

    drop(cache);
    assert_eq!(clone.status().hits, 1, "the clone keeps the table alive");
    assert_eq!(lookup(&clone, key), Some(table));
}

#[test]
fn allocation_failure_disables_new_entries_without_retry() {
    let cache = HeuristicCache::new_with_allocation_failure(4 * GROUP_BYTES);
    assert_eq!(cache.status().state, CacheState::Unallocated);

    let key = key_with(6);
    assert_eq!(lookup(&cache, key), None);
    let failed = cache.status();
    assert_eq!(failed.state, CacheState::AllocationFailed);
    assert_eq!(failed.allocated_bytes, 0);
    assert_eq!(failed.capacity_entries, 0);
    assert_eq!(failed.occupied_entries, 0);
    assert_eq!(failed.misses, 1);

    // A retry would now succeed because the one-shot hook was consumed; the
    // context must not retry.
    assert_eq!(lookup(&cache, key), None);
    assert_eq!(cache.status().state, CacheState::AllocationFailed);
    assert_eq!(cache.status().allocated_bytes, 0);

    // Computed tables are still returned to the caller, but not stored.
    let table = table_with(6);
    insert(&cache, key, &table);
    assert_eq!(cache.status().occupied_entries, 0);
    assert_eq!(lookup(&cache, key), None);

    // A clone shares the degraded state instead of trying again.
    let clone = cache.clone();
    assert_eq!(clone.status().state, CacheState::AllocationFailed);
    assert_eq!(lookup(&clone, key), None);
    assert_eq!(clone.status().allocated_bytes, 0);
}

#[test]
fn status_is_a_snapshot_of_the_moment_it_was_taken() {
    let cache = HeuristicCache::new(GROUP_BYTES).expect("valid budget");
    let before = cache.status();
    let key = key_with(8);
    assert_eq!(lookup(&cache, key), None);
    insert(&cache, key, &table_with(8));
    let after = cache.status();
    assert_eq!(before.state, CacheState::Unallocated);
    assert_eq!(before.allocated_bytes, 0);
    assert_eq!(after.state, CacheState::Ready);
    assert_eq!(after.occupied_entries, 1);
}

#[test]
fn cache_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HeuristicCache>();
    assert_send_sync::<CacheStatus>();
    assert_send_sync::<CacheConfigError>();
}
