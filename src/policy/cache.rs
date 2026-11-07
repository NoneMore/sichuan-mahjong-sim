//! Explicit, byte-bounded cross-decision cache for masked suit tables.
//!
//! [`HeuristicPolicy`](super::HeuristicPolicy) keeps its decision-local caches
//! for fast repeated queries. This module adds an optional context that
//! survives a single decision, game, or runner so exact masked suit tables can
//! be reused across decisions and seats. The context is explicit: nothing is
//! shared through TLS or a global, and binding is done with
//! [`HeuristicPolicy::with_cache`](super::HeuristicPolicy::with_cache) or
//! [`HeuristicPolicy::shared_group`](super::HeuristicPolicy::shared_group).
//!
//! # Key and value
//!
//! One cached value is the complete masked per-suit table
//! `[[(cost, local_mask); 2]; 5]` that the analysis kernel computes for a local
//! nine-rank hand. The key owns every input that can change that table:
//!
//! ```text
//! (hand[9], target_capacity[9], target_sets, Needed/Surplus, bound)
//! ```
//!
//! The suit index is omitted because the kernel is suit-local and the mask is
//! shifted into the global tile index only during the combination pass.
//! `target_capacity` holds the per-rank target capacity after declared melds
//! and the missing suit, so it fixes both the four-copy limit and the missing
//! suit for this suit. `bound` is the caller's exact full-hand deficit for a
//! truncated needed table, or [`COMPLETE_BOUND`] for a complete table. Keys are
//! matched completely: a query with any different field never aliases a stored
//! table, even if the two tables happen to be numerically equal.
//!
//! A value keeps, for every `[sets][pair]` slot, the minimum cost and the union
//! of local nine-bit marks over every equal-cost optimal partial target. An
//! unreachable slot is `(255, 0)`. The encoding is lossless: a retrieved table
//! is `Copy`, and the query-time cutoff filter of the analysis works on that
//! copy, never on the stored entry.
//!
//! # Budget and layout
//!
//! A positive budget is an upper bound on the table allocation in bytes. The
//! fixed context overhead (the one shared `Arc` block: counters, lock, inline
//! table state, and vector header) is reported separately as
//! [`CacheStatus::context_overhead_bytes`] and is never charged to the budget.
//! The layout is one flat array of replacement groups:
//!
//! ```text
//! Entry    = key (hand[9] + capacity[9] + target_sets + bound + kind)  (21 bytes)
//!          + cost[5][2]                                                (10 bytes)
//!          + mask[5][2]                                                (40 bytes)
//! Group    = 4 entries + next: u8   (four-way set, local replacement cursor)
//! groups   = budget / size_of::<Group>()
//! allocated_bytes = table_capacity * size_of::<Group>()  <= budget
//! capacity_entries = table_capacity * 4
//! ```
//!
//! `size_of` values include the alignment padding of these concrete types and
//! are pinned by a unit test. The table is one `Vec` buffer kept for the
//! context lifetime; its length is raised to its actual capacity so every
//! reserved group is usable. `allocated_bytes` therefore accounts the real
//! capacity of that single heap allocation, and a capacity that would exceed
//! the budget is rejected as an allocation failure rather than reported over
//! budget. Allocator-internal rounding is not attributed to the cache. There is
//! no stored index: the group number is the key hash modulo the group count.
//!
//! # Lifecycle
//!
//! - Budget zero disables the context: no table is ever allocated and no
//!   cross-decision entry is added, while the decision-local caches of the
//!   analysis keep working.
//! - A positive budget allocates its table lazily on the first cross-decision
//!   lookup that needs it, once per context. The setup is exactly one fallible
//!   allocation request (the vector buffer); the context, the lock, and the
//!   table state are inline in the shared block, so no later step can abort or
//!   request another allocation. A failed allocation disables new entries for
//!   that context permanently; no retry happens and the original computation
//!   continues.
//! - A full group replaces one entry cyclically and advances only that group's
//!   replacement cursor. Hits never reorder entries. The table never grows and
//!   is never cleared as a whole.
//! - [`HeuristicCache::clone`] shares the table, the lazy-initialization state,
//!   and the degradation state with its source; the last holder frees the
//!   table. The cache is [`Send`] and [`Sync`] but never enters a game
//!   checkpoint: it is derived memoization, so a fresh policy resumes a cloned
//!   game exactly.
//!
//! # Default budget
//!
//! [`DEFAULT_BUDGET_BYTES`] is 26.5 MiB, selected against the 53 MiB tier by
//! paired native batch measurements. The retained layout is four-way local
//! round-robin, not a claim of an optimal cache design. Initialization and
//! memory have a cost: very short runs can be slower; use budget zero when
//! cross-decision reuse is not worth that cost.

use std::error::Error;
use std::fmt;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::analysis::MaskTable;

/// Default cross-decision byte budget.
///
/// 26.5 MiB was selected by paired batch measurements against the 53 MiB
/// tier. This is a per-context limit, not a process-wide memory limit; short
/// cold runs can be slower than a disabled cache.
pub const DEFAULT_BUDGET_BYTES: usize = 26 * 1024 * 1024 + 512 * 1024;

/// Table slots per stored value: five set counts times two pair states.
const TABLE_SLOTS: usize = 10;

/// Entries per replacement group.
const GROUP_WAYS: usize = 4;

/// `target_sets` marker of a vacant entry.
///
/// Real keys only carry `0..=4`, so this value can never collide with one.
const VACANT_TARGET_SETS: u8 = u8::MAX;

/// Bound marker of a complete (unbounded) masked suit table.
pub(crate) const COMPLETE_BOUND: u8 = u8::MAX;

/// Which optimal-branch union a stored table holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MaskKind {
    /// Union of ranks that some optimal target still needs.
    Needed,
    /// Union of ranks that some optimal target holds fewer of.
    Surplus,
}

impl MaskKind {
    const fn code(self) -> u8 {
        match self {
            Self::Needed => 0,
            Self::Surplus => 1,
        }
    }
}

/// Availability of the cross-decision table of one [`HeuristicCache`].
///
/// See the module docs for the transitions between the variants. A library
/// never prints; a command-line tool may warn once per context that ends in
/// [`Self::AllocationFailed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheState {
    /// Budget zero: new table allocations stay at zero.
    Disabled,
    /// Positive budget; the table is allocated on the first cross-decision
    /// lookup, once per context.
    Unallocated,
    /// The table is allocated and accepts new entries.
    Ready,
    /// The single lazy allocation attempt failed. New entries are not cached
    /// and no retry happens; analysis continues with its own caches.
    AllocationFailed,
}

/// Snapshot of one [`HeuristicCache`] context.
///
/// The value is a copy: later cache activity does not change a previously
/// returned status. `hits`/`misses` count cross-decision lookups only; local
/// analysis caches and the disabled context are not counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheStatus {
    /// Availability of the cross-decision table.
    pub state: CacheState,
    /// Configured byte budget (zero means disabled).
    pub budget_bytes: usize,
    /// Requested bytes of the one shared context allocation, reported
    /// separately from the byte-budgeted table. It covers the `Arc` counters
    /// and the inline context payload (lock, table state, vector header, and
    /// any build-specific test hook), not a per-handle value and not
    /// allocator-internal rounding. It is constant for a context in every
    /// state.
    pub context_overhead_bytes: usize,
    /// Bytes actually allocated for the table buffer; zero until allocated.
    pub allocated_bytes: usize,
    /// Entry slots actually available; zero until allocated.
    pub capacity_entries: usize,
    /// Entry slots currently holding a key.
    pub occupied_entries: usize,
    /// Entries replaced because their group was full.
    pub evictions: u64,
    /// Cross-decision lookups served from the table.
    pub hits: u64,
    /// Cross-decision lookups that found no entry.
    pub misses: u64,
}

/// Rejected [`HeuristicCache::new`] budget.
///
/// A zero budget is valid and disables the cache. Every other budget must be
/// large enough for one replacement group and small enough to be addressable
/// on the platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheConfigError {
    /// A non-zero budget is below the minimum table allocation.
    BudgetTooSmall {
        /// Rejected budget.
        budget_bytes: usize,
        /// Smallest accepted non-zero budget.
        minimum_bytes: usize,
    },
    /// The budget cannot be addressed as one table allocation.
    BudgetTooLarge {
        /// Rejected budget.
        budget_bytes: usize,
        /// Largest accepted budget.
        maximum_bytes: usize,
    },
}

impl fmt::Display for CacheConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetTooSmall {
                budget_bytes,
                minimum_bytes,
            } => write!(
                formatter,
                "cache budget of {budget_bytes} bytes is below the minimum of \
                 {minimum_bytes} bytes needed for one replacement group; use 0 \
                 to disable the cache"
            ),
            Self::BudgetTooLarge {
                budget_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "cache budget of {budget_bytes} bytes exceeds the addressable \
                 maximum of {maximum_bytes} bytes"
            ),
        }
    }
}

impl Error for CacheConfigError {}

/// Owned, complete key of one masked per-suit table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SuitKey {
    hand: [u8; 9],
    capacity: [u8; 9],
    target_sets: u8,
    bound: u8,
    kind: u8,
}

impl SuitKey {
    fn new(
        hand: &[u8; 9],
        capacity: &[u8; 9],
        target_sets: usize,
        kind: MaskKind,
        bound: u8,
    ) -> Self {
        debug_assert!(
            target_sets <= 4,
            "a standard target always has at most four concealed sets"
        );
        Self {
            hand: *hand,
            capacity: *capacity,
            target_sets: target_sets as u8,
            bound,
            kind: kind.code(),
        }
    }

    const fn is_vacant(&self) -> bool {
        self.target_sets == VACANT_TARGET_SETS
    }

    /// FNV-1a over every key byte.
    ///
    /// Equal keys always land in the same group. The hash is deterministic so
    /// cache contents never influence a decision, only which exact table a
    /// lookup can reuse.
    fn hash(&self) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01B3;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &byte in self.hand.iter().chain(self.capacity.iter()) {
            hash = (hash ^ byte as u64).wrapping_mul(PRIME);
        }
        for byte in [self.target_sets, self.bound, self.kind] {
            hash = (hash ^ byte as u64).wrapping_mul(PRIME);
        }
        hash
    }
}

/// One stored `5 x 2` masked suit table.
#[derive(Clone, Copy)]
struct Entry {
    key: SuitKey,
    cost: [u8; TABLE_SLOTS],
    mask: [u32; TABLE_SLOTS],
}

const VACANT_ENTRY: Entry = Entry {
    key: SuitKey {
        hand: [0; 9],
        capacity: [0; 9],
        target_sets: VACANT_TARGET_SETS,
        bound: 0,
        kind: 0,
    },
    cost: [u8::MAX; TABLE_SLOTS],
    mask: [0; TABLE_SLOTS],
};

impl Entry {
    fn is_vacant(&self) -> bool {
        self.key.is_vacant()
    }

    fn encode(key: SuitKey, table: &MaskTable) -> Self {
        let mut cost = [u8::MAX; TABLE_SLOTS];
        let mut mask = [0u32; TABLE_SLOTS];
        // Row-major `[sets][pair]` order is exactly the flat slot order.
        for (slot, &(entry_cost, entry_mask)) in table.iter().flatten().enumerate() {
            cost[slot] = entry_cost;
            mask[slot] = entry_mask;
        }
        Self { key, cost, mask }
    }

    fn decode(&self) -> MaskTable {
        let mut table = [[(u8::MAX, 0u32); 2]; 5];
        for (slot, entry) in table.iter_mut().flatten().enumerate() {
            *entry = (self.cost[slot], self.mask[slot]);
        }
        table
    }
}

/// Four entries plus this group's local replacement cursor.
#[derive(Clone, Copy)]
struct Group {
    entries: [Entry; GROUP_WAYS],
    next: u8,
}

const VACANT_GROUP: Group = Group {
    entries: [VACANT_ENTRY; GROUP_WAYS],
    next: 0,
};

const ENTRY_BYTES: usize = std::mem::size_of::<Entry>();
const GROUP_BYTES: usize = std::mem::size_of::<Group>();

// One group allocation covers its four entries plus the replacement cursor.
const _: () = assert!(GROUP_WAYS * ENTRY_BYTES < GROUP_BYTES);

/// Smallest accepted non-zero budget: one full replacement group.
pub(crate) const MINIMUM_BUDGET_BYTES: usize = GROUP_BYTES;

/// Largest budget that one `Vec` allocation can address.
const MAXIMUM_BUDGET_BYTES: usize = isize::MAX as usize;

/// The flat group array of one context.
///
/// The [`Vec`] is kept rather than a boxed slice: converting would request a
/// second, non-fallible shrink allocation, while the vector header is already
/// part of the reported context overhead. `len == capacity`, so every reserved
/// group is usable and the accounted table bytes are exactly
/// `capacity * GROUP_BYTES`.
struct Store {
    groups: Vec<Group>,
    occupied: usize,
    evictions: u64,
}

impl Store {
    /// Reserves and fills the table, or `None` when that would fail or could
    /// exceed `budget_bytes`.
    ///
    /// `try_reserve_exact` is the only allocation request and the only
    /// fallible step; the following `resize` writes into the reserved capacity
    /// without another request. A capacity larger than the budgeted groups is
    /// rejected, so the accounted table bytes never exceed the budget even if
    /// an allocator reports more capacity than requested.
    fn try_new(groups: usize, budget_bytes: usize) -> Option<Self> {
        if groups == 0 {
            return None;
        }
        let mut slots: Vec<Group> = Vec::new();
        slots.try_reserve_exact(groups).ok()?;
        let capacity = slots.capacity();
        let capacity_bytes = capacity.checked_mul(GROUP_BYTES)?;
        if capacity_bytes > budget_bytes {
            return None;
        }
        // Use the whole reserved capacity so the accounted slots are usable.
        slots.resize(capacity, VACANT_GROUP);
        debug_assert_eq!(slots.len(), slots.capacity());
        Some(Self {
            groups: slots,
            occupied: 0,
            evictions: 0,
        })
    }

    /// Accounted table bytes: the single live heap buffer.
    fn capacity_bytes(&self) -> usize {
        self.groups.capacity() * GROUP_BYTES
    }

    /// Entry slots the table can hold.
    fn capacity_entries(&self) -> usize {
        self.groups.capacity() * GROUP_WAYS
    }

    fn group_index(&self, key: &SuitKey) -> usize {
        (key.hash() % self.groups.len() as u64) as usize
    }

    fn get(&self, index: usize, key: &SuitKey) -> Option<MaskTable> {
        let group = self.groups.get(index)?;
        group
            .entries
            .iter()
            .find(|entry| !entry.is_vacant() && entry.key == *key)
            .map(Entry::decode)
    }

    /// Stores `table` under `key`, preferring a vacant slot.
    ///
    /// A full group replaces one entry at its local cursor and advances only
    /// that cursor. An existing equal key is overwritten in place, which can
    /// only happen when two threads missed the same key concurrently.
    fn insert(&mut self, index: usize, key: SuitKey, table: &MaskTable) {
        let group = &mut self.groups[index];
        let entry = Entry::encode(key, table);
        if let Some(existing) = group
            .entries
            .iter_mut()
            .find(|existing| !existing.is_vacant() && existing.key == key)
        {
            *existing = entry;
            return;
        }
        if let Some(vacant) = group.entries.iter_mut().find(|entry| entry.is_vacant()) {
            *vacant = entry;
            self.occupied += 1;
            return;
        }
        let slot = group.next as usize;
        group.entries[slot] = entry;
        group.next = ((slot + 1) % GROUP_WAYS) as u8;
        self.evictions += 1;
    }
}

/// Mutable state behind the context lock.
///
/// The store is inline, so the lazy table setup is exactly one heap allocation
/// (the vector buffer). Boxing it would add a second allocation after the
/// fallible reservation and could abort on an out-of-memory path.
#[derive(Default)]
struct TableState {
    store: Option<Store>,
    allocation_failed: bool,
    hits: u64,
    misses: u64,
}

/// State shared by every clone of one [`HeuristicCache`].
struct CacheInner {
    budget_bytes: usize,
    table: Mutex<TableState>,
    /// Unit-test hook: consume one allocation-failure request.
    #[cfg(test)]
    fail_allocation: AtomicBool,
}

/// Requested bytes of the one shared context allocation.
///
/// The shared block is the `Arc` inner value: two `AtomicUsize` counters and
/// the [`CacheInner`] payload, including the `#[repr(C)]` alignment padding
/// between them. The payload already contains the inline lock and the vector
/// header of the table, but not the table buffer itself (that is
/// [`CacheStatus::allocated_bytes`]). A per-handle [`HeuristicCache`] value and
/// allocator-internal rounding are excluded. The value is computed for the
/// current build configuration, so a `cfg(test)`-only field can change it
/// between a test build and a tool build.
pub(crate) const CONTEXT_OVERHEAD_BYTES: usize = {
    let counters = 2 * std::mem::size_of::<usize>();
    let align = std::mem::align_of::<CacheInner>();
    // Round the counters up to the payload alignment, then append the payload.
    // The manual ceiling division keeps the expression const-compatible on the
    // project's minimum supported Rust version.
    #[allow(clippy::manual_div_ceil)]
    let padded_counters = ((counters + align - 1) / align) * align;
    padded_counters + std::mem::size_of::<CacheInner>()
};

impl CacheInner {
    #[cfg(test)]
    fn allocation_fails(&self) -> bool {
        self.fail_allocation.swap(false, Ordering::Relaxed)
    }

    #[cfg(not(test))]
    fn allocation_fails(&self) -> bool {
        false
    }
}

/// Byte-bounded cross-decision cache for masked suit tables.
///
/// Cheap to clone: every clone shares the table, the lazy-initialization state,
/// and the allocation-failure state. Dropping the last clone frees the table.
/// See the module docs for the contract.
#[derive(Clone)]
pub struct HeuristicCache {
    inner: Arc<CacheInner>,
}

impl HeuristicCache {
    /// Creates a context with a byte budget.
    ///
    /// Zero disables the context. Any other value must be at least large
    /// enough for one replacement group; the actual table allocation never
    /// exceeds the budget. The table is allocated later, on the first
    /// cross-decision lookup.
    pub fn new(budget_bytes: usize) -> Result<Self, CacheConfigError> {
        if budget_bytes != 0 {
            if budget_bytes < MINIMUM_BUDGET_BYTES {
                return Err(CacheConfigError::BudgetTooSmall {
                    budget_bytes,
                    minimum_bytes: MINIMUM_BUDGET_BYTES,
                });
            }
            if budget_bytes > MAXIMUM_BUDGET_BYTES {
                return Err(CacheConfigError::BudgetTooLarge {
                    budget_bytes,
                    maximum_bytes: MAXIMUM_BUDGET_BYTES,
                });
            }
        }
        Ok(Self {
            inner: Arc::new(CacheInner {
                budget_bytes,
                table: Mutex::new(TableState::default()),
                #[cfg(test)]
                fail_allocation: AtomicBool::new(false),
            }),
        })
    }

    /// Snapshot of the current state, allocation, and counters.
    pub fn status(&self) -> CacheStatus {
        let state = self.lock();
        let store = state.store.as_ref();
        let cache_state = if self.inner.budget_bytes == 0 {
            CacheState::Disabled
        } else if store.is_some() {
            CacheState::Ready
        } else if state.allocation_failed {
            CacheState::AllocationFailed
        } else {
            CacheState::Unallocated
        };
        let (allocated_bytes, capacity_entries, occupied_entries, evictions) = match store {
            Some(store) => (
                store.capacity_bytes(),
                store.capacity_entries(),
                store.occupied,
                store.evictions,
            ),
            None => (0, 0, 0, 0),
        };
        CacheStatus {
            state: cache_state,
            budget_bytes: self.inner.budget_bytes,
            context_overhead_bytes: CONTEXT_OVERHEAD_BYTES,
            allocated_bytes,
            capacity_entries,
            occupied_entries,
            evictions,
            hits: state.hits,
            misses: state.misses,
        }
    }

    /// Cross-decision lookup of one masked suit table.
    ///
    /// Returns `None` when the context is disabled, its table is still
    /// unallocated and the one allocation attempt fails, or no exact key is
    /// stored. A hit returns a copy; the stored entry is never modified by the
    /// caller.
    pub(crate) fn lookup_suit(
        &self,
        hand: &[u8; 9],
        capacity: &[u8; 9],
        target_sets: usize,
        kind: MaskKind,
        bound: u8,
    ) -> Option<MaskTable> {
        if self.inner.budget_bytes == 0 {
            return None;
        }
        let key = SuitKey::new(hand, capacity, target_sets, kind, bound);
        let mut state = self.lock();
        if state.store.is_none() && !self.ensure_allocated(&mut state) {
            state.misses += 1;
            return None;
        }
        let found = {
            let store = state.store.as_ref().expect("an allocated table");
            let index = store.group_index(&key);
            store.get(index, &key)
        };
        if found.is_some() {
            state.hits += 1;
        } else {
            state.misses += 1;
        }
        found
    }

    /// Stores one freshly computed masked suit table.
    ///
    /// No-op when the context is disabled, when its lazy allocation already
    /// failed, or when insertion is attempted before any lookup allocated the
    /// table.
    pub(crate) fn insert_suit(
        &self,
        hand: &[u8; 9],
        capacity: &[u8; 9],
        target_sets: usize,
        kind: MaskKind,
        bound: u8,
        table: &MaskTable,
    ) {
        if self.inner.budget_bytes == 0 {
            return;
        }
        let key = SuitKey::new(hand, capacity, target_sets, kind, bound);
        let mut state = self.lock();
        let Some(store) = state.store.as_mut() else {
            return;
        };
        let index = store.group_index(&key);
        store.insert(index, key, table);
    }

    /// Performs the one lazy table allocation.
    fn ensure_allocated(&self, state: &mut TableState) -> bool {
        if state.store.is_some() {
            return true;
        }
        if state.allocation_failed || self.inner.budget_bytes == 0 {
            return false;
        }
        let groups = self.inner.budget_bytes / GROUP_BYTES;
        let store = if self.inner.allocation_fails() {
            None
        } else {
            Store::try_new(groups, self.inner.budget_bytes)
        };
        match store {
            Some(store) => {
                state.store = Some(store);
                true
            }
            None => {
                state.allocation_failed = true;
                false
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, TableState> {
        // The cache is derived memoization: a panicked policy cannot corrupt a
        // stored table into a wrong answer, so a poisoned lock is recovered.
        self.inner
            .table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Unit-test hook: fail the single lazy allocation attempt.
    #[cfg(test)]
    pub(crate) fn new_with_allocation_failure(budget_bytes: usize) -> Self {
        let cache = Self::new(budget_bytes).expect("test budget is valid");
        cache.inner.fail_allocation.store(true, Ordering::Relaxed);
        cache
    }

    /// Unit-test hook: whether two clones share one context.
    #[cfg(test)]
    pub(crate) fn shares_context_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Default for HeuristicCache {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET_BYTES).expect("the default budget is valid")
    }
}

impl fmt::Debug for HeuristicCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = self.status();
        formatter
            .debug_struct("HeuristicCache")
            .field("state", &status.state)
            .field("budget_bytes", &status.budget_bytes)
            .field("allocated_bytes", &status.allocated_bytes)
            .field("occupied_entries", &status.occupied_entries)
            .finish()
    }
}

#[cfg(test)]
mod tests;
