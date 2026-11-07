//! Shared heuristic-cache wiring for the example tools.
//!
//! This module owns the examples' only mapping from the public cache API to the
//! flat `key: value` report vocabulary. It never prints: callers decide when a
//! degraded context is reported and keep their own output stream, so a library
//! that allocates no cache still produces no diagnostics on its own.
//!
//! One [`SharedCache`] is one explicit context lifetime. Cloning its
//! [`HeuristicCache`] handle into seat policies shares that context, and the
//! last clone releases it. Nothing here creates a thread-local or global cache.

use std::fmt;

use sichuan_mahjong_sim::policy::{
    CacheConfigError, CacheState, CacheStatus, HeuristicCache, HeuristicPolicy,
};
use sichuan_mahjong_sim::PLAYER_COUNT;

/// Plain snapshot of the public cache status.
///
/// The examples report these values rather than the original structure so the
/// text protocol stays independent of internal layout details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheSnapshot {
    /// `disabled`, `unallocated`, `ready`, or `allocation-failed`.
    pub state: &'static str,
    /// Configured byte budget, including zero for an actively disabled cache.
    pub budget_bytes: usize,
    /// Requested bytes of the one shared context allocation.
    ///
    /// This is constant for a context in every state and excludes a per-handle
    /// value and allocator-internal rounding.
    pub context_overhead_bytes: usize,
    /// Bytes actually assigned to cache tables.
    pub allocated_bytes: usize,
    /// Entry slots actually available; zero until allocated.
    pub capacity_entries: usize,
    /// Entry slots currently holding a key.
    pub occupied_entries: usize,
    /// Entries replaced after a group filled.
    pub evictions: u64,
    /// Cross-decision lookups served from the table.
    pub hits: u64,
    /// Cross-decision lookups that found no entry.
    pub misses: u64,
}

impl CacheSnapshot {
    /// Captures the current public status.
    pub fn capture(status: &CacheStatus) -> Self {
        Self {
            state: state_name(status.state),
            budget_bytes: status.budget_bytes,
            context_overhead_bytes: status.context_overhead_bytes,
            allocated_bytes: status.allocated_bytes,
            capacity_entries: status.capacity_entries,
            occupied_entries: status.occupied_entries,
            evictions: status.evictions,
            hits: status.hits,
            misses: status.misses,
        }
    }

    /// Whether the context was created with an explicit zero budget.
    // Each example is a separate crate and uses a subset of this module; the
    // shared item is not dead code across the repository.
    #[allow(dead_code)]
    pub fn is_disabled(self) -> bool {
        self.state == "disabled"
    }

    /// Whether table allocation failed and this context no longer grows.
    pub fn is_allocation_failed(self) -> bool {
        self.state == "allocation-failed"
    }

    /// Whether the table has finished its one lazy allocation.
    #[allow(dead_code)]
    pub fn is_allocated(self) -> bool {
        self.state == "ready"
    }

    /// Whether the table holds every entry its replacement groups can hold.
    ///
    /// A long measurement is capacity-exercising once this is true together
    /// with a positive eviction count; the two facts are reported separately
    /// and never inferred from elapsed time.
    #[allow(dead_code)]
    pub fn is_at_capacity(self) -> bool {
        self.capacity_entries > 0 && self.occupied_entries >= self.capacity_entries
    }
}

/// Stable report name for a public cache state.
fn state_name(state: CacheState) -> &'static str {
    match state {
        CacheState::Disabled => "disabled",
        CacheState::Unallocated => "unallocated",
        CacheState::Ready => "ready",
        CacheState::AllocationFailed => "allocation-failed",
    }
}

/// One explicitly owned cache context with four-seat policy construction.
pub struct SharedCache {
    cache: HeuristicCache,
}

impl SharedCache {
    /// Creates a context with the requested byte budget; zero disables caching.
    pub fn new(budget_bytes: usize) -> Result<Self, CacheConfigError> {
        HeuristicCache::new(budget_bytes).map(|cache| Self { cache })
    }

    /// Builds one policy seat bound to this context.
    ///
    /// The clone shares the context; it does not copy tables or reservation
    /// state.
    pub fn policy(&self) -> HeuristicPolicy {
        HeuristicPolicy::with_cache(self.cache.clone())
    }

    /// Builds the four-seat policy group that shares this context.
    #[allow(dead_code)]
    pub fn group(&self) -> [HeuristicPolicy; PLAYER_COUNT] {
        std::array::from_fn(|_| self.policy())
    }

    /// Current status snapshot.
    pub fn snapshot(&self) -> CacheSnapshot {
        CacheSnapshot::capture(&self.cache.status())
    }
}

impl fmt::Debug for SharedCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedCache")
            .field("status", &self.snapshot())
            .finish()
    }
}

/// Emits at most one allocation-failure warning for one context lifetime.
///
/// A cold benchmark run may own two lifetimes (separate warmup and measured
/// contexts), so each lifetime gets its own warner. The library itself never
/// prints; callers forward the returned message exactly once.
#[derive(Clone, Debug, Default)]
pub struct CacheWarnings {
    warned: bool,
}

impl CacheWarnings {
    /// Checks `snapshot` and returns a warning only on the first failure.
    pub fn observe(&mut self, snapshot: CacheSnapshot, scope: &str) -> Option<String> {
        if self.warned || !snapshot.is_allocation_failed() {
            return None;
        }
        self.warned = true;
        Some(format!(
            "{scope}: heuristic cache allocation failed; caching is disabled for this context \
             and the run continues"
        ))
    }
}

/// Renders the stable status lines under `prefix`.
///
/// The caller chooses the prefix (for example `cache_before_measured`), so
/// snapshots from different lifecycle phases remain distinguishable. Counters
/// are cumulative for the context lifetime.
#[allow(dead_code)]
pub fn snapshot_lines(prefix: &str, snapshot: CacheSnapshot) -> String {
    format!(
        "{prefix}_state: {}\n\
         {prefix}_budget_bytes: {}\n\
         {prefix}_context_overhead_bytes: {}\n\
         {prefix}_allocated_bytes: {}\n\
         {prefix}_capacity_entries: {}\n\
         {prefix}_occupied_entries: {}\n\
         {prefix}_evictions: {}\n\
         {prefix}_hits: {}\n\
         {prefix}_misses: {}\n",
        snapshot.state,
        snapshot.budget_bytes,
        snapshot.context_overhead_bytes,
        snapshot.allocated_bytes,
        snapshot.capacity_entries,
        snapshot.occupied_entries,
        snapshot.evictions,
        snapshot.hits,
        snapshot.misses
    )
}
