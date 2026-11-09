//! Serializable collection outcomes and corpus reports.

use std::time::Instant;

use serde::Serialize;
use sichuan_mahjong_sim::{CacheState, CacheStatus};

use super::format::Episode;
use super::manifest::Manifest;
use super::seed::Seed;
use super::source::{BuildInfo, EnvironmentInfo};
use super::stats::CorpusStats;

pub(super) fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Timing breakdown for one collection invocation, in nanoseconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CollectionTimings {
    /// Whole invocation, including commit work.
    pub total_ns: u64,
    /// `Game::new`, `next_decision`, and `submit`.
    pub rules_ns: u64,
    /// Teacher `choose` calls.
    pub teacher_ns: u64,
    /// Encoding, flattening, and record preparation.
    pub encoding_ns: u64,
    /// JSON serialization and writes to the pending shard.
    pub io_ns: u64,
    /// Shard validation, shard publication, and manifest replacement.
    pub commit_ns: u64,
    /// Games committed by this invocation.
    pub games: u64,
    /// Decisions committed by this invocation.
    pub decisions: u64,
    /// Committed games per second over total invocation time.
    pub games_per_sec: f64,
    /// Committed decisions per second over total invocation time.
    pub decisions_per_sec: f64,
}

impl CollectionTimings {
    pub(super) fn finalize_rates(&mut self) {
        self.games_per_sec = rate(self.games, self.total_ns);
        self.decisions_per_sec = rate(self.decisions, self.total_ns);
    }
}

fn rate(count: u64, elapsed_ns: u64) -> f64 {
    if elapsed_ns == 0 {
        0.0
    } else {
        count as f64 * 1_000_000_000.0 / elapsed_ns as f64
    }
}

/// Plain snapshot of the shared teacher cache.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CacheReport {
    /// `disabled`, `unallocated`, `ready`, or `allocation-failed`.
    pub state: String,
    /// Configured byte budget.
    pub budget_bytes: usize,
    /// Fixed context allocation overhead.
    pub context_overhead_bytes: usize,
    /// Bytes assigned to the cross-decision table.
    pub allocated_bytes: usize,
    /// Entry slots available.
    pub capacity_entries: usize,
    /// Entry slots holding a key.
    pub occupied_entries: usize,
    /// Replaced entries.
    pub evictions: u64,
    /// Cross-decision hits.
    pub hits: u64,
    /// Cross-decision misses.
    pub misses: u64,
}

impl CacheReport {
    /// Captures the public cache status without depending on its representation.
    pub fn capture(status: &CacheStatus) -> Self {
        Self {
            state: match status.state {
                CacheState::Disabled => "disabled",
                CacheState::Unallocated => "unallocated",
                CacheState::Ready => "ready",
                CacheState::AllocationFailed => "allocation-failed",
            }
            .to_owned(),
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
}

/// Result of one successful collection invocation.
///
/// `stats` covers the games committed by this invocation. `manifest` is the
/// full corpus commit record after the run; it is the authority for prior
/// invocations.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CollectionOutcome {
    /// Games requested by the caller.
    pub requested_games: u64,
    /// Games committed by this invocation.
    pub committed_games: u64,
    /// First seed committed by this invocation.
    pub first_seed: Seed,
    /// Last seed committed by this invocation.
    pub last_seed: Seed,
    /// Statistics over only the games committed by this invocation.
    ///
    /// Prior committed games are represented by `manifest`, not repeated here.
    pub committed_stats: CorpusStats,
    /// Full manifest after the invocation.
    pub manifest: Manifest,
    /// Sum of all committed shard file sizes.
    pub shard_bytes: u64,
    /// Manifest file size.
    pub manifest_bytes: u64,
    /// Shard bytes plus manifest bytes.
    pub total_storage_bytes: u64,
    /// Invocation timing breakdown.
    pub timings: CollectionTimings,
    /// Shared teacher cache settings and final counters.
    pub cache: CacheReport,
    /// Build facts for the collecting binary.
    pub build: BuildInfo,
    /// Runtime environment facts.
    pub environment: EnvironmentInfo,
}

/// Strictly validated corpus contents and aggregate report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CorpusReport {
    /// Corpus manifest.
    pub manifest: Manifest,
    /// Statistics over every committed game.
    pub stats: CorpusStats,
    /// Sum of all committed shard file sizes.
    pub shard_bytes: u64,
    /// Manifest file size.
    pub manifest_bytes: u64,
    /// Shard bytes plus manifest bytes.
    pub total_storage_bytes: u64,
}

/// Fully validated corpus episodes.
#[derive(Clone, Debug, PartialEq)]
pub struct CorpusData {
    /// Corpus manifest.
    pub manifest: Manifest,
    /// Every committed episode in seed order.
    pub episodes: Vec<Episode>,
}
