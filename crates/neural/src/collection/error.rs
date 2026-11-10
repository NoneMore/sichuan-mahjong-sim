//! Errors from corpus creation, collection, publication, and strict reading.

use std::fmt;
use std::io;
use std::path::PathBuf;

use sichuan_mahjong_sim::{Action, CacheConfigError, GameError, Seat};

use super::seed::{Seed, Split};
use super::source::SourceRevision;

/// Result type for collection and corpus operations.
pub type Result<T> = std::result::Result<T, CollectionError>;

/// A collection, publication, or strict corpus-reading failure.
///
/// Errors carry the seed, decision index, path, or line where the operation
/// stopped. In particular, collection never substitutes an action, skips a
/// game, or silently omits a malformed record.
#[derive(Debug)]
pub enum CollectionError {
    /// A filesystem operation failed.
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying failure.
        source: io::Error,
    },
    /// Opening or locking the corpus lock file failed.
    Lock {
        /// Lock path.
        path: PathBuf,
        /// Underlying failure.
        source: io::Error,
    },
    /// Another process or handle holds the exclusive corpus lock.
    CorpusBusy {
        /// Corpus root.
        root: PathBuf,
    },
    /// A new corpus was requested where a manifest already exists.
    CorpusExists {
        /// Corpus root.
        root: PathBuf,
    },
    /// An existing corpus manifest is missing.
    CorpusMissing {
        /// Corpus root.
        root: PathBuf,
    },
    /// A configuration value violates a positive or consistency requirement.
    InvalidConfig {
        /// Human-readable cause.
        message: String,
    },
    /// Manifest metadata violates the format invariants.
    InvalidManifest {
        /// Manifest path.
        path: PathBuf,
        /// Human-readable cause.
        message: String,
    },
    /// Manifest JSON could not be decoded.
    ManifestJson {
        /// Manifest path.
        path: PathBuf,
        /// Decoder failure.
        source: serde_json::Error,
    },
    /// The manifest format version is not supported.
    UnsupportedFormatVersion {
        /// Manifest path.
        path: PathBuf,
        /// Version found in the file.
        found: u32,
        /// Supported version.
        supported: u32,
    },
    /// The manifest encoding version is not supported.
    UnsupportedEncodingVersion {
        /// Version found in the manifest.
        found: u32,
        /// Supported version.
        supported: u32,
    },
    /// The manifest names an unknown split algorithm.
    UnsupportedSplitAlgorithm {
        /// Algorithm found in the manifest.
        found: String,
    },
    /// Append requested a different compiled source than the corpus fixes.
    SourceMismatch {
        /// Source fixed by the manifest.
        manifest: SourceRevision,
        /// Source requesting the append.
        requested: SourceRevision,
    },
    /// An append request contained zero games.
    EmptyCollection,
    /// A captured snapshot shard index is outside the listed shard range.
    SnapshotShardOutOfRange {
        /// Requested index.
        index: usize,
        /// Number of listed shards in the snapshot.
        shards: usize,
    },
    /// Every `u64` seed is already committed.
    SeedSpaceExhausted {
        /// Corpus first seed.
        seed_start: Seed,
        /// Committed game count.
        committed: u128,
    },
    /// The requested seed range would exceed `u64::MAX`.
    SeedRangeOverflow {
        /// First seed of the requested range.
        first_seed: Seed,
        /// Requested game count.
        games: u64,
    },
    /// Shard JSON could not be decoded.
    ShardJson {
        /// Shard path.
        path: PathBuf,
        /// One-based physical line number.
        line: u64,
        /// Decoder failure.
        source: serde_json::Error,
    },
    /// Shard content violates the format contract.
    ShardInvalid {
        /// Shard path.
        path: PathBuf,
        /// One-based physical line number, or zero for file-level failures.
        line: u64,
        /// Human-readable cause.
        message: String,
    },
    /// The shard checksum differs from the manifest.
    ShardChecksumMismatch {
        /// Shard path.
        path: PathBuf,
        /// Manifest checksum.
        expected: String,
        /// Recomputed checksum.
        found: String,
    },
    /// A shard's episode/decision count differs from manifest metadata.
    ShardMetadataMismatch {
        /// Shard path.
        path: PathBuf,
        /// `games` or `decisions`.
        field: &'static str,
        /// Manifest value.
        expected: u64,
        /// Parsed value.
        found: u64,
    },
    /// A shard episode names a different source revision.
    ShardSourceMismatch {
        /// Shard path.
        path: PathBuf,
        /// Manifest/corpus revision.
        expected: SourceRevision,
        /// Recorded revision.
        found: SourceRevision,
    },
    /// A shard episode split differs from the frozen seed split.
    ShardSplitMismatch {
        /// Shard path.
        path: PathBuf,
        /// Episode seed.
        seed: Seed,
        /// Frozen split.
        expected: Split,
        /// Recorded split.
        found: Split,
    },
    /// A teacher target does not index the candidate list.
    ShardTargetOutOfRange {
        /// Shard path.
        path: PathBuf,
        /// One-based physical line number.
        line: u64,
        /// Recorded target.
        target: usize,
        /// Candidate count.
        candidates: usize,
    },
    /// A stored action has an unknown kind or malformed parameters.
    ShardActionInvalid {
        /// Shard path.
        path: PathBuf,
        /// One-based physical line number.
        line: u64,
        /// Human-readable cause.
        message: String,
    },
    /// A feature vector contains NaN or infinity.
    NonFiniteFeature {
        /// Shard path.
        path: PathBuf,
        /// One-based physical line number.
        line: u64,
        /// `state` or `candidate_features`.
        field: &'static str,
    },
    /// The game rejected the teacher's candidate.
    Game {
        /// Episode seed.
        seed: Seed,
        /// Zero-based decision index with the episode.
        decision_index: u64,
        /// Seat that submitted the action.
        seat: Seat,
        /// Authoritative rejection.
        source: GameError,
    },
    /// The teacher's choice was not a candidate of the encoded decision.
    TargetNotCandidate {
        /// Episode seed.
        seed: Seed,
        /// Zero-based decision index with the episode.
        decision_index: u64,
        /// Teacher choice.
        action: Action,
    },
    /// The teacher needed another decision beyond the fixed per-game limit.
    DecisionLimit {
        /// Episode seed.
        seed: Seed,
        /// Positive limit fixed by the manifest.
        limit: u64,
    },
    /// The game reported neither a pending decision nor a terminal result.
    MissingResult {
        /// Episode seed.
        seed: Seed,
    },
    /// The configured heuristic cache budget was rejected.
    CacheConfig {
        /// Core cache configuration failure.
        source: CacheConfigError,
    },
}

impl fmt::Display for CollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CollectionError::Io { path, source } => {
                write!(f, "I/O error at {}: {source}", path.display())
            }
            CollectionError::Lock { path, source } => {
                write!(f, "could not lock {}: {source}", path.display())
            }
            CollectionError::CorpusBusy { root } => write!(
                f,
                "corpus {} is busy: another writer holds the exclusive lock",
                root.display()
            ),
            CollectionError::CorpusExists { root } => {
                write!(f, "a corpus manifest already exists at {}", root.display())
            }
            CollectionError::CorpusMissing { root } => {
                write!(f, "no corpus manifest exists at {}", root.display())
            }
            CollectionError::InvalidConfig { message } => {
                write!(f, "invalid collection configuration: {message}")
            }
            CollectionError::InvalidManifest { path, message } => {
                write!(f, "invalid manifest {}: {message}", path.display())
            }
            CollectionError::ManifestJson { path, source } => {
                write!(f, "could not decode manifest {}: {source}", path.display())
            }
            CollectionError::UnsupportedFormatVersion {
                path,
                found,
                supported,
            } => write!(
                f,
                "manifest {} uses unsupported format version {found} (supported: {supported})",
                path.display()
            ),
            CollectionError::UnsupportedEncodingVersion { found, supported } => write!(
                f,
                "manifest uses unsupported encoding version {found} (supported: {supported})"
            ),
            CollectionError::UnsupportedSplitAlgorithm { found } => {
                write!(f, "manifest uses unsupported split algorithm {found:?}")
            }
            CollectionError::SourceMismatch {
                manifest,
                requested,
            } => write!(
                f,
                "corpus fixes source revision {manifest}, but the append requested {requested}"
            ),
            CollectionError::EmptyCollection => f.write_str("a collection requires at least one game"),
            CollectionError::SnapshotShardOutOfRange { index, shards } => write!(
                f,
                "snapshot shard index {index} is outside {shards} listed shards"
            ),
            CollectionError::SeedSpaceExhausted {
                seed_start,
                committed,
            } => write!(
                f,
                "corpus starting at {seed_start} already committed {committed} games; no uncommitted u64 seed remains"
            ),
            CollectionError::SeedRangeOverflow { first_seed, games } => write!(
                f,
                "seed range starting at {first_seed} with {games} games exceeds u64::MAX"
            ),
            CollectionError::ShardJson { path, line, source } => {
                write!(f, "{}:{line}: invalid shard JSON: {source}", path.display())
            }
            CollectionError::ShardInvalid {
                path,
                line,
                message,
            } => write!(f, "{}:{line}: {message}", path.display()),
            CollectionError::ShardChecksumMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "shard {} checksum mismatch: manifest {expected}, actual {found}",
                path.display()
            ),
            CollectionError::ShardMetadataMismatch {
                path,
                field,
                expected,
                found,
            } => write!(
                f,
                "shard {} {field} mismatch: manifest {expected}, actual {found}",
                path.display()
            ),
            CollectionError::ShardSourceMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "shard {} records source {found}, expected {expected}",
                path.display()
            ),
            CollectionError::ShardSplitMismatch {
                path,
                seed,
                expected,
                found,
            } => write!(
                f,
                "shard {} records split {found} for seed {seed}, expected {expected}",
                path.display()
            ),
            CollectionError::ShardTargetOutOfRange {
                path,
                line,
                target,
                candidates,
            } => write!(
                f,
                "{}:{line}: teacher target {target} outside {candidates} candidates",
                path.display()
            ),
            CollectionError::ShardActionInvalid {
                path,
                line,
                message,
            } => write!(
                f,
                "{}:{line}: invalid stored action: {message}",
                path.display()
            ),
            CollectionError::NonFiniteFeature { path, line, field } => write!(
                f,
                "{}:{line}: {field} contains NaN or infinity",
                path.display()
            ),
            CollectionError::Game {
                seed,
                decision_index,
                seat,
                source,
            } => write!(
                f,
                "seed {seed}, decision {decision_index}, {seat}: {source}"
            ),
            CollectionError::TargetNotCandidate {
                seed,
                decision_index,
                action,
            } => write!(
                f,
                "seed {seed}, decision {decision_index}: teacher chose non-candidate {action:?}"
            ),
            CollectionError::DecisionLimit { seed, limit } => write!(
                f,
                "seed {seed} needed another decision beyond the fixed limit {limit}"
            ),
            CollectionError::MissingResult { seed } => {
                write!(f, "seed {seed} ended without a terminal result")
            }
            CollectionError::CacheConfig { source } => {
                write!(f, "invalid heuristic cache configuration: {source}")
            }
        }
    }
}

impl std::error::Error for CollectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CollectionError::Io { source, .. } | CollectionError::Lock { source, .. } => {
                Some(source)
            }
            CollectionError::ShardJson { source, .. }
            | CollectionError::ManifestJson { source, .. } => Some(source),
            CollectionError::Game { source, .. } => Some(source),
            CollectionError::CacheConfig { source } => Some(source),
            _ => None,
        }
    }
}

pub(super) fn io(path: impl Into<PathBuf>, source: io::Error) -> CollectionError {
    CollectionError::Io {
        path: path.into(),
        source,
    }
}
