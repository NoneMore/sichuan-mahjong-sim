//! Versioned corpus manifest and its fixed metadata invariants.
//!
//! The manifest is the commit record. Shards are immutable once listed, and
//! their seed ranges form one contiguous prefix starting at `seed_start`.
//! Append validates this metadata without reading listed decision records.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::encoding::ENCODING_VERSION;

use super::error::{CollectionError, Result};
use super::seed::{Seed, SPLIT_ALGORITHM};
use super::source::SourceRevision;

/// Supported manifest format version.
pub const FORMAT_VERSION: u32 = 1;

/// Manifest file name.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Shard directory name.
pub const SHARDS_DIR: &str = "shards";

/// Exclusive lock file name.
pub const LOCK_FILE: &str = "LOCK";

/// Default complete games per shard for new corpora.
pub const DEFAULT_SHARD_GAMES: u64 = 100;

/// Immutable metadata of one committed JSONL shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardEntry {
    /// First seed in the shard, also the file name stem.
    pub first_seed: Seed,
    /// Complete episodes in the shard.
    pub games: u64,
    /// Decision records in the shard.
    pub decisions: u64,
    /// Lowercase SHA-256 of the exact shard file bytes.
    pub sha256: String,
}

/// Corpus manifest.
///
/// The public fields mirror the fixed JSON object. Use [`Manifest::validate`]
/// after constructing or mutating one manually. Collection only writes
/// internally validated manifests.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Format version; must equal [`FORMAT_VERSION`].
    pub format_version: u32,
    /// Full compiled source revision shared by every episode.
    pub source_revision: SourceRevision,
    /// Encoding version shared by every episode.
    pub encoding_version: u32,
    /// Fixed seed split algorithm.
    pub split_algorithm: String,
    /// First seed of the corpus.
    pub seed_start: Seed,
    /// Positive per-game decision limit fixed for the corpus.
    pub decision_limit: u64,
    /// Positive target games per shard fixed for the corpus.
    pub shard_games: u64,
    /// Committed shards in seed order.
    pub shards: Vec<ShardEntry>,
}

impl Manifest {
    /// Builds the initial empty manifest for a new corpus.
    pub fn new(config: &CollectionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            format_version: FORMAT_VERSION,
            source_revision: config.source_revision.clone(),
            encoding_version: ENCODING_VERSION,
            split_algorithm: SPLIT_ALGORITHM.to_owned(),
            seed_start: config.seed_start,
            decision_limit: config.decision_limit,
            shard_games: config.shard_games,
            shards: Vec::new(),
        })
    }

    /// Validates all manifest invariants.
    pub fn validate(&self) -> Result<()> {
        self.validate_at(Path::new(MANIFEST_FILE))
    }

    /// Validates all manifest invariants, reporting the supplied path.
    pub fn validate_at(&self, path: &Path) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
            return Err(CollectionError::UnsupportedFormatVersion {
                path: path.to_owned(),
                found: self.format_version,
                supported: FORMAT_VERSION,
            });
        }
        if self.encoding_version != ENCODING_VERSION {
            return Err(CollectionError::UnsupportedEncodingVersion {
                found: self.encoding_version,
                supported: ENCODING_VERSION,
            });
        }
        if self.split_algorithm != SPLIT_ALGORITHM {
            return Err(CollectionError::UnsupportedSplitAlgorithm {
                found: self.split_algorithm.clone(),
            });
        }
        if self.decision_limit == 0 {
            return Err(invalid_manifest(path, "decision_limit must be positive"));
        }
        if self.shard_games == 0 {
            return Err(invalid_manifest(path, "shard_games must be positive"));
        }

        let mut next = u128::from(self.seed_start.as_u64());
        for (index, shard) in self.shards.iter().enumerate() {
            if shard.games == 0 {
                return Err(invalid_manifest(
                    path,
                    format!("shard {index} must contain at least one game"),
                ));
            }
            if shard.games > self.shard_games {
                return Err(invalid_manifest(
                    path,
                    format!(
                        "shard {index} has {} games, exceeding shard_games {}",
                        shard.games, self.shard_games
                    ),
                ));
            }
            if shard.decisions == 0 {
                return Err(invalid_manifest(
                    path,
                    format!("shard {index} must contain at least one decision"),
                ));
            }
            if !is_lower_hex_sha256(&shard.sha256) {
                return Err(invalid_manifest(
                    path,
                    format!("shard {index} sha256 is not 64 lowercase hex digits"),
                ));
            }
            if u128::from(shard.first_seed.as_u64()) != next {
                return Err(invalid_manifest(
                    path,
                    format!(
                        "shard {index} starts at {}, expected {:016x}",
                        shard.first_seed, next
                    ),
                ));
            }
            next += u128::from(shard.games);
            if next > 1u128 << 64 {
                return Err(invalid_manifest(
                    path,
                    format!("shard {index} seed range exceeds u64::MAX"),
                ));
            }
        }
        Ok(())
    }

    /// Total committed games, exact even when the range reaches `2^64`.
    pub fn total_games(&self) -> u128 {
        self.shards
            .iter()
            .map(|shard| u128::from(shard.games))
            .sum()
    }

    /// Total committed decision records.
    pub fn total_decisions(&self) -> u128 {
        self.shards
            .iter()
            .map(|shard| u128::from(shard.decisions))
            .sum()
    }

    /// First uncommitted seed, or `None` when the u64 seed space is exhausted.
    pub fn next_seed(&self) -> Option<Seed> {
        let next = u128::from(self.seed_start.as_u64()) + self.total_games();
        if next > u128::from(u64::MAX) {
            None
        } else {
            Some(Seed::from_u64(next as u64))
        }
    }

    /// Final committed seed, when the corpus is nonempty.
    pub fn last_seed(&self) -> Option<Seed> {
        let end = u128::from(self.seed_start.as_u64()) + self.total_games();
        if end == u128::from(self.seed_start.as_u64()) || end > 1u128 << 64 {
            return None;
        }
        Some(Seed::from_u64((end - 1) as u64))
    }

    /// Absolute path of one shard: `shards/<first_seed>.jsonl`.
    pub fn shard_path(&self, root: &Path, shard: &ShardEntry) -> PathBuf {
        shard_path(root, shard.first_seed)
    }
}

/// Collection settings fixed when a corpus is created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionConfig {
    /// Source revision shared by every episode.
    ///
    /// The CLI fills this from compiled, clean Git provenance. Library callers
    /// are responsible for attesting an explicitly supplied revision.
    pub source_revision: SourceRevision,
    /// First game seed.
    pub seed_start: Seed,
    /// Positive target games per shard; the final shard may be smaller.
    pub shard_games: u64,
    /// Positive decision safety limit per game.
    pub decision_limit: u64,
}

impl CollectionConfig {
    /// Creates settings with `seed_start = 0`, 100 games per shard, and a
    /// 10,000-decision per-game limit.
    pub fn new(source_revision: SourceRevision) -> Self {
        Self {
            source_revision,
            seed_start: Seed::from_u64(0),
            shard_games: DEFAULT_SHARD_GAMES,
            decision_limit: 10_000,
        }
    }

    /// Builder form of [`CollectionConfig::seed_start`].
    pub fn with_seed_start(mut self, seed_start: Seed) -> Self {
        self.seed_start = seed_start;
        self
    }

    /// Builder form of [`CollectionConfig::shard_games`].
    pub fn with_shard_games(mut self, shard_games: u64) -> Self {
        self.shard_games = shard_games;
        self
    }

    /// Builder form of [`CollectionConfig::decision_limit`].
    pub fn with_decision_limit(mut self, decision_limit: u64) -> Self {
        self.decision_limit = decision_limit;
        self
    }

    /// Requires positive shard and decision limits.
    pub fn validate(&self) -> Result<()> {
        if self.shard_games == 0 {
            return Err(CollectionError::InvalidConfig {
                message: "shard_games must be positive".to_owned(),
            });
        }
        if self.decision_limit == 0 {
            return Err(CollectionError::InvalidConfig {
                message: "decision_limit must be positive".to_owned(),
            });
        }
        Ok(())
    }
}

/// Absolute manifest path.
pub fn manifest_path(root: &Path) -> PathBuf {
    root.join(MANIFEST_FILE)
}

/// Absolute shard directory path.
pub fn shards_dir(root: &Path) -> PathBuf {
    root.join(SHARDS_DIR)
}

/// Absolute lock file path.
pub fn lock_path(root: &Path) -> PathBuf {
    root.join(LOCK_FILE)
}

/// Absolute shard path for a first seed.
pub fn shard_path(root: &Path, first_seed: Seed) -> PathBuf {
    shards_dir(root).join(format!("{}.jsonl", first_seed.to_hex()))
}

fn invalid_manifest(path: &Path, message: impl Into<String>) -> CollectionError {
    CollectionError::InvalidManifest {
        path: path.to_owned(),
        message: message.into(),
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
