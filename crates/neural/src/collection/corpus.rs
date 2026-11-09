//! Corpus coordination: seed ranges, shard boundaries, committed inventory,
//! and strict reading. Episode execution lives in `driver`; filesystem locking
//! and publication live in `storage`.
//!
//! A writer holds a nonblocking exclusive lock on `LOCK` for the whole
//! initialization or append. It streams complete games into a temporary shard,
//! validates the completed file, publishes it under `shards/`, and only then
//! atomically replaces the manifest. A crash before the manifest replacement
//! leaves an unlisted shard that a retry ignores and replays from the first
//! uncommitted seed. The lock file is never removed.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;
use sichuan_mahjong_sim::{HeuristicCache, HeuristicPolicy, PLAYER_COUNT};

use super::driver::collect_episode;
use super::error::{io, CollectionError, Result};
use super::format::{read_shard_file, ShardExpectations};
use super::manifest::{manifest_path, shards_dir, CollectionConfig, Manifest, ShardEntry};
use super::report::{
    elapsed_ns, CacheReport, CollectionOutcome, CollectionTimings, CorpusData, CorpusReport,
};
use super::seed::Seed;
use super::source::{build_info, environment_info, SourceRevision};
use super::stats::StatsAccumulator;
use super::storage::{write_manifest_atomic, ActiveShard, CorpusLock};

/// One corpus writer holding the exclusive lock.
#[derive(Debug)]
pub struct Corpus {
    root: PathBuf,
    manifest: Manifest,
    _lock: CorpusLock,
}

impl Corpus {
    /// Creates the initial manifest for a new corpus.
    ///
    /// The root directory and `shards/` are created. An existing manifest is
    /// rejected rather than overwritten. Unlisted files are left untouched.
    pub fn create(root: impl AsRef<Path>, config: &CollectionConfig) -> Result<Self> {
        let root = root.as_ref().to_owned();
        config.validate()?;
        fs::create_dir_all(&root).map_err(|error| io(&root, error))?;
        let lock = CorpusLock::try_acquire(&root)?;
        let manifest_path = manifest_path(&root);
        if manifest_path.exists() {
            return Err(CollectionError::CorpusExists { root });
        }
        fs::create_dir_all(shards_dir(&root)).map_err(|error| io(shards_dir(&root), error))?;
        let manifest = Manifest::new(config)?;
        write_manifest_atomic(&root, &manifest)?;
        Ok(Self {
            root,
            manifest,
            _lock: lock,
        })
    }

    /// Opens an existing corpus for append and acquires the exclusive lock.
    ///
    /// Manifest metadata is validated and listed shard files must exist, but
    /// their decision records are not read. The requested compiled source must
    /// match the corpus source exactly.
    pub fn open_append(root: impl AsRef<Path>, source: &SourceRevision) -> Result<Self> {
        let root = root.as_ref().to_owned();
        let lock = CorpusLock::try_acquire(&root)?;
        let path = manifest_path(&root);
        let manifest = read_manifest_at(&path, &root)?;
        if manifest.source_revision != *source {
            return Err(CollectionError::SourceMismatch {
                manifest: manifest.source_revision,
                requested: source.clone(),
            });
        }
        for shard in &manifest.shards {
            let path = manifest.shard_path(&root, shard);
            let metadata = fs::metadata(&path).map_err(|error| io(&path, error))?;
            if !metadata.is_file() {
                return Err(CollectionError::InvalidManifest {
                    path: manifest_path(&root),
                    message: format!("listed shard {} is not a file", path.display()),
                });
            }
        }
        Ok(Self {
            root,
            manifest,
            _lock: lock,
        })
    }

    /// Current committed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Corpus root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Appends the requested number of consecutive games using the default
    /// shared teacher cache budget.
    pub fn append(&mut self, games: u64) -> Result<CollectionOutcome> {
        self.append_with_cache(games, DEFAULT_BUDGET_BYTES)
    }

    /// Appends the requested number of consecutive games.
    ///
    /// `cache_budget_bytes` is the shared four-seat teacher cache budget; zero
    /// disables cross-decision caching. Committed shards remain published if a
    /// later game or publication step fails. The returned statistics cover
    /// only the games committed by this call.
    pub fn append_with_cache(
        &mut self,
        games: u64,
        cache_budget_bytes: usize,
    ) -> Result<CollectionOutcome> {
        if games == 0 {
            return Err(CollectionError::EmptyCollection);
        }
        let first_seed = self
            .manifest
            .next_seed()
            .ok_or(CollectionError::SeedSpaceExhausted {
                seed_start: self.manifest.seed_start,
                committed: self.manifest.total_games(),
            })?;
        let first_u64 = first_seed.as_u64();
        let last_u128 = u128::from(first_u64) + u128::from(games) - 1;
        if last_u128 > u128::from(u64::MAX) {
            return Err(CollectionError::SeedRangeOverflow { first_seed, games });
        }
        let last_seed = Seed::from_u64(last_u128 as u64);

        let cache = HeuristicCache::new(cache_budget_bytes)
            .map_err(|source| CollectionError::CacheConfig { source })?;
        let mut policies: [HeuristicPolicy; PLAYER_COUNT] =
            std::array::from_fn(|_| HeuristicPolicy::with_cache(cache.clone()));

        let shards = shards_dir(&self.root);
        fs::create_dir_all(&shards).map_err(|error| io(&shards, error))?;

        let mut stats = StatsAccumulator::new();
        let mut timings = CollectionTimings::default();
        let started = Instant::now();
        let mut active: Option<ActiveShard> = None;
        let mut committed_games = 0u64;

        for offset in 0..games {
            let seed = Seed::from_u64(first_u64 + offset);
            if active.is_none() {
                active = Some(ActiveShard::new(&shards, seed)?);
            }
            let shard = active.as_mut().expect("active shard exists");
            let decisions = collect_episode(
                seed,
                &self.manifest.source_revision,
                self.manifest.decision_limit,
                &mut policies,
                &mut stats,
                &mut timings,
                |record| shard.write_record(record),
            )?;
            shard.complete_episode(decisions);
            committed_games += 1;

            if shard.games() == self.manifest.shard_games || offset + 1 == games {
                let shard = active.take().expect("active shard exists");
                let timer = Instant::now();
                let entry = shard.finish(&self.root, &self.manifest)?;
                self.commit_shard(entry)?;
                timings.commit_ns += elapsed_ns(timer);
            }
        }

        debug_assert!(active.is_none());
        debug_assert_eq!(committed_games, games);
        let (shard_bytes, manifest_bytes) = storage_bytes(&self.root, &self.manifest)?;
        let stats = stats.finish();
        debug_assert_eq!(stats.games, games);
        timings.total_ns = elapsed_ns(started);
        timings.games = stats.games;
        timings.decisions = stats.decisions;
        timings.finalize_rates();
        Ok(CollectionOutcome {
            requested_games: games,
            committed_games,
            first_seed,
            last_seed,
            committed_stats: stats,
            manifest: self.manifest.clone(),
            shard_bytes,
            manifest_bytes,
            total_storage_bytes: shard_bytes + manifest_bytes,
            timings,
            cache: CacheReport::capture(&cache.status()),
            build: build_info(),
            environment: environment_info(),
        })
    }

    fn commit_shard(&mut self, entry: ShardEntry) -> Result<()> {
        let mut next = self.manifest.clone();
        next.shards.push(entry);
        next.validate_at(&manifest_path(&self.root))?;
        write_manifest_atomic(&self.root, &next)?;
        self.manifest = next;
        Ok(())
    }
}

/// Reads and validates the manifest without acquiring the writer lock.
pub fn read_manifest(root: impl AsRef<Path>) -> Result<Manifest> {
    let root = root.as_ref();
    read_manifest_at(&manifest_path(root), root)
}

fn read_manifest_at(path: &Path, root: &Path) -> Result<Manifest> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(CollectionError::CorpusMissing {
                root: root.to_owned(),
            });
        }
        Err(error) => return Err(io(path, error)),
    };
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|source| CollectionError::ManifestJson {
            path: path.to_owned(),
            source,
        })?;
    manifest.validate_at(path)?;
    Ok(manifest)
}

/// Reads every committed episode and validates all shards and checksums.
///
/// Unlisted files are ignored; listed corrupt or incomplete data is an error.
pub fn read_corpus(root: impl AsRef<Path>) -> Result<CorpusData> {
    let root = root.as_ref();
    let manifest = read_manifest(root)?;
    let mut episodes = Vec::new();
    for shard in &manifest.shards {
        read_shard(root, &manifest, shard, |episode| episodes.push(episode))?;
    }
    Ok(CorpusData { manifest, episodes })
}

/// Computes aggregate statistics by strictly reading every committed shard.
pub fn report_corpus(root: impl AsRef<Path>) -> Result<CorpusReport> {
    let root = root.as_ref();
    let manifest = read_manifest(root)?;
    let mut stats = StatsAccumulator::new();
    for shard in &manifest.shards {
        read_shard(root, &manifest, shard, |episode| {
            stats.observe_episode(episode.split, &episode.result);
            for sample in episode.decisions {
                stats.observe_decision(
                    episode.split,
                    sample.stage,
                    &sample.candidates,
                    sample.target_index,
                );
            }
        })?;
    }
    let (shard_bytes, manifest_bytes) = storage_bytes(root, &manifest)?;
    Ok(CorpusReport {
        manifest,
        stats: stats.finish(),
        shard_bytes,
        manifest_bytes,
        total_storage_bytes: shard_bytes + manifest_bytes,
    })
}

/// Strictly verifies every committed shard, returning the same report as
/// [`report_corpus`].
pub fn verify_corpus(root: impl AsRef<Path>) -> Result<CorpusReport> {
    report_corpus(root)
}

fn read_shard(
    root: &Path,
    manifest: &Manifest,
    shard: &ShardEntry,
    on_episode: impl FnMut(super::format::Episode),
) -> Result<super::format::ParsedShard> {
    let path = manifest.shard_path(root, shard);
    read_shard_file(
        &path,
        ShardExpectations {
            source_revision: &manifest.source_revision,
            first_seed: shard.first_seed,
            games: shard.games,
            decisions: shard.decisions,
            decision_limit: manifest.decision_limit,
            sha256: Some(&shard.sha256),
        },
        on_episode,
    )
}

fn storage_bytes(root: &Path, manifest: &Manifest) -> Result<(u64, u64)> {
    let mut shard_bytes = 0u64;
    for shard in &manifest.shards {
        let path = manifest.shard_path(root, shard);
        let metadata = fs::metadata(&path).map_err(|error| io(&path, error))?;
        shard_bytes = shard_bytes.saturating_add(metadata.len());
    }
    let path = manifest_path(root);
    let manifest_bytes = fs::metadata(&path).map_err(|error| io(&path, error))?.len();
    Ok((shard_bytes, manifest_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, CollectionOutcome) {
        let root = tempfile::tempdir().unwrap();
        let source = SourceRevision::parse("0123456789abcdef0123456789abcdef01234567").unwrap();
        let config = CollectionConfig::new(source).with_shard_games(2);
        let mut corpus = Corpus::create(root.path(), &config).unwrap();
        let outcome = corpus.append_with_cache(2, 0).unwrap();
        (root, outcome)
    }

    #[test]
    fn shard_reader_delivers_episodes_before_reading_the_rest_of_the_shard() {
        let (root, outcome) = fixture();
        assert_eq!(
            report_corpus(root.path()).unwrap().stats,
            outcome.committed_stats
        );
        let manifest = outcome.manifest;
        let shard = &manifest.shards[0];
        let path = manifest.shard_path(root.path(), shard);
        let text = fs::read_to_string(&path).unwrap();
        let mut prefix = String::new();
        for line in text.lines() {
            prefix.push_str(line);
            prefix.push('\n');
            if line.contains("\"type\":\"result\"") {
                break;
            }
        }
        prefix.push_str("not-json\n");
        fs::write(&path, prefix).unwrap();

        let mut delivered = 0;
        let error = read_shard(root.path(), &manifest, shard, |episode| {
            assert_eq!(episode.id.seed, shard.first_seed);
            delivered += 1;
        })
        .err()
        .expect("invalid tail must reject the shard");
        assert_eq!(delivered, 1, "reader must not buffer the entire shard");
        assert!(matches!(error, CollectionError::ShardJson { .. }));
        assert!(
            report_corpus(root.path()).is_err(),
            "no partial report on error"
        );
    }

    #[test]
    fn shard_reader_rejects_episodes_over_the_manifest_decision_limit() {
        let (root, outcome) = fixture();
        let mut manifest = outcome.manifest;
        manifest.decision_limit = 1;
        let error = read_shard(root.path(), &manifest, &manifest.shards[0], |_| {})
            .err()
            .expect("decision limit must also be checked while reading");
        assert!(error.to_string().contains("exceeds decision limit"));
    }

    #[test]
    fn shard_reader_rejects_missing_final_newline_and_inconsistent_results() {
        let (root, outcome) = fixture();
        let manifest = outcome.manifest;
        let shard = &manifest.shards[0];
        let path = manifest.shard_path(root.path(), shard);
        let original = fs::read(&path).unwrap();
        fs::write(&path, &original[..original.len() - 1]).unwrap();
        let error = read_shard(root.path(), &manifest, shard, |_| {})
            .err()
            .expect("a JSONL record must end with a newline");
        assert!(error.to_string().contains("final newline"));

        let text = String::from_utf8(original).unwrap();
        for (end_reason, winners) in [
            ("three_winners", [false; 4]),
            ("three_winners", [true; 4]),
            ("wall_exhausted", [true; 4]),
        ] {
            let mut changed = String::new();
            for line in text.lines() {
                let mut row: serde_json::Value = serde_json::from_str(line).unwrap();
                if row["type"] == "result" {
                    row["end_reason"] = end_reason.into();
                    row["winners"] = serde_json::json!(winners);
                }
                changed.push_str(&serde_json::to_string(&row).unwrap());
                changed.push('\n');
            }
            fs::write(&path, changed).unwrap();
            let error = read_shard(root.path(), &manifest, shard, |_| {})
                .err()
                .expect("terminal result must be internally consistent");
            assert!(error.to_string().contains("winner count"), "{error}");
        }
    }
}
