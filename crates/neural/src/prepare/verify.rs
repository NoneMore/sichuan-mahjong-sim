//! Strict prepared-cache verification independent of the source corpus.

use std::collections::BTreeSet;
use std::path::Path;

use crate::collection::{seed_split, Seed, Split};

use super::error::{invalid_array, invalid_manifest, PrepareError, Result};
use super::manifest::{
    expected_array, read_dataset_manifest, ArrayEntry, Counts, DatasetManifest, ShardManifest,
};
use super::npy::{self, NpyDtype};

/// Reads the manifest, verifies every listed array file, and checks packed
/// metadata alignment.
///
/// Verification is intentionally offline: it never opens the source corpus and
/// therefore demonstrates that a completed cache is usable independently of it.
/// It recomputes all per-shard counts from `candidate_offsets`, rejects episode
/// seeds outside the captured snapshot range or repeated across the whole cache,
/// and requires every captured game seed to appear exactly once.
pub fn verify_dataset(root: impl AsRef<Path>) -> Result<DatasetManifest> {
    let root = root.as_ref();
    let manifest = read_dataset_manifest(root)?;
    let seed_start = u128::from(manifest.corpus_manifest.seed_start.as_u64());
    let seed_end = seed_start + manifest.corpus_manifest.total_games();
    let mut seen_seeds = BTreeSet::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let split_manifest = manifest.splits.get(split);
        for shard in &split_manifest.shards {
            verify_shard(root, split, shard, &mut seen_seeds, seed_start, seed_end)?;
        }
    }
    if seen_seeds.len() as u128 != manifest.corpus_manifest.total_games() {
        return Err(invalid_manifest(
            root,
            format!(
                "cache contains {} distinct episode seeds, captured corpus lists {} games",
                seen_seeds.len(),
                manifest.corpus_manifest.total_games()
            ),
        ));
    }
    Ok(manifest)
}

fn verify_shard(
    root: &Path,
    split: Split,
    shard: &ShardManifest,
    seen_seeds: &mut BTreeSet<u64>,
    seed_start: u128,
    seed_end: u128,
) -> Result<()> {
    for (name, entry) in &shard.arrays {
        let (dtype_label, shape) =
            expected_array(name, shard.counts.decisions, shard.counts.candidates);
        let dtype = NpyDtype::from_label(dtype_label)
            .expect("manifest validation only accepts known dtypes");
        let path = root.join(&entry.file);
        let (sha256, bytes) = npy::verify_array(&path, dtype, &shape)?;
        if bytes != entry.bytes {
            return Err(invalid_array(
                &path,
                format!("file has {bytes} bytes, manifest lists {}", entry.bytes),
            ));
        }
        if sha256 != entry.sha256 {
            return Err(PrepareError::ChecksumMismatch {
                path,
                expected: entry.sha256.clone(),
                found: sha256,
            });
        }
    }
    verify_metadata(root, split, shard, seen_seeds, seed_start, seed_end)
}

fn verify_metadata(
    root: &Path,
    split: Split,
    shard: &ShardManifest,
    seen_seeds: &mut BTreeSet<u64>,
    seed_start: u128,
    seed_end: u128,
) -> Result<()> {
    let decisions = shard.counts.decisions;
    let candidates = shard.counts.candidates;
    let offsets = read_i64(root, shard.array("candidate_offsets")?, &[decisions + 1])?;
    let targets = read_i64(root, shard.array("targets")?, &[decisions])?;
    let seeds = read_u64(root, shard.array("episode_seeds")?, &[decisions])?;
    let indices = read_u64(root, shard.array("decision_indices")?, &[decisions])?;
    let seats = read_u8(root, shard.array("actor_seats")?, &[decisions])?;
    let stages = read_u8(root, shard.array("stages")?, &[decisions])?;
    let kinds = read_u8(root, shard.array("target_kinds")?, &[decisions])?;

    if offsets.len() != decisions as usize + 1 || offsets[0] != 0 {
        return Err(invalid_array(
            &root.join(&shard.array("candidate_offsets")?.file),
            "offsets must start at zero and have D + 1 entries",
        ));
    }
    let last = *offsets.last().expect("offsets are nonempty");
    if last != candidates as i64 {
        return Err(invalid_array(
            &root.join(&shard.array("candidate_offsets")?.file),
            format!("offsets end at {last}, candidate count is {candidates}"),
        ));
    }

    let mut previous_seed: Option<u64> = None;
    let mut previous_index = 0u64;
    let mut derived = Counts::default();
    for index in 0..decisions as usize {
        let start = offsets[index];
        let end = offsets[index + 1];
        if end <= start {
            return Err(invalid_array(
                &root.join(&shard.array("candidate_offsets")?.file),
                format!("decision {index} has an empty candidate range"),
            ));
        }
        let target = targets[index];
        if target < 0 || target >= end - start {
            return Err(invalid_array(
                &root.join(&shard.array("targets")?.file),
                format!(
                    "decision {index} target {target} is outside local candidate count {}",
                    end - start
                ),
            ));
        }
        if seats[index] > 3 {
            return Err(invalid_array(
                &root.join(&shard.array("actor_seats")?.file),
                format!(
                    "decision {index} actor seat {} is outside 0..3",
                    seats[index]
                ),
            ));
        }
        if stages[index] >= 5 {
            return Err(invalid_array(
                &root.join(&shard.array("stages")?.file),
                format!("decision {index} stage {} is outside 0..4", stages[index]),
            ));
        }
        if kinds[index] >= 9 {
            return Err(invalid_array(
                &root.join(&shard.array("target_kinds")?.file),
                format!(
                    "decision {index} target kind {} is outside 0..8",
                    kinds[index]
                ),
            ));
        }
        let seed = seeds[index];
        if previous_seed != Some(seed) {
            if u128::from(seed) < seed_start || u128::from(seed) >= seed_end {
                return Err(invalid_manifest(
                    root,
                    format!(
                        "seed {seed:016x} is outside the captured snapshot seed range [{seed_start}, {seed_end})"
                    ),
                ));
            }
            let frozen = seed_split(Seed::from_u64(seed));
            if frozen != split {
                return Err(invalid_manifest(
                    root,
                    format!("seed {seed:016x} belongs to {frozen}, not {split}"),
                ));
            }
            if !seen_seeds.insert(seed) {
                return Err(invalid_manifest(
                    root,
                    format!("seed {seed:016x} appears in more than one episode block"),
                ));
            }
            if indices[index] != 0 {
                return Err(invalid_manifest(
                    root,
                    format!(
                        "seed {seed:016x} starts with decision index {}, expected 0",
                        indices[index]
                    ),
                ));
            }
            derived.observe_episode();
            previous_seed = Some(seed);
        } else if indices[index] != previous_index + 1 {
            return Err(invalid_manifest(
                root,
                format!(
                    "seed {seed:016x} decision index {} is not the next {}",
                    indices[index],
                    previous_index + 1
                ),
            ));
        }
        previous_index = indices[index];
        derived.observe_decision((end - start) as usize);
    }
    if derived != shard.counts {
        return Err(invalid_manifest(
            root,
            format!(
                "shard {} derived counts {:?} differ from manifest counts {:?}",
                shard.index, derived, shard.counts
            ),
        ));
    }
    Ok(())
}

impl ShardManifest {
    fn array(&self, name: &str) -> Result<&ArrayEntry> {
        self.arrays
            .get(name)
            .ok_or_else(|| PrepareError::InvalidManifest {
                path: std::path::PathBuf::from("dataset.json"),
                message: format!("shard {} is missing array {name}", self.index),
            })
    }
}

fn read_i64(root: &Path, entry: &ArrayEntry, shape: &[u64]) -> Result<Vec<i64>> {
    npy::read_array::<i64>(&root.join(&entry.file), shape)
}

fn read_u64(root: &Path, entry: &ArrayEntry, shape: &[u64]) -> Result<Vec<u64>> {
    npy::read_array::<u64>(&root.join(&entry.file), shape)
}

fn read_u8(root: &Path, entry: &ArrayEntry, shape: &[u64]) -> Result<Vec<u8>> {
    npy::read_array::<u8>(&root.join(&entry.file), shape)
}
