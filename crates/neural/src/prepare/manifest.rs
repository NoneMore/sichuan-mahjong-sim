//! Version 1 prepared-dataset manifest, layout, provenance, and validation.
//!
//! `dataset.json` is the only cache entry point. Every array is listed with a
//! relative path, dtype, shape, byte length, and SHA-256. The manifest records
//! the exact captured corpus manifest bytes and both collection and preparation
//! provenance; a dirty preparation build is allowed but never presented as a
//! collection attestation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::collection::{
    BuildInfo, EnvironmentInfo, Manifest, SourceRevision, Split, ACTION_KIND_NAMES, SPLIT_ALGORITHM,
};
use crate::encoding::{ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};

use super::error::{invalid_manifest, io, PrepareError, Result};

/// Dataset manifest file name.
pub const DATASET_FILE: &str = "dataset.json";
/// Array directory name.
pub const ARRAYS_DIR: &str = "arrays";
/// Stable dataset manifest format marker.
pub const DATASET_FORMAT: &str = "mahjong-bc-dataset";
/// Prepared cache format version.
pub const PREPARATION_FORMAT_VERSION: u32 = 1;
/// Preparation algorithm version.
pub const PREPARATION_VERSION: u32 = 1;
/// Default decisions per output shard.
pub const DEFAULT_MAX_DECISIONS_PER_SHARD: usize = 20_000;
/// Default packed candidates per output shard.
pub const DEFAULT_MAX_CANDIDATES_PER_SHARD: usize = 500_000;

/// Array names in the fixed cache identity order.
pub const ARRAY_NAMES: [&str; 9] = [
    "states",
    "candidate_features",
    "candidate_offsets",
    "targets",
    "episode_seeds",
    "decision_indices",
    "actor_seats",
    "stages",
    "target_kinds",
];

/// Number of v1 decision stages.
pub const STAGE_COUNT: u64 = 5;
/// Number of v1 action kinds.
pub const ACTION_KIND_COUNT: u64 = ACTION_KIND_NAMES.len() as u64;

/// One array file listed by a prepared shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayEntry {
    /// Forward-slash relative path under the cache root.
    pub file: String,
    /// Stable dtype name.
    pub dtype: String,
    /// Complete shape.
    pub shape: Vec<u64>,
    /// Exact file byte length.
    pub bytes: u64,
    /// Lowercase SHA-256 of the exact file bytes.
    pub sha256: String,
}

/// Counts shared by the dataset, a split, and one output shard.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Complete episodes.
    pub episodes: u64,
    /// Decision records.
    pub decisions: u64,
    /// Packed candidates.
    pub candidates: u64,
    /// Decisions with exactly one candidate.
    pub forced_decisions: u64,
    /// Decisions with more than one candidate.
    pub eligible_decisions: u64,
    /// Smallest candidate count, or zero with no decisions.
    pub min_candidates: usize,
    /// Largest candidate count, or zero with no decisions.
    pub max_candidates: usize,
}

impl Counts {
    pub(super) fn observe_decision(&mut self, candidate_count: usize) {
        if self.decisions == 0 {
            self.min_candidates = candidate_count;
        } else {
            self.min_candidates = self.min_candidates.min(candidate_count);
        }
        self.max_candidates = self.max_candidates.max(candidate_count);
        self.decisions += 1;
        self.candidates += candidate_count as u64;
        if candidate_count == 1 {
            self.forced_decisions += 1;
        } else {
            self.eligible_decisions += 1;
        }
    }

    pub(super) fn observe_episode(&mut self) {
        self.episodes += 1;
    }

    pub(super) fn add_episode(&mut self, episode: &crate::collection::Episode) {
        for sample in &episode.decisions {
            self.observe_decision(sample.candidate_count());
        }
        self.observe_episode();
    }

    pub(super) fn merge(&mut self, other: &Counts) {
        if other.decisions > 0 {
            if self.decisions == 0 {
                self.min_candidates = other.min_candidates;
                self.max_candidates = other.max_candidates;
            } else {
                self.min_candidates = self.min_candidates.min(other.min_candidates);
                self.max_candidates = self.max_candidates.max(other.max_candidates);
            }
        }
        self.episodes += other.episodes;
        self.decisions += other.decisions;
        self.candidates += other.candidates;
        self.forced_decisions += other.forced_decisions;
        self.eligible_decisions += other.eligible_decisions;
    }

    fn validate(&self, path: &Path, label: &str) -> Result<()> {
        if self.forced_decisions + self.eligible_decisions != self.decisions {
            return Err(invalid_manifest(
                path,
                format!(
                    "{label} forced + eligible decisions do not equal total {}",
                    self.decisions
                ),
            ));
        }
        if self.decisions == 0 {
            if self.episodes != 0
                || self.candidates != 0
                || self.min_candidates != 0
                || self.max_candidates != 0
            {
                return Err(invalid_manifest(
                    path,
                    format!("{label} has zero decisions but nonzero counts"),
                ));
            }
        } else {
            if self.episodes == 0 {
                return Err(invalid_manifest(
                    path,
                    format!("{label} has decisions but no episodes"),
                ));
            }
            if self.candidates < self.decisions {
                return Err(invalid_manifest(
                    path,
                    format!("{label} has fewer candidates than decisions"),
                ));
            }
            if self.min_candidates == 0 || self.min_candidates > self.max_candidates {
                return Err(invalid_manifest(
                    path,
                    format!("{label} candidate range is invalid"),
                ));
            }
            if self.candidates < self.decisions.saturating_mul(self.min_candidates as u64)
                || self.candidates > self.decisions.saturating_mul(self.max_candidates as u64)
            {
                return Err(invalid_manifest(
                    path,
                    format!("{label} candidate total is inconsistent with its range"),
                ));
            }
        }
        Ok(())
    }
}

/// One listed output shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Zero-based shard index within its split.
    pub index: u64,
    /// Shard counts.
    pub counts: Counts,
    /// Arrays keyed by their fixed names.
    pub arrays: BTreeMap<String, ArrayEntry>,
}

/// One split's shards and counts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitManifest {
    /// Split name.
    pub split: Split,
    /// Split counts.
    pub counts: Counts,
    /// Output shards in creation order.
    pub shards: Vec<ShardManifest>,
}

/// The three frozen splits, in train/validation/test order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Splits {
    /// Training split.
    pub train: SplitManifest,
    /// Validation split.
    pub validation: SplitManifest,
    /// Test split.
    pub test: SplitManifest,
}

impl Splits {
    /// Returns one split by its frozen enum.
    pub fn get(&self, split: Split) -> &SplitManifest {
        match split {
            Split::Train => &self.train,
            Split::Validation => &self.validation,
            Split::Test => &self.test,
        }
    }

    fn iter(&self) -> impl Iterator<Item = &SplitManifest> {
        [&self.train, &self.validation, &self.test].into_iter()
    }
}

/// Informational array layout for cache format version 1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    /// State width.
    pub state_width: u64,
    /// Candidate width.
    pub candidate_width: u64,
    /// Stage count.
    pub stage_count: u64,
    /// Action kind count.
    pub action_kind_count: u64,
    /// Array descriptions keyed by name.
    pub arrays: BTreeMap<String, LayoutArray>,
}

/// One array's dtype and axis names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutArray {
    /// Stable dtype name.
    pub dtype: String,
    /// Axis names.
    pub dimensions: Vec<String>,
}

impl Layout {
    /// Exact layout for cache format version 1.
    pub fn expected() -> Self {
        let mut arrays = BTreeMap::new();
        let mut insert = |name: &str, dtype: &str, dimensions: &[&str]| {
            arrays.insert(
                name.to_owned(),
                LayoutArray {
                    dtype: dtype.to_owned(),
                    dimensions: dimensions.iter().map(|value| (*value).to_owned()).collect(),
                },
            );
        };
        insert("states", "float32", &["decisions", "state_features"]);
        insert(
            "candidate_features",
            "float32",
            &["candidates", "candidate_features"],
        );
        insert("candidate_offsets", "int64", &["decisions + 1"]);
        insert("targets", "int64", &["decisions"]);
        insert("episode_seeds", "uint64", &["decisions"]);
        insert("decision_indices", "uint64", &["decisions"]);
        insert("actor_seats", "uint8", &["decisions"]);
        insert("stages", "uint8", &["decisions"]);
        insert("target_kinds", "uint8", &["decisions"]);
        Self {
            state_width: STATE_WIDTH as u64,
            candidate_width: ACTION_WIDTH as u64,
            stage_count: STAGE_COUNT,
            action_kind_count: ACTION_KIND_COUNT,
            arrays,
        }
    }
}

/// Facts already recorded by the captured corpus manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionProvenance {
    /// Corpus source revision.
    pub source_revision: SourceRevision,
    /// Corpus encoding version.
    pub encoding_version: u32,
    /// Corpus split algorithm.
    pub split_algorithm: String,
}

/// Build, environment, and configuration facts of the preparation process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparationProvenance {
    /// Compile-time build facts.
    pub build: BuildInfo,
    /// Runtime environment facts.
    pub environment: EnvironmentInfo,
    /// `clean`, `dirty`, or `unavailable`; never a collection attestation.
    pub build_attestation: String,
    /// Configured output shard decision limit.
    pub max_decisions_per_shard: usize,
    /// Configured output shard candidate limit.
    pub max_candidates_per_shard: usize,
}

/// Separated collection and preparation provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Facts inherited from the captured corpus.
    pub collection: CollectionProvenance,
    /// Facts of this preparation run.
    pub preparation: PreparationProvenance,
}

/// Complete version 1 prepared-dataset manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Stable format marker.
    pub format: String,
    /// Cache format version.
    pub preparation_format_version: u32,
    /// Preparation algorithm version.
    pub preparation_version: u32,
    /// Decision encoding version.
    pub encoding_version: u32,
    /// Content identity covering the captured snapshot, versions, and inventory.
    pub cache_id: String,
    /// Split algorithm recorded by the corpus.
    pub split_algorithm: String,
    /// Corpus source revision.
    pub source_revision: SourceRevision,
    /// Lowercase SHA-256 of the exact captured `manifest.json` bytes.
    pub corpus_manifest_sha256: String,
    /// Full captured corpus manifest.
    pub corpus_manifest: Manifest,
    /// Array layout.
    pub layout: Layout,
    /// Separated provenance.
    pub provenance: Provenance,
    /// Whole-dataset counts.
    pub counts: Counts,
    /// Split-specific manifests.
    pub splits: Splits,
}

impl DatasetManifest {
    /// Validates all manifest invariants.
    pub fn validate(&self) -> Result<()> {
        self.validate_at(Path::new(DATASET_FILE))
    }

    /// Validates all manifest invariants, reporting the supplied path.
    pub fn validate_at(&self, path: &Path) -> Result<()> {
        if self.format != DATASET_FORMAT {
            return Err(invalid_manifest(
                path,
                format!("format is {:?}, expected {DATASET_FORMAT:?}", self.format),
            ));
        }
        if self.preparation_format_version != PREPARATION_FORMAT_VERSION {
            return Err(PrepareError::UnsupportedFormatVersion {
                path: path.to_owned(),
                found: self.preparation_format_version,
                supported: PREPARATION_FORMAT_VERSION,
            });
        }
        if self.preparation_version != PREPARATION_VERSION {
            return Err(PrepareError::UnsupportedPreparationVersion {
                path: path.to_owned(),
                found: self.preparation_version,
                supported: PREPARATION_VERSION,
            });
        }
        if self.encoding_version != ENCODING_VERSION {
            return Err(PrepareError::UnsupportedEncodingVersion {
                path: path.to_owned(),
                found: self.encoding_version,
                supported: ENCODING_VERSION,
            });
        }
        if self.split_algorithm != SPLIT_ALGORITHM {
            return Err(invalid_manifest(
                path,
                format!("unsupported split algorithm {:?}", self.split_algorithm),
            ));
        }
        self.corpus_manifest.validate().map_err(|error| {
            invalid_manifest(
                path,
                format!("captured corpus manifest is invalid: {error}"),
            )
        })?;
        if self.corpus_manifest.source_revision != self.source_revision
            || self.corpus_manifest.encoding_version != self.encoding_version
            || self.corpus_manifest.split_algorithm != self.split_algorithm
        {
            return Err(invalid_manifest(
                path,
                "captured corpus identity differs from top-level identity",
            ));
        }
        if !is_lower_hex_sha256(&self.corpus_manifest_sha256) {
            return Err(invalid_manifest(
                path,
                "corpus_manifest_sha256 is not 64 lowercase hex digits",
            ));
        }
        if self.layout != Layout::expected() {
            return Err(invalid_manifest(
                path,
                "layout does not match prepared-cache format version 1",
            ));
        }
        if self.counts.episodes as u128 != self.corpus_manifest.total_games() {
            return Err(invalid_manifest(
                path,
                "dataset episode count differs from the captured corpus",
            ));
        }
        if self.counts.decisions as u128 != self.corpus_manifest.total_decisions() {
            return Err(invalid_manifest(
                path,
                "dataset decision count differs from the captured corpus",
            ));
        }
        self.counts.validate(path, "dataset")?;

        let mut total = Counts::default();
        let mut files = BTreeSet::new();
        for expected in [Split::Train, Split::Validation, Split::Test] {
            let split = self.splits.get(expected);
            if split.split != expected {
                return Err(invalid_manifest(
                    path,
                    format!(
                        "split entry {:?} is stored under {:?}",
                        split.split, expected
                    ),
                ));
            }
            split.counts.validate(path, expected.as_str())?;
            let mut sum = Counts::default();
            for (position, shard) in split.shards.iter().enumerate() {
                if shard.index != position as u64 {
                    return Err(invalid_manifest(
                        path,
                        format!(
                            "{} shard index {} is not the next index {position}",
                            expected, shard.index
                        ),
                    ));
                }
                shard
                    .counts
                    .validate(path, &format!("{} shard {}", expected, shard.index))?;
                if shard.counts.episodes == 0 || shard.counts.decisions == 0 {
                    return Err(invalid_manifest(
                        path,
                        format!("{} shard {} is empty", expected, shard.index),
                    ));
                }
                sum.merge(&shard.counts);
                validate_shard_arrays(path, expected, shard, &mut files)?;
            }
            if sum != split.counts {
                return Err(invalid_manifest(
                    path,
                    format!("{} counts do not equal the sum of its shards", expected),
                ));
            }
            total.merge(&split.counts);
        }
        if total != self.counts {
            return Err(invalid_manifest(
                path,
                "dataset counts do not equal the sum of split counts",
            ));
        }
        if !is_lower_hex_sha256(&self.cache_id) {
            return Err(invalid_manifest(
                path,
                "cache_id is not 64 lowercase hex digits",
            ));
        }
        let computed = compute_cache_id(self);
        if computed != self.cache_id {
            return Err(PrepareError::CacheIdMismatch {
                expected: self.cache_id.clone(),
                found: computed,
            });
        }
        Ok(())
    }

    /// Total number of output shards.
    pub fn total_shards(&self) -> usize {
        self.splits.iter().map(|split| split.shards.len()).sum()
    }
}

fn validate_shard_arrays(
    path: &Path,
    split: Split,
    shard: &ShardManifest,
    files: &mut BTreeSet<String>,
) -> Result<()> {
    if shard.arrays.len() != ARRAY_NAMES.len() {
        return Err(invalid_manifest(
            path,
            format!(
                "{} shard {} lists {} arrays, expected {}",
                split,
                shard.index,
                shard.arrays.len(),
                ARRAY_NAMES.len()
            ),
        ));
    }
    let decisions = shard.counts.decisions;
    let candidates = shard.counts.candidates;
    for name in ARRAY_NAMES {
        let entry = shard.arrays.get(name).ok_or_else(|| {
            invalid_manifest(
                path,
                format!("{} shard {} is missing array {name}", split, shard.index),
            )
        })?;
        let (dtype, shape) = expected_array(name, decisions, candidates);
        if entry.dtype != dtype {
            return Err(invalid_manifest(
                path,
                format!(
                    "{} shard {} array {name} dtype is {:?}, expected {dtype:?}",
                    split, shard.index, entry.dtype
                ),
            ));
        }
        if entry.shape != shape {
            return Err(invalid_manifest(
                path,
                format!(
                    "{} shard {} array {name} shape is {:?}, expected {shape:?}",
                    split, shard.index, entry.shape
                ),
            ));
        }
        if entry.bytes == 0 {
            return Err(invalid_manifest(
                path,
                format!(
                    "{} shard {} array {name} has zero bytes",
                    split, shard.index
                ),
            ));
        }
        if !is_lower_hex_sha256(&entry.sha256) {
            return Err(invalid_manifest(
                path,
                format!(
                    "{} shard {} array {name} sha256 is not 64 lowercase hex digits",
                    split, shard.index
                ),
            ));
        }
        let expected_file = format!("{ARRAYS_DIR}/{}/{:05}.{name}.npy", split, shard.index);
        if entry.file != expected_file {
            return Err(invalid_manifest(
                path,
                format!(
                    "{} shard {} array {name} file is {:?}, expected {expected_file:?}",
                    split, shard.index, entry.file
                ),
            ));
        }
        validate_relative_file(path, &entry.file)?;
        if !files.insert(entry.file.clone()) {
            return Err(invalid_manifest(
                path,
                format!("duplicate array file {:?}", entry.file),
            ));
        }
    }
    Ok(())
}

pub(super) fn expected_array(
    name: &str,
    decisions: u64,
    candidates: u64,
) -> (&'static str, Vec<u64>) {
    match name {
        "states" => ("float32", vec![decisions, STATE_WIDTH as u64]),
        "candidate_features" => ("float32", vec![candidates, ACTION_WIDTH as u64]),
        "candidate_offsets" => ("int64", vec![decisions + 1]),
        "targets" => ("int64", vec![decisions]),
        "episode_seeds" => ("uint64", vec![decisions]),
        "decision_indices" => ("uint64", vec![decisions]),
        "actor_seats" => ("uint8", vec![decisions]),
        "stages" => ("uint8", vec![decisions]),
        "target_kinds" => ("uint8", vec![decisions]),
        _ => unreachable!("array names are fixed"),
    }
}

fn validate_relative_file(path: &Path, file: &str) -> Result<()> {
    let candidate = Path::new(file);
    if candidate.is_absolute() {
        return Err(invalid_manifest(path, "array path must be relative"));
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            _ => {
                return Err(invalid_manifest(
                    path,
                    format!("array path {file:?} contains a non-normal component"),
                ))
            }
        }
    }
    Ok(())
}

/// Computes the content identity of a manifest.
///
/// ```text
/// sha256(
///   "bc-dataset-v1\n"
///   "encoding_version=<u32>\n"
///   "preparation_format_version=<u32>\n"
///   "preparation_version=<u32>\n"
///   "source_revision=<hex>\n"
///   "split_algorithm=<text>\n"
///   "corpus_manifest_sha256=<hex>\n"
///   "dataset episodes=<n> decisions=<n> candidates=<n> forced_decisions=<n> eligible_decisions=<n> min_candidates=<n> max_candidates=<n>\n"
///   for train, validation, test:
///     "split=<name> episodes=<n> decisions=<n> candidates=<n> forced_decisions=<n> eligible_decisions=<n> min_candidates=<n> max_candidates=<n>\n"
///     for shard in creation order:
///       "shard=<index> episodes=<n> decisions=<n> candidates=<n> forced_decisions=<n> eligible_decisions=<n> min_candidates=<n> max_candidates=<n>\n"
///       for name in states,candidate_features,candidate_offsets,targets,
///                   episode_seeds,decision_indices,actor_seats,stages,target_kinds:
///         "array=<name> dtype=<dtype> shape=<d0,d1,...> bytes=<n> sha256=<hex>\n"
/// )
/// ```
///
/// Build, environment, timing, and filesystem paths are deliberately excluded.
pub fn compute_cache_id(manifest: &DatasetManifest) -> String {
    let mut text = String::new();
    text.push_str("bc-dataset-v1\n");
    let _ = writeln!(text, "encoding_version={}", manifest.encoding_version);
    let _ = writeln!(
        text,
        "preparation_format_version={}",
        manifest.preparation_format_version
    );
    let _ = writeln!(text, "preparation_version={}", manifest.preparation_version);
    let _ = writeln!(text, "source_revision={}", manifest.source_revision);
    let _ = writeln!(text, "split_algorithm={}", manifest.split_algorithm);
    let _ = writeln!(
        text,
        "corpus_manifest_sha256={}",
        manifest.corpus_manifest_sha256
    );
    let dataset = &manifest.counts;
    let _ = writeln!(
        text,
        "dataset episodes={} decisions={} candidates={} forced_decisions={} eligible_decisions={} min_candidates={} max_candidates={}",
        dataset.episodes,
        dataset.decisions,
        dataset.candidates,
        dataset.forced_decisions,
        dataset.eligible_decisions,
        dataset.min_candidates,
        dataset.max_candidates
    );
    for split in [Split::Train, Split::Validation, Split::Test] {
        let split_manifest = manifest.splits.get(split);
        let counts = &split_manifest.counts;
        let _ = writeln!(
            text,
            "split={} episodes={} decisions={} candidates={} forced_decisions={} eligible_decisions={} min_candidates={} max_candidates={}",
            split,
            counts.episodes,
            counts.decisions,
            counts.candidates,
            counts.forced_decisions,
            counts.eligible_decisions,
            counts.min_candidates,
            counts.max_candidates
        );
        for shard in &split_manifest.shards {
            let _ = writeln!(
                text,
                "shard={} episodes={} decisions={} candidates={} forced_decisions={} eligible_decisions={} min_candidates={} max_candidates={}",
                shard.index,
                shard.counts.episodes,
                shard.counts.decisions,
                shard.counts.candidates,
                shard.counts.forced_decisions,
                shard.counts.eligible_decisions,
                shard.counts.min_candidates,
                shard.counts.max_candidates
            );
            for name in ARRAY_NAMES {
                let entry = shard
                    .arrays
                    .get(name)
                    .expect("manifest validation requires every array");
                let shape = entry
                    .shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let _ = writeln!(
                    text,
                    "array={} dtype={} shape={} bytes={} sha256={}",
                    name, entry.dtype, shape, entry.bytes, entry.sha256
                );
            }
        }
    }
    lower_hex(&Sha256::digest(text.as_bytes()))
}

/// Reads and validates a prepared cache manifest without reading arrays.
pub fn read_dataset_manifest(root: impl AsRef<Path>) -> Result<DatasetManifest> {
    let root = root.as_ref();
    let path = root.join(DATASET_FILE);
    let bytes = fs::read(&path).map_err(|error| io(&path, error))?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&bytes).map_err(|source| PrepareError::DatasetJson {
            path: path.clone(),
            source,
        })?;
    manifest.validate_at(&path)?;
    Ok(manifest)
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).expect("hex digit"));
        out.push(char::from_digit(u32::from(byte & 0xf), 16).expect("hex digit"));
    }
    out
}
