//! Bounded preparation from a captured corpus snapshot into a version 1 cache.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::Path;

use tempfile::Builder;

use crate::collection::{
    action_kind_id, build_info, environment_info, stage_id, CorpusSnapshot, Episode,
    Result as CollectionResult, SPLIT_ALGORITHM,
};
use crate::encoding::{ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};

use super::error::{into_collection, io, PrepareError, Result};
use super::manifest::{
    compute_cache_id, ArrayEntry, CollectionProvenance, Counts, DatasetManifest, Layout,
    PreparationProvenance, Provenance, ShardManifest, SplitManifest, Splits, ARRAYS_DIR,
    DATASET_FILE, DATASET_FORMAT, DEFAULT_MAX_CANDIDATES_PER_SHARD,
    DEFAULT_MAX_DECISIONS_PER_SHARD, PREPARATION_FORMAT_VERSION, PREPARATION_VERSION,
};
use super::npy::{self, NpyDtype};

/// Bounded output-shard configuration.
///
/// Limits control buffering, not dataset semantics. Each of the three splits
/// maintains at most one active output buffer, bounded by the configured limits
/// or one larger episode. One additional input episode is parsed at a time.
/// Shards rotate only at episode boundaries; the corpus decision limit bounds
/// an episode, including when it must form an oversized output shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepareOptions {
    /// Maximum decisions buffered per output shard.
    pub max_decisions_per_shard: usize,
    /// Maximum packed candidates buffered per output shard.
    pub max_candidates_per_shard: usize,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            max_decisions_per_shard: DEFAULT_MAX_DECISIONS_PER_SHARD,
            max_candidates_per_shard: DEFAULT_MAX_CANDIDATES_PER_SHARD,
        }
    }
}

impl PrepareOptions {
    /// Requires positive limits.
    pub fn validate(&self) -> Result<()> {
        if self.max_decisions_per_shard == 0 {
            return Err(PrepareError::InvalidOptions {
                message: "max_decisions_per_shard must be positive".to_owned(),
            });
        }
        if self.max_candidates_per_shard == 0 {
            return Err(PrepareError::InvalidOptions {
                message: "max_candidates_per_shard must be positive".to_owned(),
            });
        }
        Ok(())
    }
}

/// Prepares a cache from one captured snapshot and publishes it atomically.
///
/// `output_root` must not exist. All arrays are written under a temporary
/// staging directory in the output parent; the final directory is renamed only
/// after every captured shard validates and `dataset.json` is written. A
/// failure leaves no final cache directory. The cache contains no absolute
/// corpus paths and can be memory-mapped after the source corpus is gone.
pub fn prepare(
    snapshot: &CorpusSnapshot,
    output_root: impl AsRef<Path>,
    options: &PrepareOptions,
) -> Result<DatasetManifest> {
    options.validate()?;
    let output_root = output_root.as_ref();
    match fs::symlink_metadata(output_root) {
        Ok(_) => {
            return Err(PrepareError::OutputExists {
                path: output_root.to_owned(),
            })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io(output_root, error)),
    }
    if snapshot.manifest().shards.is_empty() {
        return Err(PrepareError::EmptyCorpus);
    }

    let parent = output_parent(output_root);
    fs::create_dir_all(parent).map_err(|error| io(parent, error))?;
    let staging = Builder::new()
        .prefix(".bc-prepare-")
        .suffix(".tmp")
        .tempdir_in(parent)
        .map_err(|error| io(parent, error))?;
    let staging_root = staging.path().to_owned();
    let arrays_root = staging_root.join(ARRAYS_DIR);
    fs::create_dir_all(&arrays_root).map_err(|error| io(&arrays_root, error))?;
    for split in crate::collection::Split::ALL {
        let path = arrays_root.join(split.as_str());
        fs::create_dir_all(&path).map_err(|error| io(&path, error))?;
    }

    let mut outputs: [SplitOutput; 3] =
        std::array::from_fn(|index| SplitOutput::new(crate::collection::Split::ALL[index]));
    for index in 0..snapshot.manifest().shards.len() {
        snapshot
            .visit_shard(index, |episode| {
                outputs[episode.split.index()].push_episode(episode, &staging_root, options)
            })
            .map_err(PrepareError::from)?;
    }
    for output in &mut outputs {
        output
            .flush(&staging_root)
            .map_err(PrepareError::Collection)?;
    }

    let mut counts = Counts::default();
    for output in &outputs {
        counts.merge(&output.counts);
    }
    let splits = Splits {
        train: outputs[0].manifest(),
        validation: outputs[1].manifest(),
        test: outputs[2].manifest(),
    };

    let build = build_info();
    let environment = environment_info();
    let mut manifest = DatasetManifest {
        format: DATASET_FORMAT.to_owned(),
        preparation_format_version: PREPARATION_FORMAT_VERSION,
        preparation_version: PREPARATION_VERSION,
        encoding_version: ENCODING_VERSION,
        cache_id: String::new(),
        split_algorithm: SPLIT_ALGORITHM.to_owned(),
        source_revision: snapshot.manifest().source_revision.clone(),
        corpus_manifest_sha256: snapshot.manifest_sha256().to_owned(),
        corpus_manifest: snapshot.manifest().clone(),
        layout: Layout::expected(),
        provenance: Provenance {
            collection: CollectionProvenance {
                source_revision: snapshot.manifest().source_revision.clone(),
                encoding_version: snapshot.manifest().encoding_version,
                split_algorithm: snapshot.manifest().split_algorithm.clone(),
            },
            preparation: PreparationProvenance {
                build_attestation: build_attestation(&build).to_owned(),
                build,
                environment,
                max_decisions_per_shard: options.max_decisions_per_shard,
                max_candidates_per_shard: options.max_candidates_per_shard,
            },
        },
        counts,
        splits,
    };
    manifest.cache_id = compute_cache_id(&manifest);
    manifest.validate()?;

    let dataset_path = staging_root.join(DATASET_FILE);
    let mut bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|error| PrepareError::DatasetJson {
            path: dataset_path.clone(),
            source: error,
        })?;
    bytes.push(b'\n');
    let mut file = fs::File::create(&dataset_path).map_err(|error| io(&dataset_path, error))?;
    file.write_all(&bytes)
        .map_err(|error| io(&dataset_path, error))?;
    file.sync_all().map_err(|error| io(&dataset_path, error))?;
    drop(file);

    let staging_path = staging.keep();
    match fs::rename(&staging_path, output_root) {
        Ok(()) => Ok(manifest),
        Err(error) => {
            let _ = fs::remove_dir_all(&staging_path);
            Err(io(output_root, error))
        }
    }
}

fn output_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn build_attestation(build: &crate::collection::BuildInfo) -> &'static str {
    if !build.git_available || build.git_commit.is_none() {
        "unavailable"
    } else if build.git_dirty {
        "dirty"
    } else {
        "clean"
    }
}

struct SplitOutput {
    split: crate::collection::Split,
    counts: Counts,
    shards: Vec<ShardManifest>,
    active: Option<ShardBuffer>,
}

impl SplitOutput {
    fn new(split: crate::collection::Split) -> Self {
        Self {
            split,
            counts: Counts::default(),
            shards: Vec::new(),
            active: None,
        }
    }

    fn push_episode(
        &mut self,
        episode: Episode,
        staging_root: &Path,
        options: &PrepareOptions,
    ) -> CollectionResult<()> {
        let episode_decisions = episode.decisions.len();
        let episode_candidates: usize = episode
            .decisions
            .iter()
            .map(|sample| sample.candidate_count())
            .sum();
        if let Some(active) = &self.active {
            let exceeds_decisions = active
                .counts
                .decisions
                .saturating_add(episode_decisions as u64)
                > options.max_decisions_per_shard as u64;
            let exceeds_candidates = active
                .counts
                .candidates
                .saturating_add(episode_candidates as u64)
                > options.max_candidates_per_shard as u64;
            if exceeds_decisions || exceeds_candidates {
                self.flush(staging_root)?;
            }
        }
        if self.active.is_none() {
            self.active = Some(ShardBuffer::new(self.split, self.shards.len() as u64));
        }
        self.active
            .as_mut()
            .expect("active output shard exists")
            .push_episode(&episode);
        self.counts.add_episode(&episode);
        Ok(())
    }

    fn flush(&mut self, staging_root: &Path) -> CollectionResult<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let shard = active.finish(staging_root).map_err(into_collection)?;
        self.shards.push(shard);
        Ok(())
    }

    fn manifest(&self) -> SplitManifest {
        SplitManifest {
            split: self.split,
            counts: self.counts.clone(),
            shards: self.shards.clone(),
        }
    }
}

struct ShardBuffer {
    split: crate::collection::Split,
    index: u64,
    counts: Counts,
    states: Vec<f32>,
    candidate_features: Vec<f32>,
    candidate_offsets: Vec<i64>,
    targets: Vec<i64>,
    episode_seeds: Vec<u64>,
    decision_indices: Vec<u64>,
    actor_seats: Vec<u8>,
    stages: Vec<u8>,
    target_kinds: Vec<u8>,
}

impl ShardBuffer {
    fn new(split: crate::collection::Split, index: u64) -> Self {
        Self {
            split,
            index,
            counts: Counts::default(),
            states: Vec::new(),
            candidate_features: Vec::new(),
            candidate_offsets: vec![0],
            targets: Vec::new(),
            episode_seeds: Vec::new(),
            decision_indices: Vec::new(),
            actor_seats: Vec::new(),
            stages: Vec::new(),
            target_kinds: Vec::new(),
        }
    }

    fn push_episode(&mut self, episode: &Episode) {
        for sample in &episode.decisions {
            debug_assert_eq!(sample.state.len(), STATE_WIDTH);
            debug_assert_eq!(sample.candidate_features.len(), sample.candidates.len());
            for features in &sample.candidate_features {
                debug_assert_eq!(features.len(), ACTION_WIDTH);
                self.candidate_features.extend_from_slice(features);
            }
            self.counts.observe_decision(sample.candidate_count());
            self.states.extend_from_slice(&sample.state);
            self.candidate_offsets.push(self.counts.candidates as i64);
            self.targets.push(sample.target_index as i64);
            self.episode_seeds.push(episode.id.seed.as_u64());
            self.decision_indices.push(sample.decision_index);
            self.actor_seats.push(sample.actor_seat.index() as u8);
            self.stages.push(stage_id(sample.stage));
            self.target_kinds
                .push(action_kind_id(&sample.candidates[sample.target_index]));
        }
        self.counts.observe_episode();
    }

    fn finish(self, staging_root: &Path) -> Result<ShardManifest> {
        debug_assert!(self.counts.decisions > 0);
        let decisions = self.counts.decisions;
        let candidates = self.counts.candidates;
        let mut arrays = BTreeMap::new();
        let split = self.split.as_str();
        let mut write =
            |name: &str, dtype: NpyDtype, shape: Vec<u64>, encoded: EncodedArray| -> Result<()> {
                let file = format!("{ARRAYS_DIR}/{split}/{:05}.{name}.npy", self.index);
                let path = staging_root.join(&file);
                let (sha256, file_bytes) = match encoded {
                    EncodedArray::Float32(values) => npy::write_array(&path, &shape, &values)?,
                    EncodedArray::Int64(values) => npy::write_array(&path, &shape, &values)?,
                    EncodedArray::Uint64(values) => npy::write_array(&path, &shape, &values)?,
                    EncodedArray::Uint8(values) => npy::write_array(&path, &shape, &values)?,
                };
                arrays.insert(
                    name.to_owned(),
                    ArrayEntry {
                        file,
                        dtype: dtype.label().to_owned(),
                        shape,
                        bytes: file_bytes,
                        sha256,
                    },
                );
                Ok(())
            };

        let states_shape = vec![decisions, STATE_WIDTH as u64];
        write(
            "states",
            NpyDtype::Float32,
            states_shape,
            EncodedArray::Float32(self.states),
        )?;
        let candidates_shape = vec![candidates, ACTION_WIDTH as u64];
        write(
            "candidate_features",
            NpyDtype::Float32,
            candidates_shape,
            EncodedArray::Float32(self.candidate_features),
        )?;
        write(
            "candidate_offsets",
            NpyDtype::Int64,
            vec![decisions + 1],
            EncodedArray::Int64(self.candidate_offsets),
        )?;
        write(
            "targets",
            NpyDtype::Int64,
            vec![decisions],
            EncodedArray::Int64(self.targets),
        )?;
        write(
            "episode_seeds",
            NpyDtype::Uint64,
            vec![decisions],
            EncodedArray::Uint64(self.episode_seeds),
        )?;
        write(
            "decision_indices",
            NpyDtype::Uint64,
            vec![decisions],
            EncodedArray::Uint64(self.decision_indices),
        )?;
        write(
            "actor_seats",
            NpyDtype::Uint8,
            vec![decisions],
            EncodedArray::Uint8(self.actor_seats),
        )?;
        write(
            "stages",
            NpyDtype::Uint8,
            vec![decisions],
            EncodedArray::Uint8(self.stages),
        )?;
        write(
            "target_kinds",
            NpyDtype::Uint8,
            vec![decisions],
            EncodedArray::Uint8(self.target_kinds),
        )?;

        Ok(ShardManifest {
            index: self.index,
            counts: self.counts,
            arrays,
        })
    }
}

enum EncodedArray {
    Float32(Vec<f32>),
    Int64(Vec<i64>),
    Uint64(Vec<u64>),
    Uint8(Vec<u8>),
}
