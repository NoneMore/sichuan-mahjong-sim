//! Offline behavior-cloning cache preparation.
//!
//! [`prepare`] consumes one [`CorpusSnapshot`](crate::collection::CorpusSnapshot)
//! and publishes split-specific, sharded NumPy arrays plus `dataset.json`.
//! The source corpus is read exactly like collection `verify`: one captured
//! manifest, immutable listed shards, strict records, bounds, and SHA-256. No
//! second JSONL parser exists here.
//!
//! # Cache layout
//!
//! ```text
//! <cache-root>/
//!   dataset.json
//!   arrays/<split>/<00000>.<array>.npy
//! ```
//!
//! Each output shard belongs to one split and contains:
//!
//! | Array | Dtype | Shape |
//! | --- | --- | --- |
//! | `states` | float32 | `[D, 1013]` |
//! | `candidate_features` | float32 | `[C, 39]` |
//! | `candidate_offsets` | int64 | `[D + 1]` |
//! | `targets` | int64 | `[D]` |
//! | `episode_seeds` | uint64 | `[D]` |
//! | `decision_indices` | uint64 | `[D]` |
//! | `actor_seats` | uint8 | `[D]` |
//! | `stages` | uint8 | `[D]` |
//! | `target_kinds` | uint8 | `[D]` |
//!
//! Files are little-endian, C order. `episode_seeds` plus the dataset-level
//! `source_revision` identifies each episode. `target_kinds` is the teacher's
//! chosen action kind ID (`0..8`) and supports per-kind metrics without
//! re-reading candidate features. `candidate_offsets` spans `[0, C]`, every
//! decision has at least one candidate, and `targets` index their own local
//! range.
//!
//! # Publication and memory
//!
//! Output is streamed into a staging directory and atomically renamed to the
//! final path only after every committed shard validates and `dataset.json` is
//! written. An existing output path is rejected. Each of the three splits keeps
//! at most one active output buffer, bounded by [`PrepareOptions`] or one larger
//! episode. One additional input episode is parsed at a time. Shards rotate at
//! episode boundaries; the corpus decision limit bounds an episode, including
//! when it must form an oversized output shard.
//!
//! # Identity and provenance
//!
//! `cache_id` is computed by [`compute_cache_id`] from the preparation and
//! encoding versions, the captured corpus manifest hash and source identity,
//! and every listed array's dtype, shape, byte length, and SHA-256. Build,
//! environment, timing, and filesystem paths are excluded. Collection
//! provenance records only facts already present in the corpus manifest;
//! preparation build facts are recorded separately and may truthfully say
//! `dirty`, which is not a collection attestation.

mod error;
mod manifest;
mod npy;
mod verify;
mod writer;

pub use error::{PrepareError, Result};
pub use manifest::{
    compute_cache_id, read_dataset_manifest, ArrayEntry, CollectionProvenance, Counts,
    DatasetManifest, Layout, LayoutArray, PreparationProvenance, Provenance, ShardManifest,
    SplitManifest, Splits, ARRAYS_DIR, ARRAY_NAMES, DATASET_FILE, DATASET_FORMAT,
    DEFAULT_MAX_CANDIDATES_PER_SHARD, DEFAULT_MAX_DECISIONS_PER_SHARD, PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
};
pub use verify::verify_dataset;
pub use writer::{prepare, PrepareOptions};
