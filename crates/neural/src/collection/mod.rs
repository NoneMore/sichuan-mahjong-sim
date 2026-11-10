//! Behavior-cloning teacher data collection.
//!
//! The collector drives four-seat [`HeuristicPolicy`](sichuan_mahjong_sim::HeuristicPolicy)
//! self-play through the public [`Game`](sichuan_mahjong_sim::Game),
//! [`Policy`](sichuan_mahjong_sim::Policy), and [`crate::encoding`] interfaces.
//! For each original decision it encodes the pre-action state, asks the teacher
//! once, resolves the semantic choice to a canonical candidate index, submits
//! that candidate, and records the sample only after acceptance. Forced
//! decisions, passes, and losing games are retained without filtering.
//!
//! # Corpus layout
//!
//! ```text
//! <root>/
//!   LOCK                    exclusive, never unlinked
//!   manifest.json           atomically replaced commit record
//!   shards/<first_seed>.jsonl
//! ```
//!
//! # JSONL v1
//!
//! [`Manifest`] stores `format_version=1`, `source_revision`,
//! `encoding_version=1`, `split_algorithm`, `seed_start`, `decision_limit`,
//! `shard_games`, and a `shards` list. Each [`ShardEntry`] stores `first_seed`,
//! `games`, `decisions`, and `sha256`; ranges form one contiguous seed prefix.
//! Seeds serialize as sixteen lowercase hex digits. [`seed_split`] defines
//! seed-only train/validation/test membership, independent of source revision.
//!
//! Each newline-terminated JSON object has a `type` tag:
//!
//! | Type | Fields |
//! | --- | --- |
//! | `episode` | `episode_id`: `[revision, encoding_version, seed]`; `split` |
//! | `decision` | `decision_index`, `actor_seat`, `stage`, `state`, `candidate_features`, `candidate_actions`, `teacher_target_index` |
//! | `result` | `decisions`, four seat-ordered `scores` and `winners`, `end_reason` |
//!
//! Decision indices start at zero; seats are `0..3` and stages use [`stage_id`].
//! State has 1013 finite `f32` values; candidates have nonempty `N × 39`
//! features and matching canonical actions, with target in `[0, N)`.
//! [`encode_action`] defines `[kind_id, ...parameters]`: three sorted tiles
//! for Exchange, one suit for ChooseMissing, one tile for Discard or a
//! concealed/added kong, and no parameters otherwise. IDs follow encoding v1.
//! End reasons are `three_winners` and `wall_exhausted`. Only state and candidate
//! features are model inputs; consumers own padding.
//!
//! A shard contains one contiguous seed range and complete episode blocks. A
//! writer streams games to a temporary file, validates the completed shard,
//! publishes it under `shards/`, and only then atomically replaces the
//! manifest. Readers use only manifest-listed immutable shards and verify
//! shapes, canonical actions, target bounds, splits, completeness, counts, and
//! SHA-256. Unlisted files are ignored; they are replayed from the first
//! uncommitted seed on retry. [`CorpusSnapshot`] captures one exact committed
//! manifest and exposes fallible streaming over only its listed shards, so
//! later appends cannot change a prepared dataset.
//!
//! # Provenance
//!
//! [`SourceRevision::from_compiled_source`] uses commit and dirty facts fixed
//! by the crate's build script. A dirty or unavailable build is rejected rather
//! than relabeled with a clean commit. Library callers may pass a validated
//! [`SourceRevision`] directly for controlled tests or external pipelines; the
//! CLI deliberately uses only compiled provenance.
//!
//! # Example
//!
//! ```
//! use sichuan_mahjong_neural::collection::{seed_split, Seed, Split};
//!
//! assert_eq!(seed_split(Seed::from_u64(158)), Split::Train);
//! assert_eq!(seed_split(Seed::from_u64(167)), Split::Validation);
//! assert_eq!(seed_split(Seed::from_u64(214)), Split::Test);
//! ```

mod corpus;
mod driver;
mod error;
mod format;
mod manifest;
mod report;
mod seed;
mod source;
mod stats;
mod storage;

pub use corpus::{
    read_corpus, read_manifest, report_corpus, verify_corpus, Corpus, CorpusSnapshot,
};
pub use error::{CollectionError, Result};
pub use format::{
    action_kind_id, decode_action, encode_action, stage_from_id, stage_id, DecisionSample, Episode,
    EpisodeId, ACTION_KIND_NAMES,
};
pub use manifest::{
    lock_path, manifest_path, shard_path, shards_dir, CollectionConfig, Manifest, ShardEntry,
    DEFAULT_SHARD_GAMES, FORMAT_VERSION, LOCK_FILE, MANIFEST_FILE, SHARDS_DIR,
};
pub use report::{CacheReport, CollectionOutcome, CollectionTimings, CorpusData, CorpusReport};
pub use seed::{seed_split, split_bucket, Seed, SeedError, Split, SPLIT_ALGORITHM};
pub use source::{
    build_info, environment_info, BuildInfo, EnvironmentInfo, SourceProvenanceError,
    SourceRevision, SourceRevisionError,
};
pub use stats::{CorpusStats, KindStats, SplitStats, StageStats};
pub use storage::CorpusLock;
