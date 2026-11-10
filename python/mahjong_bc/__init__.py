"""Shared Sichuan mahjong behavior-cloning policy and offline training tools.

The package boundary mirrors the offline-training specification: the Rust
neural crate owns encoding, corpus collection, and NumPy cache preparation;
this package owns candidate scoring, batching, optimization, metrics,
checkpoint recovery, and frozen offline evaluation. PyO3/environment bindings
and online game evaluation belong to the companion Change.
"""

from __future__ import annotations

from .cache import (
    ARRAY_NAMES,
    CacheArray,
    CacheCounts,
    CacheShard,
    CacheSplit,
    PreparedCache,
    load_cache_split,
    load_prepared_cache,
)
from .checkpoint import (
    LoadedCheckpoint,
    TrainingCheckpointState,
    load_checkpoint,
    load_policy_checkpoint,
    save_checkpoint,
)
from .config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from .constants import (
    ACTION_KIND_NAMES,
    CANDIDATE_WIDTH,
    CHECKPOINT_FORMAT_VERSION,
    ENCODING_VERSION,
    MODEL_FORMAT_VERSION,
    PACKAGE_VERSION,
    PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
    SPLIT_ALGORITHM,
    STAGE_NAMES,
    STATE_WIDTH,
)
from .data import (
    DecisionBatch,
    DecisionDataset,
    DecisionSample,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    collate_decisions,
    contains_memory_mapped_arrays,
    iter_batches,
)
from .errors import (
    CacheContractError,
    CheckpointError,
    ConfigurationError,
    DataError,
    MahjongBCError,
    ResumeError,
)
from .metrics import (
    ActionKindMetrics,
    MetricAccumulator,
    MetricsReport,
    StageMetrics,
    evaluate_dataset,
)
from .policy import (
    CandidatePolicy,
    build_policy,
    decision_nll,
    greedy_indices,
    masked_cross_entropy,
    sample_indices,
)
from .train import Trainer, TrainingSummary, evaluate_checkpoint, resolve_device

__version__ = PACKAGE_VERSION

__all__ = [
    "ACTION_KIND_NAMES",
    "ARRAY_NAMES",
    "ActionKindMetrics",
    "CacheArray",
    "CacheContractError",
    "CacheCounts",
    "CacheShard",
    "CacheSplit",
    "CandidatePolicy",
    "CANDIDATE_WIDTH",
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "ConfigurationError",
    "DataError",
    "DatasetIdentity",
    "DecisionBatch",
    "DecisionDataset",
    "DecisionSample",
    "DependencyVersions",
    "ENCODING_VERSION",
    "LoadedCheckpoint",
    "MODEL_FORMAT_VERSION",
    "MahjongBCError",
    "MetricAccumulator",
    "MetricsReport",
    "ModelConfig",
    "PACKAGE_VERSION",
    "PREPARATION_FORMAT_VERSION",
    "PREPARATION_VERSION",
    "PackedDecisionDataset",
    "PreparedCache",
    "ResumeError",
    "SPLIT_ALGORITHM",
    "STAGE_NAMES",
    "STATE_WIDTH",
    "ShardedDecisionDataset",
    "StageMetrics",
    "Trainer",
    "TrainingCheckpointState",
    "TrainingConfig",
    "TrainingSummary",
    "build_policy",
    "collate_decisions",
    "contains_memory_mapped_arrays",
    "collect_dependency_versions",
    "decision_nll",
    "evaluate_checkpoint",
    "evaluate_dataset",
    "greedy_indices",
    "iter_batches",
    "load_cache_split",
    "load_checkpoint",
    "load_policy_checkpoint",
    "load_prepared_cache",
    "masked_cross_entropy",
    "resolve_device",
    "sample_indices",
    "save_checkpoint",
    "__version__",
]
