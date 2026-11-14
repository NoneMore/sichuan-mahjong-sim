"""Shared Sichuan mahjong behavior-cloning policy and offline training tools.

The Rust neural crate owns encoding, corpus collection, and NumPy cache
preparation. This package validates caches, exposes sharded memory-mapped
decisions, collates padded PyTorch batches, trains a candidate-scoring policy
with epoch-boundary recovery, and loads versioned policy checkpoints for frozen
evaluation. PyO3/environment bindings and online game evaluation remain out of
scope.
"""

from __future__ import annotations

from ._input_pipeline import BatchTransferConfig
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
    build_payload,
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
    DATASET_FORMAT,
    ENCODING_VERSION,
    FORCED_CANDIDATE_COUNT,
    MODEL_FORMAT_VERSION,
    PACKAGE_VERSION,
    PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
    SPLIT_ALGORITHM,
    SPLIT_NAMES,
    STAGE_NAMES,
    STATE_WIDTH,
)
from .data import (
    DecisionBatch,
    DecisionDataset,
    DecisionInputs,
    DecisionSample,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    collate_decisions,
    contains_memory_mapped_arrays,
    iter_batches,
    pad_decision_inputs,
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
    "BatchTransferConfig",
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
    "DATASET_FORMAT",
    "DataError",
    "DatasetIdentity",
    "DecisionBatch",
    "DecisionDataset",
    "DecisionInputs",
    "DecisionSample",
    "DependencyVersions",
    "ENCODING_VERSION",
    "FORCED_CANDIDATE_COUNT",
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
    "ResumeError",
    "PreparedCache",
    "SPLIT_ALGORITHM",
    "SPLIT_NAMES",
    "STAGE_NAMES",
    "STATE_WIDTH",
    "ShardedDecisionDataset",
    "StageMetrics",
    "Trainer",
    "TrainingCheckpointState",
    "TrainingConfig",
    "TrainingSummary",
    "build_payload",
    "build_policy",
    "collate_decisions",
    "collect_dependency_versions",
    "contains_memory_mapped_arrays",
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
    "pad_decision_inputs",
    "resolve_device",
    "sample_indices",
    "save_checkpoint",
    "__version__",
]
