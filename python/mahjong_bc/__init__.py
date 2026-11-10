"""Prepared Sichuan mahjong behavior-cloning datasets and decision batches.

The Rust neural crate owns decision encoding, corpus collection, and NumPy cache
preparation. This package validates those caches, exposes sharded memory-mapped
decision datasets, and collates padded PyTorch batches. Policy scoring,
training, checkpoint recovery, and evaluation are later deliveries and are not
imported here; PyO3 bindings and online play are out of scope.
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
from .config import DatasetIdentity, DependencyVersions, collect_dependency_versions
from .constants import (
    ACTION_KIND_NAMES,
    CANDIDATE_WIDTH,
    DATASET_FORMAT,
    ENCODING_VERSION,
    FORCED_CANDIDATE_COUNT,
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
    DecisionSample,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    collate_decisions,
    contains_memory_mapped_arrays,
    iter_batches,
)
from .errors import (
    CacheContractError,
    ConfigurationError,
    DataError,
    MahjongBCError,
)

__version__ = PACKAGE_VERSION

__all__ = [
    "ACTION_KIND_NAMES",
    "ARRAY_NAMES",
    "CacheArray",
    "CacheContractError",
    "CacheCounts",
    "CacheShard",
    "CacheSplit",
    "CANDIDATE_WIDTH",
    "ConfigurationError",
    "DATASET_FORMAT",
    "DataError",
    "DatasetIdentity",
    "DecisionBatch",
    "DecisionDataset",
    "DecisionSample",
    "DependencyVersions",
    "ENCODING_VERSION",
    "FORCED_CANDIDATE_COUNT",
    "MahjongBCError",
    "PACKAGE_VERSION",
    "PREPARATION_FORMAT_VERSION",
    "PREPARATION_VERSION",
    "PackedDecisionDataset",
    "PreparedCache",
    "SPLIT_ALGORITHM",
    "SPLIT_NAMES",
    "STAGE_NAMES",
    "STATE_WIDTH",
    "ShardedDecisionDataset",
    "collate_decisions",
    "collect_dependency_versions",
    "contains_memory_mapped_arrays",
    "iter_batches",
    "load_cache_split",
    "load_prepared_cache",
    "__version__",
]
