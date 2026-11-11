"""Strict reader for the versioned sharded NumPy prepared cache.

The Rust preparation side owns cache construction. Python validates the
manifest format, complete file inventory, checksums, array headers, structural
counts, episode metadata, and target alignment before exposing memory-mapped
split datasets. ``cache_id`` is compared as an opaque identity; this module
deliberately does not reimplement its serialization.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._array_contract import (
    ArrayContractViolation,
    expected_shapes,
    validate_packed_arrays,
)
from ._array_spec import (
    ARRAY_DIMENSIONS as _ARRAY_DIMENSIONS,
    ARRAY_DTYPES as _ARRAY_DTYPES,
    ARRAY_NAMES,
)
from .config import DatasetIdentity
from .constants import (
    ACTION_KIND_COUNT,
    ACTION_KIND_WIDTH,
    CANDIDATE_WIDTH,
    DATASET_FORMAT,
    ENCODING_VERSION,
    FORCED_CANDIDATE_COUNT,
    PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
    SPLIT_ALGORITHM,
    SPLIT_NAMES,
    STAGE_COUNT,
    STATE_WIDTH,
)
from .data import PackedDecisionDataset, ShardedDecisionDataset
from .errors import CacheContractError

_COUNT_KEYS = {
    "episodes",
    "decisions",
    "candidates",
    "forced_decisions",
    "eligible_decisions",
    "min_candidates",
    "max_candidates",
}
_DATASET_KEYS = {
    "format",
    "preparation_format_version",
    "preparation_version",
    "encoding_version",
    "cache_id",
    "split_algorithm",
    "source_revision",
    "corpus_manifest_sha256",
    "corpus_manifest",
    "layout",
    "provenance",
    "counts",
    "splits",
}
_SPLIT_KEYS = {"split", "counts", "shards"}
_SHARD_KEYS = {"index", "counts", "arrays"}
_ARRAY_KEYS = {"file", "dtype", "shape", "bytes", "sha256"}
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEED_TEXT = re.compile(r"^[0-9a-f]{16}$")
_CORPUS_FORMAT_VERSION = 1
_CORPUS_KEYS = {
    "format_version",
    "source_revision",
    "encoding_version",
    "split_algorithm",
    "seed_start",
    "decision_limit",
    "shard_games",
    "shards",
}
_CORPUS_SHARD_KEYS = {"first_seed", "games", "decisions", "sha256"}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheContractError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise CacheContractError(f"{name} has invalid keys: {', '.join(details)}")


def _require_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CacheContractError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    number = _require_integer(value, name)
    if number < 0:
        raise CacheContractError(f"{name} must be a non-negative integer")
    return number


def _require_positive_int(value: Any, name: str) -> int:
    number = _require_integer(value, name)
    if number <= 0:
        raise CacheContractError(f"{name} must be a positive integer")
    return number


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheContractError(f"{name} must be a nonempty string")
    return value


def _require_seed(value: Any, name: str) -> int:
    text = _require_nonempty_string(value, name)
    if _SEED_TEXT.fullmatch(text) is None:
        raise CacheContractError(f"{name} must be 16 lowercase hexadecimal digits")
    return int(text, 16)


def _seed_split(seed: int) -> str:
    """Frozen Rust ``seed_split`` bucket for one raw game seed."""

    digest = hashlib.sha256(b"bc-split-v1\0" + seed.to_bytes(8, "big")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _require_sha256(value: Any, name: str) -> str:
    text = _require_nonempty_string(value, name)
    if _LOWER_SHA256.fullmatch(text) is None:
        raise CacheContractError(f"{name} must be 64 lowercase hexadecimal digits")
    return text


@dataclass(frozen=True)
class CacheCounts:
    episodes: int
    decisions: int
    candidates: int
    forced_decisions: int
    eligible_decisions: int
    min_candidates: int
    max_candidates: int

    def __post_init__(self) -> None:
        for name in _COUNT_KEYS:
            value = getattr(self, name)
            _require_nonnegative_int(value, name)
        if self.forced_decisions + self.eligible_decisions != self.decisions:
            raise CacheContractError(
                "forced_decisions + eligible_decisions must equal decisions"
            )
        if self.candidates < self.decisions:
            raise CacheContractError("candidates cannot be smaller than decisions")
        if self.decisions == 0:
            if any(getattr(self, name) != 0 for name in _COUNT_KEYS):
                raise CacheContractError("an empty count must be all zero")
            return
        if self.episodes == 0:
            raise CacheContractError("a nonempty count must contain episodes")
        if self.min_candidates < FORCED_CANDIDATE_COUNT:
            raise CacheContractError("min_candidates must be at least one")
        if self.max_candidates < self.min_candidates:
            raise CacheContractError("max_candidates must be at least min_candidates")

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in (
            "episodes",
            "decisions",
            "candidates",
            "forced_decisions",
            "eligible_decisions",
            "min_candidates",
            "max_candidates",
        )}

    @classmethod
    def from_dict(cls, value: Any, name: str = "counts") -> CacheCounts:
        value = _require_mapping(value, name)
        _require_exact_keys(value, _COUNT_KEYS, name)
        return cls(
            episodes=_require_nonnegative_int(value["episodes"], f"{name}.episodes"),
            decisions=_require_nonnegative_int(
                value["decisions"], f"{name}.decisions"
            ),
            candidates=_require_nonnegative_int(
                value["candidates"], f"{name}.candidates"
            ),
            forced_decisions=_require_nonnegative_int(
                value["forced_decisions"], f"{name}.forced_decisions"
            ),
            eligible_decisions=_require_nonnegative_int(
                value["eligible_decisions"], f"{name}.eligible_decisions"
            ),
            min_candidates=_require_nonnegative_int(
                value["min_candidates"], f"{name}.min_candidates"
            ),
            max_candidates=_require_nonnegative_int(
                value["max_candidates"], f"{name}.max_candidates"
            ),
        )


def _sum_counts(counts: list[CacheCounts], name: str) -> CacheCounts:
    nonempty = [value for value in counts if value.decisions > 0]
    return CacheCounts(
        episodes=sum(value.episodes for value in counts),
        decisions=sum(value.decisions for value in counts),
        candidates=sum(value.candidates for value in counts),
        forced_decisions=sum(value.forced_decisions for value in counts),
        eligible_decisions=sum(value.eligible_decisions for value in counts),
        min_candidates=(
            min(value.min_candidates for value in nonempty) if nonempty else 0
        ),
        max_candidates=(
            max(value.max_candidates for value in nonempty) if nonempty else 0
        ),
    )


def _require_matching_identity(
    label: str,
    *,
    source_revision: str,
    encoding_version: int,
    split_algorithm: str,
    dataset_source_revision: str,
    dataset_encoding_version: int,
    dataset_split_algorithm: str,
) -> None:
    """Require a captured or provenance section to match the dataset top level."""

    if source_revision != dataset_source_revision:
        raise CacheContractError(f"{label} source_revision differs from the dataset")
    if encoding_version != dataset_encoding_version:
        raise CacheContractError(f"{label} encoding_version differs from the dataset")
    if split_algorithm != dataset_split_algorithm:
        raise CacheContractError(f"{label} split_algorithm differs from the dataset")


def _validate_captured_corpus(
    corpus: Mapping[str, Any],
    *,
    source_revision: str,
    encoding_version: int,
    split_algorithm: str,
    counts: CacheCounts,
) -> tuple[int, int]:
    """Validate the captured corpus manifest's basic protocol and totals.

    The cache carries no original manifest bytes, so
    ``corpus_manifest_sha256`` remains a declared 64-digit hash and is not
    rederived from a JSON re-encoding. Returns the captured seed start and
    total game count for episode-range and coverage checks.
    """

    _require_exact_keys(corpus, _CORPUS_KEYS, "corpus_manifest")
    format_version = _require_positive_int(
        corpus["format_version"], "corpus_manifest.format_version"
    )
    if format_version != _CORPUS_FORMAT_VERSION:
        raise CacheContractError(
            f"unsupported corpus format_version {format_version}; "
            f"supported version is {_CORPUS_FORMAT_VERSION}"
        )
    captured_revision = _require_nonempty_string(
        corpus["source_revision"], "corpus_manifest.source_revision"
    )
    captured_encoding = _require_positive_int(
        corpus["encoding_version"], "corpus_manifest.encoding_version"
    )
    captured_algorithm = _require_nonempty_string(
        corpus["split_algorithm"], "corpus_manifest.split_algorithm"
    )
    _require_matching_identity(
        "captured corpus",
        source_revision=captured_revision,
        encoding_version=captured_encoding,
        split_algorithm=captured_algorithm,
        dataset_source_revision=source_revision,
        dataset_encoding_version=encoding_version,
        dataset_split_algorithm=split_algorithm,
    )
    seed_start = _require_seed(corpus["seed_start"], "corpus_manifest.seed_start")
    _require_positive_int(
        corpus["decision_limit"], "corpus_manifest.decision_limit"
    )
    shard_games = _require_positive_int(
        corpus["shard_games"], "corpus_manifest.shard_games"
    )
    raw_shards = corpus["shards"]
    if not isinstance(raw_shards, list):
        raise CacheContractError("corpus_manifest.shards must be a list")

    next_seed = seed_start
    total_games = 0
    total_decisions = 0
    for index, raw_shard in enumerate(raw_shards):
        name = f"corpus_manifest.shards[{index}]"
        raw_shard = _require_mapping(raw_shard, name)
        _require_exact_keys(raw_shard, _CORPUS_SHARD_KEYS, name)
        first_seed = _require_seed(raw_shard["first_seed"], f"{name}.first_seed")
        games = _require_positive_int(raw_shard["games"], f"{name}.games")
        decisions = _require_positive_int(
            raw_shard["decisions"], f"{name}.decisions"
        )
        _require_sha256(raw_shard["sha256"], f"{name}.sha256")
        if games > shard_games:
            raise CacheContractError(
                f"{name} has {games} games, exceeding shard_games {shard_games}"
            )
        if first_seed != next_seed:
            raise CacheContractError(
                f"{name} starts at {first_seed:016x}, expected {next_seed:016x}"
            )
        next_seed += games
        if next_seed > 1 << 64:
            raise CacheContractError(f"{name} seed range exceeds u64::MAX")
        total_games += games
        total_decisions += decisions
    if counts.episodes != total_games:
        raise CacheContractError(
            f"dataset episodes {counts.episodes} differ from the captured "
            f"corpus total games {total_games}"
        )
    if counts.decisions != total_decisions:
        raise CacheContractError(
            f"dataset decisions {counts.decisions} differ from the captured "
            f"corpus total decisions {total_decisions}"
        )
    return seed_start, total_games


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    source_revision: str,
    encoding_version: int,
    split_algorithm: str,
) -> None:
    _require_exact_keys(provenance, {"collection", "preparation"}, "provenance")
    collection = _require_mapping(
        provenance["collection"], "provenance.collection"
    )
    _require_exact_keys(
        collection,
        {"source_revision", "encoding_version", "split_algorithm"},
        "provenance.collection",
    )
    collected_revision = _require_nonempty_string(
        collection["source_revision"], "provenance.collection.source_revision"
    )
    collected_encoding = _require_positive_int(
        collection["encoding_version"], "provenance.collection.encoding_version"
    )
    collected_algorithm = _require_nonempty_string(
        collection["split_algorithm"], "provenance.collection.split_algorithm"
    )
    _require_matching_identity(
        "collection provenance",
        source_revision=collected_revision,
        encoding_version=collected_encoding,
        split_algorithm=collected_algorithm,
        dataset_source_revision=source_revision,
        dataset_encoding_version=encoding_version,
        dataset_split_algorithm=split_algorithm,
    )
    _require_mapping(provenance["preparation"], "provenance.preparation")


@dataclass(frozen=True)
class CacheArray:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    file_bytes: int
    sha256: str

    @classmethod
    def from_dict(
        cls,
        name: str,
        value: Any,
        *,
        split: str,
        shard_index: int,
    ) -> CacheArray:
        if name not in ARRAY_NAMES:
            raise CacheContractError(f"unknown cache array {name!r}")
        value = _require_mapping(value, f"array {name!r}")
        _require_exact_keys(value, _ARRAY_KEYS, f"array {name!r}")
        relative = _require_nonempty_string(value["file"], f"array {name!r}.file")
        expected = f"arrays/{split}/{shard_index:05d}.{name}.npy"
        # Exact protocol equality is the containment check: a manifest path
        # cannot escape ``arrays/`` or name another shard's file.
        if relative != expected:
            raise CacheContractError(
                f"array {name!r}.file must be {expected!r}, got {relative!r}"
            )
        dtype = _require_nonempty_string(value["dtype"], f"array {name!r}.dtype")
        expected_dtype = _ARRAY_DTYPES[name]
        if dtype != expected_dtype:
            raise CacheContractError(
                f"array {name!r}.dtype must be {expected_dtype!r}, got {dtype!r}"
            )
        raw_shape = value["shape"]
        if not isinstance(raw_shape, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in raw_shape
        ):
            raise CacheContractError(
                f"array {name!r}.shape must be a list of non-negative integers"
            )
        file_bytes = _require_nonnegative_int(
            value["bytes"], f"array {name!r}.bytes"
        )
        sha256 = _require_sha256(value["sha256"], f"array {name!r}.sha256")
        return cls(
            name=name,
            file=relative,
            dtype=dtype,
            shape=tuple(raw_shape),
            file_bytes=file_bytes,
            sha256=sha256,
        )


@dataclass(frozen=True)
class CacheShard:
    index: int
    counts: CacheCounts
    arrays: Mapping[str, CacheArray]


@dataclass(frozen=True)
class CacheSplit:
    split: str
    counts: CacheCounts
    shards: tuple[CacheShard, ...]


@dataclass(frozen=True)
class PreparedCache:
    """A validated prepared cache, with lazily materialized split datasets."""

    root: Path
    manifest: Mapping[str, Any]
    cache_id: str
    encoding_version: int
    preparation_format_version: int
    preparation_version: int
    source_revision: str
    corpus_manifest_sha256: str
    split_algorithm: str
    counts: CacheCounts
    splits: Mapping[str, CacheSplit]
    layout: Mapping[str, Any]
    provenance: Mapping[str, Any]
    corpus_manifest: Mapping[str, Any]

    def split_identity(self, split: str) -> DatasetIdentity:
        entry = self._split_entry(split)
        return DatasetIdentity(
            cache_id=self.cache_id,
            split=split,
            encoding_version=self.encoding_version,
            preparation_format_version=self.preparation_format_version,
            preparation_version=self.preparation_version,
            source_revision=self.source_revision,
            decisions=entry.counts.decisions,
            eligible_decisions=entry.counts.eligible_decisions,
        )

    def _split_entry(self, split: str) -> CacheSplit:
        if split not in self.splits:
            raise CacheContractError(f"unknown split {split!r}")
        return self.splits[split]

    def load_split(self, split: str, *, mmap: bool = True) -> ShardedDecisionDataset:
        """Materialize one split as memory-mapped shards (or ordinary arrays)."""

        entry = self._split_entry(split)
        identity = self.split_identity(split)
        shards: list[PackedDecisionDataset] = []
        for shard in entry.shards:
            arrays: dict[str, np.ndarray] = {}
            for name in ARRAY_NAMES:
                descriptor = shard.arrays[name]
                path = self.root / descriptor.file
                try:
                    arrays[name] = np.load(
                        path,
                        mmap_mode="r" if mmap else None,
                        allow_pickle=False,
                    )
                except Exception as exc:
                    raise CacheContractError(
                        f"failed to load {descriptor.file}: {exc}"
                    ) from exc
            shards.append(
                PackedDecisionDataset(
                    states=arrays["states"],
                    candidate_features=arrays["candidate_features"],
                    offsets=arrays["candidate_offsets"],
                    targets=arrays["targets"],
                    stages=arrays["stages"],
                    identity=None,
                    actor_seats=arrays["actor_seats"],
                    decision_indices=arrays["decision_indices"],
                    episode_ids=arrays["episode_seeds"],
                    target_kinds=arrays["target_kinds"],
                )
            )
        return ShardedDecisionDataset(tuple(shards), identity)


def _read_json_no_duplicates(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CacheContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CacheContractError(f"failed to read {path}: {exc}") from exc
    def reject_constant(value: str) -> None:
        raise CacheContractError(f"non-finite JSON constant {value!r} in {path}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except CacheContractError:
        raise
    except Exception as exc:
        raise CacheContractError(f"failed to parse {path}: {exc}") from exc


def _sha256_file(path: Path, expected_bytes: int) -> str:
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise CacheContractError(f"failed to stat {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise CacheContractError(
            f"{path} is {actual_bytes} bytes, manifest declares {expected_bytes}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CacheContractError(f"failed to read {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_layout(layout: Mapping[str, Any]) -> None:
    _require_exact_keys(
        layout,
        {"state_width", "candidate_width", "stage_count", "action_kind_count", "arrays"},
        "layout",
    )
    expected_scalars = {
        "state_width": STATE_WIDTH,
        "candidate_width": CANDIDATE_WIDTH,
        "stage_count": STAGE_COUNT,
        "action_kind_count": ACTION_KIND_COUNT,
    }
    for name, expected in expected_scalars.items():
        actual = _require_integer(layout[name], f"layout.{name}")
        if actual != expected:
            raise CacheContractError(
                f"layout.{name} must be the integer {expected} for preparation "
                f"format 1, got {actual!r}"
            )
    arrays = _require_mapping(layout["arrays"], "layout.arrays")
    _require_exact_keys(arrays, set(ARRAY_NAMES), "layout.arrays")
    for name in ARRAY_NAMES:
        entry = _require_mapping(arrays[name], f"layout.arrays.{name}")
        _require_exact_keys(entry, {"dtype", "dimensions"}, f"layout.arrays.{name}")
        if entry["dtype"] != _ARRAY_DTYPES[name]:
            raise CacheContractError(
                f"layout.arrays.{name}.dtype must be {_ARRAY_DTYPES[name]!r}"
            )
        dimensions = entry["dimensions"]
        if not isinstance(dimensions, list) or tuple(dimensions) != _ARRAY_DIMENSIONS[name]:
            raise CacheContractError(
                f"layout.arrays.{name}.dimensions must be "
                f"{list(_ARRAY_DIMENSIONS[name])}"
            )


def _validate_array_header(
    root: Path,
    descriptor: CacheArray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    path = root / descriptor.file
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise CacheContractError(f"failed to read .npy header {path}: {exc}") from exc
    if tuple(array.shape) != expected_shape:
        raise CacheContractError(
            f"{descriptor.file} has shape {tuple(array.shape)}, "
            f"manifest structure requires {expected_shape}"
        )
    if array.dtype != np.dtype(descriptor.dtype):
        raise CacheContractError(
            f"{descriptor.file} has dtype {array.dtype}, "
            f"manifest declares {descriptor.dtype}"
        )
    if not array.dtype.isnative:
        raise CacheContractError(f"{descriptor.file} is not native little-endian dtype")
    if not array.flags.c_contiguous:
        raise CacheContractError(f"{descriptor.file} is not C contiguous")
    return array


def _validate_target_kinds(
    candidate_features: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    target_kinds: np.ndarray,
    *,
    file_prefix: str,
) -> None:
    starts = offsets[:-1].astype(np.int64, copy=False)
    rows = candidate_features[starts + targets.astype(np.int64, copy=False)]
    kind_values = np.asarray(rows[:, :ACTION_KIND_WIDTH])
    one_hot_count = np.count_nonzero(kind_values == 1.0, axis=1)
    if np.any(one_hot_count != 1):
        raise CacheContractError(
            f"{file_prefix}: a teacher target has no unique one-hot action kind"
        )
    derived = np.argmax(kind_values, axis=1)
    if not np.array_equal(derived, target_kinds.astype(np.int64, copy=False)):
        raise CacheContractError(
            f"{file_prefix}: target_kinds disagree with candidate kind features"
        )


def _validate_episode_arrays(
    seeds: np.ndarray,
    indices: np.ndarray,
    counts: CacheCounts,
    *,
    file_prefix: str,
) -> set[int]:
    if counts.decisions != len(seeds):
        raise CacheContractError(f"{file_prefix}: episode_seeds length mismatch")
    if counts.decisions == 0:
        return set()
    seed_values = seeds.astype(np.uint64, copy=False)
    unique = {int(value) for value in np.unique(seed_values)}
    if len(unique) != counts.episodes:
        raise CacheContractError(
            f"{file_prefix}: {len(unique)} unique episode seeds, "
            f"counts declare {counts.episodes}"
        )
    starts = np.flatnonzero(
        np.concatenate(([True], seed_values[1:] != seed_values[:-1]))
    )
    sizes = np.diff(np.concatenate((starts, [len(seed_values)])))
    if len(starts) != len(unique):
        raise CacheContractError(
            f"{file_prefix}: episode seeds repeat in non-contiguous blocks"
        )
    expected = np.arange(len(seed_values), dtype=np.int64) - np.repeat(starts, sizes)
    if not np.array_equal(indices.astype(np.int64, copy=False), expected):
        raise CacheContractError(
            f"{file_prefix}: decision_indices are not zero-based within episodes"
        )
    return unique


def _parse_split(
    name: str,
    value: Any,
    root: Path,
    listed_files: set[str],
) -> tuple[CacheSplit, set[int]]:
    value = _require_mapping(value, f"splits.{name}")
    _require_exact_keys(value, _SPLIT_KEYS, f"splits.{name}")
    if value["split"] != name:
        raise CacheContractError(
            f"splits.{name}.split must be {name!r}, got {value['split']!r}"
        )
    declared_counts = CacheCounts.from_dict(value["counts"], f"splits.{name}.counts")
    raw_shards = value["shards"]
    if not isinstance(raw_shards, list):
        raise CacheContractError(f"splits.{name}.shards must be a list")
    if declared_counts.decisions == 0 and raw_shards:
        raise CacheContractError(f"splits.{name} is empty but lists shards")
    if declared_counts.decisions > 0 and not raw_shards:
        raise CacheContractError(f"splits.{name} is nonempty but lists no shards")

    shards: list[CacheShard] = []
    shard_counts: list[CacheCounts] = []
    all_seeds: set[int] = set()
    for position, raw_shard in enumerate(raw_shards):
        raw_shard = _require_mapping(raw_shard, f"splits.{name}.shards[{position}]")
        _require_exact_keys(
            raw_shard, _SHARD_KEYS, f"splits.{name}.shards[{position}]"
        )
        index = raw_shard["index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index != position
        ):
            raise CacheContractError(
                f"splits.{name}.shards[{position}].index must be {position}"
            )
        counts = CacheCounts.from_dict(
            raw_shard["counts"], f"splits.{name}.shards[{position}].counts"
        )
        if counts.decisions == 0:
            raise CacheContractError(f"splits.{name}.shards[{position}] is empty")
        raw_arrays = _require_mapping(
            raw_shard["arrays"], f"splits.{name}.shards[{position}].arrays"
        )
        _require_exact_keys(
            raw_arrays,
            set(ARRAY_NAMES),
            f"splits.{name}.shards[{position}].arrays",
        )
        descriptors = {
            array_name: CacheArray.from_dict(
                array_name,
                raw_arrays[array_name],
                split=name,
                shard_index=position,
            )
            for array_name in ARRAY_NAMES
        }
        for descriptor in descriptors.values():
            if descriptor.file in listed_files:
                raise CacheContractError(
                    f"duplicate cache array file {descriptor.file!r}"
                )
            listed_files.add(descriptor.file)
        states_shape = descriptors["states"].shape
        candidate_shape = descriptors["candidate_features"].shape
        if len(states_shape) != 2 or states_shape[1] != STATE_WIDTH:
            raise CacheContractError(
                f"{descriptors['states'].file} declares shape {states_shape}; "
                f"expected [D, {STATE_WIDTH}]"
            )
        if len(candidate_shape) != 2 or candidate_shape[1] != CANDIDATE_WIDTH:
            raise CacheContractError(
                f"{descriptors['candidate_features'].file} declares shape "
                f"{candidate_shape}; expected [C, {CANDIDATE_WIDTH}]"
            )
        decisions = states_shape[0]
        candidates = candidate_shape[0]
        if decisions != counts.decisions or candidates != counts.candidates:
            raise CacheContractError(
                f"splits.{name}.shards[{position}] counts do not match array shapes"
            )
        shapes = expected_shapes(decisions, candidates)
        for array_name in ARRAY_NAMES:
            descriptor = descriptors[array_name]
            if descriptor.shape != shapes[array_name]:
                raise CacheContractError(
                    f"{descriptor.file} declares shape {descriptor.shape}; "
                    f"manifest structure requires {shapes[array_name]}"
                )
        arrays = {
            array_name: _validate_array_header(
                root, descriptors[array_name], shapes[array_name]
            )
            for array_name in ARRAY_NAMES
        }
        try:
            layout = validate_packed_arrays(
                states=arrays["states"],
                candidate_features=arrays["candidate_features"],
                candidate_offsets=arrays["candidate_offsets"],
                targets=arrays["targets"],
                stages=arrays["stages"],
                actor_seats=arrays["actor_seats"],
                decision_indices=arrays["decision_indices"],
                target_kinds=arrays["target_kinds"],
            )
        except ArrayContractViolation as exc:
            file_prefix = descriptors[exc.array].file
            raise CacheContractError(f"{file_prefix}: {exc}") from None
        if counts.forced_decisions != layout.forced_count:
            raise CacheContractError(
                f"{descriptors['candidate_offsets'].file} forced count mismatch"
            )
        if counts.eligible_decisions != layout.eligible_count:
            raise CacheContractError(
                f"{descriptors['candidate_offsets'].file} eligible count mismatch"
            )
        if counts.min_candidates != int(layout.candidate_counts.min()):
            raise CacheContractError("shard min_candidates mismatch")
        if counts.max_candidates != int(layout.candidate_counts.max()):
            raise CacheContractError("shard max_candidates mismatch")
        _validate_target_kinds(
            arrays["candidate_features"],
            arrays["candidate_offsets"],
            arrays["targets"],
            arrays["target_kinds"],
            file_prefix=f"splits.{name}.shards[{position}]",
        )
        seeds = _validate_episode_arrays(
            arrays["episode_seeds"],
            arrays["decision_indices"],
            counts,
            file_prefix=f"splits.{name}.shards[{position}]",
        )
        overlap = all_seeds.intersection(seeds)
        if overlap:
            raise CacheContractError(
                f"splits.{name} repeats episode seeds across shards"
            )
        all_seeds.update(seeds)
        shards.append(CacheShard(index=index, counts=counts, arrays=descriptors))
        shard_counts.append(counts)

    aggregated = _sum_counts(shard_counts, f"splits.{name}")
    if aggregated != declared_counts:
        raise CacheContractError(
            f"splits.{name}.counts do not equal the aggregate of its shards"
        )
    return CacheSplit(split=name, counts=declared_counts, shards=tuple(shards)), all_seeds


def load_prepared_cache(
    root: Path | str,
    *,
    verify_checksums: bool = True,
) -> PreparedCache:
    """Load and fully validate a prepared cache without materializing features."""

    root_path = Path(root)
    manifest_path = root_path / "dataset.json"
    if not manifest_path.is_file():
        raise CacheContractError(f"missing {manifest_path}")
    manifest = _require_mapping(
        _read_json_no_duplicates(manifest_path), "dataset.json"
    )
    _require_exact_keys(manifest, _DATASET_KEYS, "dataset.json")
    if manifest["format"] != DATASET_FORMAT:
        raise CacheContractError(
            f"dataset format must be {DATASET_FORMAT!r}, got {manifest['format']!r}"
        )
    preparation_format_version = _require_positive_int(
        manifest["preparation_format_version"],
        "preparation_format_version",
    )
    if preparation_format_version != PREPARATION_FORMAT_VERSION:
        raise CacheContractError(
            f"unsupported preparation_format_version "
            f"{preparation_format_version}; supported version is "
            f"{PREPARATION_FORMAT_VERSION}"
        )
    preparation_version = _require_positive_int(
        manifest["preparation_version"], "preparation_version"
    )
    if preparation_version != PREPARATION_VERSION:
        raise CacheContractError(
            f"unsupported preparation_version {preparation_version}; "
            f"supported version is {PREPARATION_VERSION}"
        )
    encoding_version = _require_positive_int(
        manifest["encoding_version"], "encoding_version"
    )
    if encoding_version != ENCODING_VERSION:
        raise CacheContractError(
            f"unsupported encoding_version {encoding_version}; "
            f"supported version is {ENCODING_VERSION}"
        )
    split_algorithm = _require_nonempty_string(
        manifest["split_algorithm"], "split_algorithm"
    )
    if split_algorithm != SPLIT_ALGORITHM:
        raise CacheContractError(
            f"unsupported split_algorithm {split_algorithm!r}"
        )
    cache_id = _require_nonempty_string(manifest["cache_id"], "cache_id")
    source_revision = _require_nonempty_string(
        manifest["source_revision"], "source_revision"
    )
    corpus_sha = _require_sha256(
        manifest["corpus_manifest_sha256"], "corpus_manifest_sha256"
    )
    corpus_manifest = _require_mapping(manifest["corpus_manifest"], "corpus_manifest")
    layout = _require_mapping(manifest["layout"], "layout")
    _validate_layout(layout)
    provenance = _require_mapping(manifest["provenance"], "provenance")
    declared_counts = CacheCounts.from_dict(manifest["counts"], "counts")
    seed_start, total_games = _validate_captured_corpus(
        corpus_manifest,
        source_revision=source_revision,
        encoding_version=encoding_version,
        split_algorithm=split_algorithm,
        counts=declared_counts,
    )
    _validate_provenance(
        provenance,
        source_revision=source_revision,
        encoding_version=encoding_version,
        split_algorithm=split_algorithm,
    )
    raw_splits = _require_mapping(manifest["splits"], "splits")
    _require_exact_keys(raw_splits, set(SPLIT_NAMES), "splits")

    listed_files: set[str] = set()
    splits: dict[str, CacheSplit] = {}
    split_counts: list[CacheCounts] = []
    all_seeds: set[int] = set()
    for split_name in SPLIT_NAMES:
        split_entry, seeds = _parse_split(
            split_name, raw_splits[split_name], root_path, listed_files
        )
        for seed in seeds:
            if seed < seed_start or seed >= seed_start + total_games:
                raise CacheContractError(
                    f"seed {seed:016x} is outside the captured snapshot seed "
                    f"range [{seed_start}, {seed_start + total_games})"
                )
            if _seed_split(seed) != split_name:
                raise CacheContractError(
                    f"seed {seed:016x} belongs to {_seed_split(seed)}, "
                    f"not {split_name}"
                )
        overlap = all_seeds.intersection(seeds)
        if overlap:
            raise CacheContractError("episode seeds repeat across splits")
        all_seeds.update(seeds)
        splits[split_name] = split_entry
        split_counts.append(split_entry.counts)
    if len(all_seeds) != total_games:
        raise CacheContractError(
            f"cache contains {len(all_seeds)} distinct episode seeds, "
            f"captured corpus lists {total_games} games"
        )
    aggregated = _sum_counts(split_counts, "counts")
    if aggregated != declared_counts:
        raise CacheContractError("counts do not equal the aggregate of all splits")

    expected_files = {"dataset.json", *listed_files}
    actual_files = {
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file()
    }
    missing = expected_files - actual_files
    extra = actual_files - expected_files
    if missing:
        raise CacheContractError(f"missing cache files: {sorted(missing)}")
    if extra:
        raise CacheContractError(f"unlisted cache files: {sorted(extra)}")

    for split_name in SPLIT_NAMES:
        for shard in splits[split_name].shards:
            for descriptor in shard.arrays.values():
                path = root_path / descriptor.file
                if verify_checksums:
                    actual_sha = _sha256_file(path, descriptor.file_bytes)
                    if actual_sha != descriptor.sha256:
                        raise CacheContractError(
                            f"checksum mismatch for {descriptor.file}: "
                            f"manifest {descriptor.sha256}, actual {actual_sha}"
                        )
                else:
                    try:
                        actual_bytes = path.stat().st_size
                    except OSError as exc:
                        raise CacheContractError(
                            f"failed to stat {path}: {exc}"
                        ) from exc
                    if actual_bytes != descriptor.file_bytes:
                        raise CacheContractError(
                            f"{descriptor.file} is {actual_bytes} bytes; "
                            f"manifest declares {descriptor.file_bytes}"
                        )

    return PreparedCache(
        root=root_path,
        manifest=manifest,
        cache_id=cache_id,
        encoding_version=encoding_version,
        preparation_format_version=preparation_format_version,
        preparation_version=preparation_version,
        source_revision=source_revision,
        corpus_manifest_sha256=corpus_sha,
        split_algorithm=split_algorithm,
        counts=declared_counts,
        splits=splits,
        layout=layout,
        provenance=provenance,
        corpus_manifest=corpus_manifest,
    )


def load_cache_split(
    root: Path | str,
    split: str,
    *,
    mmap: bool = True,
    verify_checksums: bool = True,
) -> tuple[ShardedDecisionDataset, PreparedCache]:
    """Validate a cache and return one materialized split plus its metadata."""

    cache = load_prepared_cache(root, verify_checksums=verify_checksums)
    return cache.load_split(split, mmap=mmap), cache
