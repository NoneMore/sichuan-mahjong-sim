"""Synthetic prepared-cache writer conforming to the approved dataset.json contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mahjong_bc.cache import ARRAY_NAMES, CacheCounts
from mahjong_bc.constants import (
    DATASET_FORMAT,
    ENCODING_VERSION,
    PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
    SPLIT_ALGORITHM,
    SPLIT_NAMES,
)
from mahjong_bc.data import PackedDecisionDataset

from .helpers import make_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_for(
    offsets: np.ndarray,
    *,
    episodes: int,
) -> CacheCounts:
    bounds = np.diff(offsets.astype(np.int64, copy=False))
    return CacheCounts(
        episodes=episodes,
        decisions=len(bounds),
        candidates=int(offsets[-1]),
        forced_decisions=int(np.count_nonzero(bounds == 1)),
        eligible_decisions=int(np.count_nonzero(bounds > 1)),
        min_candidates=int(bounds.min()) if len(bounds) else 0,
        max_candidates=int(bounds.max()) if len(bounds) else 0,
    )


def _aggregate(counts: list[CacheCounts]) -> CacheCounts:
    nonempty = [value for value in counts if value.decisions > 0]
    return CacheCounts(
        episodes=sum(value.episodes for value in counts),
        decisions=sum(value.decisions for value in counts),
        candidates=sum(value.candidates for value in counts),
        forced_decisions=sum(value.forced_decisions for value in counts),
        eligible_decisions=sum(value.eligible_decisions for value in counts),
        min_candidates=min((value.min_candidates for value in nonempty), default=0),
        max_candidates=max((value.max_candidates for value in nonempty), default=0),
    )


def _array_descriptor(root: Path, relative: str, array: np.ndarray) -> dict[str, Any]:
    path = root / relative
    return {
        "file": relative,
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


_SPLIT_ORDER = {name: index for index, name in enumerate(SPLIT_NAMES)}
_SEED_WINDOW_CACHE: dict[tuple[int, int, int], int] = {}


def _seed_split(seed: int) -> str:
    """Frozen Rust split algorithm, mirrored to choose fixture episode seeds."""

    digest = hashlib.sha256(b"bc-split-v1\0" + seed.to_bytes(8, "big")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _find_seed_start(required: tuple[int, int, int]) -> int:
    """Find the first small contiguous seed window with the required splits."""

    cached = _SEED_WINDOW_CACHE.get(required)
    if cached is not None:
        return cached
    total = sum(required)
    if total == 0:
        _SEED_WINDOW_CACHE[required] = 0
        return 0
    counts = [0, 0, 0]
    for seed in range(total):
        counts[_SPLIT_ORDER[_seed_split(seed)]] += 1
    start = 0
    while tuple(counts) != required:
        counts[_SPLIT_ORDER[_seed_split(start)]] -= 1
        counts[_SPLIT_ORDER[_seed_split(start + total)]] += 1
        start += 1
        if start > 1_000_000:
            raise AssertionError(
                f"no contiguous seed window matches {required} within 1,000,000 seeds"
            )
    _SEED_WINDOW_CACHE[required] = start
    return start


def write_cache(
    root: Path | str,
    *,
    train: PackedDecisionDataset | None = None,
    validation: PackedDecisionDataset | None = None,
    test: PackedDecisionDataset | None = None,
    cache_id: str = "synthetic-cache-id",
    source_revision: str = "synthetic-revision",
    decisions_per_episode: int = 2,
    episodes_per_shard: int = 2,
) -> Path:
    """Write a complete synthetic cache and return its root path.

    Episode seeds are selected from one small contiguous captured seed range and
    assigned to their real Rust ``seed_split`` split. Corpus shards follow the
    captured contiguous range and always end on episode boundaries.
    """

    if decisions_per_episode < 1 or episodes_per_shard < 1:
        raise ValueError("episode and shard sizes must be positive")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    supplied = {"train": train, "validation": validation, "test": test}
    planned: dict[str, list[tuple[int, int]] | None] = {}
    required = [0, 0, 0]
    for split_name in SPLIT_NAMES:
        dataset = supplied[split_name]
        if dataset is None:
            planned[split_name] = None
            continue
        decisions = len(dataset)
        ranges = [
            (start, min(start + decisions_per_episode, decisions))
            for start in range(0, decisions, decisions_per_episode)
        ]
        planned[split_name] = ranges
        required[_SPLIT_ORDER[split_name]] = len(ranges)
    required_counts = tuple(required)
    seed_start = _find_seed_start(required_counts)
    total_games = sum(required_counts)
    captured_seeds = list(range(seed_start, seed_start + total_games))
    assigned = {
        split_name: [
            seed for seed in captured_seeds if _seed_split(seed) == split_name
        ]
        for split_name in SPLIT_NAMES
    }
    if tuple(len(assigned[name]) for name in SPLIT_NAMES) != required_counts:
        raise AssertionError("fixture seed window does not match split sizes")

    episode_decisions: dict[int, int] = {}
    split_manifests: dict[str, Any] = {}
    split_counts: list[CacheCounts] = []
    for split_name in SPLIT_NAMES:
        dataset = supplied[split_name]
        if dataset is None:
            counts = CacheCounts(0, 0, 0, 0, 0, 0, 0)
            split_manifests[split_name] = {
                "split": split_name,
                "counts": counts.to_dict(),
                "shards": [],
            }
            split_counts.append(counts)
            continue

        ranges = planned[split_name]
        if ranges is None:
            raise AssertionError("fixture split plan is missing")
        decisions = len(dataset)
        seeds = np.empty(decisions, dtype=np.uint64)
        decision_indices = np.empty(decisions, dtype=np.uint64)
        for episode_index, (start, end) in enumerate(ranges):
            seed = assigned[split_name][episode_index]
            seeds[start:end] = seed
            episode_decisions[seed] = end - start
            decision_indices[start:end] = np.arange(end - start, dtype=np.uint64)

        arrays_dir = root_path / "arrays" / split_name
        arrays_dir.mkdir(parents=True, exist_ok=True)
        shards: list[dict[str, Any]] = []
        shard_counts: list[CacheCounts] = []
        for shard_index, first_episode in enumerate(
            range(0, len(ranges), episodes_per_shard)
        ):
            chosen = ranges[first_episode : first_episode + episodes_per_shard]
            start = chosen[0][0]
            end = chosen[-1][1]
            local_offsets = (
                dataset.offsets[start : end + 1].astype(np.int64)
                - int(dataset.offsets[start])
            )
            candidate_start = int(dataset.offsets[start])
            candidate_end = int(dataset.offsets[end])
            if dataset.target_kinds is None:
                raise ValueError("synthetic cache requires target_kinds metadata")
            shard_arrays = {
                "states": np.asarray(dataset.states[start:end], dtype=np.float32),
                "candidate_features": np.asarray(
                    dataset.candidate_features[candidate_start:candidate_end],
                    dtype=np.float32,
                ),
                "candidate_offsets": local_offsets,
                "targets": np.asarray(
                    dataset.targets[start:end], dtype=np.int64
                ),
                "episode_seeds": seeds[start:end],
                "decision_indices": decision_indices[start:end],
                "actor_seats": (
                    (np.arange(start, end, dtype=np.int64) % 4).astype(np.uint8)
                    if dataset.actor_seats is None
                    else np.asarray(dataset.actor_seats[start:end], dtype=np.uint8)
                ),
                "stages": np.asarray(dataset.stages[start:end], dtype=np.uint8),
                "target_kinds": np.asarray(
                    dataset.target_kinds[start:end], dtype=np.uint8
                ),
            }
            descriptors: dict[str, Any] = {}
            for name in ARRAY_NAMES:
                array = shard_arrays[name]
                relative = f"arrays/{split_name}/{shard_index:05d}.{name}.npy"
                np.save(root_path / relative, array, allow_pickle=False)
                descriptors[name] = _array_descriptor(root_path, relative, array)
            counts = _counts_for(local_offsets, episodes=len(chosen))
            shard_counts.append(counts)
            shards.append(
                {
                    "index": shard_index,
                    "counts": counts.to_dict(),
                    "arrays": descriptors,
                }
            )
        counts = _aggregate(shard_counts)
        expected_eligible = dataset.identity.eligible_decisions
        if counts.eligible_decisions != expected_eligible:
            raise AssertionError("synthetic cache eligible count mismatch")
        split_counts.append(counts)
        split_manifests[split_name] = {
            "split": split_name,
            "counts": counts.to_dict(),
            "shards": shards,
        }

    totals = _aggregate(split_counts)
    if totals.episodes != total_games:
        raise AssertionError("synthetic cache episode total mismatch")
    if totals.decisions != sum(episode_decisions.values()):
        raise AssertionError("synthetic cache decision total mismatch")
    corpus_shards = []
    for first in range(0, total_games, episodes_per_shard):
        chosen = captured_seeds[first : first + episodes_per_shard]
        corpus_shards.append(
            {
                "first_seed": f"{chosen[0]:016x}",
                "games": len(chosen),
                "decisions": sum(episode_decisions[seed] for seed in chosen),
                "sha256": hashlib.sha256(
                    f"synthetic-corpus-shard-{chosen[0]:016x}".encode("ascii")
                ).hexdigest(),
            }
        )
    corpus_manifest = {
        "format_version": 1,
        "source_revision": source_revision,
        "encoding_version": ENCODING_VERSION,
        "split_algorithm": SPLIT_ALGORITHM,
        "seed_start": f"{seed_start:016x}",
        "decision_limit": decisions_per_episode,
        "shard_games": episodes_per_shard,
        "shards": corpus_shards,
    }
    corpus_sha = hashlib.sha256(
        json.dumps(corpus_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format": DATASET_FORMAT,
        "preparation_format_version": PREPARATION_FORMAT_VERSION,
        "preparation_version": PREPARATION_VERSION,
        "encoding_version": ENCODING_VERSION,
        "cache_id": cache_id,
        "split_algorithm": SPLIT_ALGORITHM,
        "source_revision": source_revision,
        "corpus_manifest_sha256": corpus_sha,
        "corpus_manifest": corpus_manifest,
        "layout": {
            "state_width": 1013,
            "candidate_width": 39,
            "stage_count": 5,
            "action_kind_count": 9,
            "arrays": {
                "states": {
                    "dtype": "float32",
                    "dimensions": ["decisions", "state_features"],
                },
                "candidate_features": {
                    "dtype": "float32",
                    "dimensions": ["candidates", "candidate_features"],
                },
                "candidate_offsets": {
                    "dtype": "int64",
                    "dimensions": ["decisions + 1"],
                },
                "targets": {"dtype": "int64", "dimensions": ["decisions"]},
                "episode_seeds": {
                    "dtype": "uint64",
                    "dimensions": ["decisions"],
                },
                "decision_indices": {
                    "dtype": "uint64",
                    "dimensions": ["decisions"],
                },
                "actor_seats": {"dtype": "uint8", "dimensions": ["decisions"]},
                "stages": {"dtype": "uint8", "dimensions": ["decisions"]},
                "target_kinds": {
                    "dtype": "uint8",
                    "dimensions": ["decisions"],
                },
            },
        },
        "provenance": {
            "collection": {
                "source_revision": source_revision,
                "encoding_version": ENCODING_VERSION,
                "split_algorithm": SPLIT_ALGORITHM,
            },
            "preparation": {
                "build": {"dirty": False, "revision": source_revision},
                "environment": {"generator": "synthetic-test-fixture"},
            },
        },
        "counts": totals.to_dict(),
        "splits": split_manifests,
    }
    (root_path / "dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root_path


def read_manifest(root: Path | str) -> dict[str, Any]:
    return json.loads((Path(root) / "dataset.json").read_text(encoding="utf-8"))


def write_manifest(root: Path | str, manifest: Mapping[str, Any]) -> None:
    (Path(root) / "dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def synthetic_cache(
    root: Path | str,
    *,
    train_decisions: int = 9,
    validation_decisions: int = 5,
    test_decisions: int = 3,
    **kwargs: Any,
) -> Path:
    train = make_dataset(
        split="train",
        decisions=train_decisions,
        counts=[2, 3, 1] * ((train_decisions + 2) // 3),
        seed=21,
        signal=True,
    )
    train = _trim(train, train_decisions)
    validation = _trim(
        make_dataset(
            split="validation",
            decisions=validation_decisions,
            counts=[2, 1, 2] * ((validation_decisions + 2) // 3),
            seed=22,
            signal=True,
        ),
        validation_decisions,
    )
    test = _trim(
        make_dataset(
            split="test",
            decisions=test_decisions,
            counts=[2, 2, 1] * ((test_decisions + 2) // 3),
            seed=23,
            signal=True,
        ),
        test_decisions,
    )
    return write_cache(
        root,
        train=train,
        validation=validation,
        test=test,
        **kwargs,
    )


def _trim(dataset: PackedDecisionDataset, decisions: int) -> PackedDecisionDataset:
    if len(dataset) < decisions:
        extra = make_dataset(
            split=dataset.identity.split,
            decisions=decisions,
            counts=[2] * decisions,
            seed=99,
            cache_id=dataset.identity.cache_id,
            signal=True,
        )
        return extra
    if len(dataset) == decisions:
        return dataset
    counts = dataset.candidate_counts()[:decisions]
    from .helpers import identity

    eligible = int(np.count_nonzero(counts > 1))
    return PackedDecisionDataset(
        states=np.asarray(dataset.states[:decisions]),
        candidate_features=np.asarray(
            dataset.candidate_features[: int(dataset.offsets[decisions])]
        ),
        offsets=np.asarray(dataset.offsets[: decisions + 1]),
        targets=np.asarray(dataset.targets[:decisions]),
        stages=np.asarray(dataset.stages[:decisions]),
        identity=identity(
            split=dataset.identity.split,
            decisions=decisions,
            eligible=eligible,
            cache_id=dataset.identity.cache_id,
        ),
        actor_seats=None,
        decision_indices=None,
        target_kinds=(
            None
            if dataset.target_kinds is None
            else np.asarray(dataset.target_kinds[:decisions])
        ),
    )
