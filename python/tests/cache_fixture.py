"""Synthetic prepared-cache writer conforming to the approved dataset.json contract.

The fixture starts from an explicit contiguous captured seed interval. Its
episodes are assigned to train/validation/test by the fixture's frozen copy of
the Rust seed-split algorithm, so each split contains the seeds that the real
algorithm assigns to it. It never searches for a seed interval matching a
desired per-split composition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

#: The default captured interval is the 148..157 probe: train 5, validation 3,
#: test 2 episodes. It is fixed test data, not a searched composition.
DEFAULT_SEED_START = 148
DEFAULT_GAMES = 10
_U64_LIMIT = 1 << 64
_CANDIDATE_PATTERN = (2, 3, 1)


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


def _seed_split(seed: int) -> str:
    """Frozen Rust split algorithm, mirrored to assign fixture episode seeds."""

    digest = hashlib.sha256(b"bc-split-v1\0" + seed.to_bytes(8, "big")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _split_seeds(seed_start: int, games: int) -> dict[str, tuple[int, ...]]:
    assigned: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for seed in range(seed_start, seed_start + games):
        assigned[_seed_split(seed)].append(seed)
    return {name: tuple(seeds) for name, seeds in assigned.items()}


def _candidate_counts(decisions: int) -> list[int]:
    return [
        _CANDIDATE_PATTERN[index % len(_CANDIDATE_PATTERN)]
        for index in range(decisions)
    ]


@dataclass(frozen=True)
class CacheFixture:
    """One written synthetic cache plus its generated source data.

    ``seeds[split]`` is the actual captured-seed coverage of that split and
    ``sources[split]`` is the packed dataset whose arrays were written. Empty
    splits have no source dataset and zero counts.
    """

    root: Path
    seed_start: int
    games: int
    seeds: Mapping[str, tuple[int, ...]]
    sources: Mapping[str, PackedDecisionDataset | None]
    counts: Mapping[str, CacheCounts]
    manifest: Mapping[str, Any]


def write_cache(
    root: Path | str,
    *,
    seed_start: int = DEFAULT_SEED_START,
    games: int = DEFAULT_GAMES,
    cache_id: str = "synthetic-cache-id",
    source_revision: str = "synthetic-revision",
    decisions_per_episode: int = 2,
    episodes_per_shard: int = 2,
) -> CacheFixture:
    """Write a complete synthetic cache from one contiguous captured interval.

    ``seed_start`` and ``games`` define the captured corpus: every seed in
    ``range(seed_start, seed_start + games)`` is one episode. Episodes are
    assigned to splits by the real seed-split algorithm; shard boundaries stay
    on episode boundaries and follow captured seed order.
    """

    if decisions_per_episode < 1 or episodes_per_shard < 1:
        raise ValueError("episode and shard sizes must be positive")
    if seed_start < 0 or seed_start >= _U64_LIMIT:
        raise ValueError("seed_start must be a u64")
    if games < 0 or seed_start + games > _U64_LIMIT:
        raise ValueError("captured seed range must fit in u64")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    seed_coverage = _split_seeds(seed_start, games)
    captured_seeds = [
        seed for split_seeds in seed_coverage.values() for seed in split_seeds
    ]
    captured_seeds.sort()

    sources: dict[str, PackedDecisionDataset | None] = {}
    split_counts: dict[str, CacheCounts] = {}
    split_manifests: dict[str, Any] = {}
    episode_decisions: dict[int, int] = {}
    for split_name in SPLIT_NAMES:
        seeds = seed_coverage[split_name]
        if not seeds:
            source = None
            counts = CacheCounts(0, 0, 0, 0, 0, 0, 0)
            split_manifests[split_name] = {
                "split": split_name,
                "counts": counts.to_dict(),
                "shards": [],
            }
            sources[split_name] = source
            split_counts[split_name] = counts
            continue

        decisions = len(seeds) * decisions_per_episode
        source = make_dataset(
            split=split_name,
            decisions=decisions,
            counts=_candidate_counts(decisions),
            seed=seed_start + _SPLIT_ORDER[split_name],
            cache_id=cache_id,
            source_revision=source_revision,
            signal=True,
        )
        sources[split_name] = source
        ranges = [
            (start, start + decisions_per_episode)
            for start in range(0, decisions, decisions_per_episode)
        ]
        episode_seeds = np.empty(decisions, dtype=np.uint64)
        decision_indices = np.empty(decisions, dtype=np.uint64)
        for episode_index, (start, end) in enumerate(ranges):
            seed = seeds[episode_index]
            episode_seeds[start:end] = seed
            decision_indices[start:end] = np.arange(end - start, dtype=np.uint64)
            episode_decisions[seed] = end - start

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
                source.offsets[start : end + 1].astype(np.int64)
                - int(source.offsets[start])
            )
            candidate_start = int(source.offsets[start])
            candidate_end = int(source.offsets[end])
            if source.target_kinds is None:
                raise ValueError("synthetic cache requires target_kinds metadata")
            shard_arrays = {
                "states": np.asarray(source.states[start:end], dtype=np.float32),
                "candidate_features": np.asarray(
                    source.candidate_features[candidate_start:candidate_end],
                    dtype=np.float32,
                ),
                "candidate_offsets": local_offsets,
                "targets": np.asarray(source.targets[start:end], dtype=np.int64),
                "episode_seeds": episode_seeds[start:end],
                "decision_indices": decision_indices[start:end],
                "actor_seats": (
                    (np.arange(start, end, dtype=np.int64) % 4).astype(np.uint8)
                    if source.actor_seats is None
                    else np.asarray(source.actor_seats[start:end], dtype=np.uint8)
                ),
                "stages": np.asarray(source.stages[start:end], dtype=np.uint8),
                "target_kinds": np.asarray(
                    source.target_kinds[start:end], dtype=np.uint8
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
        if counts.episodes != len(seeds):
            raise AssertionError("synthetic cache split episode count mismatch")
        if counts.decisions != len(source):
            raise AssertionError("synthetic cache split decision count mismatch")
        if counts.eligible_decisions != source.identity.eligible_decisions:
            raise AssertionError("synthetic cache eligible count mismatch")
        split_counts[split_name] = counts
        split_manifests[split_name] = {
            "split": split_name,
            "counts": counts.to_dict(),
            "shards": shards,
        }

    totals = _aggregate([split_counts[name] for name in SPLIT_NAMES])
    if totals.episodes != games:
        raise AssertionError("synthetic cache episode total mismatch")
    if totals.decisions != sum(episode_decisions.values()):
        raise AssertionError("synthetic cache decision total mismatch")
    corpus_shards = []
    for first in range(0, games, episodes_per_shard):
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
    return CacheFixture(
        root=root_path,
        seed_start=seed_start,
        games=games,
        seeds=seed_coverage,
        sources=sources,
        counts=split_counts,
        manifest=manifest,
    )


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
    seed_start: int = DEFAULT_SEED_START,
    games: int = DEFAULT_GAMES,
    **kwargs: Any,
) -> CacheFixture:
    """Write the default fixed interval, or a caller-supplied fixed interval."""

    return write_cache(root, seed_start=seed_start, games=games, **kwargs)
