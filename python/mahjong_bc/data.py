"""Packed decision dataset, padded batches, and deterministic batch iteration.

This module is independent of the on-disk cache manifest. A cache loader can
back :class:`PackedDecisionDataset` with memory-mapped ``.npy`` arrays; training
and metrics consume only the packed array contract documented by the
offline-training specification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DatasetIdentity
from .constants import (
    ACTION_KIND_COUNT,
    ACTION_KIND_WIDTH,
    CANDIDATE_WIDTH,
    FORCED_CANDIDATE_COUNT,
    STAGE_COUNT,
    STATE_WIDTH,
)
from .errors import DataError


def derive_target_kind(candidate_features: np.ndarray, target: int) -> int:
    """Return the one-hot action kind of the teacher's chosen candidate."""

    if candidate_features.ndim != 2 or candidate_features.shape[1] != CANDIDATE_WIDTH:
        raise DataError(
            f"candidate_features must have shape [N, {CANDIDATE_WIDTH}], "
            f"got {candidate_features.shape}"
        )
    if target < 0 or target >= candidate_features.shape[0]:
        raise DataError(f"target {target} is outside {candidate_features.shape[0]} candidates")
    kind_values = np.asarray(candidate_features[target, :ACTION_KIND_WIDTH])
    selected = kind_values == 1.0
    if int(np.count_nonzero(selected)) != 1:
        raise DataError(
            "the target candidate does not contain exactly one one-hot action kind"
        )
    return int(np.argmax(selected))


@dataclass(frozen=True)
class DecisionSample:
    """One decision as returned by a packed dataset."""

    state: np.ndarray
    candidate_features: np.ndarray
    target: int
    stage: int
    target_kind: int
    actor_seat: int | None = None
    decision_index: int | None = None
    episode_id: object | None = None

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_features.shape[0])

    @property
    def eligible(self) -> bool:
        return self.candidate_count > FORCED_CANDIDATE_COUNT


@dataclass(frozen=True)
class DecisionBatch:
    """Padded batch of decisions, all tensors on one device."""

    states: Tensor
    candidates: Tensor
    mask: Tensor
    targets: Tensor
    stages: Tensor
    target_kinds: Tensor
    eligible: Tensor
    actor_seats: Tensor | None
    decision_indices: Tensor | None
    episode_ids: tuple[object, ...] | None

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> DecisionBatch:
        target = torch.device(device)
        return DecisionBatch(
            states=self.states.to(target, non_blocking=non_blocking),
            candidates=self.candidates.to(target, non_blocking=non_blocking),
            mask=self.mask.to(target, non_blocking=non_blocking),
            targets=self.targets.to(target, non_blocking=non_blocking),
            stages=self.stages.to(target, non_blocking=non_blocking),
            target_kinds=self.target_kinds.to(target, non_blocking=non_blocking),
            eligible=self.eligible.to(target, non_blocking=non_blocking),
            actor_seats=(
                None
                if self.actor_seats is None
                else self.actor_seats.to(target, non_blocking=non_blocking)
            ),
            decision_indices=(
                None
                if self.decision_indices is None
                else self.decision_indices.to(target, non_blocking=non_blocking)
            ),
            episode_ids=self.episode_ids,
        )


def _require_integer_array(value: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray):
        raise DataError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise DataError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        raise DataError(f"{name} must have an integer dtype, got {value.dtype}")


def _require_float32_array(value: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray):
        raise DataError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise DataError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.float32:
        raise DataError(f"{name} must be float32, got {value.dtype}")


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class DecisionDataset(Dataset[DecisionSample], ABC):
    """Common interface for single-shard and multi-shard packed splits."""

    identity: DatasetIdentity

    @abstractmethod
    def eligible_indices(self) -> np.ndarray:
        """Original split indices of decisions with more than one candidate."""

    @abstractmethod
    def forced_by_stage(self) -> tuple[int, ...]:
        """Forced decision counts in encoding stage order."""

    @abstractmethod
    def candidate_counts(self) -> np.ndarray:
        """Candidate count of every decision in split order."""


class PackedDecisionDataset(DecisionDataset):
    """Packed, optionally memory-mapped decision arrays for one split.

    ``candidate_features`` is packed across decisions. ``offsets`` delimits
    each decision's nonempty candidate range; ``targets`` are local candidate
    indices. The constructor validates the complete layout and identity counts.
    """

    def __init__(
        self,
        *,
        states: np.ndarray,
        candidate_features: np.ndarray,
        offsets: np.ndarray,
        targets: np.ndarray,
        stages: np.ndarray,
        identity: DatasetIdentity | None,
        actor_seats: np.ndarray | None = None,
        decision_indices: np.ndarray | None = None,
        episode_ids: Sequence[object] | np.ndarray | None = None,
        target_kinds: np.ndarray | None = None,
    ) -> None:
        if identity is not None and not isinstance(identity, DatasetIdentity):
            raise DataError("identity must be a DatasetIdentity or None")
        if not isinstance(candidate_features, np.ndarray):
            raise DataError("candidate_features must be a NumPy array")
        if candidate_features.ndim != 2:
            raise DataError("candidate_features must be a two-dimensional array")
        decisions = int(states.shape[0]) if isinstance(states, np.ndarray) and states.ndim else -1
        if decisions < 0:
            raise DataError("states must be a two-dimensional NumPy array")
        _require_float32_array(states, "states", (decisions, STATE_WIDTH))
        _require_float32_array(
            candidate_features,
            "candidate_features",
            (int(candidate_features.shape[0]), CANDIDATE_WIDTH),
        )
        _require_integer_array(offsets, "offsets", (decisions + 1,))
        _require_integer_array(targets, "targets", (decisions,))
        _require_integer_array(stages, "stages", (decisions,))

        candidate_count = int(candidate_features.shape[0])
        offset_values = offsets
        if int(offset_values[0]) != 0:
            raise DataError("offsets must start at zero")
        if int(offset_values[-1]) != candidate_count:
            raise DataError(
                f"offsets must end at the candidate count {candidate_count}, "
                f"got {int(offset_values[-1])}"
            )
        bounds = np.diff(offset_values.astype(np.int64, copy=False))
        if np.any(bounds <= 0):
            raise DataError("every decision must have at least one candidate")
        local_targets = targets.astype(np.int64, copy=False)
        if np.any(local_targets < 0) or np.any(local_targets >= bounds):
            raise DataError("every target must index its own candidate range")

        stage_values = stages.astype(np.int64, copy=False)
        if np.any(stage_values < 0) or np.any(stage_values >= STAGE_COUNT):
            raise DataError(f"stages must be in [0, {STAGE_COUNT})")

        if actor_seats is not None:
            _require_integer_array(actor_seats, "actor_seats", (decisions,))
            seats = actor_seats.astype(np.int64, copy=False)
            if np.any(seats < 0) or np.any(seats > 3):
                raise DataError("actor_seats must be in [0, 4)")
        if decision_indices is not None:
            _require_integer_array(decision_indices, "decision_indices", (decisions,))
            if np.any(decision_indices.astype(np.int64, copy=False) < 0):
                raise DataError("decision_indices must be non-negative")
        if episode_ids is not None and len(episode_ids) != decisions:
            raise DataError(
                f"episode_ids must contain {decisions} entries, got {len(episode_ids)}"
            )
        if target_kinds is not None:
            _require_integer_array(target_kinds, "target_kinds", (decisions,))
            kinds = target_kinds.astype(np.int64, copy=False)
            if np.any(kinds < 0) or np.any(kinds >= ACTION_KIND_COUNT):
                raise DataError(f"target_kinds must be in [0, {ACTION_KIND_COUNT})")

        eligible_count = int(np.count_nonzero(bounds > FORCED_CANDIDATE_COUNT))
        if identity is not None:
            if identity.decisions != decisions:
                raise DataError(
                    f"dataset identity declares {identity.decisions} decisions, "
                    f"arrays contain {decisions}"
                )
            if identity.eligible_decisions != eligible_count:
                raise DataError(
                    f"dataset identity declares {identity.eligible_decisions} eligible "
                    f"decisions, arrays contain {eligible_count}"
                )

        self._closed = False
        self._decision_count = decisions
        self.states = states
        self.candidate_features = candidate_features
        self.offsets = offsets
        self.targets = targets
        self.stages = stages
        self.identity = identity
        self.actor_seats = actor_seats
        self.decision_indices = decision_indices
        self.episode_ids = episode_ids
        self.target_kinds = target_kinds
        self._candidate_counts = _read_only(bounds)
        self._eligible_indices = _read_only(
            np.flatnonzero(bounds > FORCED_CANDIDATE_COUNT)
        )
        self._eligible_mask = _read_only(bounds > FORCED_CANDIDATE_COUNT)

    def __len__(self) -> int:
        return self._decision_count

    @property
    def decision_count(self) -> int:
        return len(self)

    @property
    def eligible_count(self) -> int:
        return int(self._eligible_indices.size)

    @property
    def forced_count(self) -> int:
        return len(self) - self.eligible_count

    def candidate_counts(self) -> np.ndarray:
        return self._candidate_counts

    def eligible_indices(self) -> np.ndarray:
        """Original dataset indices of decisions with more than one candidate."""

        return self._eligible_indices

    def eligible_mask(self) -> np.ndarray:
        return self._eligible_mask

    def forced_by_stage(self) -> tuple[int, ...]:
        if self._closed:
            raise DataError("dataset is closed")
        forced = self.stages[~self._eligible_mask].astype(np.int64, copy=False)
        counts = np.bincount(forced, minlength=STAGE_COUNT)
        return tuple(int(value) for value in counts)

    def close(self) -> None:
        """Release owned arrays; outstanding sample views keep their maps alive.

        Never force-close NumPy's underlying mmap: samples may still reference
        it, and invalidating those views would cause a native access violation.
        """

        self._closed = True
        self.states = None
        self.candidate_features = None
        self.offsets = None
        self.targets = None
        self.stages = None
        self.actor_seats = None
        self.decision_indices = None
        self.episode_ids = None
        self.target_kinds = None

    def __getitem__(self, index: int) -> DecisionSample:
        if self._closed:
            raise DataError("dataset is closed")
        position = int(index)
        if position < 0 or position >= len(self):
            raise IndexError(f"decision index {index} is out of range")
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        if self.target_kinds is None:
            target_kind = derive_target_kind(
                self.candidate_features[start:end], int(self.targets[position])
            )
        else:
            target_kind = int(self.target_kinds[position])
        episode_id = None if self.episode_ids is None else self.episode_ids[position]
        actor_seat = (
            None if self.actor_seats is None else int(self.actor_seats[position])
        )
        decision_index = (
            None if self.decision_indices is None else int(self.decision_indices[position])
        )
        return DecisionSample(
            state=self.states[position],
            candidate_features=self.candidate_features[start:end],
            target=int(self.targets[position]),
            stage=int(self.stages[position]),
            target_kind=target_kind,
            actor_seat=actor_seat,
            decision_index=decision_index,
            episode_id=episode_id,
        )


class ShardedDecisionDataset(DecisionDataset):
    """A split backed by multiple independently memory-mapped shards.

    Global indices follow manifest shard order. Each shard is itself a
    :class:`PackedDecisionDataset`; no array concatenation is performed.
    """

    def __init__(
        self,
        shards: Sequence[PackedDecisionDataset],
        identity: DatasetIdentity,
    ) -> None:
        if not isinstance(identity, DatasetIdentity):
            raise DataError("identity must be a DatasetIdentity")
        self._closed = False
        self.shards = tuple(shards)
        if not all(isinstance(shard, PackedDecisionDataset) for shard in self.shards):
            raise DataError("shards must contain PackedDecisionDataset instances")
        decisions = sum(len(shard) for shard in self.shards)
        eligible = sum(shard.eligible_count for shard in self.shards)
        if decisions != identity.decisions:
            raise DataError(
                f"dataset identity declares {identity.decisions} decisions, "
                f"shards contain {decisions}"
            )
        if eligible != identity.eligible_decisions:
            raise DataError(
                f"dataset identity declares {identity.eligible_decisions} eligible "
                f"decisions, shards contain {eligible}"
            )
        self.identity = identity
        lengths = [len(shard) for shard in self.shards]
        self._starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
        )
        eligible_parts = [
            shard.eligible_indices() + int(self._starts[index])
            for index, shard in enumerate(self.shards)
        ]
        self._eligible_indices = (
            np.concatenate(eligible_parts).astype(np.int64, copy=False)
            if eligible_parts
            else np.empty(0, dtype=np.int64)
        )
        self._eligible_mask = np.zeros(decisions, dtype=bool)
        if self._eligible_indices.size:
            self._eligible_mask[self._eligible_indices] = True
        self._candidate_counts = _read_only(
            np.concatenate([shard.candidate_counts() for shard in self.shards])
            if self.shards
            else np.empty(0, dtype=np.int64)
        )
        _read_only(self._eligible_indices)
        _read_only(self._eligible_mask)

    def __len__(self) -> int:
        return int(self._starts[-1])

    @property
    def decision_count(self) -> int:
        return len(self)

    @property
    def eligible_count(self) -> int:
        return int(self._eligible_indices.size)

    @property
    def forced_count(self) -> int:
        return len(self) - self.eligible_count

    def candidate_counts(self) -> np.ndarray:
        return self._candidate_counts

    def eligible_indices(self) -> np.ndarray:
        return self._eligible_indices

    def eligible_mask(self) -> np.ndarray:
        return self._eligible_mask

    def forced_by_stage(self) -> tuple[int, ...]:
        totals = [0] * STAGE_COUNT
        for shard in self.shards:
            for stage, count in enumerate(shard.forced_by_stage()):
                totals[stage] += count
        return tuple(totals)

    def close(self) -> None:
        """Release each shard's owned arrays without invalidating sample views."""

        self._closed = True
        for shard in self.shards:
            shard.close()

    def __getitem__(self, index: int) -> DecisionSample:
        if self._closed:
            raise DataError("dataset is closed")
        position = int(index)
        if position < 0 or position >= len(self):
            raise IndexError(f"decision index {index} is out of range")
        shard_index = bisect_right(self._starts, position) - 1
        return self.shards[shard_index][position - int(self._starts[shard_index])]


def collate_decisions(samples: Sequence[DecisionSample]) -> DecisionBatch:
    """Pad samples to the batch maximum and derive an explicit valid mask."""

    if not samples:
        raise DataError("cannot collate an empty batch")
    batch_size = len(samples)
    max_candidates = max(sample.candidate_count for sample in samples)
    states = torch.empty((batch_size, STATE_WIDTH), dtype=torch.float32)
    candidates = torch.zeros(
        (batch_size, max_candidates, CANDIDATE_WIDTH), dtype=torch.float32
    )
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    targets = torch.empty(batch_size, dtype=torch.int64)
    stages = torch.empty(batch_size, dtype=torch.int64)
    target_kinds = torch.empty(batch_size, dtype=torch.int64)
    actor_seats: list[int] = []
    decision_indices: list[int] = []
    episode_ids: list[object] = []
    have_actor_seats = True
    have_decision_indices = True
    have_episode_ids = True

    for row, sample in enumerate(samples):
        if sample.state.shape != (STATE_WIDTH,):
            raise DataError(
                f"sample state must have shape ({STATE_WIDTH},), "
                f"got {sample.state.shape}"
            )
        if (
            sample.candidate_features.ndim != 2
            or sample.candidate_features.shape[1] != CANDIDATE_WIDTH
        ):
            raise DataError(
                f"sample candidates must have shape [N, {CANDIDATE_WIDTH}], "
                f"got {sample.candidate_features.shape}"
            )
        count = sample.candidate_count
        if count < 1:
            raise DataError("a decision must have at least one candidate")
        if sample.target < 0 or sample.target >= count:
            raise DataError("sample target is outside its candidate range")
        states[row].copy_(torch.tensor(sample.state, dtype=torch.float32))
        candidates[row, :count].copy_(
            torch.tensor(sample.candidate_features, dtype=torch.float32)
        )
        mask[row, :count] = True
        targets[row] = sample.target
        stages[row] = sample.stage
        target_kinds[row] = sample.target_kind
        if sample.actor_seat is None:
            have_actor_seats = False
        else:
            actor_seats.append(int(sample.actor_seat))
        if sample.decision_index is None:
            have_decision_indices = False
        else:
            decision_indices.append(int(sample.decision_index))
        if sample.episode_id is None:
            have_episode_ids = False
        else:
            episode_ids.append(sample.episode_id)

    eligible = mask.sum(dim=1) > FORCED_CANDIDATE_COUNT
    return DecisionBatch(
        states=states,
        candidates=candidates,
        mask=mask,
        targets=targets,
        stages=stages,
        target_kinds=target_kinds,
        eligible=eligible,
        actor_seats=(
            torch.tensor(actor_seats, dtype=torch.int64) if have_actor_seats else None
        ),
        decision_indices=(
            torch.tensor(decision_indices, dtype=torch.int64)
            if have_decision_indices
            else None
        ),
        episode_ids=tuple(episode_ids) if have_episode_ids else None,
    )


def _iter_dataset_arrays(dataset: DecisionDataset) -> Iterator[np.ndarray]:
    if isinstance(dataset, PackedDecisionDataset):
        arrays = (
            dataset.states,
            dataset.candidate_features,
            dataset.offsets,
            dataset.targets,
            dataset.stages,
            dataset.actor_seats,
            dataset.decision_indices,
            dataset.episode_ids,
            dataset.target_kinds,
        )
    elif isinstance(dataset, ShardedDecisionDataset):
        arrays = tuple(
            array
            for shard in dataset.shards
            for array in _iter_dataset_arrays(shard)
        )
    else:
        arrays = ()
    for array in arrays:
        if isinstance(array, np.ndarray):
            yield array


def contains_memory_mapped_arrays(dataset: DecisionDataset) -> bool:
    """Whether any dataset array is a memory map that worker processes would copy."""

    return any(isinstance(array, np.memmap) for array in _iter_dataset_arrays(dataset))


class _IndexSubset(Dataset[DecisionSample]):
    def __init__(self, dataset: DecisionDataset, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> DecisionSample:
        return self.dataset[int(self.indices[position])]


def iter_batches(
    dataset: DecisionDataset,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> Iterator[DecisionBatch]:
    """Yield batches in the exact supplied index order.

    The final partial batch is always included. Memory-mapped datasets reject
    ``num_workers > 0`` because spawned workers would copy the whole map; use
    the caller process instead. Plain in-memory arrays may use a non-persistent
    ``DataLoader`` while preserving batch order.
    """

    if batch_size < 1:
        raise DataError("batch_size must be positive")
    if num_workers < 0:
        raise DataError("num_workers must be non-negative")
    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.ndim != 1:
        raise DataError("indices must be a one-dimensional sequence")
    if num_workers > 0 and contains_memory_mapped_arrays(dataset):
        raise DataError(
            "num_workers > 0 is not supported for memory-mapped datasets; "
            "worker spawning would copy whole maps. Use num_workers=0."
        )
    if num_workers == 0:
        for start in range(0, ordered.size, batch_size):
            chunk = ordered[start : start + batch_size]
            yield collate_decisions([dataset[int(index)] for index in chunk])
        return
    loader = DataLoader(
        _IndexSubset(dataset, ordered),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_decisions,
        drop_last=False,
        persistent_workers=False,
    )
    yield from loader
