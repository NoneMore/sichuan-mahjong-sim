"""Packed decision dataset, padded batches, and deterministic batch iteration.

This module is independent of the on-disk cache manifest. A cache loader can
back :class:`PackedDecisionDataset` with memory-mapped ``.npy`` arrays; training
and metrics consume only the packed array contract documented by the
offline-training specification. :func:`pad_decision_inputs` exposes the
label-free padding/mask primitive shared with online inference, while
:func:`collate_decisions` adds supervised targets and metadata. Those two are
the generic per-sample primitives; :func:`iter_batches` gathers the built-in
packed datasets column-wise instead of materializing one sample per decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ._array_contract import ArrayContractViolation, validate_packed_arrays
from .config import DatasetIdentity
from .constants import (
    ACTION_KIND_WIDTH,
    CANDIDATE_WIDTH,
    FORCED_CANDIDATE_COUNT,
    STAGE_COUNT,
    STATE_WIDTH,
)
from .errors import DataError, MahjongBCError


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
class DecisionInputs:
    """Label-free padded model inputs for a batch of decisions.

    ``states`` is ``[B, STATE_WIDTH]``; ``candidates`` is
    ``[B, max_candidates, CANDIDATE_WIDTH]`` with zero-valued padding; ``mask``
    is a boolean ``[B, max_candidates]`` valid-candidate mask. Candidate order
    is preserved. Supervision fields belong to :class:`DecisionBatch`.
    """

    states: Tensor
    candidates: Tensor
    mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidates.shape[1])

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> DecisionInputs:
        target = torch.device(device)
        return DecisionInputs(
            states=self.states.to(target, non_blocking=non_blocking),
            candidates=self.candidates.to(target, non_blocking=non_blocking),
            mask=self.mask.to(target, non_blocking=non_blocking),
        )


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


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class DecisionDataset(Dataset[DecisionSample], ABC):
    """Common interface for single-shard and multi-shard packed splits.

    ``identity`` is the split identity when the dataset represents a complete
    split. A standalone shard may have no split identity. Implementations must
    state whether their arrays are memory maps so worker spawning can reject
    them instead of copying whole maps.
    """

    identity: DatasetIdentity | None

    @property
    @abstractmethod
    def is_memory_mapped(self) -> bool:
        """Whether any array owned by this dataset is a ``np.memmap``."""

    @abstractmethod
    def eligible_indices(self) -> np.ndarray:
        """Original split indices of decisions with more than one candidate."""

    @abstractmethod
    def forced_by_stage(self) -> tuple[int, ...]:
        """Forced decision counts in encoding stage order."""

    @abstractmethod
    def candidate_counts(self) -> np.ndarray:
        """Candidate count of every decision in split order."""


def require_dataset_identity(
    dataset: DecisionDataset,
    role: str,
    *,
    dataset_error: type[MahjongBCError] = DataError,
) -> DatasetIdentity:
    """Validate a public dataset boundary and return its constructed identity.

    Third-party ``DecisionDataset`` subclasses are supported inputs, but an
    anonymous shard or a duck-typed ``identity`` that bypassed
    :class:`DatasetIdentity` construction is rejected. ``dataset_error``
    lets an established public entry point keep its own configuration error
    category for a non-dataset input.
    """

    if not isinstance(dataset, DecisionDataset):
        raise dataset_error(f"{role} must be a DecisionDataset")
    identity = dataset.identity
    if identity is None:
        raise DataError(
            f"{role} must expose a split identity; anonymous shards cannot be used"
        )
    if not isinstance(identity, DatasetIdentity):
        raise DataError(
            f"{role} identity must be a DatasetIdentity, "
            f"got {type(identity).__name__}"
        )
    return identity


class PackedDecisionDataset(DecisionDataset):
    """Packed, optionally memory-mapped decision arrays for one split or shard.

    ``candidate_features`` is packed across decisions. ``offsets`` delimits
    each decision's nonempty candidate range; ``targets`` are local candidate
    indices. The constructor validates the complete packed layout. ``identity``
    is ``None`` when this object is only a shard of a split; a complete split
    carries the :class:`DatasetIdentity` whose counts the arrays must match.
    """

    identity: DatasetIdentity | None

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
        try:
            layout = validate_packed_arrays(
                states=states,
                candidate_features=candidate_features,
                candidate_offsets=offsets,
                targets=targets,
                stages=stages,
                actor_seats=actor_seats,
                decision_indices=decision_indices,
                target_kinds=target_kinds,
            )
        except ArrayContractViolation as exc:
            raise DataError(str(exc)) from None
        if episode_ids is not None and len(episode_ids) != layout.decisions:
            raise DataError(
                f"episode_ids must contain {layout.decisions} entries, "
                f"got {len(episode_ids)}"
            )
        if identity is not None:
            if identity.decisions != layout.decisions:
                raise DataError(
                    f"dataset identity declares {identity.decisions} decisions, "
                    f"arrays contain {layout.decisions}"
                )
            if identity.eligible_decisions != layout.eligible_count:
                raise DataError(
                    f"dataset identity declares {identity.eligible_decisions} eligible "
                    f"decisions, arrays contain {layout.eligible_count}"
                )

        self._closed = False
        self._decision_count = layout.decisions
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
        self._candidate_counts = layout.candidate_counts
        self._eligible_indices = layout.eligible_indices
        self._eligible_mask = layout.eligible_mask

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

    @property
    def is_memory_mapped(self) -> bool:
        return any(
            isinstance(array, np.memmap)
            for array in (
                self.states,
                self.candidate_features,
                self.offsets,
                self.targets,
                self.stages,
                self.actor_seats,
                self.decision_indices,
                self.episode_ids,
                self.target_kinds,
            )
        )

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

    def _gather_batch(self, positions: np.ndarray) -> DecisionBatch:
        """Gather one batch column-wise instead of building a sample per row."""

        if self._closed:
            raise DataError("dataset is closed")
        _check_batch_positions(positions, len(self))
        rows = np.arange(positions.size, dtype=np.int64)
        segments = ((rows, self, positions),)
        if _unsigned_metadata_overflows(segments):
            return _reference_batch(self, positions)
        return _gather_segments(positions.size, segments)


class ShardedDecisionDataset(DecisionDataset):
    """A complete split backed by multiple independently memory-mapped shards.

    Global indices follow manifest shard order. Each shard is itself a
    :class:`PackedDecisionDataset` whose identity may be absent; this wrapper
    owns the one non-optional split :class:`DatasetIdentity`. No array
    concatenation is performed.
    """

    identity: DatasetIdentity

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

    @property
    def is_memory_mapped(self) -> bool:
        return any(shard.is_memory_mapped for shard in self.shards)

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

    def _gather_batch(self, positions: np.ndarray) -> DecisionBatch:
        """Gather one batch across shards with vectorized array indexing."""

        if self._closed:
            raise DataError("dataset is closed")
        _check_batch_positions(positions, len(self))
        if positions.size == 0:
            raise DataError("cannot collate an empty batch")
        shard_of_row = np.searchsorted(self._starts, positions, side="right") - 1
        segments: list[tuple[np.ndarray, PackedDecisionDataset, np.ndarray]] = []
        for index in np.unique(shard_of_row):
            rows = np.flatnonzero(shard_of_row == index)
            shard = self.shards[int(index)]
            if shard._closed:
                raise DataError("dataset is closed")
            segments.append((rows, shard, positions[rows] - self._starts[int(index)]))
        if _unsigned_metadata_overflows(segments):
            return _reference_batch(self, positions)
        return _gather_segments(positions.size, tuple(segments))


def _check_batch_positions(positions: np.ndarray, limit: int) -> None:
    """Raise ``__getitem__``'s error for the first out-of-range decision."""

    if positions.size == 0:
        return
    invalid = (positions < 0) | (positions >= limit)
    if invalid.any():
        first = int(np.argmax(invalid))
        raise IndexError(f"decision index {int(positions[first])} is out of range")


def _episode_values(
    values: Sequence[object] | np.ndarray, positions: np.ndarray
) -> Iterator[object]:
    """Yield one shard's episode ids, keeping their stored Python types."""

    if isinstance(values, np.ndarray):
        yield from values[positions]
        return
    for position in positions:
        yield values[int(position)]


def _reference_batch(dataset: DecisionDataset, positions: np.ndarray) -> DecisionBatch:
    """Build one batch the documented per-sample way.

    Used when a batch carries a value the vectorized path cannot convert
    faithfully, so the caller's error or conversion behavior is preserved.
    """

    return collate_decisions([dataset[int(index)] for index in positions])


def _unsigned_metadata_overflows(
    segments: Sequence[tuple[np.ndarray, PackedDecisionDataset, np.ndarray]],
) -> bool:
    """Whether gathered unsigned metadata exceeds the int64 batch dtype.

    ``decision_indices`` is unsigned on the wire but int64 in a batch, so a
    value at or above ``2**63`` must fail like the per-sample tensor build
    instead of silently wrapping. Only the rows of this batch are inspected.
    """

    limit = int(np.iinfo(np.int64).max)
    for _, shard, local in segments:
        values = shard.decision_indices
        if values is None or not local.size:
            continue
        if not np.issubdtype(values.dtype, np.unsignedinteger):
            continue
        if int(values[local].max()) > limit:
            return True
    return False


def _gather_segments(
    batch_size: int,
    segments: Sequence[tuple[np.ndarray, PackedDecisionDataset, np.ndarray]],
) -> DecisionBatch:
    """Gather one batch from packed shards with vectorized array reads.

    Each segment pairs ascending batch rows with the shard that owns them and
    their shard-local decision positions. Values, order, dtypes, padding and
    metadata presence match :func:`collate_decisions` over the same decisions;
    only the per-sample copies are replaced by column-wise gathers.
    """

    if batch_size < 1:
        raise DataError("cannot collate an empty batch")
    states = np.empty((batch_size, STATE_WIDTH), dtype=np.float32)
    counts = np.empty(batch_size, dtype=np.int64)
    starts = np.empty(batch_size, dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    stages = np.empty(batch_size, dtype=np.int64)
    target_kinds = np.empty(batch_size, dtype=np.int64)
    have_actor_seats = all(shard.actor_seats is not None for _, shard, _ in segments)
    have_decision_indices = all(
        shard.decision_indices is not None for _, shard, _ in segments
    )
    have_episode_ids = all(shard.episode_ids is not None for _, shard, _ in segments)
    actor_seats = np.empty(batch_size, dtype=np.int64) if have_actor_seats else None
    decision_indices = (
        np.empty(batch_size, dtype=np.int64) if have_decision_indices else None
    )
    episode_values: list[object] = [None] * batch_size

    for rows, shard, local in segments:
        # Counts come from the current offsets, not the cached layout: an in
        # place offsets edit stays visible to batching, as it is to
        # ``__getitem__``.
        segment_starts = np.asarray(shard.offsets[local], dtype=np.int64)
        counts[rows] = (
            np.asarray(shard.offsets[local + 1], dtype=np.int64) - segment_starts
        )
        starts[rows] = segment_starts
        states[rows] = shard.states[local]
        targets[rows] = shard.targets[local]
        stages[rows] = shard.stages[local]
        target_kinds[rows] = shard.target_kinds[local]
        if actor_seats is not None:
            actor_seats[rows] = shard.actor_seats[local]
        if decision_indices is not None:
            decision_indices[rows] = shard.decision_indices[local]
        if have_episode_ids:
            for row, value in zip(
                rows.tolist(), _episode_values(shard.episode_ids, local)
            ):
                episode_values[row] = value
    # Packed layouts validate targets at construction; recheck the gathered
    # rows so an array mutated afterwards still fails like collate_decisions.
    if np.any((targets < 0) | (targets >= counts)):
        raise DataError("sample target is outside its candidate range")

    max_count = int(counts.max())
    candidates = np.zeros((batch_size, max_count, CANDIDATE_WIDTH), dtype=np.float32)
    for rows, shard, local in segments:
        segment_counts = counts[rows]
        segment_starts = starts[rows]
        if np.all(segment_counts == segment_counts[0]):
            width = int(segment_counts[0])
            columns = np.arange(width)
            candidates[rows, :width] = shard.candidate_features[
                segment_starts[:, None] + columns
            ]
            continue
        exclusive = np.cumsum(segment_counts) - segment_counts
        total = int(segment_counts.sum())
        flat_rows = np.repeat(rows, segment_counts)
        flat_columns = np.arange(total) - np.repeat(exclusive, segment_counts)
        source_rows = np.repeat(segment_starts - exclusive, segment_counts) + np.arange(
            total
        )
        candidates[flat_rows, flat_columns] = shard.candidate_features[source_rows]

    mask = np.arange(max_count)[None, :] < counts[:, None]
    return DecisionBatch(
        states=torch.from_numpy(states),
        candidates=torch.from_numpy(candidates),
        mask=torch.from_numpy(mask),
        targets=torch.from_numpy(targets),
        stages=torch.from_numpy(stages),
        target_kinds=torch.from_numpy(target_kinds),
        eligible=torch.from_numpy(counts > FORCED_CANDIDATE_COUNT),
        actor_seats=None if actor_seats is None else torch.from_numpy(actor_seats),
        decision_indices=(
            None if decision_indices is None else torch.from_numpy(decision_indices)
        ),
        episode_ids=(
            tuple(episode_values)
            if have_episode_ids and all(value is not None for value in episode_values)
            else None
        ),
    )


def _packed_batch_gatherer(
    dataset: DecisionDataset,
) -> Callable[[np.ndarray], DecisionBatch] | None:
    """Return the vectorized gatherer for the built-in packed datasets.

    Only the exact built-in classes qualify: a subclass may override
    ``__getitem__``, and its per-sample semantics must keep applying. Shards
    without ``target_kinds`` are also excluded so target-kind derivation keeps
    its per-sample validation.
    """

    if type(dataset) is PackedDecisionDataset:
        if dataset.target_kinds is None:
            return None
        return dataset._gather_batch
    if type(dataset) is ShardedDecisionDataset:
        for shard in dataset.shards:
            if type(shard) is not PackedDecisionDataset or shard.target_kinds is None:
                return None
        return dataset._gather_batch
    return None


def _as_float32_tensor(value: object) -> Tensor:
    """Convert an array-like value to float32 for padded batch construction."""

    if isinstance(value, Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _decision_arrays(decision: object) -> tuple[Tensor, Tensor]:
    """Validate one decision's label-free arrays and return float32 tensors."""

    try:
        state = decision.state  # type: ignore[attr-defined]
        candidate_features = decision.candidate_features  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise DataError(
            "a decision must expose state and candidate_features arrays"
        ) from exc
    try:
        state_tensor = _as_float32_tensor(state)
        candidates_tensor = _as_float32_tensor(candidate_features)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DataError(f"decision features must be numeric arrays: {exc}") from exc
    if state_tensor.shape != (STATE_WIDTH,):
        raise DataError(
            f"decision state must have shape ({STATE_WIDTH},), "
            f"got {tuple(state_tensor.shape)}"
        )
    if (
        candidates_tensor.ndim != 2
        or candidates_tensor.shape[1] != CANDIDATE_WIDTH
    ):
        raise DataError(
            f"decision candidates must have shape [N, {CANDIDATE_WIDTH}], "
            f"got {tuple(candidates_tensor.shape)}"
        )
    if candidates_tensor.shape[0] < 1:
        raise DataError("a decision must have at least one candidate")
    return state_tensor, candidates_tensor


def pad_decision_inputs(decisions: Sequence[object]) -> DecisionInputs:
    """Pad decision states and candidate features to the batch maximum.

    Accepts any objects exposing ``state`` and ``candidate_features`` arrays,
    such as an offline :class:`DecisionSample` or an online binding snapshot.
    Only model inputs are produced: valid scores and canonical candidate order
    are preserved, padding rows are zero, and no labels or metadata are
    required.
    """

    if not decisions:
        raise DataError("cannot pad an empty batch of decisions")
    arrays = [_decision_arrays(decision) for decision in decisions]
    batch_size = len(arrays)
    max_candidates = max(int(features.shape[0]) for _, features in arrays)
    states = torch.empty((batch_size, STATE_WIDTH), dtype=torch.float32)
    candidates = torch.zeros(
        (batch_size, max_candidates, CANDIDATE_WIDTH), dtype=torch.float32
    )
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    for row, (state, features) in enumerate(arrays):
        count = int(features.shape[0])
        states[row].copy_(state)
        candidates[row, :count].copy_(features)
        mask[row, :count] = True
    return DecisionInputs(states=states, candidates=candidates, mask=mask)


def collate_decisions(samples: Sequence[DecisionSample]) -> DecisionBatch:
    """Pad samples to the batch maximum and add supervision and metadata."""

    if not samples:
        raise DataError("cannot collate an empty batch")
    batch_size = len(samples)
    inputs = pad_decision_inputs(samples)
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
        count = int(inputs.mask[row].sum())
        if sample.target < 0 or sample.target >= count:
            raise DataError("sample target is outside its candidate range")
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

    eligible = inputs.mask.sum(dim=1) > FORCED_CANDIDATE_COUNT
    return DecisionBatch(
        states=inputs.states,
        candidates=inputs.candidates,
        mask=inputs.mask,
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


def contains_memory_mapped_arrays(dataset: DecisionDataset) -> bool:
    """Whether the dataset owns a memory map that worker processes would copy.

    This delegates to the dataset's ``is_memory_mapped`` contract. Unknown
    implementations are rejected rather than assumed to be map-free.
    """

    if not isinstance(dataset, DecisionDataset):
        raise DataError(
            "dataset must be a DecisionDataset implementation with an "
            "is_memory_mapped contract"
        )
    return dataset.is_memory_mapped


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

    With ``num_workers=0`` the built-in packed datasets are gathered column-wise
    instead of materializing one sample per decision. Any other
    ``DecisionDataset``, and packed shards without target-kind metadata, keep
    the per-sample :func:`collate_decisions` path.
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
        gather = _packed_batch_gatherer(dataset)
        if gather is not None:
            for start in range(0, ordered.size, batch_size):
                yield gather(ordered[start : start + batch_size])
            return
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
