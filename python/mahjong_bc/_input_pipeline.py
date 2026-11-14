"""Shared host-to-device batch pipeline for training and evaluation.

``BatchTransferConfig`` selects how host batches reach the model device:

``sync``
    The historical behavior. Batches are gathered by the consumer's iterator
    and moved with one blocking ``DecisionBatch.to(device)`` call.

``cuda-prefetch``
    One CPU producer fills two fixed-capacity pinned slots; the consumer thread
    submits each finished slot on a private copy stream and makes its compute
    stream wait on that batch's event before the batch is yielded. Slots are
    rewritten only after their host-to-device copy finished, so a delivered
    device batch never aliases staging memory.

The producer never creates a stream, submits device work, or holds a device
tensor. The consumer keeps at most the delivered batch and one committed
lookahead batch on the device; both are ordinary allocator tensors whose
lifetime is tracked with ``record_stream``. Staging capacity is checked and
allocated before the producer starts, so an over-budget request fails before
any transfer begins.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from ._device import resolve_device
from .constants import CANDIDATE_WIDTH, STATE_WIDTH
from .data import (
    DecisionDataset,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    _PackedCandidateBatch,
    _iter_model_batches,
)
from .errors import ConfigurationError, DataError
from .policy import _supports_packed_candidate_scoring

#: Transfer host batches one at a time with a blocking copy.
SYNC = "sync"
#: Transfer host batches through pinned slots and a private copy stream.
CUDA_PREFETCH = "cuda-prefetch"
MODES = (SYNC, CUDA_PREFETCH)

#: Slot A is filled while slot B is read by the host-to-device copy.
_SLOT_COUNT = 2

# Element sizes of the staged dtypes. The pinned budget formula in the
# specification is ``2 * (4060B + 165BC)`` for training and
# ``2 * (4076B + 165BC)`` for evaluation, which these widths reproduce from the
# array shapes instead of hard-coded totals.
_FLOAT32_BYTES = 4
_INT64_BYTES = 8
_MASK_BYTES = 1

#: Arrays the prefetch producer reads. Provenance arrays (``actor_seats``,
#: ``decision_indices``, ``episode_ids``) and the derived ``eligible`` tensor
#: are never read: neither is staged, so prefetch also never inspects them.
_STAGED_ARRAYS = (
    "states",
    "candidate_features",
    "offsets",
    "targets",
    "stages",
    "target_kinds",
)


@dataclass(frozen=True)
class BatchTransferConfig:
    """How batches reach the device; execution-only, never training identity.

    ``cuda-prefetch`` requires an explicit positive pinned budget because the
    two staging slots are allocated up front. ``sync`` has no staging at all,
    so it rejects a pinned budget instead of ignoring it.
    """

    mode: str = SYNC
    pinned_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ConfigurationError(
                f"unknown batch transfer mode {self.mode!r}; "
                f"use one of {', '.join(MODES)}"
            )
        if self.mode == SYNC:
            if self.pinned_limit_bytes is not None:
                raise ConfigurationError(
                    "sync batch transfer does not stage batches and does not "
                    "accept pinned_limit_bytes"
                )
            return
        budget = self.pinned_limit_bytes
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ConfigurationError(
                "cuda-prefetch batch transfer requires an explicit integer "
                "pinned_limit_bytes"
            )
        if budget <= 0:
            raise ConfigurationError(
                f"pinned_limit_bytes must be a positive integer, got {budget}"
            )

    def to_dict(self) -> dict[str, object]:
        """Run-record form; both fields describe this launch only."""

        return {
            "mode": self.mode,
            "pinned_limit_bytes": self.pinned_limit_bytes,
        }


def _transfer_config(config: BatchTransferConfig | None) -> BatchTransferConfig:
    if config is None:
        return BatchTransferConfig()
    if not isinstance(config, BatchTransferConfig):
        raise ConfigurationError("batch_transfer must be a BatchTransferConfig or None")
    return config


def _prefetch_shards(dataset: DecisionDataset) -> tuple[PackedDecisionDataset, ...]:
    """Return the packed shards an explicit prefetch request may traverse.

    Only the exact built-in types qualify. A ``DecisionDataset`` subclass may
    override ``eligible_indices`` or ``candidate_counts`` with values that do
    not describe the rows it returns, so custom types keep the synchronous
    fallback instead of silently staging a different batch.
    """

    if type(dataset) is PackedDecisionDataset:
        if dataset.identity is None:
            raise ConfigurationError(
                "cuda-prefetch batch transfer requires a split identity; "
                "anonymous shards cannot be used"
            )
        shards: tuple[PackedDecisionDataset, ...] = (dataset,)
    elif type(dataset) is ShardedDecisionDataset and all(
        type(shard) is PackedDecisionDataset for shard in dataset.shards
    ):
        shards = dataset.shards
    else:
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires the built-in "
            "PackedDecisionDataset or a ShardedDecisionDataset of "
            f"PackedDecisionDataset shards, got {type(dataset).__name__}"
        )
    for shard in shards:
        _require_stageable_shard(shard)
    return shards


def _require_stageable_shard(shard: PackedDecisionDataset) -> None:
    """Check one shard exposes read-only, self-consistent packed inputs."""

    if shard.target_kinds is None:
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires target_kinds metadata"
        )
    for name in _STAGED_ARRAYS:
        array = getattr(shard, name)
        if not isinstance(array, np.ndarray):
            raise ConfigurationError(
                f"cuda-prefetch batch transfer requires an open {name} array"
            )
        if array.flags.writeable:
            raise ConfigurationError(
                f"cuda-prefetch batch transfer requires read-only input arrays; "
                f"{name} is writable and could change during asynchronous "
                "consumption"
            )
    if not _offsets_match_cached_counts(shard):
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires offsets matching the cached "
            "candidate counts validated at construction"
        )


def _offsets_match_cached_counts(shard: PackedDecisionDataset) -> bool:
    """Whether a shard's current offsets still imply its validated counts.

    Mirrors the layout predicate in :mod:`mahjong_bc.metrics`; it is imported
    lazily there because this module is itself used by that evaluation path.
    """

    from .metrics import _offsets_match_cached_counts as check

    return check(shard)


def validate_batch_transfer(
    dataset: DecisionDataset,
    *,
    model: object,
    device: torch.device | str,
    num_workers: int,
    config: BatchTransferConfig,
    training: bool,
    packed_candidates: bool,
    eligible_only: bool = False,
) -> None:
    """Reject an explicit transfer request the inputs cannot support.

    ``sync`` keeps every historical input, worker and layout allowance, so it
    validates nothing beyond the config itself. ``cuda-prefetch`` runs before
    any producer, slot or pinned allocation and raises
    :class:`ConfigurationError` instead of falling back to another mode.
    """

    resolved_config = _transfer_config(config)
    if resolved_config.mode == SYNC:
        return
    resolved_device = resolve_device(device)
    if resolved_device.type != "cuda":
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires a CUDA device, "
            f"got {resolved_device.type!r}"
        )
    if num_workers != 0:
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires num_workers=0; worker "
            "processes cannot fill the pinned slots of this pipeline"
        )
    if packed_candidates is not True:
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires flat candidates; pass "
            "packed_candidates=True"
        )
    if not _supports_packed_candidate_scoring(model):
        raise ConfigurationError(
            "cuda-prefetch batch transfer requires the built-in CandidatePolicy; "
            f"got {type(model).__name__}"
        )
    if not training and not eligible_only:
        raise ConfigurationError(
            "cuda-prefetch evaluation requires an eligible-only traversal: "
            "forced decisions are counted from split metadata, not staged"
        )
    _prefetch_shards(dataset)


@contextlib.contextmanager
def device_batches(
    dataset: DecisionDataset,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
    num_workers: int = 0,
    training: bool,
    packed_candidates: bool = False,
    model: object,
    device: torch.device | str,
    batch_transfer: BatchTransferConfig | None = None,
    eligible_only: bool = False,
    batch_iterator: Callable[..., Iterator[object]] | None = None,
    host_validator: Callable[[object], None] | None = None,
) -> Iterator[Iterator[object]]:
    """Yield an iterator of batches already on ``device``.

    The consumer must finish with the iterator before the context exits: exit
    stops publishing, cancels slot and queue waits, joins the producer, lets
    submitted host-to-device copies finish and only then releases staging.

    ``batch_iterator`` is the traversal factory ``(dataset, indices,
    **iter_kwargs)`` used by both modes; it defaults to
    :func:`mahjong_bc.data._iter_model_batches`. ``sync`` moves the batches it
    yields unchanged, calling ``host_validator`` on each host batch first when
    one is supplied. ``cuda-prefetch`` runs the same factory on its producer
    thread and stages only the fields training or evaluation consumes; the
    factory must then yield flat-candidate batches (``packed_candidates``).
    ``host_validator`` applies to ``sync`` only: prefetch batches are validated
    by the strict prefetch checks and carry no public batch fields to check.
    An empty traversal creates no producer and no pinned slots.
    """

    config = _transfer_config(batch_transfer)
    validate_batch_transfer(
        dataset,
        model=model,
        device=device,
        num_workers=num_workers,
        config=config,
        training=training,
        packed_candidates=packed_candidates,
        eligible_only=eligible_only,
    )
    factory = _batch_factory(batch_iterator)
    if config.mode == SYNC:
        source = factory(
            dataset,
            indices,
            batch_size=batch_size,
            num_workers=num_workers,
            training=training,
            packed_candidates=packed_candidates,
        )
        yield _sync_batches(source, torch.device(device), host_validator)
        return
    if batch_size < 1:
        raise DataError("batch_size must be positive")
    pipeline = _PrefetchPipeline(
        dataset=dataset,
        indices=_ordered_indices(indices),
        batch_size=batch_size,
        training=training,
        device=resolve_device(device),
        pinned_limit_bytes=config.pinned_limit_bytes,
        factory=factory,
    )
    try:
        yield _PrefetchBatches(pipeline)
    except BaseException:
        # A cleanup failure must never replace the consumer's error; the
        # pipeline is torn down on a best-effort basis either way.
        try:
            pipeline.close()
        except BaseException:
            pass
        raise
    else:
        pipeline.close()


def _batch_factory(
    batch_iterator: Callable[..., Iterator[object]] | None,
) -> Callable[..., Iterator[object]]:
    """Resolve the traversal seam shared by both transfer modes."""

    if batch_iterator is None:
        return _iter_model_batches
    if not callable(batch_iterator):
        raise ConfigurationError(
            "batch_iterator must be a callable "
            "(dataset, indices, **iter_kwargs) returning a batch iterator"
        )
    return batch_iterator


def _ordered_indices(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.ndim != 1:
        raise DataError("indices must be a one-dimensional sequence")
    return ordered


def _sync_batches(
    batches: Iterator[object],
    device: torch.device,
    host_validator: Callable[[object], None] | None,
) -> Iterator[object]:
    """Validate each host batch, then move it with the historical call."""

    for batch in batches:
        if host_validator is not None:
            host_validator(batch)
        yield batch.to(device)


def _row_bytes(*, training: bool) -> int:
    """Per-decision bytes of one slot: states and targets, plus eval metadata."""

    total = _FLOAT32_BYTES * STATE_WIDTH + _INT64_BYTES
    if not training:
        total += 2 * _INT64_BYTES
    return total


def _candidate_bytes() -> int:
    """Per-candidate bytes of one slot: flat row, position and mask element."""

    return _FLOAT32_BYTES * CANDIDATE_WIDTH + _INT64_BYTES + _MASK_BYTES


def slot_capacity_bytes(*, batch_size: int, candidates: int, training: bool) -> int:
    """Pinned bytes of the two slots for ``batch_size`` rows of ``candidates``."""

    return _SLOT_COUNT * (
        _row_bytes(training=training) * batch_size
        + _candidate_bytes() * batch_size * candidates
    )


@dataclass
class _Slot:
    """One fixed-capacity pinned staging area.

    ``state`` is ``empty`` (host-writable), ``filling`` (owned by the producer),
    ``staged`` (filled, waiting for the consumer to submit it) or ``in_flight``
    (a host-to-device copy passed the slot event but has not been observed to
    finish by the producer).
    """

    states: Tensor
    targets: Tensor
    candidates: Tensor
    positions: Tensor
    mask: Tensor
    stages: Tensor | None
    target_kinds: Tensor | None
    event: "torch.cuda.Event"
    batch_capacity: int
    candidate_capacity: int
    state: str = "empty"


@dataclass(frozen=True)
class _Staged:
    """Host-complete batch handed from the producer to the consumer thread."""

    index: int
    slot: _Slot
    states: Tensor
    targets: Tensor
    candidates: Tensor
    mask: Tensor
    candidate_positions: Tensor
    stages: Tensor | None
    target_kinds: Tensor | None


@dataclass(frozen=True)
class _Failure:
    """A host-side gather or staging error, delivered in batch order."""

    index: int
    error: BaseException


@dataclass(frozen=True)
class _End:
    """The producer published every batch."""


@dataclass(frozen=True)
class _Committed:
    """A device batch whose copies are queued behind ``event``."""

    batch: _PackedCandidateBatch
    event: "torch.cuda.Event"


def _copy_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """Stage one pinned slot view on the current (copy) stream."""

    return tensor.to(device, non_blocking=True)


def _allocate_slot(
    *, batch_capacity: int, candidate_capacity: int, training: bool
) -> _Slot:
    """Allocate one pinned slot plus the event that guards its reuse."""

    flat = batch_capacity * candidate_capacity
    return _Slot(
        states=torch.empty(
            (batch_capacity, STATE_WIDTH), dtype=torch.float32, pin_memory=True
        ),
        targets=torch.empty(batch_capacity, dtype=torch.int64, pin_memory=True),
        candidates=torch.empty(
            (flat, CANDIDATE_WIDTH), dtype=torch.float32, pin_memory=True
        ),
        positions=torch.empty(flat, dtype=torch.int64, pin_memory=True),
        mask=torch.empty(flat, dtype=torch.bool, pin_memory=True),
        stages=(
            None
            if training
            else torch.empty(batch_capacity, dtype=torch.int64, pin_memory=True)
        ),
        target_kinds=(
            None
            if training
            else torch.empty(batch_capacity, dtype=torch.int64, pin_memory=True)
        ),
        event=torch.cuda.Event(),
        batch_capacity=batch_capacity,
        candidate_capacity=candidate_capacity,
    )


class _PrefetchPipeline:
    """One producer, two pinned slots and one private copy stream.

    Batch ``n`` always uses slot ``n % 2`` and batches are delivered in index
    order. The producer waits on a slot's copy event before rewriting it, so
    slot reuse never races a running host-to-device copy.
    """

    def __init__(
        self,
        *,
        dataset: DecisionDataset,
        indices: np.ndarray,
        batch_size: int,
        training: bool,
        device: torch.device,
        pinned_limit_bytes: int,
        factory: Callable[..., Iterator[object]],
    ) -> None:
        self._dataset = dataset
        self._indices = indices
        self._batch_size = batch_size
        self._training = training
        self._device = device
        self._factory = factory
        self._cond = threading.Condition()
        self._ready: deque[_Staged | _Failure | _End] = deque()
        self._cancelled = False
        self._producer_done = False
        self._finished = False
        self._closed = False
        self._current: _Committed | None = None
        self._lookahead: _Committed | None = None
        self._slots: list[_Slot] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self._producer: threading.Thread | None = None
        if indices.size == 0:
            # An empty traversal (for example an all-forced evaluation split)
            # has nothing to stage, so it never allocates a slot or a thread.
            self._producer_done = True
            return
        batch_capacity, candidate_capacity = self._capacity()
        required = slot_capacity_bytes(
            batch_size=batch_capacity,
            candidates=candidate_capacity,
            training=training,
        )
        if required > pinned_limit_bytes:
            raise ConfigurationError(
                "cuda-prefetch pinned budget is too small for this traversal: "
                f"two slots of {batch_capacity} rows and up to "
                f"{candidate_capacity} candidates need {required} bytes, "
                f"but pinned_limit_bytes is {pinned_limit_bytes}"
            )
        self._copy_stream = torch.cuda.Stream(device=device)
        for _ in range(_SLOT_COUNT):
            self._slots.append(
                _allocate_slot(
                    batch_capacity=batch_capacity,
                    candidate_capacity=candidate_capacity,
                    training=training,
                )
            )
        self._producer = threading.Thread(
            target=self._produce,
            name="mahjong-bc-prefetch",
            daemon=True,
        )
        self._producer.start()

    # ---- capacity -------------------------------------------------------

    def _capacity(self) -> tuple[int, int]:
        """Rows and candidate columns of one slot, from validated metadata.

        ``candidate_counts`` is the read-only layout the dataset validated at
        construction, and the prefetch validation required it to still match
        the offsets batching reads. Only the selected rows are inspected.
        """

        ordered = self._indices
        counts = np.asarray(self._dataset.candidate_counts(), dtype=np.int64)
        if counts.ndim != 1 or counts.size != len(self._dataset):
            raise DataError(
                "candidate_counts must hold one count per decision"
            )
        if np.any(ordered < 0) or np.any(ordered >= counts.size):
            first = int(np.argmax((ordered < 0) | (ordered >= counts.size)))
            raise IndexError(f"decision index {int(ordered[first])} is out of range")
        rows = int(min(self._batch_size, ordered.size))
        candidates = int(counts[ordered].max())
        return rows, candidates

    # ---- producer -------------------------------------------------------

    def _produce(self) -> None:
        index = 0
        try:
            for host_batch in self._factory(
                self._dataset,
                self._indices,
                batch_size=self._batch_size,
                num_workers=0,
                training=self._training,
                packed_candidates=True,
            ):
                slot = self._acquire(index % _SLOT_COUNT)
                if slot is None:
                    return
                self._stage(slot, host_batch, index)
                index += 1
        except BaseException as error:  # delivered to the consumer in batch order
            self._publish(_Failure(index=index, error=error))
        else:
            self._publish(_End())
        finally:
            with self._cond:
                self._producer_done = True
                self._cond.notify_all()

    def _publish(self, item: _Staged | _Failure | _End) -> None:
        with self._cond:
            self._ready.append(item)
            self._cond.notify_all()

    def _acquire(self, index: int) -> _Slot | None:
        """Wait for a writable slot, or ``None`` once the pipeline is cancelled."""

        slot = self._slots[index]
        while True:
            with self._cond:
                if self._cancelled:
                    return None
                state = slot.state
                if state == "empty":
                    slot.state = "filling"
                    return slot
                if state == "staged":
                    self._cond.wait()
                    continue
                if state != "in_flight":
                    raise RuntimeError(f"unexpected slot state {state!r}")
            # ``in_flight``: the copy that read this slot may still run. Wait on
            # its event instead of polling the consumer's progress.
            slot.event.synchronize()
            with self._cond:
                if slot.state == "in_flight":
                    slot.state = "empty"

    def _stage(self, slot: _Slot, host_batch: object, index: int) -> None:
        """Copy one gathered batch into the slot's true-shape views."""

        states = host_batch.states
        candidates = host_batch.candidates
        mask = host_batch.mask
        positions = getattr(host_batch, "candidate_positions", None)
        if positions is None or mask.dim() != 2:
            raise DataError("cuda-prefetch requires flat-candidate host batches")
        rows = int(states.shape[0])
        columns = int(mask.shape[1])
        flat = int(candidates.shape[0])
        capacity = slot.batch_capacity * slot.candidate_capacity
        if (
            rows > slot.batch_capacity
            or columns > slot.candidate_capacity
            or flat > capacity
        ):
            raise DataError(
                "batch exceeds the pinned slot capacity: "
                f"{rows} rows, {columns} candidates per row and {flat} flat "
                f"candidate rows, slot holds {slot.batch_capacity} rows, "
                f"{slot.candidate_capacity} candidates per row and {capacity} "
                "flat candidate rows"
            )
        if int(positions.numel()) != flat:
            raise DataError(
                "flat candidate positions do not match the flat candidate rows: "
                f"{int(positions.numel())} positions for {flat} rows"
            )
        slot.states[:rows].copy_(states)
        slot.targets[:rows].copy_(host_batch.targets)
        slot.candidates[:flat].copy_(candidates)
        slot.positions[:flat].copy_(positions)
        mask_view = slot.mask[: rows * columns].view(rows, columns)
        mask_view.copy_(mask)
        stages = target_kinds = None
        host_stages = getattr(host_batch, "stages", None)
        host_kinds = getattr(host_batch, "target_kinds", None)
        if slot.stages is not None:
            if host_stages is None or host_kinds is None:
                raise DataError(
                    "evaluation batches must carry stages and target_kinds"
                )
            slot.stages[:rows].copy_(host_stages)
            slot.target_kinds[:rows].copy_(host_kinds)
            stages = slot.stages[:rows]
            target_kinds = slot.target_kinds[:rows]
        elif host_stages is not None or host_kinds is not None:
            raise DataError("training batches do not stage evaluation metadata")
        item = _Staged(
            index=index,
            slot=slot,
            states=slot.states[:rows],
            targets=slot.targets[:rows],
            candidates=slot.candidates[:flat],
            mask=mask_view,
            candidate_positions=slot.positions[:flat],
            stages=stages,
            target_kinds=target_kinds,
        )
        with self._cond:
            slot.state = "staged"
            self._ready.append(item)
            self._cond.notify_all()

    # ---- consumer -------------------------------------------------------

    def next_batch(self) -> _PackedCandidateBatch:
        """Return the next device batch, submitting one lookahead ahead."""

        if self._finished:
            raise StopIteration
        try:
            return self._next_batch()
        except StopIteration:
            raise
        except BaseException:
            # A host or device failure ends this traversal; later calls must not
            # resume from a batch the consumer never saw delivered.
            self._finished = True
            raise

    def _next_batch(self) -> _PackedCandidateBatch:
        current = self._lookahead
        self._lookahead = None
        if current is None:
            current = self._take()
            if current is None:
                self._finished = True
                raise StopIteration
        self._current = current
        self._submit_lookahead()
        batch = self._deliver(current)
        self._current = None
        return batch

    def _submit_lookahead(self) -> None:
        """Submit an already staged next batch; never wait for one."""

        if self._lookahead is not None or self._finished:
            return
        with self._cond:
            item = self._ready[0] if self._ready else None
            if isinstance(item, _Staged):
                self._ready.popleft()
        if isinstance(item, _Staged):
            self._lookahead = self._commit(item)

    def _take(self) -> _Committed | None:
        """Return the next committed batch, or ``None`` at the end of input."""

        item = self._next_item()
        if item is None or isinstance(item, _End):
            self._finished = True
            return None
        if isinstance(item, _Failure):
            self._finished = True
            raise item.error
        return self._commit(item)

    def _next_item(self) -> _Staged | _Failure | _End | None:
        with self._cond:
            while not self._ready:
                if self._cancelled or self._producer_done:
                    return None
                self._cond.wait()
            return self._ready.popleft()

    def _commit(self, item: _Staged) -> _Committed:
        """Submit one staged slot on the copy stream and guard its reuse."""

        assert self._copy_stream is not None
        slot = item.slot
        with self._cond:
            with torch.cuda.device(self._device), torch.cuda.stream(
                self._copy_stream
            ):
                batch = _PackedCandidateBatch(
                    states=_copy_to_device(item.states, self._device),
                    candidates=_copy_to_device(item.candidates, self._device),
                    mask=_copy_to_device(item.mask, self._device),
                    targets=_copy_to_device(item.targets, self._device),
                    stages=(
                        None
                        if item.stages is None
                        else _copy_to_device(item.stages, self._device)
                    ),
                    target_kinds=(
                        None
                        if item.target_kinds is None
                        else _copy_to_device(item.target_kinds, self._device)
                    ),
                    candidate_positions=_copy_to_device(
                        item.candidate_positions, self._device
                    ),
                )
                slot.event.record()
            slot.state = "in_flight"
            self._cond.notify_all()
        return _Committed(batch=batch, event=slot.event)

    def _deliver(self, committed: _Committed) -> _PackedCandidateBatch:
        """Make the current compute stream wait for the copies, then hand over."""

        compute_stream = torch.cuda.current_stream(self._device)
        compute_stream.wait_event(committed.event)
        for tensor in _batch_tensors(committed.batch):
            tensor.record_stream(compute_stream)
        return committed.batch

    def close(self) -> None:
        """Stop the producer, let submitted copies finish, then free staging.

        Cancellation, the producer join and the staging release always run. A
        failed copy-stream wait is reported afterwards, so a clean exit cannot
        pass a device error off as success; the caller's own exception still
        wins over any cleanup error because ``device_batches`` suppresses this
        one on that path. The compute stream is never awaited.
        """

        if self._closed:
            return
        self._closed = True
        self._finished = True
        sync_error: BaseException | None = None
        join_error: BaseException | None = None
        with self._cond:
            self._cancelled = True
            self._cond.notify_all()
        try:
            if self._copy_stream is not None:
                # Only the host-to-device copies are awaited; the compute
                # stream may still be running a delivered batch.
                self._copy_stream.synchronize()
        except BaseException as error:
            sync_error = error
        try:
            producer = self._producer
            if producer is not None:
                producer.join()
        except BaseException as error:
            join_error = error
        finally:
            self._current = None
            self._lookahead = None
            self._ready.clear()
            self._slots.clear()
        if sync_error is not None:
            raise sync_error
        if join_error is not None:
            raise join_error


def _batch_tensors(batch: _PackedCandidateBatch) -> Iterator[Tensor]:
    """All device tensors of one batch, skipping absent metadata."""

    for name in (
        "states",
        "candidates",
        "mask",
        "targets",
        "stages",
        "target_kinds",
        "candidate_positions",
    ):
        value = getattr(batch, name)
        if value is not None:
            yield value


class _PrefetchBatches:
    """Iterator over device batches produced by one prefetch pipeline."""

    def __init__(self, pipeline: _PrefetchPipeline) -> None:
        self._pipeline = pipeline

    def __iter__(self) -> "_PrefetchBatches":
        return self

    def __next__(self) -> _PackedCandidateBatch:
        return self._pipeline.next_batch()
