"""Decision-aggregated NLL/top-1 metrics and complete-split evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .constants import (
    ACTION_KIND_COUNT,
    ACTION_KIND_NAMES,
    STAGE_COUNT,
    STAGE_NAMES,
)
from .data import (
    DecisionDataset,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    _iter_model_batches,
    iter_batches,
    require_dataset_identity,
)
from .errors import DataError
from .policy import (
    CandidatePolicy,
    _batch_logits,
    _decision_nll_validated,
    _greedy_indices_validated,
    _supports_packed_candidate_scoring,
)

#: Booleans read back per ``add`` call, in the order the error checks use.
_CHECK_COUNT = 9

#: Widths of the stage and kind sections of one transferred totals vector.
_STAGE_TOTAL_WIDTH = 4 * STAGE_COUNT
_KIND_TOTAL_WIDTH = 2 * ACTION_KIND_COUNT


@dataclass(frozen=True)
class StageMetrics:
    """Metrics for one decision stage over complete eligible decisions."""

    stage_id: int
    stage: str
    support: int
    forced: int
    nll: float | None
    top1: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage": self.stage,
            "support": self.support,
            "forced": self.forced,
            "nll": self.nll,
            "top1": self.top1,
        }


@dataclass(frozen=True)
class ActionKindMetrics:
    """Top-1 agreement grouped by the teacher's chosen action kind."""

    kind_id: int
    kind: str
    support: int
    top1: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind_id": self.kind_id,
            "kind": self.kind,
            "support": self.support,
            "top1": self.top1,
        }


@dataclass(frozen=True)
class MetricsReport:
    """Complete-split behavior-cloning evaluation."""

    decisions: int
    eligible: int
    forced: int
    nll: float | None
    top1: float | None
    stages: tuple[StageMetrics, ...]
    action_kinds: tuple[ActionKindMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "eligible": self.eligible,
            "forced": self.forced,
            "nll": self.nll,
            "top1": self.top1,
            "stages": [metric.to_dict() for metric in self.stages],
            "action_kinds": [metric.to_dict() for metric in self.action_kinds],
        }


class MetricAccumulator:
    """Accumulate metrics by summing per-decision NLL and correctness.

    Forced decisions (one candidate) are counted separately and never enter
    NLL/top-1 or their groups. Zero-support groups report ``None``, not zero.

    Group totals use fixed-shape masking over the small stage and kind axes.
    Dynamic operations (boolean indexing, ``bincount``, ``masked_select``) need
    a host-side output size and therefore synchronize on CUDA; this path does
    not use them, so it also supports deterministic CUDA algorithms. Totals
    are accumulated in batch order in ``float64`` on the host. Results are
    mathematically equivalent to the earlier weighted-``bincount`` grouping,
    but that CUDA reduction used atomics and is not bitwise comparable.

    Each ``add`` validates its batch immediately and reads back one host
    vector containing the checks plus totals buffered from the previous batch.
    :meth:`report` flushes the last pending totals.
    """

    def __init__(self) -> None:
        self._decisions = 0
        self._eligible = 0
        self._forced_by_stage = [0] * STAGE_COUNT
        self._nll_total = 0.0
        self._correct_total = 0
        self._support_by_stage = [0] * STAGE_COUNT
        self._nll_by_stage = [0.0] * STAGE_COUNT
        self._correct_by_stage = [0] * STAGE_COUNT
        self._support_by_kind = [0] * ACTION_KIND_COUNT
        self._correct_by_kind = [0] * ACTION_KIND_COUNT
        self._pending_totals: Tensor | None = None

    def add(
        self,
        logits: Tensor,
        mask: Tensor,
        targets: Tensor,
        stages: Tensor,
        target_kinds: Tensor,
    ) -> None:
        """Add one padded batch containing all decisions, forced included.

        Structural checks run on their tensors first. The value checks keep
        the public NLL/selection predicates and their messages, so a batch the
        public helpers would reject is rejected here before it is accumulated.
        Logits are still tested for NaN/Inf on every batch: they come from the
        model, not from validated data.

        Totals from the previous batch ride along with this batch's check
        vector. They are applied before any error is raised, so a rejected
        batch leaves earlier batches accounted for and only the rejected batch
        unaccumulated.
        """

        if not isinstance(logits, Tensor) or logits.dtype != torch.float32:
            raise DataError("logits must be a float32 tensor")
        if logits.dim() != 2:
            raise DataError("logits must have shape [B, C]")
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise DataError("mask must be a boolean tensor")
        if mask.shape != logits.shape:
            raise DataError("mask and logits shapes differ")
        if mask.device != logits.device:
            raise DataError("logits and mask must be on the same device")
        batch_size = int(logits.shape[0])
        if batch_size == 0:
            return
        if logits.shape[1] == 0:
            raise DataError("every decision must have at least one candidate")
        for name, value in (
            ("targets", targets),
            ("stages", stages),
            ("target_kinds", target_kinds),
        ):
            if not isinstance(value, Tensor) or value.dtype not in (
                torch.int64,
                torch.int32,
            ):
                raise DataError(f"{name} must be an integer tensor")
            if value.shape != (batch_size,):
                raise DataError(
                    f"{name} must have shape ({batch_size},), got {tuple(value.shape)}"
                )
            if value.device != logits.device:
                raise DataError(f"{name} and logits must be on the same device")
        eligible = mask.sum(dim=1) > 1
        checks = torch.stack(
            (
                mask.any(dim=1).all(),
                stages.min() >= 0,
                stages.max() < STAGE_COUNT,
                target_kinds.min() >= 0,
                target_kinds.max() < ACTION_KIND_COUNT,
                targets.min() >= 0,
                targets.max() < logits.shape[1],
                mask.gather(
                    1, targets.long().clamp(0, logits.shape[1] - 1).unsqueeze(1)
                ).all(),
                # Fixed-shape replacement for ``logits.masked_select(mask)``;
                # the latter sizes its output from device data and syncs.
                (~mask | torch.isfinite(logits)).all(),
            )
        ).to(torch.float64)
        pending = self._pending_totals
        if pending is not None and pending.device != logits.device:
            # Mixed-device callers keep working; only merging needs one device.
            self._flush_pending()
            pending = None
        if pending is None:
            values = checks.cpu().tolist()
        else:
            values = torch.cat((checks, pending)).cpu().tolist()
            self._pending_totals = None
            self._apply_totals(values[_CHECK_COUNT:])
        (
            candidate_present,
            stage_min_ok,
            stage_max_ok,
            kind_min_ok,
            kind_max_ok,
            target_min_ok,
            target_max_ok,
            target_valid,
            logits_finite,
        ) = (bool(value) for value in values[:_CHECK_COUNT])
        if not candidate_present:
            raise DataError("every decision must have at least one candidate")
        if not (stage_min_ok and stage_max_ok):
            raise DataError(f"stages must be in [0, {STAGE_COUNT})")
        if not (kind_min_ok and kind_max_ok):
            raise DataError(f"target_kinds must be in [0, {ACTION_KIND_COUNT})")
        if not (target_min_ok and target_max_ok):
            raise DataError("target index is outside the padded candidate range")
        if not target_valid:
            raise DataError("target index does not reference a valid candidate")
        if not logits_finite:
            raise DataError("valid candidate logits must be finite")

        nll = _decision_nll_validated(logits, mask, targets)
        correct = _greedy_indices_validated(logits, mask).eq(targets.long())

        self._decisions += batch_size
        self._pending_totals = _batch_totals(
            nll, correct, eligible, stages, target_kinds
        )

    def _apply_totals(self, totals: Sequence[float]) -> None:
        """Add one transferred totals vector in the documented layout."""

        kind_base = _STAGE_TOTAL_WIDTH
        summary_base = kind_base + _KIND_TOTAL_WIDTH
        self._eligible += int(totals[summary_base + 2])
        self._nll_total += totals[summary_base]
        self._correct_total += int(totals[summary_base + 1])
        for index in range(STAGE_COUNT):
            self._nll_by_stage[index] += totals[index]
            self._correct_by_stage[index] += int(totals[STAGE_COUNT + index])
            self._support_by_stage[index] += int(totals[2 * STAGE_COUNT + index])
            self._forced_by_stage[index] += int(totals[3 * STAGE_COUNT + index])
        for index in range(ACTION_KIND_COUNT):
            self._support_by_kind[index] += int(totals[kind_base + index])
            self._correct_by_kind[index] += int(
                totals[kind_base + ACTION_KIND_COUNT + index]
            )

    def _flush_pending(self) -> None:
        """Apply totals held since the last completed batch, if any."""

        pending = self._pending_totals
        if pending is None:
            return
        self._pending_totals = None
        self._apply_totals(pending.cpu().tolist())

    def _add_forced_split(
        self,
        *,
        decisions: int,
        eligible: int,
        forced_by_stage: Sequence[int],
    ) -> None:
        """Record the forced rows a split's eligible-only pass skips.

        The eligible rows themselves still arrive through :meth:`add`, so this
        only adds the skipped rows and their stage counts. ``decisions`` and
        ``eligible`` come from the same validated layout as the counts.
        """

        if decisions < 0 or eligible < 0 or eligible > decisions:
            raise DataError(
                "eligible decisions must be between zero and the decision count"
            )
        if len(forced_by_stage) != STAGE_COUNT:
            raise DataError(f"forced_by_stage must contain {STAGE_COUNT} counts")
        total = 0
        for index, count in enumerate(forced_by_stage):
            if count < 0:
                raise DataError("forced_by_stage counts must be non-negative")
            self._forced_by_stage[index] += int(count)
            total += int(count)
        if total != decisions - eligible:
            raise DataError(
                "forced_by_stage must sum to decisions minus eligible decisions"
            )
        self._decisions += decisions - eligible

    def report(self) -> MetricsReport:
        self._flush_pending()
        stages = tuple(
            StageMetrics(
                stage_id=index,
                stage=STAGE_NAMES[index],
                support=support,
                forced=self._forced_by_stage[index],
                nll=(
                    self._nll_by_stage[index] / support if support > 0 else None
                ),
                top1=(
                    self._correct_by_stage[index] / support if support > 0 else None
                ),
            )
            for index, support in enumerate(self._support_by_stage)
        )
        kinds = tuple(
            ActionKindMetrics(
                kind_id=index,
                kind=ACTION_KIND_NAMES[index],
                support=support,
                top1=(
                    self._correct_by_kind[index] / support if support > 0 else None
                ),
            )
            for index, support in enumerate(self._support_by_kind)
        )
        return MetricsReport(
            decisions=self._decisions,
            eligible=self._eligible,
            forced=self._decisions - self._eligible,
            nll=(self._nll_total / self._eligible if self._eligible > 0 else None),
            top1=(
                self._correct_total / self._eligible if self._eligible > 0 else None
            ),
            stages=stages,
            action_kinds=kinds,
        )


def _batch_totals(
    nll: Tensor,
    correct: Tensor,
    eligible: Tensor,
    stages: Tensor,
    target_kinds: Tensor,
) -> Tensor:
    """Fixed-shape float64 totals for one validated batch.

    The flat layout is stage NLL, stage correct, stage support, stage forced,
    then kind support, kind correct, then batch NLL, batch correct and batch
    eligible. The stage and kind ids have already been range-checked, so one
    equality comparison against the small id axis replaces dynamic grouping.
    ``torch.where`` (rather than multiplying by a 0/1 weight) keeps a NaN that
    might reach an ineligible row from becoming ``0 * NaN``.
    """

    device = nll.device
    stage_hit = stages.to(torch.int64).unsqueeze(1) == torch.arange(
        STAGE_COUNT, device=device
    )
    kind_hit = target_kinds.to(torch.int64).unsqueeze(1) == torch.arange(
        ACTION_KIND_COUNT, device=device
    )
    eligible_2d = eligible.unsqueeze(1)
    stage_eligible = stage_hit & eligible_2d
    kind_eligible = kind_hit & eligible_2d
    zero = nll.new_zeros(())
    nll64 = nll.to(torch.float64)
    correct64 = correct.to(torch.float64)
    return torch.cat(
        (
            torch.where(stage_eligible, nll64.unsqueeze(1), zero).sum(dim=0),
            torch.where(stage_eligible, correct64.unsqueeze(1), zero).sum(dim=0),
            stage_eligible.sum(dim=0).to(torch.float64),
            (stage_hit & ~eligible_2d).sum(dim=0).to(torch.float64),
            kind_eligible.sum(dim=0).to(torch.float64),
            torch.where(kind_eligible, correct64.unsqueeze(1), zero).sum(dim=0),
            torch.stack(
                (
                    torch.where(eligible, nll64, zero).sum(),
                    torch.where(eligible, correct64, zero).sum(),
                    eligible.sum().to(torch.float64),
                )
            ),
        )
    )


def _offsets_match_cached_counts(shard: PackedDecisionDataset) -> bool:
    """Whether a shard's current offsets still imply its validated counts.

    Batching reads the current offsets, so an in-place edit can change which
    decisions are eligible after construction cached the original layout.
    """

    offsets = shard.offsets
    if offsets is None:
        # A closed shard has no offsets. Reach its documented "closed" error
        # through the full traversal instead of failing on a missing array.
        return False
    counts = shard.candidate_counts()
    current = np.diff(np.asarray(offsets, dtype=np.int64))
    return current.shape == counts.shape and np.array_equal(current, counts)


def _builtin_eligible_layout(
    dataset: DecisionDataset,
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Return eligible indices and forced counts for a built-in packed split.

    Only the exact built-in classes qualify: a ``DecisionDataset`` subclass may
    override ``eligible_indices`` or ``forced_by_stage`` with values that do
    not describe the samples it returns, so such datasets keep the full
    traversal whose forced counts come from the collated mask itself. A sharded
    split qualifies only when every shard is a packed dataset too. A built-in
    split qualifies only while its offsets still match the counts it was
    validated with, because only then does its cached layout describe the rows
    that batching will return.
    """

    if type(dataset) is PackedDecisionDataset:
        shards: Sequence[PackedDecisionDataset] = (dataset,)
    elif type(dataset) is ShardedDecisionDataset and all(
        type(shard) is PackedDecisionDataset for shard in dataset.shards
    ):
        shards = dataset.shards
    else:
        return None
    if not all(_offsets_match_cached_counts(shard) for shard in shards):
        return None
    return dataset.eligible_indices(), dataset.forced_by_stage()


def evaluate_dataset(
    model: CandidatePolicy,
    dataset: DecisionDataset,
    *,
    batch_size: int,
    device: torch.device | str,
    num_workers: int = 0,
) -> MetricsReport:
    """Evaluate every decision in one split, aggregating by decision.

    Forced decisions are included only in forced counts. Built-in packed and
    sharded splits whose current offsets still match their validated layout
    read those counts from that metadata and never infer forced rows; their
    eligible decisions alone are batched through the model. Other datasets,
    including built-in splits whose offsets changed after construction, visit
    every decision so forced counts are derived from the samples actually
    scored. The model is not modified and its previous train/eval mode is
    restored.
    """

    require_dataset_identity(dataset, "evaluation dataset")
    if batch_size < 1:
        raise DataError("batch_size must be positive")
    target_device = torch.device(device)
    accumulator = MetricAccumulator()
    builtin_layout = _builtin_eligible_layout(dataset)
    if builtin_layout is None:
        indices = np.arange(len(dataset), dtype=np.int64)
        batches = iter_batches(
            dataset,
            indices,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        indices, forced_by_stage = builtin_layout
        accumulator._add_forced_split(
            decisions=len(dataset),
            eligible=int(indices.size),
            forced_by_stage=forced_by_stage,
        )
        batches = _iter_model_batches(
            dataset,
            indices,
            batch_size=batch_size,
            num_workers=num_workers,
            training=False,
            packed_candidates=_supports_packed_candidate_scoring(model),
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                moved = batch.to(target_device)
                logits = _batch_logits(model, moved)
                accumulator.add(
                    logits,
                    moved.mask,
                    moved.targets,
                    moved.stages,
                    moved.target_kinds,
                )
    finally:
        model.train(was_training)
    return accumulator.report()
