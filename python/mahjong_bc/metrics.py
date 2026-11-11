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
    iter_batches,
    require_dataset_identity,
)
from .errors import DataError
from .policy import CandidatePolicy, decision_nll, greedy_indices


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

    def add(
        self,
        logits: Tensor,
        mask: Tensor,
        targets: Tensor,
        stages: Tensor,
        target_kinds: Tensor,
    ) -> None:
        """Add one padded batch containing all decisions, forced included."""

        if logits.dim() != 2:
            raise DataError("logits must have shape [B, C]")
        batch_size = int(logits.shape[0])
        if batch_size == 0:
            return
        if mask.shape != logits.shape:
            raise DataError("mask and logits shapes differ")
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
        if not bool(mask.any(dim=1).all()):
            raise DataError("every decision must have at least one candidate")
        if int(stages.min()) < 0 or int(stages.max()) >= STAGE_COUNT:
            raise DataError(f"stages must be in [0, {STAGE_COUNT})")
        if int(target_kinds.min()) < 0 or int(target_kinds.max()) >= ACTION_KIND_COUNT:
            raise DataError(f"target_kinds must be in [0, {ACTION_KIND_COUNT})")

        eligible = mask.sum(dim=1) > 1
        nll = decision_nll(logits, mask, targets)
        correct = greedy_indices(logits, mask).eq(targets.long())

        self._decisions += batch_size
        eligible_count = int(eligible.sum())
        self._eligible += eligible_count
        if eligible_count == 0:
            forced_stages = stages.to(torch.int64)
            for stage, count in enumerate(
                torch.bincount(forced_stages, minlength=STAGE_COUNT).cpu().tolist()
            ):
                self._forced_by_stage[stage] += int(count)
            return

        eligible_stages = stages[eligible].to(torch.int64)
        eligible_kinds = target_kinds[eligible].to(torch.int64)
        eligible_nll = nll[eligible].to(torch.float64)
        eligible_correct = correct[eligible].to(torch.float64)
        forced_stages = stages[~eligible].to(torch.int64)

        self._nll_total += float(eligible_nll.sum().item())
        self._correct_total += int(eligible_correct.sum().item())
        stage_nll = torch.bincount(
            eligible_stages, weights=eligible_nll, minlength=STAGE_COUNT
        ).cpu().tolist()
        stage_correct = torch.bincount(
            eligible_stages, weights=eligible_correct, minlength=STAGE_COUNT
        ).cpu().tolist()
        stage_support = torch.bincount(
            eligible_stages, minlength=STAGE_COUNT
        ).cpu().tolist()
        stage_forced = torch.bincount(
            forced_stages, minlength=STAGE_COUNT
        ).cpu().tolist()
        kind_support = torch.bincount(
            eligible_kinds, minlength=ACTION_KIND_COUNT
        ).cpu().tolist()
        kind_correct = torch.bincount(
            eligible_kinds, weights=eligible_correct, minlength=ACTION_KIND_COUNT
        ).cpu().tolist()
        for index in range(STAGE_COUNT):
            self._support_by_stage[index] += int(stage_support[index])
            self._nll_by_stage[index] += float(stage_nll[index])
            self._correct_by_stage[index] += int(stage_correct[index])
            self._forced_by_stage[index] += int(stage_forced[index])
        for index in range(ACTION_KIND_COUNT):
            self._support_by_kind[index] += int(kind_support[index])
            self._correct_by_kind[index] += int(kind_correct[index])

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


def _builtin_eligible_layout(
    dataset: DecisionDataset,
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Return eligible indices and forced counts for a built-in packed split.

    Only the exact built-in classes qualify: a ``DecisionDataset`` subclass may
    override ``eligible_indices`` or ``forced_by_stage`` with values that do
    not describe the samples it returns, so such datasets keep the full
    traversal whose forced counts come from the collated mask itself. A sharded
    split qualifies only when every shard is a packed dataset too.
    """

    if type(dataset) is PackedDecisionDataset:
        return dataset.eligible_indices(), dataset.forced_by_stage()
    if type(dataset) is ShardedDecisionDataset and all(
        type(shard) is PackedDecisionDataset for shard in dataset.shards
    ):
        return dataset.eligible_indices(), dataset.forced_by_stage()
    return None


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
    sharded splits read those counts from their validated layout metadata and
    never infer forced rows; their eligible decisions alone are batched
    through the model. Other datasets still visit every decision so forced
    counts are derived from the samples actually scored. The model is not
    modified and its previous train/eval mode is restored.
    """

    require_dataset_identity(dataset, "evaluation dataset")
    if batch_size < 1:
        raise DataError("batch_size must be positive")
    target_device = torch.device(device)
    accumulator = MetricAccumulator()
    builtin_layout = _builtin_eligible_layout(dataset)
    if builtin_layout is None:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices, forced_by_stage = builtin_layout
        accumulator._add_forced_split(
            decisions=len(dataset),
            eligible=int(indices.size),
            forced_by_stage=forced_by_stage,
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in iter_batches(
                dataset,
                indices,
                batch_size=batch_size,
                num_workers=num_workers,
            ):
                moved = batch.to(target_device)
                logits = model(moved.states, moved.candidates, moved.mask)
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
