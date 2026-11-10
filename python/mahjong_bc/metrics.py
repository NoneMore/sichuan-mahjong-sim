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
    ENCODING_VERSION,
    STAGE_COUNT,
    STAGE_NAMES,
)
from .data import DecisionDataset, iter_batches
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


def evaluate_dataset(
    model: CandidatePolicy,
    dataset: DecisionDataset,
    *,
    batch_size: int,
    device: torch.device | str,
    num_workers: int = 0,
) -> MetricsReport:
    """Evaluate every decision in one split, aggregating by decision.

    Forced decisions are included only in forced counts. The model is not
    modified and its previous train/eval mode is restored.
    """

    if dataset.identity is None:
        raise DataError("evaluation dataset must expose a split identity")
    if dataset.identity.encoding_version != ENCODING_VERSION:
        raise DataError(
            f"dataset encoding version {dataset.identity.encoding_version} "
            f"does not match {ENCODING_VERSION}"
        )
    if batch_size < 1:
        raise DataError("batch_size must be positive")
    target_device = torch.device(device)
    accumulator = MetricAccumulator()
    indices = np.arange(len(dataset), dtype=np.int64)
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


def reports_equal(
    left: MetricsReport,
    right: MetricsReport,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Compare complete-split metric reports, optionally with NLL tolerance."""

    if left == right:
        return True
    if (
        left.decisions != right.decisions
        or left.eligible != right.eligible
        or left.forced != right.forced
        or len(left.stages) != len(right.stages)
        or len(left.action_kinds) != len(right.action_kinds)
    ):
        return False

    def close(first: float | None, second: float | None) -> bool:
        if first is None or second is None:
            return first is None and second is None
        return abs(first - second) <= tolerance

    if not close(left.nll, right.nll) or not close(left.top1, right.top1):
        return False
    for first_stage, second_stage in zip(left.stages, right.stages):
        if (
            first_stage.stage_id != second_stage.stage_id
            or first_stage.support != second_stage.support
            or first_stage.forced != second_stage.forced
            or not close(first_stage.nll, second_stage.nll)
            or not close(first_stage.top1, second_stage.top1)
        ):
            return False
    for first_kind, second_kind in zip(left.action_kinds, right.action_kinds):
        if (
            first_kind.kind_id != second_kind.kind_id
            or first_kind.support != second_kind.support
            or not close(first_kind.top1, second_kind.top1)
        ):
            return False
    return True


def support_by_stage(report: MetricsReport) -> Sequence[int]:
    """Convenience accessor for stage supports in encoding order."""

    return tuple(metric.support for metric in report.stages)
