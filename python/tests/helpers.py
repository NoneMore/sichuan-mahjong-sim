"""Synthetic packed-dataset helpers shared by the CPU test suite."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from mahjong_bc.config import DatasetIdentity
from mahjong_bc.constants import (
    ACTION_KIND_WIDTH,
    CANDIDATE_WIDTH,
    STATE_WIDTH,
)
from mahjong_bc.data import DecisionDataset, PackedDecisionDataset
from mahjong_bc.metrics import MetricsReport


def identity(
    *,
    split: str,
    decisions: int,
    eligible: int,
    cache_id: str = "cache-fixture",
    preparation_format_version: int = 1,
    source_revision: str = "fixture-revision",
) -> DatasetIdentity:
    return DatasetIdentity(
        cache_id=cache_id,
        split=split,
        encoding_version=1,
        preparation_format_version=preparation_format_version,
        source_revision=source_revision,
        decisions=decisions,
        eligible_decisions=eligible,
    )


def make_dataset(
    *,
    split: str = "train",
    decisions: int = 8,
    counts: Sequence[int] | None = None,
    seed: int = 0,
    cache_id: str = "cache-fixture",
    source_revision: str = "fixture-revision",
    preparation_format_version: int = 1,
    stages: Sequence[int] | None = None,
    target_kinds: Sequence[int] | None = None,
    signal: bool = False,
) -> PackedDecisionDataset:
    """Build a valid packed split with optional learnable candidate signal."""

    rng = np.random.default_rng(seed)
    if counts is None:
        counts = [int(value) for value in rng.integers(1, 4, size=decisions)]
        counts = [max(1, int(value)) for value in counts]
    else:
        counts = [int(value) for value in counts]
        decisions = len(counts)
    if decisions == 0:
        raise ValueError("fixture must contain at least one decision")
    if any(count < 1 for count in counts):
        raise ValueError("fixture candidate counts must be positive")

    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    ).astype(np.int64)
    candidate_count = int(offsets[-1])
    states = rng.normal(size=(decisions, STATE_WIDTH)).astype(np.float32)
    candidate_features = np.zeros(
        (candidate_count, CANDIDATE_WIDTH), dtype=np.float32
    )
    targets = np.empty(decisions, dtype=np.int64)
    stage_values = np.empty(decisions, dtype=np.int64)
    kind_values = np.empty(decisions, dtype=np.int64)

    for decision in range(decisions):
        start = int(offsets[decision])
        count = counts[decision]
        targets[decision] = int(rng.integers(0, count))
        stage_values[decision] = (
            int(stages[decision]) if stages is not None else int(rng.integers(0, 5))
        )
        chosen_kind = (
            int(target_kinds[decision])
            if target_kinds is not None
            else int(rng.integers(0, ACTION_KIND_WIDTH))
        )
        kind_values[decision] = chosen_kind
        for local in range(count):
            row = candidate_features[start + local]
            row[chosen_kind if local == targets[decision] else 2] = 1.0
            if signal:
                row[ACTION_KIND_WIDTH] = 1.0 if local == targets[decision] else 0.0
                row[ACTION_KIND_WIDTH + 1] = float(local)

    eligible = int(np.count_nonzero(np.asarray(counts) > 1))
    return PackedDecisionDataset(
        states=states,
        candidate_features=candidate_features,
        offsets=offsets,
        targets=targets,
        stages=stage_values,
        identity=identity(
            split=split,
            decisions=decisions,
            eligible=eligible,
            cache_id=cache_id,
            preparation_format_version=preparation_format_version,
            source_revision=source_revision,
        ),
        target_kinds=kind_values,
    )


def learnable_pair(
    *,
    train_decisions: int = 12,
    validation_decisions: int = 6,
    cache_id: str = "learnable-fixture",
) -> tuple[PackedDecisionDataset, PackedDecisionDataset]:
    """Two-candidate decisions where one feature identifies the target."""

    train = make_dataset(
        split="train",
        decisions=train_decisions,
        counts=[2] * train_decisions,
        seed=11,
        cache_id=cache_id,
        signal=True,
    )
    validation = make_dataset(
        split="validation",
        decisions=validation_decisions,
        counts=[2] * validation_decisions,
        seed=12,
        cache_id=cache_id,
        signal=True,
    )
    return train, validation


class DelegatingDataset(DecisionDataset):
    """Third-party-style wrapper used to exercise public dataset boundaries."""

    def __init__(
        self,
        inner: DecisionDataset,
        identity: object | None = None,
    ) -> None:
        self.inner = inner
        self.identity = inner.identity if identity is None else identity

    @property
    def is_memory_mapped(self) -> bool:
        return self.inner.is_memory_mapped

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        return self.inner[index]

    def eligible_indices(self) -> np.ndarray:
        return self.inner.eligible_indices()

    def forced_by_stage(self) -> tuple[int, ...]:
        return self.inner.forced_by_stage()

    def candidate_counts(self) -> np.ndarray:
        return self.inner.candidate_counts()


def support_by_stage(report: MetricsReport) -> tuple[int, ...]:
    return tuple(metric.support for metric in report.stages)


def count_host_reads(action: Callable[[], Any]) -> Counter[str]:
    """Count the tensor operations that read values back to the host.

    ``cpu``, ``item``, ``tolist``, ``__bool__`` and ``__int__`` each wait for
    the device to finish, so their per-batch count exposes redundant
    synchronization. Device-to-device torch calls are not counted.
    """

    seen: list[str] = []

    class _Recorder(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    with _Recorder():
        action()
    return Counter(seen)


def host_read_count(counts: Counter[str]) -> int:
    return sum(counts[name] for name in _HOST_READ_NAMES)


_HOST_READ_NAMES = ("cpu", "item", "tolist", "__bool__", "__int__")


def assert_reports_equal(
    left: MetricsReport,
    right: MetricsReport,
    *,
    tolerance: float = 0.0,
) -> None:
    """Compare serialized reports so newly added fields are not skipped."""

    _assert_metric_value(left.to_dict(), right.to_dict(), tolerance=tolerance)


def _assert_metric_value(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, expected_value in expected.items():
            _assert_metric_value(
                actual[key], expected_value, tolerance=tolerance
            )
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_metric_value(
                actual_item, expected_item, tolerance=tolerance
            )
    elif expected is None:
        assert actual is None
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=tolerance)
    else:
        assert actual == expected
