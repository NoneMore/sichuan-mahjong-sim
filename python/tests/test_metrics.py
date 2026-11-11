from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mahjong_bc.config import ModelConfig
from mahjong_bc.constants import (
    ACTION_KIND_COUNT,
    CANDIDATE_WIDTH,
    STAGE_COUNT,
    STATE_WIDTH,
)
from mahjong_bc.data import (
    PackedDecisionDataset,
    ShardedDecisionDataset,
    collate_decisions,
    iter_batches,
)
from mahjong_bc.errors import DataError
from mahjong_bc.metrics import MetricAccumulator, evaluate_dataset
from mahjong_bc.policy import build_policy, decision_nll, greedy_indices

from .helpers import (
    DelegatingDataset,
    assert_reports_equal,
    count_host_reads,
    host_read_count,
    identity,
    make_dataset,
    support_by_stage,
)


class _RecordingPolicy:
    """Report how many rows actually reach a model forward pass."""

    def __init__(self, inner, *, fail_after: int | None = None) -> None:
        self.inner = inner
        self.fail_after = fail_after
        self.rows: list[int] = []

    @property
    def training(self) -> bool:
        return self.inner.training

    def eval(self):
        self.inner.eval()
        return self

    def train(self, mode: bool = True):
        self.inner.train(mode)
        return self

    def __call__(self, states, candidates, mask):
        if self.fail_after is not None and len(self.rows) >= self.fail_after:
            raise RuntimeError("injected forward failure")
        self.rows.append(int(states.shape[0]))
        return self.inner(states, candidates, mask)


class _LyingMetadataDataset(DelegatingDataset):
    """Third-party style wrapper whose layout metadata describes other samples."""

    def eligible_indices(self) -> np.ndarray:
        return np.array([0], dtype=np.int64)

    def forced_by_stage(self) -> tuple[int, ...]:
        return (9, 0, 0, 0, 0)


def _empty_dataset(*, split: str = "validation") -> PackedDecisionDataset:
    return PackedDecisionDataset(
        states=np.empty((0, STATE_WIDTH), dtype=np.float32),
        candidate_features=np.empty((0, CANDIDATE_WIDTH), dtype=np.float32),
        offsets=np.zeros(1, dtype=np.int64),
        targets=np.empty(0, dtype=np.int64),
        stages=np.empty(0, dtype=np.int64),
        identity=identity(split=split, decisions=0, eligible=0),
        target_kinds=np.empty(0, dtype=np.uint8),
    )


def _subclass_shard(source: PackedDecisionDataset) -> PackedDecisionDataset:
    """Rebuild one shard as a subclass that overrides its layout metadata."""

    class _LyingShard(PackedDecisionDataset):
        def eligible_indices(self) -> np.ndarray:
            return np.empty(0, dtype=np.int64)

        def forced_by_stage(self) -> tuple[int, ...]:
            return (7, 0, 0, 0, 0)

    return _LyingShard(
        states=source.states,
        candidate_features=source.candidate_features,
        offsets=source.offsets,
        targets=source.targets,
        stages=source.stages,
        identity=None,
        target_kinds=source.target_kinds,
    )


def _full_traversal_report(model, dataset, *, batch_size: int = 3):
    """Score every decision, including forced rows, through ``add``."""

    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in iter_batches(
            dataset, np.arange(len(dataset), dtype=np.int64), batch_size=batch_size
        ):
            accumulator.add(
                model(batch.states, batch.candidates, batch.mask),
                batch.mask,
                batch.targets,
                batch.stages,
                batch.target_kinds,
            )
    return accumulator.report()


def _manual_accumulator() -> MetricAccumulator:
    accumulator = MetricAccumulator()
    logits = torch.tensor(
        [
            [0.0, 0.0, -100.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-5.0, 5.0, -100.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])
    stages = torch.tensor([0, 0, 4, 4])
    kinds = torch.tensor([2, 3, 2, 2])
    accumulator.add(logits, mask, targets, stages, kinds)
    return accumulator


def test_accumulator_reports_primary_groups_and_forced_denominators():
    report = _manual_accumulator().report()
    assert report.decisions == 4
    assert report.eligible == 3
    assert report.forced == 1
    expected_nll = (math.log(2.0) + math.log(3.0) + math.log1p(math.exp(-10.0))) / 3.0
    assert report.nll == pytest.approx(expected_nll, abs=1e-6)
    assert report.top1 == pytest.approx(2.0 / 3.0)

    stage0, stage1, stage2, stage3, stage4 = report.stages
    assert stage0.support == 2
    assert stage0.forced == 0
    assert stage0.nll == pytest.approx((math.log(2.0) + math.log(3.0)) / 2.0, abs=1e-6)
    assert stage0.top1 == pytest.approx(0.5)
    assert stage4.support == 1
    assert stage4.forced == 1
    assert stage4.nll == pytest.approx(math.log1p(math.exp(-10.0)), abs=1e-6)
    assert stage4.top1 == 1.0
    assert stage1.support == stage2.support == stage3.support == 0
    assert stage1.nll is None and stage2.top1 is None
    assert support_by_stage(report) == (2, 0, 0, 0, 1)

    kind2, kind3 = report.action_kinds[2], report.action_kinds[3]
    assert kind2.support == 2
    assert kind2.top1 == pytest.approx(1.0)
    assert kind3.support == 1
    assert kind3.top1 == 0.0
    for index, kind in enumerate(report.action_kinds):
        if index not in (2, 3):
            assert kind.support == 0
            assert kind.top1 is None
    assert sum(kind.support for kind in report.action_kinds) == report.eligible
    assert report.to_dict()["forced"] == 1


def test_accumulator_rejects_target_on_padding():
    accumulator = MetricAccumulator()
    logits = torch.zeros(1, 3)
    mask = torch.tensor([[True, False, False]])
    with pytest.raises(DataError):
        accumulator.add(
            logits, mask, torch.tensor([2]), torch.tensor([0]), torch.tensor([0])
        )


def test_accumulator_is_additive_across_batches():
    accumulator = MetricAccumulator()
    logits = torch.tensor(
        [[0.0, 0.0], [1.0, -1.0], [-3.0, 3.0], [0.0, 0.0]]
    )
    mask = torch.tensor(
        [[True, True], [True, True], [True, True], [True, False]]
    )
    targets = torch.tensor([0, 0, 1, 0])
    stages = torch.tensor([0, 1, 2, 4])
    kinds = torch.tensor([0, 1, 2, 2])
    accumulator.add(logits[:2], mask[:2], targets[:2], stages[:2], kinds[:2])
    accumulator.add(logits[2:], mask[2:], targets[2:], stages[2:], kinds[2:])
    combined = accumulator.report()

    one_batch = MetricAccumulator()
    one_batch.add(logits, mask, targets, stages, kinds)
    assert combined == one_batch.report()


def test_accumulator_rejects_invalid_group_and_mask_inputs():
    accumulator = MetricAccumulator()
    logits = torch.zeros(2, 2)
    mask = torch.tensor([[True, True], [True, False]])
    targets = torch.tensor([0, 0])
    with pytest.raises(DataError):
        accumulator.add(logits, mask, targets, torch.tensor([0, 5]), torch.tensor([0, 1]))
    with pytest.raises(DataError):
        accumulator.add(logits, mask, targets, torch.tensor([0, 1]), torch.tensor([0, 9]))
    with pytest.raises(DataError):
        accumulator.add(
            logits,
            torch.tensor([[True, True], [False, False]]),
            targets,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )


def test_evaluate_dataset_is_complete_and_batch_size_independent():
    dataset = make_dataset(
        split="validation", counts=[2, 1, 3, 2, 1, 2, 3], seed=41
    )
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=5)
    model.train()
    reports = [
        evaluate_dataset(model, dataset, batch_size=size, device="cpu")
        for size in (1, 3, 64)
    ]
    assert model.training is True
    assert reports[0].decisions == len(dataset)
    assert reports[0].eligible == dataset.eligible_count
    assert reports[0].forced == dataset.forced_count
    assert reports[0].forced == sum(stage.forced for stage in reports[0].stages)
    assert reports[0].eligible == sum(stage.support for stage in reports[0].stages)
    assert reports[0].eligible == sum(
        kind.support for kind in reports[0].action_kinds
    )
    for other in reports[1:]:
        assert_reports_equal(reports[0], other, tolerance=1e-12)
    model.eval()
    evaluate_dataset(model, dataset, batch_size=2, device="cpu")
    assert model.training is False


def test_evaluate_dataset_uses_padded_batches_without_affecting_metrics():
    dataset = make_dataset(split="validation", counts=[1, 3, 2], seed=42)
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=6)
    single = evaluate_dataset(model, dataset, batch_size=1, device="cpu")
    samples = [dataset[index] for index in range(len(dataset))]
    batch = collate_decisions(samples)
    accumulator = MetricAccumulator()
    accumulator.add(
        model(batch.states, batch.candidates, batch.mask),
        batch.mask,
        batch.targets,
        batch.stages,
        batch.target_kinds,
    )
    assert_reports_equal(single, accumulator.report(), tolerance=1e-12)
    assert single.forced == 1
    assert single.eligible == 2


def test_evaluate_dataset_accepts_third_party_subclass_identity_boundary():
    dataset = make_dataset(split="validation", counts=[2, 1, 2], seed=44)
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=8)
    wrapped = DelegatingDataset(dataset)
    report = evaluate_dataset(model, wrapped, batch_size=2, device="cpu")
    assert report.decisions == len(dataset)

    duck = DelegatingDataset(
        dataset, identity=SimpleNamespace(**dataset.identity.to_dict())
    )
    with pytest.raises(DataError, match="DatasetIdentity"):
        evaluate_dataset(model, duck, batch_size=2, device="cpu")


def test_zero_support_groups_are_undefined():
    dataset = make_dataset(
        split="validation",
        counts=[2, 2],
        target_kinds=[0, 0],
        stages=[0, 0],
        seed=43,
    )
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=7)
    report = evaluate_dataset(model, dataset, batch_size=4, device="cpu")
    assert report.stages[0].support == 2
    for stage in report.stages[1:]:
        assert stage.support == 0
        assert stage.nll is None and stage.top1 is None
    assert report.action_kinds[0].support == 2
    for kind in report.action_kinds[1:]:
        assert kind.support == 0 and kind.top1 is None


def test_evaluate_dataset_skips_forced_rows_for_builtin_packed_split():
    dataset = make_dataset(
        split="validation", counts=[2, 1, 3, 2, 1, 2, 3, 1], seed=51
    )
    assert (dataset.eligible_count, dataset.forced_count) == (5, 3)
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=12)
    recording = _RecordingPolicy(model)
    report = evaluate_dataset(recording, dataset, batch_size=3, device="cpu")

    assert recording.rows == [3, 2]
    assert sum(recording.rows) == dataset.eligible_count
    assert (report.decisions, report.eligible, report.forced) == (8, 5, 3)
    assert tuple(metric.forced for metric in report.stages) == dataset.forced_by_stage()
    baseline = _full_traversal_report(model, dataset)
    # Forced rows carry no primary metric, so counts and top-1 stay exact;
    # eligible-only batch composition can move NLL in its last digits.
    assert report.top1 == baseline.top1
    assert support_by_stage(report) == support_by_stage(baseline)
    assert tuple(metric.forced for metric in baseline.stages) == dataset.forced_by_stage()
    assert_reports_equal(report, baseline, tolerance=1e-6)


@pytest.mark.parametrize("sharded", [False, True])
def test_evaluation_uses_current_layout_after_offsets_change(sharded):
    first = make_dataset(
        split="validation", counts=[1, 2, 3], stages=[0, 1, 2], seed=61
    )
    first.targets[:] = 0
    if sharded:
        second = make_dataset(
            split="validation", counts=[1, 2], stages=[3, 4], seed=62
        )
        dataset = ShardedDecisionDataset(
            (first, second), identity(split="validation", decisions=5, eligible=3)
        )
    else:
        dataset = first
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=21)
    recording = _RecordingPolicy(model)
    evaluate_dataset(recording, dataset, batch_size=2, device="cpu")
    assert sum(recording.rows) == dataset.eligible_count
    recording.rows.clear()

    # Eligible indices are cached, but sampling follows the current spans.
    # This swaps a forced and an eligible row without changing the total count.
    first.offsets[1] = 2
    assert np.diff(first.offsets).tolist() == [2, 1, 3]
    report = evaluate_dataset(recording, dataset, batch_size=2, device="cpu")
    reference = evaluate_dataset(
        model, DelegatingDataset(dataset), batch_size=2, device="cpu"
    )
    assert sum(recording.rows) == len(dataset)
    assert_reports_equal(report, reference)
    assert (report.eligible, report.forced) == ((3, 2) if sharded else (2, 1))
    assert support_by_stage(report) == ((1, 0, 1, 0, 1) if sharded else (1, 0, 1, 0, 0))
    assert tuple(stage.forced for stage in report.stages) == (
        (0, 1, 0, 1, 0) if sharded else (0, 1, 0, 0, 0)
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_evaluation_closed_layout_keeps_closed_error(sharded):
    first = make_dataset(split="validation", counts=[1, 2], seed=63)
    dataset = (
        ShardedDecisionDataset(
            (first,), identity(split="validation", decisions=2, eligible=1)
        )
        if sharded else first
    )
    first.close()
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=22)
    with pytest.raises(DataError, match="closed"):
        evaluate_dataset(model, dataset, batch_size=2, device="cpu")


def test_evaluate_dataset_all_forced_split_needs_no_forward():
    dataset = make_dataset(
        split="validation",
        counts=[1, 1, 1, 1],
        stages=[0, 1, 4, 4],
        seed=52,
    )
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=13)
    model.train()
    recording = _RecordingPolicy(model)
    report = evaluate_dataset(recording, dataset, batch_size=2, device="cpu")

    assert recording.rows == []
    assert model.training is True
    assert (report.decisions, report.eligible, report.forced) == (4, 0, 4)
    assert report.nll is None and report.top1 is None
    assert tuple(metric.forced for metric in report.stages) == (1, 1, 0, 0, 2)
    for stage in report.stages:
        assert stage.support == 0
        assert stage.nll is None and stage.top1 is None
    for kind in report.action_kinds:
        assert kind.support == 0 and kind.top1 is None
    assert_reports_equal(report, _full_traversal_report(model, dataset), tolerance=1e-6)


def test_evaluate_dataset_empty_split_needs_no_forward():
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=14)
    recording = _RecordingPolicy(model)
    datasets = (
        _empty_dataset(),
        ShardedDecisionDataset((), identity(split="validation", decisions=0, eligible=0)),
    )
    for dataset in datasets:
        report = evaluate_dataset(recording, dataset, batch_size=4, device="cpu")
        assert (report.decisions, report.eligible, report.forced) == (0, 0, 0)
        assert report.nll is None and report.top1 is None
        assert support_by_stage(report) == (0, 0, 0, 0, 0)
        assert all(metric.nll is None and metric.top1 is None for metric in report.stages)
        assert all(
            kind.support == 0 and kind.top1 is None for kind in report.action_kinds
        )
        assert_reports_equal(report, _full_traversal_report(model, dataset))
    assert recording.rows == []


def test_evaluate_dataset_skips_forced_rows_across_mixed_shards():
    first = make_dataset(split="validation", counts=[2, 1, 3], seed=53)
    second = make_dataset(split="validation", counts=[1, 2, 1, 2], seed=54)
    dataset = ShardedDecisionDataset(
        (first, _empty_dataset(), second),
        identity(split="validation", decisions=7, eligible=4),
    )
    assert dataset.forced_count == 3
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=15)
    recording = _RecordingPolicy(model)
    report = evaluate_dataset(recording, dataset, batch_size=3, device="cpu")

    assert recording.rows == [3, 1]
    assert (report.decisions, report.eligible, report.forced) == (7, 4, 3)
    assert tuple(metric.forced for metric in report.stages) == dataset.forced_by_stage()
    baseline = _full_traversal_report(model, dataset)
    assert report.top1 == baseline.top1
    assert tuple(metric.forced for metric in baseline.stages) == dataset.forced_by_stage()
    assert_reports_equal(report, baseline, tolerance=1e-6)


def test_evaluate_dataset_sharded_subclass_shard_keeps_full_traversal():
    first = make_dataset(split="validation", counts=[2, 1, 3], seed=55)
    second = _subclass_shard(
        make_dataset(split="validation", counts=[1, 2], seed=56)
    )
    dataset = ShardedDecisionDataset(
        (first, second),
        identity(split="validation", decisions=5, eligible=3),
    )
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=16)
    recording = _RecordingPolicy(model)
    report = evaluate_dataset(recording, dataset, batch_size=3, device="cpu")

    # The shard's layout metadata is unusable, so every decision is visited.
    assert recording.rows == [3, 2]
    assert (report.decisions, report.eligible, report.forced) == (5, 3, 2)
    baseline = _full_traversal_report(model, dataset)
    assert report.top1 == baseline.top1
    assert_reports_equal(report, baseline, tolerance=1e-6)


def test_evaluate_dataset_does_not_trust_third_party_layout_metadata():
    dataset = make_dataset(split="validation", counts=[2, 1, 3, 2], seed=57)
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=17)
    recording = _RecordingPolicy(model)
    report = evaluate_dataset(
        recording, _LyingMetadataDataset(dataset), batch_size=3, device="cpu"
    )

    assert recording.rows == [3, 1]
    assert (report.decisions, report.eligible, report.forced) == (4, 3, 1)
    assert tuple(metric.forced for metric in report.stages) == dataset.forced_by_stage()
    baseline = _full_traversal_report(model, _LyingMetadataDataset(dataset))
    assert report.top1 == baseline.top1
    assert_reports_equal(report, baseline, tolerance=1e-6)


def test_evaluate_dataset_restores_mode_when_forward_fails():
    dataset = make_dataset(split="validation", counts=[2, 2, 1, 2], seed=58)
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=18)
    recording = _RecordingPolicy(model, fail_after=1)
    model.train()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        evaluate_dataset(recording, dataset, batch_size=2, device="cpu")
    assert model.training is True
    assert recording.rows == [2]
    model.eval()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        evaluate_dataset(recording, dataset, batch_size=2, device="cpu")
    assert model.training is False


def test_accumulator_rejects_inconsistent_forced_split_counts():
    accumulator = MetricAccumulator()
    with pytest.raises(DataError):
        accumulator._add_forced_split(decisions=2, eligible=3, forced_by_stage=(0,) * 5)
    with pytest.raises(DataError):
        accumulator._add_forced_split(decisions=2, eligible=1, forced_by_stage=(1,) * 4)
    with pytest.raises(DataError):
        accumulator._add_forced_split(
            decisions=2, eligible=0, forced_by_stage=(-1, 1, 0, 0, 0)
        )
    with pytest.raises(DataError):
        accumulator._add_forced_split(decisions=2, eligible=2, forced_by_stage=(1, 0, 0, 0, 0))
    assert accumulator.report().decisions == 0


def test_accumulator_matches_per_decision_reference_groups():
    dataset = make_dataset(
        split="validation", counts=[2, 1, 3, 2, 1, 2, 3, 1, 2], seed=61
    )
    model = build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=21)
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in iter_batches(
            dataset,
            np.arange(len(dataset), dtype=np.int64),
            batch_size=4,
        ):
            accumulator.add(
                model(batch.states, batch.candidates, batch.mask),
                batch.mask,
                batch.targets,
                batch.stages,
                batch.target_kinds,
            )
    report = accumulator.report()

    # Independent reference through the public per-decision helpers.
    decisions = eligible = 0
    nll_total = 0.0
    correct_total = 0
    forced = [0] * STAGE_COUNT
    support = [0] * STAGE_COUNT
    nll_by_stage = [0.0] * STAGE_COUNT
    correct_by_stage = [0] * STAGE_COUNT
    kind_support = [0] * ACTION_KIND_COUNT
    kind_correct = [0] * ACTION_KIND_COUNT
    with torch.no_grad():
        for batch in iter_batches(
            dataset, np.arange(len(dataset), dtype=np.int64), batch_size=4
        ):
            logits = model(batch.states, batch.candidates, batch.mask)
            nll = decision_nll(logits, batch.mask, batch.targets)
            chosen = greedy_indices(logits, batch.mask).eq(batch.targets)
            for row in range(batch.batch_size):
                stage = int(batch.stages[row])
                kind = int(batch.target_kinds[row])
                decisions += 1
                if int(batch.mask[row].sum()) == 1:
                    forced[stage] += 1
                    continue
                value = float(nll[row].item())
                hit = int(chosen[row].item())
                eligible += 1
                nll_total += value
                correct_total += hit
                support[stage] += 1
                nll_by_stage[stage] += value
                correct_by_stage[stage] += hit
                kind_support[kind] += 1
                kind_correct[kind] += hit

    def assert_ratio(actual, total, total_support):
        if total_support == 0:
            assert actual is None
        else:
            assert actual == pytest.approx(total / total_support, rel=1e-12, abs=1e-12)

    assert report.decisions == decisions
    assert report.eligible == eligible
    assert report.forced == decisions - eligible
    assert_ratio(report.nll, nll_total, eligible)
    assert_ratio(report.top1, correct_total, eligible)
    for index, metric in enumerate(report.stages):
        assert metric.support == support[index]
        assert metric.forced == forced[index]
        assert_ratio(metric.nll, nll_by_stage[index], support[index])
        assert_ratio(metric.top1, correct_by_stage[index], support[index])
    for index, kind in enumerate(report.action_kinds):
        assert kind.support == kind_support[index]
        assert_ratio(kind.top1, kind_correct[index], kind_support[index])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_accumulator_rejects_nonfinite_valid_logits_before_accumulating(bad):
    accumulator = MetricAccumulator()
    logits = torch.zeros(2, 3)
    logits[1, 2] = bad
    mask = torch.tensor([[True, False, False], [True, True, True]])
    with pytest.raises(DataError, match="finite"):
        accumulator.add(
            logits,
            mask,
            torch.tensor([0, 2]),
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        )
    report = accumulator.report()
    assert (report.decisions, report.eligible, report.forced) == (0, 0, 0)
    assert report.nll is None and report.top1 is None
    assert all(stage.support == 0 and stage.forced == 0 for stage in report.stages)
    assert all(kind.support == 0 and kind.top1 is None for kind in report.action_kinds)


def test_accumulator_ignores_nonfinite_padding_and_keeps_public_messages():
    logits = torch.tensor([[0.0, 0.0, float("nan")], [1.0, 0.0, float("inf")]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    accumulator = MetricAccumulator()
    accumulator.add(
        logits, mask, torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([2, 2])
    )
    report = accumulator.report()
    assert (report.decisions, report.eligible, report.forced) == (2, 2, 0)

    padded_mask = torch.tensor([[True, False, True], [True, True, True]])
    cases = (
        (
            torch.zeros(2, 3).double(),
            mask,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            "float32",
        ),
        (
            torch.zeros(2, 3),
            mask.to(torch.int64),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            "boolean",
        ),
        (
            torch.zeros(2, 3),
            padded_mask,
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            "does not reference a valid candidate",
        ),
        (
            torch.zeros(2, 3),
            mask,
            torch.tensor([0, 3]),
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            "outside the padded candidate range",
        ),
        (
            torch.zeros(2, 3),
            mask,
            torch.tensor([0, 1]),
            torch.tensor([0, 5]),
            torch.tensor([2, 2]),
            "stages must be in",
        ),
        (
            torch.zeros(2, 3),
            mask,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([2, 9]),
            "target_kinds must be in",
        ),
        (
            torch.zeros(2, 3),
            torch.tensor([[True, True, True], [False, False, False]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            "at least one candidate",
        ),
    )
    for logits_case, mask_case, targets, stages, kinds, message in cases:
        rejected = MetricAccumulator()
        with pytest.raises(DataError, match=message):
            rejected.add(logits_case, mask_case, targets, stages, kinds)
        assert rejected.report().decisions == 0


def test_accumulator_host_transfers_per_batch_are_bounded():
    logits = torch.tensor(
        [[0.0, 0.0, -100.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
    targets = torch.tensor([0, 1, 0])
    stages = torch.tensor([0, 0, 4])
    kinds = torch.tensor([2, 3, 2])
    reads = count_host_reads(
        lambda: MetricAccumulator().add(logits, mask, targets, stages, kinds)
    )
    # One validation transfer and one aggregation transfer. Implicit device
    # reads inside masked_select, boolean indexing and bincount are not counted
    # here, so this bounds explicit scalar fetches only.
    assert reads["cpu"] == 2
    assert reads["item"] == reads["__bool__"] == reads["__int__"] == 0
    forced = count_host_reads(
        lambda: MetricAccumulator().add(
            logits[:1, :1],
            torch.tensor([[True]]),
            targets[:1],
            stages[:1],
            kinds[:1],
        )
    )
    assert forced["cpu"] == 2
