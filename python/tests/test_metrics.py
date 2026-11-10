from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mahjong_bc.config import ModelConfig
from mahjong_bc.data import collate_decisions
from mahjong_bc.errors import DataError
from mahjong_bc.metrics import MetricAccumulator, evaluate_dataset
from mahjong_bc.policy import build_policy

from .helpers import (
    DelegatingDataset,
    assert_reports_equal,
    make_dataset,
    support_by_stage,
)


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
