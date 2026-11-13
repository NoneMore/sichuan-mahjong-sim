"""Packed scoring invariants: layout, valid-row work, gradients and consumers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import mahjong_bc.train as train_module
from mahjong_bc.config import ModelConfig, TrainingConfig
from mahjong_bc.data import (
    ShardedDecisionDataset,
    _PackedCandidateBatch,
    _iter_model_batches,
)
from mahjong_bc.errors import DataError
from mahjong_bc.metrics import evaluate_dataset
from mahjong_bc.policy import (
    CandidatePolicy,
    _batch_logits,
    _decision_nll_validated,
    build_policy,
    greedy_indices,
)
from mahjong_bc.train import TRAIN_OUTPUT_NAMES, Trainer

from .helpers import (
    assert_reports_equal, count_cuda_syncs, identity, learnable_pair, make_dataset,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]


def _batches(dataset, indices, *, batch_size, packed):
    return list(
        _iter_model_batches(
            dataset, indices, batch_size=batch_size, training=False,
            packed_candidates=packed,
        )
    )


def test_packed_gather_preserves_current_spans_order_and_metadata():
    shards = [
        make_dataset(split="validation", counts=[3, 3, 2], seed=61),
        make_dataset(split="validation", counts=[1, 2, 5], seed=62),
    ]
    dataset = ShardedDecisionDataset(
        shards, identity(split="validation", decisions=6, eligible=5)
    )
    shards[0].targets[:] = 0
    shards[0].offsets[1] = 4  # Current spans, not construction-time counts.
    indices = np.array([5, 1, 0, 4, 1, 2, 3])  # Cross-shard, repeated, tail row.
    padded = _batches(dataset, indices, batch_size=3, packed=False)
    packed = _batches(dataset, indices, batch_size=3, packed=True)
    assert [b.batch_size for b in packed] == [3, 3, 1]
    for reference, actual in zip(padded, packed, strict=True):
        assert isinstance(actual, _PackedCandidateBatch)
        assert actual.candidates.ndim == 2
        assert torch.equal(actual.candidates, reference.candidates[reference.mask])
        for name in ("states", "mask", "targets", "stages", "target_kinds"):
            assert torch.equal(getattr(actual, name), getattr(reference, name)), name


def _forward_backward(model, batch):
    batch = replace(
        batch, states=batch.states.detach().clone().requires_grad_(True),
        candidates=batch.candidates.detach().clone().requires_grad_(True),
    )
    model.zero_grad(set_to_none=True)
    logits = _batch_logits(model, batch)
    nll = _decision_nll_validated(logits, batch.mask, batch.targets)
    nll.mean().backward()
    return (
        logits.detach(), nll.detach(),
        {name: p.grad.detach().clone() for name, p in model.named_parameters()},
        batch.states.grad, batch.candidates.grad,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_only_valid_rows_are_scored_without_changing_loss_or_gradients(device):
    dataset = make_dataset(
        split="validation", counts=[1, 2, 3, 8, 20, 88], seed=67, signal=True,
    )
    indices = np.arange(len(dataset))
    padded, packed = (
        _batches(dataset, indices, batch_size=len(dataset), packed=flat)[0].to(device)
        for flat in (False, True)
    )
    model = build_policy(ModelConfig(), seed=19).to(device)
    scorer_shapes, forward_layouts = [], []
    scorer_hook = model.candidate_scorer.register_forward_pre_hook(
        lambda module, args: scorer_shapes.append(tuple(args[0].shape))
    )
    forward_hook = model.register_forward_hook(
        lambda module, args, output: forward_layouts.append(args[1].ndim)
    )
    try:
        expected = _forward_backward(model, padded)
        actual = _forward_backward(model, packed)
    finally:
        scorer_hook.remove()
        forward_hook.remove()
    # Both layouts use normal Module calls, but packed MLP work excludes padding.
    assert forward_layouts == [3, 2]
    assert scorer_shapes[-1] == (
        int(packed.mask.sum()),
        model.config.state_hidden[-1] + model.config.candidate_width,
    )
    assert scorer_shapes[-1][0] < padded.mask.numel()
    torch.testing.assert_close(
        actual[0][packed.mask], expected[0][padded.mask], atol=1e-5, rtol=1e-5
    )
    assert torch.count_nonzero(actual[0][~packed.mask]) == 0
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-5)
    for name, gradient in expected[2].items():
        torch.testing.assert_close(actual[2][name], gradient, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual[3], expected[3], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        actual[4], expected[4][padded.mask], atol=1e-5, rtol=1e-4
    )
    # Restored candidate positions retain canonical-first ties.
    with torch.no_grad():
        model.candidate_scorer[-1].weight.zero_()
        model.candidate_scorer[-1].bias.zero_()
    logits = _batch_logits(model, packed)
    assert torch.equal(
        greedy_indices(logits, packed.mask),
        torch.zeros(len(dataset), device=device, dtype=torch.int64),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_packed_state_gradient_avoids_duplicate_index_accumulation():
    dataset = make_dataset(counts=[2, 3, 8, 20, 88], seed=67, signal=True)
    batch = _batches(
        dataset, dataset.eligible_indices(), batch_size=5, packed=True
    )[0].to("cuda")
    model = build_policy(ModelConfig(), seed=23).cuda()
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(False)
    try:
        gradients = []
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            logits = _batch_logits(model, batch)
            _decision_nll_validated(logits, batch.mask, batch.targets).mean().backward()
            gradients.append(
                torch.cat([
                    p.grad.detach().flatten() for p in model.state_encoder.parameters()
                ]).cpu()
            )
        assert all(torch.equal(gradients[0], g) for g in gradients[1:])
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_packed_evaluation_preserves_decision_metrics_and_padded_fallback():
    class PaddedPolicy(CandidatePolicy):
        pass

    dataset = make_dataset(
        split="validation", counts=[1, 3, 2, 1, 4], seed=63, signal=True,
    )
    model = build_policy(
        ModelConfig(state_hidden=(16,), scorer_hidden=(16,)), seed=17
    )
    reference = PaddedPolicy(model.config)
    reference.load_state_dict(model.state_dict())
    packed = evaluate_dataset(model, dataset, batch_size=3, device="cpu")
    padded = evaluate_dataset(reference, dataset, batch_size=3, device="cpu")
    assert_reports_equal(packed, padded, tolerance=1e-5)
    assert (packed.decisions, packed.eligible, packed.forced) == (5, 3, 2)
    assert model.training and reference.training


def test_packed_training_rejects_nonfinite_output_before_any_update(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair()

    def build(config=None, *, seed=None):
        model = build_policy(config, seed=seed)

        def nonfinite(module, args, output):
            assert args[1].ndim == 2
            return torch.full_like(output, float("nan"))

        model.register_forward_hook(nonfinite)
        return model

    monkeypatch.setattr(train_module, "build_policy", build)
    trainer = Trainer(
        train_dataset=train, validation_dataset=validation,
        model_config=ModelConfig(state_hidden=(16,), scorer_hidden=(16,)),
        training_config=TrainingConfig(
            seed=7, epochs=1, batch_size=3, learning_rate=0.01
        ),
        output_dir=tmp_path / "nonfinite", device="cpu",
    )
    with pytest.raises(DataError, match="finite"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state
    assert not any((trainer.output_dir / name).exists() for name in TRAIN_OUTPUT_NAMES)


def test_packed_candidate_positions_describe_the_mask_layout():
    dataset = make_dataset(counts=[1, 2, 3, 8, 1], seed=71)
    batches = _batches(
        dataset, np.arange(len(dataset)), batch_size=2, packed=True
    )
    assert len(batches) == 3
    for batch in batches:
        positions = batch.candidate_positions
        assert positions is not None
        assert positions.dtype == torch.int64
        assert positions.shape == (batch.candidates.shape[0],)
        expected = batch.mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        assert torch.equal(positions, expected)
        # Strictly increasing means every gather/scatter index is unique.
        assert torch.all(positions[1:] > positions[:-1])
        assert int(positions.max()) < batch.mask.numel()
        moved = batch.to("cpu")
        assert isinstance(moved, _PackedCandidateBatch)
        assert torch.equal(moved.candidate_positions, positions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_internal_packed_scoring_does_not_synchronize_cuda():
    dataset = make_dataset(counts=[1, 3, 2, 5], seed=73, signal=True)
    batch = _batches(
        dataset, np.arange(len(dataset)), batch_size=4, packed=True
    )[0].to("cuda")
    model = build_policy(
        ModelConfig(state_hidden=(16,), scorer_hidden=(16,)), seed=19
    ).cuda()
    _forward_backward(model, batch)
    assert count_cuda_syncs(lambda: _forward_backward(model, batch)) == 0


def test_packed_forward_without_positions_keeps_the_mask_fallback():
    dataset = make_dataset(counts=[1, 3, 2], seed=75, signal=True)
    batch = _batches(
        dataset, np.arange(len(dataset)), batch_size=3, packed=True
    )[0]
    model = build_policy(
        ModelConfig(state_hidden=(16,), scorer_hidden=(16,)), seed=21
    )
    direct = model(batch.states, batch.candidates, batch.mask, packed=True)
    mapped = model(
        batch.states,
        batch.candidates,
        batch.mask,
        packed=True,
        candidate_positions=batch.candidate_positions,
    )
    assert torch.equal(direct, mapped)


@pytest.mark.parametrize(
    "make_positions, message",
    [
        (
            lambda batch: torch.zeros(
                batch.candidates.shape[0], dtype=torch.int32
            ),
            "int64",
        ),
        (lambda batch: [], "int64"),
        (lambda batch: torch.zeros(1, dtype=torch.int64), "shape"),
    ],
)
def test_packed_scoring_rejects_malformed_candidate_positions(
    make_positions, message
):
    dataset = make_dataset(counts=[2, 3], seed=77)
    batch = _batches(dataset, np.arange(len(dataset)), batch_size=2, packed=True)[0]
    malformed = replace(batch, candidate_positions=make_positions(batch))
    model = build_policy(ModelConfig(), seed=23)
    with pytest.raises(DataError, match=message):
        _batch_logits(model, malformed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_packed_scoring_rejects_candidate_positions_on_another_device():
    dataset = make_dataset(counts=[2, 3], seed=79)
    batch = _batches(dataset, np.arange(len(dataset)), batch_size=2, packed=True)[0]
    malformed = replace(
        batch, candidate_positions=batch.candidate_positions.cuda()
    )
    model = build_policy(ModelConfig(), seed=25)
    with pytest.raises(DataError, match="same device"):
        _batch_logits(model, malformed)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_gradients_match_padded_path_and_repeat_exactly(device):
    dataset = make_dataset(counts=[1, 2, 3, 8, 2], seed=81, signal=True)
    indices = np.arange(len(dataset))
    padded, packed = (
        _batches(dataset, indices, batch_size=5, packed=flat)[0].to(device)
        for flat in (False, True)
    )
    model = build_policy(
        ModelConfig(state_hidden=(16, 8), scorer_hidden=(12,)), seed=29
    ).to(device)
    first = _forward_backward(model, packed)
    second = _forward_backward(model, packed)
    expected = _forward_backward(model, padded)
    # The fixed-shape backward repeats bitwise; nothing is atomically ordered.
    assert torch.equal(first[0], second[0])
    for name, gradient in first[2].items():
        assert torch.equal(gradient, second[2][name]), name
    assert torch.equal(first[3], second[3])
    assert torch.equal(first[4], second[4])
    # Valid scores, parameters and inputs match the padded model computation.
    torch.testing.assert_close(
        first[0][packed.mask], expected[0][padded.mask], atol=1e-5, rtol=1e-5
    )
    for name, gradient in expected[2].items():
        torch.testing.assert_close(
            first[2][name], gradient, atol=1e-5, rtol=1e-4
        )
    torch.testing.assert_close(first[3], expected[3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        first[4], expected[4][padded.mask], atol=1e-6, rtol=1e-6
    )


def test_packed_scoring_ignores_hostile_padding_in_the_padded_reference():
    dataset = make_dataset(counts=[2, 1, 4, 3], seed=83, signal=True)
    padded, packed = (
        _batches(dataset, np.arange(len(dataset)), batch_size=4, packed=flat)[0]
        for flat in (False, True)
    )
    padded = replace(padded, candidates=padded.candidates.clone())
    padded.candidates[~padded.mask] = float("nan")
    model = build_policy(
        ModelConfig(state_hidden=(16,), scorer_hidden=(16,)), seed=31
    )
    expected = _forward_backward(model, padded)
    actual = _forward_backward(model, packed)
    torch.testing.assert_close(
        actual[0][packed.mask], expected[0][padded.mask], atol=1e-5, rtol=1e-5
    )
    assert torch.count_nonzero(actual[0][~packed.mask]) == 0
    torch.testing.assert_close(
        actual[4], expected[4][padded.mask], atol=1e-6, rtol=1e-6
    )
