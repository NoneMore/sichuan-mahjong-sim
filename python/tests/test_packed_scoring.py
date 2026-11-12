"""Flat valid-candidate scoring: equivalence, fallbacks and packed training.

Packed scoring changes the scorer GEMM row layout, so the packed and padded
paths are compared with tolerances frozen before implementation rather than for
bitwise identity. Everything that is a structural property of the batch
(mask, counts, membership, targets, stages, kinds) is compared exactly.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import pytest
import torch

import mahjong_bc.metrics as metrics_module
import mahjong_bc.train as train_module
from mahjong_bc.config import ModelConfig, TrainingConfig
from mahjong_bc.checkpoint import load_checkpoint
from mahjong_bc.constants import CANDIDATE_WIDTH
from mahjong_bc.data import (
    DecisionBatch,
    PackedDecisionDataset,
    _iter_model_batches,
    _PackedCandidateBatch,
    _PrunedDecisionBatch,
)
from mahjong_bc.errors import DataError
from mahjong_bc.metrics import evaluate_dataset
from mahjong_bc.policy import (
    CandidatePolicy,
    _batch_logits,
    _decision_nll_validated,
    _supports_packed_candidate_scoring,
    build_policy,
    greedy_indices,
)
from mahjong_bc.train import TRAIN_OUTPUT_NAMES, Trainer

from .helpers import assert_reports_equal, learnable_pair, make_dataset

#: Frozen before implementation in the candidate record.
LOGIT_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
NLL_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
GRADIENT_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
#: Paired short-run metric guard from the same record.
TRAIN_METRIC_TOLERANCE = 1e-3

DEVICES = (
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
)


def _model_config() -> ModelConfig:
    return ModelConfig(state_hidden=(16,), scorer_hidden=(16,))


def _training_config(**overrides: Any) -> TrainingConfig:
    values = dict(
        seed=7,
        epochs=2,
        batch_size=3,
        learning_rate=0.05,
        num_workers=0,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _policy(device: str = "cpu", *, seed: int = 17) -> CandidatePolicy:
    return build_policy(_model_config(), seed=seed).to(device)


def _ragged() -> PackedDecisionDataset:
    return make_dataset(
        split="train", counts=[4, 2, 3, 5, 2, 3], seed=61, cache_id="packed-scoring"
    )


def _pair(
    dataset: PackedDecisionDataset,
    indices: np.ndarray,
    batch_size: int,
    *,
    training: bool = True,
    device: str = "cpu",
) -> tuple[_PrunedDecisionBatch | DecisionBatch, _PackedCandidateBatch]:
    def one(**kwargs: Any):
        return next(
            _iter_model_batches(
                dataset,
                indices,
                batch_size=batch_size,
                training=training,
                **kwargs,
            )
        )

    padded = one().to(device)
    packed = one(packed_candidates=True).to(device)
    assert isinstance(packed, _PackedCandidateBatch)
    return padded, packed


def _forward_backward(model: CandidatePolicy, batch: Any) -> dict[str, Any]:
    """Score one internal batch and collect logits, per-row NLL and gradients."""

    model.zero_grad(set_to_none=True)
    batch.states.requires_grad_(True)
    batch.candidates.requires_grad_(True)
    logits = _batch_logits(model, batch)
    nll = _decision_nll_validated(logits, batch.mask, batch.targets)
    nll.mean().backward()
    return {
        "logits": logits.detach(),
        "nll": nll.detach(),
        "mean": nll.detach().mean(),
        "parameters": {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        },
        "states": batch.states.grad.detach().clone(),
        "candidates": batch.candidates.grad.detach().clone(),
    }


@pytest.mark.parametrize("device", DEVICES)
def test_packed_logits_loss_and_gradients_match_the_padded_reference(device):
    dataset = _ragged()
    model = _policy(device)
    padded, packed = _pair(
        dataset, dataset.eligible_indices(), len(dataset), device=device
    )
    _assert_packed_matches_padded(model, padded, packed)


@pytest.mark.parametrize("device", DEVICES)
def test_default_model_long_tail_packed_numerics_match_the_padded_reference(device):
    """The default 256/128 scorer keeps the frozen tolerances on long tails."""

    long_tail = [2, 3, 8, 20, 88]
    counts = (long_tail * 7)[:32]
    dataset = make_dataset(
        split="train", counts=counts, seed=67, cache_id="packed-default-model"
    )
    model_config = ModelConfig()
    assert model_config.state_hidden == (256,) and model_config.scorer_hidden == (128,)
    model = build_policy(model_config, seed=19).to(device)
    padded, packed = _pair(
        dataset, dataset.eligible_indices(), len(dataset), device=device
    )
    assert padded.states.shape[0] == 32
    assert int(packed.candidate_counts.min()) == 2
    assert int(packed.candidate_counts.max()) == 88
    _assert_packed_matches_padded(model, padded, packed)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_backward_is_bitwise_repeatable_without_global_determinism(device):
    """The packed state reduction itself is stable, not global deterministic mode."""

    counts = ([2, 3, 8, 20, 88] * 7)[:32]
    dataset = make_dataset(
        split="train", counts=counts, seed=67, cache_id=f"packed-repeat-{device}"
    )
    model_config = ModelConfig()
    assert model_config.state_hidden == (256,) and model_config.scorer_hidden == (128,)
    model = build_policy(model_config, seed=23).to(device)
    _, packed = _pair(
        dataset, dataset.eligible_indices(), len(dataset), device=device
    )
    assert int(packed.candidate_counts.max()) == 88
    states = packed.states.detach().clone().requires_grad_(True)
    candidates = packed.candidates.detach().clone().requires_grad_(True)
    batch = _PackedCandidateBatch(
        states=states,
        candidates=candidates,
        candidate_counts=packed.candidate_counts,
        mask=packed.mask,
        targets=packed.targets,
    )
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        assert not torch.are_deterministic_algorithms_enabled()
        snapshots = []
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            logits = _batch_logits(model, batch)
            _decision_nll_validated(logits, batch.mask, batch.targets).mean().backward()
            snapshots.append(
                (
                    {
                        name: parameter.grad.detach().cpu().clone()
                        for name, parameter in model.named_parameters()
                    },
                    states.grad.detach().cpu().clone(),
                    candidates.grad.detach().cpu().clone(),
                )
            )
            states.grad = None
            candidates.grad = None
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)

    first = snapshots[0]
    for repeated in snapshots[1:]:
        for name, gradient in first[0].items():
            assert torch.equal(repeated[0][name], gradient), name
        assert torch.equal(repeated[1], first[1])
        assert torch.equal(repeated[2], first[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_packed_training_repeats_and_resumes_bitwise(tmp_path, monkeypatch):
    """Fixed seeds repeat and resume exactly on CUDA without deterministic mode."""

    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        assert not torch.are_deterministic_algorithms_enabled()
        train, validation = learnable_pair(
            train_decisions=12,
            validation_decisions=6,
            cache_id="cuda-packed-replay",
        )
        config = _training_config(epochs=2, batch_size=3, learning_rate=0.01)
        real_batches = train_module._iter_model_batches

        def run(name, *, interrupt=False, resume_from=None):
            order: list[list[int]] = []

            def traced(dataset, indices, **kwargs):
                assert kwargs.get("packed_candidates") is True
                order.append([int(index) for index in indices])
                for number, batch in enumerate(
                    real_batches(dataset, indices, **kwargs)
                ):
                    assert isinstance(batch, _PackedCandidateBatch)
                    yield batch
                    if interrupt and len(order) == 2 and number == 0:
                        raise KeyboardInterrupt("injected mid-epoch stop")

            with monkeypatch.context() as patch:
                patch.setattr(train_module, "_iter_model_batches", traced)
                trainer = Trainer(
                    train_dataset=train,
                    validation_dataset=validation,
                    model_config=_model_config(),
                    training_config=config,
                    output_dir=tmp_path / name,
                    device="cuda",
                )
                summary = trainer.train(resume_from=resume_from)
            assert trainer.model is not None
            weights = {
                key: value.detach().clone()
                for key, value in trainer.model.state_dict().items()
            }
            return summary, weights, order

        expected_summary, expected_weights, expected_order = run("continuous")
        repeated_summary, repeated_weights, repeated_order = run("repeated")
        assert repeated_summary.to_dict() == expected_summary.to_dict()
        assert repeated_order == expected_order
        _assert_state_tree_exact(repeated_weights, expected_weights)

        with pytest.raises(KeyboardInterrupt, match="injected"):
            run("interrupted", interrupt=True)
        resumed_summary, resumed_weights, resumed_order = run(
            "resumed", resume_from=tmp_path / "interrupted" / "last.pt"
        )
        assert expected_order[1:] == resumed_order
        assert resumed_summary.to_dict() == expected_summary.to_dict()
        _assert_state_tree_exact(resumed_weights, expected_weights)
        assert (tmp_path / "resumed" / "last.pt").is_file()
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


def _assert_packed_matches_padded(
    model: CandidatePolicy, padded: Any, packed: _PackedCandidateBatch
) -> None:
    mask = padded.mask
    # Structural batch facts are exact, including the flat block membership.
    assert torch.equal(packed.mask, mask)
    assert torch.equal(packed.candidate_counts, mask.sum(dim=1))
    assert torch.equal(packed.targets, padded.targets)
    assert packed.states.shape == padded.states.shape
    assert torch.equal(packed.candidates, padded.candidates[mask])
    assert bool((packed.candidate_counts > 1).all())

    expected = _forward_backward(model, padded)
    actual = _forward_backward(model, packed)

    torch.testing.assert_close(
        actual["logits"][mask], expected["logits"][mask], **LOGIT_TOLERANCE
    )
    assert torch.equal(actual["logits"][~mask], torch.zeros_like(actual["logits"][~mask]))
    torch.testing.assert_close(actual["nll"], expected["nll"], **NLL_TOLERANCE)
    torch.testing.assert_close(actual["mean"], expected["mean"], **NLL_TOLERANCE)
    assert set(actual["parameters"]) == set(expected["parameters"])
    for name, expected_gradient in expected["parameters"].items():
        torch.testing.assert_close(
            actual["parameters"][name], expected_gradient, **GRADIENT_TOLERANCE
        )
    torch.testing.assert_close(
        actual["states"], expected["states"], **GRADIENT_TOLERANCE
    )
    # Padded candidate slots receive no gradient, so only valid rows are summed.
    torch.testing.assert_close(
        actual["candidates"], expected["candidates"][mask], **GRADIENT_TOLERANCE
    )
    assert torch.equal(
        greedy_indices(actual["logits"], mask), greedy_indices(expected["logits"], mask)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_packed_evaluation_batches_keep_forced_rows_and_metadata(device):
    dataset = make_dataset(
        split="validation", counts=[1, 3, 2, 1, 4], seed=63, cache_id="packed-eval"
    )
    padded, packed = _pair(
        dataset,
        np.arange(len(dataset), dtype=np.int64),
        len(dataset),
        training=False,
        device=device,
    )
    assert torch.equal(packed.stages, padded.stages)
    assert torch.equal(packed.target_kinds, padded.target_kinds)
    assert torch.equal(packed.candidate_counts, padded.mask.sum(dim=1))
    assert torch.equal(
        packed.candidate_counts,
        torch.from_numpy(np.diff(dataset.offsets).astype(np.int64)).to(device),
    )
    assert bool((packed.candidate_counts == 1).any())
    model = _policy(device)
    expected = model(padded.states, padded.candidates, padded.mask)
    actual = _batch_logits(model, packed)
    torch.testing.assert_close(
        actual[padded.mask], expected[padded.mask], **LOGIT_TOLERANCE
    )


def test_padded_batch_metadata_cannot_select_packed_scoring():
    dataset = _ragged()
    model = _policy()
    padded, _ = _pair(dataset, dataset.eligible_indices(), len(dataset))
    # A future metadata field must not silently change this batch's layout.
    object.__setattr__(padded, "candidate_counts", padded.mask.sum(dim=1))
    assert torch.equal(
        _batch_logits(model, padded),
        model(padded.states, padded.candidates, padded.mask),
    )


def test_compiled_policy_call_keeps_the_public_path(monkeypatch):
    model = _policy()
    monkeypatch.setattr(model, "_compiled_call_impl", lambda *args: None)
    assert not _supports_packed_candidate_scoring(model)


def test_packed_scoring_keeps_greedy_ties_and_duplicate_rows_exact():
    dataset = _ragged()
    features = dataset.candidate_features
    start = int(dataset.offsets[0])
    features[start + 1] = features[start]  # One decision now has an exact tie.
    model = _policy()
    padded, packed = _pair(dataset, dataset.eligible_indices(), len(dataset))
    mask = padded.mask
    packed_logits = _batch_logits(model, packed)
    reference_logits = model(padded.states, padded.candidates, padded.mask)
    # The duplicated rows keep an exact two-way tie in the flat layout too.
    assert torch.equal(packed_logits[mask][0], packed_logits[mask][1])
    assert torch.equal(reference_logits[mask][0], reference_logits[mask][1])
    assert torch.equal(
        greedy_indices(packed_logits, mask), greedy_indices(reference_logits, mask)
    )
    # A zeroed scorer output is a tie for every row in both layouts.
    with torch.no_grad():
        model.candidate_scorer[-1].weight.zero_()
        model.candidate_scorer[-1].bias.zero_()
    logits = _batch_logits(model, packed)
    assert torch.equal(logits[mask], torch.zeros_like(logits[mask]))
    # An exact all-zero tie resolves to the lowest candidate index everywhere.
    assert torch.equal(
        greedy_indices(logits, mask), torch.zeros(len(dataset), dtype=torch.int64)
    )


def test_packed_evaluation_report_matches_the_padded_reference(monkeypatch):
    _, validation = learnable_pair(cache_id="packed-report")
    model = _policy()
    packed = evaluate_dataset(model, validation, batch_size=4, device="cpu")
    with monkeypatch.context() as patch:
        patch.setattr(
            metrics_module, "_supports_packed_candidate_scoring", lambda model: False
        )
        padded = evaluate_dataset(model, validation, batch_size=4, device="cpu")
    assert_reports_equal(packed, padded, tolerance=1e-5)
    assert (packed.decisions, packed.eligible, packed.forced) == (
        padded.decisions,
        padded.eligible,
        padded.forced,
    )
    assert packed.top1 == padded.top1


def test_packed_training_matches_padded_training_within_tolerance(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(cache_id="packed-train")
    config = _training_config(epochs=2, batch_size=3, learning_rate=0.05)
    runs: dict[str, Any] = {}
    for name in ("packed", "padded"):
        trainer = Trainer(
            train_dataset=train,
            validation_dataset=validation,
            model_config=_model_config(),
            training_config=config,
            output_dir=tmp_path / name,
            device="cpu",
        )
        with monkeypatch.context() as patch:
            if name == "padded":
                for module in (train_module, metrics_module):
                    patch.setattr(
                        module,
                        "_supports_packed_candidate_scoring",
                        lambda model: False,
                    )
            summary = trainer.train()
        assert trainer.model is not None
        runs[name] = (
            summary,
            {
                key: value.clone()
                for key, value in trainer.model.state_dict().items()
            },
        )

    packed_summary, packed_weights = runs["packed"]
    padded_summary, padded_weights = runs["padded"]
    assert packed_summary.completed_epochs == padded_summary.completed_epochs
    assert packed_summary.global_step == padded_summary.global_step
    for packed_epoch, padded_epoch in zip(
        packed_summary.history, padded_summary.history
    ):
        assert abs(packed_epoch["train_nll"] - padded_epoch["train_nll"]) <= (
            TRAIN_METRIC_TOLERANCE
        )
        assert abs(
            packed_epoch["validation"]["nll"] - padded_epoch["validation"]["nll"]
        ) <= TRAIN_METRIC_TOLERANCE
    assert set(packed_weights) == set(padded_weights)
    for name, expected in padded_weights.items():
        torch.testing.assert_close(packed_weights[name], expected, **GRADIENT_TOLERANCE)


def test_internal_batches_stay_padded_unless_packing_is_requested():
    dataset = _ragged()
    indices = dataset.eligible_indices()
    (batch,) = list(
        _iter_model_batches(dataset, indices, batch_size=len(dataset), training=True)
    )
    assert isinstance(batch, _PrunedDecisionBatch)
    assert not hasattr(batch, "candidate_counts")
    candidate_columns = batch.candidates.shape[1]
    assert candidate_columns == int(batch.mask.sum(dim=1).max())
    assert batch.candidates.shape == (len(dataset), candidate_columns, CANDIDATE_WIDTH)
    model = _policy()
    # The unchanged external warmup/consumer contract still scores it directly.
    logits = model(batch.states, batch.candidates, batch.mask)
    assert torch.equal(logits, _batch_logits(model, batch))


def test_only_the_untouched_builtin_policy_uses_packed_scoring(monkeypatch):
    model = _policy()
    assert _supports_packed_candidate_scoring(model)

    class Subclass(CandidatePolicy):
        pass

    assert not _supports_packed_candidate_scoring(Subclass(model.config))

    overridden = _policy()
    overridden.forward = lambda states, candidates, mask: overridden.score(
        states, candidates, mask
    )
    assert not _supports_packed_candidate_scoring(overridden)

    scored = _policy()
    scored.score = MethodType(CandidatePolicy.score, scored)
    assert not _supports_packed_candidate_scoring(scored)

    with monkeypatch.context() as patch:
        patch.setattr(
            CandidatePolicy,
            "forward",
            lambda self, states, candidates, mask: self.score(states, candidates, mask),
        )
        assert not _supports_packed_candidate_scoring(model)

    with monkeypatch.context() as patch:
        patch.setattr(
            CandidatePolicy,
            "score",
            lambda self, states, candidates, mask: self.score(states, candidates, mask),
        )
        assert not _supports_packed_candidate_scoring(model)

    hooked = _policy()
    hooked.register_forward_hook(lambda module, args, output: output)
    assert not _supports_packed_candidate_scoring(hooked)
    pre_hooked = _policy()
    pre_hooked.register_forward_pre_hook(lambda module, args: None)
    assert not _supports_packed_candidate_scoring(pre_hooked)
    pre_hooked_with_kwargs = _policy()
    pre_hooked_with_kwargs.register_forward_pre_hook(
        lambda module, args, kwargs: None, with_kwargs=True
    )
    assert not _supports_packed_candidate_scoring(pre_hooked_with_kwargs)

    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda module, args, output: output
    )
    try:
        assert not _supports_packed_candidate_scoring(_policy())
    finally:
        handle.remove()
    assert _supports_packed_candidate_scoring(_policy())


def test_custom_models_and_hooks_keep_padded_internal_batches(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="packed-fallback")
    seen: list[Any] = []

    class HookedPolicy(CandidatePolicy):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.forward_calls = 0
            self.hook_calls = 0

        def forward(self, states, candidates, mask):
            self.forward_calls += 1
            return super().forward(states, candidates, mask)

    def build(config=None, *, seed=None):
        torch.manual_seed(seed)
        policy = HookedPolicy(config).float()
        policy.register_forward_hook(
            lambda module, args, output: setattr(
                module, "hook_calls", module.hook_calls + 1
            )
        )
        seen.append(policy)
        return policy

    monkeypatch.setattr(train_module, "build_policy", build)
    real_batches = train_module._iter_model_batches
    flags: list[Any] = []

    def recording(dataset, indices, **kwargs):
        for batch in real_batches(dataset, indices, **kwargs):
            flags.append(kwargs.get("packed_candidates"))
            assert not isinstance(batch, _PackedCandidateBatch)
            yield batch

    monkeypatch.setattr(train_module, "_iter_model_batches", recording)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=_training_config(epochs=1, batch_size=4),
        output_dir=tmp_path / "hooked",
        device="cpu",
    )
    summary = trainer.train()
    (policy,) = seen
    assert summary.completed_epochs == 1
    assert flags and set(flags) == {False}
    assert policy.forward_calls > 0
    assert policy.hook_calls == policy.forward_calls


def test_score_overrides_keep_padded_internal_batches(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="packed-score-override")
    real_batches = train_module._iter_model_batches
    flags: list[Any] = []
    calls: list[str] = []

    def recording(dataset, indices, **kwargs):
        for batch in real_batches(dataset, indices, **kwargs):
            flags.append(kwargs.get("packed_candidates"))
            assert not isinstance(batch, _PackedCandidateBatch)
            yield batch

    monkeypatch.setattr(train_module, "_iter_model_batches", recording)
    original_score = CandidatePolicy.score

    for kind in ("instance", "class"):
        calls.clear()
        built: list[CandidatePolicy] = []

        def build(config=None, *, seed=None):
            policy = build_policy(config, seed=seed)
            if kind == "instance":

                def score(self, states, candidates, mask):
                    calls.append("instance")
                    return original_score(self, states, candidates, mask)

                policy.score = MethodType(score, policy)
            built.append(policy)
            return policy

        with monkeypatch.context() as patch:
            patch.setattr(train_module, "build_policy", build)
            if kind == "class":

                def score(self, states, candidates, mask):
                    calls.append("class")
                    return original_score(self, states, candidates, mask)

                patch.setattr(CandidatePolicy, "score", score)
            trainer = Trainer(
                train_dataset=train,
                validation_dataset=validation,
                model_config=_model_config(),
                training_config=_training_config(epochs=1, batch_size=3),
                output_dir=tmp_path / kind,
                device="cpu",
            )
            summary = trainer.train()
            (policy,) = built
            assert not _supports_packed_candidate_scoring(policy)

        assert summary.completed_epochs == 1
        assert calls and set(calls) == {kind}
        assert flags and set(flags) == {False}


def test_builtin_policy_forward_hook_is_not_bypassed(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="packed-model-hook")
    seen: list[CandidatePolicy] = []

    def build(config=None, *, seed=None):
        policy = build_policy(config, seed=seed)
        policy.register_forward_hook(
            lambda module, args, output: setattr(
                module, "hook_calls", getattr(module, "hook_calls", 0) + 1
            )
        )
        seen.append(policy)
        return policy

    monkeypatch.setattr(train_module, "build_policy", build)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=_training_config(epochs=1, batch_size=2),
        output_dir=tmp_path / "hook-fallback",
        device="cpu",
    )
    trainer.train()
    (policy,) = seen
    assert type(policy) is CandidatePolicy
    assert not _supports_packed_candidate_scoring(policy)
    assert policy.hook_calls > 0


def test_packed_batches_follow_current_offsets():
    dataset = make_dataset(
        split="train", counts=[3, 3, 4], seed=64, cache_id="packed-offsets"
    )
    dataset.targets[:] = 0
    dataset.offsets[1] = 4  # counts become [4, 2, 4]
    indices = dataset.eligible_indices()
    padded, packed = _pair(dataset, indices, len(dataset))
    assert torch.equal(
        packed.candidate_counts,
        torch.from_numpy(np.diff(dataset.offsets)[indices].astype(np.int64)),
    )
    assert torch.equal(packed.candidate_counts, padded.mask.sum(dim=1))
    assert torch.equal(packed.candidates, padded.candidates[padded.mask])
    model = _policy()
    torch.testing.assert_close(
        _batch_logits(model, packed)[padded.mask],
        model(padded.states, padded.candidates, padded.mask)[padded.mask],
        **LOGIT_TOLERANCE,
    )


def test_packed_scoring_rejects_nonfinite_logits_before_updating(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(cache_id="packed-finite")
    seen: list[CandidatePolicy] = []
    flags: list[Any] = []

    def build(config=None, *, seed=None):
        policy = build_policy(config, seed=seed)
        with torch.no_grad():
            policy.candidate_scorer[-1].bias.fill_(float("inf"))
        seen.append(policy)
        return policy

    monkeypatch.setattr(train_module, "build_policy", build)
    real_batches = train_module._iter_model_batches

    def recording(dataset, indices, **kwargs):
        flags.append(kwargs.get("packed_candidates"))
        yield from real_batches(dataset, indices, **kwargs)

    monkeypatch.setattr(train_module, "_iter_model_batches", recording)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=_training_config(epochs=1, batch_size=3),
        output_dir=tmp_path / "nonfinite-packed",
        device="cpu",
    )
    with pytest.raises(DataError, match="finite"):
        trainer.train()
    (policy,) = seen
    assert type(policy) is CandidatePolicy
    assert flags and set(flags) == {True}
    assert trainer.optimizer is not None and not trainer.optimizer.state
    for name in TRAIN_OUTPUT_NAMES:
        assert not (tmp_path / "nonfinite-packed" / name).exists()


def test_packed_training_replays_exactly_from_the_epoch_boundary(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(
        train_decisions=12, validation_decisions=6, cache_id="packed-replay"
    )
    config = _training_config(epochs=4, batch_size=3, learning_rate=0.01)
    real_batches = train_module._iter_model_batches
    continuous_order: list[list[int]] = []
    resumed_order: list[list[int]] = []

    def traced(trace, *, interrupt=False):
        def iterator(dataset, indices, **kwargs):
            assert kwargs.get("packed_candidates") is True
            trace.append([int(index) for index in indices])
            for number, batch in enumerate(real_batches(dataset, indices, **kwargs)):
                assert isinstance(batch, _PackedCandidateBatch)
                yield batch
                if interrupt and len(trace) == 2 and number == 0:
                    raise KeyboardInterrupt("injected packed mid-epoch stop")

        return iterator

    continuous = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=config,
        output_dir=tmp_path / "continuous",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "_iter_model_batches", traced(continuous_order)
    )
    expected = continuous.train()

    interrupted = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=config,
        output_dir=tmp_path / "interrupted",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "_iter_model_batches", traced([], interrupt=True)
    )
    with pytest.raises(KeyboardInterrupt, match="injected"):
        interrupted.train()

    resumed = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_model_config(),
        training_config=config,
        output_dir=tmp_path / "resumed",
        device="cpu",
    )
    monkeypatch.setattr(train_module, "_iter_model_batches", traced(resumed_order))
    actual = resumed.train(resume_from=tmp_path / "interrupted" / "last.pt")
    assert continuous_order[1:] == resumed_order
    assert expected.to_dict() == actual.to_dict()
    assert continuous.model is not None and resumed.model is not None
    for name, value in continuous.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    expected_last = load_checkpoint(
        tmp_path / "continuous" / "last.pt", require_training=True
    )
    actual_last = load_checkpoint(
        tmp_path / "resumed" / "last.pt", require_training=True
    )
    assert expected_last.metrics == actual_last.metrics
    _assert_state_tree_exact(
        expected_last.training.optimizer_state,
        actual_last.training.optimizer_state,
    )


def _assert_state_tree_exact(left: Any, right: Any, *, path: str = "") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_state_tree_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_state_tree_exact(first, second, path=f"{path}[{index}]")
        return
    if isinstance(left, float):
        assert right == left, path
        return
    assert left == right, path
