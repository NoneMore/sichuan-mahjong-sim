"""Restored AdamW slot placement across epoch-boundary recovery."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import mahjong_bc.train as train_module
from mahjong_bc.checkpoint import load_checkpoint
from mahjong_bc.config import ModelConfig, TrainingConfig
from mahjong_bc.train import (
    Trainer,
    _move_optimizer_state,
    _optimizer_state_slot_device,
)

from .helpers import learnable_pair


def _training_config(**overrides: Any) -> TrainingConfig:
    values = dict(
        seed=13,
        epochs=2,
        batch_size=3,
        learning_rate=0.02,
        num_workers=0,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _model_config() -> ModelConfig:
    return ModelConfig(state_hidden=(16,), scorer_hidden=(16,))


def _assert_tree_exact(left: Any, right: Any, *, path: str = "") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_tree_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_tree_exact(first, second, path=f"{path}[{index}]")
        return
    assert left == right, path


@pytest.mark.parametrize(
    ("group", "step_device"),
    [
        ({}, torch.device("cpu")),
        ({"capturable": True}, torch.device("meta")),
        ({"fused": True}, torch.device("meta")),
    ],
)
def test_restored_step_slot_follows_the_optimizer_step_path(group, step_device):
    parameter = torch.empty(2, device="meta")

    assert _optimizer_state_slot_device("step", parameter, group) == step_device
    for slot in ("exp_avg", "exp_avg_sq"):
        assert _optimizer_state_slot_device(slot, parameter, group) == parameter.device


def test_move_optimizer_state_keeps_non_capturable_step_on_the_host():
    parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.zeros(2),
        "exp_avg_sq": torch.zeros(2),
    }

    _move_optimizer_state(optimizer)

    state = optimizer.state[parameter]
    assert state["step"].device == torch.device("cpu")
    assert state["exp_avg"].device == parameter.device
    assert state["exp_avg_sq"].device == parameter.device


def test_move_optimizer_state_keeps_capturable_step_with_the_parameters():
    parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
    optimizer = torch.optim.AdamW([parameter], capturable=True)
    optimizer.state[parameter] = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.zeros(2),
        "exp_avg_sq": torch.zeros(2),
    }

    _move_optimizer_state(optimizer)

    state = optimizer.state[parameter]
    for slot in ("step", "exp_avg", "exp_avg_sq"):
        assert state[slot].device == parameter.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_move_optimizer_state_moves_cuda_step_back_to_the_host():
    parameter = torch.nn.Parameter(torch.zeros(2, device="cuda"))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {
        "step": torch.tensor(4.0, device="cuda"),
        "exp_avg": torch.zeros(2, device="cuda"),
        "exp_avg_sq": torch.zeros(2, device="cuda"),
    }

    _move_optimizer_state(optimizer)

    state = optimizer.state[parameter]
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device.type == "cuda"
    assert state["exp_avg_sq"].device.type == "cuda"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_resume_restores_slot_placement_and_matches_uninterrupted_continuation(
    tmp_path, monkeypatch, device
):
    train, validation = learnable_pair(
        train_decisions=9, validation_decisions=4, cache_id="optimizer-slots"
    )
    model_config = _model_config()
    config = _training_config(epochs=2)

    interrupted = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model_config,
        training_config=config,
        output_dir=tmp_path / "boundary",
        device=device,
    )
    real_batches = train_module._iter_model_batches
    epoch_calls = 0

    def stop_before_second_epoch(dataset, indices, **kwargs):
        nonlocal epoch_calls
        epoch_calls += 1
        if epoch_calls == 2:
            raise KeyboardInterrupt("injected epoch-boundary stop")
        yield from real_batches(dataset, indices, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            train_module, "_iter_model_batches", stop_before_second_epoch
        )
        with pytest.raises(KeyboardInterrupt, match="injected"):
            interrupted.train()
    assert epoch_calls == 2

    resumed = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model_config,
        training_config=config,
        output_dir=tmp_path / "resumed",
        device=device,
    )
    resumed_summary = resumed.train(resume_from=tmp_path / "boundary" / "last.pt")

    assert resumed.optimizer is not None
    for group in resumed.optimizer.param_groups:
        for parameter in group["params"]:
            state = resumed.optimizer.state[parameter]
            assert state["step"].device == torch.device("cpu")
            for slot in ("exp_avg", "exp_avg_sq"):
                assert state[slot].device == parameter.device

    uninterrupted = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model_config,
        training_config=config,
        output_dir=tmp_path / "uninterrupted",
        device=device,
    )
    uninterrupted_summary = uninterrupted.train()

    # Exact continuation also guards against nondeterministic CUDA operations
    # entering either the packed backward or validation path.
    assert resumed_summary.to_dict() == uninterrupted_summary.to_dict()
    assert resumed.model is not None and uninterrupted.model is not None
    _assert_tree_exact(
        resumed.model.state_dict(), uninterrupted.model.state_dict()
    )
    _assert_tree_exact(
        resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict()
    )

    resumed_last = load_checkpoint(
        tmp_path / "resumed" / "last.pt", require_training=True
    )
    uninterrupted_last = load_checkpoint(
        tmp_path / "uninterrupted" / "last.pt", require_training=True
    )
    _assert_tree_exact(
        resumed_last.training.optimizer_state,
        uninterrupted_last.training.optimizer_state,
    )
