from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mahjong_bc.checkpoint as checkpoint_module
from mahjong_bc.checkpoint import (
    TrainingCheckpointState,
    atomic_torch_save,
    build_payload,
    check_model_config_identity,
    load_checkpoint,
    load_policy_checkpoint,
    restore_random_state,
    save_checkpoint,
    validate_training_state_compatible,
)
from mahjong_bc.config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from mahjong_bc.constants import ROLE_BEST, ROLE_LAST, ROLE_POLICY
from mahjong_bc.errors import (
    CheckpointError,
    ConfigurationError,
    ResumeError,
)
from mahjong_bc.policy import build_policy

from .helpers import identity


def _model() -> object:
    return build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=2)


def _identities():
    return (
        identity(split="train", decisions=10, eligible=7),
        identity(split="validation", decisions=5, eligible=4),
    )


def _state(
    *,
    training_config: TrainingConfig | None = None,
    dependencies: DependencyVersions | None = None,
    device_type: str = "cpu",
    completed_epochs: int = 1,
) -> TrainingCheckpointState:
    train, validation = _identities()
    generator = torch.Generator().manual_seed(77)
    return TrainingCheckpointState(
        training_config=training_config
        or TrainingConfig(seed=3, epochs=2, batch_size=2, learning_rate=0.01),
        train_dataset_identity=train,
        validation_dataset_identity=validation,
        dependencies=dependencies or collect_dependency_versions(),
        device_type=device_type,
        completed_epochs=completed_epochs,
        global_step=4,
        samples_seen=8,
        best_validation_nll=0.75,
        best_epoch=0,
        history=[{"epoch": 0, "steps": 2, "samples": 4}],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=None,
        order_rng_state=generator.get_state(),
    )


def test_policy_checkpoint_roundtrip_and_shared_loader(tmp_path):
    path = tmp_path / "policy.pt"
    model = _model()
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model, role=ROLE_POLICY, metrics={"nll": 1.25})
    loaded = load_policy_checkpoint(path)
    assert loaded.role == ROLE_POLICY
    assert loaded.model_config == model.config
    assert loaded.training is None
    assert loaded.metrics == {"nll": 1.25}
    assert loaded.model.training is False
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, before[name])
    with pytest.raises(CheckpointError):
        load_checkpoint(path, require_training=True)


def test_training_checkpoint_roundtrip_and_resume_comparison(tmp_path):
    model = _model()
    state = _state()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, role=ROLE_LAST, training=state, metrics={"nll": 0.5})
    loaded = load_checkpoint(path, require_training=True)
    assert loaded.role == ROLE_LAST
    assert loaded.training is not None
    loaded_state = loaded.training
    assert loaded_state.training_config == state.training_config
    assert loaded_state.train_dataset_identity == state.train_dataset_identity
    assert loaded_state.validation_dataset_identity == state.validation_dataset_identity
    assert loaded_state.dependencies == state.dependencies
    assert loaded_state.device_type == "cpu"
    assert loaded_state.completed_epochs == 1
    assert loaded_state.global_step == 4
    assert loaded_state.samples_seen == 8
    assert loaded_state.best_validation_nll == 0.75
    assert loaded_state.best_epoch == 0
    assert torch.equal(loaded_state.torch_rng_state, state.torch_rng_state)
    assert torch.equal(loaded_state.order_rng_state, state.order_rng_state)
    assert loaded_state.optimizer_state["param_groups"][0]["lr"] == 0.01

    validate_training_state_compatible(
        loaded_state,
        training_config=state.training_config,
        train_dataset_identity=state.train_dataset_identity,
        validation_dataset_identity=state.validation_dataset_identity,
        dependencies=state.dependencies,
        device_type="cpu",
    )
    check_model_config_identity(loaded, model.config)
    with pytest.raises(ResumeError):
        check_model_config_identity(
            loaded, ModelConfig(state_hidden=(16,), scorer_hidden=(8,))
        )


def test_resume_comparison_rejects_every_identity_dimension(tmp_path):
    model = _model()
    state = _state()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, role=ROLE_LAST, training=state)
    loaded_state = load_checkpoint(path, require_training=True).training
    assert loaded_state is not None

    base = dict(
        training_config=state.training_config,
        train_dataset_identity=state.train_dataset_identity,
        validation_dataset_identity=state.validation_dataset_identity,
        dependencies=state.dependencies,
        device_type="cpu",
    )
    validate_training_state_compatible(loaded_state, **base)

    with pytest.raises(ResumeError):
        validate_training_state_compatible(
            loaded_state,
            **{**base, "training_config": TrainingConfig(seed=3, epochs=2, batch_size=3, learning_rate=0.01)},
        )
    with pytest.raises(ResumeError):
        validate_training_state_compatible(
            loaded_state,
            **{
                **base,
                "train_dataset_identity": identity(
                    split="train", decisions=10, eligible=7, cache_id="other"
                ),
            },
        )
    with pytest.raises(ResumeError):
        validate_training_state_compatible(
            loaded_state,
            **{
                **base,
                "validation_dataset_identity": identity(
                    split="validation", decisions=6, eligible=4
                ),
            },
        )
    with pytest.raises(ResumeError):
        validate_training_state_compatible(
            loaded_state,
            **{
                **base,
                "dependencies": DependencyVersions(
                    python="different",
                    torch="different",
                    numpy="different",
                    cuda_build=None,
                ),
            },
        )
    with pytest.raises(ResumeError):
        validate_training_state_compatible(
            loaded_state, **{**base, "device_type": "cuda"}
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("checkpoint_format_version", 2),
        ("model_format_version", 2),
        ("encoding_version", 2),
        ("role", "unknown"),
        ("kind", "other-checkpoint"),
    ],
)
def test_loader_rejects_unsupported_checkpoint_versions_and_roles(tmp_path, key, value):
    model = _model()
    payload = build_payload(model, role=ROLE_POLICY, training=None, metrics=None)
    payload[key] = value
    path = tmp_path / "bad.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_loader_rejects_unknown_keys_missing_state_and_corruption(tmp_path):
    model = _model()
    payload = build_payload(model, role=ROLE_POLICY, training=None, metrics=None)
    payload["unknown"] = 1
    path = tmp_path / "unknown.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, training=None, metrics=None)
    del payload["model_state"]["state_encoder.0.weight"]
    path = tmp_path / "missing.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(corrupt)


def test_training_roles_require_training_state(tmp_path):
    model = _model()
    with pytest.raises(ConfigurationError):
        save_checkpoint(tmp_path / "best.pt", model, role=ROLE_BEST)
    with pytest.raises(ConfigurationError):
        save_checkpoint(tmp_path / "last.pt", model, role=ROLE_LAST)


def test_atomic_save_preserves_old_checkpoint_on_write_and_replace_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, _model(), role=ROLE_POLICY)
    original = path.read_bytes()

    def failing_save(payload, handle):
        handle.write(b"partial checkpoint")
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", failing_save)
    with pytest.raises(RuntimeError):
        save_checkpoint(path, _model(), role=ROLE_POLICY)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
    assert load_policy_checkpoint(path).role == ROLE_POLICY
    monkeypatch.undo()

    def failing_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)
    with pytest.raises(OSError):
        save_checkpoint(path, _model(), role=ROLE_POLICY)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
    assert load_policy_checkpoint(path).role == ROLE_POLICY


def test_restore_random_state_recovers_torch_and_order_streams():
    torch.manual_seed(123)
    torch_state = torch.get_rng_state()
    expected_torch = torch.rand(5)
    order_source = torch.Generator().manual_seed(456)
    order_state = order_source.get_state()
    expected_order = torch.randperm(11, generator=order_source)

    torch.manual_seed(999)
    destination = torch.Generator().manual_seed(999)
    state = _state()
    state.torch_rng_state = torch_state
    state.order_rng_state = order_state
    restore_random_state(state, order_generator=destination, device=torch.device("cpu"))
    assert torch.equal(torch.rand(5), expected_torch)
    assert torch.equal(torch.randperm(11, generator=destination), expected_order)


def test_training_state_parsing_rejects_invalid_epoch_metadata(tmp_path):
    model = _model()
    state = _state()
    payload = build_payload(model, role=ROLE_LAST, training=state, metrics=None)
    payload["training"]["completed_epochs"] = 3
    payload["training"]["best_epoch"] = 2
    path = tmp_path / "bad-epoch.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, require_training=True)
