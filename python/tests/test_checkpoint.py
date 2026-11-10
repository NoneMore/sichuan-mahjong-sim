from __future__ import annotations

import pytest
import torch

import mahjong_bc.checkpoint as checkpoint_module
from mahjong_bc.checkpoint import (
    atomic_torch_save,
    build_payload,
    load_policy_checkpoint,
    save_checkpoint,
)
from mahjong_bc.config import ModelConfig
from mahjong_bc.constants import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    ROLE_BEST,
    ROLE_POLICY,
)
from mahjong_bc.errors import CheckpointError, ConfigurationError
from mahjong_bc.policy import build_policy


def _model() -> object:
    return build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=2)


def test_policy_checkpoint_roundtrip_and_shared_loader(tmp_path):
    path = tmp_path / "policy.pt"
    model = _model()
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model, role=ROLE_POLICY, metrics={"nll": 1.25})
    loaded = load_policy_checkpoint(path)
    assert loaded.role == ROLE_POLICY
    assert loaded.model_config == model.config
    assert loaded.metrics == {"nll": 1.25}
    assert loaded.model.training is False
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, before[name])
    metadata = loaded.metadata()
    assert metadata["has_training_state"] is False
    assert metadata["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["model_format_version"] == MODEL_FORMAT_VERSION


def test_best_checkpoint_without_recovery_state_roundtrip(tmp_path):
    model = _model()
    metrics = {"eligible": 4, "nll": 0.75, "top1": 0.5}
    payload = build_payload(
        model, role=ROLE_BEST, training=None, metrics=metrics
    )
    assert payload["role"] == ROLE_BEST
    assert payload["training"] is None
    assert payload["checkpoint_format_version"] == 2
    assert payload["model_format_version"] == 1
    assert payload["encoding_version"] == 1

    path = tmp_path / "best.pt"
    save_checkpoint(path, model, role=ROLE_BEST, metrics=metrics)
    loaded = load_policy_checkpoint(path)
    assert loaded.role == "best"
    assert loaded.metrics == metrics
    assert loaded.metadata()["has_training_state"] is False


def test_loader_rejects_non_null_training_recovery_state(tmp_path):
    payload = build_payload(_model(), role=ROLE_POLICY, metrics=None)
    payload["training"] = {"completed_epochs": 1}
    path = tmp_path / "recovery.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="recovery state"):
        load_policy_checkpoint(path)

    with pytest.raises(ConfigurationError, match="recovery state"):
        build_payload(_model(), role=ROLE_BEST, training={"completed_epochs": 1})
    with pytest.raises(ConfigurationError, match="recovery state"):
        save_checkpoint(
            tmp_path / "bad.pt",
            _model(),
            role=ROLE_BEST,
            training={"completed_epochs": 1},
        )
    assert not (tmp_path / "bad.pt").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("checkpoint_format_version", 1),
        ("checkpoint_format_version", 3),
        ("model_format_version", 2),
        ("encoding_version", 2),
        ("role", "last"),
        ("role", "unknown"),
        ("kind", "other-checkpoint"),
        ("package_version", ""),
    ],
)
def test_loader_rejects_unsupported_versions_roles_and_markers(
    tmp_path, key, value
):
    payload = build_payload(_model(), role=ROLE_POLICY, metrics=None)
    payload[key] = value
    path = tmp_path / "bad.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_policy_checkpoint(path)


def test_loader_rejects_unknown_keys_missing_state_and_corruption(tmp_path):
    model = _model()
    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["unknown"] = 1
    path = tmp_path / "unknown.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="invalid keys"):
        load_policy_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    del payload["model_state"]["state_encoder.0.weight"]
    path = tmp_path / "missing.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="model_state"):
        load_policy_checkpoint(path)

    payload = build_payload(model, role=ROLE_BEST, metrics=None)
    payload["model_config"] = ModelConfig(
        state_hidden=(4,), scorer_hidden=(8,)
    ).to_dict()
    path = tmp_path / "config-mismatch.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="model_state"):
        load_policy_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["metrics"] = []
    path = tmp_path / "bad-metrics.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="metrics"):
        load_policy_checkpoint(path)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="failed to read"):
        load_policy_checkpoint(corrupt)


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
        save_checkpoint(path, _model(), role=ROLE_BEST)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
    assert load_policy_checkpoint(path).role == ROLE_POLICY
