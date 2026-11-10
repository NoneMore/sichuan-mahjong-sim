from __future__ import annotations

import dataclasses

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
from mahjong_bc.constants import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    ROLE_BEST,
    ROLE_LAST,
    ROLE_POLICY,
)
from mahjong_bc.errors import CheckpointError, ConfigurationError, ResumeError
from mahjong_bc.policy import build_policy

from .helpers import identity


def _model() -> object:
    return build_policy(ModelConfig(state_hidden=(8,), scorer_hidden=(8,)), seed=2)


def _config() -> TrainingConfig:
    return TrainingConfig(seed=3, epochs=3, batch_size=2, learning_rate=0.01)


def _identities() -> tuple[DatasetIdentity, DatasetIdentity]:
    return (
        identity(split="train", decisions=10, eligible=7),
        identity(split="validation", decisions=5, eligible=4),
    )


def _state(
    model,
    *,
    training_config: TrainingConfig | None = None,
    dependencies: DependencyVersions | None = None,
    device_type: str = "cpu",
    completed_epochs: int = 1,
) -> TrainingCheckpointState:
    train, validation = _identities()
    generator = torch.Generator().manual_seed(77)
    cuda_states = (
        [torch.get_rng_state()] if device_type == "cuda" else None
    )
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return TrainingCheckpointState(
        training_config=training_config or _config(),
        train_dataset_identity=train,
        validation_dataset_identity=validation,
        dependencies=dependencies or collect_dependency_versions(),
        device_type=device_type,
        completed_epochs=completed_epochs,
        global_step=4,
        samples_seen=8,
        best_validation_nll=0.75,
        best_epoch=0,
        history=[
            {"epoch": index, "steps": 2, "samples": 4}
            for index in range(completed_epochs)
        ],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=cuda_states,
        order_rng_state=generator.get_state(),
        best_model_state=best_state,
        best_metrics={"nll": 0.75},
    )


def _legacy_format2_payload(model, role: str = ROLE_BEST) -> dict:
    payload = build_payload(
        model,
        role=ROLE_POLICY,
        metrics={"nll": 0.75},
    )
    payload["role"] = role
    payload["checkpoint_format_version"] = 2
    payload["training"] = None
    return payload


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
    metadata = loaded.metadata()
    assert metadata["has_training_state"] is False
    assert metadata["completed_epochs"] is None
    assert metadata["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["model_format_version"] == MODEL_FORMAT_VERSION
    with pytest.raises(CheckpointError, match="recoverable training state"):
        load_checkpoint(path, require_training=True)


def test_training_checkpoint_roundtrip_and_resume_comparison(tmp_path):
    model = _model()
    state = _state(model)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, model, role=ROLE_LAST, training=state, metrics={"nll": 0.5}
    )
    loaded = load_checkpoint(path, require_training=True)
    assert loaded.role == ROLE_LAST
    assert loaded.training is not None
    loaded_state = loaded.training
    assert loaded_state.training_config == state.training_config
    assert loaded_state.train_dataset_identity == state.train_dataset_identity
    assert (
        loaded_state.validation_dataset_identity
        == state.validation_dataset_identity
    )
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
    assert loaded_state.best_model_state is not None
    assert loaded.metadata()["has_training_state"] is True
    assert loaded.metadata()["completed_epochs"] == 1

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


def test_legacy_format2_policy_is_read_only_and_cannot_resume(tmp_path):
    model = _model()
    path = tmp_path / "legacy-best.pt"
    atomic_torch_save(_legacy_format2_payload(model), path)
    loaded = load_policy_checkpoint(path)
    assert loaded.role == ROLE_BEST
    assert loaded.training is None
    assert loaded.metadata()["checkpoint_format_version"] == 2
    with pytest.raises(CheckpointError, match="recoverable training state"):
        load_checkpoint(path, require_training=True)


def test_build_payload_enforces_role_and_training_boundary(tmp_path):
    model = _model()
    state = _state(model)
    assert build_payload(model, role=ROLE_POLICY)["training"] is None
    for role in (ROLE_BEST, ROLE_LAST):
        assert (
            build_payload(model, role=role, training=state)["role"] == role
        )
        with pytest.raises(ConfigurationError, match="requires training state"):
            build_payload(model, role=role)
        assert not (tmp_path / f"{role}.pt").exists()
    with pytest.raises(ConfigurationError, match="cannot carry training state"):
        build_payload(model, role=ROLE_POLICY, training=state)
    with pytest.raises(ConfigurationError, match="TrainingCheckpointState"):
        save_checkpoint(tmp_path / "bad.pt", model, role=ROLE_LAST, training={})
    assert not (tmp_path / "bad.pt").exists()


def test_resume_comparison_rejects_every_identity_dimension(tmp_path):
    model = _model()
    state = _state(model)
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

    mismatches = (
        {"training_config": dataclasses.replace(_config(), epochs=2)},
        {
            "train_dataset_identity": identity(
                split="train", decisions=10, eligible=7, cache_id="other"
            )
        },
        {
            "validation_dataset_identity": identity(
                split="validation", decisions=6, eligible=4
            )
        },
        {
            "dependencies": DependencyVersions(
                python="different",
                torch="different",
                numpy="different",
                cuda_build=None,
            )
        },
        {"device_type": "cuda"},
    )
    for mismatch in mismatches:
        with pytest.raises(ResumeError):
            validate_training_state_compatible(
                loaded_state, **{**base, **mismatch}
            )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("checkpoint_format_version", 1),
        ("checkpoint_format_version", 4),
        ("model_format_version", 2),
        ("encoding_version", 2),
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
        load_checkpoint(path)


def test_loader_rejects_legacy_recovery_state_and_format2_last_role(tmp_path):
    model = _model()
    payload = _legacy_format2_payload(model)
    payload["training"] = _state(model).to_payload()
    path = tmp_path / "legacy-recovery.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="non-null training state"):
        load_checkpoint(path)

    payload = _legacy_format2_payload(model)
    payload["role"] = ROLE_LAST
    path = tmp_path / "legacy-last.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="format 2"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_LAST, training=_state(model))
    payload["role"] = ROLE_POLICY
    path = tmp_path / "policy-with-state.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="policy checkpoint"):
        load_checkpoint(path)


def test_loader_rejects_unknown_keys_missing_state_and_corruption(tmp_path):
    model = _model()
    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["unknown"] = 1
    path = tmp_path / "unknown.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="invalid keys"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    del payload["model_state"]["state_encoder.0.weight"]
    path = tmp_path / "missing.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="model_state"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_BEST, training=_state(model))
    payload["model_config"] = ModelConfig(
        state_hidden=(4,), scorer_hidden=(8,)
    ).to_dict()
    path = tmp_path / "config-mismatch.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="model_state"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["metrics"] = []
    path = tmp_path / "bad-metrics.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="metrics"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["model_state"] = []
    path = tmp_path / "bad-state.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="model_state must be a mapping"):
        load_checkpoint(path)

    payload = build_payload(model, role=ROLE_POLICY, metrics=None)
    payload["model_config"]["encoding_version"] = 2
    path = tmp_path / "bad-encoding.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="invalid model configuration"):
        load_checkpoint(path)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="failed to read"):
        load_checkpoint(corrupt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["training"].__setitem__("completed_epochs", 4),
        lambda payload: payload["training"].__setitem__("best_epoch", 3),
        lambda payload: payload["training"]["history"].pop(),
        lambda payload: payload["training"].__setitem__("device_type", "mps"),
        lambda payload: payload["training"].__setitem__(
            "torch_rng_state", "not-a-tensor"
        ),
        lambda payload: payload["training"].__setitem__(
            "optimizer_state", []
        ),
        lambda payload: payload["training"].__setitem__(
            "best_model_state", None
        ),
        lambda payload: payload["training"]["best_model_state"].pop(
            "state_encoder.0.weight"
        ),
        lambda payload: payload["training"]["validation_dataset_identity"].__setitem__(
            "cache_id", "different"
        ),
        lambda payload: payload["training"].__setitem__(
            "torch_cuda_rng_states", [torch.get_rng_state()]
        ),
    ],
)
def test_training_state_parsing_rejects_invalid_recovery_metadata(
    tmp_path, mutate
):
    model = _model()
    payload = build_payload(model, role=ROLE_LAST, training=_state(model))
    mutate(payload)
    path = tmp_path / "bad-recovery.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, require_training=True)


def test_cuda_device_type_requires_cuda_rng_state(tmp_path):
    model = _model()
    payload = build_payload(
        model,
        role=ROLE_LAST,
        training=_state(model, device_type="cuda"),
    )
    payload["training"]["torch_cuda_rng_states"] = None
    path = tmp_path / "cuda-without-state.pt"
    atomic_torch_save(payload, path)
    with pytest.raises(CheckpointError, match="CUDA RNG"):
        load_checkpoint(path)


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
    state = _state(_model())
    state.torch_rng_state = torch_state
    state.order_rng_state = order_state
    restore_random_state(
        state, order_generator=destination, device=torch.device("cpu")
    )
    assert torch.equal(torch.rand(5), expected_torch)
    assert torch.equal(torch.randperm(11, generator=destination), expected_order)


def test_cuda_rng_state_restores_on_available_cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    torch.cuda.manual_seed(321)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state(device), device=device)
    saved = torch.cuda.get_rng_state(device)
    expected = torch.empty(8, device=device).uniform_()
    torch.cuda.manual_seed(999)
    state = _state(_model(), device_type="cuda")
    state.torch_cuda_rng_states = [saved]
    restore_random_state(
        state,
        order_generator=torch.Generator().manual_seed(1),
        device=device,
    )
    actual = torch.empty(8, device=device).uniform_()
    assert torch.equal(actual, expected)
