"""Versioned policy/training checkpoints and atomic publication.

The same loader is used by offline training recovery and by Python-driven
closed-loop evaluation. Unsupported encoding, model, or checkpoint versions are
rejected before model construction.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
)
from .constants import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_KIND,
    ENCODING_VERSION,
    MODEL_FORMAT_VERSION,
    PACKAGE_VERSION,
    ROLE_BEST,
    ROLE_LAST,
    ROLES,
)
from .errors import CheckpointError, ConfigurationError, ResumeError
from .policy import CandidatePolicy

_TRAINING_KEYS = {
    "training_config",
    "train_dataset_identity",
    "validation_dataset_identity",
    "dependencies",
    "device_type",
    "completed_epochs",
    "global_step",
    "samples_seen",
    "best_validation_nll",
    "best_epoch",
    "history",
    "optimizer_state",
    "torch_rng_state",
    "torch_cuda_rng_states",
    "order_rng_state",
    "best_model_state",
    "best_metrics",
}
_CHECKPOINT_KEYS = {
    "kind",
    "checkpoint_format_version",
    "model_format_version",
    "encoding_version",
    "package_version",
    "role",
    "model_config",
    "model_state",
    "training",
    "metrics",
}


@dataclass
class TrainingCheckpointState:
    """Recoverable training state at an epoch boundary."""

    training_config: TrainingConfig
    train_dataset_identity: DatasetIdentity
    validation_dataset_identity: DatasetIdentity
    dependencies: DependencyVersions
    device_type: str
    completed_epochs: int
    global_step: int
    samples_seen: int
    best_validation_nll: float | None
    best_epoch: int | None
    history: list[dict[str, Any]]
    optimizer_state: dict[str, Any]
    torch_rng_state: Tensor
    torch_cuda_rng_states: list[Tensor] | None
    order_rng_state: Tensor
    best_model_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "training_config": self.training_config.to_dict(),
            "train_dataset_identity": self.train_dataset_identity.to_dict(),
            "validation_dataset_identity": self.validation_dataset_identity.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "device_type": self.device_type,
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "best_validation_nll": self.best_validation_nll,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "optimizer_state": self.optimizer_state,
            "torch_rng_state": self.torch_rng_state,
            "torch_cuda_rng_states": self.torch_cuda_rng_states,
            "order_rng_state": self.order_rng_state,
            "best_model_state": self.best_model_state,
            "best_metrics": self.best_metrics,
        }


@dataclass
class LoadedCheckpoint:
    """A loaded, eval-mode policy plus optional recoverable training state."""

    path: Path
    model: CandidatePolicy
    model_config: ModelConfig
    role: str
    package_version: str
    training: TrainingCheckpointState | None
    metrics: dict[str, Any] | None

    @property
    def encoding_version(self) -> int:
        return self.model_config.encoding_version

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "role": self.role,
            "package_version": self.package_version,
            "encoding_version": self.encoding_version,
            "model_format_version": MODEL_FORMAT_VERSION,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": self.model_config.to_dict(),
            "has_training_state": self.training is not None,
            "completed_epochs": (
                None if self.training is None else self.training.completed_epochs
            ),
            "metrics": self.metrics,
        }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by writing and renaming within one directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _fsync_directory(destination.parent)


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Atomically publish a sorted, newline-terminated JSON document."""

    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(path, encoded.encode("utf-8"))


def atomic_torch_save(payload: Any, path: Path | str) -> None:
    """Atomically publish a ``torch.save`` payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _fsync_directory(destination.parent)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise CheckpointError(f"{name} has invalid keys: {', '.join(details)}")


def _require_rng_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise CheckpointError(f"{name} must be a torch tensor")
    if value.dtype != torch.uint8:
        raise CheckpointError(f"{name} must have uint8 dtype")
    if value.device.type != "cpu":
        raise CheckpointError(f"{name} must be a CPU tensor")
    return value


def _validate_versioned_payload(payload: Any, path: Path) -> Mapping[str, Any]:
    payload = _require_mapping(payload, "checkpoint")
    _require_exact_keys(payload, _CHECKPOINT_KEYS, "checkpoint")
    if payload["kind"] != CHECKPOINT_KIND:
        raise CheckpointError(
            f"{path} is not a {CHECKPOINT_KIND!r} checkpoint "
            f"(found {payload['kind']!r})"
        )
    checkpoint_version = payload["checkpoint_format_version"]
    if checkpoint_version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{path} has checkpoint format version {checkpoint_version}; "
            f"supported version is {CHECKPOINT_FORMAT_VERSION}"
        )
    model_version = payload["model_format_version"]
    if model_version != MODEL_FORMAT_VERSION:
        raise CheckpointError(
            f"{path} has model format version {model_version}; "
            f"supported version is {MODEL_FORMAT_VERSION}"
        )
    encoding_version = payload["encoding_version"]
    if encoding_version != ENCODING_VERSION:
        raise CheckpointError(
            f"{path} has encoding version {encoding_version}; "
            f"supported version is {ENCODING_VERSION}"
        )
    if payload["role"] not in ROLES:
        raise CheckpointError(
            f"{path} has unsupported checkpoint role {payload['role']!r}"
        )
    package_version = payload["package_version"]
    if not isinstance(package_version, str) or not package_version:
        raise CheckpointError(f"{path} has an invalid package_version")
    return payload


def _parse_training_state(value: Any) -> TrainingCheckpointState:
    value = _require_mapping(value, "training")
    _require_exact_keys(value, _TRAINING_KEYS, "training")
    try:
        training_config = TrainingConfig.from_dict(value["training_config"])
        train_identity = DatasetIdentity.from_dict(value["train_dataset_identity"])
        validation_identity = DatasetIdentity.from_dict(
            value["validation_dataset_identity"]
        )
        dependencies = DependencyVersions.from_dict(value["dependencies"])
    except ConfigurationError as exc:
        raise CheckpointError(f"invalid training checkpoint configuration: {exc}") from exc
    if train_identity.split != "train":
        raise CheckpointError("train_dataset_identity must describe the train split")
    if validation_identity.split != "validation":
        raise CheckpointError(
            "validation_dataset_identity must describe the validation split"
        )
    device_type = value["device_type"]
    if device_type not in ("cpu", "cuda"):
        raise CheckpointError(
            f"unsupported checkpoint device type {device_type!r}; "
            "only 'cpu' and 'cuda' are supported"
        )
    for name in ("completed_epochs", "global_step", "samples_seen"):
        field = value[name]
        if not isinstance(field, int) or isinstance(field, bool) or field < 0:
            raise CheckpointError(f"{name} must be a non-negative integer")
    if value["completed_epochs"] > training_config.epochs:
        raise CheckpointError(
            "completed_epochs cannot exceed the configured epoch count"
        )
    best_validation_nll = value["best_validation_nll"]
    if best_validation_nll is not None and not isinstance(
        best_validation_nll, (int, float)
    ):
        raise CheckpointError("best_validation_nll must be a number or null")
    best_epoch = value["best_epoch"]
    if best_epoch is not None and (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch < 0
    ):
        raise CheckpointError("best_epoch must be a non-negative integer or null")
    if value["completed_epochs"] > 0 and best_epoch is None:
        raise CheckpointError("a checkpoint with completed epochs must record best_epoch")
    if best_epoch is not None and best_epoch >= value["completed_epochs"]:
        raise CheckpointError("best_epoch must reference a completed epoch")
    history = value["history"]
    if not isinstance(history, list) or not all(
        isinstance(entry, Mapping) for entry in history
    ):
        raise CheckpointError("history must be a list of mappings")
    optimizer_state = value["optimizer_state"]
    if not isinstance(optimizer_state, Mapping):
        raise CheckpointError("optimizer_state must be a mapping")
    torch_rng_state = _require_rng_tensor(
        value["torch_rng_state"], "torch_rng_state"
    )
    order_rng_state = _require_rng_tensor(
        value["order_rng_state"], "order_rng_state"
    )
    raw_cuda_states = value["torch_cuda_rng_states"]
    if raw_cuda_states is None:
        cuda_states = None
    else:
        if not isinstance(raw_cuda_states, (list, tuple)):
            raise CheckpointError("torch_cuda_rng_states must be a list or null")
        cuda_states = [
            _require_rng_tensor(item, "torch_cuda_rng_states entry")
            for item in raw_cuda_states
        ]
    raw_best_state = value["best_model_state"]
    if raw_best_state is None:
        best_model_state = None
    else:
        if not isinstance(raw_best_state, Mapping):
            raise CheckpointError("best_model_state must be a mapping or null")
        best_model_state = {}
        for name, tensor in raw_best_state.items():
            if not isinstance(name, str) or not isinstance(tensor, Tensor):
                raise CheckpointError(
                    "best_model_state must map parameter names to tensors"
                )
            best_model_state[name] = tensor
    raw_best_metrics = value["best_metrics"]
    if raw_best_metrics is not None and not isinstance(raw_best_metrics, Mapping):
        raise CheckpointError("best_metrics must be a mapping or null")
    best_metrics = None if raw_best_metrics is None else dict(raw_best_metrics)
    return TrainingCheckpointState(
        training_config=training_config,
        train_dataset_identity=train_identity,
        validation_dataset_identity=validation_identity,
        dependencies=dependencies,
        device_type=device_type,
        completed_epochs=value["completed_epochs"],
        global_step=value["global_step"],
        samples_seen=value["samples_seen"],
        best_validation_nll=(
            None if best_validation_nll is None else float(best_validation_nll)
        ),
        best_epoch=best_epoch,
        history=[dict(entry) for entry in history],
        optimizer_state=dict(optimizer_state),
        torch_rng_state=torch_rng_state,
        torch_cuda_rng_states=cuda_states,
        order_rng_state=order_rng_state,
        best_model_state=best_model_state,
        best_metrics=best_metrics,
    )


def build_payload(
    model: CandidatePolicy,
    *,
    role: str,
    training: TrainingCheckpointState | None,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the versioned checkpoint payload without touching the filesystem."""

    if role not in ROLES:
        raise ConfigurationError(f"unsupported checkpoint role {role!r}")
    if role in (ROLE_BEST, ROLE_LAST) and training is None:
        raise ConfigurationError(f"a {role} checkpoint requires training state")
    return {
        "kind": CHECKPOINT_KIND,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_format_version": MODEL_FORMAT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "package_version": PACKAGE_VERSION,
        "role": role,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "training": None if training is None else training.to_payload(),
        "metrics": None if metrics is None else dict(metrics),
    }


def save_checkpoint(
    path: Path | str,
    model: CandidatePolicy,
    *,
    role: str,
    training: TrainingCheckpointState | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Atomically publish a policy/training checkpoint."""

    payload = build_payload(model, role=role, training=training, metrics=metrics)
    atomic_torch_save(payload, path)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"failed to read checkpoint {path}: {exc}") from exc


def load_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
    require_training: bool = False,
) -> LoadedCheckpoint:
    """Load and validate a checkpoint; rejects unsupported versions."""

    source = Path(path)
    payload = _validate_versioned_payload(_torch_load(source), source)
    try:
        model_config = ModelConfig.from_dict(payload["model_config"])
    except ConfigurationError as exc:
        raise CheckpointError(f"invalid model configuration: {exc}") from exc
    if payload["encoding_version"] != model_config.encoding_version:
        raise CheckpointError(
            "checkpoint encoding version does not match its model configuration"
        )
    raw_model_state = payload["model_state"]
    if not isinstance(raw_model_state, Mapping):
        raise CheckpointError("model_state must be a mapping")
    model = CandidatePolicy(model_config)
    try:
        model.load_state_dict(raw_model_state, strict=True)
    except Exception as exc:
        raise CheckpointError(f"incompatible model_state: {exc}") from exc
    target_device = torch.device(device)
    model.to(target_device).float().eval()

    training = None
    if payload["training"] is not None:
        training = _parse_training_state(payload["training"])
    if require_training and training is None:
        raise CheckpointError(f"{source} does not contain recoverable training state")
    metrics = payload["metrics"]
    if metrics is not None and not isinstance(metrics, Mapping):
        raise CheckpointError("metrics must be a mapping or null")
    return LoadedCheckpoint(
        path=source,
        model=model,
        model_config=model_config,
        role=payload["role"],
        package_version=payload["package_version"],
        training=training,
        metrics=None if metrics is None else dict(metrics),
    )


def load_policy_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Shared inference entry point for closed-loop evaluation."""

    return load_checkpoint(path, device=device, require_training=False)


def validate_training_state_compatible(
    state: TrainingCheckpointState,
    *,
    training_config: TrainingConfig,
    train_dataset_identity: DatasetIdentity,
    validation_dataset_identity: DatasetIdentity,
    dependencies: DependencyVersions,
    device_type: str,
) -> None:
    """Require the complete retraining identity to match the checkpoint.

    Output and log paths are deliberately not part of this identity. The
    caller separately compares the loaded checkpoint's model configuration via
    :func:`check_model_config_identity`.
    """

    if state.training_config != training_config:
        raise ResumeError(
            "training configuration differs from the checkpoint; "
            f"checkpoint={state.training_config.to_dict()} "
            f"requested={training_config.to_dict()}"
        )
    if state.train_dataset_identity != train_dataset_identity:
        raise ResumeError(
            "train dataset identity differs from the checkpoint; "
            f"checkpoint={state.train_dataset_identity.to_dict()} "
            f"requested={train_dataset_identity.to_dict()}"
        )
    if state.validation_dataset_identity != validation_dataset_identity:
        raise ResumeError(
            "validation dataset identity differs from the checkpoint; "
            f"checkpoint={state.validation_dataset_identity.to_dict()} "
            f"requested={validation_dataset_identity.to_dict()}"
        )
    if state.dependencies != dependencies:
        raise ResumeError(
            "dependency versions differ from the checkpoint; "
            f"checkpoint={state.dependencies.to_dict()} "
            f"requested={dependencies.to_dict()}"
        )
    if state.device_type != device_type:
        raise ResumeError(
            f"device type differs from the checkpoint; "
            f"checkpoint={state.device_type!r} requested={device_type!r}"
        )


def restore_random_state(
    state: TrainingCheckpointState,
    *,
    order_generator: torch.Generator,
    device: torch.device,
) -> None:
    """Restore CPU/CUDA torch RNG states and the dedicated epoch-order stream."""

    torch.set_rng_state(state.torch_rng_state)
    if device.type == "cuda":
        if state.torch_cuda_rng_states is None:
            raise ResumeError(
                "checkpoint has no CUDA RNG state but this run uses a CUDA device"
            )
        if not state.torch_cuda_rng_states:
            raise ResumeError("checkpoint has an empty CUDA RNG state list")
        torch.cuda.set_rng_state(state.torch_cuda_rng_states[0], device=device)
    order_generator.set_state(state.order_rng_state)


def check_model_config_identity(
    loaded: LoadedCheckpoint,
    model_config: ModelConfig,
) -> None:
    """Require a checkpoint's model architecture to match a requested run."""

    if loaded.model_config != model_config:
        raise ResumeError(
            "model configuration differs from the checkpoint; "
            f"checkpoint={loaded.model_config.to_dict()} "
            f"requested={model_config.to_dict()}"
        )
