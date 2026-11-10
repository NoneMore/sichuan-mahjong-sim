"""Versioned policy checkpoints and atomic publication.

Checkpoint format 2 stores model configuration, weights, and optional metrics.
A top-level ``training`` key is always present as ``null``: it reserves the
boundary for a future recovery payload but this release neither writes nor
accepts recovery state. The same loader will serve offline evaluation and
future closed-loop consumers, so unsupported checkpoint, model, or encoding
versions are rejected before a model is constructed.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ._device import resolve_device
from .config import ModelConfig
from .constants import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_KIND,
    ENCODING_VERSION,
    MODEL_FORMAT_VERSION,
    PACKAGE_VERSION,
    ROLES,
)
from .errors import CheckpointError, ConfigurationError
from .policy import CandidatePolicy

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


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A loaded, eval-mode policy checkpoint without recovery state."""

    path: Path
    model: CandidatePolicy
    model_config: ModelConfig
    role: str
    package_version: str
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
            "has_training_state": False,
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
    if payload["training"] is not None:
        raise CheckpointError(
            f"{path} carries training recovery state, which is not supported by "
            "this policy-checkpoint release; only training = null is accepted"
        )
    if not isinstance(payload["model_state"], Mapping):
        raise CheckpointError("model_state must be a mapping")
    metrics = payload["metrics"]
    if metrics is not None and not isinstance(metrics, Mapping):
        raise CheckpointError("metrics must be a mapping or null")
    return payload


def build_payload(
    model: CandidatePolicy,
    *,
    role: str,
    training: None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned policy payload without touching the filesystem.

    ``training`` exists only to make the reserved boundary explicit. This
    release accepts ``None`` and rejects every other value; a future recovery
    payload therefore needs an intentional schema/version change.
    """

    if not isinstance(model, CandidatePolicy):
        raise ConfigurationError("model must be a CandidatePolicy")
    if role not in ROLES:
        raise ConfigurationError(f"unsupported checkpoint role {role!r}")
    if training is not None:
        raise ConfigurationError(
            "training recovery state is not supported by this release"
        )
    if metrics is not None and not isinstance(metrics, Mapping):
        raise ConfigurationError("metrics must be a mapping or null")
    return {
        "kind": CHECKPOINT_KIND,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_format_version": MODEL_FORMAT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "package_version": PACKAGE_VERSION,
        "role": role,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "training": None,
        "metrics": None if metrics is None else dict(metrics),
    }


def save_checkpoint(
    path: Path | str,
    model: CandidatePolicy,
    *,
    role: str,
    training: None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Atomically publish a policy checkpoint."""

    payload = build_payload(
        model, role=role, training=training, metrics=metrics
    )
    atomic_torch_save(payload, path)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"failed to read checkpoint {path}: {exc}") from exc


def load_policy_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Load a policy/best checkpoint, rejecting unsupported formats and state."""

    source = Path(path)
    payload = _validate_versioned_payload(_torch_load(source), source)
    try:
        model_config = ModelConfig.from_dict(payload["model_config"])
    except ConfigurationError as exc:
        raise CheckpointError(f"invalid model configuration: {exc}") from exc
    raw_model_state = payload["model_state"]
    model = CandidatePolicy(model_config)
    try:
        model.load_state_dict(raw_model_state, strict=True)
    except Exception as exc:
        raise CheckpointError(f"incompatible model_state: {exc}") from exc
    target_device = resolve_device(device)
    model.to(target_device).float().eval()
    metrics = payload["metrics"]
    return LoadedCheckpoint(
        path=source,
        model=model,
        model_config=model_config,
        role=payload["role"],
        package_version=payload["package_version"],
        metrics=None if metrics is None else dict(metrics),
    )
