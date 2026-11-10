"""Shared CPU/CUDA device resolution for training and checkpoint loading."""

from __future__ import annotations

import torch

from .errors import ConfigurationError


def resolve_device(value: str | torch.device) -> torch.device:
    """Resolve a supported training device or raise a configuration error."""

    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ConfigurationError(f"invalid device {value!r}: {exc}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ConfigurationError(
            f"unsupported device type {device.type!r}; use 'cpu' or 'cuda'"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ConfigurationError(
                f"CUDA device index {device.index} is unavailable "
                f"({torch.cuda.device_count()} device(s) present)"
            )
    return device
