"""Validated immutable dataset and dependency identities."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .constants import (
    ENCODING_VERSION,
    PREPARATION_VERSION,
    SPLIT_NAMES,
)
from .errors import ConfigurationError


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise ConfigurationError(f"{name} has invalid keys: {', '.join(details)}")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer, got {value!r}")
    return value


@dataclass(frozen=True)
class DatasetIdentity:
    """Stable identity of one prepared split.

    ``cache_id`` is supplied by the cache loader and is derived by Rust from the
    captured input snapshot and output inventory, never from a mutable corpus
    root. Consumers compare it as an opaque value and must not recompute it.
    """

    cache_id: str
    split: str
    encoding_version: int
    preparation_format_version: int
    source_revision: str
    decisions: int
    eligible_decisions: int
    preparation_version: int = PREPARATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.cache_id, str) or not self.cache_id:
            raise ConfigurationError("cache_id must be a nonempty string")
        if self.split not in SPLIT_NAMES:
            raise ConfigurationError(
                f"split must be one of {list(SPLIT_NAMES)}, got {self.split!r}"
            )
        if self.encoding_version != ENCODING_VERSION:
            raise ConfigurationError(
                f"unsupported encoding_version {self.encoding_version}; "
                f"supported version is {ENCODING_VERSION}"
            )
        _positive_int(
            self.preparation_format_version,
            "preparation_format_version",
        )
        _positive_int(self.preparation_version, "preparation_version")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ConfigurationError("source_revision must be a nonempty string")
        _non_negative_int(self.decisions, "decisions")
        _non_negative_int(self.eligible_decisions, "eligible_decisions")
        if self.eligible_decisions > self.decisions:
            raise ConfigurationError(
                "eligible_decisions cannot exceed total decisions"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> DatasetIdentity:
        value = _require_mapping(value, "dataset_identity")
        _require_exact_keys(
            value,
            {
                "cache_id",
                "split",
                "encoding_version",
                "preparation_format_version",
                "source_revision",
                "decisions",
                "eligible_decisions",
                "preparation_version",
            },
            "dataset_identity",
        )
        return cls(**value)


@dataclass(frozen=True)
class DependencyVersions:
    """Runtime dependency versions reported by the ``version`` command."""

    python: str
    torch: str
    numpy: str
    cuda_build: str | None

    def __post_init__(self) -> None:
        for name in ("python", "torch", "numpy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ConfigurationError(f"{name} version must be a nonempty string")
        if self.cuda_build is not None and not isinstance(self.cuda_build, str):
            raise ConfigurationError("cuda_build must be a string or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> DependencyVersions:
        value = _require_mapping(value, "dependencies")
        _require_exact_keys(
            value,
            {"python", "torch", "numpy", "cuda_build"},
            "dependencies",
        )
        return cls(**value)


def collect_dependency_versions() -> DependencyVersions:
    """Return the locally installed versions relevant to reproducibility."""

    import numpy
    import torch

    cuda_build = None if torch.version.cuda is None else str(torch.version.cuda)
    return DependencyVersions(
        python=str(platform.python_version()),
        torch=str(torch.__version__),
        numpy=str(numpy.__version__),
        cuda_build=cuda_build,
    )
