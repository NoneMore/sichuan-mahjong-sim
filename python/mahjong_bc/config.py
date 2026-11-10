"""Validated immutable model, training, dataset, and dependency identities."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .constants import (
    CANDIDATE_WIDTH,
    ENCODING_VERSION,
    PREPARATION_VERSION,
    SPLIT_NAMES,
    STATE_WIDTH,
)
from .errors import ConfigurationError

#: Activation names accepted by :class:`ModelConfig`; the policy factory maps
#: each name to its ``nn.Module`` implementation.
ACTIVATION_NAMES = ("relu", "gelu", "tanh")


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


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ConfigurationError(f"{name} must be finite, got {value!r}")
    return number


def _normalize_widths(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a sequence of positive integers")
    try:
        widths = tuple(value)
    except TypeError as exc:
        raise ConfigurationError(
            f"{name} must be a sequence of positive integers"
        ) from exc
    if not widths:
        raise ConfigurationError(f"{name} must contain at least one layer width")
    for width in widths:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ConfigurationError(
                f"{name} entries must be positive integers, got {width!r}"
            )
    return widths


@dataclass(frozen=True)
class ModelConfig:
    """Candidate-scoring model architecture.

    Hidden widths and the activation are experimental parameters. The
    encoding-versioned input widths are fixed and validated.
    """

    encoding_version: int = ENCODING_VERSION
    state_width: int = STATE_WIDTH
    candidate_width: int = CANDIDATE_WIDTH
    state_hidden: tuple[int, ...] = (256,)
    scorer_hidden: tuple[int, ...] = (128,)
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.encoding_version != ENCODING_VERSION:
            raise ConfigurationError(
                f"unsupported encoding_version {self.encoding_version}; "
                f"supported version is {ENCODING_VERSION}"
            )
        if self.state_width != STATE_WIDTH:
            raise ConfigurationError(
                f"state_width must be {STATE_WIDTH} for encoding v1, "
                f"got {self.state_width}"
            )
        if self.candidate_width != CANDIDATE_WIDTH:
            raise ConfigurationError(
                f"candidate_width must be {CANDIDATE_WIDTH} for encoding v1, "
                f"got {self.candidate_width}"
            )
        object.__setattr__(
            self, "state_hidden", _normalize_widths(self.state_hidden, "state_hidden")
        )
        object.__setattr__(
            self,
            "scorer_hidden",
            _normalize_widths(self.scorer_hidden, "scorer_hidden"),
        )
        if self.activation not in ACTIVATION_NAMES:
            raise ConfigurationError(
                f"activation must be one of {list(ACTIVATION_NAMES)}, "
                f"got {self.activation!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state_hidden"] = list(self.state_hidden)
        value["scorer_hidden"] = list(self.scorer_hidden)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> ModelConfig:
        value = _require_mapping(value, "model_config")
        _require_exact_keys(
            value,
            {
                "encoding_version",
                "state_width",
                "candidate_width",
                "state_hidden",
                "scorer_hidden",
                "activation",
            },
            "model_config",
        )
        return cls(
            encoding_version=value["encoding_version"],
            state_width=value["state_width"],
            candidate_width=value["candidate_width"],
            state_hidden=value["state_hidden"],
            scorer_hidden=value["scorer_hidden"],
            activation=value["activation"],
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and epoch configuration; no scheduler is used."""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    optimizer: str = "adamw"
    num_workers: int = 0

    def __post_init__(self) -> None:
        _non_negative_int(self.seed, "seed")
        if self.seed > 2**64 - 1:
            raise ConfigurationError("seed must fit in an unsigned 64-bit integer")
        _positive_int(self.epochs, "epochs")
        _positive_int(self.batch_size, "batch_size")
        learning_rate = _finite_float(self.learning_rate, "learning_rate")
        if learning_rate <= 0.0:
            raise ConfigurationError("learning_rate must be positive")
        if _finite_float(self.weight_decay, "weight_decay") < 0.0:
            raise ConfigurationError("weight_decay must be non-negative")
        beta1 = _finite_float(self.beta1, "beta1")
        beta2 = _finite_float(self.beta2, "beta2")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ConfigurationError("beta1 and beta2 must be in [0, 1)")
        if self.optimizer != "adamw":
            raise ConfigurationError(
                f"unsupported optimizer {self.optimizer!r}; only 'adamw' is implemented"
            )
        _non_negative_int(self.num_workers, "num_workers")

    @property
    def betas(self) -> tuple[float, float]:
        return (float(self.beta1), float(self.beta2))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> TrainingConfig:
        value = _require_mapping(value, "training_config")
        _require_exact_keys(
            value,
            {
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "beta1",
                "beta2",
                "optimizer",
                "num_workers",
            },
            "training_config",
        )
        return cls(**value)


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
