"""Fresh behavior-cloning training to a validation-selected best policy.

Each epoch visits every eligible training decision once in a seeded shuffled
order, including the final partial batch. Validation mean NLL alone selects
``best.pt``. This delivery trains from scratch only: it writes no ``last.pt``
and exposes no resume path. The policy checkpoint carries no optimizer or RNG
state, although its top-level ``training: null`` field reserves that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ._source import collect_training_source
from .checkpoint import atomic_write_json, save_checkpoint
from .config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
    require_dataset_encoding,
)
from .constants import PACKAGE_VERSION, ROLE_BEST
from .data import DecisionDataset, contains_memory_mapped_arrays, iter_batches
from .errors import ConfigurationError, DataError
from .metrics import MetricsReport, evaluate_dataset
from .policy import CandidatePolicy, build_policy, masked_cross_entropy

_ORDER_SEED_MASK = 0x9E3779B97F4A7C15
_BEST_CHECKPOINT_NAME = "best.pt"
_RESERVED_LAST_CHECKPOINT_NAME = "last.pt"
_RUN_RECORD_NAME = "run.json"
_PROTECTED_OUTPUT_NAMES = (
    _BEST_CHECKPOINT_NAME,
    _RESERVED_LAST_CHECKPOINT_NAME,
    _RUN_RECORD_NAME,
)


@dataclass(frozen=True)
class TrainingSummary:
    """Final fresh-run progress and epoch history."""

    completed_epochs: int
    global_step: int
    samples_seen: int
    best_validation_nll: float | None
    best_epoch: int | None
    history: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "best_validation_nll": self.best_validation_nll,
            "best_epoch": self.best_epoch,
            "history": [dict(entry) for entry in self.history],
        }


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


def _order_seed(seed: int) -> int:
    return (int(seed) ^ _ORDER_SEED_MASK) & (2**64 - 1)


def _require_same_capture(
    train: DatasetIdentity, validation: DatasetIdentity
) -> None:
    if train.split != "train":
        raise ConfigurationError(
            f"training dataset must use the 'train' split, got {train.split!r}"
        )
    if validation.split != "validation":
        raise ConfigurationError(
            "validation dataset must use the 'validation' split, "
            f"got {validation.split!r}"
        )
    for field in (
        "cache_id",
        "encoding_version",
        "preparation_format_version",
        "preparation_version",
        "source_revision",
    ):
        if getattr(train, field) != getattr(validation, field):
            raise ConfigurationError(
                f"train and validation dataset identities disagree on {field}"
            )


class Trainer:
    """Train one candidate-scoring policy and publish its best checkpoint."""

    def __init__(
        self,
        *,
        train_dataset: DecisionDataset,
        validation_dataset: DecisionDataset,
        training_config: TrainingConfig,
        output_dir: Path | str,
        model_config: ModelConfig | None = None,
        device: str | torch.device = "cpu",
        dependencies: DependencyVersions | None = None,
        dataset_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(train_dataset, DecisionDataset):
            raise ConfigurationError("train_dataset must be a DecisionDataset")
        if not isinstance(validation_dataset, DecisionDataset):
            raise ConfigurationError(
                "validation_dataset must be a DecisionDataset"
            )
        if not isinstance(training_config, TrainingConfig):
            raise ConfigurationError("training_config must be a TrainingConfig")
        if train_dataset.identity is None:
            raise DataError(
                "train_dataset must expose a split identity; anonymous shards "
                "cannot be used for training"
            )
        if validation_dataset.identity is None:
            raise DataError(
                "validation_dataset must expose a split identity; anonymous "
                "shards cannot be used for training"
            )
        if training_config.num_workers > 0 and (
            contains_memory_mapped_arrays(train_dataset)
            or contains_memory_mapped_arrays(validation_dataset)
        ):
            raise DataError(
                "num_workers > 0 is not supported with memory-mapped datasets; "
                "use num_workers=0"
            )
        if dataset_provenance is not None and not isinstance(
            dataset_provenance, Mapping
        ):
            raise ConfigurationError("dataset_provenance must be a mapping or None")
        self.model_config = model_config if model_config is not None else ModelConfig()
        _require_same_capture(train_dataset.identity, validation_dataset.identity)
        require_dataset_encoding(train_dataset.identity)
        require_dataset_encoding(validation_dataset.identity)
        if train_dataset.eligible_count == 0:
            raise ConfigurationError(
                "training split has no decisions with more than one candidate"
            )
        if validation_dataset.eligible_count == 0:
            raise ConfigurationError(
                "validation split has no decisions with more than one candidate; "
                "best-checkpoint selection would be undefined"
            )
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.device = resolve_device(device)
        self.dependencies = (
            dependencies if dependencies is not None else collect_dependency_versions()
        )
        # Capture the imported source once per Trainer; epoch records reuse it.
        self.source = collect_training_source()
        self.dataset_provenance = (
            None if dataset_provenance is None else dict(dataset_provenance)
        )
        self.best_path = self.output_dir / _BEST_CHECKPOINT_NAME
        self.run_record_path = self.output_dir / _RUN_RECORD_NAME
        self.model: CandidatePolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.order_generator = torch.Generator(device="cpu")
        self.order_generator.manual_seed(_order_seed(training_config.seed))

    # ---- public training entry point ------------------------------------

    def train(self) -> TrainingSummary:
        """Train from scratch, refusing to overwrite existing run artifacts."""

        self._require_fresh_output()
        model, optimizer = self._new_model_and_optimizer()
        self.model = model
        self.optimizer = optimizer
        return self._run_epochs()

    # ---- run setup -------------------------------------------------------

    def _new_model_and_optimizer(
        self,
    ) -> tuple[CandidatePolicy, torch.optim.Optimizer]:
        seed = self.training_config.seed
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = build_policy(self.model_config, seed=seed).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            betas=self.training_config.betas,
            weight_decay=self.training_config.weight_decay,
        )
        return model, optimizer

    def _require_fresh_output(self) -> None:
        existing = [
            name
            for name in _PROTECTED_OUTPUT_NAMES
            if (self.output_dir / name).exists()
        ]
        if existing:
            raise ConfigurationError(
                f"output directory {self.output_dir} already contains "
                f"{', '.join(existing)}; choose a new output directory "
                "(existing files are not deleted or overwritten)"
            )

    # ---- epoch loop ------------------------------------------------------

    def _run_epochs(self) -> TrainingSummary:
        model = self.model
        optimizer = self.optimizer
        assert model is not None and optimizer is not None
        eligible_indices = self.train_dataset.eligible_indices()
        eligible_count = int(eligible_indices.size)
        batch_size = self.training_config.batch_size
        global_step = 0
        samples_seen = 0
        best_validation_nll: float | None = None
        best_epoch: int | None = None
        history: list[dict[str, Any]] = []
        summary: TrainingSummary | None = None
        for epoch in range(self.training_config.epochs):
            model.train()
            permutation = torch.randperm(
                eligible_count, generator=self.order_generator
            ).numpy()
            epoch_indices = eligible_indices[permutation]
            train_nll_sum = 0.0
            train_decisions = 0
            epoch_steps = 0
            for batch in iter_batches(
                self.train_dataset,
                epoch_indices,
                batch_size=batch_size,
                num_workers=self.training_config.num_workers,
            ):
                moved = batch.to(self.device)
                if not bool(moved.eligible.all()):
                    raise DataError(
                        "training traversal included a forced decision"
                    )
                optimizer.zero_grad(set_to_none=True)
                logits = model(moved.states, moved.candidates, moved.mask)
                loss = masked_cross_entropy(
                    logits,
                    moved.mask,
                    moved.targets,
                    eligible=moved.eligible,
                )
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch_steps += 1
                decisions = moved.batch_size
                samples_seen += decisions
                train_decisions += decisions
                train_nll_sum += float(loss.detach().double().item()) * decisions
            if train_decisions != eligible_count:
                raise RuntimeError(
                    f"epoch {epoch} visited {train_decisions} decisions, "
                    f"expected {eligible_count}"
                )
            train_nll = train_nll_sum / train_decisions
            validation = evaluate_dataset(
                model,
                self.validation_dataset,
                batch_size=batch_size,
                device=self.device,
                num_workers=self.training_config.num_workers,
            )
            if validation.nll is None:
                raise ConfigurationError(
                    "validation split produced no eligible primary metric"
                )
            improved = (
                best_validation_nll is None or validation.nll < best_validation_nll
            )
            if improved:
                best_validation_nll = validation.nll
                best_epoch = epoch
                save_checkpoint(
                    self.best_path,
                    model,
                    role=ROLE_BEST,
                    metrics=validation.to_dict(),
                )
            completed_epochs = epoch + 1
            history.append(
                {
                    "epoch": epoch,
                    "completed_epochs": completed_epochs,
                    "steps": epoch_steps,
                    "samples": train_decisions,
                    "train_nll": train_nll,
                    "validation": validation.to_dict(),
                    "best_updated": improved,
                    "best_validation_nll": best_validation_nll,
                    "best_epoch": best_epoch,
                }
            )
            summary = TrainingSummary(
                completed_epochs=completed_epochs,
                global_step=global_step,
                samples_seen=samples_seen,
                best_validation_nll=best_validation_nll,
                best_epoch=best_epoch,
                history=tuple(history),
            )
            self._write_run_record(summary)
        assert summary is not None
        return summary

    def _write_run_record(self, summary: TrainingSummary) -> None:
        train_identity = self.train_dataset.identity
        validation_identity = self.validation_dataset.identity
        assert train_identity is not None and validation_identity is not None
        atomic_write_json(
            self.run_record_path,
            {
                "source": self.source,
                "package_version": PACKAGE_VERSION,
                "model_config": self.model_config.to_dict(),
                "training_config": self.training_config.to_dict(),
                "dependencies": self.dependencies.to_dict(),
                "device_type": self.device.type,
                "seed": self.training_config.seed,
                "dataset": {
                    "train_dataset_identity": train_identity.to_dict(),
                    "validation_dataset_identity": validation_identity.to_dict(),
                    "provenance": self.dataset_provenance,
                },
                "best_checkpoint": self.best_path.name,
                "run_record": self.run_record_path.name,
                "summary": summary.to_dict(),
            },
        )


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    dataset: DecisionDataset,
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> MetricsReport:
    """Evaluate a frozen policy checkpoint on a complete validation/test split."""

    from .checkpoint import load_policy_checkpoint

    if dataset.identity is None:
        raise DataError(
            "evaluation dataset must expose a split identity; anonymous shards "
            "cannot be evaluated"
        )
    if dataset.identity.split not in ("validation", "test"):
        raise ConfigurationError(
            "frozen evaluation requires a validation or test split, "
            f"got {dataset.identity.split!r}"
        )
    resolved = resolve_device(device)
    loaded = load_policy_checkpoint(checkpoint_path, device=resolved)
    if loaded.encoding_version != dataset.identity.encoding_version:
        raise ConfigurationError(
            "checkpoint encoding version does not match the dataset"
        )
    return evaluate_dataset(
        loaded.model,
        dataset,
        batch_size=batch_size,
        device=resolved,
        num_workers=num_workers,
    )
