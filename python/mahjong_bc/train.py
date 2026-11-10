"""Epoch-boundary recoverable behavior-cloning training.

Each epoch visits every eligible training decision once in a seeded shuffled
order, including the final partial batch. Recovery checkpoints are written only
after a completed validation pass, so the last usable checkpoint always names a
completed epoch. An interrupted epoch is replayed from that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    TrainingCheckpointState,
    atomic_write_json,
    check_model_config_identity,
    load_checkpoint,
    restore_random_state,
    save_checkpoint,
    validate_training_state_compatible,
)
from .config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
    require_dataset_encoding,
)
from .constants import PACKAGE_VERSION, ROLE_BEST, ROLE_LAST
from .data import (
    DecisionDataset,
    contains_memory_mapped_arrays,
    iter_batches,
)
from .errors import ConfigurationError, DataError, ResumeError
from .metrics import MetricsReport, evaluate_dataset
from .policy import CandidatePolicy, build_policy, masked_cross_entropy

_ORDER_SEED_MASK = 0x9E3779B97F4A7C15
_RUN_RECORD_NAME = "run.json"


@dataclass(frozen=True)
class TrainingSummary:
    """Final training progress and epoch history."""

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


def _cpu_state_dict(model: CandidatePolicy) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to("cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class Trainer:
    """Train one candidate-scoring policy with recoverable epoch boundaries."""

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
            raise DataError("train_dataset must expose a split identity")
        if validation_dataset.identity is None:
            raise DataError("validation_dataset must expose a split identity")
        if training_config.num_workers > 0 and (
            contains_memory_mapped_arrays(train_dataset)
            or contains_memory_mapped_arrays(validation_dataset)
        ):
            raise DataError(
                "num_workers > 0 is not supported with memory-mapped datasets; "
                "use num_workers=0"
            )
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
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"
        self.run_record_path = self.output_dir / _RUN_RECORD_NAME
        self.model: CandidatePolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.order_generator = torch.Generator(device="cpu")
        self.order_generator.manual_seed(_order_seed(training_config.seed))

    # ---- public training entry point ------------------------------------

    def train(self, *, resume_from: Path | str | None = None) -> TrainingSummary:
        if resume_from is None:
            return self._train_from_scratch()
        return self._resume_and_train(Path(resume_from))

    # ---- run setup -------------------------------------------------------

    def _new_model_and_optimizer(self) -> tuple[
        CandidatePolicy, torch.optim.Optimizer
    ]:
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

    def _train_from_scratch(self) -> TrainingSummary:
        self._require_fresh_output()
        model, optimizer = self._new_model_and_optimizer()
        self.model = model
        self.optimizer = optimizer
        return self._run_epochs(
            start_epoch=0,
            global_step=0,
            samples_seen=0,
            best_validation_nll=None,
            best_epoch=None,
            best_model_state=None,
            best_metrics=None,
            history=[],
        )

    def _resume_and_train(self, checkpoint_path: Path) -> TrainingSummary:
        loaded = load_checkpoint(
            checkpoint_path,
            device=self.device,
            require_training=True,
        )
        if loaded.role != ROLE_LAST:
            raise ResumeError(
                f"resume requires a {ROLE_LAST!r} checkpoint, "
                f"got role {loaded.role!r}"
            )
        check_model_config_identity(loaded, self.model_config)
        state = loaded.training
        assert state is not None
        validate_training_state_compatible(
            state,
            training_config=self.training_config,
            train_dataset_identity=self.train_dataset.identity,
            validation_dataset_identity=self.validation_dataset.identity,
            dependencies=self.dependencies,
            device_type=self.device.type,
        )
        model = loaded.model.to(self.device).float()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            betas=self.training_config.betas,
            weight_decay=self.training_config.weight_decay,
        )
        optimizer.load_state_dict(state.optimizer_state)
        _move_optimizer_state(optimizer, self.device)
        restore_random_state(
            state,
            order_generator=self.order_generator,
            device=self.device,
        )
        self.model = model
        self.optimizer = optimizer
        if state.best_model_state is None or state.best_metrics is None:
            raise ResumeError(
                f"{checkpoint_path} does not embed the selected best model state; "
                "output paths cannot be migrated safely from this checkpoint"
            )
        # Publish both destination artifacts before any further epoch. This
        # makes a changed output directory self-contained even if training is
        # already complete or never improves on the embedded best.
        self._publish_migrated_outputs(loaded)
        if state.completed_epochs >= self.training_config.epochs:
            summary = TrainingSummary(
                completed_epochs=state.completed_epochs,
                global_step=state.global_step,
                samples_seen=state.samples_seen,
                best_validation_nll=state.best_validation_nll,
                best_epoch=state.best_epoch,
                history=tuple(state.history),
            )
            self._write_run_record(summary)
            return summary
        return self._run_epochs(
            start_epoch=state.completed_epochs,
            global_step=state.global_step,
            samples_seen=state.samples_seen,
            best_validation_nll=state.best_validation_nll,
            best_epoch=state.best_epoch,
            best_model_state=dict(state.best_model_state),
            best_metrics=dict(state.best_metrics),
            history=list(state.history),
        )

    def _publish_migrated_outputs(self, loaded) -> None:
        """Re-publish best and last artifacts into this run's output directory."""

        state = loaded.training
        assert state is not None
        if state.best_model_state is None or state.best_metrics is None:
            raise ResumeError("checkpoint has no embedded best model state")
        # Constructing the destination model must not consume training RNG.
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            [torch.cuda.get_rng_state(self.device)]
            if self.device.type == "cuda"
            else None
        )
        try:
            best_model = CandidatePolicy(self.model_config)
        finally:
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng[0], device=self.device)
        try:
            best_model.load_state_dict(state.best_model_state, strict=True)
        except Exception as exc:
            raise ResumeError(
                f"checkpoint best_model_state is incompatible with the model: {exc}"
            ) from exc
        best_model.float()
        save_checkpoint(
            self.best_path,
            best_model,
            role=ROLE_BEST,
            training=state,
            metrics=state.best_metrics,
        )
        save_checkpoint(
            self.last_path,
            loaded.model,
            role=ROLE_LAST,
            training=state,
            metrics=loaded.metrics,
        )

    def _require_fresh_output(self) -> None:
        existing = [
            path.name
            for path in (self.best_path, self.last_path, self.run_record_path)
            if path.exists()
        ]
        if existing:
            raise ConfigurationError(
                f"output directory {self.output_dir} already contains "
                f"{', '.join(existing)}; use --resume with last.pt or choose a "
                "new output directory (existing files are not deleted)"
            )

    # ---- epoch loop ------------------------------------------------------

    def _run_epochs(
        self,
        *,
        start_epoch: int,
        global_step: int,
        samples_seen: int,
        best_validation_nll: float | None,
        best_epoch: int | None,
        best_model_state: dict[str, torch.Tensor] | None,
        best_metrics: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> TrainingSummary:
        model = self.model
        optimizer = self.optimizer
        assert model is not None and optimizer is not None
        eligible_indices = self.train_dataset.eligible_indices()
        eligible_count = int(eligible_indices.size)
        batch_size = self.training_config.batch_size
        for epoch in range(start_epoch, self.training_config.epochs):
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
                best_metrics = validation.to_dict()
                best_model_state = _cpu_state_dict(model)
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
            state = self._training_state(
                completed_epochs=completed_epochs,
                global_step=global_step,
                samples_seen=samples_seen,
                best_validation_nll=best_validation_nll,
                best_epoch=best_epoch,
                best_model_state=best_model_state,
                best_metrics=best_metrics,
                history=history,
            )
            metrics = validation.to_dict()
            if improved:
                save_checkpoint(
                    self.best_path,
                    model,
                    role=ROLE_BEST,
                    training=state,
                    metrics=metrics,
                )
            save_checkpoint(
                self.last_path,
                model,
                role=ROLE_LAST,
                training=state,
                metrics=metrics,
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
        return TrainingSummary(
            completed_epochs=self.training_config.epochs,
            global_step=global_step,
            samples_seen=samples_seen,
            best_validation_nll=best_validation_nll,
            best_epoch=best_epoch,
            history=tuple(history),
        )

    def _training_state(
        self,
        *,
        completed_epochs: int,
        global_step: int,
        samples_seen: int,
        best_validation_nll: float | None,
        best_epoch: int | None,
        best_model_state: dict[str, torch.Tensor] | None,
        best_metrics: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> TrainingCheckpointState:
        assert self.optimizer is not None
        cuda_states = (
            [torch.cuda.get_rng_state(self.device)]
            if self.device.type == "cuda"
            else None
        )
        return TrainingCheckpointState(
            training_config=self.training_config,
            train_dataset_identity=self.train_dataset.identity,
            validation_dataset_identity=self.validation_dataset.identity,
            dependencies=self.dependencies,
            device_type=self.device.type,
            completed_epochs=completed_epochs,
            global_step=global_step,
            samples_seen=samples_seen,
            best_validation_nll=best_validation_nll,
            best_epoch=best_epoch,
            history=list(history),
            optimizer_state=self.optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_states=cuda_states,
            order_rng_state=self.order_generator.get_state(),
            best_model_state=best_model_state,
            best_metrics=best_metrics,
        )

    def _write_run_record(self, summary: TrainingSummary) -> None:
        atomic_write_json(
            self.run_record_path,
            {
                "package_version": PACKAGE_VERSION,
                "model_config": self.model_config.to_dict(),
                "training_config": self.training_config.to_dict(),
                "dependencies": self.dependencies.to_dict(),
                "device_type": self.device.type,
                "train_dataset_identity": self.train_dataset.identity.to_dict(),
                "validation_dataset_identity": self.validation_dataset.identity.to_dict(),
                "best_checkpoint": self.best_path.name,
                "last_checkpoint": self.last_path.name,
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
    """Evaluate a frozen checkpoint on one complete validation/test split."""

    from .checkpoint import load_policy_checkpoint

    if dataset.identity is None:
        raise DataError("evaluation dataset must expose a split identity")
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
