"""Epoch-boundary recoverable behavior-cloning training.

Each epoch visits every eligible training decision once in a seeded shuffled
order, including the final partial batch. Validation mean NLL alone selects
``best.pt``. Recovery state is written only after a completed validation pass,
so ``last.pt`` always names a completed epoch; an interrupted epoch is replayed
from that boundary. ``last.pt`` is published after ``best.pt`` and is the commit
point for each recoverable epoch.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from ._device import resolve_device
from ._source import collect_training_source
from .checkpoint import (
    LoadedCheckpoint,
    TrainingCheckpointState,
    atomic_write_json,
    check_model_config_identity,
    load_checkpoint,
    load_policy_checkpoint,
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
)
from .constants import PACKAGE_VERSION, ROLE_BEST, ROLE_LAST
from .data import (
    DecisionBatch,
    DecisionDataset,
    _iter_model_batches,
    contains_memory_mapped_arrays,
    require_dataset_identity,
)
from .errors import ConfigurationError, DataError, ResumeError
from .metrics import MetricsReport, evaluate_dataset
from .policy import (
    CandidatePolicy,
    _batch_logits,
    _decision_nll_validated,
    _require_finite_valid,
    _validate_logits_mask_structure,
    _validate_mask_structure,
    _supports_packed_candidate_scoring,
    _validate_targets_structure,
    build_policy,
)

_ORDER_SEED_MASK = 0x9E3779B97F4A7C15
#: Batch losses buffered on the device before one host transfer. Bounds the
#: scalar allocation peak without changing the accumulation order.
_LOSS_CHUNK_SIZE = 1024
BEST_CHECKPOINT_NAME = "best.pt"
LAST_CHECKPOINT_NAME = "last.pt"
RUN_RECORD_NAME = "run.json"
TRAIN_OUTPUT_NAMES = (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    RUN_RECORD_NAME,
)


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


@dataclass
class _TrainingProgress:
    """Mutable counters and best-selection state for one training run.

    Fresh and resumed runs differ only in how this object is initialized; the
    epoch loop and its checkpoint/summary conversions read and update one
    source of truth instead of threading the same fields through call chains.
    """

    completed_epochs: int = 0
    global_step: int = 0
    samples_seen: int = 0
    best_validation_nll: float | None = None
    best_epoch: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_checkpoint(
        cls, state: TrainingCheckpointState
    ) -> "_TrainingProgress":
        return cls(
            completed_epochs=state.completed_epochs,
            global_step=state.global_step,
            samples_seen=state.samples_seen,
            best_validation_nll=state.best_validation_nll,
            best_epoch=state.best_epoch,
            best_model_state=dict(state.best_model_state or {}),
            best_metrics=dict(state.best_metrics or {}),
            history=list(state.history),
        )

    def summary(self) -> TrainingSummary:
        return TrainingSummary(
            completed_epochs=self.completed_epochs,
            global_step=self.global_step,
            samples_seen=self.samples_seen,
            best_validation_nll=self.best_validation_nll,
            best_epoch=self.best_epoch,
            history=tuple(self.history),
        )


def _require_host_training_batch(batch: DecisionBatch) -> None:
    """Check a collated training batch before it is copied to the device.

    The traversal is eligible-only and the packed layout is validated when the
    cache is loaded, so these invariants are host-side facts about the batch.
    Checking them here keeps the public predicates and messages of the loss
    helpers without a device round trip per batch. Model outputs are still
    structurally checked and tested for NaN/Inf on the device before every
    update.
    """

    if batch.states.dim() != 2 or batch.candidates.dim() != 3:
        raise DataError(
            "collated batches must carry states [B, state_width] and "
            f"candidates [B, C, candidate_width], got {tuple(batch.states.shape)} "
            f"and {tuple(batch.candidates.shape)}"
        )
    if batch.batch_size == 0:
        raise DataError("at least one eligible decision is required")
    _validate_mask_structure(
        batch.mask,
        (batch.batch_size, batch.candidates.shape[1]),
        batch.states.device,
    )
    _validate_targets_structure(
        batch.targets, batch.batch_size, batch.states.device
    )
    if not isinstance(batch.eligible, torch.Tensor) or batch.eligible.dtype != torch.bool:
        raise DataError("eligible must be a boolean tensor")
    if batch.eligible.shape != (batch.batch_size,):
        raise DataError(
            f"eligible must have shape ({batch.batch_size},), "
            f"got {tuple(batch.eligible.shape)}"
        )
    if batch.eligible.device != batch.states.device:
        raise DataError("eligible and logits must be on the same device")
    if not bool(batch.eligible.all()):
        raise DataError("training traversal included a forced decision")
    if not bool(batch.mask.any(dim=1).all()):
        raise DataError("every decision must have at least one valid candidate")
    if int(batch.targets.min()) < 0 or int(
        batch.targets.max()
    ) >= batch.candidates.shape[1]:
        raise DataError("target index is outside the padded candidate range")
    if not bool(batch.mask.gather(1, batch.targets.long().unsqueeze(1)).all()):
        raise DataError("target index does not reference a valid candidate")


class _EpochLossTotals:
    """Decision-weighted batch-loss sum with bounded device buffering.

    Batch losses are fetched in fixed-size chunks, so device residency does not
    grow with the epoch's step count. Each chunk is widened to float64 and then
    accumulated in batch order, which reproduces the per-batch
    ``float(loss.detach().double().item()) * decisions`` sum it replaces.
    """

    def __init__(self, chunk_size: int | None = None) -> None:
        self._chunk_size = _LOSS_CHUNK_SIZE if chunk_size is None else chunk_size
        self._losses: list[torch.Tensor] = []
        self._batch_decisions: list[int] = []
        self._total = 0.0
        self.decisions = 0

    def add(self, loss: torch.Tensor, decisions: int) -> None:
        self._losses.append(loss.detach())
        self._batch_decisions.append(decisions)
        self.decisions += decisions
        if len(self._losses) >= self._chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._losses:
            return
        values = torch.stack(self._losses).double().cpu().tolist()
        for value, decisions in zip(values, self._batch_decisions):
            self._total += float(value) * decisions
        self._losses.clear()
        self._batch_decisions.clear()

    def mean(self) -> float:
        self._flush()
        if self.decisions == 0:
            raise DataError("at least one eligible decision is required")
        return self._total / self.decisions


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


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


_OPTIMIZER_STATE_DICT_KEYS = {"state", "param_groups"}
_ADAMW_SLOT_KEYS = {"step", "exp_avg", "exp_avg_sq"}


def _invalid_optimizer(detail: str) -> ResumeError:
    return ResumeError(f"checkpoint optimizer state is invalid: {detail}")


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Mapping[str, Any],
    *,
    model: CandidatePolicy,
    completed_epochs: int,
    global_step: int,
) -> None:
    """Validate an AdamW state dict before the destination publishes anything.

    ``Optimizer.load_state_dict`` accepts arbitrary mappings and only fails, or
    silently changes the configured hyperparameters, after model reconstruction.
    Recovery instead requires the exact single parameter group and per-parameter
    slots this Trainer would have produced at an epoch boundary.
    """

    if set(state_dict) != _OPTIMIZER_STATE_DICT_KEYS:
        raise _invalid_optimizer(
            f"expected keys {sorted(_OPTIMIZER_STATE_DICT_KEYS)}, "
            f"got {sorted(state_dict)}"
        )
    raw_state = state_dict["state"]
    raw_groups = state_dict["param_groups"]
    if not isinstance(raw_state, Mapping):
        raise _invalid_optimizer("'state' must be a mapping")
    if not isinstance(raw_groups, list) or len(raw_groups) != 1:
        raise _invalid_optimizer("exactly one AdamW parameter group is required")
    group = raw_groups[0]
    if not isinstance(group, Mapping):
        raise _invalid_optimizer("the parameter group must be a mapping")

    expected = optimizer.state_dict()
    expected_group = expected["param_groups"][0]
    expected_keys = set(expected_group)
    actual_keys = set(group)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise _invalid_optimizer(
            "parameter group keys do not match AdamW: "
            f"missing {missing}, unknown {extra}"
        )
    for key in sorted(expected_keys - {"params"}):
        value = group[key]
        # Optimizer options are plain scalars/tuples. Reject tensors and other
        # containers before ``!=``, which would otherwise resolve to an
        # ambiguous tensor boolean or an unrelated comparison error.
        plain_value = isinstance(value, (bool, int, float, type(None))) or (
            isinstance(value, tuple)
            and all(
                isinstance(item, (bool, int, float)) for item in value
            )
        )
        if not plain_value:
            raise _invalid_optimizer(
                f"parameter group {key!r} has unsupported value type "
                f"{type(value).__name__}"
            )
        if value != expected_group[key]:
            raise _invalid_optimizer(
                f"parameter group {key!r} is {group[key]!r}, but the requested "
                f"training configuration requires {expected_group[key]!r}"
            )

    saved_params = group["params"]
    expected_params = list(expected_group["params"])
    if not isinstance(saved_params, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in saved_params
    ):
        raise _invalid_optimizer(
            "parameter group 'params' must be a list of integer indices"
        )
    if saved_params != expected_params:
        raise _invalid_optimizer(
            "parameter group 'params' does not list the model parameters once "
            "in model order"
        )
    parameters_by_index = dict(
        zip(expected_params, model.parameters())
    )

    expected_state_indices = set(expected_params)
    for index, entry in raw_state.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise _invalid_optimizer(
                f"state keys must be parameter indices, got {index!r}"
            )
        if index not in expected_state_indices:
            raise _invalid_optimizer(
                f"state references unknown parameter index {index!r}"
            )
        if completed_epochs == 0:
            raise _invalid_optimizer(
                "a checkpoint with no completed epoch must have empty AdamW state"
            )
        if not isinstance(entry, Mapping) or set(entry) != _ADAMW_SLOT_KEYS:
            raise _invalid_optimizer(
                f"parameter {index} state must contain exactly "
                f"{sorted(_ADAMW_SLOT_KEYS)}"
            )
        parameter = parameters_by_index[index]
        step = entry["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.dim() != 0
            or not step.dtype.is_floating_point
        ):
            raise _invalid_optimizer(
                f"parameter {index} 'step' must be a scalar floating tensor"
            )
        step_value = float(step.detach().cpu().item())
        if not math.isfinite(step_value) or step_value != float(global_step):
            raise _invalid_optimizer(
                f"parameter {index} 'step' is {step_value!r}, expected "
                f"global_step {global_step}"
            )
        for slot in ("exp_avg", "exp_avg_sq"):
            value = entry[slot]
            if not isinstance(value, torch.Tensor):
                raise _invalid_optimizer(
                    f"parameter {index} {slot!r} must be a tensor"
                )
            if value.shape != parameter.shape:
                raise _invalid_optimizer(
                    f"parameter {index} {slot!r} shape {tuple(value.shape)} "
                    f"does not match {tuple(parameter.shape)}"
                )
            if value.dtype != parameter.dtype:
                raise _invalid_optimizer(
                    f"parameter {index} {slot!r} dtype {value.dtype} does not "
                    f"match {parameter.dtype}"
                )

    if completed_epochs > 0:
        missing = sorted(expected_state_indices - set(raw_state))
        if missing:
            raise _invalid_optimizer(
                f"missing AdamW state for parameter indices {missing}"
            )


def _require_last_role(loaded: LoadedCheckpoint) -> None:
    if loaded.role != ROLE_LAST:
        raise ResumeError(
            f"resume requires a {ROLE_LAST!r} checkpoint, "
            f"got role {loaded.role!r}"
        )


def load_resume_checkpoint(path: Path | str) -> LoadedCheckpoint:
    """Load a recoverable ``last`` checkpoint once for resume orchestration.

    The CLI calls this before constructing datasets and a :class:`Trainer` so
    it can read the checkpoint's configuration; passing the returned value to
    ``Trainer.train(resume_from=...)`` reuses the same deserialized artifact.
    """

    loaded = load_checkpoint(path, device="cpu", require_training=True)
    _require_last_role(loaded)
    return loaded


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
        dataset_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        train_identity = require_dataset_identity(
            train_dataset, "train_dataset", dataset_error=ConfigurationError
        )
        validation_identity = require_dataset_identity(
            validation_dataset,
            "validation_dataset",
            dataset_error=ConfigurationError,
        )
        if not isinstance(training_config, TrainingConfig):
            raise ConfigurationError("training_config must be a TrainingConfig")
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
        _require_same_capture(train_identity, validation_identity)
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
        self.train_identity = train_identity
        self.validation_identity = validation_identity
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
        self.best_path = self.output_dir / BEST_CHECKPOINT_NAME
        self.last_path = self.output_dir / LAST_CHECKPOINT_NAME
        self.run_record_path = self.output_dir / RUN_RECORD_NAME
        self.model: CandidatePolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.order_generator = torch.Generator(device="cpu")
        self.order_generator.manual_seed(_order_seed(training_config.seed))

    # ---- public training entry point ------------------------------------

    def train(
        self,
        *,
        resume_from: Path | str | LoadedCheckpoint | None = None,
    ) -> TrainingSummary:
        """Train from scratch or continue from a completed-epoch ``last``.

        ``resume_from`` normally names a checkpoint. A loader-validated
        ``LoadedCheckpoint`` may be passed unchanged to reuse an artifact read
        while resolving configuration. Its original source path determines
        in-place versus migrated output; resume identity and recovery checks
        still run here.
        """

        if resume_from is None:
            return self._train_from_scratch()
        loaded = (
            resume_from
            if isinstance(resume_from, LoadedCheckpoint)
            else load_resume_checkpoint(resume_from)
        )
        return self._resume_and_train(loaded)

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

    def _train_from_scratch(self) -> TrainingSummary:
        self._require_fresh_output()
        model, optimizer = self._new_model_and_optimizer()
        self.model = model
        self.optimizer = optimizer
        return self._run_epochs(_TrainingProgress())

    def _resume_and_train(self, loaded: LoadedCheckpoint) -> TrainingSummary:
        _require_last_role(loaded)
        check_model_config_identity(loaded, self.model_config)
        state = loaded.training
        assert state is not None
        validate_training_state_compatible(
            state,
            training_config=self.training_config,
            train_dataset_identity=self.train_identity,
            validation_dataset_identity=self.validation_identity,
            dependencies=self.dependencies,
            device_type=self.device.type,
        )
        self._require_resume_progress(state)
        self._require_resume_output_target(loaded.path)
        model = loaded.model.to(self.device).float()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            betas=self.training_config.betas,
            weight_decay=self.training_config.weight_decay,
        )
        try:
            _validate_optimizer_state(
                optimizer,
                state.optimizer_state,
                model=model,
                completed_epochs=state.completed_epochs,
                global_step=state.global_step,
            )
            optimizer.load_state_dict(state.optimizer_state)
        except ResumeError:
            raise
        except Exception as exc:
            raise ResumeError(
                f"checkpoint optimizer state cannot be loaded: {exc}"
            ) from exc
        _move_optimizer_state(optimizer, self.device)
        restore_random_state(
            state,
            order_generator=self.order_generator,
            device=self.device,
        )
        self.model = model
        self.optimizer = optimizer
        # Publish both destination artifacts before any further epoch. This
        # makes a changed output directory self-contained even if training is
        # already complete or never improves on the embedded best.
        self._publish_migrated_outputs(loaded)
        progress = _TrainingProgress.from_checkpoint(state)
        if state.completed_epochs >= self.training_config.epochs:
            summary = progress.summary()
            self._write_run_record(summary)
            return summary
        return self._run_epochs(progress)

    def _publish_migrated_outputs(self, loaded: LoadedCheckpoint) -> None:
        """Re-publish best and last artifacts into this run's output directory."""

        state = loaded.training
        assert state is not None
        if state.best_model_state is None or state.best_metrics is None:
            raise ResumeError("checkpoint has no embedded best model state")
        # Copy the already constructed model instead of initializing a new one:
        # migration must not consume the restored training RNG streams. Keep
        # the migrated best model on CPU with its saved state tensors. ``last``
        # is written second and is the self-contained commit point.
        best_model = copy.deepcopy(loaded.model).cpu()
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

    def _require_resume_progress(self, state: TrainingCheckpointState) -> None:
        """Check the checkpoint is at a fixed full-traversal epoch boundary."""

        eligible = self.train_dataset.eligible_count
        batch_size = self.training_config.batch_size
        steps_per_epoch = (eligible + batch_size - 1) // batch_size
        expected_steps = state.completed_epochs * steps_per_epoch
        expected_samples = state.completed_epochs * eligible
        if (
            state.global_step != expected_steps
            or state.samples_seen != expected_samples
        ):
            raise ResumeError(
                "checkpoint training progress is inconsistent with its completed "
                f"epochs: completed_epochs={state.completed_epochs}, "
                f"global_step={state.global_step}, "
                f"samples_seen={state.samples_seen}; expected "
                f"global_step={expected_steps} and "
                f"samples_seen={expected_samples} for {eligible} eligible "
                f"decisions and batch size {batch_size}"
            )

    def _require_resume_output_target(self, checkpoint_path: Path) -> None:
        """Protect another run's artifacts from cross-directory migration.

        In-place resume may replace this run's own artifacts. A different
        output directory may be existing but must not contain any reserved run
        artifact, including a dangling symlink. Unrelated files are preserved.
        """

        checkpoint_directory = Path(checkpoint_path).resolve(strict=False).parent
        output_directory = self.output_dir.resolve(strict=False)
        if checkpoint_directory == output_directory:
            return
        occupied = [
            name
            for name in TRAIN_OUTPUT_NAMES
            if os.path.lexists(self.output_dir / name)
        ]
        if occupied:
            raise ConfigurationError(
                f"output directory {self.output_dir} already contains "
                f"{', '.join(occupied)}; cross-directory resume writes only "
                "into a directory without best.pt, last.pt, or run.json. "
                "Resume in place or choose a clean output directory "
                "(existing files are not deleted)"
            )

    def _require_fresh_output(self) -> None:
        existing = [
            name
            for name in TRAIN_OUTPUT_NAMES
            if (self.output_dir / name).exists()
        ]
        if existing:
            raise ConfigurationError(
                f"output directory {self.output_dir} already contains "
                f"{', '.join(existing)}; use --resume with last.pt or choose a "
                "new output directory (existing files are not deleted)"
            )

    # ---- epoch loop ------------------------------------------------------

    def _run_epochs(self, progress: _TrainingProgress) -> TrainingSummary:
        model = self.model
        optimizer = self.optimizer
        assert model is not None and optimizer is not None
        eligible_indices = self.train_dataset.eligible_indices()
        eligible_count = int(eligible_indices.size)
        batch_size = self.training_config.batch_size
        packed_candidates = _supports_packed_candidate_scoring(model)
        for epoch in range(progress.completed_epochs, self.training_config.epochs):
            model.train()
            permutation = torch.randperm(
                eligible_count, generator=self.order_generator
            ).numpy()
            epoch_indices = eligible_indices[permutation]
            epoch_losses = _EpochLossTotals()
            epoch_steps = 0
            for batch in _iter_model_batches(
                self.train_dataset,
                epoch_indices,
                batch_size=batch_size,
                num_workers=self.training_config.num_workers,
                training=True,
                packed_candidates=packed_candidates,
            ):
                if isinstance(batch, DecisionBatch):
                    _require_host_training_batch(batch)
                moved = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = _batch_logits(model, moved)
                _validate_logits_mask_structure(logits, moved.mask)
                _require_finite_valid(logits, moved.mask)
                loss = _decision_nll_validated(
                    logits, moved.mask, moved.targets
                ).mean()
                loss.backward()
                optimizer.step()
                progress.global_step += 1
                epoch_steps += 1
                decisions = moved.batch_size
                progress.samples_seen += decisions
                epoch_losses.add(loss, decisions)
            train_decisions = epoch_losses.decisions
            if train_decisions != eligible_count:
                raise RuntimeError(
                    f"epoch {epoch} visited {train_decisions} decisions, "
                    f"expected {eligible_count}"
                )
            train_nll = epoch_losses.mean()
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
                progress.best_validation_nll is None
                or validation.nll < progress.best_validation_nll
            )
            if improved:
                progress.best_validation_nll = validation.nll
                progress.best_epoch = epoch
                progress.best_metrics = validation.to_dict()
                progress.best_model_state = _cpu_state_dict(model)
            progress.completed_epochs = epoch + 1
            progress.history.append(
                {
                    "epoch": epoch,
                    "completed_epochs": progress.completed_epochs,
                    "steps": epoch_steps,
                    "samples": train_decisions,
                    "train_nll": train_nll,
                    "validation": validation.to_dict(),
                    "best_updated": improved,
                    "best_validation_nll": progress.best_validation_nll,
                    "best_epoch": progress.best_epoch,
                }
            )
            state = self._training_state(progress)
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
            self._write_run_record(progress.summary())
        return progress.summary()

    def _training_state(
        self, progress: _TrainingProgress
    ) -> TrainingCheckpointState:
        assert self.optimizer is not None
        cuda_states = (
            [torch.cuda.get_rng_state(self.device)]
            if self.device.type == "cuda"
            else None
        )
        return TrainingCheckpointState(
            training_config=self.training_config,
            train_dataset_identity=self.train_identity,
            validation_dataset_identity=self.validation_identity,
            dependencies=self.dependencies,
            device_type=self.device.type,
            completed_epochs=progress.completed_epochs,
            global_step=progress.global_step,
            samples_seen=progress.samples_seen,
            best_validation_nll=progress.best_validation_nll,
            best_epoch=progress.best_epoch,
            history=list(progress.history),
            optimizer_state=self.optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_states=cuda_states,
            order_rng_state=self.order_generator.get_state(),
            best_model_state=progress.best_model_state,
            best_metrics=progress.best_metrics,
        )

    def run_payload(self, summary: TrainingSummary) -> dict[str, Any]:
        """Build the reproducibility payload shared by ``run.json`` and the CLI."""

        return {
            "source": self.source,
            "package_version": PACKAGE_VERSION,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "device_type": self.device.type,
            "seed": self.training_config.seed,
            "dataset": {
                "train_dataset_identity": self.train_identity.to_dict(),
                "validation_dataset_identity": self.validation_identity.to_dict(),
                "provenance": self.dataset_provenance,
            },
            "best_checkpoint": self.best_path.name,
            "last_checkpoint": self.last_path.name,
            "run_record": self.run_record_path.name,
            "summary": summary.to_dict(),
        }

    def _write_run_record(self, summary: TrainingSummary) -> None:
        atomic_write_json(self.run_record_path, self.run_payload(summary))


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    dataset: DecisionDataset,
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> MetricsReport:
    """Evaluate a frozen policy checkpoint on a complete validation/test split."""

    identity = require_dataset_identity(dataset, "evaluation dataset")
    if identity.split not in ("validation", "test"):
        raise ConfigurationError(
            "frozen evaluation requires a validation or test split, "
            f"got {identity.split!r}"
        )
    resolved = resolve_device(device)
    loaded = load_policy_checkpoint(checkpoint_path, device=resolved)
    if loaded.encoding_version != identity.encoding_version:
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
