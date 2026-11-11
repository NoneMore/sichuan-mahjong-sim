"""Explicit offline checkpoint construction shared by the online test modules.

The fixture values stay parameters rather than module defaults: dataset
identity, model width/seed, role, and metrics are visible at each call site.
Tests may share a published checkpoint as a read-only baseline, but a test
that replaces, corrupts, or removes checkpoint bytes must first copy it with
:func:`copy_checkpoint` so no other test observes the mutation.

This module imports only the offline ``mahjong_bc`` package and PyTorch. It
deliberately does not import ``mahjong_eval``: the records tests load their
worktree module by path without the native extension.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from mahjong_bc.checkpoint import TrainingCheckpointState, save_checkpoint
from mahjong_bc.config import (
    DatasetIdentity,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from mahjong_bc.constants import ENCODING_VERSION, ROLE_BEST, ROLE_POLICY
from mahjong_bc.policy import build_policy


def model_config(hidden: int) -> ModelConfig:
    """One hidden width for both the state encoder and candidate scorer."""

    return ModelConfig(state_hidden=(hidden,), scorer_hidden=(hidden,))


def build_model(*, hidden: int, seed: int):
    """Build a deterministic policy without leaking the fixture seed to callers.

    ``build_policy(seed=...)`` normally reseeds the process-wide torch RNG.
    Restoring the surrounding state keeps unrelated tests independent of how
    often a fixture model is constructed.
    """

    config = model_config(hidden)
    with torch.random.fork_rng(devices=[]):
        return build_policy(config, seed=seed)


def build_training_state(
    model,
    *,
    cache_id: str,
    train_eligible_decisions: int = 2,
    validation_eligible_decisions: int = 1,
    training_seed: int = 3,
    batch_size: int = 1,
    global_step: int = 4,
    samples_seen: int = 6,
    best_validation_nll: float = 0.25,
) -> TrainingCheckpointState:
    """Recoverable training metadata with an explicit dataset identity."""

    def identity(split: str, decisions: int, eligible: int) -> DatasetIdentity:
        return DatasetIdentity(
            cache_id=cache_id,
            split=split,
            encoding_version=ENCODING_VERSION,
            preparation_format_version=1,
            source_revision="fixture-revision",
            decisions=decisions,
            eligible_decisions=eligible,
        )

    return TrainingCheckpointState(
        training_config=TrainingConfig(
            seed=training_seed,
            epochs=1,
            batch_size=batch_size,
            learning_rate=0.01,
        ),
        train_dataset_identity=identity(
            "train", 4, train_eligible_decisions
        ),
        validation_dataset_identity=identity(
            "validation", 2, validation_eligible_decisions
        ),
        dependencies=collect_dependency_versions(),
        device_type="cpu",
        completed_epochs=1,
        global_step=global_step,
        samples_seen=samples_seen,
        best_validation_nll=best_validation_nll,
        best_epoch=0,
        history=[
            {
                "epoch": 0,
                "steps": global_step,
                "samples": samples_seen,
            }
        ],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=None,
        order_rng_state=torch.Generator().manual_seed(7).get_state(),
        best_model_state={
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        best_metrics={"nll": best_validation_nll},
    )


@dataclass(frozen=True)
class CheckpointFixture:
    """A published checkpoint and the in-memory model that produced it."""

    path: Path
    model: Any
    training: TrainingCheckpointState | None


def build_checkpoint(
    path: Path,
    *,
    role: str = ROLE_BEST,
    hidden: int,
    model_seed: int,
    metrics: Mapping[str, Any] | None = None,
    cache_id: str | None = None,
    training: TrainingCheckpointState | None = None,
    model: Any | None = None,
) -> CheckpointFixture:
    """Write one role-consistent checkpoint and return its source model.

    ``best``/``last`` checkpoints carry training state (supplied or built from
    ``cache_id``), while ``policy`` checkpoints never do. Callers pass the
    dataset cache id explicitly because it is part of the evaluation identity.
    A caller that already built the model (for example to construct custom
    training state) may pass it instead of building an equivalent one twice.
    """

    target = Path(path)
    if model is None:
        model = build_model(hidden=hidden, seed=model_seed)
    if role == ROLE_POLICY:
        if training is not None:
            raise ValueError("policy checkpoints cannot carry training state")
    elif training is None:
        if cache_id is None:
            raise ValueError(f"{role} checkpoints require an explicit cache_id")
        training = build_training_state(model, cache_id=cache_id)
    save_checkpoint(
        target,
        model,
        role=role,
        training=training,
        metrics=metrics,
    )
    return CheckpointFixture(path=target, model=model, training=training)


def copy_checkpoint(source: Path, destination: Path) -> Path:
    """Copy checkpoint bytes for a test that mutates or removes its path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
