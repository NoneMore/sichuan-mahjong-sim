from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch

import mahjong_bc
import mahjong_bc.train as train_module
from mahjong_bc.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_policy_checkpoint,
    save_checkpoint,
)
from mahjong_bc.config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
)
from mahjong_bc.constants import ROLE_BEST, ROLE_LAST, ROLE_POLICY
from mahjong_bc.data import PackedDecisionDataset
from mahjong_bc.errors import CheckpointError, ConfigurationError, DataError, ResumeError
from mahjong_bc.policy import build_policy
from mahjong_bc.train import (
    TRAIN_OUTPUT_NAMES,
    Trainer,
    evaluate_checkpoint,
    resolve_device,
)

from .helpers import DelegatingDataset, learnable_pair, make_dataset


class RecordingDataset(PackedDecisionDataset):
    def __init__(self, source: PackedDecisionDataset) -> None:
        super().__init__(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=source.identity,
            actor_seats=source.actor_seats,
            decision_indices=source.decision_indices,
            target_kinds=source.target_kinds,
        )
        self.visited: list[int] = []

    def __getitem__(self, index: int):
        self.visited.append(int(index))
        return super().__getitem__(index)


def _record(source: PackedDecisionDataset) -> RecordingDataset:
    return RecordingDataset(source)


def _config(**overrides: Any) -> TrainingConfig:
    values = dict(
        seed=7,
        epochs=3,
        batch_size=3,
        learning_rate=0.02,
        num_workers=0,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _small_model() -> ModelConfig:
    return ModelConfig(state_hidden=(16,), scorer_hidden=(16,))


def _summary_metrics(summary) -> list[float]:
    return [entry["validation"]["nll"] for entry in summary.history]


def _install_validation_sequence(monkeypatch, values):
    real_evaluate = train_module.evaluate_dataset
    sequence = iter(values)

    def fake_evaluate(model, dataset, **kwargs):
        report = real_evaluate(model, dataset, **kwargs)
        return dataclasses.replace(report, nll=next(sequence))

    monkeypatch.setattr(train_module, "evaluate_dataset", fake_evaluate)
    return real_evaluate


def _assert_state_tree_close(left: Any, right: Any, *, path: str = "") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_state_tree_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_state_tree_close(
                first, second, path=f"{path}[{index}]"
            )
        return
    if isinstance(left, float):
        assert right == pytest.approx(left, rel=1e-6, abs=1e-7), path
        return
    assert left == right, path


def _assert_state_dict_equal(left, right) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _reidentify(
    source: PackedDecisionDataset, new_identity: DatasetIdentity
) -> PackedDecisionDataset:
    return PackedDecisionDataset(
        states=source.states,
        candidate_features=source.candidate_features,
        offsets=source.offsets,
        targets=source.targets,
        stages=source.stages,
        identity=new_identity,
        actor_seats=source.actor_seats,
        decision_indices=source.decision_indices,
        target_kinds=source.target_kinds,
    )


def test_tiny_cpu_fixture_reduces_bc_loss_and_selects_validation_best(tmp_path):
    train, validation = learnable_pair(train_decisions=12, validation_decisions=6)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=40, batch_size=4, learning_rate=0.05),
        output_dir=tmp_path / "run",
        device="cpu",
    )
    summary = trainer.train()
    first = summary.history[0]["train_nll"]
    last = summary.history[-1]["train_nll"]
    assert last < first * 0.8
    validation_nlls = _summary_metrics(summary)
    assert summary.best_epoch == int(np.argmin(validation_nlls))
    assert summary.best_validation_nll == pytest.approx(min(validation_nlls))
    assert summary.completed_epochs == 40
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "last.pt").is_file()
    assert (tmp_path / "run" / "run.json").is_file()

    loaded_best = load_checkpoint(
        tmp_path / "run" / "best.pt", require_training=True
    )
    assert loaded_best.role == ROLE_BEST
    assert loaded_best.metadata()["has_training_state"] is True
    assert loaded_best.metrics["nll"] == pytest.approx(
        summary.best_validation_nll
    )
    assert loaded_best.training.best_epoch == summary.best_epoch

    loaded_last = load_checkpoint(
        tmp_path / "run" / "last.pt", require_training=True
    )
    assert loaded_last.role == ROLE_LAST
    assert loaded_last.training.completed_epochs == 40
    assert loaded_last.training.global_step == summary.global_step
    assert loaded_last.training.samples_seen == summary.samples_seen
    assert loaded_last.metrics["nll"] == pytest.approx(
        summary.history[-1]["validation"]["nll"]
    )
    assert loaded_last.training.best_model_state is not None


def test_each_epoch_visits_every_eligible_decision_once_with_partial_batch(tmp_path):
    train_source = make_dataset(
        split="train", counts=[2] * 7 + [1] * 4, seed=51, cache_id="traversal"
    )
    validation = make_dataset(
        split="validation", counts=[2, 2, 1], seed=52, cache_id="traversal"
    )
    train = _record(train_source)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=3, learning_rate=0.01),
        output_dir=tmp_path / "run",
        device="cpu",
    )
    summary = trainer.train()
    eligible = train_source.eligible_indices()
    assert train_source.eligible_count == 7
    assert len(train.visited) == 14
    for epoch in range(2):
        visited = train.visited[epoch * 7 : (epoch + 1) * 7]
        assert sorted(visited) == sorted(int(index) for index in eligible)
        assert len(set(visited)) == 7
        assert summary.history[epoch]["steps"] == 3
        assert summary.history[epoch]["samples"] == 7
    assert summary.global_step == 6
    assert summary.samples_seen == 14
    forced = set(int(index) for index in range(len(train_source))) - set(
        int(index) for index in eligible
    )
    assert not forced.intersection(train.visited)


def test_seeded_full_epoch_order_is_reproducible(tmp_path):
    source = make_dataset(
        split="train", counts=[2] * 6, seed=53, cache_id="order"
    )
    validation = make_dataset(
        split="validation", counts=[2] * 3, seed=54, cache_id="order"
    )
    first = _record(source)
    second = _record(source)
    config = _config(epochs=2, batch_size=2, learning_rate=0.01)
    summaries = []
    for name, dataset in (("first", first), ("second", second)):
        trainer = Trainer(
            train_dataset=dataset,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=config,
            output_dir=tmp_path / name,
            device="cpu",
        )
        summaries.append(trainer.train())
    assert first.visited == second.visited
    assert summaries[0].to_dict() == summaries[1].to_dict()


def test_midepoch_interruption_resumes_with_exact_following_order_and_state(
    tmp_path, monkeypatch
):
    base_train, base_validation = learnable_pair(
        train_decisions=12, validation_decisions=6, cache_id="resume-cache"
    )
    first_train = _record(base_train)
    second_train = _record(base_train)
    config = _config(epochs=4, batch_size=3, learning_rate=0.01)
    model = _small_model()
    continuous_order: list[list[int]] = []
    interrupted_order: list[list[int]] = []
    resumed_order: list[list[int]] = []
    real_batches = train_module.iter_batches

    def traced_batches(trace, *, interrupt=False):
        def iterator(dataset, indices, **kwargs):
            trace.append([int(index) for index in indices])
            for number, batch in enumerate(
                real_batches(dataset, indices, **kwargs)
            ):
                yield batch
                # The consumer has completed one optimizer step with this
                # batch before control returns here. Raise instead of fetching
                # the next batch, so the interruption happens mid-epoch after
                # exactly one executed step.
                if interrupt and len(trace) == 2 and number == 0:
                    raise KeyboardInterrupt("injected mid-epoch stop")

        return iterator

    uninterrupted = Trainer(
        train_dataset=first_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "uninterrupted",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "iter_batches", traced_batches(continuous_order)
    )
    summary_a = uninterrupted.train()

    interrupted = Trainer(
        train_dataset=second_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "interrupted",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module,
        "iter_batches",
        traced_batches(interrupted_order, interrupt=True),
    )
    with pytest.raises(KeyboardInterrupt, match="injected"):
        interrupted.train()
    boundary = load_checkpoint(
        tmp_path / "interrupted" / "last.pt", require_training=True
    )
    assert boundary.training.completed_epochs == 1
    # Epoch 1 completed all 12 decisions; epoch 2 loaded and trained exactly
    # its first batch before the injected mid-epoch stop. Those uncommitted
    # samples must be replayed after resuming from the epoch boundary.
    assert len(second_train.visited) == base_train.eligible_count + config.batch_size

    resumed = Trainer(
        train_dataset=second_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "resumed-output",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "iter_batches", traced_batches(resumed_order)
    )
    summary_b = resumed.train(resume_from=tmp_path / "interrupted" / "last.pt")

    assert continuous_order[:2] == interrupted_order[:2]
    assert continuous_order[1:] == resumed_order
    # The first 12 visits are completed epoch 1. The next batch was executed
    # before the stop but never committed, so resume must skip no training yet
    # still replay the whole epoch from its boundary. Compare the committed
    # prefix, the uncommitted batch, and every replayed decision separately.
    boundary_visits = base_train.eligible_count
    uncommitted_batch = config.batch_size
    assert (
        second_train.visited[:boundary_visits]
        == first_train.visited[:boundary_visits]
    )
    assert (
        second_train.visited[boundary_visits : boundary_visits + uncommitted_batch]
        == first_train.visited[
            boundary_visits : boundary_visits + uncommitted_batch
        ]
    )
    assert (
        second_train.visited[boundary_visits + uncommitted_batch :]
        == first_train.visited[boundary_visits:]
    )
    assert summary_b.completed_epochs == summary_a.completed_epochs == 4
    assert summary_b.global_step == summary_a.global_step
    assert summary_b.samples_seen == summary_a.samples_seen
    assert summary_b.best_epoch == summary_a.best_epoch
    assert summary_b.best_validation_nll == pytest.approx(
        summary_a.best_validation_nll, rel=1e-6, abs=1e-7
    )
    assert len(summary_b.history) == len(summary_a.history) == 4
    for entry_a, entry_b in zip(summary_a.history, summary_b.history):
        assert entry_a["steps"] == entry_b["steps"]
        assert entry_a["samples"] == entry_b["samples"]
        assert entry_a["train_nll"] == pytest.approx(
            entry_b["train_nll"], rel=1e-6, abs=1e-7
        )
        assert entry_a["validation"]["nll"] == pytest.approx(
            entry_b["validation"]["nll"], rel=1e-6, abs=1e-7
        )
    assert (tmp_path / "resumed-output" / "best.pt").is_file()
    assert (tmp_path / "resumed-output" / "last.pt").is_file()
    assert (tmp_path / "interrupted" / "last.pt").is_file()

    loaded_a = load_checkpoint(
        tmp_path / "uninterrupted" / "last.pt", require_training=True
    )
    loaded_b = load_checkpoint(
        tmp_path / "resumed-output" / "last.pt", require_training=True
    )
    assert loaded_a.training.global_step == loaded_b.training.global_step
    assert loaded_a.training.samples_seen == loaded_b.training.samples_seen
    _assert_state_tree_close(
        loaded_a.model.state_dict(), loaded_b.model.state_dict()
    )
    _assert_state_tree_close(
        loaded_a.training.optimizer_state, loaded_b.training.optimizer_state
    )
    assert torch.equal(
        loaded_a.training.torch_rng_state, loaded_b.training.torch_rng_state
    )
    assert torch.equal(
        loaded_a.training.order_rng_state, loaded_b.training.order_rng_state
    )
    assert loaded_a.training.best_model_state is not None
    _assert_state_tree_close(
        loaded_a.training.best_model_state,
        loaded_b.training.best_model_state,
    )


@dataclasses.dataclass(frozen=True)
class _ResumeBaseline:
    train: PackedDecisionDataset
    validation: PackedDecisionDataset
    model_config: ModelConfig
    training_config: TrainingConfig
    last: Path
    best: Path
    policy: Path


@dataclasses.dataclass(frozen=True)
class _ResumeRejectionCase:
    name: str
    expected: type[Exception]
    match: str | None = None
    source: str = "last"
    loaded: bool = False
    model_config: Callable[[ModelConfig], ModelConfig] | None = None
    training_config: Callable[[TrainingConfig], TrainingConfig] | None = None
    train_dataset: (
        Callable[[PackedDecisionDataset], PackedDecisionDataset] | None
    ) = None
    validation_dataset: (
        Callable[[PackedDecisionDataset], PackedDecisionDataset] | None
    ) = None
    dependencies: DependencyVersions | None = None
    preparation_version: int | None = None


def _other_capture_train(_: PackedDecisionDataset) -> PackedDecisionDataset:
    return make_dataset(
        split="train", counts=[2] * 8, seed=61, cache_id="other-dataset"
    )


def _other_capture_validation(_: PackedDecisionDataset) -> PackedDecisionDataset:
    return make_dataset(
        split="validation", counts=[2] * 4, seed=62, cache_id="other-dataset"
    )


def _other_validation_only(_: PackedDecisionDataset) -> PackedDecisionDataset:
    return make_dataset(
        split="validation", counts=[2, 2, 2], seed=63, cache_id="strict-resume"
    )


_RESUME_REJECTION_CASES = [
    _ResumeRejectionCase(
        "model-config",
        ResumeError,
        model_config=lambda _: ModelConfig(
            state_hidden=(32,), scorer_hidden=(16,)
        ),
    ),
    _ResumeRejectionCase(
        "learning-rate",
        ResumeError,
        training_config=lambda config: dataclasses.replace(
            config, learning_rate=0.02
        ),
    ),
    _ResumeRejectionCase(
        "epoch-count",
        ResumeError,
        training_config=lambda config: dataclasses.replace(config, epochs=2),
    ),
    _ResumeRejectionCase(
        "dataset-capture",
        ResumeError,
        train_dataset=_other_capture_train,
        validation_dataset=_other_capture_validation,
    ),
    _ResumeRejectionCase(
        "validation-identity",
        ResumeError,
        validation_dataset=_other_validation_only,
    ),
    _ResumeRejectionCase(
        "dependencies",
        ResumeError,
        dependencies=DependencyVersions(
            python="different",
            torch="different",
            numpy="different",
            cuda_build=None,
        ),
    ),
    _ResumeRejectionCase(
        "preparation-version",
        ResumeError,
        preparation_version=2,
    ),
    _ResumeRejectionCase("best-path", ResumeError, match="last", source="best"),
    _ResumeRejectionCase(
        "best-loaded", ResumeError, match="last", source="best", loaded=True
    ),
    _ResumeRejectionCase(
        "policy-path",
        CheckpointError,
        match="recoverable training state",
        source="policy",
    ),
    _ResumeRejectionCase(
        "policy-loaded", ResumeError, match="last", source="policy", loaded=True
    ),
]


@pytest.fixture(scope="module")
def resume_baseline(tmp_path_factory) -> _ResumeBaseline:
    """Train the shared rejection baseline once, without sharing a live Trainer."""

    root = tmp_path_factory.mktemp("resume-rejections")
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="strict-resume"
    )
    config = _config(epochs=1, batch_size=2, learning_rate=0.01)
    model = _small_model()
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=root / "base",
        device="cpu",
    )
    trainer.train()
    policy = root / "policy.pt"
    save_checkpoint(policy, build_policy(model, seed=1), role=ROLE_POLICY)
    return _ResumeBaseline(
        train=train,
        validation=validation,
        model_config=model,
        training_config=config,
        last=root / "base" / "last.pt",
        best=root / "base" / "best.pt",
        policy=policy,
    )


def _resume_case_trainer(
    baseline: _ResumeBaseline,
    output_dir: Path,
    case: _ResumeRejectionCase,
) -> Trainer:
    train = (
        baseline.train
        if case.train_dataset is None
        else case.train_dataset(baseline.train)
    )
    validation = (
        baseline.validation
        if case.validation_dataset is None
        else case.validation_dataset(baseline.validation)
    )
    if case.preparation_version is not None:
        train = _reidentify(
            train,
            dataclasses.replace(
                train.identity, preparation_version=case.preparation_version
            ),
        )
        validation = _reidentify(
            validation,
            dataclasses.replace(
                validation.identity,
                preparation_version=case.preparation_version,
            ),
        )
    return Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=(
            baseline.model_config
            if case.model_config is None
            else case.model_config(baseline.model_config)
        ),
        training_config=(
            baseline.training_config
            if case.training_config is None
            else case.training_config(baseline.training_config)
        ),
        output_dir=output_dir,
        device="cpu",
        dependencies=case.dependencies,
    )


def _resume_case_source(
    baseline: _ResumeBaseline, case: _ResumeRejectionCase
):
    path = getattr(baseline, case.source)
    if not case.loaded:
        return path
    if case.source == "policy":
        return load_policy_checkpoint(path)
    return load_checkpoint(path)


@pytest.mark.parametrize(
    "case", _RESUME_REJECTION_CASES, ids=lambda case: case.name
)
def test_resume_rejects_named_incompatibility(
    resume_baseline, tmp_path, case
):
    trainer = _resume_case_trainer(
        resume_baseline, tmp_path / case.name, case
    )
    with pytest.raises(case.expected, match=case.match):
        trainer.train(resume_from=_resume_case_source(resume_baseline, case))
    assert not trainer.output_dir.exists()


def test_resume_rejects_checkpoint_device_type_mismatch(tmp_path):
    train, validation = learnable_pair(
        train_decisions=4, validation_decisions=2, cache_id="device-resume"
    )
    config = _config(epochs=1, batch_size=2, learning_rate=0.01)
    model = _small_model()
    base = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "base",
        device="cpu",
    )
    base.train()
    payload = torch.load(
        tmp_path / "base" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    payload["training"]["device_type"] = "cuda"
    payload["training"]["torch_cuda_rng_states"] = [torch.get_rng_state()]
    cuda_last = tmp_path / "last-cuda.pt"
    atomic_torch_save(payload, cuda_last)
    resumed = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "resumed",
        device="cpu",
    )
    with pytest.raises(ResumeError, match="device type"):
        resumed.train(resume_from=cuda_last)
    assert not resumed.output_dir.exists()


def test_validation_only_best_selection_and_frozen_test_evaluation(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="selection"
    )
    test = make_dataset(
        split="test", counts=[2, 2, 1], seed=71, cache_id="selection"
    )
    config = _config(epochs=4, batch_size=2, learning_rate=0.01)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=config,
        output_dir=tmp_path / "selection",
        device="cpu",
    )
    real_evaluate = _install_validation_sequence(
        monkeypatch, [0.5, 0.6, 0.4, 0.7]
    )
    summary = trainer.train()
    monkeypatch.setattr(train_module, "evaluate_dataset", real_evaluate)
    assert summary.best_epoch == 2
    assert summary.best_validation_nll == pytest.approx(0.4)
    loaded_best = load_checkpoint(
        tmp_path / "selection" / "best.pt", require_training=True
    )
    loaded_last = load_checkpoint(
        tmp_path / "selection" / "last.pt", require_training=True
    )
    assert loaded_best.role == ROLE_BEST
    assert loaded_best.training.best_epoch == 2
    assert loaded_best.metrics["nll"] == pytest.approx(0.4)
    assert loaded_last.role == ROLE_LAST
    assert loaded_last.training.best_epoch == 2
    assert loaded_last.metrics["nll"] == pytest.approx(0.7)

    before = (tmp_path / "selection" / "best.pt").read_bytes()
    report = evaluate_checkpoint(
        tmp_path / "selection" / "best.pt",
        test,
        batch_size=2,
        device="cpu",
    )
    after = (tmp_path / "selection" / "best.pt").read_bytes()
    assert before == after
    assert report.decisions == len(test)
    assert report.forced == test.forced_count
    assert report.eligible == test.eligible_count


@pytest.mark.parametrize("protected_name", TRAIN_OUTPUT_NAMES)
def test_from_scratch_refuses_existing_run_artifacts_without_deleting(
    tmp_path, protected_name
):
    train, validation = learnable_pair()
    output = tmp_path / protected_name.replace(".", "-")
    output.mkdir()
    sentinel = output / protected_name
    sentinel.write_bytes(b"existing user artifact")
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
        output_dir=output,
        device="cpu",
    )
    with pytest.raises(ConfigurationError, match="already contains"):
        trainer.train()
    assert sentinel.read_bytes() == b"existing user artifact"


def test_training_rejects_unusable_datasets_and_memmap_workers(tmp_path):
    forced_only = make_dataset(split="train", counts=[1, 1], seed=81, cache_id="bad")
    validation = make_dataset(
        split="validation", counts=[1, 1], seed=82, cache_id="bad"
    )
    with pytest.raises(ConfigurationError):
        Trainer(
            train_dataset=forced_only,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "forced",
            device="cpu",
        )
    train = make_dataset(split="train", counts=[2, 2], seed=83, cache_id="bad2")
    with pytest.raises(ConfigurationError):
        Trainer(
            train_dataset=train,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "validation-forced",
            device="cpu",
        )

    anonymous = PackedDecisionDataset(
        states=train.states,
        candidate_features=train.candidate_features,
        offsets=train.offsets,
        targets=train.targets,
        stages=train.stages,
        identity=None,
    )
    with pytest.raises(DataError, match="identity"):
        Trainer(
            train_dataset=anonymous,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "anonymous",
            device="cpu",
        )

    class MappedDataset(RecordingDataset):
        @property
        def is_memory_mapped(self) -> bool:
            return True

    mapped = MappedDataset(train)
    with pytest.raises(DataError, match="num_workers"):
        Trainer(
            train_dataset=mapped,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(
                epochs=1, batch_size=2, learning_rate=0.01, num_workers=1
            ),
            output_dir=tmp_path / "workers",
            device="cpu",
        )

    other_train = make_dataset(
        split="train", counts=[2, 2], seed=84, cache_id="other"
    )
    with pytest.raises(ConfigurationError, match="disagree"):
        Trainer(
            train_dataset=other_train,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "different-cache",
            device="cpu",
        )

    wrong_split = make_dataset(
        split="validation", counts=[2, 2], seed=85, cache_id="bad2"
    )
    with pytest.raises(ConfigurationError, match="train"):
        Trainer(
            train_dataset=wrong_split,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "wrong-split",
            device="cpu",
        )


@pytest.mark.parametrize("failed_role", [ROLE_BEST, ROLE_LAST])
def test_training_checkpoint_write_failure_preserves_last_usable_epoch(
    tmp_path, monkeypatch, failed_role
):
    train, validation = learnable_pair(cache_id="write-failure")
    config = _config(epochs=3, batch_size=2, learning_rate=0.01)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=config,
        output_dir=tmp_path / "write-failure",
        device="cpu",
    )
    _install_validation_sequence(monkeypatch, [0.5, 0.4, 0.3])
    real_save = train_module.save_checkpoint
    calls = []

    def fail_second_role_write(path, model, **kwargs):
        if kwargs.get("role") == failed_role and Path(path).exists():
            calls.append(Path(path))
            if len(calls) == 1:
                raise RuntimeError("injected interrupted checkpoint write")
        return real_save(path, model, **kwargs)

    monkeypatch.setattr(
        train_module, "save_checkpoint", fail_second_role_write
    )
    with pytest.raises(RuntimeError, match="injected"):
        trainer.train()

    loaded = load_checkpoint(
        tmp_path / "write-failure" / "last.pt", require_training=True
    )
    assert loaded.role == ROLE_LAST
    assert loaded.training.completed_epochs == 1
    record = json.loads(
        (tmp_path / "write-failure" / "run.json").read_text(encoding="utf-8")
    )
    assert record["summary"]["completed_epochs"] == 1
    assert record["summary"]["best_validation_nll"] == pytest.approx(0.5)


def test_run_record_contains_configuration_identity_provenance_and_history(tmp_path):
    train, validation = learnable_pair(
        train_decisions=4, validation_decisions=2, cache_id="run-record"
    )
    dependencies = DependencyVersions(
        python="3.12.0", torch="2.0.0", numpy="1.24.0", cuda_build=None
    )
    provenance = {
        "cache_id": "run-record",
        "source_revision": "fixture-revision",
        "corpus_manifest_sha256": "a" * 64,
        "provenance": {"collection": {}, "preparation": {}},
    }
    config = _config(epochs=1, batch_size=2, learning_rate=0.01)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=config,
        output_dir=tmp_path / "record",
        device="cpu",
        dependencies=dependencies,
        dataset_provenance=provenance,
    )
    summary = trainer.train()
    record = json.loads(
        (tmp_path / "record" / "run.json").read_text(encoding="utf-8")
    )
    assert record["seed"] == config.seed
    assert record["source"] == trainer.source
    assert record["source"]["package"] == "mahjong-bc"
    assert record["source"]["package_version"] == record["package_version"]
    assert set(record["source"]) == {
        "package",
        "package_version",
        "hash",
        "revision",
        "dirty",
    }
    assert record["device_type"] == "cpu"
    assert record["model_config"] == _small_model().to_dict()
    assert record["training_config"] == config.to_dict()
    assert record["dependencies"] == dependencies.to_dict()
    assert record["dataset"]["train_dataset_identity"] == train.identity.to_dict()
    assert (
        record["dataset"]["validation_dataset_identity"]
        == validation.identity.to_dict()
    )
    assert record["dataset"]["provenance"] == provenance
    assert record["best_checkpoint"] == "best.pt"
    assert record["last_checkpoint"] == "last.pt"
    assert set(record) == {
        "source",
        "package_version",
        "model_config",
        "training_config",
        "dependencies",
        "device_type",
        "seed",
        "dataset",
        "best_checkpoint",
        "last_checkpoint",
        "run_record",
        "summary",
    }
    assert set(record["dataset"]) == {
        "train_dataset_identity",
        "validation_dataset_identity",
        "provenance",
    }
    assert record == trainer.run_payload(summary)
    assert record["summary"]["history"][0]["validation"]["nll"] == pytest.approx(
        summary.best_validation_nll
    )
    assert len(record["summary"]["history"]) == 1


def test_evaluate_checkpoint_rejects_training_and_anonymous_datasets(tmp_path):
    train, validation = learnable_pair(
        train_decisions=4, validation_decisions=2, cache_id="eval-guard"
    )
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "eval-guard",
        device="cpu",
    )
    trainer.train()
    with pytest.raises(ConfigurationError, match="validation or test"):
        evaluate_checkpoint(
            tmp_path / "eval-guard" / "best.pt",
            train,
            batch_size=2,
            device="cpu",
        )
    anonymous = PackedDecisionDataset(
        states=validation.states,
        candidate_features=validation.candidate_features,
        offsets=validation.offsets,
        targets=validation.targets,
        stages=validation.stages,
        identity=None,
    )
    with pytest.raises(DataError, match="identity"):
        evaluate_checkpoint(
            tmp_path / "eval-guard" / "best.pt",
            anonymous,
            batch_size=2,
            device="cpu",
        )
    duck = DelegatingDataset(
        validation, identity=SimpleNamespace(**validation.identity.to_dict())
    )
    with pytest.raises(DataError, match="DatasetIdentity"):
        evaluate_checkpoint(
            tmp_path / "eval-guard" / "best.pt",
            duck,
            batch_size=2,
            device="cpu",
        )


def test_trainer_requires_constructed_dataset_identity(tmp_path):
    train, validation = learnable_pair(cache_id="identity-guard")
    with pytest.raises(ConfigurationError, match="must be a DecisionDataset"):
        Trainer(
            train_dataset=object(),
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "not-a-dataset",
            device="cpu",
        )
    duck_train = DelegatingDataset(
        train, identity=SimpleNamespace(**train.identity.to_dict())
    )
    with pytest.raises(DataError, match="DatasetIdentity"):
        Trainer(
            train_dataset=duck_train,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "duck-train",
            device="cpu",
        )
    duck_validation = DelegatingDataset(
        validation, identity=SimpleNamespace(**validation.identity.to_dict())
    )
    with pytest.raises(DataError, match="DatasetIdentity"):
        Trainer(
            train_dataset=train,
            validation_dataset=duck_validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "duck-validation",
            device="cpu",
        )


def test_device_resolution_is_shared_by_training_and_checkpoint_loading(tmp_path):
    path = tmp_path / "device.pt"
    save_checkpoint(path, build_policy(_small_model(), seed=1), role=ROLE_POLICY)
    assert mahjong_bc.resolve_device is train_module.resolve_device is resolve_device
    assert resolve_device("cpu") == torch.device("cpu")
    loaded = load_policy_checkpoint(path, device="cpu")
    assert next(loaded.model.parameters()).device.type == "cpu"

    cases = [
        ("not-a-device", "invalid device"),
        ("mps", "unsupported device"),
    ]
    if not torch.cuda.is_available():
        cases.append(("cuda", "not available"))
    elif torch.cuda.device_count() > 0:
        cases.append((f"cuda:{torch.cuda.device_count()}", "unavailable"))
    for value, match in cases:
        with pytest.raises(ConfigurationError, match=match):
            resolve_device(value)
        with pytest.raises(ConfigurationError, match=match):
            load_policy_checkpoint(path, device=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training_and_checkpoint_reload_smoke(tmp_path):
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="cuda-smoke"
    )
    test = make_dataset(
        split="test", counts=[2, 2, 1], seed=91, cache_id="cuda-smoke"
    )
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "cuda",
        device="cuda",
    )
    summary = trainer.train()
    assert summary.completed_epochs == 2
    loaded = load_checkpoint(
        tmp_path / "cuda" / "last.pt",
        device="cuda",
        require_training=True,
    )
    assert loaded.role == ROLE_LAST
    assert loaded.training.device_type == "cuda"
    assert loaded.training.torch_cuda_rng_states
    expected_torch_rng = loaded.training.torch_rng_state.clone()
    expected_cuda_rng = loaded.training.torch_cuda_rng_states[0].clone()
    best_state = {
        name: value.clone()
        for name, value in loaded.training.best_model_state.items()
    }
    migrated = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "cuda-migrated",
        device="cuda",
    )
    migrated.train(resume_from=loaded.path)
    # Copying the loaded model replaces random initialization during migration,
    # so both restored streams must survive unchanged.
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert torch.equal(
        torch.cuda.get_rng_state(torch.device("cuda")), expected_cuda_rng
    )
    migrated_best = load_checkpoint(migrated.best_path, require_training=True)
    _assert_state_dict_equal(migrated_best.model.state_dict(), best_state)
    report = evaluate_checkpoint(
        tmp_path / "cuda" / "best.pt", test, batch_size=2, device="cuda"
    )
    assert report.eligible == test.eligible_count


def test_trainer_captures_source_once_and_reuses_it(tmp_path, monkeypatch):
    calls: list[None] = []
    source = {
        "package": "mahjong-bc",
        "package_version": "fixture-version",
        "hash": "f" * 64,
        "revision": None,
        "dirty": None,
    }

    def fake_collect_source() -> dict[str, object]:
        calls.append(None)
        return dict(source)

    train, validation = learnable_pair(
        train_decisions=4, validation_decisions=2, cache_id="source-once"
    )
    monkeypatch.setattr(
        train_module, "collect_training_source", fake_collect_source
    )
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "source-once",
        device="cpu",
    )
    assert len(calls) == 1
    trainer.train()
    assert len(calls) == 1
    record = json.loads(
        (tmp_path / "source-once" / "run.json").read_text(encoding="utf-8")
    )
    assert record["source"] == source


def test_resume_to_new_output_migrates_embedded_best_without_improvement(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="migration"
    )
    config = _config(epochs=3, batch_size=2, learning_rate=0.01)
    model = _small_model()
    old = tmp_path / "old"
    real_evaluate = _install_validation_sequence(monkeypatch, [0.5, 0.6, 0.7])

    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=old,
        device="cpu",
    )
    original_write = Trainer._write_run_record

    def stop_after_second_epoch(self, summary):
        original_write(self, summary)
        if summary.completed_epochs == 2:
            raise KeyboardInterrupt("injected stop with best before last")

    monkeypatch.setattr(Trainer, "_write_run_record", stop_after_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    monkeypatch.setattr(Trainer, "_write_run_record", original_write)

    old_best = load_checkpoint(old / "best.pt", require_training=True)
    old_last = load_checkpoint(old / "last.pt", require_training=True)
    assert old_last.training.best_epoch < old_last.training.completed_epochs - 1
    assert old_last.training.best_model_state is not None
    _assert_state_dict_equal(
        old_last.training.best_model_state, old_best.model.state_dict()
    )
    best_weights = {
        name: value.clone() for name, value in old_best.model.state_dict().items()
    }
    last_weights = {
        name: value.clone() for name, value in old_last.model.state_dict().items()
    }
    # Prove the new directory does not depend on the old best artifact.
    (old / "best.pt").unlink()

    new = tmp_path / "new"
    resumed = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=new,
        device="cpu",
    )
    summary = resumed.train(resume_from=old / "last.pt")
    monkeypatch.setattr(train_module, "evaluate_dataset", real_evaluate)
    assert summary.completed_epochs == 3
    assert summary.best_epoch == 0
    assert summary.best_validation_nll == pytest.approx(0.5)
    new_best = load_checkpoint(new / "best.pt", require_training=True)
    new_last = load_checkpoint(new / "last.pt", require_training=True)
    _assert_state_dict_equal(new_best.model.state_dict(), best_weights)
    assert new_best.training.best_epoch == 0
    assert new_best.metrics["nll"] == pytest.approx(0.5)
    assert new_last.training.completed_epochs == 3
    assert new_last.metrics["nll"] == pytest.approx(0.7)
    assert new_last.training.best_model_state is not None
    _assert_state_dict_equal(
        new_last.training.best_model_state, best_weights
    )
    assert not torch.equal(
        new_last.model.state_dict()["candidate_scorer.2.weight"],
        last_weights["candidate_scorer.2.weight"],
    )
    assert (new / "run.json").is_file()


def test_completed_resume_to_new_output_copies_best_and_last_and_does_not_train(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="completed-migration"
    )
    config = _config(epochs=2, batch_size=2, learning_rate=0.01)
    model = _small_model()
    old = tmp_path / "old"
    real_evaluate = _install_validation_sequence(monkeypatch, [0.5, 0.6])
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=old,
        device="cpu",
    )
    summary = trainer.train()
    monkeypatch.setattr(train_module, "evaluate_dataset", real_evaluate)
    assert summary.best_epoch == 0
    assert summary.completed_epochs == 2
    old_best = load_checkpoint(old / "best.pt", require_training=True)
    old_last = load_checkpoint(old / "last.pt", require_training=True)
    expected_torch_rng = old_last.training.torch_rng_state.clone()
    best_weights = {
        name: value.clone() for name, value in old_best.model.state_dict().items()
    }
    last_weights = {
        name: value.clone() for name, value in old_last.model.state_dict().items()
    }
    (old / "best.pt").unlink()

    recorder = _record(train)
    new = tmp_path / "completed-new"
    resumed = Trainer(
        train_dataset=recorder,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=new,
        device="cpu",
    )
    resumed_summary = resumed.train(resume_from=old / "last.pt")
    # Completed migration restores and then must not consume either RNG stream.
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert recorder.visited == []
    assert resumed_summary.to_dict() == summary.to_dict()
    new_best = load_checkpoint(new / "best.pt", require_training=True)
    new_last = load_checkpoint(new / "last.pt", require_training=True)
    _assert_state_dict_equal(new_best.model.state_dict(), best_weights)
    _assert_state_dict_equal(new_last.model.state_dict(), last_weights)
    assert new_best.metrics["nll"] == pytest.approx(0.5)
    assert new_last.metrics["nll"] == pytest.approx(0.6)
    assert (new / "run.json").is_file()
