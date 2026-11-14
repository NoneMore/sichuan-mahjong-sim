from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import mahjong_bc
from mahjong_bc import BatchTransferConfig
import mahjong_bc.metrics as metrics_module
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
from mahjong_bc.data import PackedDecisionDataset, _PackedCandidateBatch, iter_batches
from mahjong_bc.errors import CheckpointError, ConfigurationError, DataError, ResumeError
import mahjong_bc.policy as policy_module
from mahjong_bc.policy import CandidatePolicy, build_policy
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


def _assert_state_tree_exact(left: Any, right: Any, *, path: str = "") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_state_tree_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_state_tree_exact(first, second, path=f"{path}[{index}]")
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
    real_batches = train_module._iter_model_batches

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
        train_module, "_iter_model_batches", traced_batches(continuous_order)
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
        "_iter_model_batches",
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
        train_module, "_iter_model_batches", traced_batches(resumed_order)
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


_RESUME_REJECTION_CASES = (
    "model-config",
    "learning-rate",
    "epoch-count",
    "dataset-capture",
    "validation-identity",
    "dependencies",
    "preparation-version",
    "best-path",
    "best-loaded",
    "policy-path",
    "policy-loaded",
)


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


@pytest.mark.parametrize("case", _RESUME_REJECTION_CASES)
def test_resume_rejects_named_incompatibility(resume_baseline, tmp_path, case):
    baseline = resume_baseline
    options = dict(
        train_dataset=baseline.train,
        validation_dataset=baseline.validation,
        model_config=baseline.model_config,
        training_config=baseline.training_config,
        output_dir=tmp_path / case,
        device="cpu",
    )
    source = baseline.last
    expected = ResumeError
    message = None
    if case == "model-config":
        options["model_config"] = ModelConfig(
            state_hidden=(32,), scorer_hidden=(16,)
        )
    elif case == "learning-rate":
        options["training_config"] = dataclasses.replace(
            baseline.training_config, learning_rate=0.02
        )
    elif case == "epoch-count":
        options["training_config"] = dataclasses.replace(
            baseline.training_config, epochs=2
        )
    elif case == "dataset-capture":
        options["train_dataset"] = make_dataset(
            split="train", counts=[2] * 8, seed=61, cache_id="other-dataset"
        )
        options["validation_dataset"] = make_dataset(
            split="validation", counts=[2] * 4, seed=62, cache_id="other-dataset"
        )
    elif case == "validation-identity":
        options["validation_dataset"] = make_dataset(
            split="validation", counts=[2, 2, 2], seed=63,
            cache_id="strict-resume",
        )
    elif case == "dependencies":
        options["dependencies"] = DependencyVersions(
            python="different", torch="different", numpy="different",
            cuda_build=None,
        )
    elif case == "preparation-version":
        for split in ("train", "validation"):
            dataset = getattr(baseline, split)
            options[f"{split}_dataset"] = _reidentify(
                dataset, dataclasses.replace(dataset.identity, preparation_version=2)
            )
    elif case in ("best-path", "best-loaded"):
        source = baseline.best
        message = "last"
        if case == "best-loaded":
            source = load_checkpoint(source)
    elif case in ("policy-path", "policy-loaded"):
        source = baseline.policy
        if case == "policy-loaded":
            source = load_policy_checkpoint(source)
            message = "last"
        else:
            expected = CheckpointError
            message = "recoverable training state"
    else:
        pytest.fail(f"unknown resume rejection case {case!r}")

    trainer = Trainer(**options)
    with pytest.raises(expected, match=message):
        trainer.train(resume_from=source)
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
    assert record["batch_transfer"] == {
        "mode": "sync",
        "pinned_limit_bytes": None,
    }
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
        "batch_transfer",
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


class _OutputPolicy(CandidatePolicy):
    """Custom policy whose forward output is inspected after ``__call__``."""

    def __init__(self, config, transform) -> None:
        super().__init__(config)
        self.transform = transform
        self.forward_calls = 0

    def forward(self, states, candidates, mask):
        self.forward_calls += 1
        return self.transform(super().forward(states, candidates, mask))


def _install_output_policy(monkeypatch, transform):
    created = []

    def build(config=None, *, seed=None):
        torch.manual_seed(seed)
        policy = _OutputPolicy(config, transform).float()
        policy.register_forward_pre_hook(
            lambda module, args: setattr(
                module, "hook_calls", getattr(module, "hook_calls", 0) + 1
            )
        )
        created.append(policy)
        return policy

    monkeypatch.setattr(train_module, "build_policy", build)
    return created


@pytest.mark.parametrize("offset", [float("nan"), float("inf")])
def test_training_rejects_nonfinite_model_output_before_any_update(
    tmp_path, monkeypatch, offset
):
    train, validation = learnable_pair(cache_id="nonfinite-output")
    _install_output_policy(monkeypatch, lambda logits: logits + offset)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=3, learning_rate=0.01),
        output_dir=tmp_path / "nonfinite",
        device="cpu",
    )
    with pytest.raises(DataError, match="finite"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state
    for name in TRAIN_OUTPUT_NAMES:
        assert not (tmp_path / "nonfinite" / name).exists()


@pytest.mark.parametrize(
    "transform, message",
    [
        (lambda logits: logits.double(), "logits must be a float32 tensor"),
        (lambda logits: logits.unsqueeze(-1), "logits must have shape"),
        (lambda logits: logits[:, :-1], "mask must have shape"),
    ],
)
def test_training_rejects_structural_model_output_errors(
    tmp_path, monkeypatch, transform, message
):
    train, validation = learnable_pair(cache_id="structure-output")
    _install_output_policy(monkeypatch, transform)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=3, learning_rate=0.01),
        output_dir=tmp_path / "structure",
        device="cpu",
    )
    with pytest.raises(DataError, match=message):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state


def test_training_calls_the_custom_model_through_module_hooks(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="hook-cache")
    created = _install_output_policy(monkeypatch, lambda logits: logits)
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=4, learning_rate=0.01),
        output_dir=tmp_path / "hooks",
        device="cpu",
    )
    summary = trainer.train()
    (policy,) = created
    assert summary.completed_epochs == 1
    assert policy.forward_calls > 0
    assert policy.hook_calls == policy.forward_calls


def test_training_loss_matches_the_public_reference_exactly(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="loss-equivalence")
    config = _config(epochs=3, batch_size=3, learning_rate=0.05)
    model_config = _small_model()
    summaries = []
    weights = []
    reference_calls: list[int] = []
    for name in ("trusted", "reference"):
        trainer = Trainer(
            train_dataset=train,
            validation_dataset=validation,
            model_config=model_config,
            training_config=config,
            output_dir=tmp_path / name,
            device="cpu",
        )
        with monkeypatch.context() as patch:
            if name == "reference":
                def public_reference(logits, mask, targets):
                    reference_calls.append(int(logits.shape[0]))
                    nll = policy_module.decision_nll(logits, mask, targets)
                    torch.testing.assert_close(
                        nll.mean(),
                        policy_module.masked_cross_entropy(logits, mask, targets),
                        rtol=0, atol=0,
                    )
                    return nll

                patch.setattr(
                    train_module,
                    "_decision_nll_validated",
                    public_reference,
                )
            summaries.append(trainer.train())
        assert trainer.model is not None
        weights.append(
            {key: value.clone() for key, value in trainer.model.state_dict().items()}
        )
    assert reference_calls
    assert summaries[0].to_dict() == summaries[1].to_dict()
    assert summaries[0].history[0]["train_nll"] == summaries[1].history[0]["train_nll"]
    _assert_state_dict_equal(weights[0], weights[1])


def test_batch_loss_buffering_stays_bounded_and_keeps_accumulation_order():
    losses = [torch.tensor(value, dtype=torch.float32) for value in (0.5, 1.25, 2.0, 0.75, 3.5)]
    decisions = [3, 2, 4, 1, 5]
    expected = 0.0
    for loss, count in zip(losses, decisions):
        expected += float(loss.double().item()) * count
    for chunk_size in (1, 2, 5):
        totals = train_module._EpochLossTotals(chunk_size=chunk_size)
        for loss, count in zip(losses, decisions):
            totals.add(loss, count)
            assert len(totals._losses) < chunk_size
        assert totals.decisions == sum(decisions)
        assert totals.mean() == expected / sum(decisions)


def test_epoch_history_is_identical_across_loss_chunk_sizes(tmp_path, monkeypatch):
    train, validation = learnable_pair(cache_id="chunk-size")
    config = _config(epochs=2, batch_size=2, learning_rate=0.02)
    model_config = _small_model()
    summaries = []
    for name, chunk_size in (("unchunked", 4096), ("chunked", 4)):
        trainer = Trainer(
            train_dataset=train,
            validation_dataset=validation,
            model_config=model_config,
            training_config=config,
            output_dir=tmp_path / name,
            device="cpu",
        )
        with monkeypatch.context() as patch:
            patch.setattr(train_module, "_LOSS_CHUNK_SIZE", chunk_size)
            summaries.append(trainer.train())
    assert summaries[0].to_dict() == summaries[1].to_dict()


def _mask_off_targets(batch):
    mask = batch.mask.clone()
    mask[:, 0] = False
    mask[:, 1] = True
    return dataclasses.replace(
        batch, mask=mask, targets=torch.zeros_like(batch.targets)
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda batch: dataclasses.replace(
                batch, targets=batch.targets.clone().fill_(batch.candidates.shape[1])
            ),
            "outside the padded candidate range",
        ),
        (
            lambda batch: dataclasses.replace(
                batch, targets=batch.targets.float()
            ),
            "integer tensor",
        ),
        (_mask_off_targets, "does not reference a valid candidate"),
        (
            lambda batch: dataclasses.replace(
                batch, mask=torch.zeros_like(batch.mask)
            ),
            "at least one valid candidate",
        ),
    ],
)
def test_training_rejects_malformed_fallback_batches_before_updating(
    tmp_path, monkeypatch, mutate, message
):
    train, validation = learnable_pair(cache_id="host-batches")
    real_batches = train_module._iter_model_batches

    def corrupting(dataset, indices, **kwargs):
        for batch in real_batches(dataset, indices, **kwargs):
            assert hasattr(batch, "eligible")
            yield mutate(batch)

    monkeypatch.setattr(train_module, "_iter_model_batches", corrupting)
    trainer = Trainer(
        train_dataset=_record(train),
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=3, learning_rate=0.01),
        output_dir=tmp_path / "host-batches",
        device="cpu",
    )
    with pytest.raises(DataError, match=message):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state
    for name in TRAIN_OUTPUT_NAMES:
        assert not (tmp_path / "host-batches" / name).exists()


def test_training_rejects_forced_public_batch_on_fallback(
    tmp_path, monkeypatch
):
    train, validation = learnable_pair(cache_id="fallback-forced")
    real_batches = train_module._iter_model_batches

    def corrupting(dataset, indices, **kwargs):
        for batch in real_batches(dataset, indices, **kwargs):
            assert hasattr(batch, "eligible")
            yield dataclasses.replace(
                batch, eligible=torch.zeros_like(batch.eligible)
            )

    monkeypatch.setattr(train_module, "_iter_model_batches", corrupting)
    trainer = Trainer(
        train_dataset=_record(train),
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=3, learning_rate=0.01),
        output_dir=tmp_path / "fallback-forced",
        device="cpu",
    )
    with pytest.raises(DataError, match="forced decision"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state


def test_training_rejects_forced_row_from_current_offsets(tmp_path):
    train = make_dataset(
        split="train", counts=[2], seed=131, cache_id="current-offsets"
    )
    validation = make_dataset(
        split="validation", counts=[2], seed=132, cache_id="current-offsets"
    )
    train.targets[0] = 0
    train.offsets[1] = 1  # The cached eligible index is now a forced row.
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=1, learning_rate=0.01),
        output_dir=tmp_path / "current-offsets",
        device="cpu",
    )
    with pytest.raises(DataError, match="forced decision"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state


def test_training_checks_target_legality_before_forced_rejection(tmp_path):
    train = make_dataset(
        split="train", counts=[2, 2], seed=133, cache_id="target-first"
    )
    validation = make_dataset(
        split="validation", counts=[2], seed=134, cache_id="target-first"
    )
    train.offsets[1] = 1
    assert np.diff(train.offsets).tolist() == [1, 3]
    train.targets[0] = 0  # Valid for the now-forced first row.
    train.targets[1] = 4  # Outside the current three-candidate range.
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "target-first",
        device="cpu",
    )
    with pytest.raises(DataError, match="sample target is outside"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state


def test_pruned_training_matches_full_batch_baseline_exactly(
    tmp_path, monkeypatch
):
    """Padded internal gathering preserves public collation's arithmetic."""
    train = make_dataset(
        split="train", counts=[2, 1, 3, 4, 2, 3, 2, 4],
        seed=141, cache_id="pruned-baseline"
    )
    validation = make_dataset(
        split="validation", counts=[2, 3, 2, 4],
        seed=142, cache_id="pruned-baseline"
    )
    config = _config(epochs=3, batch_size=3, learning_rate=0.03)
    model_config = _small_model()

    def reference_batches(dataset, indices, *, training, **kwargs):
        kwargs.pop("packed_candidates", None)
        return iter_batches(dataset, indices, **kwargs)

    summaries = []
    models = []
    for name, dataset, reference in (
        ("pruned", train, False),
        ("full", train, True),
    ):
        trainer = Trainer(
            train_dataset=dataset,
            validation_dataset=(
                _record(validation) if reference else validation
            ),
            model_config=model_config,
            training_config=config,
            output_dir=tmp_path / name,
            device="cpu",
        )
        with monkeypatch.context() as patch:
            for module in (train_module, metrics_module):
                patch.setattr(
                    module, "_supports_packed_candidate_scoring", lambda model: False
                )
            if reference:
                patch.setattr(
                    train_module,
                    "_iter_model_batches",
                    reference_batches,
                )
            summaries.append(trainer.train())
        assert trainer.model is not None
        models.append(
            {
                key: value.clone()
                for key, value in trainer.model.state_dict().items()
            }
        )
    assert summaries[0].to_dict() == summaries[1].to_dict()
    assert summaries[0].best_epoch == summaries[1].best_epoch
    assert summaries[0].global_step == summaries[1].global_step
    assert summaries[0].samples_seen == summaries[1].samples_seen
    _assert_state_dict_equal(models[0], models[1])
    first_best = load_checkpoint(
        tmp_path / "pruned" / "best.pt", require_training=True
    )
    second_best = load_checkpoint(
        tmp_path / "full" / "best.pt", require_training=True
    )
    assert first_best.training.best_epoch == second_best.training.best_epoch
    assert first_best.metrics == second_best.metrics
    _assert_state_dict_equal(
        first_best.model.state_dict(), second_best.model.state_dict()
    )
    first_last = load_checkpoint(
        tmp_path / "pruned" / "last.pt", require_training=True
    )
    second_last = load_checkpoint(
        tmp_path / "full" / "last.pt", require_training=True
    )
    assert first_last.training.global_step == second_last.training.global_step
    assert first_last.training.samples_seen == second_last.training.samples_seen
    _assert_state_dict_equal(
        first_last.model.state_dict(), second_last.model.state_dict()
    )
    _assert_state_tree_exact(
        first_last.training.optimizer_state,
        second_last.training.optimizer_state,
    )


def test_pruned_training_recovery_replays_exactly(tmp_path, monkeypatch):
    train, validation = learnable_pair(
        train_decisions=12, validation_decisions=6, cache_id="pruned-replay"
    )
    config = _config(epochs=4, batch_size=3, learning_rate=0.01)
    model = _small_model()
    real_batches = train_module._iter_model_batches
    continuous_order: list[list[int]] = []
    resumed_order: list[list[int]] = []

    def traced_batches(trace, *, interrupt=False):
        def iterator(dataset, indices, **kwargs):
            trace.append([int(index) for index in indices])
            for number, batch in enumerate(
                real_batches(dataset, indices, **kwargs)
            ):
                assert isinstance(batch, _PackedCandidateBatch)
                yield batch
                if interrupt and len(trace) == 2 and number == 0:
                    raise KeyboardInterrupt("injected pruned mid-epoch stop")

        return iterator

    continuous = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "continuous",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "_iter_model_batches", traced_batches(continuous_order)
    )
    expected = continuous.train()

    interrupted = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "interrupted",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module,
        "_iter_model_batches",
        traced_batches([], interrupt=True),
    )
    with pytest.raises(KeyboardInterrupt, match="injected"):
        interrupted.train()

    resumed = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "resumed",
        device="cpu",
    )
    monkeypatch.setattr(
        train_module, "_iter_model_batches", traced_batches(resumed_order)
    )
    actual = resumed.train(resume_from=tmp_path / "interrupted" / "last.pt")
    assert continuous_order[1:] == resumed_order
    assert expected.to_dict() == actual.to_dict()
    expected_last = load_checkpoint(
        tmp_path / "continuous" / "last.pt", require_training=True
    )
    actual_last = load_checkpoint(
        tmp_path / "resumed" / "last.pt", require_training=True
    )
    _assert_state_dict_equal(
        expected_last.model.state_dict(), actual_last.model.state_dict()
    )
    _assert_state_tree_exact(
        expected_last.training.optimizer_state,
        actual_last.training.optimizer_state,
    )



def test_training_missing_target_kinds_keeps_public_batch_fallback(
    tmp_path, monkeypatch
):
    train_source = make_dataset(
        split="train", counts=[2, 2, 2, 2], seed=151, cache_id="no-kinds"
    )
    validation_source = make_dataset(
        split="validation", counts=[2, 2], seed=152, cache_id="no-kinds"
    )

    def without_kinds(source):
        return PackedDecisionDataset(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=source.identity,
        )

    trainer = Trainer(
        train_dataset=without_kinds(train_source),
        validation_dataset=without_kinds(validation_source),
        model_config=_small_model(),
        training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
        output_dir=tmp_path / "no-kinds",
        device="cpu",
    )
    real_batches = train_module._iter_model_batches
    public_batches: list[bool] = []

    def recording(dataset, indices, **kwargs):
        for batch in real_batches(dataset, indices, **kwargs):
            public_batches.append(hasattr(batch, "eligible"))
            yield batch

    monkeypatch.setattr(train_module, "_iter_model_batches", recording)
    summary = trainer.train()
    assert summary.completed_epochs == 1
    assert summary.samples_seen == 4
    assert public_batches and all(public_batches)


# ---- optional batch transfer ---------------------------------------------

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
_MIB = 2**20
_STAGED_ARRAY_NAMES = (
    "states",
    "candidate_features",
    "offsets",
    "targets",
    "stages",
    "target_kinds",
)


def _freeze(dataset: PackedDecisionDataset) -> PackedDecisionDataset:
    """Make a fixture's consumed inputs read-only, like a loaded split."""

    for name in _STAGED_ARRAY_NAMES:
        array = getattr(dataset, name)
        if array is not None:
            array.setflags(write=False)
    return dataset


def _prefetch(pinned_mib: int = 8) -> BatchTransferConfig:
    return BatchTransferConfig(
        mode="cuda-prefetch", pinned_limit_bytes=pinned_mib * _MIB
    )


def _trainer(
    tmp_path: Path,
    train: PackedDecisionDataset,
    validation: PackedDecisionDataset,
    *,
    name: str,
    config: TrainingConfig | None = None,
    device: str = "cpu",
    batch_transfer: BatchTransferConfig | None = None,
) -> Trainer:
    return Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=_small_model(),
        training_config=_config(epochs=2, batch_size=3, learning_rate=0.01)
        if config is None
        else config,
        output_dir=tmp_path / name,
        device=device,
        batch_transfer=batch_transfer,
    )


def _saved_state(trainer: Trainer) -> dict[str, torch.Tensor]:
    assert trainer.model is not None
    return {
        name: tensor.detach().clone()
        for name, tensor in trainer.model.state_dict().items()
    }


def test_sync_transfer_config_matches_the_default_exactly(tmp_path):
    train, validation = learnable_pair(cache_id="sync-equality")
    summaries = []
    states = []
    for name, transfer in (("default", None), ("sync", BatchTransferConfig())):
        trainer = _trainer(
            tmp_path, train, validation, name=name, batch_transfer=transfer
        )
        summaries.append(trainer.train())
        states.append(_saved_state(trainer))
    assert summaries[0].to_dict() == summaries[1].to_dict()
    _assert_state_dict_equal(states[0], states[1])
    for name in ("default", "sync"):
        record = json.loads(
            (tmp_path / name / "run.json").read_text(encoding="utf-8")
        )
        assert record["batch_transfer"] == {
            "mode": "sync",
            "pinned_limit_bytes": None,
        }
        # Supply choices never enter the training identity.
        assert set(record["training_config"]) == set(
            TrainingConfig(
                seed=1, epochs=1, batch_size=1, learning_rate=0.01
            ).to_dict()
        )


def test_prefetch_requires_cuda_before_any_update(tmp_path):
    train, validation = learnable_pair(cache_id="prefetch-cpu")
    trainer = _trainer(
        tmp_path,
        _freeze(train),
        _freeze(validation),
        name="cpu-prefetch",
        device="cpu",
        batch_transfer=_prefetch(),
    )
    with pytest.raises(ConfigurationError, match="CUDA device"):
        trainer.train()
    assert trainer.optimizer is not None and not trainer.optimizer.state
    for name in TRAIN_OUTPUT_NAMES:
        assert not (tmp_path / "cpu-prefetch" / name).exists()


@CUDA
def test_prefetch_rejects_unsupported_inputs_before_any_update(tmp_path):
    train_source, validation_source = learnable_pair(cache_id="prefetch-guard")
    train = _freeze(train_source)
    validation = _freeze(validation_source)
    cases = [
        (
            "workers",
            train,
            validation,
            _config(num_workers=2),
            "num_workers=0",
        ),
        (
            "writable",
            learnable_pair(cache_id="prefetch-guard")[0],
            validation,
            _config(),
            "read-only input arrays",
        ),
        (
            "custom-dataset",
            _record(train),
            validation,
            _config(),
            "requires the built-in",
        ),
        (
            # A subclass validation split keeps the full traversal, so it
            # cannot provide the metadata this staging path requires.
            "custom-validation",
            train,
            _record(validation),
            _config(),
            "eligible-only traversal",
        ),
    ]
    for name, train_dataset, validation_dataset, config, message in cases:
        trainer = _trainer(
            tmp_path,
            train_dataset,
            validation_dataset,
            name=name,
            config=config,
            device="cuda",
            batch_transfer=_prefetch(),
        )
        with pytest.raises(ConfigurationError, match=message):
            trainer.train()
        assert trainer.optimizer is not None and not trainer.optimizer.state
        for output in TRAIN_OUTPUT_NAMES:
            assert not (tmp_path / name / output).exists(), (name, output)


def test_trainer_rejects_a_non_config_transfer(tmp_path):
    train, validation = learnable_pair(cache_id="transfer-type")
    with pytest.raises(ConfigurationError, match="BatchTransferConfig"):
        _trainer(
            tmp_path,
            train,
            validation,
            name="bad-transfer",
            batch_transfer="sync",  # type: ignore[arg-type]
        )


@CUDA
def test_prefetch_training_matches_sync_exactly(tmp_path):
    train, validation = learnable_pair(cache_id="prefetch-equality")
    _freeze(train)
    _freeze(validation)
    config = _config(epochs=2, batch_size=3, learning_rate=0.01)
    summaries = {}
    states = {}
    last_states = {}
    for mode, transfer in (
        ("sync", BatchTransferConfig()),
        ("cuda-prefetch", _prefetch()),
    ):
        trainer = _trainer(
            tmp_path,
            train,
            validation,
            name=mode,
            config=config,
            device="cuda",
            batch_transfer=transfer,
        )
        summaries[mode] = trainer.train()
        states[mode] = _saved_state(trainer)
        last = load_checkpoint(
            tmp_path / mode / "last.pt", require_training=True
        )
        last_states[mode] = last.model.state_dict()
        record = json.loads(
            (tmp_path / mode / "run.json").read_text(encoding="utf-8")
        )
        assert record["batch_transfer"] == transfer.to_dict()
    assert summaries["sync"].to_dict() == summaries["cuda-prefetch"].to_dict()
    _assert_state_dict_equal(states["sync"], states["cuda-prefetch"])
    _assert_state_dict_equal(last_states["sync"], last_states["cuda-prefetch"])


@CUDA
def test_prefetch_interruption_and_resume_replay_exactly(tmp_path, monkeypatch):
    train, validation = learnable_pair(
        train_decisions=12, validation_decisions=6, cache_id="prefetch-recovery"
    )
    _freeze(train)
    _freeze(validation)
    config = _config(epochs=3, batch_size=2, learning_rate=0.01)
    transfer = _prefetch()

    continuous = _trainer(
        tmp_path,
        train,
        validation,
        name="continuous",
        config=config,
        device="cuda",
        batch_transfer=transfer,
    )
    continuous_summary = continuous.train()
    continuous_state = _saved_state(continuous)
    assert continuous.optimizer is not None
    continuous_optimizer = continuous.optimizer.state_dict()

    # Interrupt the same run after the first step of the second epoch: the
    # first epoch is committed, the failed epoch must not publish.
    real_add = train_module._EpochLossTotals.add
    calls = {"count": 0}

    def failing_add(self, loss, decisions):
        calls["count"] += 1
        if calls["count"] == 7:
            raise RuntimeError("injected mid-epoch stop")
        return real_add(self, loss, decisions)

    interrupted = _trainer(
        tmp_path,
        train,
        validation,
        name="resumed",
        config=config,
        device="cuda",
        batch_transfer=transfer,
    )
    with monkeypatch.context() as patch:
        patch.setattr(train_module._EpochLossTotals, "add", failing_add)
        with pytest.raises(RuntimeError, match="injected mid-epoch stop"):
            interrupted.train()
    committed = load_checkpoint(
        tmp_path / "resumed" / "last.pt", require_training=True
    )
    assert committed.training.completed_epochs == 1
    record = json.loads(
        (tmp_path / "resumed" / "run.json").read_text(encoding="utf-8")
    )
    assert record["summary"]["completed_epochs"] == 1
    assert record["batch_transfer"] == transfer.to_dict()

    resumed = _trainer(
        tmp_path,
        train,
        validation,
        name="resumed",
        config=config,
        device="cuda",
        batch_transfer=transfer,
    )
    resumed_summary = resumed.train(resume_from=tmp_path / "resumed" / "last.pt")
    assert resumed_summary.to_dict() == continuous_summary.to_dict()
    _assert_state_dict_equal(_saved_state(resumed), continuous_state)
    assert resumed.optimizer is not None
    _assert_state_tree_exact(
        resumed.optimizer.state_dict(), continuous_optimizer
    )
    final_record = json.loads(
        (tmp_path / "resumed" / "run.json").read_text(encoding="utf-8")
    )
    assert final_record["summary"] == json.loads(
        (tmp_path / "continuous" / "run.json").read_text(encoding="utf-8")
    )["summary"]


@CUDA
def test_evaluate_checkpoint_accepts_a_prefetch_request(tmp_path):
    train, validation = learnable_pair(cache_id="eval-transfer")
    _freeze(train)
    _freeze(validation)
    trainer = _trainer(
        tmp_path,
        train,
        validation,
        name="frozen",
        config=_config(epochs=1, batch_size=2, learning_rate=0.01),
        device="cuda",
    )
    trainer.train()
    best = tmp_path / "frozen" / "best.pt"
    test = _freeze(
        make_dataset(
            split="test", counts=[2, 1, 3, 2, 2, 1], seed=91, cache_id="eval-transfer"
        )
    )
    sync_report = evaluate_checkpoint(
        best, test, batch_size=2, device="cuda"
    )
    prefetch_report = evaluate_checkpoint(
        best,
        test,
        batch_size=2,
        device="cuda",
        batch_transfer=_prefetch(),
    )
    assert prefetch_report == sync_report
    assert prefetch_report.decisions == len(test)
    assert prefetch_report.forced == test.forced_count
    with pytest.raises(ConfigurationError, match="CUDA device"):
        evaluate_checkpoint(
            best,
            test,
            batch_size=2,
            device="cpu",
            batch_transfer=_prefetch(),
        )
