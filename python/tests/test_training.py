from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mahjong_bc.train as train_module
from mahjong_bc.checkpoint import load_checkpoint
from mahjong_bc.config import (
    DatasetIdentity,
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
)
from mahjong_bc.constants import ROLE_BEST, ROLE_LAST
from mahjong_bc.data import PackedDecisionDataset
from mahjong_bc.errors import ConfigurationError, DataError, ResumeError
from mahjong_bc.train import Trainer, evaluate_checkpoint

from .helpers import learnable_pair, make_dataset


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


def _assert_tensor_tree_close(left: Any, right: Any, *, path: str = "") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_tensor_tree_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_tensor_tree_close(first, second, path=f"{path}[{index}]")
        return
    if isinstance(left, float):
        assert right == pytest.approx(left, abs=1e-6), path
        return
    assert left == right, path


def _summary_metrics(summary) -> list[float]:
    return [entry["validation"]["nll"] for entry in summary.history]


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
    loaded_last = load_checkpoint(
        tmp_path / "run" / "last.pt", require_training=True
    )
    assert loaded_last.role == ROLE_LAST
    assert loaded_last.training.completed_epochs == 40
    loaded_best = load_checkpoint(
        tmp_path / "run" / "best.pt", require_training=True
    )
    assert loaded_best.metrics["nll"] == pytest.approx(summary.best_validation_nll)


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
    assert train.visited[:7].count(eligible[0]) == 1
    assert len(train.visited) == 14
    for epoch in range(2):
        visited = train.visited[epoch * 7 : (epoch + 1) * 7]
        assert sorted(visited) == sorted(int(index) for index in eligible)
        assert len(set(visited)) == 7
        assert summary.history[epoch]["steps"] == 3
        assert summary.history[epoch]["samples"] == 7
    assert summary.global_step == 6
    assert summary.samples_seen == 14
    assert not any(int(index) in set(int(i) for i in range(len(train_source))) - set(int(i) for i in eligible) for index in train.visited)


def test_epoch_boundary_resume_matches_uninterrupted_run_exactly(tmp_path, monkeypatch):
    base_train, base_validation = learnable_pair(
        train_decisions=12, validation_decisions=6, cache_id="resume-cache"
    )
    first_train = _record(base_train)
    second_train = _record(base_train)
    config = _config(epochs=4, batch_size=3, learning_rate=0.01)
    model = _small_model()

    uninterrupted = Trainer(
        train_dataset=first_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "uninterrupted",
        device="cpu",
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
    original_write = Trainer._write_run_record

    def stop_after_first_epoch(self, summary):
        original_write(self, summary)
        if summary.completed_epochs == 1:
            raise KeyboardInterrupt("injected epoch-boundary stop")

    monkeypatch.setattr(Trainer, "_write_run_record", stop_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        interrupted.train()
    monkeypatch.setattr(Trainer, "_write_run_record", original_write)
    assert len(second_train.visited) == base_train.eligible_count
    first_epoch_order = list(second_train.visited)

    resumed = Trainer(
        train_dataset=second_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "resumed-output",
        device="cpu",
    )
    summary_b = resumed.train(resume_from=tmp_path / "interrupted" / "last.pt")

    assert second_train.visited == first_train.visited
    assert second_train.visited[: base_train.eligible_count] == first_epoch_order
    assert summary_b.completed_epochs == summary_a.completed_epochs == 4
    assert summary_b.global_step == summary_a.global_step
    assert summary_b.samples_seen == summary_a.samples_seen
    assert summary_b.best_epoch == summary_a.best_epoch
    assert summary_b.best_validation_nll == pytest.approx(
        summary_a.best_validation_nll
    )
    assert len(summary_b.history) == len(summary_a.history) == 4
    for entry_a, entry_b in zip(summary_a.history, summary_b.history):
        assert entry_a["steps"] == entry_b["steps"]
        assert entry_a["samples"] == entry_b["samples"]
        assert entry_a["train_nll"] == pytest.approx(entry_b["train_nll"], abs=1e-6)
        assert entry_a["validation"]["nll"] == pytest.approx(
            entry_b["validation"]["nll"], abs=1e-6
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
    _assert_tensor_tree_close(
        loaded_a.model.state_dict(), loaded_b.model.state_dict()
    )
    _assert_tensor_tree_close(
        loaded_a.training.optimizer_state, loaded_b.training.optimizer_state
    )
    assert torch.equal(
        loaded_a.training.torch_rng_state, loaded_b.training.torch_rng_state
    )
    assert torch.equal(
        loaded_a.training.order_rng_state, loaded_b.training.order_rng_state
    )


def test_resume_rejects_configuration_identity_dependency_and_role_mismatches(
    tmp_path,
):
    base_train, base_validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="strict-resume"
    )
    config = _config(epochs=1, batch_size=2, learning_rate=0.01)
    model = _small_model()
    base = Trainer(
        train_dataset=base_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "base",
        device="cpu",
    )
    base.train()
    last = tmp_path / "base" / "last.pt"
    best = tmp_path / "base" / "best.pt"

    cases = []
    cases.append(
        Trainer(
            train_dataset=base_train,
            validation_dataset=base_validation,
            model_config=ModelConfig(state_hidden=(32,), scorer_hidden=(16,)),
            training_config=config,
            output_dir=tmp_path / "model-mismatch",
            device="cpu",
        )
    )
    cases.append(
        Trainer(
            train_dataset=base_train,
            validation_dataset=base_validation,
            model_config=model,
            training_config=dataclasses.replace(config, learning_rate=0.02),
            output_dir=tmp_path / "config-mismatch",
            device="cpu",
        )
    )
    other_train = make_dataset(
        split="train", counts=[2] * 8, seed=61, cache_id="other-dataset"
    )
    other_validation = make_dataset(
        split="validation", counts=[2] * 4, seed=62, cache_id="other-dataset"
    )
    cases.append(
        Trainer(
            train_dataset=other_train,
            validation_dataset=other_validation,
            model_config=model,
            training_config=config,
            output_dir=tmp_path / "data-mismatch",
            device="cpu",
        )
    )
    other_validation_only = make_dataset(
        split="validation", counts=[2, 2, 2], seed=63, cache_id="strict-resume"
    )
    cases.append(
        Trainer(
            train_dataset=base_train,
            validation_dataset=other_validation_only,
            model_config=model,
            training_config=config,
            output_dir=tmp_path / "validation-mismatch",
            device="cpu",
        )
    )
    cases.append(
        Trainer(
            train_dataset=base_train,
            validation_dataset=base_validation,
            model_config=model,
            training_config=config,
            output_dir=tmp_path / "dependency-mismatch",
            device="cpu",
            dependencies=DependencyVersions(
                python="different",
                torch="different",
                numpy="different",
                cuda_build=None,
            ),
        )
    )
    prep_versions = (
        dataclasses.replace(base_train.identity, preparation_version=2),
        dataclasses.replace(base_validation.identity, preparation_version=2),
    )
    prep_train = _reidentify(base_train, prep_versions[0])
    prep_validation = _reidentify(base_validation, prep_versions[1])
    cases.append(
        Trainer(
            train_dataset=prep_train,
            validation_dataset=prep_validation,
            model_config=model,
            training_config=config,
            output_dir=tmp_path / "preparation-mismatch",
            device="cpu",
        )
    )
    for index, trainer in enumerate(cases):
        with pytest.raises(ResumeError):
            trainer.train(resume_from=last)

    best_role = Trainer(
        train_dataset=base_train,
        validation_dataset=base_validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "best-role",
        device="cpu",
    )
    with pytest.raises(ResumeError):
        best_role.train(resume_from=best)


def _reidentify(source: PackedDecisionDataset, new_identity: DatasetIdentity):
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


def test_validation_only_best_selection_and_frozen_test_evaluation(tmp_path, monkeypatch):
    train, validation = learnable_pair(
        train_decisions=8, validation_decisions=4, cache_id="selection"
    )
    test = make_dataset(
        split="test", counts=[2, 2, 1], seed=71, cache_id="selection"
    )
    config = _config(epochs=4, batch_size=2, learning_rate=0.01)
    model = _small_model()
    trainer = Trainer(
        train_dataset=train,
        validation_dataset=validation,
        model_config=model,
        training_config=config,
        output_dir=tmp_path / "selection",
        device="cpu",
    )
    real_evaluate = train_module.evaluate_dataset
    injected = iter([0.5, 0.6, 0.4, 0.7])

    def fake_evaluate(model, dataset, **kwargs):
        report = real_evaluate(model, dataset, **kwargs)
        return dataclasses.replace(report, nll=next(injected))

    monkeypatch.setattr(train_module, "evaluate_dataset", fake_evaluate)
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
    assert loaded_best.training.best_epoch == 2
    assert loaded_best.metrics["nll"] == pytest.approx(0.4)
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


def test_from_scratch_refuses_existing_run_artifacts_without_deleting(tmp_path):
    train, validation = learnable_pair()
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "last.pt"
    sentinel.write_bytes(b"existing user checkpoint")
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
    assert sentinel.read_bytes() == b"existing user checkpoint"


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


def test_trainer_and_evaluation_reject_datasets_without_split_identity(tmp_path):
    source, validation = learnable_pair()
    anonymous = PackedDecisionDataset(
        states=source.states,
        candidate_features=source.candidate_features,
        offsets=source.offsets,
        targets=source.targets,
        stages=source.stages,
        identity=None,
    )
    with pytest.raises(DataError):
        Trainer(
            train_dataset=anonymous,
            validation_dataset=validation,
            model_config=_small_model(),
            training_config=_config(epochs=1, batch_size=2, learning_rate=0.01),
            output_dir=tmp_path / "anonymous",
            device="cpu",
        )
    from mahjong_bc.metrics import evaluate_dataset

    with pytest.raises(DataError):
        evaluate_dataset(
            __import__("mahjong_bc").build_policy(_small_model()),
            anonymous,
            batch_size=2,
            device="cpu",
        )


def test_training_stops_safely_when_last_checkpoint_write_is_interrupted(tmp_path, monkeypatch):
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
    real_save = train_module.save_checkpoint

    def fail_last_after_first_epoch(path, model, **kwargs):
        if kwargs.get("role") == ROLE_LAST and Path(path).exists():
            raise RuntimeError("injected interrupted last checkpoint write")
        return real_save(path, model, **kwargs)

    monkeypatch.setattr(train_module, "save_checkpoint", fail_last_after_first_epoch)
    with pytest.raises(RuntimeError):
        trainer.train()
    loaded = load_checkpoint(
        tmp_path / "write-failure" / "last.pt", require_training=True
    )
    assert loaded.role == ROLE_LAST
    assert loaded.training.completed_epochs == 1


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
        tmp_path / "cuda" / "last.pt", device="cuda", require_training=True
    )
    assert loaded.training.device_type == "cuda"
    report = evaluate_checkpoint(
        tmp_path / "cuda" / "best.pt", test, batch_size=2, device="cuda"
    )
    assert report.eligible == test.eligible_count


def _install_validation_sequence(monkeypatch, values):
    real_evaluate = train_module.evaluate_dataset
    sequence = iter(values)

    def fake_evaluate(model, dataset, **kwargs):
        report = real_evaluate(model, dataset, **kwargs)
        return dataclasses.replace(report, nll=next(sequence))

    monkeypatch.setattr(train_module, "evaluate_dataset", fake_evaluate)
    return real_evaluate


def _assert_state_dict_equal(left, right):
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


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
    assert recorder.visited == []
    assert resumed_summary.completed_epochs == 2
    assert resumed_summary.global_step == summary.global_step
    assert resumed_summary.samples_seen == summary.samples_seen
    new_best = load_checkpoint(new / "best.pt", require_training=True)
    new_last = load_checkpoint(new / "last.pt", require_training=True)
    _assert_state_dict_equal(new_best.model.state_dict(), best_weights)
    _assert_state_dict_equal(new_last.model.state_dict(), last_weights)
    assert new_best.metrics["nll"] == pytest.approx(0.5)
    assert new_last.metrics["nll"] == pytest.approx(0.6)
    assert (new / "run.json").is_file()
