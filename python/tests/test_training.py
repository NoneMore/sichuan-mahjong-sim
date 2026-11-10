from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mahjong_bc.train as train_module
from mahjong_bc.checkpoint import load_policy_checkpoint
from mahjong_bc.config import (
    DependencyVersions,
    ModelConfig,
    TrainingConfig,
)
from mahjong_bc.data import PackedDecisionDataset
from mahjong_bc.errors import ConfigurationError, DataError
from mahjong_bc.train import Trainer, evaluate_checkpoint, resolve_device

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
    assert (tmp_path / "run" / "run.json").is_file()
    assert not (tmp_path / "run" / "last.pt").exists()

    loaded = load_policy_checkpoint(tmp_path / "run" / "best.pt")
    assert loaded.role == "best"
    assert loaded.metadata()["has_training_state"] is False
    assert loaded.metrics["nll"] == pytest.approx(summary.best_validation_nll)


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
    loaded_best = load_policy_checkpoint(tmp_path / "selection" / "best.pt")
    assert loaded_best.role == "best"
    assert loaded_best.metrics["nll"] == pytest.approx(0.4)

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
    assert not (tmp_path / "selection" / "last.pt").exists()


@pytest.mark.parametrize("protected_name", ["best.pt", "last.pt", "run.json"])
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


def test_training_stops_safely_when_best_checkpoint_write_is_interrupted(
    tmp_path, monkeypatch
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

    def fail_second_save(path, model, **kwargs):
        calls.append(Path(path))
        if len(calls) == 2:
            raise RuntimeError("injected interrupted best checkpoint write")
        return real_save(path, model, **kwargs)

    monkeypatch.setattr(train_module, "save_checkpoint", fail_second_save)
    with pytest.raises(RuntimeError, match="injected"):
        trainer.train()

    loaded = load_policy_checkpoint(tmp_path / "write-failure" / "best.pt")
    assert loaded.role == "best"
    assert loaded.metrics["nll"] == pytest.approx(0.5)
    record = json.loads(
        (tmp_path / "write-failure" / "run.json").read_text(encoding="utf-8")
    )
    assert record["summary"]["completed_epochs"] == 1
    assert record["summary"]["best_validation_nll"] == pytest.approx(0.5)
    assert not (tmp_path / "write-failure" / "last.pt").exists()


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


def test_resolve_device_rejects_unsupported_or_unavailable_devices():
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ConfigurationError, match="unsupported device"):
        resolve_device("mps")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigurationError, match="not available"):
            resolve_device("cuda")


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
    loaded = load_policy_checkpoint(
        tmp_path / "cuda" / "best.pt", device="cuda"
    )
    assert loaded.role == "best"
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
