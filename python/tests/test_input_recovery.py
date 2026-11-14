"""Exact epoch replay while reselecting independent input execution options."""

from __future__ import annotations

import json

import pytest
import torch

import mahjong_bc.train as training
from mahjong_bc import BatchTransferConfig, ModelConfig, Trainer, TrainingConfig
from mahjong_bc.cache import load_prepared_cache
from mahjong_bc.checkpoint import load_checkpoint


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact(a, b)
    else:
        assert left == right


@CUDA
@pytest.mark.parametrize("storage", ["mmap", "ram"])
@pytest.mark.parametrize("transfer", ["sync", "cuda-prefetch"])
def test_epoch_replay_is_exact_after_reselecting_supply_options(
    shared_cache, tmp_path, monkeypatch, storage, transfer
):
    cache = load_prepared_cache(shared_cache.root)
    config = TrainingConfig(seed=7, epochs=2, batch_size=2, learning_rate=0.02)
    model_config = ModelConfig(state_hidden=(8,), scorer_hidden=(8,))
    original_factory = training._iter_model_batches

    class HostFailure(RuntimeError):
        pass

    def factory(trace, *, fail=False):
        def batches(dataset, indices, **kwargs):
            trace.append(tuple(int(value) for value in indices))
            for number, batch in enumerate(original_factory(dataset, indices, **kwargs)):
                # Fail while gathering a future batch, after the prior batch
                # can be delivered; producer run-ahead must not reorder it.
                if fail and len(trace) == 2 and number == 1:
                    raise HostFailure("future host batch failed in epoch two")
                yield batch
        return batches

    def execute(directory, storage, transfer, trace, *, fail=False, resume=None):
        with cache.open_splits(
            ("train", "validation"), storage=storage,
            ram_limit_bytes=8 * 2**20 if storage == "ram" else None,
        ) as inputs:
            trainer = Trainer(
                train_dataset=inputs.datasets["train"],
                validation_dataset=inputs.datasets["validation"],
                model_config=model_config, training_config=config,
                output_dir=directory, device="cuda",
                dataset_provenance={"input_loading": inputs.loading_record},
                batch_transfer=BatchTransferConfig(
                    mode=transfer,
                    pinned_limit_bytes=8 * 2**20 if transfer == "cuda-prefetch" else None,
                ),
            )
            with monkeypatch.context() as patch:
                patch.setattr(training, "_iter_model_batches", factory(trace, fail=fail))
                return trainer.train(resume_from=resume)

    continuous_order, interrupted_order, replayed_order = [], [], []
    expected_dir, actual_dir = tmp_path / "continuous", tmp_path / "replayed"
    expected = execute(expected_dir, storage, transfer, continuous_order)
    with pytest.raises(HostFailure, match="future host batch failed"):
        execute(actual_dir, storage, transfer, interrupted_order, fail=True)
    boundary = load_checkpoint(actual_dir / "last.pt", require_training=True)
    assert boundary.training.completed_epochs == 1
    assert boundary.training.global_step == (cache.splits["train"].counts.eligible_decisions + 1) // 2
    record = json.loads((actual_dir / "run.json").read_text(encoding="utf-8"))
    assert record["summary"]["completed_epochs"] == 1

    # Resume on the same device with both execution choices changed, without
    # treating either choice as a training-configuration override.
    resumed_storage = "ram" if storage == "mmap" else "mmap"
    resumed_transfer = "cuda-prefetch" if transfer == "sync" else "sync"
    actual = execute(
        actual_dir, resumed_storage, resumed_transfer, replayed_order,
        resume=actual_dir / "last.pt",
    )
    assert expected.to_dict() == actual.to_dict()
    assert continuous_order == interrupted_order
    assert continuous_order[1:] == replayed_order
    expected_last = load_checkpoint(expected_dir / "last.pt", require_training=True)
    actual_last = load_checkpoint(actual_dir / "last.pt", require_training=True)
    _assert_exact(expected_last.model.state_dict(), actual_last.model.state_dict())
    _assert_exact(expected_last.training.to_payload(), actual_last.training.to_payload())
    _assert_exact(
        load_checkpoint(expected_dir / "best.pt").model.state_dict(),
        load_checkpoint(actual_dir / "best.pt").model.state_dict(),
    )
    current = json.loads((actual_dir / "run.json").read_text(encoding="utf-8"))
    assert current["dataset"]["provenance"]["input_loading"]["storage"] == resumed_storage
    assert current["batch_transfer"]["mode"] == resumed_transfer
