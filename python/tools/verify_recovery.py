"""Probe epoch replay and directory migration on a real prepared cache.

Run after installing python/; outputs must be outside the cache/source tree:
    python python/tools/verify_recovery.py --dataset ../bc-cache --output ../bc-recovery
The output directory must not exist. This bounded acceptance probe uses one CPU
thread and injects an exception after the first batch of the second epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from mahjong_bc import ModelConfig, Trainer, TrainingConfig, load_prepared_cache
from mahjong_bc.checkpoint import load_checkpoint
from mahjong_bc.train import evaluate_checkpoint
import mahjong_bc.train as training


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        # Fixed CPU runtime: explicitly require exact model/optimizer equality.
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def verify(dataset_root: Path, output: Path) -> dict:
    # Do not create probe outputs in immutable cache inventory.
    root = dataset_root.resolve()
    destination = output.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("output must be outside the prepared cache")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cache = load_prepared_cache(dataset_root)
    train = cache.load_split("train")
    validation = cache.load_split("validation")
    test = cache.load_split("test")
    config = TrainingConfig(seed=7, epochs=3, batch_size=17, learning_rate=0.01)
    model_config = ModelConfig(state_hidden=(16,), scorer_hidden=(16,))
    original_batches = training.iter_batches

    def trainer(name):
        return Trainer(
            train_dataset=train,
            validation_dataset=validation,
            training_config=config,
            model_config=model_config,
            output_dir=output / name,
            device="cpu",
            dataset_provenance=cache.provenance,
        )

    class InterruptedEpoch(RuntimeError):
        pass

    def batches(trace, *, interrupt=False):
        def iterator(dataset, indices, **kwargs):
            trace.append([int(index) for index in indices])
            for number, batch in enumerate(original_batches(dataset, indices, **kwargs)):
                if interrupt and len(trace) == 2 and number == 1:
                    raise InterruptedEpoch("injected after one batch of epoch 1")
                yield batch
        return iterator

    try:
        continuous_order, interrupted_order, resumed_order = [], [], []
        continuous = trainer("continuous")
        with patch.object(training, "iter_batches", batches(continuous_order)):
            expected = continuous.train()
        interrupted = trainer("interrupted")
        try:
            with patch.object(
                training, "iter_batches", batches(interrupted_order, interrupt=True)
            ):
                interrupted.train()
        except InterruptedEpoch:
            pass
        else:
            raise AssertionError("interruption was not reached")
        source = output / "interrupted" / "last.pt"
        boundary = load_checkpoint(source, require_training=True)
        assert boundary.training.completed_epochs == 1
        resumed = trainer("resumed")
        with patch.object(training, "iter_batches", batches(resumed_order)):
            actual = resumed.train(resume_from=source)
        assert continuous_order[:2] == interrupted_order
        assert continuous_order[1:] == resumed_order
        assert expected.to_dict() == actual.to_dict()
        expected_last = load_checkpoint(continuous.last_path, require_training=True)
        actual_last = load_checkpoint(resumed.last_path, require_training=True)
        _assert_equal(expected_last.model.state_dict(), actual_last.model.state_dict())
        _assert_equal(
            expected_last.training.optimizer_state, actual_last.training.optimizer_state
        )
        # A completed run must still republish both artifacts to a fresh directory.
        migrated = trainer("completed-migration")
        completed = migrated.train(resume_from=resumed.last_path)
        assert completed.to_dict() == actual.to_dict()
        _assert_equal(
            load_checkpoint(resumed.best_path).model.state_dict(),
            load_checkpoint(migrated.best_path).model.state_dict(),
        )
        assert migrated.last_path.is_file()
        best_bytes = migrated.best_path.read_bytes()
        evaluation = evaluate_checkpoint(migrated.best_path, test, batch_size=17)
        assert migrated.best_path.read_bytes() == best_bytes
        result = {
            "cache_id": cache.cache_id,
            "corpus_manifest_sha256": cache.corpus_manifest_sha256,
            "source": resumed.source,
            "dependencies": resumed.dependencies.to_dict(),
            "device_type": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "tolerance": {"rtol": 0, "atol": 0},
            "interrupted_completed_epochs": boundary.training.completed_epochs,
            "sample_order_exact": True,
            "model_optimizer_exact": True,
            "completed_migration": True,
            "summary": actual.to_dict(),
            "frozen_test": evaluation.to_dict(),
            "best_sha256": hashlib.sha256(best_bytes).hexdigest(),
        }
        (output / "verification.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        test.close()
        validation.close()
        train.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
