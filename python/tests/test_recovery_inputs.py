"""Reject damaged recovery state and occupied migration targets before writing."""

from __future__ import annotations

import shutil

import pytest
import torch

from mahjong_bc import ModelConfig, Trainer, TrainingConfig
from mahjong_bc.checkpoint import atomic_torch_save, load_checkpoint
from mahjong_bc.errors import ConfigurationError, ResumeError

from .helpers import learnable_pair


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("recovery-inputs")
    train, validation = learnable_pair(train_decisions=7, validation_decisions=5)
    options = dict(
        train_dataset=train,
        validation_dataset=validation,
        training_config=TrainingConfig(
            seed=7, epochs=2, batch_size=4, learning_rate=0.01
        ),
        model_config=ModelConfig(state_hidden=(8,), scorer_hidden=(8,)),
        device="cpu",
    )
    trainer = Trainer(output_dir=root, **options)
    summary = trainer.train()
    try:
        yield trainer, summary, options
    finally:
        torch.set_num_threads(old_threads)


def _damage(payload, kind):
    state = payload["training"]
    optimizer = state["optimizer_state"]
    group = optimizer["param_groups"][0]
    first = next(iter(optimizer["state"].values()))
    if kind == "missing_keys":
        del optimizer["param_groups"]
    elif kind == "missing_params":
        del group["params"]
    elif kind == "lr":
        group["lr"] *= 2
    elif kind == "tensor_lr":
        group["lr"] = torch.ones(2)
    elif kind == "param_order":
        group["params"].reverse()
    elif kind == "shape":
        first["exp_avg"] = torch.zeros(1)
    elif kind == "dtype":
        first["exp_avg_sq"] = first["exp_avg_sq"].double()
    elif kind == "step":
        first["step"] += 1
    elif kind == "missing_slot":
        del first["exp_avg"]
    elif kind == "missing_parameter":
        del optimizer["state"][next(iter(optimizer["state"]))]
    elif kind in ("global_step", "samples_seen"):
        state[kind] += 1
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", [
    "missing_keys", "missing_params", "lr", "tensor_lr", "param_order",
    "shape", "dtype", "step", "missing_slot", "missing_parameter",
    "global_step", "samples_seen",
])
def test_damaged_resume_rejected_before_publication(completed_run, tmp_path, kind):
    original, _, options = completed_run
    payload = torch.load(original.last_path, weights_only=True)
    _damage(payload, kind)
    source = tmp_path / "damaged" / "last.pt"
    atomic_torch_save(payload, source)
    before = source.read_bytes()
    output = tmp_path / "destination"
    trainer = Trainer(output_dir=output, **options)
    with pytest.raises(ResumeError, match="optimizer|progress"):
        trainer.train(resume_from=source)
    assert not output.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("name", ["best.pt", "last.pt", "run.json"])
def test_migration_refuses_reserved_artifact_without_clobber(
    completed_run, tmp_path, name
):
    original, _, options = completed_run
    output = tmp_path / "other-run"
    output.mkdir()
    (output / name).write_bytes(b"another run")
    (output / "notes.txt").write_bytes(b"retain unrelated files")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ConfigurationError, match="already contains"):
        Trainer(output_dir=output, **options).train(resume_from=original.last_path)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_migration_refuses_dangling_reserved_symlink(completed_run, tmp_path):
    original, _, options = completed_run
    output = tmp_path / "other-run"
    output.mkdir()
    reserved = output / "last.pt"
    try:
        reserved.symlink_to(tmp_path / "missing.pt")
    except OSError:
        pytest.skip("symlink creation is not permitted on this host")
    with pytest.raises(ConfigurationError, match="already contains"):
        Trainer(output_dir=output, **options).train(resume_from=original.last_path)
    assert reserved.is_symlink()
    assert not (output / "best.pt").exists()
    assert not (output / "run.json").exists()


@pytest.mark.parametrize("in_place", [False, True])
def test_completed_resume_allows_in_place_or_clean_existing_directory(
    completed_run, tmp_path, in_place
):
    original, summary, options = completed_run
    output = tmp_path / "destination"
    if in_place:
        shutil.copytree(original.output_dir, output)
        source = output / "last.pt"
    else:
        output.mkdir()
        source = original.last_path
    (output / "notes.txt").write_bytes(b"keep")
    trainer = Trainer(output_dir=output, **options)
    actual = trainer.train(resume_from=source)
    assert actual == summary
    assert (output / "notes.txt").read_bytes() == b"keep"
    assert (output / "run.json").is_file()
    assert load_checkpoint(trainer.best_path).role == "best"
    loaded = load_checkpoint(trainer.last_path, require_training=True)
    assert loaded.training.global_step == 4
    assert loaded.training.samples_seen == 14
