from __future__ import annotations

import json
from pathlib import Path

import pytest

from mahjong_bc.cli import main
from mahjong_bc.checkpoint import load_checkpoint

from .cache_fixture import synthetic_cache


def _parse(capsys) -> dict:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_cache_info_and_version_commands(tmp_path, capsys):
    cache = synthetic_cache(tmp_path / "cache")
    assert main(["cache-info", "--dataset", str(cache)]) == 0
    info = _parse(capsys)
    assert info["format"] == "mahjong-bc-dataset"
    assert info["cache_id"] == "synthetic-cache-id"
    assert info["counts"]["decisions"] == 17
    assert set(info["splits"]) == {"train", "validation", "test"}

    assert main(["version"]) == 0
    version = _parse(capsys)
    assert version["package"] == "mahjong-bc"
    assert version["dependencies"]["torch"]


def test_train_evaluate_inspect_synthetic_cache_end_to_end(tmp_path, capsys):
    cache = synthetic_cache(tmp_path / "cache")
    output = tmp_path / "run"
    log = tmp_path / "train.log.json"
    rc = main(
        [
            "train",
            "--dataset",
            str(cache),
            "--output",
            str(output),
            "--seed",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.02",
            "--state-hidden",
            "8",
            "--scorer-hidden",
            "8",
            "--log-file",
            str(log),
        ]
    )
    assert rc == 0
    result = _parse(capsys)
    assert result["command"] == "train"
    assert result["summary"]["completed_epochs"] == 2
    assert result["device_type"] == "cpu"
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert (output / "run.json").is_file()
    assert log.is_file()
    assert json.loads(log.read_text(encoding="utf-8"))["summary"]["completed_epochs"] == 2
    last = load_checkpoint(output / "last.pt", require_training=True)
    assert last.training.best_model_state is not None

    metrics_path = tmp_path / "metrics.json"
    eval_log = tmp_path / "eval.log.json"
    rc = main(
        [
            "evaluate",
            "--dataset",
            str(cache),
            "--checkpoint",
            str(output / "best.pt"),
            "--split",
            "test",
            "--batch-size",
            "2",
            "--output",
            str(metrics_path),
            "--log-file",
            str(eval_log),
        ]
    )
    assert rc == 0
    evaluation = _parse(capsys)
    assert evaluation["command"] == "evaluate"
    assert evaluation["split"] == "test"
    assert evaluation["metrics"]["forced"] == 1
    assert evaluation["metrics"]["eligible"] == 2
    assert metrics_path.is_file()
    assert eval_log.is_file()

    assert main(["inspect", "--checkpoint", str(output / "best.pt")]) == 0
    inspected = _parse(capsys)
    assert inspected["role"] == "best"
    assert inspected["has_training_state"] is True


def test_cli_resume_to_new_output_and_conflicting_options(tmp_path, capsys):
    cache = synthetic_cache(tmp_path / "cache")
    old = tmp_path / "old"
    assert main(
        [
            "train",
            "--dataset",
            str(cache),
            "--output",
            str(old),
            "--seed",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.02",
            "--state-hidden",
            "8",
            "--scorer-hidden",
            "8",
        ]
    ) == 0
    capsys.readouterr()

    new = tmp_path / "new"
    log = tmp_path / "resume.log.json"
    rc = main(
        [
            "train",
            "--dataset",
            str(cache),
            "--output",
            str(new),
            "--resume",
            str(old / "last.pt"),
            "--log-file",
            str(log),
        ]
    )
    assert rc == 0
    result = _parse(capsys)
    assert result["summary"]["completed_epochs"] == 1
    assert (new / "best.pt").is_file()
    assert (new / "last.pt").is_file()
    assert (new / "run.json").is_file()
    assert log.is_file()

    rc = main(
        [
            "train",
            "--dataset",
            str(cache),
            "--output",
            str(tmp_path / "conflict"),
            "--resume",
            str(old / "last.pt"),
            "--seed",
            "3",
        ]
    )
    assert rc == 2
    assert "uses the checkpoint's" in capsys.readouterr().err
    assert not (tmp_path / "conflict").exists()


def test_cli_rejects_new_run_in_existing_output_and_missing_config(tmp_path, capsys):
    cache = synthetic_cache(tmp_path / "cache")
    output = tmp_path / "existing"
    args = [
        "train",
        "--dataset",
        str(cache),
        "--output",
        str(output),
        "--seed",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--learning-rate",
        "0.02",
        "--state-hidden",
        "8",
        "--scorer-hidden",
        "8",
    ]
    assert main(args) == 0
    capsys.readouterr()
    before = (output / "last.pt").read_bytes()
    assert main(args) == 2
    assert "already contains" in capsys.readouterr().err
    assert (output / "last.pt").read_bytes() == before

    missing = tmp_path / "missing"
    assert main(
        [
            "train",
            "--dataset",
            str(cache),
            "--output",
            str(missing),
        ]
    ) == 2
    assert "requires --seed" in capsys.readouterr().err
    assert not missing.exists()


def test_cli_evaluate_rejects_non_evaluation_split(tmp_path):
    cache = synthetic_cache(tmp_path / "cache")
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                "--dataset",
                str(cache),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--split",
                "train",
            ]
        )
