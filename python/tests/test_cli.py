from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

import mahjong_bc
import mahjong_bc.checkpoint as checkpoint_module
from mahjong_bc.checkpoint import (
    load_checkpoint,
    load_policy_checkpoint,
    save_checkpoint,
)
from mahjong_bc.cli import build_parser, main
from mahjong_bc.config import ACTIVATION_NAMES, ModelConfig, TrainingConfig
from mahjong_bc.constants import PACKAGE_VERSION, ROLE_POLICY
from mahjong_bc.policy import build_policy
from mahjong_bc.train import TRAIN_OUTPUT_NAMES

from .cache_fixture import read_manifest


def _parse(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _cache_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _train_args(
    cache_root: Path,
    output: Path,
    *,
    log_file: Path | None = None,
) -> list[str]:
    args = [
        "train",
        "--dataset",
        str(cache_root),
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
    if log_file is not None:
        args.extend(["--log-file", str(log_file)])
    return args


def _resume_args(
    cache_root: Path,
    output: Path,
    resume: Path,
    *,
    log_file: Path | None = None,
) -> list[str]:
    args = [
        "train",
        "--dataset",
        str(cache_root),
        "--output",
        str(output),
        "--resume",
        str(resume),
    ]
    if log_file is not None:
        args.extend(["--log-file", str(log_file)])
    return args


def _evaluate_args(
    cache_root: Path,
    checkpoint: Path,
    *,
    output: Path | None = None,
    log_file: Path | None = None,
) -> list[str]:
    args = [
        "evaluate",
        "--dataset",
        str(cache_root),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--batch-size",
        "2",
    ]
    if output is not None:
        args.extend(["--output", str(output)])
    if log_file is not None:
        args.extend(["--log-file", str(log_file)])
    return args


def _write_checkpoint(path: Path) -> Path:
    model = build_policy(
        ModelConfig(state_hidden=(4,), scorer_hidden=(4,)), seed=3
    )
    save_checkpoint(path, model, role=ROLE_POLICY, metrics={"nll": 0.5})
    return path


@pytest.fixture(scope="module")
def completed_resume_run(tmp_path_factory, shared_cache):
    """One real one-epoch run reused read-only by the resume-path cases."""

    old = tmp_path_factory.mktemp("completed-resume-run") / "old"
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        rc = main(_train_args(shared_cache.root, old))
    assert rc == 0, captured.getvalue()
    return shared_cache, old


def test_cache_info_reports_manifest_inventory(shared_cache, capsys):
    fixture = shared_cache
    manifest = read_manifest(fixture.root)

    assert main(["cache-info", "--dataset", str(fixture.root)]) == 0
    info = _parse(capsys)

    assert info["command"] == "cache-info"
    assert info["package_version"] == PACKAGE_VERSION
    for key in (
        "format",
        "cache_id",
        "encoding_version",
        "preparation_format_version",
        "preparation_version",
        "source_revision",
        "corpus_manifest_sha256",
        "split_algorithm",
        "counts",
    ):
        assert info[key] == manifest[key]
    assert set(info["splits"]) == {"train", "validation", "test"}
    for split in info["splits"]:
        assert info["splits"][split]["counts"] == manifest["splits"][split]["counts"]
        assert info["splits"][split]["shards"] == len(
            manifest["splits"][split]["shards"]
        )
    assert info["counts"]["decisions"] == sum(
        info["splits"][split]["counts"]["decisions"] for split in info["splits"]
    )
    assert info["counts"] == manifest["counts"]
    assert info["counts"]["eligible_decisions"] > 0


def test_cache_info_rejects_missing_cache(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert main(["cache-info", "--dataset", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "dataset.json" in captured.err


def test_version_reports_package_and_dependency_versions(capsys):
    assert main(["version"]) == 0
    version = _parse(capsys)
    assert version["package"] == "mahjong-bc"
    assert version["package_version"] == PACKAGE_VERSION == mahjong_bc.__version__
    dependencies = version["dependencies"]
    assert set(dependencies) == {"python", "torch", "numpy", "cuda_build"}
    assert dependencies["python"]
    assert dependencies["torch"]
    assert dependencies["numpy"]
    assert dependencies["cuda_build"] is None or isinstance(
        dependencies["cuda_build"], str
    )


def test_help_lists_available_commands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "cache-info" in output
    assert "train" in output
    assert "evaluate" in output
    assert "inspect" in output
    assert "version" in output

    with pytest.raises(SystemExit) as exc_info:
        main(["cache-info", "--help"])
    assert exc_info.value.code == 0
    assert "--dataset" in capsys.readouterr().out


def test_module_entry_point_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "mahjong_bc", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["package"] == "mahjong-bc"


def test_installed_metadata_and_console_script():
    try:
        distribution = importlib_metadata.distribution("mahjong-bc")
    except importlib_metadata.PackageNotFoundError:
        pytest.skip("package is not installed")
    assert distribution.version == mahjong_bc.__version__
    entry_points = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert entry_points.get("mahjong-bc") == "mahjong_bc.cli:main"

    executable = Path(sys.executable).parent / (
        "mahjong-bc.exe" if os.name == "nt" else "mahjong-bc"
    )
    if not executable.is_file():
        pytest.skip("console script is not installed next to the interpreter")
    result = subprocess.run(
        [str(executable), "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["package"] == "mahjong-bc"


def test_train_evaluate_inspect_synthetic_cache_end_to_end(shared_cache, tmp_path, capsys):
    fixture = shared_cache
    manifest = read_manifest(fixture.root)
    output = tmp_path / "run"
    log = tmp_path / "train.log.json"
    rc = main(
        [
            "train",
            "--dataset",
            str(fixture.root),
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
    assert result["dataset"]["cache_id"] == manifest["cache_id"]
    assert result["dataset"]["provenance"] == manifest["provenance"]
    assert set(result["source"]) == {
        "package",
        "package_version",
        "hash",
        "revision",
        "dirty",
    }
    assert result["source"]["package"] == "mahjong-bc"
    assert result["source"]["package_version"] == PACKAGE_VERSION
    assert len(result["source"]["hash"]) == 64
    assert result["checkpoints"]["best"] == str(output / "best.pt")
    assert result["checkpoints"]["last"] == str(output / "last.pt")
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert (output / "run.json").is_file()
    assert log.is_file()
    logged = json.loads(log.read_text(encoding="utf-8"))
    assert logged["summary"]["completed_epochs"] == 2
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["source"] == result["source"]
    assert record["package_version"] == result["package_version"]
    assert record["model_config"] == result["model_config"]
    assert record["training_config"] == result["training_config"]
    assert record["dependencies"] == result["dependencies"]
    assert record["device_type"] == result["device_type"]
    assert (
        record["dataset"]["train_dataset_identity"]
        == result["train_dataset_identity"]
    )
    assert (
        record["dataset"]["validation_dataset_identity"]
        == result["validation_dataset_identity"]
    )
    assert record["summary"] == result["summary"]
    assert record["dataset"]["provenance"]["cache_id"] == manifest["cache_id"]
    assert record["training_config"]["seed"] == 1
    assert record["best_checkpoint"] == "best.pt"
    assert record["last_checkpoint"] == "last.pt"
    assert set(result) == {
        "command",
        "package_version",
        "source",
        "dataset",
        "model_config",
        "training_config",
        "dependencies",
        "device_type",
        "train_dataset_identity",
        "validation_dataset_identity",
        "summary",
        "checkpoints",
        "run_record",
    }
    assert set(result["dataset"]) == {
        "cache_id",
        "encoding_version",
        "preparation_format_version",
        "preparation_version",
        "source_revision",
        "corpus_manifest_sha256",
        "split_algorithm",
        "provenance",
    }
    loaded = load_policy_checkpoint(output / "best.pt")
    assert loaded.role == "best"
    loaded_last = load_checkpoint(
        output / "last.pt", require_training=True
    )
    assert loaded_last.role == "last"
    assert loaded_last.training.completed_epochs == 2

    metrics_path = tmp_path / "metrics.json"
    eval_log = tmp_path / "eval.log.json"
    rc = main(
        [
            "evaluate",
            "--dataset",
            str(fixture.root),
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
    test_counts = manifest["splits"]["test"]["counts"]
    assert evaluation["metrics"]["eligible"] == test_counts["eligible_decisions"]
    assert evaluation["metrics"]["forced"] == test_counts["forced_decisions"]
    assert metrics_path.is_file()
    assert eval_log.is_file()

    assert main(["inspect", "--checkpoint", str(output / "best.pt")]) == 0
    inspected = _parse(capsys)
    assert inspected["command"] == "inspect"
    assert inspected["role"] == "best"
    assert inspected["has_training_state"] is True
    assert inspected["completed_epochs"] == 2
    assert inspected["checkpoint_format_version"] == 3
    assert inspected["model_format_version"] == 1


def test_cli_fresh_run_protects_existing_outputs_and_requires_config(
    shared_cache, tmp_path, capsys
):
    fixture = shared_cache
    output = tmp_path / "existing"
    args = [
        "train",
        "--dataset",
        str(fixture.root),
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
    before = (output / "best.pt").read_bytes()
    assert main(args) == 2
    assert "already contains" in capsys.readouterr().err
    assert (output / "best.pt").read_bytes() == before

    reserved = tmp_path / "reserved"
    reserved.mkdir()
    sentinel = reserved / "last.pt"
    sentinel.write_bytes(b"future recovery artifact")
    assert main([*args[:4], str(reserved), *args[5:]]) == 2
    assert "last.pt" in capsys.readouterr().err
    assert sentinel.read_bytes() == b"future recovery artifact"

    missing = tmp_path / "missing"
    assert main(
        [
            "train",
            "--dataset",
            str(fixture.root),
            "--output",
            str(missing),
        ]
    ) == 2
    assert "requires" in capsys.readouterr().err
    assert not missing.exists()


def test_cli_train_omitted_options_use_config_defaults(shared_cache, tmp_path, capsys):
    fixture = shared_cache
    output = tmp_path / "defaults"
    assert main(
        [
            "train",
            "--dataset",
            str(fixture.root),
            "--output",
            str(output),
            "--seed",
            "3",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.02",
        ]
    ) == 0
    result = _parse(capsys)
    expected_training = TrainingConfig(
        seed=3, epochs=1, batch_size=2, learning_rate=0.02
    )
    assert result["model_config"] == ModelConfig().to_dict()
    assert result["training_config"] == expected_training.to_dict()


def test_cli_activation_choices_match_config_names():
    parser = build_parser()
    for name in ACTIVATION_NAMES:
        args = parser.parse_args(
            [
                "train",
                "--dataset",
                "cache",
                "--output",
                "output",
                "--activation",
                name,
            ]
        )
        assert args.activation == name
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--dataset",
                "cache",
                "--output",
                "output",
                "--activation",
                "unknown",
            ]
        )


def test_cli_resume_to_new_output_and_conflicting_options(
    completed_resume_run, tmp_path, capsys
):
    fixture, old = completed_resume_run
    new = tmp_path / "new"
    log = tmp_path / "resume.log.json"
    assert main(
        _resume_args(fixture.root, new, old / "last.pt", log_file=log)
    ) == 0
    result = _parse(capsys)
    assert result["summary"]["completed_epochs"] == 1
    assert (new / "best.pt").is_file()
    assert (new / "last.pt").is_file()
    assert (new / "run.json").is_file()
    assert log.is_file()
    resumed_last = load_checkpoint(new / "last.pt", require_training=True)
    assert resumed_last.role == "last"
    assert resumed_last.training.completed_epochs == 1

    conflict = tmp_path / "conflict"
    args = _resume_args(fixture.root, conflict, old / "last.pt")
    args.extend(["--seed", "3"])
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "uses the checkpoint's" in captured.err
    assert not conflict.exists()

    policy = _write_checkpoint(tmp_path / "policy.pt")
    policy_output = tmp_path / "policy-resume"
    assert main(
        _resume_args(fixture.root, policy_output, policy)
    ) == 2
    assert "recoverable training state" in capsys.readouterr().err
    assert not policy_output.exists()


def test_cli_resume_reads_last_checkpoint_once(
    completed_resume_run, tmp_path, capsys, monkeypatch
):
    fixture, old = completed_resume_run
    real_load = checkpoint_module.torch.load
    reads: list[Path] = []

    def counting_load(*args, **kwargs):
        reads.append(Path(args[0]))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", counting_load)
    new = tmp_path / "new"
    assert main(_resume_args(fixture.root, new, old / "last.pt")) == 0
    result = _parse(capsys)
    assert result["summary"]["completed_epochs"] == 1
    assert reads == [old / "last.pt"]
    assert (new / "last.pt").is_file()


def test_cli_resume_output_and_log_paths_remain_protected(
    completed_resume_run, fresh_cache, tmp_path, capsys
):
    _, old = completed_resume_run
    fixture = fresh_cache
    before = _cache_snapshot(fixture.root)

    inside = fixture.root / "resumed"
    assert main(
        _resume_args(fixture.root, inside, old / "last.pt")
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "inside prepared cache" in captured.err
    assert _cache_snapshot(fixture.root) == before

    output = tmp_path / "resume-output"
    log = output / "best.pt"
    assert main(
        _resume_args(fixture.root, output, old / "last.pt", log_file=log)
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "aliases the training output" in captured.err
    assert not output.exists()

    existing = tmp_path / "existing-resume.log.json"
    sentinel = b"existing report"
    existing.write_bytes(sentinel)
    assert main(
        _resume_args(
            fixture.root, output, old / "last.pt", log_file=existing
        )
    ) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert existing.read_bytes() == sentinel
    assert not output.exists()


def test_cli_rejects_training_split_evaluation(shared_cache, tmp_path):
    fixture = shared_cache
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                "--dataset",
                str(fixture.root),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--split",
                "train",
            ]
        )


@pytest.mark.parametrize("relative", [None, "nested-output"])
def test_cli_train_output_must_be_outside_cache_root(
    fresh_cache, tmp_path, capsys, relative
):
    fixture = fresh_cache
    output = fixture.root if relative is None else fixture.root / relative
    before = _cache_snapshot(fixture.root)
    assert main(_train_args(fixture.root, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "inside prepared cache" in captured.err
    assert _cache_snapshot(fixture.root) == before
    assert not (tmp_path / "best.pt").exists()
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize(
    "log_path",
    [*TRAIN_OUTPUT_NAMES, None],
)
def test_cli_train_log_file_cannot_alias_training_outputs(
    shared_cache, tmp_path, capsys, log_path
):
    fixture = shared_cache
    output = tmp_path / "train"
    log = output if log_path is None else output / log_path
    before = _cache_snapshot(fixture.root)
    assert main(_train_args(fixture.root, output, log_file=log)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "aliases the training output" in captured.err
    assert all(name in captured.err for name in TRAIN_OUTPUT_NAMES)
    assert not output.exists()
    assert _cache_snapshot(fixture.root) == before


def test_cli_train_log_file_must_be_outside_cache_and_not_preexisting(
    fresh_cache, tmp_path, capsys
):
    fixture = fresh_cache
    before = _cache_snapshot(fixture.root)
    output = tmp_path / "train"

    inside = fixture.root / "train.log.json"
    assert main(_train_args(fixture.root, output, log_file=inside)) == 2
    captured = capsys.readouterr()
    assert "inside prepared cache" in captured.err
    assert not output.exists()
    assert _cache_snapshot(fixture.root) == before

    existing = tmp_path / "existing.log.json"
    sentinel = b"existing report"
    existing.write_bytes(sentinel)
    assert main(_train_args(fixture.root, output, log_file=existing)) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert not output.exists()
    assert existing.read_bytes() == sentinel
    assert _cache_snapshot(fixture.root) == before


def test_cli_evaluate_report_paths_cannot_alias_checkpoint_or_touch_cache(
    fresh_cache, tmp_path, capsys
):
    fixture = fresh_cache
    checkpoint = _write_checkpoint(tmp_path / "policy.pt")
    checkpoint_before = checkpoint.read_bytes()
    cache_before = _cache_snapshot(fixture.root)

    cases = (
        ("--output", checkpoint, "checkpoint"),
        ("--log-file", checkpoint, "checkpoint"),
        ("--output", fixture.root / "dataset.json", "cache"),
        ("--log-file", fixture.root / "extra-report.json", "cache"),
    )
    for option, destination, reason in cases:
        kwargs = {
            "output": destination if option == "--output" else None,
            "log_file": destination if option == "--log-file" else None,
        }
        assert main(
            _evaluate_args(fixture.root, checkpoint, **kwargs)
        ) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert reason in captured.err
        assert checkpoint.read_bytes() == checkpoint_before
        assert _cache_snapshot(fixture.root) == cache_before


@pytest.mark.parametrize("destination_option", ["--output", "--log-file"])
def test_cli_evaluate_report_outputs_do_not_overwrite_existing_files(
    shared_cache, tmp_path, capsys, destination_option
):
    fixture = shared_cache
    checkpoint = _write_checkpoint(tmp_path / "policy.pt")
    checkpoint_before = checkpoint.read_bytes()
    cache_before = _cache_snapshot(fixture.root)
    destination = tmp_path / "existing-report.json"
    sentinel = b"existing report bytes"
    destination.write_bytes(sentinel)
    kwargs = {
        "output": destination if destination_option == "--output" else None,
        "log_file": destination if destination_option == "--log-file" else None,
    }
    assert main(_evaluate_args(fixture.root, checkpoint, **kwargs)) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert destination.read_bytes() == sentinel
    assert checkpoint.read_bytes() == checkpoint_before
    assert _cache_snapshot(fixture.root) == cache_before


def test_cli_cache_info_output_cannot_be_inside_cache_root(fresh_cache, capsys):
    fixture = fresh_cache
    manifest = fixture.root / "dataset.json"
    before_bytes = manifest.read_bytes()
    before = _cache_snapshot(fixture.root)
    assert main(
        ["cache-info", "--dataset", str(fixture.root), "--output", str(manifest)]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "inside prepared cache" in captured.err
    assert manifest.read_bytes() == before_bytes
    assert _cache_snapshot(fixture.root) == before


def test_cli_report_output_and_log_may_share_a_fresh_destination(shared_cache, tmp_path, capsys):
    fixture = shared_cache
    checkpoint = _write_checkpoint(tmp_path / "policy.pt")
    destination = tmp_path / "evaluation.json"
    assert main(
        _evaluate_args(
            fixture.root,
            checkpoint,
            output=destination,
            log_file=destination,
        )
    ) == 0
    reported = _parse(capsys)
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert reported == written
    assert written["command"] == "evaluate"


def test_cli_path_checks_resolve_symlinked_parent_directories(shared_cache, tmp_path, capsys):
    fixture = shared_cache
    output = tmp_path / "train"
    output.mkdir()
    alias = tmp_path / "output-alias"
    try:
        alias.symlink_to(output, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create directory symlinks in this environment")
    before = _cache_snapshot(fixture.root)
    assert main(
        _train_args(fixture.root, output, log_file=alias / "best.pt")
    ) == 2
    captured = capsys.readouterr()
    assert "aliases the training output" in captured.err
    assert not (output / "best.pt").exists()
    assert not (output / "run.json").exists()
    assert _cache_snapshot(fixture.root) == before


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive path alias test")
def test_cli_path_checks_normalize_filesystem_case(shared_cache, tmp_path, capsys):
    fixture = shared_cache
    output = tmp_path / "TrainCase"
    output.mkdir()
    log = Path(str(output).upper()) / "BEST.PT"
    before = _cache_snapshot(fixture.root)
    assert main(
        _train_args(fixture.root, output, log_file=log)
    ) == 2
    captured = capsys.readouterr()
    assert "aliases the training output" in captured.err
    assert not (output / "best.pt").exists()
    assert _cache_snapshot(fixture.root) == before
