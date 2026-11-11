"""Formal comparison CLI: artifacts, provenance, rejection, and failure gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
import torch

import mahjong_eval
import mahjong_eval.inference as inference
from mahjong_bc.checkpoint import (
    TrainingCheckpointState,
    save_checkpoint,
)
from mahjong_bc.config import (
    DatasetIdentity,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from mahjong_bc.constants import (
    ENCODING_VERSION,
    ROLE_BEST,
    ROLE_LAST,
    ROLE_POLICY,
)
from mahjong_bc.errors import DataError
from mahjong_bc.policy import build_policy

from mahjong_eval import cli, records, scheduler


def _model_config() -> ModelConfig:
    return ModelConfig(state_hidden=(4,), scorer_hidden=(4,))


def _training_state(model) -> TrainingCheckpointState:
    identity = DatasetIdentity(
        cache_id="compare-fixture",
        split="train",
        encoding_version=ENCODING_VERSION,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=4,
        eligible_decisions=2,
    )
    validation = DatasetIdentity(
        cache_id="compare-fixture",
        split="validation",
        encoding_version=ENCODING_VERSION,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=2,
        eligible_decisions=1,
    )
    return TrainingCheckpointState(
        training_config=TrainingConfig(
            seed=3, epochs=1, batch_size=1, learning_rate=0.01
        ),
        train_dataset_identity=identity,
        validation_dataset_identity=validation,
        dependencies=collect_dependency_versions(),
        device_type="cpu",
        completed_epochs=1,
        global_step=4,
        samples_seen=6,
        best_validation_nll=0.25,
        best_epoch=0,
        history=[{"epoch": 0, "steps": 4, "samples": 6}],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=None,
        order_rng_state=torch.Generator().manual_seed(7).get_state(),
        best_model_state={
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        best_metrics={"nll": 0.25},
    )


def _write_checkpoint(path: Path, role: str) -> Path:
    model = build_policy(_model_config(), seed=5)
    if role == ROLE_POLICY:
        save_checkpoint(path, model, role=role, metrics={"nll": 0.25})
    else:
        save_checkpoint(
            path,
            model,
            role=role,
            training=_training_state(model),
            metrics={"nll": 0.25},
        )
    return path


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("compare-checkpoints")
    return {
        role: _write_checkpoint(directory / f"{role}.pt", role)
        for role in (ROLE_BEST, ROLE_LAST, ROLE_POLICY)
    }


def _first_seed(split: str) -> int:
    for seed in range(200):
        if mahjong_eval.seed_split(seed) == split:
            return seed
    raise AssertionError(f"no {split} seed found")


@pytest.fixture(scope="module")
def validation_seed() -> int:
    return _first_seed("validation")


@pytest.fixture(scope="module")
def test_seed() -> int:
    return _first_seed("test")


def _compare_args(checkpoint, out, *, split, seeds, extra=()):
    return [
        "compare",
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--out",
        str(out),
        *extra,
    ]


def _run_compare(checkpoint, out, *, split, seeds, extra=()):
    return cli.main(_compare_args(checkpoint, out, split=split, seeds=seeds, extra=extra))


def test_compare_publishes_reproducible_artifacts(checkpoints, validation_seed, tmp_path, capsys):
    out = tmp_path / "run"
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="validation",
        seeds=[validation_seed],
        extra=[
            "--batch-size",
            "3",
            "--concurrent-games",
            "2",
            "--teacher-cache-bytes",
            "0",
            "--device",
            "cpu",
        ],
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    printed = json.loads(captured.out)
    assert printed["command"] == "compare"
    assert printed["split"] == "validation"

    paths = records.run_paths(out)
    assert paths.state.is_file()
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == "complete"
    costs_path = out / cli.COSTS_FILENAME
    assert costs_path.is_file()
    assert paths.summary.is_file()

    plan = records.read_plan(paths.plan)
    assert plan.seeds == (validation_seed,)
    assert plan.placements == records.PLACEMENTS
    metadata = plan.metadata
    assert metadata["split"] == "validation"
    assert metadata["split_algorithm"] == "seed-sha256-80-10-10-v1"
    assert metadata["split_buckets"] == {
        str(validation_seed): mahjong_eval.split_bucket(validation_seed)
    }
    assert metadata["selection"] == {
        "mode": "greedy",
        "device": "cpu",
        "batch_size": 3,
        "concurrent_games": 2,
    }
    assert metadata["teacher_cache"]["budget_bytes"] == 0
    assert metadata["teacher_cache"]["scope"] == "shared"
    contents = checkpoints[ROLE_BEST].read_bytes()
    assert metadata["checkpoint"]["sha256"] == hashlib.sha256(contents).hexdigest()
    assert metadata["checkpoint"]["size_bytes"] == len(contents)
    assert metadata["checkpoint"]["role"] == ROLE_BEST
    assert metadata["python_source"]["online_evaluator"]["package"] == "mahjong-eval"
    assert metadata["python_source"]["offline_policy"]["package"] == "mahjong-bc"
    assert len(metadata["python_source"]["online_evaluator"]["hash"]) == 64
    extension = metadata["rust_runtime"]["extension"]
    assert len(extension["source_sha256"]) == 64
    assert extension["source_complete"] is True
    assert extension["source_identity"] == (
        "sha256-paths-bytes-v1:crates/python+src+crates/neural"
    )
    assert metadata["training_provenance"]["best_epoch"] == 0
    assert metadata["dependencies"]["evaluator"]["package"] == "mahjong-eval"

    persisted = records.read_records(paths.records, plan, require_complete=True)
    assert len(persisted) == 6
    assert {record.neural_seats for record in persisted} == set(records.PLACEMENTS)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert summary == records.summarize_run(out)
    assert summary == records.compute_summary(plan, persisted)

    costs = json.loads(costs_path.read_text(encoding="utf-8"))
    assert costs["format"] == cli.COSTS_FORMAT
    assert costs["format_version"] == cli.COSTS_FORMAT_VERSION
    assert costs["games"] == 6
    assert costs["seeds"] == [validation_seed]
    assert costs["checkpoint_sha256"] == metadata["checkpoint"]["sha256"]
    assert costs["counts"]["decisions"]["total"] == sum(
        record.decisions for record in persisted
    )
    assert costs["counts"]["games"] == 6
    assert costs["counts"]["teacher"]["cache"]["budget_bytes"] == 0
    assert len(costs["per_game"]) == 6
    timings = costs["timings"]
    assert timings["calls"]["env_resets"] == 6
    assert timings["calls"]["env_steps"] == costs["counts"]["decisions"]["total"]
    assert timings["calls"]["inference_batches"] == costs["counts"]["inference"]["batches"]
    assert timings["calls"]["teacher_queries"] == costs["counts"]["teacher"]["queries"]
    assert timings["cuda_synchronized_inference"] is False
    assert timings["seconds"]["total"] >= sum(
        timings["seconds"][name]
        for name in ("env_reset", "env_step", "teacher", "inference")
    )


def test_test_split_run_is_the_frozen_comparison(checkpoints, test_seed, tmp_path):
    out = tmp_path / "run"
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="test",
        seeds=[test_seed],
        extra=["--concurrent-games", "6", "--batch-size", "6"],
    )
    assert code == 0
    plan = records.read_plan(records.run_paths(out).plan)
    assert plan.metadata["split"] == "test"
    persisted = records.read_records(
        records.run_paths(out).records, plan, require_complete=True
    )
    assert len(persisted) == 6


@pytest.mark.parametrize(
    ("seeds", "fragment"),
    [
        ("", "at least one seed"),
        ("1,,2", "empty entry"),
        ("abc", "expected a decimal integer"),
        ("1,1", "unique"),
        ("-1", "must fit in an unsigned 64-bit integer"),
        (str(2**64), "must fit in an unsigned 64-bit integer"),
    ],
)
def test_seed_parser_rejects_invalid_lists(seeds, fragment):
    with pytest.raises(cli.CompareError) as raised:
        cli._parse_seeds(seeds)
    assert fragment in str(raised.value)


def test_seed_split_mismatch_is_rejected_before_output(checkpoints, validation_seed, tmp_path, capsys):
    train_seed = _first_seed("train")
    out = tmp_path / "run"
    code = _run_compare(
        checkpoints[ROLE_BEST], out, split="validation", seeds=[train_seed]
    )
    assert code == 2
    assert "belongs to the 'train' split" in capsys.readouterr().err
    assert not out.exists()

    other = tmp_path / "other"
    code = _run_compare(
        checkpoints[ROLE_BEST],
        other,
        split="validation",
        seeds=[validation_seed, train_seed],
    )
    assert code == 2
    assert not other.exists()


@pytest.mark.parametrize("role", [ROLE_LAST, ROLE_POLICY])
def test_non_best_checkpoints_are_rejected(checkpoints, role, validation_seed, tmp_path, capsys):
    out = tmp_path / "run"
    code = _run_compare(
        checkpoints[role], out, split="validation", seeds=[validation_seed]
    )
    assert code == 2
    message = capsys.readouterr().err
    assert ROLE_BEST in message
    assert not out.exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--batch-size", "0"),
        ("--concurrent-games", "0"),
        ("--decision-limit", "0"),
        ("--teacher-cache-bytes", "-1"),
    ],
)
def test_invalid_numeric_options_are_rejected(
    checkpoints, validation_seed, tmp_path, capsys, flag, value
):
    out = tmp_path / "run"
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="validation",
        seeds=[validation_seed],
        extra=[flag, value],
    )
    assert code == 2
    assert "must be a" in capsys.readouterr().err
    assert not out.exists()


def test_missing_checkpoint_reports_an_error(validation_seed, tmp_path, capsys):
    out = tmp_path / "run"
    code = _run_compare(
        tmp_path / "absent.pt", out, split="validation", seeds=[validation_seed]
    )
    assert code == 2
    assert "cannot read checkpoint" in capsys.readouterr().err
    assert not out.exists()


def test_plan_and_completed_records_survive_an_inference_failure(
    checkpoints, validation_seed, tmp_path, capsys, monkeypatch
):
    out = tmp_path / "run"
    seen_record = {"value": False}
    original_append = records.RecordWriter.append
    original_select = inference.FrozenPolicy.select_indices

    def append(self, record):
        original_append(self, record)
        seen_record["value"] = True

    def select(self, decisions):
        if seen_record["value"]:
            raise DataError("synthetic inference failure")
        return original_select(self, decisions)

    monkeypatch.setattr(records.RecordWriter, "append", append)
    monkeypatch.setattr(inference.FrozenPolicy, "select_indices", select)
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="validation",
        seeds=[validation_seed],
        extra=["--concurrent-games", "1"],
    )
    assert code == 2
    message = capsys.readouterr().err
    assert "comparison stopped" in message
    assert "synthetic inference failure" in message

    paths = records.run_paths(out)
    assert paths.state.is_file()
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == "incomplete"
    assert paths.plan.is_file()
    assert not paths.summary.exists()
    assert not (out / cli.COSTS_FILENAME).exists()
    plan = records.read_plan(paths.plan)
    persisted = records.read_records(paths.records, plan)
    assert 1 <= len(persisted) < 6 * len(plan.seeds)
    with pytest.raises(records.RecordError):
        records.finalize_run(out)
    assert not paths.summary.exists()


def test_game_failure_after_a_completed_game_keeps_records_and_blocks_summary(
    checkpoints, validation_seed, tmp_path, capsys, monkeypatch
):
    out = tmp_path / "run"
    state = {"envs": 0, "fail": False}

    class FlakyEnv:
        def __init__(self, inner):
            self.inner = inner

        def reset(self, seed):
            return self.inner.reset(seed)

        def teacher_index(self, cache=None):
            return self.inner.teacher_index(cache=cache)

        def step(self, decision_id, candidate_index):
            if state["fail"]:
                raise RuntimeError("synthetic game failure")
            return self.inner.step(decision_id, candidate_index)

    def factory(*, decision_limit):
        state["envs"] += 1
        if state["envs"] > 1:
            state["fail"] = True
        return FlakyEnv(mahjong_eval.Env(decision_limit=decision_limit))

    monkeypatch.setattr(scheduler, "_default_env", lambda: factory)
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="validation",
        seeds=[validation_seed],
        extra=["--concurrent-games", "1"],
    )
    assert code == 2
    message = capsys.readouterr().err
    assert "synthetic game failure" in message
    assert f"seed {validation_seed}" in message
    assert "placement [0, 2]" in message

    paths = records.run_paths(out)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == "incomplete"
    assert not paths.summary.exists()
    assert not (out / cli.COSTS_FILENAME).exists()
    plan = records.read_plan(paths.plan)
    persisted = records.read_records(paths.records, plan)
    assert 1 <= len(persisted) < 6 * len(plan.seeds)


def test_costs_failure_keeps_completed_records_and_blocks_summary(
    checkpoints, validation_seed, tmp_path, capsys, monkeypatch
):
    out = tmp_path / "run"

    def failing_costs(path, document):
        raise cli.CompareError("synthetic costs failure")

    monkeypatch.setattr(cli, "write_costs", failing_costs)
    code = _run_compare(
        checkpoints[ROLE_BEST],
        out,
        split="validation",
        seeds=[validation_seed],
        extra=["--concurrent-games", "6"],
    )
    assert code == 2
    assert "synthetic costs failure" in capsys.readouterr().err

    paths = records.run_paths(out)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == "incomplete"
    assert paths.plan.is_file()
    assert not paths.summary.exists()
    assert not (out / cli.COSTS_FILENAME).exists()
    plan = records.read_plan(paths.plan)
    persisted = records.read_records(paths.records, plan, require_complete=True)
    assert len(persisted) == 6
    # The T6 public gate must keep refusing the summary after costs failed.
    with pytest.raises(records.RecordError):
        records.finalize_run(out)
    assert not paths.summary.exists()


def test_existing_run_artifacts_are_refused(checkpoints, validation_seed, tmp_path, capsys):
    out = tmp_path / "run"
    assert (
        _run_compare(
            checkpoints[ROLE_BEST],
            out,
            split="validation",
            seeds=[validation_seed],
            extra=["--concurrent-games", "6"],
        )
        == 0
    )
    capsys.readouterr()
    code = _run_compare(
        checkpoints[ROLE_BEST], out, split="validation", seeds=[validation_seed]
    )
    assert code == 2
    assert "already contains" in capsys.readouterr().err


def test_checkpoint_runs_without_corpus_cache_or_training_run(
    checkpoints, validation_seed, tmp_path, capsys, monkeypatch
):
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    copied = isolated / "best.pt"
    copied.write_bytes(checkpoints[ROLE_BEST].read_bytes())
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    out = tmp_path / "run"
    code = _run_compare(
        copied,
        out,
        split="validation",
        seeds=[validation_seed],
        extra=["--concurrent-games", "6", "--teacher-cache-bytes", "0"],
    )
    assert code == 0, capsys.readouterr().err
    plan = records.read_plan(records.run_paths(out).plan)
    assert plan.metadata["checkpoint"]["path"] == str(copied.resolve())
    assert plan.metadata["checkpoint"]["sha256"] == hashlib.sha256(
        copied.read_bytes()
    ).hexdigest()
    assert list(Path(workdir).iterdir()) == []


def test_module_entry_point_prints_version_provenance(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "mahjong_eval", "version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["package"] == "mahjong-eval"
    assert document["native"]["extension"]["source_complete"] is True


def test_persisted_score_disagreement_blocks_publication(
    checkpoints, validation_seed, tmp_path, capsys, monkeypatch
):
    from dataclasses import replace

    original = records.read_records

    def inconsistent_read(*args, **kwargs):
        values = original(*args, **kwargs)
        first = values[0]
        scores = list(first.scores)
        scores[0] += 1
        scores[1] -= 1
        values[0] = replace(first, scores=tuple(scores))
        return values

    monkeypatch.setattr(records, "read_records", inconsistent_read)
    out = tmp_path / "inconsistent-run"
    assert _run_compare(
        checkpoints[ROLE_BEST], out, split="validation", seeds=[validation_seed]
    ) == 2
    message = capsys.readouterr().err
    assert "persisted terminal result differs" in message
    assert "completed records are retained" in message
    assert json.loads((out / "state.json").read_text())["status"] == "incomplete"
    assert not (out / "summary.json").exists()
    assert not (out / cli.COSTS_FILENAME).exists()


def test_installed_console_script_is_available():
    scripts = Path(sysconfig.get_path("scripts"))
    candidates = [
        scripts / "mahjong-eval",
        scripts / "mahjong-eval.exe",
        scripts / "mahjong_eval",
    ]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("console script is not installed in this environment")
    completed = subprocess.run(
        [str(executable), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "compare" in completed.stdout
