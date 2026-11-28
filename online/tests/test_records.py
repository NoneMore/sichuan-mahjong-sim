from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import mahjong_bc
import pytest
import torch
from _checkpoint_fixtures import (
    build_checkpoint,
    build_model,
    build_training_state,
)
from mahjong_bc._source import collect_source_record, hash_source_directory
from mahjong_bc.constants import ROLE_BEST, ROLE_POLICY
from mahjong_eval import records

_FAKE_PYTHON_SOURCE = {
    "online_evaluator": {
        "package": "mahjong-eval",
        "package_version": None,
        "hash": "a" * 64,
        "revision": None,
        "dirty": None,
    },
    "offline_policy": {
        "package": "mahjong-bc",
        "package_version": "0.1.0",
        "hash": "b" * 64,
        "revision": "fixture-revision",
        "dirty": False,
    },
}

_RUST_RUNTIME = {
    "source": {"revision": "rust-revision", "dirty": False},
    "build": {"profile": "debug", "features": ["neural"]},
    "nested": [1, True, None],
}


def _plan(
    identity: dict,
    *,
    seeds: tuple[int, ...] = (1, 2),
    decision_limit: int = 100,
    metadata: dict | None = None,
) -> records.EvaluationPlan:
    if metadata is None:
        metadata = records.build_plan_metadata(
            checkpoint=identity,
            rust_runtime=_RUST_RUNTIME,
            split_algorithm="seed-sha256-80-10-10-v1",
            selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
            teacher_cache={"scope": "shared"},
            dependencies={"python": "3.12", "torch": "2.14"},
            training_provenance=None,
            python_source=_FAKE_PYTHON_SOURCE,
        )
    return records.EvaluationPlan(
        seeds=seeds, metadata=metadata, decision_limit=decision_limit
    )


def _record(
    seed: int,
    placement: tuple[int, int],
    *,
    neural_mean: int,
    decisions: int,
    winners: tuple[int, ...],
    end_reason: str,
) -> records.TerminalRecord:
    scores = [0, 0, 0, 0]
    for seat in placement:
        scores[seat] = neural_mean
    for seat in range(records.SEAT_COUNT):
        if seat not in placement:
            scores[seat] = -neural_mean
    return records.TerminalRecord(
        seed=seed,
        neural_seats=placement,
        scores=tuple(scores),
        winners=winners,
        end_reason=end_reason,
        decisions=decisions,
    )


def _full_records(seed: int) -> list[records.TerminalRecord]:
    result = []
    for index, placement in enumerate(records.PLACEMENTS):
        result.append(
            _record(
                seed,
                placement,
                neural_mean=index + (seed - 1),
                decisions=10 + index + 10 * (seed - 1),
                winners=(0,) if index % 2 == 0 else (0, 1, 2),
                end_reason=(
                    "wall_exhausted" if index % 2 == 0 else "three_winners"
                ),
            )
        )
    return result


def _record_line(record: records.TerminalRecord) -> str:
    return json.dumps(
        record.to_dict(), sort_keys=True, separators=(",", ":")
    ) + "\n"


def _with_duplicate_line(lines: list[str], _record) -> str:
    return "\n".join([lines[0], *lines]) + "\n"


def _with_malformed_tail(lines: list[str], _record) -> str:
    return "\n".join([*lines, "{"]) + "\n"


def _with_duplicate_json_key(_lines: list[str], _record) -> str:
    return (
        '{"seed":1,"neural_seats":[0,1],"scores":[1,-1,0,0],'
        '"winners":[0],"end_reason":"wall_exhausted","decisions":10,'
        '"seed":1}\n'
    )


def _with_nonzero_scores(_lines: list[str], record) -> str:
    value = record.to_dict()
    value["scores"] = [3, -1, -1, 0]
    return json.dumps(value) + "\n"


@pytest.fixture(scope="module")
def policy_identity(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("policy-checkpoint")
    path = directory / "policy.pt"
    build_checkpoint(
        path,
        role=ROLE_POLICY,
        hidden=4,
        model_seed=5,
        metrics={"nll": 0.75},
    )
    loaded, identity = records.load_checkpoint_with_identity(path)
    assert loaded.role == ROLE_POLICY
    return identity


def _append_full_run(directory: Path, plan: records.EvaluationPlan) -> None:
    writer = records.create_run(directory, plan)
    for seed in plan.seeds:
        for record in _full_records(seed):
            writer.append(record)
    writer.close()


def test_plan_roundtrip_preserves_raw_rust_metadata_and_extra_fields(
    tmp_path, policy_identity
):
    metadata = records.build_plan_metadata(
        checkpoint=policy_identity,
        rust_runtime=_RUST_RUNTIME,
        split_algorithm="seed-sha256-80-10-10-v1",
        selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
        teacher_cache={"scope": "shared"},
        dependencies={"python": "3.12", "torch": "2.14"},
        training_provenance=None,
        python_source=_FAKE_PYTHON_SOURCE,
        extra={
            "invocation": ["python", "-m", "mahjong_eval"],
            "environment": {"device": "cpu"},
        },
    )
    plan = records.EvaluationPlan(
        seeds=(3, 5), metadata=metadata, decision_limit=100
    )
    path = tmp_path / "plan.json"
    records.write_plan(plan, path)

    loaded = records.read_plan(path)
    assert loaded == plan
    assert loaded.metadata["rust_runtime"] == _RUST_RUNTIME
    assert loaded.metadata["invocation"] == ["python", "-m", "mahjong_eval"]
    assert loaded.metadata["environment"] == {"device": "cpu"}
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == records.PLAN_FORMAT
    assert document["placements"] == [list(pair) for pair in records.PLACEMENTS]


def test_single_seed_summary_reports_unavailable_se(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    _append_full_run(directory, plan)
    summary = records.summarize_run(directory)
    assert summary["seeds"] == 1
    assert summary["games"] == 6
    assert summary["neural_mean"] == pytest.approx(2.5)
    assert summary["delta"] == pytest.approx(5.0)
    assert summary["neural_mean_se"] is None
    assert summary["delta_se"] is None
    assert len(summary["seed_blocks"]) == 1


def test_summary_is_seed_blocked_and_recomputed_from_persisted_records(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity)
    directory = tmp_path / "run"
    _append_full_run(directory, plan)

    summary = records.summarize_run(directory)
    assert summary["primary_metric"] == "neural_mean"
    assert summary["seeds"] == 2
    assert summary["games"] == 12
    assert summary["placements_per_seed"] == 6
    assert summary["neural_mean"] == pytest.approx(3.0)
    assert summary["heuristic_mean"] == pytest.approx(-3.0)
    assert summary["delta"] == pytest.approx(6.0)
    assert summary["neural_mean_se"] == pytest.approx(0.5)
    assert summary["delta_se"] == pytest.approx(1.0)
    assert summary["spread"]["neural_seat_score_sd"] == pytest.approx(
        (76 / 23) ** 0.5
    )
    assert summary["spread"]["neural_game_mean_sd"] == pytest.approx(
        (38 / 11) ** 0.5
    )
    assert summary["finish_reasons"] == {
        "three_winners": 6,
        "wall_exhausted": 6,
    }
    assert summary["decision_counts"] == {
        "total": 210,
        "mean": pytest.approx(17.5),
        "min": 10,
        "max": 25,
    }
    blocks = summary["seed_blocks"]
    assert [block["seed"] for block in blocks] == [1, 2]
    assert [block["neural_mean"] for block in blocks] == pytest.approx(
        [2.5, 3.5]
    )
    assert [block["delta"] for block in blocks] == pytest.approx([5.0, 7.0])
    assert blocks[0]["finish_reasons"] == {
        "three_winners": 3,
        "wall_exhausted": 3,
    }
    assert blocks[0]["decision_counts"]["total"] == 75
    assert blocks[1]["decision_counts"]["total"] == 135

    # Change one persisted row and recompute without any in-memory game list.
    paths = records.run_paths(directory)
    lines = paths.records.read_text(encoding="utf-8").splitlines()
    replacement = _record(
        1,
        (0, 1),
        neural_mean=101,
        decisions=30,
        winners=(0,),
        end_reason="wall_exhausted",
    )
    lines[0] = _record_line(replacement).strip()
    paths.records.write_text("\n".join(lines) + "\n", encoding="utf-8")

    changed = records.summarize_run(directory)
    assert changed["neural_mean"] == pytest.approx(274 / 24)
    assert changed["heuristic_mean"] == pytest.approx(-274 / 24)
    assert changed["delta"] == pytest.approx(2 * 274 / 24)
    assert changed["decision_counts"]["total"] == 230
    assert changed["seed_blocks"][0]["neural_mean"] == pytest.approx(232 / 12)


def test_finalize_publishes_only_a_complete_recomputed_summary(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity)
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    for seed in plan.seeds:
        for record in _full_records(seed):
            writer.append(record)
    summary = writer.finalize()
    paths = records.run_paths(directory)
    assert paths.summary.exists()
    assert json.loads(paths.summary.read_text(encoding="utf-8")) == summary
    assert records.summarize_run(directory) == summary
    assert len(records.read_records(paths.records, plan, require_complete=True)) == 12


def test_missing_record_blocks_summary_and_keeps_completed_records(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity)
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    records_to_write = [
        record
        for seed in plan.seeds
        for record in _full_records(seed)
    ][:-1]
    for record in records_to_write:
        writer.append(record)
    writer.close()

    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    paths = records.run_paths(directory)
    assert not paths.summary.exists()
    assert len(records.read_records(paths.records, plan)) == len(records_to_write)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param(_with_duplicate_line, "duplicate", id="duplicate-line"),
        pytest.param(
            _with_malformed_tail, "not valid JSON", id="malformed-json"
        ),
        pytest.param(
            _with_duplicate_json_key,
            "duplicate JSON key",
            id="duplicate-json-key",
        ),
        pytest.param(
            _with_nonzero_scores, "sum to zero", id="nonzero-score-sum"
        ),
    ],
)
def test_duplicate_and_corrupt_records_block_summary(
    tmp_path, policy_identity, mutation, message
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    _append_full_run(directory, plan)
    paths = records.run_paths(directory)
    lines = paths.records.read_text(encoding="utf-8").splitlines()
    paths.records.write_text(
        mutation(lines, _full_records(1)[0]), encoding="utf-8"
    )
    with pytest.raises(records.RecordError, match=message):
        records.finalize_run(directory)
    assert not paths.summary.exists()


def test_record_write_failure_keeps_completed_records_and_blocks_summary(
    tmp_path, policy_identity, monkeypatch
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    all_records = _full_records(1)
    writer.append(all_records[0])

    original = records._append_line
    calls = {"count": 0}

    def failing_append(handle, line):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("disk full")
        return original(handle, line)

    monkeypatch.setattr(records, "_append_line", failing_append)
    with pytest.raises(records.RecordError, match="failed to write"):
        writer.append(all_records[1])
    assert writer.failed is True
    with pytest.raises(records.RecordError, match="cannot continue"):
        writer.append(all_records[2])

    writer.close()
    paths = records.run_paths(directory)
    assert len(paths.records.read_text(encoding="utf-8").splitlines()) == 1
    assert records.read_records(paths.records, plan) == [all_records[0]]
    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    assert not paths.summary.exists()


def test_fsync_failure_after_complete_write_keeps_records_but_blocks_public_finalize(
    tmp_path, policy_identity, monkeypatch
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    all_records = _full_records(1)
    for record in all_records[:-1]:
        writer.append(record)

    real_fsync = os.fsync

    def failing_fsync(descriptor):
        real_fsync(descriptor)
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(records.os, "fsync", failing_fsync)
    with pytest.raises(records.RecordError, match="failed to write"):
        writer.append(all_records[-1])
    assert writer.failed is True
    monkeypatch.setattr(records.os, "fsync", real_fsync)

    writer.close()
    paths = records.run_paths(directory)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == (
        records.STATE_INCOMPLETE
    )
    assert len(paths.records.read_text(encoding="utf-8").splitlines()) == 6
    assert records.read_records(
        paths.records, plan, require_complete=True
    ) == all_records
    with pytest.raises(records.RecordError, match="refusing to publish"):
        writer.finalize()
    # The separate recompute interface remains usable for diagnosis without
    # publishing a complete run artifact.
    assert records.summarize_run(directory)["games"] == 6
    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    assert not paths.summary.exists()


class _CloseFailingHandle:
    def __init__(self, handle):
        self._handle = handle
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def close(self):
        self._handle.close()
        self.closed = True
        raise OSError("simulated close failure")


def test_close_failure_after_complete_records_keeps_records_but_blocks_public_finalize(
    tmp_path, policy_identity, monkeypatch
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    opened: dict[str, _CloseFailingHandle] = {}
    original_open = records._open_records_file

    def wrapping_open(path):
        handle = _CloseFailingHandle(original_open(path))
        opened["handle"] = handle
        return handle

    monkeypatch.setattr(records, "_open_records_file", wrapping_open)
    writer = records.create_run(directory, plan)
    all_records = _full_records(1)
    for record in all_records:
        writer.append(record)
    assert opened["handle"].closed is False

    with pytest.raises(records.RecordError, match="failed to close"):
        writer.close()
    assert writer.failed is True
    assert opened["handle"].closed is True
    with pytest.raises(records.RecordError, match="refusing to publish"):
        writer.finalize()

    paths = records.run_paths(directory)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == (
        records.STATE_INCOMPLETE
    )
    assert len(paths.records.read_text(encoding="utf-8").splitlines()) == 6
    assert records.read_records(
        paths.records, plan, require_complete=True
    ) == all_records
    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    assert not paths.summary.exists()


def test_public_finalizer_requires_a_cleanly_closed_run(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    for record in _full_records(1):
        writer.append(record)
    paths = records.run_paths(directory)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == (
        records.STATE_INCOMPLETE
    )
    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    assert not paths.summary.exists()

    writer.close()
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == (
        records.STATE_COMPLETE
    )
    summary = records.finalize_run(directory)
    assert summary["games"] == 6
    assert json.loads(paths.summary.read_text(encoding="utf-8")) == summary


def test_context_exception_after_complete_records_keeps_run_incomplete(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="game failed"):
        with records.create_run(directory, plan) as writer:
            for record in _full_records(1):
                writer.append(record)
            raise RuntimeError("game failed")

    paths = records.run_paths(directory)
    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == (
        records.STATE_INCOMPLETE
    )
    assert len(
        records.read_records(paths.records, plan, require_complete=True)
    ) == 6
    with pytest.raises(records.RecordError, match="incomplete"):
        records.finalize_run(directory)
    assert not paths.summary.exists()


def test_summary_write_failure_publishes_nothing_and_keeps_records(
    tmp_path, policy_identity, monkeypatch
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    _append_full_run(directory, plan)
    paths = records.run_paths(directory)
    original = records._atomic_write_json

    def failing_write(path, value):
        if Path(path).name == records.SUMMARY_FILENAME:
            raise OSError("disk full")
        return original(path, value)

    monkeypatch.setattr(records, "_atomic_write_json", failing_write)
    with pytest.raises(records.RecordError, match="failed to write summary"):
        records.finalize_run(directory)
    assert not paths.summary.exists()
    assert len(records.read_records(paths.records, plan)) == 6

    monkeypatch.setattr(records, "_atomic_write_json", original)
    summary = records.finalize_run(directory)
    assert paths.summary.exists()
    assert json.loads(paths.summary.read_text(encoding="utf-8")) == summary


def test_plan_is_published_before_records_open_and_existing_runs_are_refused(
    tmp_path, policy_identity, monkeypatch
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    original = records._open_records_file

    def failing_open(path):
        raise OSError("cannot open")

    monkeypatch.setattr(records, "_open_records_file", failing_open)
    with pytest.raises(records.RecordError, match="failed to open"):
        records.create_run(directory, plan)
    assert (directory / records.PLAN_FILENAME).exists()
    assert not (directory / records.RECORDS_FILENAME).exists()

    monkeypatch.setattr(records, "_open_records_file", original)
    with pytest.raises(records.RecordError, match="already contains"):
        records.create_run(directory, plan)


@pytest.mark.parametrize(
    ("case", "bad_record", "message"),
    [
        pytest.param(
            "wrong-seed",
            _record(
                2,
                (0, 1),
                neural_mean=1,
                decisions=12,
                winners=(0,),
                end_reason="wall_exhausted",
            ),
            "not a planned game seed",
            id="wrong-seed",
        ),
        pytest.param(
            "too-many-decisions",
            _record(
                1,
                (0, 2),
                neural_mean=1,
                decisions=13,
                winners=(0,),
                end_reason="wall_exhausted",
            ),
            "exceeds the configured limit",
            id="too-many-decisions",
        ),
    ],
)
def test_append_rejects_plan_mismatches_and_poisons_the_run(
    tmp_path, policy_identity, case, bad_record, message
):
    plan = _plan(policy_identity, seeds=(1,), decision_limit=12)
    directory = tmp_path / case
    writer = records.create_run(directory, plan)
    with pytest.raises(records.RecordError, match=message):
        writer.append(bad_record)
    assert writer.failed is True
    with pytest.raises(records.RecordError, match="cannot continue"):
        writer.append(bad_record)
    paths = records.run_paths(directory)
    assert paths.records.read_text(encoding="utf-8") == ""
    writer.close()
    with pytest.raises(records.RecordError, match="refusing to publish"):
        writer.finalize()
    assert not paths.summary.exists()


def test_duplicate_append_poisons_the_run_but_keeps_the_first_line(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    valid = _record(
        1,
        (0, 1),
        neural_mean=1,
        decisions=12,
        winners=(0,),
        end_reason="wall_exhausted",
    )
    writer.append(valid)
    with pytest.raises(records.RecordError, match="duplicate"):
        writer.append(valid)
    assert writer.failed is True
    writer.close()
    paths = records.run_paths(directory)
    assert len(paths.records.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(records.RecordError, match="refusing to publish"):
        writer.finalize()
    assert not paths.summary.exists()


def test_read_records_rejects_unknown_record_keys(tmp_path, policy_identity):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    writer = records.create_run(directory, plan)
    writer.close()
    record = _full_records(1)[0].to_dict()
    record["unexpected"] = 1
    paths = records.run_paths(directory)
    paths.records.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(records.RecordError, match="unknown"):
        records.read_records(paths.records, plan)


@pytest.mark.parametrize(
    "changes, message",
    [
        pytest.param(
            {"scores": (1, -1, 0, 1)}, "sum to zero", id="scores-nonzero"
        ),
        pytest.param(
            {"scores": (1, -1, 0)}, "4 entries", id="scores-count"
        ),
        pytest.param(
            {"scores": (True, -1, 0, 0)},
            "score must be an integer",
            id="scores-bool",
        ),
        pytest.param(
            {"winners": (0, 0)}, "more than once", id="winner-duplicate"
        ),
        pytest.param(
            {"end_reason": "three_winners", "winners": (0,)},
            "three winners",
            id="three-winners-count",
        ),
        pytest.param(
            {"end_reason": "wall_exhausted", "winners": (0, 1, 2)},
            "fewer than three",
            id="wall-exhausted-count",
        ),
        pytest.param(
            {"end_reason": "draw"},
            "unsupported end_reason",
            id="unsupported-end-reason",
        ),
        pytest.param({"decisions": 0}, "at least 1", id="zero-decisions"),
        pytest.param({"seed": -1}, "at least 0", id="negative-seed"),
        pytest.param(
            {"neural_seats": (0, 0)}, "distinct", id="duplicate-neural-seat"
        ),
        pytest.param(
            {"neural_seats": (0, 4)}, "at most 3", id="neural-seat-range"
        ),
        pytest.param(
            {"neural_seats": (0,)}, "2 seats", id="neural-seat-count"
        ),
        pytest.param({"winners": (4,)}, "at most 3", id="winner-range"),
        pytest.param(
            {"scores": (2**31, -(2**31), 0, 0)},
            "at most",
            id="score-range",
        ),
    ],
)
def test_terminal_record_rejects_invalid_shape(changes, message):
    value = {
        "seed": 1,
        "neural_seats": (0, 1),
        "scores": (1, -1, 0, 0),
        "winners": (0,),
        "end_reason": "wall_exhausted",
        "decisions": 10,
    }
    value.update(changes)
    with pytest.raises(records.RecordError, match=message):
        records.TerminalRecord(**value)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda metadata: metadata.pop("rust_runtime"),
            id="missing-rust-runtime",
        ),
        pytest.param(
            lambda metadata: metadata["checkpoint"].__setitem__(
                "sha256", "0" * 63
            ),
            id="short-checkpoint-sha",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("encoding_version", 2),
            id="encoding-version",
        ),
        pytest.param(
            lambda metadata: metadata["checkpoint"].__setitem__(
                "has_training_state", "yes"
            ),
            id="nonboolean-training-state",
        ),
        pytest.param(
            lambda metadata: metadata["selection"].pop("device"),
            id="missing-selection-device",
        ),
        pytest.param(
            lambda metadata: metadata["selection"].__setitem__(
                "batch_size", 0
            ),
            id="zero-selection-batch",
        ),
        pytest.param(
            lambda metadata: metadata["python_source"].pop(
                "online_evaluator"
            ),
            id="missing-online-source",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("training_provenance", 3),
            id="invalid-training-provenance",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("dependencies", []),
            id="dependencies-array",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("split_algorithm", ""),
            id="empty-split-algorithm",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("rust_runtime", None),
            id="null-rust-runtime",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("rust_runtime", {}),
            id="empty-rust-runtime",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("rust_runtime", []),
            id="array-rust-runtime",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("teacher_cache", "shared"),
            id="string-teacher-cache",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("teacher_cache", True),
            id="boolean-teacher-cache",
        ),
    ],
)
def test_plan_metadata_rejects_inconsistent_required_fields(
    policy_identity, mutate
):
    metadata = records.build_plan_metadata(
        checkpoint=policy_identity,
        rust_runtime=_RUST_RUNTIME,
        split_algorithm="seed-sha256-80-10-10-v1",
        selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
        teacher_cache={"scope": "shared"},
        dependencies={"python": "3.12"},
        training_provenance=None,
        python_source=_FAKE_PYTHON_SOURCE,
    )
    mutate(metadata)
    with pytest.raises(records.RecordError):
        records.EvaluationPlan(
            seeds=(1,), metadata=metadata, decision_limit=100
        )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda metadata: metadata.__setitem__("encoding_version", 2),
            id="encoding-version",
        ),
        pytest.param(
            lambda metadata: metadata["checkpoint"].__setitem__(
                "sha256", "0" * 63
            ),
            id="short-checkpoint-sha",
        ),
        pytest.param(
            lambda metadata: metadata["selection"].__setitem__(
                "batch_size", 0
            ),
            id="zero-selection-batch",
        ),
        pytest.param(
            lambda metadata: metadata["python_source"].pop(
                "online_evaluator"
            ),
            id="missing-online-source",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("rust_runtime", None),
            id="null-rust-runtime",
        ),
    ],
)
def test_write_plan_rejects_metadata_mutated_after_construction(
    tmp_path, policy_identity, mutate
):
    plan = _plan(policy_identity, seeds=(1,))
    mutate(plan.metadata)
    path = tmp_path / "nested" / "plan.json"
    with pytest.raises(records.RecordError):
        records.write_plan(plan, path)
    assert not (tmp_path / "nested").exists()


def test_write_plan_revalidation_does_not_replace_a_valid_plan(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    path = tmp_path / "plan.json"
    records.write_plan(plan, path)
    original = path.read_bytes()
    original_document = json.loads(original)

    plan.metadata["encoding_version"] = 2
    with pytest.raises(records.RecordError):
        records.write_plan(plan, path)
    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8")) == original_document
    assert records.read_plan(path).metadata["encoding_version"] == 1


def test_create_run_rejects_mutated_plan_before_creating_output(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    plan.metadata["checkpoint"]["sha256"] = "0" * 63
    directory = tmp_path / "run"
    with pytest.raises(records.RecordError):
        records.create_run(directory, plan)
    assert not directory.exists()


@pytest.mark.parametrize(
    "teacher_cache",
    [0, {"scope": "shared"}],
    ids=["disabled", "configured"],
)
def test_plan_metadata_accepts_disabled_or_configured_teacher_cache(
    tmp_path, policy_identity, teacher_cache
):
    metadata = records.build_plan_metadata(
        checkpoint=policy_identity,
        rust_runtime=_RUST_RUNTIME,
        split_algorithm="seed-sha256-80-10-10-v1",
        selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
        teacher_cache=teacher_cache,
        dependencies={"python": "3.12"},
        training_provenance=None,
        python_source=_FAKE_PYTHON_SOURCE,
    )
    plan = records.EvaluationPlan(
        seeds=(1,), metadata=metadata, decision_limit=100
    )
    path = tmp_path / "plan.json"
    records.write_plan(plan, path)
    loaded = records.read_plan(path)
    assert loaded.metadata["teacher_cache"] == teacher_cache


@pytest.mark.parametrize(
    "changes, message",
    [
        pytest.param({"seeds": ()}, "at least one", id="empty-seeds"),
        pytest.param({"seeds": (1, 1)}, "unique", id="duplicate-seeds"),
        pytest.param({"seeds": (-1,)}, "at least 0", id="negative-seed"),
        pytest.param({"seeds": (2**64,)}, "at most", id="seed-range"),
        pytest.param(
            {"decision_limit": 0}, "at least 1", id="zero-decision-limit"
        ),
        pytest.param(
            {"placements": ((0, 1),)}, "fixed six", id="placement-count"
        ),
        pytest.param(
            {"neural_policy": ""}, "nonempty", id="empty-neural-policy"
        ),
        pytest.param(
            {"heuristic_policy": ""},
            "nonempty",
            id="empty-heuristic-policy",
        ),
    ],
)
def test_plan_rejects_invalid_structure(policy_identity, changes, message):
    values = {
        "seeds": (1,),
        "metadata": _plan(policy_identity, seeds=(1,)).metadata,
        "decision_limit": 100,
    }
    values.update(changes)
    with pytest.raises(records.RecordError, match=message):
        records.EvaluationPlan(**values)


def test_read_plan_rejects_unsupported_format_version(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,))
    path = tmp_path / "plan.json"
    records.write_plan(plan, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["format_version"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(records.RecordError, match="unsupported plan format version"):
        records.read_plan(path)


def test_build_plan_metadata_rejects_extra_key_collisions(policy_identity):
    with pytest.raises(records.RecordError, match="cannot replace"):
        records.build_plan_metadata(
            checkpoint=policy_identity,
            rust_runtime=_RUST_RUNTIME,
            split_algorithm="seed-sha256-80-10-10-v1",
            selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
            teacher_cache={"scope": "shared"},
            dependencies={"python": "3.12"},
            training_provenance=None,
            python_source=_FAKE_PYTHON_SOURCE,
            extra={"checkpoint": {}},
        )


def test_checkpoint_identity_matches_exact_loaded_bytes(tmp_path):
    path = tmp_path / "policy.pt"
    original_model = build_checkpoint(
        path,
        role=ROLE_POLICY,
        hidden=4,
        model_seed=1,
        metrics={"nll": 0.5},
    ).model
    original_bytes = path.read_bytes()
    loaded, identity = records.load_checkpoint_with_identity(path)
    assert identity["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert identity["size_bytes"] == len(original_bytes)
    assert identity["role"] == ROLE_POLICY
    assert identity["metrics"] == {"nll": 0.5}
    assert identity["has_training_state"] is False
    assert identity["path"] == str(path)
    assert identity["path"] == str(path.resolve())
    assert records.training_provenance(loaded) is None

    # The recorded identity must still describe the weights already loaded,
    # not a replacement checkpoint now at the same path.
    build_checkpoint(
        path,
        role=ROLE_POLICY,
        hidden=4,
        model_seed=2,
        metrics={"nll": 9.0},
    )
    assert identity["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in original_model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], value)


def test_checkpoint_identity_records_resolved_path_and_content_hash(
    tmp_path, monkeypatch
):
    path = tmp_path / "policy.pt"
    build_checkpoint(
        path,
        role=ROLE_POLICY,
        hidden=4,
        model_seed=4,
        metrics={"nll": 0.25},
    )

    monkeypatch.chdir(tmp_path)
    loaded, identity = records.load_checkpoint_with_identity("policy.pt")
    assert loaded.role == ROLE_POLICY
    assert identity["path"] == str(path.resolve())
    assert Path(identity["path"]).is_absolute()
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_provenance_is_separate_from_evaluation_identity(tmp_path):
    model = build_model(hidden=4, seed=3)
    state = build_training_state(
        model,
        cache_id="fixture-cache",
        train_eligible_decisions=3,
        training_seed=1,
        batch_size=2,
        global_step=2,
        samples_seen=4,
        best_validation_nll=0.5,
    )
    path = tmp_path / "best.pt"
    build_checkpoint(
        path,
        role=ROLE_BEST,
        hidden=4,
        model_seed=3,
        model=model,
        training=state,
        metrics={"nll": 0.5},
    )
    loaded, identity = records.load_checkpoint_with_identity(path)
    assert identity["has_training_state"] is True
    assert "optimizer_state" not in identity
    assert "training_config" not in identity

    provenance = records.training_provenance(loaded)
    assert provenance is not None
    assert provenance["train_dataset_identity"]["split"] == "train"
    assert provenance["validation_dataset_identity"]["split"] == "validation"
    assert provenance["completed_epochs"] == 1
    assert "optimizer_state" not in provenance
    assert "torch_rng_state" not in provenance
    assert "best_model_state" not in provenance

    plan = _plan(
        identity,
        seeds=(1,),
        metadata=records.build_plan_metadata(
            checkpoint=identity,
            rust_runtime=_RUST_RUNTIME,
            split_algorithm="seed-sha256-80-10-10-v1",
            selection={"mode": "greedy", "device": "cpu", "batch_size": 4},
            teacher_cache={"scope": "shared"},
            dependencies={"python": "3.12"},
            training_provenance=provenance,
            python_source=_FAKE_PYTHON_SOURCE,
        ),
    )
    path = tmp_path / "plan.json"
    records.write_plan(plan, path)
    loaded_plan = records.read_plan(path)
    assert loaded_plan.metadata["training_provenance"] == provenance
    assert "training_provenance" not in loaded_plan.metadata["checkpoint"]


def test_python_source_records_both_packages_independent_of_cwd(
    tmp_path, monkeypatch
):
    source = records.collect_python_source()
    assert set(source) == {"online_evaluator", "offline_policy"}
    online = source["online_evaluator"]
    offline = source["offline_policy"]
    assert online["package"] == "mahjong-eval"
    assert offline["package"] == "mahjong-bc"
    assert online["hash"] == hash_source_directory(Path(records.__file__).parent)
    assert offline["hash"] == hash_source_directory(
        Path(mahjong_bc.__file__).parent
    )
    assert offline == collect_source_record(Path(mahjong_bc.__file__).parent)

    monkeypatch.chdir(tmp_path)
    assert records.collect_python_source() == source


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git executable is not available")
    return subprocess.run(
        [git, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable is not available"
)


@requires_git
def test_python_source_uses_package_directories_and_git_metadata(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    offline = root / "python" / "mahjong_bc"
    online = root / "online" / "mahjong_eval"
    offline.mkdir(parents=True)
    online.mkdir(parents=True)
    (offline / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    (online / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Source Test",
        "-c",
        "user.email=source-test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    first = records.collect_python_source(
        offline_directory=offline,
        online_directory=online,
        offline_package_version="1.2.3",
        online_package_version="4.5.6",
    )
    for role in ("online_evaluator", "offline_policy"):
        assert first[role]["revision"] == head
        assert first[role]["dirty"] is False
    assert first["offline_policy"]["package_version"] == "1.2.3"
    assert first["online_evaluator"]["package_version"] == "4.5.6"

    (online / "extra.py").write_text("value = 2\n", encoding="utf-8")
    second = records.collect_python_source(
        offline_directory=offline,
        online_directory=online,
        offline_package_version="1.2.3",
        online_package_version="4.5.6",
    )
    assert second["online_evaluator"]["hash"] != first["online_evaluator"]["hash"]
    assert second["offline_policy"]["hash"] == first["offline_policy"]["hash"]
    assert second["online_evaluator"]["dirty"] is True

    (online / "extra.py").unlink()
    (offline / "extra.py").write_text("value = 2\n", encoding="utf-8")
    third = records.collect_python_source(
        offline_directory=offline,
        online_directory=online,
        offline_package_version="1.2.3",
        online_package_version="4.5.6",
    )
    assert third["online_evaluator"]["hash"] == first["online_evaluator"]["hash"]
    assert third["offline_policy"]["hash"] != first["offline_policy"]["hash"]


@requires_git
def test_python_source_does_not_borrow_unrelated_outer_git(tmp_path):
    root = tmp_path / "outer"
    offline = root / "python" / "mahjong_bc"
    online = root / "vendor" / "mahjong_eval"
    offline.mkdir(parents=True)
    online.mkdir(parents=True)
    (offline / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    (online / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Source Test",
        "-c",
        "user.email=source-test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    (offline / "__init__.py").write_text("value = 2\n", encoding="utf-8")

    source = records.collect_python_source(
        offline_directory=offline,
        online_directory=online,
        offline_package_version="1.2.3",
        online_package_version="4.5.6",
    )
    assert source["online_evaluator"]["revision"] is None
    assert source["online_evaluator"]["dirty"] is None
    # The offline package does sit at its expected repository path, so the same
    # outer repository legitimately supplies its revision and dirty flag.
    assert source["offline_policy"]["revision"] == head
    assert source["offline_policy"]["dirty"] is True
