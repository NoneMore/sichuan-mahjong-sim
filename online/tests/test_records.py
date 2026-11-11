from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import mahjong_bc
from mahjong_bc._source import collect_source_record, hash_source_directory
from mahjong_bc.checkpoint import TrainingCheckpointState, save_checkpoint
from mahjong_bc.config import (
    DatasetIdentity,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from mahjong_bc.constants import ROLE_BEST, ROLE_POLICY
from mahjong_bc.policy import build_policy

# Load this checkout's module directly. An installed ``mahjong_eval`` regular
# package would otherwise shadow this checkout's namespace package, and its
# package initializer imports the native extension; the records tests must stay
# independent of that extension.
_RECORDS_PATH = Path(__file__).resolve().parents[1] / "mahjong_eval" / "records.py"
_RECORDS_SPEC = importlib.util.spec_from_file_location(
    "mahjong_eval_worktree_records", _RECORDS_PATH
)
assert _RECORDS_SPEC is not None and _RECORDS_SPEC.loader is not None
records = importlib.util.module_from_spec(_RECORDS_SPEC)
sys.modules[_RECORDS_SPEC.name] = records
_RECORDS_SPEC.loader.exec_module(records)

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


@pytest.fixture(scope="module")
def policy_identity(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("policy-checkpoint")
    path = directory / "policy.pt"
    model = build_policy(
        ModelConfig(state_hidden=(4,), scorer_hidden=(4,)), seed=5
    )
    save_checkpoint(path, model, role=ROLE_POLICY, metrics={"nll": 0.75})
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


def test_duplicate_and_corrupt_records_block_summary(tmp_path, policy_identity):
    plan = _plan(policy_identity, seeds=(1,))
    directory = tmp_path / "run"
    _append_full_run(directory, plan)
    paths = records.run_paths(directory)
    lines = paths.records.read_text(encoding="utf-8").splitlines()

    paths.records.write_text(
        "\n".join([lines[0], *lines]) + "\n", encoding="utf-8"
    )
    with pytest.raises(records.RecordError, match="duplicate"):
        records.finalize_run(directory)
    assert not paths.summary.exists()

    paths.records.write_text(
        "\n".join([*lines, "{"]) + "\n", encoding="utf-8"
    )
    with pytest.raises(records.RecordError, match="not valid JSON"):
        records.finalize_run(directory)
    assert not paths.summary.exists()

    duplicate_key = (
        '{"seed":1,"neural_seats":[0,1],"scores":[1,-1,0,0],'
        '"winners":[0],"end_reason":"wall_exhausted","decisions":10,'
        '"seed":1}\n'
    )
    paths.records.write_text(duplicate_key, encoding="utf-8")
    with pytest.raises(records.RecordError, match="duplicate JSON key"):
        records.finalize_run(directory)
    assert not paths.summary.exists()

    record = _full_records(1)[0].to_dict()
    record["scores"] = [3, -1, -1, 0]
    paths.records.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(records.RecordError, match="sum to zero"):
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


def test_append_rejects_plan_mismatches_and_poisons_the_run(
    tmp_path, policy_identity
):
    plan = _plan(policy_identity, seeds=(1,), decision_limit=12)
    bad_records = [
        (
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
        ),
        (
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
        ),
    ]
    for name, bad_record, message in bad_records:
        directory = tmp_path / name
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
        ({"scores": (1, -1, 0, 1)}, "sum to zero"),
        ({"scores": (1, -1, 0)}, "4 entries"),
        ({"scores": (True, -1, 0, 0)}, "score must be an integer"),
        ({"winners": (0, 0)}, "more than once"),
        ({"end_reason": "three_winners", "winners": (0,)}, "three winners"),
        (
            {"end_reason": "wall_exhausted", "winners": (0, 1, 2)},
            "fewer than three",
        ),
        ({"end_reason": "draw"}, "unsupported end_reason"),
        ({"decisions": 0}, "at least 1"),
        ({"seed": -1}, "at least 0"),
        ({"neural_seats": (0, 0)}, "distinct"),
        ({"neural_seats": (0, 4)}, "at most 3"),
        ({"neural_seats": (0,)}, "2 seats"),
        ({"winners": (4,)}, "at most 3"),
        ({"scores": (2**31, -(2**31), 0, 0)}, "at most"),
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
        lambda metadata: metadata.pop("rust_runtime"),
        lambda metadata: metadata["checkpoint"].__setitem__("sha256", "0" * 63),
        lambda metadata: metadata.__setitem__("encoding_version", 2),
        lambda metadata: metadata["checkpoint"].__setitem__(
            "has_training_state", "yes"
        ),
        lambda metadata: metadata["selection"].pop("device"),
        lambda metadata: metadata["selection"].__setitem__("batch_size", 0),
        lambda metadata: metadata["python_source"].pop("online_evaluator"),
        lambda metadata: metadata.__setitem__("training_provenance", 3),
        lambda metadata: metadata.__setitem__("dependencies", []),
        lambda metadata: metadata.__setitem__("split_algorithm", ""),
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
    "changes, message",
    [
        ({"seeds": ()}, "at least one"),
        ({"seeds": (1, 1)}, "unique"),
        ({"seeds": (-1,)}, "at least 0"),
        ({"seeds": (2**64,)}, "at most"),
        ({"decision_limit": 0}, "at least 1"),
        ({"placements": ((0, 1),)}, "fixed six"),
        ({"neural_policy": ""}, "nonempty"),
        ({"heuristic_policy": ""}, "nonempty"),
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
    original_model = build_policy(
        ModelConfig(state_hidden=(4,), scorer_hidden=(4,)), seed=1
    )
    save_checkpoint(
        path, original_model, role=ROLE_POLICY, metrics={"nll": 0.5}
    )
    original_bytes = path.read_bytes()
    loaded, identity = records.load_checkpoint_with_identity(path)
    assert identity["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert identity["size_bytes"] == len(original_bytes)
    assert identity["role"] == ROLE_POLICY
    assert identity["metrics"] == {"nll": 0.5}
    assert identity["has_training_state"] is False
    assert identity["path"] == str(path)
    assert records.training_provenance(loaded) is None

    # The recorded identity must still describe the weights already loaded,
    # not a replacement checkpoint now at the same path.
    replacement = build_policy(
        ModelConfig(state_hidden=(4,), scorer_hidden=(4,)), seed=2
    )
    save_checkpoint(
        path, replacement, role=ROLE_POLICY, metrics={"nll": 9.0}
    )
    assert identity["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in original_model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], value)


def test_training_provenance_is_separate_from_evaluation_identity(tmp_path):
    model = build_policy(
        ModelConfig(state_hidden=(4,), scorer_hidden=(4,)), seed=3
    )
    train_identity = DatasetIdentity(
        cache_id="fixture-cache",
        split="train",
        encoding_version=1,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=4,
        eligible_decisions=3,
    )
    validation_identity = DatasetIdentity(
        cache_id="fixture-cache",
        split="validation",
        encoding_version=1,
        preparation_format_version=1,
        source_revision="fixture-revision",
        decisions=2,
        eligible_decisions=1,
    )
    state = TrainingCheckpointState(
        training_config=TrainingConfig(
            seed=1, epochs=1, batch_size=2, learning_rate=0.01
        ),
        train_dataset_identity=train_identity,
        validation_dataset_identity=validation_identity,
        dependencies=collect_dependency_versions(),
        device_type="cpu",
        completed_epochs=1,
        global_step=2,
        samples_seen=4,
        best_validation_nll=0.5,
        best_epoch=0,
        history=[{"epoch": 0, "steps": 2, "samples": 4}],
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.01}]},
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=None,
        order_rng_state=torch.Generator().manual_seed(7).get_state(),
        best_model_state={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        best_metrics={"nll": 0.5},
    )
    path = tmp_path / "best.pt"
    save_checkpoint(
        path, model, role=ROLE_BEST, training=state, metrics={"nll": 0.5}
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
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


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
