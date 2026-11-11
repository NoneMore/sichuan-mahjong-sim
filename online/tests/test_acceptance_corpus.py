"""Optional read-only probe against an external acceptance corpus.

Set ``MAHJONG_EVAL_ACCEPTANCE_CORPUS`` to a corpus directory whose first shard
starts at seed 0 (for example the parent acceptance artifact). The test never
writes to the corpus. It compares the environment's full seed-0 trajectory and
the native teacher replay with the recorded state, candidate order, semantic
action, and terminal result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import mahjong_eval

CORPUS_ENV = "MAHJONG_EVAL_ACCEPTANCE_CORPUS"


def first_episode(corpus: Path) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    shards = manifest["shards"]
    assert shards, "corpus manifest has no shards"
    first_seed = shards[0]["first_seed"]
    assert first_seed == "0000000000000000", (
        f"acceptance probe expects a seed-0 first shard, got {first_seed}"
    )
    shard = corpus / "shards" / f"{first_seed}.jsonl"
    episode = None
    decisions: list[dict] = []
    result = None
    for line in shard.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "episode":
            if episode is not None:
                break
            episode = record
        elif record["type"] == "decision":
            decisions.append(record)
        elif record["type"] == "result":
            result = record
            break
    assert episode is not None and result is not None and decisions
    return episode, decisions, result


def same_f32(actual: np.ndarray, values) -> None:
    expected = np.asarray(values, dtype="<f4")
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_external_corpus_handoff_matches_environment_and_teacher() -> None:
    configured = os.environ.get(CORPUS_ENV)
    if not configured:
        pytest.skip(f"set {CORPUS_ENV} to run the external corpus probe")
    corpus = Path(configured)
    if not (corpus / "manifest.json").is_file():
        pytest.skip(f"acceptance corpus is unavailable: {corpus}")
    episode, records, result = first_episode(corpus)
    seed = int(episode["episode_id"][2], 16)

    native = mahjong_eval.native_teacher_replay(seed)
    assert native["seed"] == seed
    assert native["decisions"] == result["decisions"] == len(records)
    assert list(native["scores"]) == result["scores"]
    assert list(native["winners"]) == result["winners"]
    assert native["end_reason"] == result["end_reason"]

    env = mahjong_eval.Env()
    decision = env.reset(seed)
    for position, (record, native_entry) in enumerate(
        zip(records, native["trace"])
    ):
        same_f32(decision.state, record["state"])
        same_f32(decision.candidate_features, record["candidate_features"])
        assert decision.actor_seat == record["actor_seat"]
        assert decision.stage == record["stage"]
        assert decision.candidate_count == len(record["candidate_actions"])
        assert decision.decision_id == native_entry["decision_id"]
        index = env.teacher_index()
        assert index == record["teacher_target_index"]
        assert native_entry["teacher_index"] == index
        assert native_entry["action"]["wire"] == record["candidate_actions"][index]

        outcome = env.step(decision.decision_id, index)
        if position + 1 < len(records):
            assert isinstance(outcome, mahjong_eval.Decision)
            decision = outcome
        else:
            assert isinstance(outcome, mahjong_eval.Terminal)

    assert outcome.decisions == result["decisions"]
    assert list(outcome.scores) == result["scores"]
    assert list(outcome.winners) == result["winners"]
    assert outcome.end_reason == result["end_reason"]
