"""T1 acceptance: install, import, reset, and first-decision fixture equality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mahjong_eval
import mahjong_eval._native as native

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "crates"
    / "python"
    / "tests"
    / "fixtures"
    / "first_decision_seed_0.json"
)


def load_fixture() -> dict:
    assert FIXTURE.is_file(), f"Rust first-decision fixture is missing: {FIXTURE}"
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def f32_array(hex_text: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = bytes.fromhex(hex_text)
    values = np.frombuffer(raw, dtype="<f4")
    assert values.nbytes == int(np.prod(shape)) * 4
    return values.reshape(shape)


def test_native_module_and_versions() -> None:
    fixture = load_fixture()

    assert mahjong_eval.__version__ == native.__version__ == "0.1.0"
    assert mahjong_eval.ENCODING_VERSION == fixture["encoding_version"] == 1
    assert mahjong_eval.STATE_WIDTH == fixture["state_width"] == 1013
    assert mahjong_eval.CANDIDATE_WIDTH == fixture["candidate_width"] == 39
    assert mahjong_eval.DEFAULT_DECISION_LIMIT == 10_000
    assert native.Decision is mahjong_eval.Decision
    assert native.Env is mahjong_eval.Env


def test_reset_matches_rust_first_decision_fixture() -> None:
    fixture = load_fixture()
    decision = mahjong_eval.Env().reset(fixture["seed"])

    assert isinstance(decision, mahjong_eval.Decision)
    assert decision.decision_id == fixture["decision_id"]
    assert decision.actor_seat == fixture["actor_seat"]
    assert decision.stage == fixture["stage"]
    assert decision.candidate_count == fixture["candidate_count"]

    state = decision.state
    candidates = decision.candidate_features
    assert isinstance(state, np.ndarray)
    assert isinstance(candidates, np.ndarray)
    assert state.dtype == np.float32
    assert candidates.dtype == np.float32
    assert state.shape == (fixture["state_width"],)
    assert candidates.shape == (
        fixture["candidate_count"],
        fixture["candidate_width"],
    )
    assert state.flags["C_CONTIGUOUS"]
    assert candidates.flags["C_CONTIGUOUS"]
    assert state.flags["WRITEABLE"]
    assert candidates.flags["WRITEABLE"]
    assert state.flags["OWNDATA"]
    assert candidates.flags["OWNDATA"]
    assert state.base is None
    assert candidates.base is None

    expected_state = f32_array(fixture["state_le_f32_hex"], state.shape)
    expected_candidates = f32_array(
        fixture["candidate_features_le_f32_hex"], candidates.shape
    )
    np.testing.assert_array_equal(state.view(np.uint32), expected_state.view(np.uint32))
    np.testing.assert_array_equal(
        candidates.view(np.uint32), expected_candidates.view(np.uint32)
    )


def test_snapshots_are_independent_of_game_and_later_resets() -> None:
    env = mahjong_eval.Env()
    first = env.reset(0)
    state = first.state.copy()
    candidates = first.candidate_features.copy()

    # Writable Python-owned snapshots may be mutated freely; the authoritative
    # game and every later reset remain unchanged.
    first.state[:] = -123.0
    first.candidate_features[:] = 456.0

    second = env.reset(0)
    np.testing.assert_array_equal(second.state, state)
    np.testing.assert_array_equal(second.candidate_features, candidates)
    assert second.decision_id != first.decision_id
    assert second.decision_id == first.decision_id + (1 << 32)


def test_constructor_limit_and_argument_rejection() -> None:
    assert mahjong_eval.Env().decision_limit == 10_000
    assert mahjong_eval.Env(decision_limit=7).decision_limit == 7

    with pytest.raises(OverflowError):
        mahjong_eval.Env(decision_limit=-1)
    with pytest.raises(TypeError):
        mahjong_eval.Env(decision_limit=1.5)

    env = mahjong_eval.Env()
    with pytest.raises(OverflowError):
        env.reset(-1)
    with pytest.raises(TypeError):
        env.reset("0")


def test_reset_is_seed_deterministic_and_seed_sensitive() -> None:
    env = mahjong_eval.Env()
    one = env.reset(11)
    two = env.reset(11)
    other = env.reset(12)

    np.testing.assert_array_equal(one.state, two.state)
    np.testing.assert_array_equal(one.candidate_features, two.candidate_features)
    assert not np.array_equal(one.state, other.state)
