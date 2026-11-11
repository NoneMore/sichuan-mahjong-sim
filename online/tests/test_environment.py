"""Environment lifecycle and Rust-handoff acceptance for the online binding."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mahjong_eval
import mahjong_eval._native as native

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "crates" / "python" / "tests" / "fixtures"
FIRST_DECISION_FIXTURE = FIXTURES / "first_decision_seed_0.json"
TRACE_FIXTURE = FIXTURES / "environment_trace_seed_61.json"

# Frozen v1 state layout offsets used only to observe public win flags.
PLAYERS_OFFSET = 27
PLAYER_WIDTH = 155
PLAYER_HAS_WON_OFFSET = 1 + 4 + 1


def load_fixture(path: Path) -> dict:
    assert path.is_file(), f"Rust handoff fixture is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def f32_array(hex_text: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = bytes.fromhex(hex_text)
    values = np.frombuffer(raw, dtype="<f4")
    assert values.nbytes == int(np.prod(shape)) * 4
    return values.reshape(shape)


def assert_snapshot(
    decision: mahjong_eval.Decision,
    expected: dict,
    *,
    check_storage: bool = True,
) -> None:
    assert isinstance(decision, mahjong_eval.Decision)
    assert decision.decision_id == expected["decision_id"]
    assert decision.actor_seat == expected["actor_seat"]
    assert decision.stage == expected["stage"]
    assert decision.candidate_count == expected["candidate_count"]

    state = decision.state
    candidates = decision.candidate_features
    assert isinstance(state, np.ndarray) and state.dtype == np.float32
    assert isinstance(candidates, np.ndarray) and candidates.dtype == np.float32
    assert state.shape == (mahjong_eval.STATE_WIDTH,)
    assert candidates.shape == (
        expected["candidate_count"],
        mahjong_eval.CANDIDATE_WIDTH,
    )
    if check_storage:
        assert state.flags["C_CONTIGUOUS"]
        assert candidates.flags["C_CONTIGUOUS"]
        assert state.flags["WRITEABLE"]
        assert candidates.flags["WRITEABLE"]
        assert state.flags["OWNDATA"]
        assert candidates.flags["OWNDATA"]
        assert state.base is None
        assert candidates.base is None
        assert not np.shares_memory(state, candidates)

    np.testing.assert_array_equal(
        state.view(np.uint32),
        f32_array(expected["state_le_f32_hex"], state.shape).view(np.uint32),
    )
    np.testing.assert_array_equal(
        candidates.view(np.uint32),
        f32_array(
            expected["candidate_features_le_f32_hex"], candidates.shape
        ).view(np.uint32),
    )


def trace_states(fixture: dict) -> list[np.ndarray]:
    return [
        f32_array(item["state_le_f32_hex"], (fixture["state_width"],))
        for item in fixture["decisions"]
    ]


def visible_winner(state: np.ndarray) -> bool:
    """Whether any relative player group exposes the public won flag."""

    for relative in range(4):
        offset = PLAYERS_OFFSET + relative * PLAYER_WIDTH + PLAYER_HAS_WON_OFFSET
        if state[offset] == np.float32(1.0):
            return True
    return False


def test_native_module_and_versions() -> None:
    fixture = load_fixture(FIRST_DECISION_FIXTURE)

    assert mahjong_eval.__version__ == native.__version__ == "0.1.0"
    assert mahjong_eval.ENCODING_VERSION == fixture["encoding_version"] == 1
    assert mahjong_eval.STATE_WIDTH == fixture["state_width"] == 1013
    assert mahjong_eval.CANDIDATE_WIDTH == fixture["candidate_width"] == 39
    assert mahjong_eval.DEFAULT_DECISION_LIMIT == 10_000
    assert native.Decision is mahjong_eval.Decision
    assert native.Env is mahjong_eval.Env
    assert native.Terminal is mahjong_eval.Terminal
    assert issubclass(native.DecisionLimitError, native.MahjongEvalError)


def test_reset_matches_rust_first_decision_fixture() -> None:
    fixture = load_fixture(FIRST_DECISION_FIXTURE)
    decision = mahjong_eval.Env().reset(fixture["seed"])

    assert_snapshot(
        decision,
        {
            **fixture,
            "decision_id": fixture["decision_id"],
            "candidate_count": fixture["candidate_count"],
            "state_width": fixture["state_width"],
            "candidate_width": fixture["candidate_width"],
        },
    )


def test_scripted_trace_matches_rust_across_all_stages() -> None:
    fixture = load_fixture(TRACE_FIXTURE)
    decisions = fixture["decisions"]
    states = trace_states(fixture)
    terminal = fixture["terminal"]
    assert len(decisions) == terminal["decisions"]

    env = mahjong_eval.Env(decision_limit=fixture["decision_limit"])
    decision = env.reset(fixture["seed"])

    stages: set[int] = set()
    winner_indexes = []
    for index, expected in enumerate(decisions):
        assert_snapshot(decision, expected)
        stages.add(decision.stage)
        if visible_winner(states[index]):
            winner_indexes.append(index)
        next_id = expected["decision_id"]
        if index + 1 < len(decisions):
            outcome = env.step(next_id, expected["chosen_index"])
            assert isinstance(outcome, mahjong_eval.Decision)
            assert outcome.decision_id == decisions[index + 1]["decision_id"]
            decision = outcome
        else:
            result = env.step(next_id, expected["chosen_index"])
            assert isinstance(result, mahjong_eval.Terminal)
            assert not isinstance(result, mahjong_eval.Decision)

    assert stages == {0, 1, 2, 3, 4}, "the trace must cover every stage"
    assert winner_indexes, "the trace must expose a won player"
    assert winner_indexes[0] < len(decisions) - 1, (
        "a game must continue after the first winner rather than terminate early"
    )

    assert result.seed == fixture["seed"]
    assert result.decisions == len(decisions)
    assert result.end_reason == terminal["end_reason"]
    assert result.end_reason in {"three_winners", "wall_exhausted"}
    assert result.scores == tuple(terminal["scores"])
    assert result.winners == tuple(terminal["winners"])
    assert len(result.scores) == len(result.winners) == 4
    assert sum(result.scores) == 0

    with pytest.raises(native.EnvironmentStateError, match="terminal"):
        env.step(decisions[-1]["decision_id"], 0)

    # Reset is explicit, starts a fresh generation, and makes the previous
    # identity stale even when the new game reaches the same first decision.
    reset = env.reset(fixture["seed"])
    assert reset.decision_id == (2 << 32) | 1
    assert_snapshot(reset, {**decisions[0], "decision_id": reset.decision_id})
    with pytest.raises(native.StaleDecisionError, match="stale decision id"):
        env.step(decisions[-1]["decision_id"], 0)
    advanced = env.step(reset.decision_id, decisions[0]["chosen_index"])
    assert isinstance(advanced, mahjong_eval.Decision)
    assert advanced.decision_id == reset.decision_id + 1


def test_request_rejection_does_not_change_the_game() -> None:
    fixture = load_fixture(TRACE_FIXTURE)
    decisions = fixture["decisions"]
    env = mahjong_eval.Env()
    decision = env.reset(fixture["seed"])
    assert decision.decision_id == decisions[0]["decision_id"]

    with pytest.raises(native.CandidateIndexError, match="out of range"):
        env.step(decision.decision_id, decision.candidate_count)
    with pytest.raises(native.CandidateIndexError, match="out of range"):
        env.step(decision.decision_id, 1 << 30)
    with pytest.raises(native.StaleDecisionError, match="current pending decision"):
        env.step(decision.decision_id + 1, 0)
    with pytest.raises(native.StaleDecisionError, match="stale decision id"):
        env.step(decision.decision_id - 1, 0)

    # The authoritative game and cached candidates were not modified by any
    # rejected request: the fixture path continues exactly.
    advanced = env.step(decision.decision_id, decisions[0]["chosen_index"])
    assert isinstance(advanced, mahjong_eval.Decision)
    assert_snapshot(advanced, decisions[1])

    # A reset generation invalidates identities from the earlier game.
    old_id = advanced.decision_id
    env.reset(fixture["seed"])
    with pytest.raises(native.StaleDecisionError):
        env.step(old_id, 0)


def test_snapshots_are_owned_and_do_not_affect_authority() -> None:
    fixture = load_fixture(TRACE_FIXTURE)
    decisions = fixture["decisions"]
    env = mahjong_eval.Env()
    first = env.reset(fixture["seed"])
    assert_snapshot(first, decisions[0])

    # Mutating every Python-owned value must not reach Rust game state, cached
    # candidate actions, or the next encoded snapshot.
    first.state[:] = -123.0
    first.candidate_features[:] = 456.0
    second = env.step(first.decision_id, decisions[0]["chosen_index"])
    assert isinstance(second, mahjong_eval.Decision)
    assert_snapshot(second, decisions[1])

    second.state[:] = 789.0
    third = env.step(second.decision_id, decisions[1]["chosen_index"])
    assert isinstance(third, mahjong_eval.Decision)
    assert_snapshot(third, decisions[2])

    # A reset creates a new generation but reproduces the same deal.
    reset = env.reset(fixture["seed"])
    assert reset.decision_id == (2 << 32) | 1
    assert reset.decision_id != first.decision_id
    assert_snapshot(reset, {**decisions[0], "decision_id": reset.decision_id})


def test_reset_is_seed_deterministic_and_seed_sensitive() -> None:
    env = mahjong_eval.Env()
    one = env.reset(11)
    two = env.reset(11)
    other = env.reset(12)

    np.testing.assert_array_equal(one.state, two.state)
    np.testing.assert_array_equal(one.candidate_features, two.candidate_features)
    assert not np.array_equal(one.state, other.state)
    assert one.decision_id != two.decision_id
    assert other.decision_id == (3 << 32) | 1


def test_constructor_and_argument_rejection() -> None:
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

    with pytest.raises(native.EnvironmentStateError, match="reset"):
        mahjong_eval.Env().step(0, 0)


def test_decision_limit_is_distinct_from_terminal() -> None:
    fixture = load_fixture(TRACE_FIXTURE)
    decisions = fixture["decisions"]

    with pytest.raises(native.DecisionLimitError, match="decision limit 0"):
        mahjong_eval.Env(decision_limit=0).reset(fixture["seed"])

    limited = mahjong_eval.Env(decision_limit=3)
    decision = limited.reset(fixture["seed"])
    for index in range(3):
        decision = limited.step(decision.decision_id, decisions[index]["chosen_index"])
        assert isinstance(decision, mahjong_eval.Decision)
    with pytest.raises(
        native.DecisionLimitError,
        match=r"decision limit 3 reached after 3 accepted decisions \(seed 61\)",
    ):
        limited.step(decision.decision_id, decisions[3]["chosen_index"])

    # A game that terminates exactly at the limit is a normal terminal result.
    exact = mahjong_eval.Env(decision_limit=len(decisions))
    decision = exact.reset(fixture["seed"])
    for index, expected in enumerate(decisions):
        outcome = exact.step(decision.decision_id, expected["chosen_index"])
        if index + 1 < len(decisions):
            assert isinstance(outcome, mahjong_eval.Decision)
            decision = outcome
    assert isinstance(outcome, mahjong_eval.Terminal)
    assert outcome.decisions == len(decisions)


def test_observation_surface_is_isolated_and_hidden_information_free() -> None:
    env = mahjong_eval.Env()
    decision = env.reset(0)
    assert_snapshot(decision, load_fixture(FIRST_DECISION_FIXTURE) | {
        "candidate_count": decision.candidate_count,
    })

    public_names = {
        name for name in dir(decision) if not name.startswith("__")
    }
    assert public_names == {
        "actor_seat",
        "candidate_count",
        "candidate_features",
        "decision_id",
        "stage",
        "state",
    }
    env_names = {name for name in dir(env) if not name.startswith("__")}
    assert env_names == {"decision_limit", "reset", "step"}

    # The encoded state has exactly one per-tile own-hand vector; every other
    # seat contributes only the public player group (concealed count, missing
    # suit, score, won flag, melds). Wall order and opponent tile identities
    # have no representation or accessor.
    state = decision.state
    assert state.shape == (mahjong_eval.STATE_WIDTH,)
    assert np.all(state[:27] >= 0.0)
    assert np.all(state[:27] <= 1.0)
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(decision.candidate_features))


def test_package_import_does_not_require_the_native_extension() -> None:
    code = """
import sys

class BlockNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mahjong_eval._native":
            raise ImportError("native extension intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockNative())
import mahjong_eval
assert mahjong_eval.__version__ == "0.1.0"
from mahjong_eval.inference import FrozenPolicy  # noqa: F401
assert "mahjong_eval._native" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
