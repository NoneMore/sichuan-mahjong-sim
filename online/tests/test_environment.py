"""Environment lifecycle and Rust-handoff acceptance for the online binding."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mahjong_eval
import mahjong_eval._native as native

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "crates" / "python" / "tests" / "fixtures"
FIRST_DECISION_FIXTURE = FIXTURES / "first_decision_seed_0.json"
TRACE_FIXTURE = FIXTURES / "environment_trace_seed_61.json"
TEACHER_FIXTURE = FIXTURES / "teacher_replay_seed_0.json"

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


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()


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
    assert any(item["candidate_count"] == 1 for item in decisions), (
        "the trace must include explicit forced single-candidate choices"
    )

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
    assert env_names == {"decision_limit", "reset", "step", "teacher_index"}

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


def _replay_teacher_game(seed: int, cache=None):
    env = mahjong_eval.Env()
    decision = env.reset(seed)
    indices: list[int] = []
    while True:
        index = (
            env.teacher_index(cache=cache)
            if cache is not None
            else env.teacher_index()
        )
        indices.append(index)
        outcome = env.step(decision.decision_id, index)
        if isinstance(outcome, mahjong_eval.Terminal):
            return indices, outcome
        assert isinstance(outcome, mahjong_eval.Decision)
        decision = outcome


def test_teacher_replay_matches_rust_fixture_and_native_replay() -> None:
    fixture = load_fixture(TEACHER_FIXTURE)
    terminal = fixture["terminal"]
    decisions = fixture["decisions"]

    env = mahjong_eval.Env(decision_limit=fixture["decision_limit"])
    decision = env.reset(fixture["seed"])
    indices = []
    for position, expected in enumerate(decisions):
        assert decision.decision_id == expected["decision_id"]
        assert decision.actor_seat == expected["actor_seat"]
        assert decision.stage == expected["stage"]
        assert decision.candidate_count == expected["candidate_count"]
        assert array_sha256(decision.state) == expected["state_sha256"]
        assert (
            array_sha256(decision.candidate_features)
            == expected["candidate_features_sha256"]
        )

        # Repeated teacher queries observe the original pending Decision and
        # never advance it or change the snapshot.
        for _ in range(3):
            assert env.teacher_index() == expected["teacher_index"]
        assert array_sha256(decision.state) == expected["state_sha256"]

        indices.append(expected["teacher_index"])
        outcome = env.step(decision.decision_id, expected["teacher_index"])
        if position + 1 < len(decisions):
            assert isinstance(outcome, mahjong_eval.Decision)
            decision = outcome
        else:
            assert isinstance(outcome, mahjong_eval.Terminal)

    assert outcome.seed == fixture["seed"]
    assert outcome.decisions == terminal["decisions"] == len(decisions)
    assert outcome.end_reason == terminal["end_reason"]
    assert outcome.scores == tuple(terminal["scores"])
    assert outcome.winners == tuple(terminal["winners"])
    assert sum(outcome.scores) == 0

    with pytest.raises(native.EnvironmentStateError, match="terminal"):
        env.teacher_index()

    native_result = native.native_teacher_replay(fixture["seed"])
    assert native_result["format"] == "mahjong-eval-native-teacher-replay-v1"
    assert native_result["seed"] == fixture["seed"]
    assert native_result["decisions"] == len(decisions)
    assert native_result["scores"] == tuple(terminal["scores"])
    assert native_result["winners"] == tuple(terminal["winners"])
    assert native_result["end_reason"] == terminal["end_reason"]
    trace = native_result["trace"]
    assert len(trace) == len(decisions)
    for got, expected, python_index in zip(trace, decisions, indices):
        assert got["decision_id"] == expected["decision_id"]
        assert got["actor_seat"] == expected["actor_seat"]
        assert got["stage"] == expected["stage"]
        assert got["candidate_count"] == expected["candidate_count"]
        assert got["teacher_index"] == expected["teacher_index"] == python_index
        assert got["action"] == expected["action"]


def test_teacher_index_rejects_an_unstarted_environment() -> None:
    env = mahjong_eval.Env()
    with pytest.raises(native.EnvironmentStateError, match="reset"):
        env.teacher_index()


def test_teacher_cache_is_explicit_shared_and_choice_preserving() -> None:
    fixture = load_fixture(TEACHER_FIXTURE)
    expected_indices = [item["teacher_index"] for item in fixture["decisions"]]

    cache = mahjong_eval.TeacherCache()
    assert cache.budget_bytes == mahjong_eval.DEFAULT_TEACHER_CACHE_BYTES
    initial = cache.status()
    assert initial["state"] == "unallocated"
    assert initial["hits"] == initial["misses"] == 0
    indices, result = _replay_teacher_game(fixture["seed"], cache=cache)
    assert indices == expected_indices
    assert isinstance(result, mahjong_eval.Terminal)
    status = cache.status()
    assert status["state"] == "ready"
    assert status["hits"] + status["misses"] > 0
    assert status["allocated_bytes"] > 0

    # The same explicit context may be shared by independent handles without
    # changing either handle's pending choice.
    first = mahjong_eval.Env()
    second = mahjong_eval.Env()
    first.reset(fixture["seed"])
    second.reset(fixture["seed"])
    assert first.teacher_index(cache=cache) == expected_indices[0]
    assert second.teacher_index(cache=cache) == expected_indices[0]
    assert first.teacher_index(cache=cache) == second.teacher_index(cache=cache)
    after = cache.status()
    assert after["hits"] + after["misses"] >= status["hits"] + status["misses"]

    disabled = mahjong_eval.TeacherCache(0)
    assert disabled.budget_bytes == 0
    assert disabled.status()["state"] == "disabled"
    disabled_indices, disabled_result = _replay_teacher_game(
        fixture["seed"], cache=disabled
    )
    assert disabled_indices == expected_indices
    assert disabled_result.scores == result.scores
    assert disabled_result.decisions == result.decisions

    warm_cache = mahjong_eval.TeacherCache()
    cached_native = native.native_teacher_replay(fixture["seed"], cache=warm_cache)
    uncached_native = native.native_teacher_replay(fixture["seed"])
    assert cached_native["trace"] == uncached_native["trace"]
    assert cached_native["decisions"] == uncached_native["decisions"]
    assert warm_cache.status()["hits"] + warm_cache.status()["misses"] > 0

    with pytest.raises(ValueError):
        mahjong_eval.TeacherCache(1)
    with pytest.raises(OverflowError):
        mahjong_eval.TeacherCache(-1)
    with pytest.raises(TypeError):
        mahjong_eval.TeacherCache(1.5)


def reference_split_bucket(seed: int) -> int:
    digest = hashlib.sha256(
        b"bc-split-v1" + b"\0" + seed.to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 100


def test_seed_split_matches_frozen_rust_algorithm() -> None:
    seeds = [0, 1, 2, 6, 12, 18, 23, 158, 167, 214, (1 << 64) - 1]
    for seed in seeds:
        bucket = reference_split_bucket(seed)
        expected = (
            "train" if bucket <= 79 else "validation" if bucket <= 89 else "test"
        )
        assert native.split_bucket(seed) == bucket
        assert native.seed_split(seed) == expected


def test_compiled_source_identity_is_cwd_independent() -> None:
    metadata = native.compiled_source()
    assert metadata["format"] == "mahjong-eval-compiled-source-v1"
    assert metadata["encoding_version"] == 1
    extension = metadata["extension"]
    assert extension["source_complete"] is True
    assert extension["source_scope"] == ["crates/python", "src", "crates/neural"]
    assert len(extension["source_sha256"]) == 64
    assert extension["source_identity"].startswith("sha256-paths-bytes-v1:")
    assert extension["package_version"] == "0.1.0"
    assert metadata["neural"]["package_version"] == "0.1.0"
    assert metadata["environment"]["pointer_width"] in {32, 64}

    files = extension["source_files"]
    assert files == sorted(files)
    assert "crates/python/src/lib.rs" in files
    assert "src/game.rs" in files
    assert "crates/neural/src/encoding/mod.rs" in files
    hasher = hashlib.sha256()
    for relative in files:
        path = REPO_ROOT / relative
        assert path.is_file(), f"compiled source file is missing: {path}"
        data = path.read_bytes()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    assert hasher.hexdigest() == extension["source_sha256"]

    git = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if git.returncode == 0 and extension["git_available"]:
        assert extension["git_commit"] == git.stdout.strip()

    code = (
        "import json\n"
        "import mahjong_eval._native as native\n"
        "print(json.dumps(native.compiled_source(), sort_keys=True))\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        from_other_cwd = json.loads(completed.stdout)
    assert from_other_cwd == json.loads(json.dumps(metadata))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seeded_teacher_smoke(seed: int) -> None:
    normal = native.native_teacher_replay(seed)
    assert normal["seed"] == seed
    assert normal["decisions"] == len(normal["trace"])
    assert sum(normal["scores"]) == 0

    env = mahjong_eval.Env(decision_limit=normal["decision_limit"])
    decision = env.reset(seed)
    for position, entry in enumerate(normal["trace"]):
        assert decision.decision_id == entry["decision_id"]
        assert decision.actor_seat == entry["actor_seat"]
        assert decision.stage == entry["stage"]
        assert decision.candidate_count == entry["candidate_count"]
        assert env.teacher_index() == entry["teacher_index"]
        outcome = env.step(decision.decision_id, entry["teacher_index"])
        if position + 1 < len(normal["trace"]):
            assert isinstance(outcome, mahjong_eval.Decision)
            decision = outcome
        else:
            assert isinstance(outcome, mahjong_eval.Terminal)

    assert outcome.decisions == normal["decisions"]
    assert outcome.scores == tuple(normal["scores"])
    assert outcome.winners == tuple(normal["winners"])
    assert outcome.end_reason == normal["end_reason"]
    assert outcome.seed == seed


def test_native_teacher_replay_decision_limit() -> None:
    with pytest.raises(native.DecisionLimitError, match="decision limit 0"):
        native.native_teacher_replay(0, decision_limit=0)
    with pytest.raises(native.DecisionLimitError, match="decision limit 1"):
        native.native_teacher_replay(0, decision_limit=1)


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
from mahjong_eval import execution, records  # noqa: F401
assert "torch" not in sys.modules
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
