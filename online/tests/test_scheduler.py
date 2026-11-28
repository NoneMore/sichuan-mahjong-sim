"""Multi-game scheduling: grouping, routing, failures, and accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import mahjong_eval
from mahjong_eval import scheduler
from mahjong_eval.scheduler import GameSpec, SchedulingError

PLACEMENT = (0, 1)
OTHER_PLACEMENT = (2, 3)


class FakeDecision:
    def __init__(self, *, handle, seed, index, actor_seat, stage, candidate_count):
        self.handle = handle
        self.seed = seed
        self.index = index
        self.actor_seat = actor_seat
        self.stage = stage
        self.candidate_count = candidate_count

    @property
    def decision_id(self) -> int:
        # Deliberately per-handle: concurrent handles reuse the same ids.
        return (1 << 32) | (self.index + 1)

    @property
    def tag(self) -> tuple[int, int, int]:
        return (self.handle, self.seed, self.index)


class FakeTerminal:
    def __init__(self, *, seed, decisions):
        self.seed = seed
        self.decisions = decisions
        self.scores = (1, -1, 0, 0)
        self.winners = (True, False, False, False)
        self.end_reason = "wall_exhausted"


@dataclass
class Script:
    """One game's pending decisions as (actor_seat, stage, candidate_count)."""

    seed: int
    decisions: list[tuple[int, int, int]]
    teacher_choice: int | None = None
    fail_on_step: int | None = None
    fail_message: str = "synthetic step failure"


class FakeEnv:
    def __init__(self, env_log, scripts, handle):
        self.log = env_log
        self.handle = handle
        self.scripts = {script.seed: script for script in scripts}
        self.script = None
        self.index = 0
        self.seed = None

    def reset(self, seed):
        self.log.events.append(("reset", seed))
        self.seed = seed
        self.script = self.scripts[seed]
        self.index = 0
        return self._decision()

    def _decision(self):
        actor, stage, count = self.script.decisions[self.index]
        return FakeDecision(
            handle=self.handle,
            seed=self.seed,
            index=self.index,
            actor_seat=actor,
            stage=stage,
            candidate_count=count,
        )

    def teacher_index(self, cache=None):
        self.log.cache_ids.append(id(cache))
        decision = self._decision()
        if decision.candidate_count == 1:
            raise AssertionError("forced teacher decision must bypass the teacher")
        if self.script.teacher_choice is None:
            return decision.candidate_count - 1
        return self.script.teacher_choice

    def step(self, decision_id, candidate_index):
        decision = self._decision()
        assert decision_id == decision.decision_id
        assert 0 <= candidate_index < decision.candidate_count
        if self.script.fail_on_step == self.index:
            raise ValueError(self.script.fail_message)
        self.log.steps.append((self.seed, self.index, candidate_index))
        self.index += 1
        if self.index >= len(self.script.decisions):
            self.log.events.append(("terminal", self.seed))
            return FakeTerminal(seed=self.seed, decisions=self.index)
        return self._decision()


@dataclass
class EnvLog:
    events: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    cache_ids: list = field(default_factory=list)
    envs: int = 0
    active: int = 0
    max_active: int = 0


class RecordingPolicy:
    device = "cpu"

    def __init__(self, *, choice=None, fail_on_batch=None, indices=None):
        self.batches: list[list[tuple[tuple[int, int, int], int, int]]] = []
        self.choice = choice or (lambda decision: decision.candidate_count - 1)
        self.fail_on_batch = fail_on_batch
        self.indices = indices

    def select_indices(self, decisions):
        items = list(decisions)
        self.batches.append(
            [
                (getattr(item, "tag", None), item.decision_id, item.candidate_count)
                for item in items
            ]
        )
        for item in items:
            assert item.candidate_count > 1, "forced decision reached inference"
        if self.fail_on_batch is not None and len(self.batches) == self.fail_on_batch:
            raise RuntimeError("synthetic inference failure")
        if self.indices is not None:
            return self.indices
        return [self.choice(item) for item in items]


def make_env_factory(log: EnvLog, scripts):
    def factory(*, decision_limit):
        log.envs += 1
        env = FakeEnv(log, scripts, log.envs)

        original_reset = env.reset
        original_step = env.step

        def reset(seed):
            log.active += 1
            log.max_active = max(log.max_active, log.active)
            return original_reset(seed)

        def step(decision_id, candidate_index):
            outcome = original_step(decision_id, candidate_index)
            if isinstance(outcome, FakeTerminal):
                log.active -= 1
            return outcome

        env.reset = reset
        env.step = step
        return env

    return factory


def script(seed, decisions, **kwargs):
    return Script(seed=seed, decisions=decisions, **kwargs)


def run(log, scripts, **kwargs):
    kwargs.setdefault("teacher_cache", object())
    kwargs.setdefault("env_factory", make_env_factory(log, scripts))
    return scheduler.run_games(**kwargs)


def specs(*pairs):
    return [GameSpec(seed=seed, neural_seats=placement) for seed, placement in pairs]


def recompute(result):
    timings = result.timings["calls"]
    return scheduler.summarize_games(
        result.games,
        {
            "inference_batches": timings["inference_batches"],
            "teacher_queries": timings["teacher_queries"],
            "inference_rows": timings["inference_rows"],
        },
    )


def assert_counts_recompute(result):
    actual = dict(result.counts)
    actual["teacher"] = {
        key: value for key, value in actual["teacher"].items() if key != "cache"
    }
    assert actual == recompute(result)


def test_mixed_games_group_by_policy_and_route_indices_across_handles():
    pairs = [
        (2, PLACEMENT),
        (1, PLACEMENT),
        (1, OTHER_PLACEMENT),
        (2, OTHER_PLACEMENT),
    ]
    formulas = {
        1: [(0, 0, 30), (1, 2, 5), (2, 1, 1), (3, 3, 8), (0, 2, 4), (1, 3, 2)],
        2: [
            (0, 0, 12),
            (2, 1, 4),
            (3, 4, 1),
            (1, 3, 20),
            (0, 2, 3),
            (2, 0, 7),
            (3, 2, 1),
        ],
    }
    log = EnvLog()
    scripts = [script(seed, formula) for seed, formula in formulas.items()]
    policy = RecordingPolicy()
    terminal = []
    result = run(
        log,
        scripts,
        specs=specs(*pairs),
        neural=policy,
        batch_size=2,
        concurrent_games=3,
        on_terminal=terminal.append,
        trace=True,
    )

    assert [game.key for game in result.games] == [pair for pair in pairs]
    assert len(terminal) == len(pairs)
    assert [game.key for game in terminal] != [game.key for game in result.games]
    assert all(len(batch) <= 2 for batch in policy.batches)

    rows = [row for batch in policy.batches for row in batch]
    assert len(rows) == len({row[0] for row in rows})
    decisions = [entry for game in result.games for entry in game.trace]
    neural = [entry for entry in decisions if entry["policy"] == "neural"]
    assert len(rows) == len([entry for entry in neural if not entry["forced"]])
    assert len(decisions) == sum(game.decisions for game in result.games)

    for game in result.games:
        for entry in game.trace:
            if entry["forced"]:
                assert entry["index"] == 0
            elif entry["policy"] == "neural":
                assert entry["index"] == entry["candidate_count"] - 1
            else:
                assert entry["index"] == entry["candidate_count"] - 1

    # Concurrent handles share the same packed decision ids; routing must not
    # key on them.
    repeated = [
        batch
        for batch in policy.batches
        if len({row[1] for row in batch}) < len(batch)
    ]
    assert repeated, "fixture should exercise identical decision ids across handles"

    assert_counts_recompute(result)


def test_teacher_queries_use_one_explicit_cache_and_skip_forced():
    decisions = [(0, 0, 6), (1, 1, 1), (2, 2, 3), (3, 3, 4), (2, 2, 1), (0, 0, 2)]
    log = EnvLog()
    cache = object()
    policy = RecordingPolicy()
    result = scheduler.run_games(
        specs((7, PLACEMENT)),
        neural=policy,
        teacher_cache=cache,
        env_factory=make_env_factory(log, [script(7, decisions)]),
        trace=True,
        batch_size=4,
    )
    teacher = [entry for entry in result.games[0].trace if entry["policy"] == "heuristic"]
    teacher_active = [entry for entry in teacher if not entry["forced"]]
    assert len(log.cache_ids) == len(teacher_active)
    assert set(log.cache_ids) == {id(cache)}
    assert result.counts["teacher"]["queries"] == len(teacher_active)
    assert result.counts["decisions"]["teacher_forced"] == sum(
        entry["forced"] for entry in teacher
    )
    assert all(
        entry["index"] == entry["candidate_count"] - 1 for entry in teacher_active
    )


def test_forced_single_candidate_choices_bypass_both_policies():
    decisions = [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]

    class ExplodingPolicy(RecordingPolicy):
        def select_indices(self, decisions):
            raise AssertionError("forced neural decision must bypass inference")

    log = EnvLog()
    result = run(
        log,
        [script(4, decisions)],
        specs=specs((4, PLACEMENT)),
        neural=ExplodingPolicy(),
        trace=True,
    )
    game = result.games[0]
    assert [entry["index"] for entry in game.trace] == [0, 0, 0, 0]
    assert game.forced_decisions == game.decisions == 4
    assert game.inference_rows == 0
    assert result.counts["inference"]["batches"] == 0
    assert result.counts["inference"]["rows"] == 0
    assert result.counts["teacher"]["queries"] == 0
    assert log.cache_ids == []
    assert_counts_recompute(result)


def test_concurrent_progress_differences_and_immediate_record_order():
    log = EnvLog()
    scripts = [
        script(1, [(0, 0, 2)] * 8),
        script(2, [(0, 0, 2)] * 2),
        script(3, [(0, 0, 2)] * 4),
    ]
    completed = []
    result = run(
        log,
        scripts,
        specs=specs((1, PLACEMENT), (2, PLACEMENT), (3, PLACEMENT)),
        neural=RecordingPolicy(),
        concurrent_games=2,
        on_terminal=completed.append,
    )
    assert log.max_active == 2
    assert [game.seed for game in completed] == [2, 3, 1]
    assert [game.seed for game in result.games] == [1, 2, 3]
    events = log.events
    for position, event in enumerate(events):
        if event[0] != "terminal" or position + 1 >= len(events):
            continue
        if events[position + 1][0] == "reset":
            assert events[position + 1] == ("reset", 3)
    assert len([event for event in events if event[0] == "reset"]) == 3
    assert len([event for event in events if event[0] == "terminal"]) == 3


def test_terminal_decision_count_mismatch_stops_with_context():
    log = EnvLog()
    scripts = [script(7, [(0, 0, 3)])]
    factory = make_env_factory(log, scripts)

    def mismatched_factory(*, decision_limit):
        env = factory(decision_limit=decision_limit)
        step = env.step

        def step_with_wrong_count(decision_id, candidate_index):
            outcome = step(decision_id, candidate_index)
            if isinstance(outcome, FakeTerminal):
                return FakeTerminal(
                    seed=outcome.seed, decisions=outcome.decisions + 1
                )
            return outcome

        env.step = step_with_wrong_count
        return env

    with pytest.raises(SchedulingError) as raised:
        run(
            log,
            scripts,
            specs=specs((7, PLACEMENT)),
            neural=RecordingPolicy(),
            env_factory=mismatched_factory,
        )
    message = str(raised.value)
    assert "environment terminal decision count 2" in message
    assert "scheduler decision count 1" in message
    assert "seed 7" in message and "placement [0, 1]" in message


def test_terminal_callback_failure_stops_scheduling_with_context():
    def failing(record):
        raise OSError("disk full while appending")

    log = EnvLog()
    scripts = [script(5, [(0, 0, 2), (1, 2, 2)]), script(6, [(0, 0, 2)])]
    with pytest.raises(SchedulingError) as raised:
        run(
            log,
            scripts,
            specs=specs((5, PLACEMENT), (6, PLACEMENT)),
            neural=RecordingPolicy(),
            concurrent_games=1,
            on_terminal=failing,
        )
    message = str(raised.value)
    assert "terminal callback failed" in message
    assert "seed 5" in message and "placement [0, 1]" in message
    assert "OSError: disk full while appending" in message
    assert [event for event in log.events if event[0] == "reset"] == [("reset", 5)]


def test_game_failure_reports_seed_placement_and_decision_context():
    log = EnvLog()
    scripted = script(
        9,
        [(0, 0, 2), (1, 1, 3), (2, 2, 4), (3, 3, 2)],
        fail_on_step=2,
    )
    with pytest.raises(SchedulingError) as raised:
        run(
            log,
            [scripted],
            specs=specs((9, PLACEMENT)),
            neural=RecordingPolicy(),
            batch_size=4,
        )
    message = str(raised.value)
    assert "environment step failed for candidate index 3" in message
    assert "seed 9" in message
    assert "placement [0, 1]" in message
    assert "decision 4294967299" in message
    assert "seat 2" in message
    assert "stage 2" in message
    assert "candidates 4" in message
    assert "ValueError: synthetic step failure" in message
    assert log.steps == [(9, 0, 1), (9, 1, 2)]


def test_inference_failure_reports_batch_context_and_stops():
    log = EnvLog()
    policy = RecordingPolicy(fail_on_batch=2)
    with pytest.raises(SchedulingError) as raised:
        run(
            log,
            [script(3, [(0, 0, 2)] * 6)],
            specs=specs((3, PLACEMENT)),
            neural=policy,
            batch_size=1,
        )
    message = str(raised.value)
    assert "neural inference failed for a batch of 1 decision(s)" in message
    assert "seed 3" in message and "placement [0, 1]" in message
    assert "RuntimeError: synthetic inference failure" in message
    # One batch was committed before the failure; no substitute action.
    assert len(log.steps) == 1


@pytest.mark.parametrize(
    ("indices", "fragment"),
    [
        ([], "returned 0 indices for 1 decisions"),
        ([99], "outside 2 candidates"),
        ([True], "non-integer candidate index"),
    ],
)
def test_invalid_inference_indices_are_rejected_before_commit(indices, fragment):
    log = EnvLog()
    with pytest.raises(SchedulingError) as raised:
        run(
            log,
            [script(11, [(0, 0, 2), (1, 1, 2)])],
            specs=specs((11, PLACEMENT)),
            neural=RecordingPolicy(indices=indices),
        )
    assert fragment in str(raised.value)
    assert log.steps == []


def test_stats_are_recomputable_and_timings_are_separated():
    log = EnvLog()
    scripts = [script(seed, [(0, 0, 3), (1, 1, 1), (2, 2, 2)]) for seed in (1, 2)]
    result = run(
        log,
        scripts,
        specs=specs((1, PLACEMENT), (2, PLACEMENT)),
        neural=RecordingPolicy(),
        concurrent_games=2,
    )
    assert_counts_recompute(result)
    timings = result.timings
    assert timings["instrument"] == "time.perf_counter"
    assert timings["device"] == "cpu"
    assert timings["cuda_synchronized_inference"] is False
    calls = timings["calls"]
    seconds = timings["seconds"]
    assert calls["env_resets"] == 2
    assert calls["env_steps"] == sum(game.decisions for game in result.games)
    assert calls["inference_batches"] == result.counts["inference"]["batches"]
    assert calls["teacher_queries"] == result.counts["teacher"]["queries"]
    assert calls["inference_rows"] == result.counts["inference"]["rows"]
    assert all(value >= 0 for value in seconds.values())
    assert seconds["total"] >= sum(
        seconds[name] for name in ("env_reset", "env_step", "teacher", "inference")
    )


def test_trace_option_retains_only_small_scalar_entries():
    log = EnvLog()
    scripts = [script(1, [(0, 0, 3), (1, 1, 2)])]
    spec = specs((1, PLACEMENT))
    policy = RecordingPolicy()
    without = run(log, scripts, specs=spec, neural=policy)
    assert without.games[0].trace is None

    with_trace = run(
        EnvLog(), scripts, specs=spec, neural=RecordingPolicy(), trace=True
    )
    entries = with_trace.games[0].trace
    expected = {
        "decision",
        "decision_id",
        "actor_seat",
        "stage",
        "candidate_count",
        "policy",
        "forced",
        "index",
    }
    assert all(set(entry) == expected for entry in entries)
    assert all(
        isinstance(value, (int, bool)) or isinstance(value, str)
        for entry in entries
        for value in entry.values()
    )


def test_plan_order_results_with_teacher_trace_and_shared_native_cache():
    cache = mahjong_eval.TeacherCache(budget_bytes=0)
    result = scheduler.run_games(
        [GameSpec(seed=0, neural_seats=PLACEMENT)],
        neural=RecordingPolicy(choice=lambda decision: 0),
        teacher_cache=cache,
        decision_limit=10000,
        trace=True,
    )
    game = result.games[0]
    assert game.end_reason in ("three_winners", "wall_exhausted")
    assert sum(game.scores) == 0
    assert game.decisions == len(game.trace)
    assert all(entry["index"] < entry["candidate_count"] for entry in game.trace)
    assert_counts_recompute(result)
    assert result.counts["teacher"]["cache"]["budget_bytes"] == 0
    assert result.timings["calls"]["teacher_queries"] == result.counts["teacher"][
        "queries"
    ]


def test_real_environment_decision_limit_failure_has_context():
    with pytest.raises(SchedulingError) as raised:
        scheduler.run_games(
            [GameSpec(seed=0, neural_seats=PLACEMENT)],
            neural=RecordingPolicy(choice=lambda decision: 0),
            teacher_cache=mahjong_eval.TeacherCache(budget_bytes=0),
            decision_limit=3,
        )
    message = str(raised.value)
    assert "decision limit 3 reached after 3 accepted decisions" in message
    assert "seed 0" in message and "placement [0, 1]" in message


def test_invalid_specs_and_options_are_rejected_before_play():
    cache = object()

    async def observe(game):
        pytest.fail("an asynchronous observer must not be called")

    with pytest.raises(SchedulingError):
        scheduler.run_games([], neural=RecordingPolicy())
    with pytest.raises(SchedulingError):
        scheduler.run_games(
            [GameSpec(seed=1, neural_seats=PLACEMENT)] * 2,
            neural=RecordingPolicy(),
        )
    with pytest.raises(SchedulingError):
        GameSpec(seed=1, neural_seats=(1, 1))
    for kwargs in (
        {"batch_size": 0},
        {"concurrent_games": 0},
        {"decision_limit": 0},
        {"on_terminal": 5},
        {"on_terminal": observe},
        {"neural": object()},
    ):
        log = EnvLog()
        arguments = {
            "neural": RecordingPolicy(),
            "teacher_cache": cache,
            "env_factory": make_env_factory(log, []),
        }
        arguments.update(kwargs)
        with pytest.raises(SchedulingError):
            scheduler.run_games(
                [GameSpec(seed=1, neural_seats=PLACEMENT)], **arguments
            )
        assert log.envs == 0
