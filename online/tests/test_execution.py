"""Contract tests for policy-neutral game execution."""

from dataclasses import replace
from types import SimpleNamespace
from weakref import WeakSet

import numpy as np
import pytest
import torch
from mahjong_eval.execution import (
    Choice,
    ExecutionError,
    GameSpec,
    NativeTeacher,
    execute_games,
)
from test_scheduler import EnvLog, FakeTerminal, make_env_factory, script


class Selector:
    def __init__(self, select):
        self.select = select


def game(game_id, seed=0, policy="actor"):
    return GameSpec(game_id, seed, (policy,) * 4)


async def asynchronous_observer(event):
    pytest.fail("asynchronous observer was accepted")


@pytest.mark.parametrize(
    ("game_id", "seed", "seats"),
    [
        ("", 0, ("actor",) * 4),
        (1, 0, ("actor",) * 4),
        ("g", True, ("actor",) * 4),
        ("g", -1, ("actor",) * 4),
        ("g", 2**64, ("actor",) * 4),
        ("g", 0, "abcd"),
        ("g", 0, ("actor",) * 3),
        ("g", 0, ("actor", "actor", "actor", None)),
    ],
)
def test_invalid_game_spec(game_id, seed, seats):
    with pytest.raises(ExecutionError):
        GameSpec(game_id, seed, seats)


@pytest.mark.parametrize(
    "change",
    [
        {"specs": []},
        {"specs": [object()]},
        {"specs": [game("same"), game("same", 1)]},
        {"specs": [game("g", policy="unknown")]},
        {"policies": [("actor", NativeTeacher())]},
        {"policies": {"actor": object()}},
        {"policies": {"actor": Selector(asynchronous_observer)}},
        {"policies": {"actor": NativeTeacher(), "": NativeTeacher()}},
        {"batch_size": 0},
        {"concurrent_games": True},
        {"decision_limit": -1},
        {"decision_limit": 2**64},
        {"on_decision": False},
        {"on_terminal": True},
        {"on_decision": asynchronous_observer},
        {"env_factory": True},
        {"trace": 1},
        {"diagnostics": "yes"},
    ],
)
def test_validation_precedes_handle_creation(change):
    log = EnvLog()
    args = {
        "specs": [game("g")],
        "policies": {"actor": Selector(lambda _: [0])},
        "env_factory": make_env_factory(log, []),
    }
    with pytest.raises(ExecutionError):
        execute_games(**(args | change))
    assert log.envs == 0


def test_group_order_chunking_refill_routing_and_observers():
    log = EnvLog()
    scripted = [
        script(0, [(0, 0, 1), (1, 2, 2)]),
        script(1, [(2, 0, 3)]),
        script(2, [(0, 0, 1)]),
        script(3, [(3, 0, 3)]),
        script(4, [(3, 0, 2)]),
        script(5, [(1, 0, 3)]),
        script(6, [(2, 0, 1)]),
    ]
    plan = [
        GameSpec("head", 0, ("A", "B", "B", "A")),
        game("b", 1, "B"),
        game("first-T", 2, "T"),
        game("other-teacher", 3, "U"),
        game("later-T", 4, "T"),
        game("a1", 5, "A"),
        game("a2", 5, "A"),
        game("refill", 6, "A"),
    ]
    base_factory = make_env_factory(log, scripted)

    def factory(*, decision_limit):
        env = base_factory(decision_limit=decision_limit)
        teacher_index = env.teacher_index

        def teacher(*, cache):
            log.events.append(("teacher", env.seed))
            return teacher_index(cache=cache)

        env.teacher_index = teacher
        return env

    observed = []
    metadata = {}

    def select(policy):
        def choose(requests):
            ids = tuple(request.game_id for request in requests)
            log.events.append((policy, ids))
            assert all(req.policy_id == policy for req in requests)
            assert all(req.decision.candidate_count > 1 for req in requests)
            assert all(not hasattr(req, "env") for req in requests)
            choices = []
            for req in requests:
                key = (req.game_id, req.decision.decision_id)
                metadata[key] = object()
                index = np.array(2 if req.game_id == "a2" else 0)
                choices.append(Choice(index, metadata[key]))
            return choices

        return Selector(choose)

    def observe(event):
        assert log.steps[-1] == (
            event.request.seed,
            event.request.decision.index,
            event.index,
        )
        observed.append(event)
        log.events.append(("accepted", event.request.game_id))

    cache = object()
    result = execute_games(
        plan,
        policies={
            "T": NativeTeacher(),
            "U": NativeTeacher(),
            "A": select("A"),
            "B": select("B"),
        },
        teacher_cache=cache,
        env_factory=factory,
        batch_size=2,
        concurrent_games=7,
        decision_limit=2,
        diagnostics=True,
        trace=True,
        on_decision=observe,
        on_terminal=lambda result: log.events.append(("notified", result.game_id)),
    )

    selection = [event for event in log.events if event[0] in ("teacher", "A", "B")]
    assert selection == [
        ("teacher", 4),
        ("teacher", 3),
        ("A", ("a1",)),
        ("A", ("a2",)),
        ("B", ("b",)),
        ("B", ("head",)),
    ]
    assert log.max_active == 7
    assert log.cache_ids == [id(cache), id(cache)]
    assert log.steps[:7] == [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 2),
        (4, 0, 1),
        (5, 0, 0),
        (5, 0, 2),
    ]
    assert result.games[5].trace[0].decision_id == result.games[6].trace[0].decision_id
    assert [item.game_id for item in result.games] == [item.game_id for item in plan]
    complete = [item for event, item in log.events if event == "notified"]
    assert complete == [
        "b",
        "first-T",
        "other-teacher",
        "later-T",
        "a1",
        "a2",
        "head",
        "refill",
    ]
    assert log.events.index(("notified", "a2")) < log.events.index(("reset", 6))
    for game_id in complete:
        notify = log.events.index(("notified", game_id))
        assert log.events[notify - 1] == ("accepted", game_id)
    for event in observed:
        request = event.request
        if event.forced:
            assert event.index == 0 and event.metadata is None
        else:
            assert event.metadata is metadata.get(
                (request.game_id, request.decision.decision_id)
            )
    assert result.games[0].decisions_by_seat == (1, 1, 0, 0)
    assert result.games[0].forced_by_seat == (1, 0, 0, 0)

    diag = result.diagnostics
    assert diag.env_resets == 8 and diag.env_steps == 9
    assert diag.policies["A"].accepted_by_seat == (1, 2, 1, 0)
    assert diag.policies["A"].forced_by_seat == (1, 0, 1, 0)
    assert diag.policies["A"].selector_rows == 2
    assert diag.policies["A"].selector_batches == 2
    assert diag.policies["T"].selector_rows == diag.policies["T"].selector_batches == 0
    assert diag.policies["T"].teacher_queries == diag.policies["U"].teacher_queries == 1


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "returned 0 indices"),
        ([0, 0], "returned 2 indices"),
        ([True], "non-integer"),
        ([np.bool_(False)], "non-integer"),
        (torch.tensor([True]), "non-integer"),
        ([Choice(1.0)], "non-integer"),
        ([Choice(-1)], "outside 2 candidates"),
        ([2], "outside 2 candidates"),
        (None, "did not return an index sequence"),
        ({0: Choice(1)}, "choices must be returned in request order"),
        ({0}, "choices must be returned in request order"),
    ],
)
def test_invalid_choice_prevents_the_entire_round_from_stepping(values, message):
    log = EnvLog()
    selector_calls = []
    specs = [
        game("teacher", 0, "T"),
        game("valid", 1, "A"),
        game("invalid", 1, "B"),
        game("waiting", 1, "A"),
    ]
    policies = {
        "T": NativeTeacher(),
        "A": Selector(lambda _: selector_calls.append("A") or [0]),
        "B": Selector(lambda _: selector_calls.append("B") or values),
    }
    with pytest.raises(ExecutionError, match=message) as error:
        execute_games(
            specs,
            policies=policies,
            teacher_cache=object(),
            decision_limit=2,
            env_factory=make_env_factory(
                log, [script(seed, [(0, 0, 2)]) for seed in (0, 1)]
            ),
            concurrent_games=3,
            on_decision=lambda _: pytest.fail("unexpected decision"),
            on_terminal=lambda _: pytest.fail("unexpected terminal"),
        )
    assert error.value.game_id == "invalid"
    assert "seed 1" in str(error.value) and "policy 'B'" in str(error.value)
    assert log.steps == [] and log.envs == 3
    assert log.cache_ids and selector_calls == ["A", "B"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": 7}, "terminal seed"),
        ({"decisions": 2}, "terminal decision count"),
        ({"scores": (1, 0, 0, 0)}, "sum to zero"),
        ({"scores": (False, 0, 0, 0)}, "must be an integer"),
        ({"winners": (True, False, False)}, "four seat flags"),
        ({"winners": (0, 0, 0, 0)}, "four seat flags"),
        ({"end_reason": "three_winners"}, "exactly three winners"),
        ({"winners": (True, True, True, False)}, "fewer than three winners"),
    ],
)
def test_invalid_terminal_is_never_published(change, message):
    log = EnvLog()
    observed = []
    base_factory = make_env_factory(log, [script(0, [(0, 0, 1)])])

    def factory(*, decision_limit):
        env = base_factory(decision_limit=decision_limit)
        step = env.step

        def bad_terminal(*args):
            terminal = step(*args)
            assert isinstance(terminal, FakeTerminal)
            vars(terminal).update(change)
            return terminal

        env.step = bad_terminal
        return env

    with pytest.raises(ExecutionError, match=message):
        execute_games(
            [game("g"), game("waiting")],
            policies={"actor": NativeTeacher()},
            env_factory=factory,
            decision_limit=1,
            teacher_cache=object(),
            on_decision=observed.append,
            on_terminal=lambda _: pytest.fail("invalid terminal"),
            concurrent_games=1,
        )
    assert len(observed) == 1 and log.envs == 1


@pytest.mark.parametrize("failure", ["observer", "step", "terminal", "deferred"])
def test_failures_stop_immediately_without_fabricating_completion(failure):
    log = EnvLog()
    broken = script(0, [(0, 0, 1)])
    if failure == "step":
        broken = replace(broken, fail_on_step=0)
    observed = []
    terminals = []

    def observe(event):
        observed.append(event)
        if failure == "observer":
            raise OSError("decision observer failed")

    def finish(result):
        terminals.append(result)
        if failure == "terminal":
            raise OSError("terminal observer failed")
        if failure == "deferred":
            return asynchronous_observer(result)

    with pytest.raises(ExecutionError, match="failed|failure") as error:
        execute_games(
            [game("broken"), game("active", 1), game("waiting")],
            policies={"actor": Selector(lambda _: pytest.fail("forced selection"))},
            env_factory=make_env_factory(log, [broken, script(1, [(0, 0, 1)])]),
            concurrent_games=2,
            decision_limit=1,
            on_decision=observe,
            on_terminal=finish,
        )
    assert error.value.game_id == "broken"
    assert len(observed) == (0 if failure == "step" else 1)
    assert len(terminals) == (1 if failure in ("terminal", "deferred") else 0)
    assert len(log.steps) == len(observed) and log.envs == 2


def test_refill_releases_finished_handles_and_does_not_retain_metadata():
    handles = WeakSet()
    metadata = WeakSet()

    class Env:
        def __init__(self, *, decision_limit):
            assert len(handles) == 0
            handles.add(self)

        def reset(self, seed):
            self.seed = seed
            return SimpleNamespace(
                actor_seat=0, stage=0, decision_id=1, candidate_count=2
            )

        def step(self, decision_id, index):
            assert decision_id == 1 and index == 0
            return FakeTerminal(seed=self.seed, decisions=1)

    class Metadata:
        pass

    def select(requests):
        item = Metadata()
        metadata.add(item)
        return [Choice(0, item)]

    result = execute_games(
        [game("before"), game("after")],
        policies={"actor": Selector(select)},
        env_factory=Env,
        decision_limit=1,
        concurrent_games=1,
    )
    assert len(result.games) == 2
    assert not handles and not metadata


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("create", "environment creation failed"),
        ("reset", "environment reset failed"),
        ("teacher", "teacher query failed"),
    ],
)
def test_handle_failures_include_context_and_stop_before_stepping(operation, message):
    log = EnvLog()
    base_factory = make_env_factory(log, [script(0, [(0, 0, 2)])])

    def fail(*args, **kwargs):
        raise OSError("native failure")

    def factory(*, decision_limit):
        if operation == "create":
            fail()
        env = base_factory(decision_limit=decision_limit)
        if operation == "reset":
            env.reset = fail
        else:
            env.teacher_index = fail
        return env

    with pytest.raises(ExecutionError, match=message) as error:
        execute_games(
            [game("failed"), game("waiting")],
            policies={"actor": NativeTeacher()},
            teacher_cache=object(),
            decision_limit=1,
            env_factory=factory,
            concurrent_games=1,
        )
    assert error.value.game_id == "failed"
    assert "seed 0" in str(error.value) and "OSError: native failure" in str(
        error.value
    )
    assert log.steps == [] and log.envs <= 1
