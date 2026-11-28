"""Exercise neutral policy assignments on the real native environment."""

import mahjong_eval
import numpy as np
import pytest
import torch
from _checkpoint_fixtures import build_checkpoint
from mahjong_bc.constants import ROLE_POLICY
from mahjong_eval.execution import (
    Choice,
    ExecutionError,
    GameSpec,
    NativeTeacher,
    execute_games,
)
from mahjong_eval.inference import FrozenPolicy, SamplingSelector
from test_environment import TRACE_FIXTURE, load_fixture, visible_winner
from test_scheduler import EnvLog, make_env_factory, script


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    target = tmp_path_factory.mktemp("execution-policies")
    return {
        name: FrozenPolicy(
            build_checkpoint(
                target / f"{name}.pt",
                role=ROLE_POLICY,
                hidden=4,
                model_seed=seed,
            ).path
        )
        for name, seed in (("A", 5), ("B", 7))
    }


def action_wire(row):
    """Decode the selected Encoding v1 candidate as a native wire action."""
    kind = int(row[:9].argmax())
    if kind == 1:
        return [kind, int(row[36:39].argmax())]
    if kind in (0, 2, 5, 6):
        tiles = np.repeat(np.arange(27), (row[9:36] * 4).astype(int))
        return [kind, *tiles.tolist()]
    return [kind]


def test_native_lineups_and_teacher_replay(policies):
    selections = (
        ("A", "T", "T", "A"),
        ("A",) * 4,
        ("T",) * 4,
        ("A", "B", "A", "B"),
    )
    specs = [GameSpec(str(index), 0, seats) for index, seats in enumerate(selections)]
    observed = []

    result = execute_games(
        specs,
        policies=policies | {"T": NativeTeacher()},
        diagnostics=True,
        trace=True,
        on_decision=observed.append,
        batch_size=2,
        concurrent_games=3,
    )
    assert [item.seat_policies for item in result.games] == list(selections)
    assert all(
        sum(item.scores) == 0 and item.decisions == len(item.trace)
        for item in result.games
    )
    assert result.diagnostics.policies["A"].selector_rows > 0
    assert result.diagnostics.policies["B"].selector_rows > 0
    assert result.diagnostics.policies["T"].teacher_queries > 0

    all_teacher = result.games[2]
    expected = mahjong_eval.native_teacher_replay(
        0, cache=mahjong_eval.TeacherCache(budget_bytes=0)
    )
    assert all_teacher.scores == expected["scores"]
    assert all_teacher.end_reason == expected["end_reason"]
    assert all_teacher.winners == tuple(
        i for i, won in enumerate(expected["winners"]) if won
    )
    teacher_actions = [
        event for event in observed if event.request.game_id == all_teacher.game_id
    ]
    assert len(teacher_actions) == expected["decisions"]
    for event, replay in zip(teacher_actions, expected["trace"]):
        decision = event.request.decision
        assert decision.decision_id == replay["decision_id"]
        assert event.request.seat == replay["actor_seat"]
        assert decision.stage == replay["stage"]
        assert event.index == replay["teacher_index"]
        assert (
            action_wire(decision.candidate_features[event.index])
            == replay["action"]["wire"]
        )

    # Snapshots retained by the caller still own their arrays after refill and return.
    first, second = teacher_actions[:2]
    assert first.request.decision.state.flags.owndata
    assert first.request.decision.candidate_features.flags.owndata
    assert not np.shares_memory(
        first.request.decision.state, second.request.decision.state
    )


def test_native_early_hu_continues_to_final_settlement():
    fixture = load_fixture(TRACE_FIXTURE)
    decisions = fixture["decisions"]
    seen = []

    class ScriptedPolicy:
        def select(self, requests):
            choices = []
            for request in requests:
                assert request.game_id == "script"
                turn = request.decision.decision_id - decisions[0]["decision_id"]
                choices.append(Choice(decisions[turn]["chosen_index"], decisions[turn]))
            return choices

    def observe(event):
        assert event.request.decision.decision_id == decisions[len(seen)]["decision_id"]
        assert event.index == decisions[len(seen)]["chosen_index"]
        if event.forced:
            assert event.metadata is None
        else:
            assert event.metadata is decisions[len(seen)]
        seen.append(visible_winner(event.request.decision.state))

    result = execute_games(
        [GameSpec("script", fixture["seed"], ("scripted",) * 4)],
        policies={"scripted": ScriptedPolicy()},
        decision_limit=len(decisions),
        on_decision=observe,
        on_terminal=lambda _: seen.append("terminal"),
    )
    assert seen.index(True) < len(decisions) - 1
    assert len(seen) == len(decisions) + 1 and seen[-1] == "terminal"
    game = result.games[0]
    terminal = fixture["terminal"]
    assert game.scores == tuple(terminal["scores"])
    assert game.winners == tuple(i for i, won in enumerate(terminal["winners"]) if won)
    assert game.end_reason == terminal["end_reason"]
    assert game.decisions == len(decisions)
    assert game.trace is None and result.diagnostics is None


def test_native_sampling_replays_at_fixed_configuration(policies):
    spec = [GameSpec(str(index), index, ("sampled",) * 4) for index in range(3)]
    generator = torch.Generator().manual_seed(10)
    original = generator.get_state()
    selector = SamplingSelector(policies["A"], generator=generator)

    def play():
        return execute_games(
            spec,
            policies={"sampled": selector},
            batch_size=2,
            concurrent_games=3,
            trace=True,
        )

    first = play()
    after = generator.get_state()
    assert not torch.equal(original, after)
    generator.set_state(original)
    assert play() == first
    assert torch.equal(generator.get_state(), after)


def test_forced_kernel_choices_leave_sampling_generator_untouched(policies):
    generator = torch.Generator().manual_seed(10)
    original = generator.get_state()
    selector = SamplingSelector(policies["A"], generator=generator)
    log = EnvLog()
    result = execute_games(
        [GameSpec("forced", 0, ("sampled",) * 4)],
        policies={"sampled": selector},
        env_factory=make_env_factory(log, [script(0, [(0, 0, 1), (3, 1, 1)])]),
        decision_limit=2,
        diagnostics=True,
    )
    assert torch.equal(generator.get_state(), original)
    assert result.games[0].forced_by_seat == (1, 0, 0, 1)
    assert result.diagnostics.policies["sampled"].selector_batches == 0


def test_native_limit_stops_before_another_selection():
    events = []
    selected = []

    class FirstCandidate:
        def select(self, requests):
            selected.extend(requests)
            return [0] * len(requests)

    with pytest.raises(ExecutionError, match="decision limit 1"):
        execute_games(
            [GameSpec("limited", 0, ("first",) * 4)],
            policies={"first": FirstCandidate()},
            decision_limit=1,
            on_decision=events.append,
            on_terminal=lambda _: pytest.fail("unexpected terminal"),
        )
    assert len(events) == 1
    assert selected == [events[0].request]
