"""Synchronous execution of finite, caller-assigned plans of Mahjong games."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from inspect import (
    isasyncgenfunction,
    isawaitable,
    iscoroutine,
    iscoroutinefunction,
    isgeneratorfunction,
)
from itertools import islice
from operator import index as integer_index
from sys import modules
from typing import TYPE_CHECKING, Any, Protocol

from ._terminal import SEAT_COUNT, U64_MAX, is_int, validate_settlement

if TYPE_CHECKING:
    from . import Decision, TeacherCache

DEFAULT_BATCH_SIZE = 64
DEFAULT_CONCURRENT_GAMES = 6
FORCED_CANDIDATE_COUNT = 1

__all__ = [
    "AcceptedDecision",
    "Choice",
    "DecisionRequest",
    "DecisionTrace",
    "ExecutionDiagnostics",
    "ExecutionError",
    "ExecutionResult",
    "GameResult",
    "GameSpec",
    "NativeTeacher",
    "PolicyDiagnostics",
    "Selector",
    "execute_games",
]


class ExecutionError(RuntimeError):
    """Invalid input or an execution failure, with game context when available."""

    def __init__(self, message: str, *, game_id: str | None = None) -> None:
        super().__init__(message)
        self.game_id = game_id


@dataclass(frozen=True, slots=True)
class GameSpec:
    """A caller-assigned game ID, deal seed, and policy IDs in seat order."""

    game_id: str
    seed: int
    seat_policies: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if not _valid_id(self.game_id):
            raise ExecutionError("game_id must be a nonempty string")
        if not is_int(self.seed) or not 0 <= self.seed <= U64_MAX:
            raise ExecutionError(f"game {self.game_id!r} seed must be an unsigned u64")
        seats = self.seat_policies
        if (
            not isinstance(seats, Sequence)
            or isinstance(seats, (str, bytes, bytearray))
            or len(seats) != SEAT_COUNT
            or any(not _valid_id(policy) for policy in seats)
        ):
            raise ExecutionError(
                f"game {self.game_id!r} seat_policies must contain four policy IDs"
            )
        object.__setattr__(self, "seat_policies", tuple(seats))


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Policy identity and an owned, player-visible native decision snapshot."""

    game_id: str
    seed: int
    policy_id: str
    decision: Decision

    @property
    def seat(self) -> int:
        return self.decision.actor_seat


@dataclass(frozen=True, slots=True)
class Choice:
    """A local canonical candidate index and optional caller-owned metadata."""

    index: int
    metadata: object = None


class Selector(Protocol):
    """Ordinary policy interface. Choices correspond positionally to requests."""

    def select(
        self, requests: Sequence[DecisionRequest], /
    ) -> Sequence[int | Choice]: ...


@dataclass(frozen=True, slots=True)
class NativeTeacher:
    """Registration for the native heuristic, queried on the originating Env."""


@dataclass(frozen=True, slots=True)
class _SynchronizedSelector:
    """Preserve the legacy CUDA timing boundary around a comparison batch."""

    select: Callable[[Sequence[DecisionRequest]], Sequence[int]]
    synchronize: Callable[[], None]


@dataclass(frozen=True, slots=True)
class AcceptedDecision:
    """An accepted action with its pre-action request and optional metadata."""

    request: DecisionRequest
    index: int
    forced: bool
    metadata: object = None


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Scalar facts about an accepted decision within a game."""

    decision: int
    decision_id: int
    actor_seat: int
    stage: int
    candidate_count: int
    policy_id: str
    forced: bool
    index: int


@dataclass(frozen=True, slots=True)
class GameResult:
    """Validated terminal facts and decision counts in seat order."""

    game_id: str
    seed: int
    seat_policies: tuple[str, str, str, str]
    scores: tuple[int, int, int, int]
    winners: tuple[int, ...]
    end_reason: str
    decisions_by_seat: tuple[int, int, int, int]
    forced_by_seat: tuple[int, int, int, int]
    trace: tuple[DecisionTrace, ...] | None = None

    @property
    def decisions(self) -> int:
        return sum(self.decisions_by_seat)

    @property
    def forced_decisions(self) -> int:
        return sum(self.forced_by_seat)


@dataclass(frozen=True, slots=True)
class PolicyDiagnostics:
    accepted_by_seat: tuple[int, int, int, int]
    forced_by_seat: tuple[int, int, int, int]
    selector_rows: int
    selector_batches: int
    teacher_queries: int
    selection_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    policies: Mapping[str, PolicyDiagnostics]
    env_resets: int
    env_steps: int
    env_reset_seconds: float
    env_step_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Completed games in plan order, with optional run diagnostics."""

    games: tuple[GameResult, ...]
    diagnostics: ExecutionDiagnostics | None


@dataclass(slots=True)
class _SelectionStats:
    rows: int = 0
    batches: int = 0
    queries: int = 0
    seconds: float = 0.0


class _RunStats:
    def __init__(self, registry: Mapping[str, Selector | NativeTeacher]) -> None:
        self.selections = {policy: _SelectionStats() for policy in registry}
        self.reset_seconds = 0.0
        self.step_seconds = 0.0

    def finish(
        self, games: tuple[GameResult, ...], total: float
    ) -> ExecutionDiagnostics:
        accepted = {policy: [0] * SEAT_COUNT for policy in self.selections}
        forced = {policy: [0] * SEAT_COUNT for policy in self.selections}
        for game in games:
            for seat, policy in enumerate(game.seat_policies):
                accepted[policy][seat] += game.decisions_by_seat[seat]
                forced[policy][seat] += game.forced_by_seat[seat]
        return ExecutionDiagnostics(
            policies={
                policy: PolicyDiagnostics(
                    tuple(accepted[policy]),
                    tuple(forced[policy]),
                    stats.rows,
                    stats.batches,
                    stats.queries,
                    stats.seconds,
                )
                for policy, stats in self.selections.items()
            },
            env_resets=len(games),
            env_steps=sum(game.decisions for game in games),
            env_reset_seconds=self.reset_seconds,
            env_step_seconds=self.step_seconds,
            total_seconds=total,
        )


class _RunningGame:
    __slots__ = (
        "accepted",
        "count",
        "decision_id",
        "decisions",
        "env",
        "forced",
        "index",
        "metadata",
        "pending",
        "policy_id",
        "seat",
        "spec",
        "trace",
    )

    def __init__(
        self, spec: GameSpec, env: Any, pending: Decision, trace: bool
    ) -> None:
        self.spec = spec
        self.env = env
        self.decisions = 0
        self.accepted = [0] * SEAT_COUNT
        self.forced = [0] * SEAT_COUNT
        self.trace: list[DecisionTrace] | None = [] if trace else None
        self.metadata: object = None
        self.set_pending(pending)

    def set_pending(self, pending: Decision) -> None:
        try:
            seat = pending.actor_seat
            count = pending.candidate_count
            decision_id = pending.decision_id
            if type(seat) is not int or not 0 <= seat < SEAT_COUNT:
                raise ValueError(f"invalid actor seat {seat!r}")
            if type(count) is not int or count < 1:
                raise ValueError(f"invalid candidate count {count!r}")
        except (AttributeError, ValueError) as exc:
            raise _game_error(self.spec, "invalid pending decision", cause=exc) from exc
        self.pending = pending
        self.seat = seat
        self.policy_id = self.spec.seat_policies[seat]
        self.count = count
        self.decision_id = decision_id
        self.index = 0

    def request(self) -> DecisionRequest:
        return DecisionRequest(
            self.spec.game_id, self.spec.seed, self.policy_id, self.pending
        )

    def accept(self, on_decision: Callable[[AcceptedDecision], None] | None) -> None:
        forced = self.count == FORCED_CANDIDATE_COUNT
        self.decisions += 1
        self.accepted[self.seat] += 1
        self.forced[self.seat] += forced
        if self.trace is not None:
            self.trace.append(
                DecisionTrace(
                    self.decisions,
                    self.decision_id,
                    self.seat,
                    self.pending.stage,
                    self.count,
                    self.policy_id,
                    forced,
                    self.index,
                )
            )
        if on_decision is not None:
            self.notify(
                "accepted-decision",
                on_decision,
                AcceptedDecision(self.request(), self.index, forced, self.metadata),
            )
        self.metadata = None

    def notify(self, name: str, callback: Callable, event: object) -> None:
        try:
            returned = callback(event)
            if isawaitable(returned):
                if iscoroutine(returned):
                    returned.close()
                raise TypeError("observer must run synchronously")
        except Exception as exc:
            raise self.error(f"{name} callback failed", exc) from exc

    def finish(self, terminal: Any) -> GameResult:
        try:
            if not is_int(terminal.seed) or terminal.seed != self.spec.seed:
                raise ValueError(
                    f"environment terminal seed {terminal.seed!r} does not match "
                    f"planned seed {self.spec.seed}"
                )
            decisions = terminal.decisions
            if not is_int(decisions) or decisions != self.decisions:
                raise ValueError(
                    f"environment terminal decision count {decisions!r} does not match "
                    f"scheduler decision count {self.decisions}"
                )
            scores = tuple(terminal.scores)
            flags = tuple(terminal.winners)
            if len(flags) != SEAT_COUNT or any(type(won) is not bool for won in flags):
                raise ValueError("environment terminal winners must be four seat flags")
            winners = tuple(seat for seat, won in enumerate(flags) if won)
            winners = validate_settlement(scores, winners, terminal.end_reason)
        except (AttributeError, TypeError, ValueError) as exc:
            raise self.error("terminal result is invalid", exc) from exc
        return GameResult(
            self.spec.game_id,
            self.spec.seed,
            self.spec.seat_policies,
            scores,
            winners,
            terminal.end_reason,
            tuple(self.accepted),
            tuple(self.forced),
            None if self.trace is None else tuple(self.trace),
        )

    def error(self, message: str, cause: Exception | None = None) -> ExecutionError:
        context = (
            f"decision {self.decision_id}, seat {self.seat}, "
            f"stage {getattr(self.pending, 'stage', '?')}, "
            f"candidates {self.count}, policy {self.policy_id!r}"
        )
        return _game_error(self.spec, message, context=context, cause=cause)


def execute_games(
    specs: Iterable[GameSpec],
    *,
    policies: Mapping[str, Selector | NativeTeacher],
    teacher_cache: TeacherCache | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrent_games: int = DEFAULT_CONCURRENT_GAMES,
    decision_limit: int | None = None,
    on_decision: Callable[[AcceptedDecision], None] | None = None,
    on_terminal: Callable[[GameResult], None] | None = None,
    trace: bool = False,
    diagnostics: bool = False,
    env_factory: Callable[..., Any] | None = None,
) -> ExecutionResult:
    """Execute one plan synchronously, returning only validated completed games.

    Callbacks run after acceptance and before refill. A failure raises
    :class:`ExecutionError`. Steps and callback effects already accepted remain
    in effect. Keep policies stable for the run and callbacks non-reentrant.
    ``env_factory`` supplies a trusted Env-compatible test double when needed.
    """

    plan, registry = _validate_inputs(
        specs,
        policies,
        batch_size,
        concurrent_games,
        decision_limit,
        on_decision,
        on_terminal,
        trace,
        diagnostics,
        env_factory,
    )
    teachers = {
        policy
        for policy, selector in registry.items()
        if isinstance(selector, NativeTeacher)
    }
    if teachers:
        if teacher_cache is None:
            teacher_cache = _disabled_cache()
        elif env_factory is None:
            from . import TeacherCache

            if not isinstance(teacher_cache, TeacherCache):
                raise ExecutionError("teacher_cache must be a TeacherCache")
    if env_factory is None:
        from . import Env

        env_factory = Env
    if decision_limit is None:
        from . import DEFAULT_DECISION_LIMIT

        decision_limit = DEFAULT_DECISION_LIMIT

    stats = _RunStats(registry) if diagnostics else None
    started = time.perf_counter() if stats is not None else 0.0
    running: list[_RunningGame] = []
    results: dict[str, GameResult] = {}
    next_game = 0

    while next_game < len(plan) or running:
        for game in running:
            if game.decisions == decision_limit:
                raise game.error(
                    f"decision limit {decision_limit} reached after "
                    f"{game.decisions} accepted decisions"
                )
        while next_game < len(plan) and len(running) < concurrent_games:
            running.append(
                _start_game(plan[next_game], env_factory, decision_limit, trace, stats)
            )
            next_game += 1

        _select_round(running, registry, teachers, teacher_cache, batch_size, stats)
        # Every choice in this round has been validated before the first step.
        remaining = []
        for game in running:
            begin = time.perf_counter() if stats is not None else 0.0
            try:
                outcome = game.env.step(game.decision_id, game.index)
            except Exception as exc:
                raise game.error(
                    f"environment step failed for candidate index {game.index}", exc
                ) from exc
            if stats is not None:
                stats.step_seconds += time.perf_counter() - begin
            game.accept(on_decision)
            if hasattr(outcome, "end_reason"):
                result = game.finish(outcome)
                if on_terminal is not None:
                    game.notify("terminal", on_terminal, result)
                results[result.game_id] = result
            else:
                game.set_pending(outcome)
                remaining.append(game)
        running = remaining
        del game  # The loop variable can otherwise keep a finished Env alive during refill.

    total = time.perf_counter() - started if stats is not None else 0.0
    ordered = tuple(results[spec.game_id] for spec in plan)
    return ExecutionResult(
        ordered, stats.finish(ordered, total) if stats is not None else None
    )


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_inputs(
    specs: Iterable[GameSpec],
    policies: Mapping[str, Selector | NativeTeacher],
    batch_size: int,
    concurrent_games: int,
    decision_limit: int | None,
    on_decision: Callable[[AcceptedDecision], None] | None,
    on_terminal: Callable[[GameResult], None] | None,
    trace: bool,
    diagnostics: bool,
    env_factory: Callable[..., Any] | None,
) -> tuple[tuple[GameSpec, ...], dict[str, Selector | NativeTeacher]]:
    try:
        plan = tuple(specs)
    except TypeError as exc:
        raise ExecutionError("specs must be an iterable of GameSpec values") from exc
    if not plan:
        raise ExecutionError("the plan requires at least one game")
    if not isinstance(policies, Mapping):
        raise ExecutionError("policies must be a mapping from policy IDs to selectors")
    registry = dict(policies)
    for policy, selector in registry.items():
        if not _valid_id(policy):
            raise ExecutionError(f"invalid policy ID {policy!r}")
        if not isinstance(selector, NativeTeacher):
            _require_synchronous(
                getattr(selector, "select", None),
                f"policy {policy!r} select",
                allow_generator=True,
            )
    seen = set()
    for spec in plan:
        if not isinstance(spec, GameSpec):
            raise ExecutionError("every spec must be a GameSpec")
        if spec.game_id in seen:
            raise ExecutionError(f"duplicate game ID {spec.game_id!r}")
        seen.add(spec.game_id)
        for policy in spec.seat_policies:
            if policy not in registry:
                raise ExecutionError(
                    f"game {spec.game_id!r} references unregistered policy {policy!r}"
                )
    limits = {"batch_size": batch_size, "concurrent_games": concurrent_games}
    if decision_limit is not None:
        limits["decision_limit"] = decision_limit
    for name, limit in limits.items():
        if not is_int(limit) or limit < 1:
            raise ExecutionError(f"{name} must be a positive integer, got {limit!r}")
    if decision_limit is not None and decision_limit > U64_MAX:
        raise ExecutionError("decision_limit must fit in an unsigned u64")
    for name, callback in (
        ("on_decision", on_decision),
        ("on_terminal", on_terminal),
        ("env_factory", env_factory),
    ):
        if callback is not None:
            _require_synchronous(callback, name)
    if type(trace) is not bool or type(diagnostics) is not bool:
        raise ExecutionError("trace and diagnostics must be booleans")
    return plan, registry


def _require_synchronous(
    callback: object, name: str, *, allow_generator: bool = False
) -> None:
    if not callable(callback):
        raise ExecutionError(f"{name} must be callable")
    for function in (callback, type(callback).__call__):
        if (
            iscoroutinefunction(function)
            or isasyncgenfunction(function)
            or (not allow_generator and isgeneratorfunction(function))
        ):
            raise ExecutionError(f"{name} must run synchronously")


def _disabled_cache() -> TeacherCache:
    from . import TeacherCache

    return TeacherCache(budget_bytes=0)


def _game_error(
    spec: GameSpec,
    message: str,
    *,
    context: str = "",
    cause: Exception | None = None,
) -> ExecutionError:
    context = f", {context}" if context else ""
    suffix = "" if cause is None else f": {type(cause).__name__}: {cause}"
    return ExecutionError(
        f"{message} (game {spec.game_id!r}, seed {spec.seed}{context}){suffix}",
        game_id=spec.game_id,
    )


def _start_game(
    spec: GameSpec,
    env_factory: Callable[..., Any],
    decision_limit: int,
    trace: bool,
    stats: _RunStats | None,
) -> _RunningGame:
    try:
        env = env_factory(decision_limit=decision_limit)
    except Exception as exc:
        raise _game_error(spec, "environment creation failed", cause=exc) from exc
    begin = time.perf_counter() if stats is not None else 0.0
    try:
        pending = env.reset(spec.seed)
    except Exception as exc:
        raise _game_error(spec, "environment reset failed", cause=exc) from exc
    if stats is not None:
        stats.reset_seconds += time.perf_counter() - begin
    return _RunningGame(spec, env, pending, trace)


def _select_round(
    running: list[_RunningGame],
    registry: Mapping[str, Selector | NativeTeacher],
    teachers: set[str],
    cache: TeacherCache | None,
    batch_size: int,
    stats: _RunStats | None,
) -> None:
    teacher_groups: defaultdict[str, list[_RunningGame]] = defaultdict(list)
    selector_groups: defaultdict[str, list[_RunningGame]] = defaultdict(list)
    for game in running:
        groups = teacher_groups if game.policy_id in teachers else selector_groups
        groups[game.policy_id].append(game)

    for policy, group in teacher_groups.items():
        selection = stats.selections[policy] if stats is not None else None
        for game in group:
            if game.count > FORCED_CANDIDATE_COUNT:
                _query_teacher(game, cache, selection)
    for policy, group in selector_groups.items():
        selection = stats.selections[policy] if stats is not None else None
        selector = registry[policy]
        assert not isinstance(selector, NativeTeacher)
        for offset in range(0, len(group), batch_size):
            eligible = [
                game
                for game in group[offset : offset + batch_size]
                if game.count > FORCED_CANDIDATE_COUNT
            ]
            if eligible:
                _select_batch(eligible, selector, selection)


def _query_teacher(
    game: _RunningGame, cache: TeacherCache | None, stats: _SelectionStats | None
) -> None:
    begin = time.perf_counter() if stats is not None else 0.0
    try:
        index = game.env.teacher_index(cache=cache)
    except Exception as exc:
        raise game.error("teacher query failed", exc) from exc
    if stats is not None:
        stats.seconds += time.perf_counter() - begin
        stats.queries += 1
    game.index = _require_index(game, index, "teacher")


def _select_batch(
    games: list[_RunningGame],
    selector: Selector,
    stats: _SelectionStats | None,
) -> None:
    first = games[0]
    source = f"selector {first.policy_id!r}"
    requests = tuple(game.request() for game in games)
    sync = selector.synchronize if isinstance(selector, _SynchronizedSelector) else None
    try:
        if sync is not None:
            sync()
        begin = time.perf_counter() if stats is not None else 0.0
        values = selector.select(requests)
        if sync is not None:
            sync()
    except Exception as exc:
        raise first.error(
            f"{source} failed for a batch of {len(games)} decision(s)", exc
        ) from exc
    if stats is not None:
        stats.seconds += time.perf_counter() - begin
        stats.rows += len(games)
        stats.batches += 1
    if not isinstance(values, (list, tuple)):
        try:
            if isinstance(values, (Mapping, AbstractSet)):
                raise TypeError("choices must be returned in request order")
            if iscoroutine(values):
                values.close()
                raise TypeError("selector must run synchronously")
            values = list(islice(values, len(games) + 1))
        except Exception as exc:
            raise first.error(
                f"{source} did not return an index sequence", exc
            ) from exc
    if len(values) != len(games):
        raise first.error(
            f"{source} returned {len(values)} indices for {len(games)} decisions"
        )
    for game, value in zip(games, values):
        if isinstance(value, Choice):
            game.metadata = value.metadata
            value = value.index
        game.index = _require_index(game, value, source)


def _require_index(game: _RunningGame, index: int, source: str) -> int:
    if type(index) is not int:
        from numpy import bool_

        try:
            torch = modules.get("torch")
            torch_bool = getattr(torch, "bool", None)
            if isinstance(index, (bool, bool_)) or (
                torch_bool is not None and getattr(index, "dtype", None) is torch_bool
            ):
                raise TypeError
            index = integer_index(index)
        except TypeError as exc:
            raise game.error(
                f"{source} returned a non-integer candidate index {index!r}"
            ) from exc
    if not 0 <= index < game.count:
        raise game.error(
            f"{source} returned candidate index {index} outside {game.count} candidates"
        )
    return index
