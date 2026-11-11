"""Synchronous multi-game scheduling for mixed neural/teacher evaluation.

One :class:`~mahjong_eval.Env` handle owns each game. The scheduler advances
every active game one pending decision at a time:

1. pending neural decisions are grouped by assigned policy and scored in
   ``batch_size`` chunks, and indices are routed back by chunk position;
2. teacher seats query the original Rust decision through one explicit,
   shared teacher cache;
3. a forced single-candidate decision is committed as index zero without
   inference or a teacher query;
4. every chosen index is committed through the originating handle's cached
   canonical actions.

Games start in plan order up to ``concurrent_games`` and are refilled as soon
as one finishes, so handles naturally sit at different stages and decision
counts. A terminal result is reported to ``on_terminal`` before the next game
starts; a callback or output failure stops scheduling instead of substituting
an action. A win for one seat never stops a game early: only the environment's
terminal result ends it.

The module does not import the native extension or PyTorch at import time, so
its result structure stays inspectable without a built wheel. The engine and
teacher cache are resolved lazily when a run starts.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .records import RecordError, TerminalRecord

POLICY_NEURAL = "neural"
POLICY_TEACHER = "heuristic"
POLICIES = (POLICY_NEURAL, POLICY_TEACHER)

FORCED_CANDIDATE_COUNT = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_CONCURRENT_GAMES = 6

_SEAT_COUNT = 4
_U64_MAX = 2**64 - 1

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONCURRENT_GAMES",
    "FORCED_CANDIDATE_COUNT",
    "GameResult",
    "GameSpec",
    "POLICIES",
    "POLICY_NEURAL",
    "POLICY_TEACHER",
    "SchedulingError",
    "SchedulingResult",
    "run_games",
    "summarize_games",
]


class SchedulingError(RuntimeError):
    """A game, teacher, inference, or terminal-record failure during a run.

    The message carries the seed, neural-seat placement, and (when available)
    the decision identity, actor, stage, and candidate count of the affected
    pending decision.
    """


@dataclass(frozen=True)
class GameSpec:
    """One planned game: a seed and the two seats assigned the neural policy."""

    seed: int
    neural_seats: tuple[int, int]

    def __post_init__(self) -> None:
        seed = self.seed
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SchedulingError("game seed must be an integer")
        if seed < 0 or seed > _U64_MAX:
            raise SchedulingError(f"game seed {seed} must fit in an unsigned u64")
        try:
            seats = tuple(self.neural_seats)
        except TypeError as exc:
            raise SchedulingError("neural_seats must be a two-seat sequence") from exc
        if len(seats) != 2:
            raise SchedulingError("neural_seats must contain exactly two seats")
        for seat in seats:
            if not isinstance(seat, int) or isinstance(seat, bool):
                raise SchedulingError("neural seats must be integers")
            if seat < 0 or seat >= _SEAT_COUNT:
                raise SchedulingError(
                    f"neural seat {seat} must be in [0, {_SEAT_COUNT})"
                )
        if seats[0] == seats[1]:
            raise SchedulingError("neural seats must be distinct")
        object.__setattr__(self, "neural_seats", tuple(sorted(seats)))

    @property
    def key(self) -> tuple[int, tuple[int, int]]:
        return (self.seed, self.neural_seats)

    def placement_text(self) -> str:
        return f"[{self.neural_seats[0]}, {self.neural_seats[1]}]"


@dataclass(frozen=True)
class GameResult:
    """One completed game plus the scheduler's own decision accounting.

    ``decisions`` includes forced choices. ``inference_rows`` counts neural
    decisions that required inference (more than one candidate). Trace entries,
    when requested, are small scalar records; no state or candidate arrays are
    retained.
    """

    seed: int
    neural_seats: tuple[int, int]
    scores: tuple[int, int, int, int]
    winners: tuple[int, ...]
    end_reason: str
    decisions: int
    neural_decisions: int
    neural_forced_decisions: int
    forced_decisions: int
    inference_rows: int
    trace: tuple[Mapping[str, Any], ...] | None = None

    @property
    def key(self) -> tuple[int, tuple[int, int]]:
        return (self.seed, self.neural_seats)

    @property
    def teacher_decisions(self) -> int:
        return self.decisions - self.neural_decisions

    @property
    def teacher_forced_decisions(self) -> int:
        return self.forced_decisions - self.neural_forced_decisions

    def to_record(self) -> TerminalRecord:
        """Validate and return the persisted terminal-record shape."""

        return TerminalRecord(
            seed=self.seed,
            neural_seats=self.neural_seats,
            scores=self.scores,
            winners=self.winners,
            end_reason=self.end_reason,
            decisions=self.decisions,
        )


@dataclass(frozen=True)
class SchedulingResult:
    """Completed games in plan order plus recomputable counts and timings."""

    games: tuple[GameResult, ...]
    counts: Mapping[str, Any]
    timings: Mapping[str, Any]

    def records(self) -> tuple[TerminalRecord, ...]:
        return tuple(game.to_record() for game in self.games)


class _Timings:
    def __init__(self) -> None:
        self.seconds = {
            "env_reset": 0.0,
            "env_step": 0.0,
            "teacher": 0.0,
            "inference": 0.0,
        }
        self.calls = {
            "env_resets": 0,
            "env_steps": 0,
            "teacher_queries": 0,
            "inference_batches": 0,
            "inference_rows": 0,
        }

    def add(self, name: str, call: str, seconds: float) -> None:
        self.seconds[name] += seconds
        self.calls[call] += 1

    def document(self, total: float, device: Any) -> dict[str, Any]:
        measured = sum(self.seconds.values())
        device_text = "unknown" if device is None else str(device)
        cuda = getattr(device, "type", None) == "cuda"
        return {
            "instrument": "time.perf_counter",
            "device": device_text,
            "cuda_synchronized_inference": cuda,
            "boundary": (
                "CUDA is synchronized before and after each inference batch, "
                "so inference seconds include queued GPU work and the other "
                "timers exclude it. Environment and teacher timers cover the "
                "single-threaded Rust/CPU call only."
            ),
            "seconds": {
                "env_reset": self.seconds["env_reset"],
                "env_step": self.seconds["env_step"],
                "teacher": self.seconds["teacher"],
                "inference": self.seconds["inference"],
                "unaccounted": max(0.0, total - measured),
                "total": total,
            },
            "calls": dict(self.calls),
        }


class _RunningGame:
    def __init__(self, spec: GameSpec, env: Any, pending: Any, trace: bool) -> None:
        self.spec = spec
        self.env = env
        self.pending = pending
        self.decisions = 0
        self.neural_decisions = 0
        self.neural_forced_decisions = 0
        self.forced_decisions = 0
        self.inference_rows = 0
        self.trace: list[dict[str, Any]] | None = [] if trace else None
        self.chosen: int | None = None

    @property
    def policy(self) -> str:
        if int(self.pending.actor_seat) in self.spec.neural_seats:
            return POLICY_NEURAL
        return POLICY_TEACHER


def run_games(
    specs: Iterable[GameSpec],
    *,
    neural: Any,
    teacher_cache: Any = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrent_games: int | None = None,
    decision_limit: int | None = None,
    on_terminal: Callable[[GameResult], None] | None = None,
    trace: bool = False,
    env_factory: Callable[..., Any] | None = None,
) -> SchedulingResult:
    """Play every planned game to its terminal result.

    ``neural`` must expose ``select_indices(decisions)`` and, optionally,
    ``device``. ``teacher_cache`` is one explicit cache shared by every
    teacher query; ``None`` creates a disabled cache instead of falling back to
    per-handle implicit state. ``on_terminal`` is called with each completed
    game before the next game starts. Any failure raises
    :class:`SchedulingError` with game/decision context and stops the run;
    games already delivered to ``on_terminal`` remain the caller's
    responsibility.
    """

    games = _normalize_specs(specs)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise SchedulingError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )
    if concurrent_games is None:
        limit = len(games)
    else:
        if (
            not isinstance(concurrent_games, int)
            or isinstance(concurrent_games, bool)
            or concurrent_games < 1
        ):
            raise SchedulingError(
                "concurrent_games must be a positive integer, "
                f"got {concurrent_games!r}"
            )
        limit = min(concurrent_games, len(games))
    if decision_limit is None:
        decision_limit = _default_decision_limit()
    elif (
        not isinstance(decision_limit, int)
        or isinstance(decision_limit, bool)
        or decision_limit < 1
    ):
        raise SchedulingError(
            f"decision_limit must be a positive integer, got {decision_limit!r}"
        )
    if on_terminal is not None and not callable(on_terminal):
        raise SchedulingError("on_terminal must be callable")
    if not callable(getattr(neural, "select_indices", None)):
        raise SchedulingError("neural policy must expose select_indices(decisions)")

    engine = env_factory if env_factory is not None else _default_env()
    cache = teacher_cache if teacher_cache is not None else _default_cache()
    device = getattr(neural, "device", None)
    timings = _Timings()
    running: list[_RunningGame] = []
    next_spec = 0
    results: dict[tuple[int, tuple[int, int]], GameResult] = {}
    started = time.perf_counter()

    while next_spec < len(games) or running:
        while next_spec < len(games) and len(running) < limit:
            spec = games[next_spec]
            env = engine(decision_limit=decision_limit)
            begin = time.perf_counter()
            try:
                pending = env.reset(spec.seed)
            except Exception as exc:
                raise _context_error(
                    spec, None, "environment reset failed", exc
                ) from exc
            timings.add("env_reset", "env_resets", time.perf_counter() - begin)
            running.append(_RunningGame(spec, env, pending, trace))
            next_spec += 1

        teacher_games = [game for game in running if game.policy == POLICY_TEACHER]
        neural_games = [game for game in running if game.policy == POLICY_NEURAL]

        for game in teacher_games:
            game.chosen = _teacher_choice(game, cache, timings)
        for offset in range(0, len(neural_games), batch_size):
            _neural_choices(
                neural_games[offset : offset + batch_size], neural, timings
            )

        finished: list[_RunningGame] = []
        for game in running:
            if game.chosen is None:
                raise SchedulingError(
                    f"internal error: no chosen index for seed {game.spec.seed} "
                    f"placement {game.spec.placement_text()}"
                )
            begin = time.perf_counter()
            try:
                outcome = game.env.step(game.pending.decision_id, game.chosen)
            except Exception as exc:
                raise _context_error(
                    game.spec,
                    game.pending,
                    f"environment step failed for candidate index {game.chosen}",
                    exc,
                ) from exc
            timings.add("env_step", "env_steps", time.perf_counter() - begin)
            _record_choice(game, game.chosen)
            game.chosen = None
            if _is_terminal(outcome):
                finished.append(game)
                game.pending = None
                result = _finish_game(game, outcome)
                try:
                    result.to_record()
                except RecordError as exc:
                    raise _context_error(
                        game.spec, None, "terminal result is invalid", exc
                    ) from exc
                results[result.key] = result
                if on_terminal is not None:
                    try:
                        on_terminal(result)
                    except Exception as exc:
                        raise _context_error(
                            game.spec, None, "terminal callback failed", exc
                        ) from exc
            else:
                game.pending = outcome

        if finished:
            done = {id(game) for game in finished}
            running = [game for game in running if id(game) not in done]

    total = time.perf_counter() - started
    ordered = tuple(results[spec.key] for spec in games)
    counts = summarize_games(ordered, timings.calls)
    counts["teacher"]["cache"] = _cache_status(cache)
    _validate_counts(ordered, counts)
    return SchedulingResult(
        games=ordered,
        counts=counts,
        timings=timings.document(total, device),
    )


def _normalize_specs(specs: Iterable[GameSpec]) -> list[GameSpec]:
    try:
        values = list(specs)
    except TypeError as exc:
        raise SchedulingError("specs must be an iterable of GameSpec values") from exc
    if not values:
        raise SchedulingError("the comparison requires at least one planned game")
    normalized: list[GameSpec] = []
    seen: set[tuple[int, tuple[int, int]]] = set()
    for value in values:
        if not isinstance(value, GameSpec):
            if not hasattr(value, "seed") or not hasattr(value, "neural_seats"):
                raise SchedulingError(
                    "every spec must be a GameSpec or expose seed/neural_seats"
                )
            value = GameSpec(seed=value.seed, neural_seats=value.neural_seats)
        if value.key in seen:
            raise SchedulingError(
                f"duplicate planned game: seed {value.seed} placement "
                f"{value.placement_text()}"
            )
        seen.add(value.key)
        normalized.append(value)
    return normalized


def _default_env() -> Any:
    import mahjong_eval

    return mahjong_eval.Env


def _default_decision_limit() -> int:
    import mahjong_eval

    return int(mahjong_eval.DEFAULT_DECISION_LIMIT)


def _default_cache() -> Any:
    import mahjong_eval

    return mahjong_eval.TeacherCache(budget_bytes=0)


def _is_terminal(outcome: Any) -> bool:
    return hasattr(outcome, "end_reason") and hasattr(outcome, "scores")


def _candidate_count(pending: Any) -> int:
    return int(pending.candidate_count)


def _teacher_choice(game: _RunningGame, cache: Any, timings: _Timings) -> int:
    if _candidate_count(game.pending) <= FORCED_CANDIDATE_COUNT:
        return 0
    begin = time.perf_counter()
    try:
        index = game.env.teacher_index(cache=cache)
    except Exception as exc:
        raise _context_error(
            game.spec, game.pending, "teacher query failed", exc
        ) from exc
    timings.add("teacher", "teacher_queries", time.perf_counter() - begin)
    return _require_index(game, index, "teacher")


def _neural_choices(
    chunk: Sequence[_RunningGame], neural: Any, timings: _Timings
) -> None:
    active: list[_RunningGame] = []
    for game in chunk:
        if _candidate_count(game.pending) <= FORCED_CANDIDATE_COUNT:
            game.chosen = 0
        else:
            active.append(game)
    if not active:
        return
    device = getattr(neural, "device", None)
    _synchronize(device)
    begin = time.perf_counter()
    try:
        indices = neural.select_indices([game.pending for game in active])
    except Exception as exc:
        raise _context_error(
            active[0].spec,
            active[0].pending,
            f"neural inference failed for a batch of {len(active)} decision(s)",
            exc,
        ) from exc
    _synchronize(device)
    timings.add("inference", "inference_batches", time.perf_counter() - begin)
    try:
        values = list(indices)
    except TypeError as exc:
        raise _context_error(
            active[0].spec,
            active[0].pending,
            "neural inference did not return an index sequence",
            exc,
        ) from exc
    if len(values) != len(active):
        raise _context_error(
            active[0].spec,
            active[0].pending,
            f"neural inference returned {len(values)} indices for "
            f"{len(active)} decisions",
        )
    for game, value in zip(active, values):
        game.chosen = _require_index(game, value, "neural")
        game.inference_rows += 1
    timings.calls["inference_rows"] += len(active)


def _require_index(game: _RunningGame, index: Any, source: str) -> int:
    count = _candidate_count(game.pending)
    if isinstance(index, bool) or isinstance(index, (float, str, bytes)):
        raise _context_error(
            game.spec,
            game.pending,
            f"{source} returned a non-integer candidate index {index!r}",
        )
    try:
        value = int(operator.index(index))
    except TypeError as exc:
        raise _context_error(
            game.spec,
            game.pending,
            f"{source} returned a non-integer candidate index {index!r}",
            exc,
        ) from exc
    if value < 0 or value >= count:
        raise _context_error(
            game.spec,
            game.pending,
            f"{source} returned candidate index {value} outside "
            f"{count} candidates",
        )
    return value


def _synchronize(device: Any) -> None:
    """Make CUDA inference timing include queued GPU work."""

    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _record_choice(game: _RunningGame, index: int) -> None:
    pending = game.pending
    forced = _candidate_count(pending) <= FORCED_CANDIDATE_COUNT
    game.decisions += 1
    if forced:
        game.forced_decisions += 1
    if game.policy == POLICY_NEURAL:
        game.neural_decisions += 1
        if forced:
            game.neural_forced_decisions += 1
    if game.trace is not None:
        game.trace.append(
            {
                "decision": game.decisions,
                "decision_id": int(pending.decision_id),
                "actor_seat": int(pending.actor_seat),
                "stage": int(pending.stage),
                "candidate_count": _candidate_count(pending),
                "policy": game.policy,
                "forced": forced,
                "index": index,
            }
        )


def _finish_game(game: _RunningGame, terminal: Any) -> GameResult:
    decisions = int(terminal.decisions)
    if decisions != game.decisions:
        raise _context_error(
            game.spec,
            None,
            f"environment terminal decision count {decisions} does not match "
            f"scheduler decision count {game.decisions}",
        )
    return GameResult(
        seed=game.spec.seed,
        neural_seats=game.spec.neural_seats,
        scores=tuple(int(score) for score in terminal.scores),
        winners=tuple(
            seat for seat, won in enumerate(tuple(terminal.winners)) if won
        ),
        end_reason=str(terminal.end_reason),
        decisions=decisions,
        neural_decisions=game.neural_decisions,
        neural_forced_decisions=game.neural_forced_decisions,
        forced_decisions=game.forced_decisions,
        inference_rows=game.inference_rows,
        trace=None if game.trace is None else tuple(game.trace),
    )


def _cache_status(cache: Any) -> Any:
    status = getattr(cache, "status", None)
    if status is None:
        return None
    try:
        return status()
    except Exception as exc:
        raise SchedulingError(
            f"teacher cache status failed: {type(exc).__name__}: {exc}"
        ) from exc


def _context_error(
    spec: GameSpec,
    pending: Any,
    message: str,
    cause: BaseException | None = None,
) -> SchedulingError:
    parts = [f"seed {spec.seed}", f"placement {spec.placement_text()}"]
    if pending is not None:
        parts.extend(
            [
                f"decision {int(pending.decision_id)}",
                f"seat {int(pending.actor_seat)}",
                f"stage {int(pending.stage)}",
                f"candidates {int(pending.candidate_count)}",
            ]
        )
    suffix = "" if cause is None else f": {type(cause).__name__}: {cause}"
    return SchedulingError(f"{message} ({', '.join(parts)}){suffix}")


def summarize_games(
    games: Sequence[GameResult],
    calls: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Recompute decision, inference, and teacher counts from completed games.

    ``calls`` adds scheduler-observed call totals (inference batches, teacher
    queries, inference rows). Those call counts are cross-checked against the
    per-game accounting by :func:`run_games`.
    """

    values = list(games)
    total = sum(game.decisions for game in values)
    neural = sum(game.neural_decisions for game in values)
    neural_forced = sum(game.neural_forced_decisions for game in values)
    forced = sum(game.forced_decisions for game in values)
    inference_rows = sum(game.inference_rows for game in values)
    call_counts = dict(calls or {})
    return {
        "games": len(values),
        "decisions": {
            "total": total,
            "neural": neural,
            "teacher": total - neural,
            "forced": forced,
            "neural_forced": neural_forced,
            "teacher_forced": forced - neural_forced,
        },
        "inference": {
            "rows": inference_rows,
            "batches": int(call_counts.get("inference_batches", 0)),
            "observed_rows": int(call_counts.get("inference_rows", inference_rows)),
        },
        "teacher": {
            "queries": int(call_counts.get("teacher_queries", 0)),
        },
    }


def _validate_counts(games: Sequence[GameResult], counts: Mapping[str, Any]) -> None:
    decisions = counts["decisions"]
    if decisions["forced"] > decisions["total"]:
        raise SchedulingError(
            "internal error: forced decision count exceeds the total"
        )
    if counts["inference"]["rows"] != counts["inference"]["observed_rows"]:
        raise SchedulingError(
            "internal error: inference row count cannot be recomputed from games"
        )
    if counts["teacher"]["queries"] != decisions["teacher"] - decisions["teacher_forced"]:
        raise SchedulingError(
            "internal error: teacher query count does not match teacher decisions "
            "with more than one candidate"
        )
    for game in games:
        if game.inference_rows != game.neural_decisions - game.neural_forced_decisions:
            raise SchedulingError(
                "internal error: inference rows do not match non-forced neural "
                f"decisions for seed {game.seed} placement {game.neural_seats}"
            )
        if game.neural_forced_decisions > game.forced_decisions:
            raise SchedulingError(
                "internal error: neural forced decisions exceed the forced total "
                f"for seed {game.seed} placement {game.neural_seats}"
            )
