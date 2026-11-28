"""2N2T compatibility adapter for the synchronous execution kernel."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import execution
from ._terminal import SEAT_COUNT, U64_MAX, is_int
from .execution import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENT_GAMES,
    FORCED_CANDIDATE_COUNT,
)
from .records import TerminalRecord

POLICY_NEURAL = "neural"
POLICY_TEACHER = "heuristic"
POLICIES = (POLICY_NEURAL, POLICY_TEACHER)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONCURRENT_GAMES",
    "FORCED_CANDIDATE_COUNT",
    "POLICIES",
    "POLICY_NEURAL",
    "POLICY_TEACHER",
    "GameResult",
    "GameSpec",
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
        if not is_int(seed):
            raise SchedulingError("game seed must be an integer")
        if seed < 0 or seed > U64_MAX:
            raise SchedulingError(f"game seed {seed} must fit in an unsigned u64")
        try:
            seats = tuple(self.neural_seats)
        except TypeError as exc:
            raise SchedulingError("neural_seats must be a two-seat sequence") from exc
        if len(seats) != 2:
            raise SchedulingError("neural_seats must contain exactly two seats")
        for seat in seats:
            if not is_int(seat):
                raise SchedulingError("neural seats must be integers")
            if seat < 0 or seat >= SEAT_COUNT:
                raise SchedulingError(
                    f"neural seat {seat} must be in [0, {SEAT_COUNT})"
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
    """Play a mixed 2N2T plan using the shared execution kernel.

    The greedy comparison policy exposes ``select_indices(decisions)``. Every
    completed game is delivered to ``on_terminal`` before refill. A failure
    raises :class:`SchedulingError` and leaves earlier callback effects intact.
    """

    games = _normalize_specs(specs)
    select_indices = getattr(neural, "select_indices", None)
    try:
        execution._require_synchronous(
            select_indices, "neural policy select_indices", allow_generator=True
        )
        if on_terminal is not None:
            execution._require_synchronous(on_terminal, "on_terminal")
    except execution.ExecutionError as exc:
        raise SchedulingError(str(exc)) from exc
    by_id = {str(index): spec for index, spec in enumerate(games)}
    plan = [
        execution.GameSpec(
            game_id,
            spec.seed,
            tuple(
                POLICY_NEURAL if seat in spec.neural_seats else POLICY_TEACHER
                for seat in range(SEAT_COUNT)
            ),
        )
        for game_id, spec in by_id.items()
    ]
    device = getattr(neural, "device", None)
    cache = teacher_cache if teacher_cache is not None else execution._disabled_cache()
    completed: dict[str, GameResult] = {}

    def select(requests: Sequence[execution.DecisionRequest]) -> Sequence[int]:
        return select_indices([request.decision for request in requests])

    def notify(game: execution.GameResult) -> None:
        projected = _project_game(game, by_id[game.game_id])
        completed[game.game_id] = projected
        return on_terminal(projected)

    try:
        result = execution.execute_games(
            plan,
            policies={
                POLICY_NEURAL: execution._SynchronizedSelector(
                    select, lambda: _synchronize(device)
                ),
                POLICY_TEACHER: execution.NativeTeacher(),
            },
            teacher_cache=cache,
            batch_size=batch_size,
            concurrent_games=len(plan)
            if concurrent_games is None
            else concurrent_games,
            decision_limit=decision_limit,
            on_terminal=notify if on_terminal is not None else None,
            trace=trace,
            diagnostics=True,
            env_factory=env_factory,
        )
    except execution.ExecutionError as exc:
        message = str(exc)
        prefix = f"selector {POLICY_NEURAL!r}"
        if message.startswith(prefix):
            message = "neural inference" + message[len(prefix) :]
        if exc.game_id is not None:
            message += f" (placement {by_id[exc.game_id].placement_text()})"
        raise SchedulingError(message) from exc

    ordered = tuple(
        completed[game.game_id]
        if on_terminal is not None
        else _project_game(game, by_id[game.game_id])
        for game in result.games
    )
    assert result.diagnostics is not None
    timings = _legacy_timings(result.diagnostics, device)
    counts = summarize_games(ordered, timings["calls"])
    counts["teacher"]["cache"] = _cache_status(cache)
    _validate_counts(counts)
    return SchedulingResult(games=ordered, counts=counts, timings=timings)


def _project_game(game: execution.GameResult, spec: GameSpec) -> GameResult:
    neural = sum(game.decisions_by_seat[seat] for seat in spec.neural_seats)
    neural_forced = sum(game.forced_by_seat[seat] for seat in spec.neural_seats)
    trace = (
        None
        if game.trace is None
        else tuple(
            {
                "decision": entry.decision,
                "decision_id": entry.decision_id,
                "actor_seat": entry.actor_seat,
                "stage": entry.stage,
                "candidate_count": entry.candidate_count,
                "policy": entry.policy_id,
                "forced": entry.forced,
                "index": entry.index,
            }
            for entry in game.trace
        )
    )
    return GameResult(
        seed=game.seed,
        neural_seats=spec.neural_seats,
        scores=game.scores,
        winners=game.winners,
        end_reason=game.end_reason,
        decisions=game.decisions,
        neural_decisions=neural,
        neural_forced_decisions=neural_forced,
        forced_decisions=game.forced_decisions,
        inference_rows=neural - neural_forced,
        trace=trace,
    )


def _legacy_timings(
    stats: execution.ExecutionDiagnostics, device: Any
) -> dict[str, Any]:
    neural = stats.policies[POLICY_NEURAL]
    teacher = stats.policies[POLICY_TEACHER]
    seconds = {
        "env_reset": stats.env_reset_seconds,
        "env_step": stats.env_step_seconds,
        "teacher": teacher.selection_seconds,
        "inference": neural.selection_seconds,
    }
    unaccounted = max(0.0, stats.total_seconds - sum(seconds.values()))
    return {
        "instrument": "time.perf_counter",
        "device": "unknown" if device is None else str(device),
        "cuda_synchronized_inference": getattr(device, "type", None) == "cuda",
        "boundary": (
            "CUDA is synchronized before and after each inference batch, "
            "so inference seconds include queued GPU work and the other "
            "timers exclude it. Environment and teacher timers cover the "
            "single-threaded Rust/CPU call only."
        ),
        "seconds": {
            **seconds,
            "unaccounted": unaccounted,
            "total": stats.total_seconds,
        },
        "calls": {
            "env_resets": stats.env_resets,
            "env_steps": stats.env_steps,
            "teacher_queries": teacher.teacher_queries,
            "inference_batches": neural.selector_batches,
            "inference_rows": neural.selector_rows,
        },
    }


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


def _synchronize(device: Any) -> None:
    """Make CUDA inference timing include queued GPU work."""

    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


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


def _validate_counts(counts: Mapping[str, Any]) -> None:
    decisions = counts["decisions"]
    if counts["inference"]["rows"] != counts["inference"]["observed_rows"]:
        raise SchedulingError(
            "internal error: inference row count cannot be recomputed from games"
        )
    if (
        counts["teacher"]["queries"]
        != decisions["teacher"] - decisions["teacher_forced"]
    ):
        raise SchedulingError(
            "internal error: teacher query count does not match teacher decisions "
            "with more than one candidate"
        )
