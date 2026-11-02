#!/usr/bin/env python3
"""Reproducible mixed-policy A/B experiment (Python standard library only).

This module drives the experiment described in
``specs/active/reproducible-policy-ab/design.md``: two A seats running the
imported ninone heuristic against two B seats running the simulator's
``FirstLegalPolicy``, on six balanced seat placements per game seed.

Workflow: ``pilot -> plan -> main -> summarize``.

* ``pilot`` runs a fixed, non-overlapping pilot seed sample to measure the
  variation of seed-block contrasts and the runtime per seed.
* ``plan`` uses the pilot to freeze the main sample size from an explicit
  target precision (95% interval half-width) or an explicit runtime budget.
  The main seed list, seed procedure, analysis settings, and adapter identity
  are written to ``plan.json`` before any main result is observed.
* ``main`` runs the frozen main sample and validates the driver output.
  A non-zero exit, any runner failure record, and missing, duplicate, or
  malformed entries invalidate the run: failures are preserved and no normal
  summary is published.
* ``summarize`` recomputes the summary from stored per-game results with
  equal-weight seed blocks, sample SD, ``sample_sd / sqrt(n)``, and a
  whole-block percentile bootstrap interval. Symmetric modes additionally
  require every seed block contrast to be zero.

``plan`` inherits omitted ``--mode``/``--decision-limit`` from the pilot report
and rejects any explicit value that differs from what the pilot actually ran.
``main`` inherits omitted mode, decision limit, and bootstrap settings from
``plan.json``; explicitly supplied values must still match the plan.

Before ``pilot``/``main`` start, both repositories' revisions/diffs and the
actual bytes of identity-relevant sources (driver source, worker, analysis
script, Rust build inputs, ninone heuristic) are captured under the work
directory. The manifest reuses those pre-run captures, and a post-run identity
comparison invalidates the stage if any of them changed while it ran.
``summarize`` writes recomputation diagnostics to
``main/summarize_failures.json`` and never overwrites the original
``main/failures.json`` from the main process.

Frozen cross-component contract (Rust driver, implemented elsewhere)::

    policy_ab --seeds PATH --output PATH --python EXE --worker PATH
              --ninone-root PATH --mode mixed|first-only|heuristic-only
              --decision-limit N

``--seeds`` is a JSON array of u64 game seeds. ``--output`` is JSON Lines with
one object per successfully completed game::

    {"seed": u64, "a_seats": [i, j], "scores": [i64, i64, i64, i64],
     "end_reason": string, "decisions": u64}

Placements are the six fixed pairs in lexicographic order:
``[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]`` (``a_seats`` is normalized by
sorting, so either pair order is accepted). Extra fields such as timing are
ignored because they are not part of deterministic result identity.
``end_reason`` must be one of the simulator's snake_case terminal reasons
(``three_winners``, ``wall_exhausted``), and ``decisions`` must be positive.
Records may carry ``trace_hash`` (FNV-1a64 of the action trace, 16 lowercase
hex characters); it is validated, preserved, and compared during the
deterministic-replay check so that equal scores and decision counts cannot
mask different action sequences.

The manifest binds recorded sources to the executed driver: it hashes the
driver executable, its Rust build inputs (``Cargo.toml``, ``Cargo.lock``,
``src/**/*.rs``, ``examples/**/*.rs``), the worker and analysis sources, the
actual Python interpreter, and the imported ninone revision, and records a
build-order freshness check between the driver binary and those sources.

No early stopping is implemented, and pilot seeds are never mixed into the
main estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = 1

#: Fixed lexicographic seat placements. Each policy occupies each seat three
#: times across the six placements of a seed block.
PLACEMENTS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PLACEMENT_INDEX = {placement: index for index, placement in enumerate(PLACEMENTS)}

MODES = ("mixed", "first-only", "heuristic-only")
#: Modes where every seat uses the same policy; every block contrast must be
#: exactly zero after seat averaging.
SYMMETRIC_MODES = ("first-only", "heuristic-only")

#: End reasons the simulator reports for completed games (snake_case). Records
#: with other strings are still accepted, but the summary flags them so an
#: unplanned termination cannot pass as a normal result.
KNOWN_END_REASONS = ("three_winners", "wall_exhausted")

#: Inference needs at least two seed blocks for a sample SD.
MIN_INFERENCE_SEEDS = 2
U64_MAX = (1 << 64) - 1

#: Two-sided 95% normal quantile, used only to translate the pilot block SD
#: into a planned sample size (the reported interval comes from the bootstrap).
Z_95 = 1.959963984540054
CONFIDENCE = 0.95

DEFAULT_SEED_PROCEDURE_SEED = 20250814
DEFAULT_BOOTSTRAP_SEED = 20250815
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_DECISION_LIMIT = 10000

#: Artifact evidence the manifest must carry before a normal summary is
#: published. The summary is recomputed from stored per-game records only; it
#: never triggers a replay.
REQUIRED_MANIFEST_ARTIFACTS = (
    "main_seeds_path",
    "main_seeds_sha256",
    "main_results_path",
    "main_results_sha256",
    "main_run_path",
    "main_run_sha256",
)

MAX_SNAPSHOT_FILE_BYTES = 256 * 1024
MAX_SNAPSHOT_TOTAL_BYTES = 8 * 1024 * 1024

#: Budget for copying the actual bytes of identity-relevant sources (worker,
#: analysis script, Rust build inputs, ninone heuristic) before a run. Content
#: copies make even ignored or untracked source files recoverable; a file that
#: cannot be copied is reported explicitly instead of being reduced to a hash.
MAX_SOURCE_SNAPSHOT_TOTAL_BYTES = 32 * 1024 * 1024

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_WORKER_PATH = SCRIPT_DIR / "worker.py"
DEFAULT_NINONE_ROOT = DEFAULT_REPO_ROOT.parent / "ninone-mahjong-agent"


class ExperimentError(Exception):
    """Invalid inputs, artifacts, or environment; the step cannot continue."""


class PlanningError(ExperimentError):
    """The pilot cannot support a plan for the requested precision/budget."""


class SummaryError(ExperimentError):
    """The collected data cannot be summarized as a normal A/B result."""


class SummarizeFailure(ExperimentError):
    """Re-summarizing stored artifacts failed; carries derived diagnostics."""

    def __init__(self, message: str, problems: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ExperimentError(f"expected artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"artifact is not valid JSON: {path}: {exc}") from exc


def remove_if_exists(path: Path) -> None:
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def relative_display(path: Path, reference: Path) -> str:
    try:
        return os.path.relpath(str(path), str(reference)).replace("\\", "/")
    except ValueError:  # different drives on Windows
        return str(path)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation with ``n - 1`` in the denominator."""
    n = len(values)
    if n < 2:
        raise ValueError(f"sample standard deviation requires at least 2 values, got {n}")
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (n - 1))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (R type 7 / numpy default) of sorted values."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile quantile out of range: {q}")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower])


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _mtime_utc_from_epoch(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mtime_utc(path: Path) -> str | None:
    try:
        return _mtime_utc_from_epoch(path.stat().st_mtime)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Reproducible seed corpus
# ---------------------------------------------------------------------------


class SplitMix64:
    """Deterministic PRNG (Steele/Lea/Flood splitmix64).

    Used for the seed corpus and for bootstrap resampling so that both are
    reproducible independently of the Python version.
    """

    MASK = (1 << 64) - 1
    GOLDEN = 0x9E3779B97F4A7C15

    def __init__(self, seed: int) -> None:
        if not is_int(seed) or not 0 <= seed <= U64_MAX:
            raise ExperimentError(f"splitmix64 seed must be a u64, got {seed!r}")
        self._state = seed

    def next_u64(self) -> int:
        self._state = (self._state + self.GOLDEN) & self.MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def index(self, n: int) -> int:
        """Strictly uniform index in ``[0, n)`` by u64 rejection sampling.

        Unbiased and version stable: values at or above the largest multiple of
        ``n`` below ``2**64`` are rejected and redrawn.
        """
        if n <= 0:
            raise ValueError("index bound must be positive")
        if n == 1:
            return 0
        bound = 1 << 64
        limit = bound - (bound % n)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % n


SEED_PROCEDURE_GENERATOR = "splitmix64-v1"


def seed_procedure(master_seed: int = DEFAULT_SEED_PROCEDURE_SEED) -> dict[str, Any]:
    return {
        "generator": SEED_PROCEDURE_GENERATOR,
        "master_seed": master_seed,
        "draw": "sequential u64 outputs of splitmix64-v1",
        "duplicates": "rejected; first accepted draw order is the corpus order",
        "bounds": [0, U64_MAX],
    }


def validate_seed_procedure(procedure: dict[str, Any]) -> None:
    if not isinstance(procedure, dict):
        raise ExperimentError("seed procedure must be an object")
    if procedure.get("generator") != SEED_PROCEDURE_GENERATOR:
        raise ExperimentError(
            f"unsupported seed procedure generator: {procedure.get('generator')!r}"
        )
    master = procedure.get("master_seed")
    if not is_int(master) or not 0 <= master <= U64_MAX:
        raise ExperimentError(f"seed procedure master_seed must be a u64, got {master!r}")


def generate_seeds(procedure: dict[str, Any], count: int) -> list[int]:
    """Return the first ``count`` distinct u64 seeds of the procedure stream."""
    validate_seed_procedure(procedure)
    if count < 0:
        raise ExperimentError("seed count must be non-negative")
    rng = SplitMix64(int(procedure["master_seed"]))
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < count:
        value = rng.next_u64()
        if value in seen:
            continue
        seen.add(value)
        seeds.append(value)
    return seeds


def plan_main_seeds(procedure: dict[str, Any], pilot_size: int, main_size: int) -> list[int]:
    """Main seeds are the corpus positions after the pilot block (no overlap)."""
    if pilot_size < 0 or main_size < 0:
        raise ExperimentError("pilot/main seed counts must be non-negative")
    corpus = generate_seeds(procedure, pilot_size + main_size)
    main = corpus[pilot_size:]
    overlap = set(main) & set(corpus[:pilot_size])
    if overlap:  # defensive: the procedure rejects duplicates already
        raise ExperimentError(f"seed corpus produced overlapping pilot/main seeds: {sorted(overlap)[:5]}")
    return main


# ---------------------------------------------------------------------------
# Per-game records, problems, and run validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameResult:
    seed: int
    placement: tuple[int, int]
    scores: tuple[int, ...]
    end_reason: str
    decisions: int
    trace_hash: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "seed": self.seed,
            "a_seats": list(self.placement),
            "scores": list(self.scores),
            "end_reason": self.end_reason,
            "decisions": self.decisions,
        }
        if self.trace_hash is not None:
            payload["trace_hash"] = self.trace_hash
        return payload


@dataclass(frozen=True)
class Problem:
    kind: str
    detail: str
    seed: int | None = None
    placement: tuple[int, int] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "detail": self.detail}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.placement is not None:
            payload["placement"] = list(self.placement)
        if self.context:
            payload["context"] = self.context
        return payload


@dataclass
class RunEvaluation:
    ok: bool
    results: list[GameResult]
    problems: list[Problem]
    by_seed: dict[int, dict[tuple[int, int], GameResult]]


def _normalize_placement(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(is_int(seat) for seat in value):
        return None
    seat_a, seat_b = int(value[0]), int(value[1])
    if seat_a == seat_b:
        return None
    pair = (min(seat_a, seat_b), max(seat_a, seat_b))
    if pair not in PLACEMENT_INDEX:
        return None
    return pair


#: Record keys that mark a runner-reported failure instead of a completed
#: game. Their presence rejects the record even if all game fields are present.
FAILURE_MARKER_KEYS = ("error", "errors", "failure", "exception", "traceback")
FAILURE_STATUS_KEYS = ("status", "record_type", "kind", "type")
FAILURE_STATUS_VALUES = ("failure", "failed", "error", "fatal")


def _failure_marker(obj: dict[str, Any]) -> str | None:
    for key in FAILURE_MARKER_KEYS:
        if key in obj and obj[key] not in (None, "", [], {}, False):
            return key
    for key in FAILURE_STATUS_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip().lower() in FAILURE_STATUS_VALUES:
            return f"{key}={value}"
    return None


def _is_trace_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_game_object(
    obj: dict[str, Any], line_no: int, problems: list[Problem], require_zero_sum: bool
) -> GameResult | None:
    marker = _failure_marker(obj)
    if marker is not None:
        # Any error/failure record invalidates the run, even when it also
        # carries complete-looking game fields.
        problems.append(
            Problem(
                "runner_failure_record",
                f"line {line_no}: record carries failure marker {marker!r}; "
                "treated as a runner failure record",
                seed=obj.get("seed") if is_int(obj.get("seed")) else None,
                placement=_normalize_placement(obj.get("a_seats")),
                context={"line": line_no, "marker": marker, "record": _truncate_json(obj)},
            )
        )
        return None

    missing = [key for key in ("seed", "a_seats", "scores", "end_reason", "decisions") if key not in obj]
    if missing:
        # A runner-reported failure record: preserve its seed/placement context.
        problems.append(
            Problem(
                "runner_failure_record",
                f"line {line_no}: record is missing {sorted(missing)}; treated as a runner failure record",
                seed=obj.get("seed") if is_int(obj.get("seed")) else None,
                placement=_normalize_placement(obj.get("a_seats")),
                context={"line": line_no, "record": _truncate_json(obj)},
            )
        )
        return None

    seed = obj["seed"]
    if not is_int(seed) or not 0 <= seed <= U64_MAX:
        problems.append(
            Problem(
                "invalid_seed",
                f"line {line_no}: seed must be a u64, got {seed!r}",
                context={"line": line_no},
            )
        )
        return None

    placement = _normalize_placement(obj["a_seats"])
    if placement is None:
        problems.append(
            Problem(
                "invalid_placement",
                f"line {line_no}: a_seats {obj['a_seats']!r} is not one of the six fixed placements",
                seed=seed,
                context={"line": line_no, "a_seats": obj["a_seats"]},
            )
        )
        return None

    scores = obj["scores"]
    if (
        not isinstance(scores, list)
        or len(scores) != 4
        or not all(is_int(score) for score in scores)
    ):
        problems.append(
            Problem(
                "invalid_scores",
                f"line {line_no}: scores must be four integers, got {scores!r}",
                seed=seed,
                placement=placement,
                context={"line": line_no},
            )
        )
        return None

    end_reason = obj["end_reason"]
    if not isinstance(end_reason, str) or end_reason not in KNOWN_END_REASONS:
        problems.append(
            Problem(
                "invalid_end_reason",
                f"line {line_no}: end_reason {end_reason!r} is not one of the terminal "
                f"reasons {list(KNOWN_END_REASONS)}",
                seed=seed,
                placement=placement,
                context={"line": line_no},
            )
        )
        return None

    decisions = obj["decisions"]
    if not is_int(decisions) or decisions <= 0:
        problems.append(
            Problem(
                "invalid_decisions",
                f"line {line_no}: decisions must be a positive integer, got {decisions!r}",
                seed=seed,
                placement=placement,
                context={"line": line_no},
            )
        )
        return None

    if require_zero_sum and sum(scores) != 0:
        problems.append(
            Problem(
                "nonzero_score_sum",
                f"line {line_no}: scores {scores} sum to {sum(scores)}, expected 0",
                seed=seed,
                placement=placement,
                context={"line": line_no, "scores": list(scores)},
            )
        )
        return None

    trace_hash = obj.get("trace_hash")
    if trace_hash is not None and not _is_trace_hash(trace_hash):
        problems.append(
            Problem(
                "invalid_trace_hash",
                f"line {line_no}: trace_hash must be 16 lowercase hex characters, "
                f"got {trace_hash!r}",
                seed=seed,
                placement=placement,
                context={"line": line_no},
            )
        )
        return None

    return GameResult(
        seed=int(seed),
        placement=placement,
        scores=tuple(int(score) for score in scores),
        end_reason=end_reason,
        decisions=int(decisions),
        trace_hash=trace_hash,
    )


def _truncate_json(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= limit:
        return value
    return {"truncated": True, "text": text[:limit]}


def _group(results: Iterable[GameResult]) -> dict[int, dict[tuple[int, int], GameResult]]:
    grouped: dict[int, dict[tuple[int, int], GameResult]] = {}
    for result in results:
        grouped.setdefault(result.seed, {})[result.placement] = result
    return grouped


def load_output(
    seeds: Sequence[int], output_path: Path, *, require_zero_sum: bool = True
) -> RunEvaluation:
    """Parse and validate a driver JSONL output against the planned seeds."""
    planned = [int(seed) for seed in seeds]
    planned_set = set(planned)
    output_path = Path(output_path)
    problems: list[Problem] = []
    results: list[GameResult] = []

    if not output_path.is_file():
        problems.append(
            Problem("missing_output", f"driver output does not exist: {output_path}")
        )
        return RunEvaluation(ok=False, results=[], problems=problems, by_seed={})

    with output_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                problems.append(
                    Problem(
                        "malformed_record",
                        f"line {line_no}: not valid JSON: {exc}",
                        context={"line": line_no, "raw": stripped[:4000]},
                    )
                )
                continue
            if not isinstance(obj, dict):
                problems.append(
                    Problem(
                        "malformed_record",
                        f"line {line_no}: record is not a JSON object",
                        context={"line": line_no, "raw": stripped[:4000]},
                    )
                )
                continue
            parsed = _parse_game_object(obj, line_no, problems, require_zero_sum)
            if parsed is not None:
                results.append(parsed)

    by_seed = _group(results)
    seen: set[tuple[int, tuple[int, int]]] = set()
    for result in results:
        key = (result.seed, result.placement)
        if key in seen:
            problems.append(
                Problem(
                    "duplicate_entry",
                    f"duplicate result for seed {result.seed} placement {list(result.placement)}",
                    seed=result.seed,
                    placement=result.placement,
                )
            )
        seen.add(key)

    for seed in sorted(by_seed):
        if seed not in planned_set:
            problems.append(
                Problem(
                    "unexpected_seed",
                    f"seed {seed} was not part of the planned sample",
                    seed=seed,
                )
            )

    for seed in planned:
        block = by_seed.get(seed)
        if not block:
            problems.append(
                Problem("missing_seed", f"no results for planned seed {seed}", seed=seed)
            )
            continue
        missing = [placement for placement in PLACEMENTS if placement not in block]
        if missing:
            problems.append(
                Problem(
                    "missing_placement",
                    f"seed {seed} is missing placements {[list(p) for p in missing]}",
                    seed=seed,
                    placement=missing[0],
                    context={"missing_placements": [list(p) for p in missing]},
                )
            )

    return RunEvaluation(
        ok=not problems,
        results=results,
        problems=problems,
        by_seed=by_seed,
    )


# ---------------------------------------------------------------------------
# Driver invocation
# ---------------------------------------------------------------------------


@dataclass
class DriverRun:
    command: list[str]
    returncode: int
    elapsed_seconds: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    output_path: Path
    timeout_seconds: float | None = None


def resolve_driver_command(driver: Path) -> list[str]:
    """Resolve the driver invocation.

    ``--driver`` normally points at the built ``policy_ab`` example binary. A
    ``*.py`` driver runs under the current interpreter, which keeps smoke tests
    and adapter experiments possible without extra wrappers.
    """
    driver = Path(driver)
    if not driver.exists():
        raise ExperimentError(f"driver executable not found: {driver}")
    if driver.suffix.lower() == ".py":
        return [sys.executable, str(driver)]
    return [str(driver)]


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def invoke_driver(
    *,
    driver_command: Sequence[str],
    seeds_path: Path,
    output_path: Path,
    python_exe: Path,
    worker_path: Path,
    ninone_root: Path,
    mode: str,
    decision_limit: int,
    timeout: float | None = None,
) -> DriverRun:
    if mode not in MODES:
        raise ExperimentError(f"unsupported mode {mode!r}; expected one of {MODES}")
    if not is_int(decision_limit) or decision_limit <= 0:
        raise ExperimentError(f"decision limit must be a positive integer, got {decision_limit!r}")
    if Path(output_path).exists():
        # Preserve prior attempt evidence; a failed run must not be silently
        # overwritten by a second one.
        raise ExperimentError(
            f"driver output already exists: {output_path}; refusing to overwrite a "
            "previous attempt. Use a new work directory for a new attempt."
        )
    command = [
        *[str(part) for part in driver_command],
        "--seeds",
        str(seeds_path),
        "--output",
        str(output_path),
        "--python",
        str(python_exe),
        "--worker",
        str(worker_path),
        "--ninone-root",
        str(ninone_root),
        "--mode",
        mode,
        "--decision-limit",
        str(decision_limit),
    ]
    started = time.monotonic()
    timed_out = False
    returncode = -1
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
        returncode = -1
    elapsed = time.monotonic() - started
    return DriverRun(
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        output_path=Path(output_path),
        timeout_seconds=timeout,
    )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def evaluate_run(seeds: Sequence[int], run: DriverRun) -> RunEvaluation:
    """Combine process-level and record-level validation for one driver run."""
    stage = load_output(seeds, run.output_path)
    problems: list[Problem] = []
    if run.timed_out:
        problems.append(
            Problem(
                "driver_timeout",
                "driver exceeded the configured timeout",
                context={
                    "timeout_seconds": run.timeout_seconds,
                    "elapsed_seconds": run.elapsed_seconds,
                },
            )
        )
    if run.returncode != 0:
        problems.append(
            Problem(
                "nonzero_exit",
                f"driver exited with status {run.returncode}",
                context={"exit_code": run.returncode, "stderr_tail": run.stderr_tail},
            )
        )
    problems.extend(stage.problems)
    return RunEvaluation(
        ok=not problems,
        results=stage.results,
        problems=problems,
        by_seed=stage.by_seed,
    )


def failures_payload(
    evaluation: RunEvaluation, *, stage: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "stage": stage,
        "generated_at_utc": utc_now(),
        "normal_summary_published": False,
        "note": (
            "A run with any problem is invalid. The failing seeds, placements, "
            "and context are preserved here; no normal A/B summary is published."
        ),
        "problems": [problem.to_json() for problem in evaluation.problems],
        "failing_seeds": sorted({p.seed for p in evaluation.problems if p.seed is not None}),
        "failing_placements": [
            list(p.placement) for p in evaluation.problems if p.placement is not None
        ],
        "preserved_results": [result.to_json() for result in evaluation.results],
    }
    if extra:
        payload["extra"] = extra
    return payload


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastStats:
    seeds: tuple[int, ...]
    block_contrasts: tuple[float, ...]
    placement_a_means: tuple[float, ...]
    placement_b_means: tuple[float, ...]
    a_mean: float
    b_mean: float
    delta: float
    sample_sd: float
    standard_error: float


def compute_contrast_stats(
    by_seed: dict[int, dict[tuple[int, int], GameResult]], seeds: Sequence[int]
) -> ContrastStats:
    """Compute equal-weight seed-block contrasts.

    ``a[s,c]`` and ``b[s,c]`` are the mean terminal net scores of the placement's
    two A and two B players, ``d[s,c] = a - b``, ``D[s]`` the mean over the six
    placements, and ``Delta`` the mean over seeds.
    """
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < MIN_INFERENCE_SEEDS:
        raise SummaryError(
            f"at least {MIN_INFERENCE_SEEDS} seed blocks are required for inference; got {len(seed_list)}"
        )
    blocks: list[float] = []
    a_values: list[float] = []
    b_values: list[float] = []
    placement_a: list[list[float]] = [[] for _ in PLACEMENTS]
    placement_b: list[list[float]] = [[] for _ in PLACEMENTS]
    for seed in seed_list:
        block = by_seed.get(seed)
        if not block:
            raise SummaryError(f"missing results for seed {seed}")
        deltas: list[float] = []
        for index, placement in enumerate(PLACEMENTS):
            record = block.get(placement)
            if record is None:
                raise SummaryError(
                    f"missing placement {list(placement)} for seed {seed}"
                )
            others = tuple(seat for seat in range(4) if seat not in placement)
            a_score = (record.scores[placement[0]] + record.scores[placement[1]]) / 2.0
            b_score = (record.scores[others[0]] + record.scores[others[1]]) / 2.0
            deltas.append(a_score - b_score)
            a_values.append(a_score)
            b_values.append(b_score)
            placement_a[index].append(a_score)
            placement_b[index].append(b_score)
        blocks.append(mean(deltas))
    n_blocks = len(blocks)
    sd = sample_sd(blocks)
    return ContrastStats(
        seeds=tuple(seed_list),
        block_contrasts=tuple(blocks),
        placement_a_means=tuple(mean(values) for values in placement_a),
        placement_b_means=tuple(mean(values) for values in placement_b),
        a_mean=mean(a_values),
        b_mean=mean(b_values),
        delta=mean(blocks),
        sample_sd=sd,
        standard_error=sd / math.sqrt(n_blocks),
    )


def bootstrap_delta_ci(
    blocks: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """95% percentile bootstrap interval from whole-block resampling."""
    if len(blocks) < MIN_INFERENCE_SEEDS:
        raise SummaryError(
            f"bootstrap needs at least {MIN_INFERENCE_SEEDS} seed blocks; got {len(blocks)}"
        )
    if not is_int(resamples) or resamples < 1:
        raise ExperimentError(f"bootstrap resample count must be a positive integer, got {resamples!r}")
    if not is_int(seed) or not 0 <= seed <= U64_MAX:
        raise ExperimentError(f"bootstrap seed must be a u64, got {seed!r}")
    rng = SplitMix64(seed)
    n_blocks = len(blocks)
    estimates: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n_blocks):
            total += blocks[rng.index(n_blocks)]
        estimates.append(total / n_blocks)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return percentile(estimates, alpha), percentile(estimates, 1.0 - alpha)


def zero_block_check(
    stats: ContrastStats, *, tolerance: float = 1e-9
) -> dict[str, Any]:
    nonzero = [
        {"seed": seed, "block_contrast": value}
        for seed, value in zip(stats.seeds, stats.block_contrasts)
        if abs(value) > tolerance
    ]
    max_abs = max((abs(value) for value in stats.block_contrasts), default=0.0)
    return {
        "all_zero": not nonzero,
        "tolerance": tolerance,
        "max_abs_block_contrast": max_abs,
        "nonzero_blocks": nonzero,
    }


# ---------------------------------------------------------------------------
# Pilot stage
# ---------------------------------------------------------------------------


def prepare_corpus(
    work_dir: Path,
    *,
    procedure: dict[str, Any],
    pilot_size: int,
) -> list[int]:
    """Create or verify the pilot seed corpus artifact."""
    if pilot_size < MIN_INFERENCE_SEEDS:
        raise ExperimentError(
            f"pilot needs at least {MIN_INFERENCE_SEEDS} seeds to estimate a block SD; got {pilot_size}"
        )
    corpus_path = Path(work_dir) / "corpus.json"
    if corpus_path.is_file():
        corpus = read_json(corpus_path)
        stored = corpus.get("procedure") or {}
        if stored.get("generator") != procedure.get("generator") or stored.get(
            "master_seed"
        ) != procedure.get("master_seed"):
            raise ExperimentError(
                f"existing corpus {corpus_path} used a different seed procedure; "
                "use a new work directory or the recorded procedure seed"
            )
        pilot_seeds = [int(seed) for seed in corpus.get("pilot_seeds") or []]
        if len(pilot_seeds) != pilot_size:
            raise ExperimentError(
                f"existing corpus {corpus_path} has {len(pilot_seeds)} pilot seeds, "
                f"but --pilot-size expects {pilot_size}"
            )
        return pilot_seeds
    pilot_seeds = generate_seeds(procedure, pilot_size)
    write_json(
        corpus_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "procedure": procedure,
            "pilot_size": pilot_size,
            "pilot_seeds": pilot_seeds,
        },
    )
    return pilot_seeds


def run_pilot(
    work_dir: Path,
    *,
    seeds: Sequence[int],
    driver_args: dict[str, Any],
    adapter_identity: dict[str, Any],
    pre_evidence: dict[str, Any],
    post_evidence_provider: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run the pilot sample and write its report (block SD and runtime).

    Repository/source evidence is captured immediately before the driver
    starts and compared after it exits. A configuration that changed mid-pilot
    cannot support a plan for the sample it measured, so such a pilot is
    invalid and no report is published.
    """
    work_dir = Path(work_dir)
    pilot_dir = work_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    for existing in (pilot_dir / "results.jsonl", pilot_dir / "failures.json"):
        if existing.exists():
            raise ExperimentError(
                f"pilot attempt already exists at {existing}; refusing to overwrite "
                "preserved pilot evidence. Use a new work directory for a new attempt."
            )
    write_json(pilot_dir / "pre-run-evidence.json", pre_evidence)
    seeds_path = pilot_dir / "seeds.json"
    output_path = pilot_dir / "results.jsonl"
    write_json(seeds_path, list(seeds))
    run = invoke_driver(
        seeds_path=seeds_path,
        output_path=output_path,
        **driver_args,
    )
    write_json(
        pilot_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "pilot",
            "command": run.command,
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "timeout_seconds": run.timeout_seconds,
            "elapsed_seconds": run.elapsed_seconds,
            "stdout_tail": run.stdout_tail,
            "stderr_tail": run.stderr_tail,
            "seeds_path": relative_display(seeds_path, work_dir),
            "output_path": relative_display(output_path, work_dir),
            "results_sha256": sha256_file(output_path),
            "pre_run_evidence": "pilot/pre-run-evidence.json",
        },
    )
    evaluation = evaluate_run(seeds, run)
    post_evidence, identity_check = capture_post_evidence(
        post_evidence_provider, pre_evidence
    )
    write_json(
        pilot_dir / "post-run-evidence.json",
        post_evidence
        if post_evidence is not None
        else {"stage": "pilot-post-run", "capture_error": identity_check.get("error")},
    )
    problems = list(evaluation.problems)
    stage = "pilot"
    if identity_check["status"] != "stable":
        stage = "pilot-identity-drift"
        problems.append(
            Problem(
                "identity_drifted",
                "driver/worker/heuristic/repository identity changed while the pilot ran; "
                "the measured sample does not describe one configuration",
                context=identity_check,
            )
        )
    if problems:
        failed = RunEvaluation(
            ok=False,
            results=evaluation.results,
            problems=problems,
            by_seed=evaluation.by_seed,
        )
        write_json(
            pilot_dir / "failures.json",
            failures_payload(
                failed, stage=stage, extra={"identity_check": identity_check}
            ),
        )
        raise ExperimentError(
            f"pilot run invalid ({len(problems)} problem(s)); see {pilot_dir / 'failures.json'}"
        )
    stats = compute_contrast_stats(evaluation.by_seed, seeds)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "pilot",
        "created_at_utc": utc_now(),
        "mode": driver_args["mode"],
        "decision_limit": int(driver_args["decision_limit"]),
        "seeds": list(stats.seeds),
        "count": len(stats.seeds),
        "block_contrasts": list(stats.block_contrasts),
        "block_mean": stats.delta,
        "block_sd": stats.sample_sd,
        "elapsed_seconds": run.elapsed_seconds,
        "mean_seconds_per_seed": run.elapsed_seconds / len(stats.seeds),
        "results_sha256": sha256_file(output_path),
        "adapter_identity": adapter_identity,
        "identity_check": identity_check,
        "note": "Pilot results estimate block SD and runtime only; they never enter the main estimate.",
    }
    write_json(pilot_dir / "report.json", report)
    return report


# ---------------------------------------------------------------------------
# Plan stage
# ---------------------------------------------------------------------------


def build_plan(
    *,
    pilot_report: dict[str, Any],
    procedure: dict[str, Any],
    mode: str | None,
    decision_limit: int | None,
    target_precision: float | None,
    runtime_budget: float | None,
    max_main_seeds: int | None,
    analysis: dict[str, Any],
    adapter_identity: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the main sample from the pilot and an explicit precision/budget.

    Omitted ``mode``/``decision_limit`` inherit the pilot's recorded
    configuration. The pilot *ran* one driver configuration, so a requested
    value that differs is rejected instead of planning a mixed configuration.
    """
    pilot_mode = pilot_report.get("mode")
    pilot_limit = pilot_report.get("decision_limit")
    if pilot_mode not in MODES:
        raise PlanningError(
            "pilot report has no recorded mode; re-run the pilot with this script so "
            "the plan can be tied to the configuration that produced its sample"
        )
    if not is_int(pilot_limit) or pilot_limit <= 0:
        raise PlanningError(
            "pilot report has no recorded decision limit; re-run the pilot with this "
            "script so the plan can be tied to the configuration that produced its sample"
        )
    if mode is None:
        mode = pilot_mode
    elif mode != pilot_mode:
        raise PlanningError(
            f"requested mode {mode!r} does not match the pilot mode {pilot_mode!r}; "
            "the pilot measured a different configuration. Re-run the pilot or plan "
            "with the pilot's mode"
        )
    if decision_limit is None:
        decision_limit = int(pilot_limit)
    elif not is_int(decision_limit) or int(decision_limit) != int(pilot_limit):
        raise PlanningError(
            f"requested decision limit {decision_limit!r} does not match the pilot "
            f"decision limit {pilot_limit!r}; the pilot measured a different configuration"
        )
    if target_precision is None and runtime_budget is None:
        raise ExperimentError(
            "plan requires an explicit --target-precision (95% half-width) and/or --runtime-budget"
        )
    if target_precision is not None and target_precision <= 0:
        raise ExperimentError("--target-precision must be positive")
    if runtime_budget is not None and runtime_budget <= 0:
        raise ExperimentError("--runtime-budget must be positive seconds")
    if max_main_seeds is not None and max_main_seeds < MIN_INFERENCE_SEEDS:
        raise ExperimentError(
            f"--max-main-seeds must be at least {MIN_INFERENCE_SEEDS}"
        )
    required_analysis = ("bootstrap_seed", "bootstrap_resamples", "confidence")
    for key in required_analysis:
        if key not in analysis:
            raise ExperimentError(f"analysis settings are missing {key!r}")

    pilot_seeds = [int(seed) for seed in pilot_report.get("seeds") or []]
    if len(pilot_seeds) < MIN_INFERENCE_SEEDS:
        raise PlanningError(
            f"pilot report has {len(pilot_seeds)} seeds; at least {MIN_INFERENCE_SEEDS} are needed"
        )
    expected_pilot_seeds = generate_seeds(procedure, len(pilot_seeds))
    if pilot_seeds != expected_pilot_seeds:
        raise PlanningError(
            "pilot seeds are not the recorded seed procedure's initial prefix; the "
            "pilot sample and the frozen seed procedure disagree"
        )
    pilot_identity = (pilot_report.get("adapter_identity") or {}).get("fingerprint")
    current_identity = (adapter_identity or {}).get("fingerprint")
    if not pilot_identity:
        raise PlanningError(
            "pilot report has no adapter identity; re-run the pilot with this script so "
            "the plan can be tied to the measured implementation"
        )
    if pilot_identity != current_identity:
        raise PlanningError(
            "adapter identity changed between pilot and plan (driver/worker/analysis "
            "source/ninone heuristic source); re-run the pilot with the same "
            f"implementation. pilot {pilot_identity!r}, current {current_identity!r}"
        )
    block_sd = float(pilot_report.get("block_sd") or 0.0)
    seconds_per_seed = float(pilot_report.get("mean_seconds_per_seed") or 0.0)
    if block_sd < 0:
        raise PlanningError("pilot block SD is negative; the pilot report is corrupt")

    limitations: list[str] = []
    n_precision: int | None = None
    n_budget: int | None = None
    if target_precision is not None:
        if block_sd > 0:
            n_precision = math.ceil((Z_95 * block_sd / target_precision) ** 2)
        else:
            n_precision = MIN_INFERENCE_SEEDS
            limitations.append(
                "pilot block SD is zero, so the precision target cannot set a sample size "
                f"and zero uncertainty is not claimed; using the minimum of {MIN_INFERENCE_SEEDS} seeds"
            )
    if runtime_budget is not None:
        if seconds_per_seed <= 0:
            raise PlanningError(
                "pilot runtime per seed must be positive to translate a runtime budget"
            )
        n_budget = math.floor(runtime_budget / seconds_per_seed)
        if n_budget < MIN_INFERENCE_SEEDS:
            raise PlanningError(
                f"runtime budget {runtime_budget:g}s allows {n_budget} seed(s) at "
                f"{seconds_per_seed:.4g}s/seed; at least {MIN_INFERENCE_SEEDS} seeds are "
                "required for inference"
            )

    candidates = [value for value in (n_precision, n_budget) if value is not None]
    n_main = min(candidates)
    if max_main_seeds is not None and n_main > max_main_seeds:
        limitations.append(
            f"main sample capped at --max-main-seeds {max_main_seeds} "
            f"(uncapped size was {n_main})"
        )
        n_main = max_main_seeds
    n_main = max(MIN_INFERENCE_SEEDS, n_main)

    predicted_half_width = Z_95 * block_sd / math.sqrt(n_main) if n_main else None
    precision_met: bool | None = None
    if target_precision is not None:
        if block_sd > 0 and predicted_half_width is not None:
            precision_met = predicted_half_width <= target_precision
            if not precision_met:
                limitations.append(
                    f"planned sample of {n_main} seeds predicts a 95% half-width of "
                    f"{predicted_half_width:.4f} (normal approximation from the pilot block SD), "
                    f"above the target {target_precision:g}; the runtime budget binds"
                )
        else:
            # A zero pilot SD is a small-sample accident, not evidence of zero
            # uncertainty; leave the target unclaimed.
            precision_met = None

    main_seeds = plan_main_seeds(procedure, len(pilot_seeds), n_main)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "mode": mode,
        "decision_limit": int(decision_limit),
        "pilot": {
            "seeds": pilot_seeds,
            "count": len(pilot_seeds),
            "mode": pilot_mode,
            "decision_limit": int(pilot_limit),
            "block_sd": block_sd,
            "block_mean": float(pilot_report.get("block_mean") or 0.0),
            "mean_seconds_per_seed": seconds_per_seed,
            "elapsed_seconds": float(pilot_report.get("elapsed_seconds") or 0.0),
            "results_sha256": pilot_report.get("results_sha256"),
        },
        "target": {
            "target_precision": target_precision,
            "runtime_budget_seconds": runtime_budget,
            "confidence": CONFIDENCE,
            "z": Z_95,
            "precision_formula": "n = ceil((z * pilot_block_sd / target_precision) ** 2)",
            "runtime_formula": "n = floor(runtime_budget / pilot_mean_seconds_per_seed)",
            "max_main_seeds": max_main_seeds,
            "precision_note": (
                "Target precision is a planning goal derived from pilot variability via a "
                "normal approximation; the bootstrap interval measured on the main sample "
                "is reported independently and is not guaranteed to match the target."
            ),
        },
        "sample": {
            "n_main": n_main,
            "main_seeds": main_seeds,
            "min_inference_seeds": MIN_INFERENCE_SEEDS,
            "predicted_half_width": predicted_half_width,
            "predicted_half_width_is_estimate": True,
            "prediction_method": (
                "z * pilot_block_sd / sqrt(n_main) (normal approximation); predicted "
                "from pilot variability, not a guarantee for the reported bootstrap interval"
            ),
            "precision_target_met": precision_met,
            "estimated_main_seconds": n_main * seconds_per_seed,
        },
        "seed_procedure": procedure,
        "analysis": {
            "bootstrap_seed": int(analysis["bootstrap_seed"]),
            "bootstrap_resamples": int(analysis["bootstrap_resamples"]),
            "confidence": float(analysis["confidence"]),
            "interval": "percentile bootstrap over resampled whole seed blocks",
            "resample_unit": "seed block",
            "prng": SEED_PROCEDURE_GENERATOR,
            "estimator": (
                "Delta = mean over seed blocks s of D[s]; "
                "D[s] = mean over the six placements c of (mean A score - mean B score)"
            ),
        },
        "adapter_identity": adapter_identity,
        "rules": {
            "early_stopping": False,
            "pilot_excluded_from_main": True,
            "sample_frozen_before_main": True,
            "failures_invalidate_run": True,
        },
        "limitations": limitations,
    }


def write_plan(work_dir: Path, plan: dict[str, Any]) -> Path:
    """Write a frozen plan without overwriting earlier decisions.

    A frozen plan (or any work directory that already contains a main attempt)
    must not be silently replaced: the main sample was already decided, so a
    new decision requires a new work directory.
    """
    work_dir = Path(work_dir)
    path = work_dir / "plan.json"
    if path.is_file():
        existing = None
        try:
            existing = read_json(path)
        except ExperimentError:
            existing = None
        frozen = isinstance(existing, dict) and existing.get("status") == "frozen"
        if frozen:
            raise ExperimentError(
                f"a frozen plan already exists at {path}; refusing to overwrite the "
                "decided main sample (use a new work directory)"
            )
    for attempt in (work_dir / "main" / "results.jsonl", work_dir / "main" / "failures.json"):
        if attempt.exists():
            raise ExperimentError(
                f"a main attempt already exists at {attempt}; refusing to overwrite the "
                "plan of a run that has already started (use a new work directory)"
            )
    write_json(path, plan)
    return path


def load_plan(work_dir: Path) -> dict[str, Any]:
    plan = read_json(Path(work_dir) / "plan.json")
    if not isinstance(plan, dict):
        raise ExperimentError("plan.json is not a JSON object")
    if plan.get("status") != "frozen":
        raise ExperimentError(
            f"plan.json is not frozen (status {plan.get('status')!r}); refusing to run or summarize"
        )
    return plan


def resolve_plan_seeds(plan: dict[str, Any]) -> list[int]:
    """Validate and return the frozen main seeds of a plan."""
    sample = plan.get("sample") or {}
    seeds = sample.get("main_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ExperimentError("plan has no frozen main seed list")
    if sample.get("n_main") != len(seeds):
        raise ExperimentError("plan n_main does not match its main seed list")
    if len(seeds) < MIN_INFERENCE_SEEDS:
        raise ExperimentError(
            f"plan has {len(seeds)} main seeds; at least {MIN_INFERENCE_SEEDS} are required"
        )
    resolved: list[int] = []
    for seed in seeds:
        if not is_int(seed) or not 0 <= seed <= U64_MAX:
            raise ExperimentError(f"plan contains an invalid u64 seed: {seed!r}")
        resolved.append(int(seed))
    if len(set(resolved)) != len(resolved):
        raise ExperimentError("plan main seeds contain duplicates")
    pilot = {int(seed) for seed in (plan.get("pilot") or {}).get("seeds") or []}
    overlap = sorted(set(resolved) & pilot)
    if overlap:
        raise ExperimentError(f"plan main seeds overlap pilot seeds: {overlap[:5]}")
    return resolved


def verify_plan_for_main(
    plan: dict[str, Any],
    *,
    mode: str,
    decision_limit: int,
    analysis: dict[str, Any],
    adapter_identity: dict[str, Any],
) -> list[int]:
    """Check that the frozen plan matches the current run configuration."""
    if plan.get("mode") != mode:
        raise ExperimentError(
            f"plan mode {plan.get('mode')!r} does not match requested mode {mode!r}"
        )
    if int(plan.get("decision_limit", -1)) != int(decision_limit):
        raise ExperimentError(
            f"plan decision limit {plan.get('decision_limit')!r} does not match requested {decision_limit}"
        )
    planned_analysis = plan.get("analysis") or {}
    for key in ("bootstrap_seed", "bootstrap_resamples", "confidence"):
        if planned_analysis.get(key) != analysis[key]:
            raise ExperimentError(
                f"analysis setting {key!r} changed after the plan was frozen "
                f"({planned_analysis.get(key)!r} -> {analysis[key]!r})"
            )
    planned_fingerprint = (plan.get("adapter_identity") or {}).get("fingerprint")
    current_fingerprint = (adapter_identity or {}).get("fingerprint")
    if planned_fingerprint != current_fingerprint:
        raise ExperimentError(
            "adapter identity (driver/worker/analysis source/ninone heuristic source) "
            "changed after the plan was frozen; rebuild the plan with the same "
            f"components. planned {planned_fingerprint!r}, current {current_fingerprint!r}"
        )
    return resolve_plan_seeds(plan)


# ---------------------------------------------------------------------------
# Adapter identity, repositories, environment
# ---------------------------------------------------------------------------


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    info: dict[str, Any] = {"path": str(path), "present": path.exists()}
    if path.is_file():
        info["sha256"] = sha256_file(path)
        info["size"] = path.stat().st_size
    else:
        info["sha256"] = None
        info["size"] = None
    return info


def capture_rust_sources(repo_root: Path) -> dict[str, Any]:
    """Hash the Rust build inputs that determine the driver's behavior.

    Only manifest and source files are read. Build outputs (``target/``) and
    generated experiment outputs are never scanned, so the provenance record
    cannot recursively contain itself.
    """
    repo_root = Path(repo_root)
    files: list[dict[str, Any]] = []
    epochs: list[float] = []
    for relative in ("Cargo.toml", "Cargo.lock"):
        path = repo_root / relative
        if not path.is_file():
            continue
        files.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
        epochs.append(path.stat().st_mtime)
    for directory in ("src", "examples"):
        base = repo_root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            if not path.is_file():
                continue
            relative = relative_display(path, repo_root)
            if _is_generated_path(relative):
                continue
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
            epochs.append(path.stat().st_mtime)
    files.sort(key=lambda item: item["path"])
    cargo_lock = next((item for item in files if item["path"] == "Cargo.lock"), None)
    cargo_toml = next((item for item in files if item["path"] == "Cargo.toml"), None)
    sources = [item for item in files if item["path"].endswith(".rs")]
    newest = max(epochs) if epochs else None
    return {
        "root": str(repo_root),
        "files": files,
        "file_count": len(files),
        "rust_source_count": len(sources),
        "rust_source_aggregate_sha256": sha256_json(
            [{"path": item["path"], "sha256": item["sha256"]} for item in sources]
        ),
        "cargo_toml_sha256": (cargo_toml or {}).get("sha256"),
        "cargo_lock_sha256": (cargo_lock or {}).get("sha256"),
        "newest_source_mtime_utc": _mtime_utc_from_epoch(newest),
        "newest_source_epoch": newest,
    }


NINONE_SOURCE_PACKAGES = ("ninone",)
NINONE_EXCLUDED_DIRS = (
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "runtime",
    ".local",
    "node_modules",
)


def capture_ninone_sources(ninone_root: Path) -> dict[str, Any]:
    """Hash the imported heuristic's Python source tree.

    A tracked git diff misses untracked or ignored files, so the imported
    package is hashed directly: any local edit to the heuristic (including an
    untracked module that shadows an import) changes the adapter identity.
    """
    ninone_root = Path(ninone_root)
    files: list[dict[str, Any]] = []
    for package in NINONE_SOURCE_PACKAGES:
        base = ninone_root / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if not path.is_file():
                continue
            relative = relative_display(path, ninone_root)
            if _is_generated_path(relative):
                continue
            if set(relative.split("/")) & set(NINONE_EXCLUDED_DIRS):
                continue
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    files.sort(key=lambda item: item["path"])
    return {
        "root": str(ninone_root),
        "package_dirs": list(NINONE_SOURCE_PACKAGES),
        "files": files,
        "file_count": len(files),
        "present": bool(files),
        "aggregate_sha256": sha256_json(
            [{"path": item["path"], "sha256": item["sha256"]} for item in files]
        ),
    }


def driver_freshness(driver: Path, rust_sources: dict[str, Any]) -> dict[str, Any]:
    """Build-order check tying the executed driver to the recorded sources."""
    driver = Path(driver)
    newest = rust_sources.get("newest_source_epoch")
    info: dict[str, Any] = {
        "checked": False,
        "driver_mtime_utc": None,
        "newest_source_mtime_utc": rust_sources.get("newest_source_mtime_utc"),
        "tolerance_seconds": 1.0,
    }
    if driver.suffix.lower() == ".py":
        info["reason"] = "python-script driver; build-order check does not apply"
        return info
    if not driver.is_file():
        info["reason"] = "driver executable is missing"
        return info
    if newest is None:
        info["reason"] = "no Rust build inputs were found next to the driver"
        return info
    info["checked"] = True
    info["driver_mtime_utc"] = _mtime_utc(driver)
    info["driver_newer_than_sources"] = driver.stat().st_mtime + info["tolerance_seconds"] >= float(
        newest
    )
    info["stale"] = not info["driver_newer_than_sources"]
    info["reason"] = (
        "driver is at least as new as the recorded Rust sources"
        if not info["stale"]
        else "driver is older than the recorded Rust sources; rebuild it"
    )
    return info


PYTHON_PROBE = (
    "import platform, sys; "
    "print(platform.python_version()); "
    "print(platform.python_implementation()); "
    "print(sys.executable)"
)


def resolve_python_executable(python_exe: Path) -> Path | None:
    """Resolve the worker interpreter (``--python`` may be a bare command)."""
    candidate = Path(python_exe)
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(str(python_exe))
    return Path(found).resolve() if found else None


def probe_python_interpreter(resolved: Path | None) -> dict[str, Any]:
    """Query the actual worker interpreter, not the analysis interpreter."""
    info: dict[str, Any] = {
        "version": None,
        "implementation": None,
        "executable": None,
        "error": None,
    }
    if resolved is None:
        info["error"] = "interpreter executable could not be resolved"
        return info
    try:
        completed = subprocess.run(
            [str(resolved), "-c", PYTHON_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) < 3:
        info["error"] = (
            f"probe exited with {completed.returncode}: {_tail(completed.stderr or '')}"
        )
        return info
    info["version"] = lines[0]
    info["implementation"] = lines[1]
    info["executable"] = lines[2]
    return info


def adapter_identity(
    *,
    driver: Path,
    worker: Path,
    python_exe: Path,
    repo_root: Path,
    ninone_capture: dict[str, Any],
    script_path: Path | None = None,
    rust_sources: dict[str, Any] | None = None,
    ninone_sources: dict[str, Any] | None = None,
    allow_stale_driver: bool = False,
) -> dict[str, Any]:
    """Behavioral identity of everything between seeds and terminal scores.

    Records the executed driver binary, its Rust build inputs, the worker and
    analysis sources, the actual Python interpreter, and the imported ninone
    revision, so recorded sources can be tied to the binary that produced the
    results. The ``fingerprint`` is what a frozen plan pins.
    """
    script = Path(script_path) if script_path is not None else SCRIPT_PATH
    driver_info = file_identity(driver)
    if driver_info["present"]:
        driver_info["kind"] = (
            "python-script" if Path(driver).suffix.lower() == ".py" else "binary"
        )
        driver_info["resolved_path"] = str(Path(driver).resolve())
        driver_info["mtime_utc"] = _mtime_utc(Path(driver))
    else:
        driver_info["kind"] = None

    rust = rust_sources if rust_sources is not None else capture_rust_sources(Path(repo_root))
    if ninone_sources is not None:
        sources = ninone_sources
    elif ninone_capture.get("root"):
        sources = capture_ninone_sources(Path(str(ninone_capture["root"])))
    else:
        sources = {
            "files": [],
            "file_count": 0,
            "present": False,
            "aggregate_sha256": None,
        }
    freshness = driver_freshness(Path(driver), rust)
    if freshness.get("checked") and freshness.get("stale"):
        if not allow_stale_driver:
            raise ExperimentError(
                f"driver executable {driver} is older than the recorded Rust build inputs "
                f"(newest source mtime {freshness['newest_source_mtime_utc']}); rebuild it "
                "(cargo build --example policy_ab) or pass --allow-stale-driver to "
                "record the mismatch explicitly"
            )
        freshness["waived"] = True

    source_path = Path(repo_root) / "examples" / "policy_ab.rs"
    source_info = file_identity(source_path) if source_path.is_file() else None
    worker_info = file_identity(worker)
    script_info = file_identity(script)

    python_path = Path(python_exe)
    resolved_python = resolve_python_executable(python_path)
    if resolved_python is None:
        raise ExperimentError(
            f"worker python interpreter not found: {python_exe}; pass --python with a "
            "resolvable executable path or command name"
        )
    probe = probe_python_interpreter(resolved_python)
    python_info = {
        "requested": str(python_exe),
        "resolved_path": str(resolved_python),
        "present": resolved_python.is_file(),
        "sha256": sha256_file(resolved_python),
        "version": probe["version"],
        "implementation": probe["implementation"],
        "probe_executable": probe["executable"],
        "probe_error": probe["error"],
        "version_source": "worker interpreter probe" if probe["version"] else "unavailable",
        "analysis_python_version": platform.python_version(),
        "analysis_python_executable": sys.executable,
    }

    identity: dict[str, Any] = {
        "driver": driver_info,
        "driver_source": source_info,
        "driver_freshness": freshness,
        "rust_sources": rust,
        "worker": worker_info,
        "script": script_info,
        "python": python_info,
        "ninone": {
            "root": str(ninone_capture.get("root")),
            "is_git_repo": bool(ninone_capture.get("is_git_repo")),
            "head": ninone_capture.get("head"),
            "diff_sha256": ninone_capture.get("diff_sha256"),
            "dirty": ninone_capture.get("dirty"),
            "sources": sources,
        },
    }
    identity["fingerprint_components"] = {
        "driver_sha256": driver_info.get("sha256"),
        "driver_source_sha256": (source_info or {}).get("sha256"),
        "rust_source_aggregate_sha256": rust.get("rust_source_aggregate_sha256"),
        "cargo_lock_sha256": rust.get("cargo_lock_sha256"),
        "worker_sha256": worker_info.get("sha256"),
        "script_sha256": script_info.get("sha256"),
        "python_resolved_path": python_info["resolved_path"],
        "python_sha256": python_info["sha256"],
        "python_version": python_info["version"],
        "python_implementation": python_info["implementation"],
        "ninone_head": identity["ninone"]["head"],
        "ninone_diff_sha256": identity["ninone"]["diff_sha256"],
        "ninone_source_aggregate_sha256": sources.get("aggregate_sha256"),
    }
    identity["fingerprint"] = sha256_json(identity["fingerprint_components"])
    return identity


def _git_available() -> bool:
    try:
        completed = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_text(root: Path, *args: str) -> str | None:
    completed = _git(root, *args)
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _git_bytes(root: Path, *args: str) -> bytes | None:
    completed = _git(root, *args)
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout


@dataclass(frozen=True)
class StatusEntry:
    status: str
    path: str


def _parse_status_z(raw: bytes) -> list[StatusEntry]:
    entries: list[StatusEntry] = []
    fields = raw.split(b"\0")
    index = 0
    while index < len(fields):
        field = fields[index]
        if not field or len(field) < 4:
            index += 1
            continue
        status = field[:2].decode("utf-8", errors="replace")
        path = os.fsdecode(field[3:])
        entries.append(StatusEntry(status=status, path=path))
        # Rename/copy entries carry the original path as an extra record.
        index += 2 if status[:1] in ("R", "C") else 1
    return entries


def _safe_relative_path(relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    if any(part == ".." for part in candidate.parts):
        return None
    return candidate


#: Paths that hold build outputs or generated experiment artifacts. They are
#: excluded from provenance snapshots so the manifest never recursively contains
#: itself or a build tree.
GENERATED_PATH_PREFIXES = (
    "target/",
    "experiments/policy_ab/runs/",
)
GENERATED_PATH_PARTS = ("__pycache__", ".git")
GENERATED_PATH_SUFFIXES = (".pyc", ".pyo")


def _is_generated_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if any(normalized.startswith(prefix) for prefix in GENERATED_PATH_PREFIXES):
        return True
    parts = normalized.split("/")
    if any(part in GENERATED_PATH_PARTS for part in parts):
        return True
    return normalized.endswith(GENERATED_PATH_SUFFIXES)


def _is_text_bytes(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def capture_repository(
    root: Path,
    *,
    name: str,
    reference_dir: Path | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Capture revision, local changes, and (optionally) untracked snapshots."""
    root = Path(root)
    reference_dir = Path(reference_dir) if reference_dir is not None else root
    info: dict[str, Any] = {
        "name": name,
        "root": str(root),
        "is_git_repo": False,
        "git_available": _git_available(),
    }
    if not info["git_available"]:
        info["note"] = "git executable unavailable; revision evidence could not be captured"
        return info

    head = _git_text(root, "rev-parse", "HEAD")
    if head is None:
        info["note"] = "not a git repository (or no commits); revision evidence unavailable"
        return info
    info["is_git_repo"] = True
    info["head"] = head
    info["head_subject"] = _git_text(root, "log", "-1", "--pretty=%s")
    info["branch"] = _git_text(root, "rev-parse", "--abbrev-ref", "HEAD")

    status_raw = _git_bytes(root, "status", "--porcelain", "-uall", "-z") or b""
    all_entries = _parse_status_z(status_raw)
    entries = [entry for entry in all_entries if not _is_generated_path(entry.path)]
    info["omitted_generated_entries"] = len(all_entries) - len(entries)
    info["status_entries"] = [
        {"status": entry.status, "path": entry.path} for entry in entries
    ]
    info["status_lines"] = [f"{entry.status} {entry.path}" for entry in entries]
    info["dirty"] = bool(entries)
    info["tracked_changes"] = [
        {"status": entry.status, "path": entry.path}
        for entry in entries
        if not entry.status.startswith("??")
    ]

    diff = _git_bytes(root, "diff", "HEAD", "--binary", "--no-color") or b""
    info["diff_sha256"] = sha256_bytes(diff)
    info["diff_bytes"] = len(diff)

    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if diff:
            patch_path = snapshot_dir / "changes.patch"
            patch_path.write_bytes(diff)
            info["diff_patch"] = relative_display(patch_path, reference_dir)

        untracked_entries = [entry for entry in entries if entry.status.startswith("??")]
        info["untracked"] = []
        snapshot_total = 0
        for entry in untracked_entries:
            record: dict[str, Any] = {"path": entry.path}
            relative = _safe_relative_path(entry.path)
            if relative is None or _is_generated_path(entry.path):
                record["present"] = False
                record["note"] = "generated or out-of-tree path; not captured as provenance"
                info["untracked"].append(record)
                continue
            full = root / entry.path
            if relative is None or not full.is_file():
                record["present"] = False
                record["note"] = "not a regular file inside the repository"
                info["untracked"].append(record)
                continue
            record["present"] = True
            data = full.read_bytes()
            record["sha256"] = sha256_bytes(data)
            record["size"] = len(data)
            if not _is_text_bytes(data):
                record["snapshot"] = None
                record["snapshot_skipped"] = "binary content"
            elif len(data) > MAX_SNAPSHOT_FILE_BYTES:
                record["snapshot"] = None
                record["snapshot_skipped"] = (
                    f"file larger than {MAX_SNAPSHOT_FILE_BYTES} bytes"
                )
            elif snapshot_total + len(data) > MAX_SNAPSHOT_TOTAL_BYTES:
                record["snapshot"] = None
                record["snapshot_skipped"] = (
                    f"snapshot budget of {MAX_SNAPSHOT_TOTAL_BYTES} bytes exhausted"
                )
            else:
                destination = snapshot_dir / "untracked" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                snapshot_total += len(data)
                record["snapshot"] = relative_display(destination, reference_dir)
            info["untracked"].append(record)
        info["untracked_snapshot_bytes"] = snapshot_total
    return info


#: Identity-relevant source groups are copied as actual bytes before a run.
#: Their content must remain recoverable even when a file is ignored by Git or
#: lives outside both repositories.
def identity_source_files(identity: dict[str, Any]) -> list[tuple[str, Path, str]]:
    """Return ``(label, path, relative_name)`` for identity-relevant sources."""
    entries: list[tuple[str, Path, str]] = []

    def add_single(label: str, record: dict[str, Any] | None) -> None:
        if not record or not record.get("path"):
            return
        path = Path(str(record["path"]))
        entries.append((label, path, path.name))

    def add_group(
        label: str, root: str | None, records: Iterable[dict[str, Any]]
    ) -> None:
        base = Path(root) if root else None
        for record in records:
            relative = record.get("path")
            if not relative:
                continue
            path = Path(str(relative))
            if base is not None and not path.is_absolute():
                path = base / path
            entries.append((label, path, str(relative)))

    add_single("driver_source", identity.get("driver_source"))
    add_single("worker", identity.get("worker"))
    add_single("analysis_script", identity.get("script"))
    rust = identity.get("rust_sources") or {}
    add_group("rust_sources", rust.get("root"), rust.get("files") or [])
    ninone_sources = (identity.get("ninone") or {}).get("sources") or {}
    add_group(
        "ninone_sources",
        ninone_sources.get("root"),
        ninone_sources.get("files") or [],
    )
    return entries


def capture_source_contents(
    identity: dict[str, Any],
    *,
    snapshot_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    """Copy identity-relevant source bytes so a run's inputs are recoverable.

    Only hashes cannot restore an ignored or out-of-tree worker/heuristic source
    that changed or disappeared later. Missing or oversized content is recorded
    explicitly as missing evidence rather than silently reduced to a hash.
    """
    snapshot_dir = Path(snapshot_dir)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[str] = []
    copied = 0
    total = 0
    seen: set[str] = set()
    for label, path, relative in identity_source_files(identity):
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "present": path.is_file(),
            "snapshot": None,
        }
        if not record["present"]:
            record["note"] = "missing evidence: source file is absent at capture time"
            missing.append(str(path))
            records.append(record)
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            record["present"] = False
            record["note"] = f"missing evidence: source file is unreadable ({exc})"
            missing.append(str(path))
            records.append(record)
            continue
        record["sha256"] = sha256_bytes(data)
        record["size"] = len(data)
        if total + len(data) > MAX_SOURCE_SNAPSHOT_TOTAL_BYTES:
            record["note"] = (
                "missing evidence: source snapshot budget of "
                f"{MAX_SOURCE_SNAPSHOT_TOTAL_BYTES} bytes exhausted"
            )
            missing.append(str(path))
            records.append(record)
            continue
        safe_relative = _safe_relative_path(relative.replace("\\", "/"))
        if safe_relative is None:
            safe_relative = Path(path.name)
        destination = snapshot_dir / label / safe_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        total += len(data)
        copied += 1
        record["snapshot"] = relative_display(destination, reference_dir)
        record["sha256_matches_identity"] = record["sha256"] == _identity_source_sha(
            identity, label, relative
        )
        if record["sha256_matches_identity"] is False:
            mismatched.append(str(path))
        records.append(record)
    records.sort(key=lambda item: (item["label"], item.get("snapshot") or item["path"]))
    return {
        "snapshot_dir": relative_display(snapshot_dir, reference_dir),
        "files": records,
        "file_count": len(records),
        "copied_count": copied,
        "total_bytes": total,
        "missing_evidence": missing,
        "changed_before_capture": mismatched,
        "aggregate_sha256": sha256_json(
            [
                {
                    "label": item["label"],
                    "path": item["path"],
                    "sha256": item.get("sha256"),
                }
                for item in records
            ]
        ),
    }


def _identity_source_sha(
    identity: dict[str, Any], label: str, relative: str
) -> str | None:
    if label == "driver_source":
        return (identity.get("driver_source") or {}).get("sha256")
    if label == "worker":
        return (identity.get("worker") or {}).get("sha256")
    if label == "analysis_script":
        return (identity.get("script") or {}).get("sha256")
    if label == "rust_sources":
        for item in (identity.get("rust_sources") or {}).get("files") or []:
            if item.get("path") == relative:
                return item.get("sha256")
        return None
    if label == "ninone_sources":
        sources = (identity.get("ninone") or {}).get("sources") or {}
        for item in sources.get("files") or []:
            if item.get("path") == relative:
                return item.get("sha256")
        return None
    return None


def identity_evidence(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": identity.get("fingerprint"),
        "components": identity.get("fingerprint_components"),
    }


def capture_run_evidence(
    *,
    work_dir: Path,
    stage: str,
    repo_root: Path,
    ninone_root: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Capture repository revisions/diffs and source bytes at one run boundary."""
    work_dir = Path(work_dir)
    stage_dir = work_dir / "evidence" / stage
    repositories: dict[str, Any] = {}
    for root, name in (
        (Path(repo_root), DEFAULT_REPO_ROOT.name),
        (Path(ninone_root), Path(ninone_root).name),
    ):
        repositories[name] = capture_repository(
            root,
            name=name,
            reference_dir=work_dir,
            snapshot_dir=stage_dir / "repos" / name,
        )
    sources = capture_source_contents(
        identity,
        snapshot_dir=stage_dir / "sources",
        reference_dir=work_dir,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "captured_at_utc": utc_now(),
        "identity": identity_evidence(identity),
        "repositories": repositories,
        "source_snapshots": sources,
    }


def compare_run_evidence(
    pre_evidence: dict[str, Any] | None,
    post_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare pre-run and post-run identity/repository evidence."""
    if not pre_evidence or not post_evidence:
        return {"status": "not-checked", "identity_changes": [], "repository_changes": []}
    pre_identity = pre_evidence.get("identity") or {}
    post_identity = post_evidence.get("identity") or {}
    pre_components = pre_identity.get("components") or {}
    post_components = post_identity.get("components") or {}
    identity_changes = [
        {
            "component": key,
            "pre": pre_components.get(key),
            "post": post_components.get(key),
        }
        for key in sorted(set(pre_components) | set(post_components))
        if pre_components.get(key) != post_components.get(key)
    ]
    pre_repositories = pre_evidence.get("repositories") or {}
    post_repositories = post_evidence.get("repositories") or {}
    repository_changes: list[dict[str, Any]] = []
    for name in sorted(set(pre_repositories) | set(post_repositories)):
        before = pre_repositories.get(name) or {}
        after = post_repositories.get(name) or {}
        for field_name in ("head", "diff_sha256"):
            if before.get(field_name) != after.get(field_name):
                repository_changes.append(
                    {
                        "repository": name,
                        "field": field_name,
                        "pre": before.get(field_name),
                        "post": after.get(field_name),
                    }
                )
    drifted = bool(
        identity_changes
        or repository_changes
        or pre_identity.get("fingerprint") != post_identity.get("fingerprint")
    )
    return {
        "status": "drifted" if drifted else "stable",
        "pre_fingerprint": pre_identity.get("fingerprint"),
        "post_fingerprint": post_identity.get("fingerprint"),
        "identity_changes": identity_changes,
        "repository_changes": repository_changes,
    }


def capture_post_evidence(
    provider: Callable[[], dict[str, Any]],
    pre_evidence: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Capture post-run evidence, treating re-identity failures as drift.

    Re-deriving identity after a run can itself fail (for example when a source
    changed so the driver now looks stale). That still means the executed
    implementation cannot be tied to the pre-run capture, so it invalidates the
    run instead of escaping as an undiagnosed error.
    """
    try:
        post_evidence = provider()
    except ExperimentError as exc:
        return None, {
            "status": "drifted",
            "pre_fingerprint": (pre_evidence.get("identity") or {}).get("fingerprint"),
            "post_fingerprint": None,
            "identity_changes": [],
            "repository_changes": [],
            "error": str(exc),
        }
    return post_evidence, compare_run_evidence(pre_evidence, post_evidence)


def _tool_version(command: str) -> str | None:
    try:
        completed = subprocess.run(
            [command, "-V"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or completed.stderr).strip() or None


def environment_info() -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "rustc": _tool_version("rustc"),
        "cargo": _tool_version("cargo"),
    }


def build_manifest(
    *,
    work_dir: Path,
    repo_root: Path,
    ninone_root: Path,
    plan: dict[str, Any],
    adapter: dict[str, Any],
    invocation: dict[str, Any],
    timing: dict[str, Any],
    deterministic_replay: dict[str, Any],
    argv: Sequence[str],
    artifacts: dict[str, Any] | None = None,
    captures: dict[str, Any] | None = None,
    identity_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    if captures is not None:
        # Reuse pre-run captures: re-reading repositories after the run could
        # record sources that were not the ones actually executed.
        repositories = captures.get("repositories") or {}
        capture_stage = captures.get("stage")
        capture_time = captures.get("captured_at_utc")
        source_snapshots = captures.get("source_snapshots")
    else:
        repos_dir = work_dir / "repos"
        si_chuan = capture_repository(
            repo_root,
            name=DEFAULT_REPO_ROOT.name,
            reference_dir=work_dir,
            snapshot_dir=repos_dir / DEFAULT_REPO_ROOT.name,
        )
        ninone = capture_repository(
            ninone_root,
            name=Path(ninone_root).name,
            reference_dir=work_dir,
            snapshot_dir=repos_dir / Path(ninone_root).name,
        )
        repositories = {si_chuan["name"]: si_chuan, ninone["name"]: ninone}
        capture_stage = "captured-at-manifest-build"
        capture_time = utc_now()
        source_snapshots = None
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "policy": (
            "Both repository revisions and local changes are captured, including a "
            "binary-capable diff patch and hashes/snapshots of untracked text files. "
            "Identity-relevant source bytes are copied before the run so ignored or "
            "out-of-tree sources stay recoverable."
        ),
        "repositories": repositories,
        "evidence": {
            "repository_capture_stage": capture_stage,
            "repository_capture_time_utc": capture_time,
            "source_snapshots": source_snapshots,
            "identity_check": identity_check or {"status": "not-checked"},
        },
        "adapter": adapter,
        "source_binding": {
            "driver_executable": adapter.get("driver", {}).get("resolved_path"),
            "driver_executable_sha256": adapter.get("driver", {}).get("sha256"),
            "driver_source_sha256": (adapter.get("driver_source") or {}).get("sha256"),
            "rust_source_aggregate_sha256": adapter.get("rust_sources", {}).get(
                "rust_source_aggregate_sha256"
            ),
            "cargo_toml_sha256": adapter.get("rust_sources", {}).get("cargo_toml_sha256"),
            "cargo_lock_sha256": adapter.get("rust_sources", {}).get("cargo_lock_sha256"),
            "worker_sha256": adapter.get("worker", {}).get("sha256"),
            "analysis_script_sha256": adapter.get("script", {}).get("sha256"),
            "python_executable": adapter.get("python", {}).get("resolved_path"),
            "worker_python_version": adapter.get("python", {}).get("version"),
            "worker_python_implementation": adapter.get("python", {}).get(
                "implementation"
            ),
            "ninone_head": adapter.get("ninone", {}).get("head"),
            "ninone_source_aggregate_sha256": (adapter.get("ninone") or {})
            .get("sources", {})
            .get("aggregate_sha256"),
            "ninone_source_file_count": (adapter.get("ninone") or {})
            .get("sources", {})
            .get("file_count"),
            "identity_fingerprint": adapter.get("fingerprint"),
            "driver_freshness": adapter.get("driver_freshness"),
            "note": (
                "Recorded source hashes plus the executed driver hash. The freshness "
                "check is build-order evidence, not a proof that the binary was built "
                "from exactly these sources."
            ),
        },
        "experiment": {
            "mode": plan.get("mode"),
            "decision_limit": plan.get("decision_limit"),
            "seed_procedure": plan.get("seed_procedure"),
            "pilot": {
                "count": (plan.get("pilot") or {}).get("count"),
                "seeds": (plan.get("pilot") or {}).get("seeds"),
                "block_sd": (plan.get("pilot") or {}).get("block_sd"),
                "mean_seconds_per_seed": (plan.get("pilot") or {}).get(
                    "mean_seconds_per_seed"
                ),
            },
            "main": {
                "count": (plan.get("sample") or {}).get("n_main"),
                "seeds": (plan.get("sample") or {}).get("main_seeds"),
                "predicted_half_width": (plan.get("sample") or {}).get(
                    "predicted_half_width"
                ),
                "precision_target_met": (plan.get("sample") or {}).get(
                    "precision_target_met"
                ),
            },
            "target": plan.get("target"),
            "limitations": plan.get("limitations"),
        },
        "analysis": plan.get("analysis"),
        "plan": {
            "path": "plan.json",
            "sha256": sha256_file(work_dir / "plan.json"),
        },
        "artifacts": artifacts or {},
        "invocation": invocation,
        "timing": {
            **timing,
            "note": "Timing is diagnostic and not part of deterministic result identity.",
        },
        "deterministic_replay": deterministic_replay,
        "environment": environment_info(),
        "argv": list(argv),
        "cwd": os.getcwd(),
    }


# ---------------------------------------------------------------------------
# Summary stage
# ---------------------------------------------------------------------------


def build_summary(
    *,
    plan: dict[str, Any],
    evaluation: RunEvaluation,
    timing: dict[str, Any] | None,
    deterministic_replay: dict[str, Any] | None,
    integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not evaluation.ok:
        raise SummaryError(
            f"run has {len(evaluation.problems)} problem(s); refusing to publish a normal summary"
        )
    seeds = resolve_plan_seeds(plan)
    mode = plan.get("mode")
    if mode not in MODES:
        raise SummaryError(f"plan has an unsupported mode: {mode!r}")
    analysis = plan.get("analysis") or {}
    stats = compute_contrast_stats(evaluation.by_seed, seeds)
    ci_low, ci_high = bootstrap_delta_ci(
        stats.block_contrasts,
        resamples=int(analysis.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
        seed=int(analysis.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
        confidence=float(analysis.get("confidence", CONFIDENCE)),
    )

    zero = zero_block_check(stats)
    symmetric_required = mode in SYMMETRIC_MODES
    if symmetric_required and not zero["all_zero"]:
        nonzero = zero["nonzero_blocks"][:5]
        raise SummaryError(
            f"symmetric mode {mode!r} requires zero block contrasts, but "
            f"{len(zero['nonzero_blocks'])} block(s) are non-zero (e.g. {nonzero}); "
            "the Rust harness or the adapter is inconsistent"
        )

    delta_from_means = stats.a_mean - stats.b_mean
    consistency_error = abs(stats.delta - delta_from_means)
    if consistency_error > 1e-6:
        raise SummaryError(
            "Delta does not equal mean(A) - mean(B); per-game data is inconsistent"
        )

    end_reasons: dict[str, int] = {}
    total_decisions = 0
    trace_hashes = 0
    for seed in seeds:
        for placement in PLACEMENTS:
            record = evaluation.by_seed[seed][placement]
            end_reasons[record.end_reason] = end_reasons.get(record.end_reason, 0) + 1
            total_decisions += record.decisions
            if record.trace_hash is not None:
                trace_hashes += 1

    unexpected_end_reasons = sorted(
        reason for reason in end_reasons if reason not in KNOWN_END_REASONS
    )

    replay = deterministic_replay or {"status": "not-checked"}
    warnings: list[str] = []
    if replay.get("status") not in ("verified",):
        warnings.append(
            "deterministic replay was not verified for this run; "
            "see the manifest for the recorded status"
        )
    if unexpected_end_reasons:
        warnings.append(
            f"unexpected end_reason value(s) {unexpected_end_reasons}; "
            f"known values are {list(KNOWN_END_REASONS)}"
        )

    limits = list(plan.get("limitations") or [])
    if (plan.get("sample") or {}).get("precision_target_met") is False:
        warnings.append(
            "the pilot-based normal approximation could not plan the requested sample "
            "size within the runtime budget, so the target precision may not be reached; "
            "the bootstrap interval measured on the main sample is authoritative"
        )
    pilot_block_sd = float((plan.get("pilot") or {}).get("block_sd") or 0.0)
    if pilot_block_sd == 0.0:
        warnings.append(
            "pilot block SD was exactly zero, so the planned sample size is a minimum "
            "and the results do not establish zero uncertainty"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "created_at_utc": utc_now(),
        "mode": mode,
        "decision_limit": plan.get("decision_limit"),
        "seeds": {
            "count": len(seeds),
            "list": seeds,
            "procedure": plan.get("seed_procedure"),
            "pilot_excluded": True,
            "unit": "seed block (six placements per seed)",
        },
        "placements": [list(placement) for placement in PLACEMENTS],
        "analysis": {
            **(plan.get("analysis") or {}),
            "block_weights": "equal across seed blocks and placements",
            "early_stopping": False,
        },
        "means_per_player_per_game": {
            "a": stats.a_mean,
            "b": stats.b_mean,
            "delta": stats.delta,
            "delta_from_means": delta_from_means,
            "consistency_abs_error": consistency_error,
        },
        "uncertainty": {
            "sample_sd": stats.sample_sd,
            "standard_error": stats.standard_error,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_half_width": (ci_high - ci_low) / 2.0,
            "confidence": float(analysis.get("confidence", CONFIDENCE)),
            "resamples": int(
                analysis.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
            ),
            "bootstrap_seed": int(analysis.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
            "method": "percentile bootstrap over whole resampled seed blocks",
            "interval_type": "two-sided",
            "se_formula": "sample_sd(D) / sqrt(number of seeds)",
        },
        "checks": {
            "at_least_two_seed_blocks": len(seeds) >= MIN_INFERENCE_SEEDS,
            "complete_placement_coverage": True,
            "zero_sum_scores": True,
            "delta_matches_policy_means": consistency_error <= 1e-6,
            "symmetric_zero_blocks": {
                "required": symmetric_required,
                "all_zero": zero["all_zero"],
                "tolerance": zero["tolerance"],
                "max_abs_block_contrast": zero["max_abs_block_contrast"],
                "nonzero_blocks": zero["nonzero_blocks"],
            },
            "deterministic_replay": replay.get("status", "not-checked"),
        },
        "diagnostics": {
            "block_contrasts": {
                str(seed): value for seed, value in zip(stats.seeds, stats.block_contrasts)
            },
            "placement_a_means": {
                str(list(placement)): value
                for placement, value in zip(PLACEMENTS, stats.placement_a_means)
            },
            "placement_b_means": {
                str(list(placement)): value
                for placement, value in zip(PLACEMENTS, stats.placement_b_means)
            },
            "total_decisions": total_decisions,
            "end_reasons": end_reasons,
            "known_end_reasons": list(KNOWN_END_REASONS),
            "unexpected_end_reasons": unexpected_end_reasons,
            "trace_hash_coverage": {
                "records_with_trace_hash": trace_hashes,
                "records": len(seeds) * len(PLACEMENTS),
            },
            "timing": {
                **(timing or {}),
                "diagnostic_only": True,
            },
        },
        "limitations": limits,
        "warnings": warnings,
        "integrity": integrity
        or {
            "verified": False,
            "note": "summary built without artifact-integrity evidence",
        },
        "planning": {
            "target_precision": (plan.get("target") or {}).get("target_precision"),
            "runtime_budget_seconds": (plan.get("target") or {}).get(
                "runtime_budget_seconds"
            ),
            "pilot_block_sd": pilot_block_sd,
            "predicted_half_width": (plan.get("sample") or {}).get(
                "predicted_half_width"
            ),
            "predicted_half_width_is_estimate": True,
            "precision_target_met": (plan.get("sample") or {}).get(
                "precision_target_met"
            ),
            "measured_ci_half_width": (ci_high - ci_low) / 2.0,
            "note": (
                "The sample size used a normal approximation from the pilot block SD. "
                "The reported bootstrap interval is measured on the main sample and the "
                "target precision is a planning estimate, not a guarantee."
            ),
        },
        "provenance": {
            "plan_sha256": None,  # filled by write_summary
            "results_sha256": None,
            "manifest": "manifest.json",
            "manifest_sha256": None,  # filled by write_summary
        },
    }


def verify_artifact_integrity(
    work_dir: Path, plan: dict[str, Any], *, results_path: Path
) -> dict[str, Any]:
    """Check that stored results, seeds, plan, and manifest agree.

    Summarizing recomputes the estimate from saved per-game records only; it
    never invokes the driver or replays games. The manifest hash evidence makes
    post-hoc edits to results visible instead of silently changing the estimate.
    """
    work_dir = Path(work_dir)
    results_path = Path(results_path)
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ExperimentError(
            f"manifest.json is missing from {work_dir}; refusing to publish a summary "
            "without provenance evidence"
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ExperimentError("manifest.json is not a JSON object")
    artifacts = manifest.get("artifacts") or {}
    missing = [key for key in REQUIRED_MANIFEST_ARTIFACTS if not artifacts.get(key)]
    if missing:
        raise ExperimentError(f"manifest is missing artifact evidence {missing}")

    recorded_results = work_dir / str(artifacts["main_results_path"])
    if recorded_results.resolve() != results_path.resolve():
        raise ExperimentError(
            "manifest main results path does not match the file being summarized"
        )
    actual_results_sha = sha256_file(results_path)
    if actual_results_sha is None:
        raise ExperimentError(f"results file is missing: {results_path}")
    if artifacts["main_results_sha256"] != actual_results_sha:
        raise ExperimentError(
            "main results hash does not match the manifest; the stored results were "
            "modified after the run, so no summary can be published"
        )

    seeds_path = work_dir / str(artifacts["main_seeds_path"])
    actual_seeds_sha = sha256_file(seeds_path)
    if actual_seeds_sha is None:
        raise ExperimentError(f"seeds file is missing: {seeds_path}")
    if artifacts["main_seeds_sha256"] != actual_seeds_sha:
        raise ExperimentError(
            "main seeds hash does not match the manifest; the frozen sample was changed"
        )

    run_record_path = work_dir / str(artifacts["main_run_path"])
    actual_run_sha = sha256_file(run_record_path)
    if actual_run_sha is None:
        raise ExperimentError(f"main run record is missing: {run_record_path}")
    if artifacts["main_run_sha256"] != actual_run_sha:
        raise ExperimentError(
            "main run record hash does not match the manifest; the recorded process "
            "outcome was changed"
        )
    run_record = read_json(run_record_path)
    if run_record.get("returncode") != 0:
        raise ExperimentError(
            f"the recorded main driver exited with status {run_record.get('returncode')!r}; "
            "no summary can be published for a non-zero exit"
        )
    if run_record.get("timed_out"):
        raise ExperimentError(
            "the recorded main driver timed out; no summary can be published"
        )
    if run_record.get("results_sha256") not in (None, actual_results_sha):
        raise ExperimentError(
            "the main run record describes different results than the stored file"
        )
    replay_record = run_record.get("deterministic_replay") or {}
    replay_status = replay_record.get("status")
    if replay_status == "failed":
        raise ExperimentError(
            "the recorded deterministic-replay check failed; no summary can be published"
        )
    if replay_status not in ("verified", "not-checked"):
        raise ExperimentError(
            f"unexpected deterministic-replay status {replay_status!r}; refusing to publish"
        )
    recorded_identity_check = run_record.get("identity_check") or {}
    identity_status = recorded_identity_check.get("status")
    if identity_status == "drifted":
        raise ExperimentError(
            "the recorded main run changed its driver/worker/heuristic/repository "
            "identity while running; no summary can be published"
        )
    if identity_status not in (None, "not-checked", "stable"):
        raise ExperimentError(
            f"unexpected identity-check status {identity_status!r}; refusing to publish"
        )

    actual_plan_sha = sha256_file(work_dir / "plan.json")
    if (manifest.get("plan") or {}).get("sha256") != actual_plan_sha:
        raise ExperimentError(
            "plan.json changed after the main run; re-run the frozen plan instead of "
            "summarizing a modified plan"
        )

    binding = manifest.get("source_binding") or {}
    planned_fingerprint = (plan.get("adapter_identity") or {}).get("fingerprint")
    if binding.get("identity_fingerprint") != planned_fingerprint:
        raise ExperimentError(
            "manifest adapter identity does not match the frozen plan; the run used "
            "different driver/worker/heuristic components than the plan recorded"
        )

    recorded_experiment = manifest.get("experiment") or {}
    recorded_main = recorded_experiment.get("main") or {}
    planned_seeds = list((plan.get("sample") or {}).get("main_seeds") or [])
    if recorded_main.get("seeds") != planned_seeds:
        raise ExperimentError("manifest main seeds do not match the frozen plan")
    if recorded_main.get("count") != (plan.get("sample") or {}).get("n_main"):
        raise ExperimentError("manifest main sample size does not match the frozen plan")
    if recorded_experiment.get("mode") != plan.get("mode"):
        raise ExperimentError("manifest mode does not match the frozen plan")
    if recorded_experiment.get("decision_limit") != plan.get("decision_limit"):
        raise ExperimentError("manifest decision limit does not match the frozen plan")

    return {
        "verified": True,
        "manifest_sha256": sha256_file(manifest_path),
        "plan_sha256": actual_plan_sha,
        "results_sha256": actual_results_sha,
        "seeds_sha256": actual_seeds_sha,
        "run_record_sha256": actual_run_sha,
        "driver_returncode": run_record.get("returncode"),
        "replay_status": replay_status,
        "identity_fingerprint": planned_fingerprint,
        "results_path": relative_display(results_path, work_dir),
        "seeds_path": relative_display(seeds_path, work_dir),
        "replay_triggered": False,
        "note": (
            "Summary recomputed from stored per-game records; no driver invocation "
            "or replay was needed."
        ),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    means = summary["means_per_player_per_game"]
    uncertainty = summary["uncertainty"]
    checks = summary["checks"]
    symmetric = checks["symmetric_zero_blocks"]
    lines = [
        f"# Policy A/B summary ({summary['mode']})",
        "",
        "A = imported ninone heuristic (2 seats), B = simulator FirstLegalPolicy (2 seats).",
        "Scores are reported per player per game in score units.",
        "",
        f"- Seed blocks: {summary['seeds']['count']}",
        f"- A mean: {means['a']:.4f}",
        f"- B mean: {means['b']:.4f}",
        f"- Delta (A - B): {means['delta']:.4f}",
        f"- Sample SD of block contrasts D[s]: {uncertainty['sample_sd']:.4f}",
        f"- Standard error: {uncertainty['standard_error']:.4f}",
        (
            f"- 95% percentile bootstrap interval (whole seed blocks, "
            f"{uncertainty['resamples']} resamples, bootstrap seed "
            f"{uncertainty['bootstrap_seed']}): "
            f"[{uncertainty['ci_low']:.4f}, {uncertainty['ci_high']:.4f}]"
        ),
        "",
        "## Checks",
        "",
        f"- Complete placement coverage: {checks['complete_placement_coverage']}",
        f"- Zero-sum scores: {checks['zero_sum_scores']}",
        f"- Delta equals mean(A) - mean(B): {checks['delta_matches_policy_means']}",
        f"- Deterministic replay: {checks['deterministic_replay']}",
        (
            f"- Symmetric zero-block check required: {symmetric['required']}; "
            f"all blocks zero: {symmetric['all_zero']} "
            f"(max |D[s]| = {symmetric['max_abs_block_contrast']:.3e})"
        ),
        "",
        "## Notes",
        "",
        "- Seeds are listed in `summary.json`; the pilot sample is excluded.",
        "- The interval estimates variation across sampled seeds, not replay variability.",
    ]
    for limitation in summary.get("limitations") or []:
        lines.append(f"- Limitation: {limitation}")
    for warning in summary.get("warnings") or []:
        lines.append(f"- Warning: {warning}")
    lines.append("")
    return "\n".join(lines)


def write_summary(
    work_dir: Path, summary: dict[str, Any], *, results_path: Path
) -> tuple[Path, Path]:
    work_dir = Path(work_dir)
    summary["provenance"]["plan_sha256"] = sha256_file(work_dir / "plan.json")
    summary["provenance"]["results_sha256"] = sha256_file(results_path)
    summary["provenance"]["manifest_sha256"] = sha256_file(work_dir / "manifest.json")
    json_path = work_dir / "summary.json"
    markdown_path = work_dir / "summary.md"
    write_json(json_path, summary)
    markdown_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    # This derived diagnostic describes the latest recomputation, not an
    # immutable failed main attempt. Do not leave it contradicting a valid
    # summary; original main/failures.json is never removed here.
    remove_if_exists(work_dir / "main" / "summarize_failures.json")
    return json_path, markdown_path


# ---------------------------------------------------------------------------
# Main stage orchestration
# ---------------------------------------------------------------------------


def _driver_args(
    args: argparse.Namespace,
    *,
    mode: str | None = None,
    decision_limit: int | None = None,
) -> dict[str, Any]:
    resolved_mode = mode if mode is not None else getattr(args, "mode", None)
    if resolved_mode is None:
        resolved_mode = "mixed"
    resolved_limit = decision_limit
    if resolved_limit is None:
        resolved_limit = getattr(args, "decision_limit", None)
    if resolved_limit is None:
        resolved_limit = DEFAULT_DECISION_LIMIT
    return {
        "driver_command": resolve_driver_command(Path(args.driver)),
        "python_exe": Path(args.python),
        "worker_path": Path(args.worker),
        "ninone_root": Path(args.ninone_root),
        "mode": resolved_mode,
        "decision_limit": int(resolved_limit),
        "timeout": args.driver_timeout,
    }


def _analysis_settings(
    args: argparse.Namespace, *, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve analysis settings, inheriting frozen values when not given.

    For ``main`` the parser defaults are ``None``: an omitted bootstrap setting
    comes from the frozen plan, while an explicitly supplied one must match it
    through :func:`verify_plan_for_main`.
    """
    recorded = (plan or {}).get("analysis") or {}
    resolved: dict[str, int] = {}
    for key in ("bootstrap_seed", "bootstrap_resamples"):
        value = getattr(args, key, None)
        if value is None:
            if plan is not None:
                if key not in recorded:
                    raise ExperimentError(
                        f"the frozen plan has no analysis setting {key!r}; refusing to "
                        "silently substitute a CLI default"
                    )
                value = recorded[key]
            else:
                value = (
                    DEFAULT_BOOTSTRAP_SEED
                    if key == "bootstrap_seed"
                    else DEFAULT_BOOTSTRAP_RESAMPLES
                )
        resolved[key] = int(value)
    if resolved["bootstrap_resamples"] < 1:
        raise ExperimentError("--bootstrap-resamples must be at least 1")
    if not 0 <= resolved["bootstrap_seed"] <= U64_MAX:
        raise ExperimentError("--bootstrap-seed must be a u64")
    return {
        "bootstrap_seed": resolved["bootstrap_seed"],
        "bootstrap_resamples": resolved["bootstrap_resamples"],
        "confidence": CONFIDENCE,
        "interval": "percentile bootstrap over resampled whole seed blocks",
    }


def _ninone_capture(ninone_root: Path) -> dict[str, Any]:
    return capture_repository(
        Path(ninone_root), name=Path(ninone_root).name, reference_dir=Path(ninone_root).parent
    )


def _current_identity(args: argparse.Namespace) -> dict[str, Any]:
    return adapter_identity(
        driver=Path(args.driver),
        worker=Path(args.worker),
        python_exe=Path(args.python),
        repo_root=Path(args.repo_root),
        ninone_capture=_ninone_capture(Path(args.ninone_root)),
        allow_stale_driver=bool(getattr(args, "allow_stale_driver", False)),
    )


def determinism_check(
    *,
    seeds: Sequence[int],
    main_dir: Path,
    driver_args: dict[str, Any],
    checks: int,
    primary: dict[int, dict[tuple[int, int], GameResult]],
) -> tuple[dict[str, Any], list[Problem]]:
    """Replay a prefix of the main seeds and compare deterministic results."""
    subset = list(seeds)[: max(0, checks)]
    if not subset:
        return {"status": "not-checked", "seeds": []}, []
    replay_dir = Path(main_dir) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = replay_dir / "seeds.json"
    output_path = replay_dir / "results.jsonl"
    write_json(seeds_path, subset)
    run = invoke_driver(seeds_path=seeds_path, output_path=output_path, **driver_args)
    evaluation = evaluate_run(subset, run)
    problems = list(evaluation.problems)
    mismatches: list[dict[str, Any]] = []

    if evaluation.ok:
        for seed in subset:
            for placement in PLACEMENTS:
                original = primary[seed][placement]
                replay = evaluation.by_seed[seed][placement]
                if (
                    original.scores != replay.scores
                    or original.end_reason != replay.end_reason
                    or original.decisions != replay.decisions
                    or original.trace_hash != replay.trace_hash
                ):
                    mismatches.append(
                        {
                            "seed": seed,
                            "placement": list(placement),
                            "main": original.to_json(),
                            "replay": replay.to_json(),
                        }
                    )
        if mismatches:
            problems = [
                Problem(
                    "deterministic_replay_mismatch",
                    f"{len(mismatches)} game(s) differed on replay of {len(subset)} seed(s)",
                    context={"mismatches": mismatches[:20]},
                )
            ]

    if problems:
        status = "failed"
    else:
        status = "verified"
    info = {
        "status": status,
        "seeds": subset,
        "elapsed_seconds": run.elapsed_seconds,
        "returncode": run.returncode,
        "results_sha256": sha256_file(output_path),
        "mismatches": mismatches[:20],
        "results_path": "main/replay/results.jsonl",
    }
    return info, problems


def run_main(work_dir: Path, *, plan: dict[str, Any], args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    work_dir = Path(work_dir)
    main_dir = work_dir / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    # Preserve any earlier attempt instead of overwriting its evidence.
    for existing in (main_dir / "results.jsonl", main_dir / "failures.json"):
        if existing.exists():
            raise ExperimentError(
                f"main attempt already exists at {existing}; refusing to overwrite "
                "preserved results or failure evidence. Use a new work directory for a "
                "new attempt."
            )
    # A summary is published only after full validation of this run.
    remove_if_exists(work_dir / "summary.json")
    remove_if_exists(work_dir / "summary.md")

    mode = args.mode if args.mode is not None else plan.get("mode")
    decision_limit = (
        args.decision_limit
        if args.decision_limit is not None
        else int(plan.get("decision_limit", DEFAULT_DECISION_LIMIT))
    )
    analysis = _analysis_settings(args, plan=plan)
    identity = _current_identity(args)
    seeds = verify_plan_for_main(
        plan,
        mode=mode,
        decision_limit=decision_limit,
        analysis=analysis,
        adapter_identity=identity,
    )

    driver_args = _driver_args(args, mode=mode, decision_limit=decision_limit)

    # Capture what will actually be executed before the process starts. The
    # manifest reuses these captures instead of re-reading repositories after
    # the run, so recorded evidence cannot be satisfied by later edits.
    pre_evidence = capture_run_evidence(
        work_dir=work_dir,
        stage="main-pre-run",
        repo_root=Path(args.repo_root),
        ninone_root=Path(args.ninone_root),
        identity=identity,
    )
    write_json(main_dir / "pre-run-evidence.json", pre_evidence)

    seeds_path = main_dir / "seeds.json"
    output_path = main_dir / "results.jsonl"
    write_json(seeds_path, seeds)
    run = invoke_driver(seeds_path=seeds_path, output_path=output_path, **driver_args)
    evaluation = evaluate_run(seeds, run)

    # Never replay against an incomplete run: the primary blocks would be
    # missing and diagnostics must be written before anything else.
    replay_info: dict[str, Any] = {"status": "not-checked", "seeds": []}
    replay_problems: list[Problem] = []
    if evaluation.ok:
        replay_info, replay_problems = determinism_check(
            seeds=seeds,
            main_dir=main_dir,
            driver_args=driver_args,
            checks=int(args.determinism_check),
            primary=evaluation.by_seed,
        )
    else:
        replay_info = {
            "status": "skipped",
            "reason": "main run invalid; deterministic replay not attempted",
            "seeds": [],
        }

    # Re-derive the executed identity after main and replay finished and compare
    # it with the pre-run capture. Sources changing mid-run invalidate the run.
    post_evidence, identity_check = capture_post_evidence(
        _evidence_provider(args, work_dir=work_dir, stage="main-post-run"),
        pre_evidence,
    )
    write_json(
        main_dir / "post-run-evidence.json",
        post_evidence
        if post_evidence is not None
        else {"stage": "main-post-run", "capture_error": identity_check.get("error")},
    )
    write_json(main_dir / "identity-check.json", identity_check)

    timing = {
        "elapsed_seconds": run.elapsed_seconds,
        "mean_seconds_per_seed": run.elapsed_seconds / max(1, len(seeds)),
        "pilot_mean_seconds_per_seed": (plan.get("pilot") or {}).get(
            "mean_seconds_per_seed"
        ),
    }

    write_json(
        main_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "main",
            "command": run.command,
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "timeout_seconds": run.timeout_seconds,
            "elapsed_seconds": run.elapsed_seconds,
            "mean_seconds_per_seed": timing["mean_seconds_per_seed"],
            "stdout_tail": run.stdout_tail,
            "stderr_tail": run.stderr_tail,
            "seeds_path": relative_display(seeds_path, work_dir),
            "output_path": relative_display(output_path, work_dir),
            "results_sha256": sha256_file(output_path),
            "pre_run_evidence": relative_display(
                main_dir / "pre-run-evidence.json", work_dir
            ),
            "post_run_evidence": relative_display(
                main_dir / "post-run-evidence.json", work_dir
            ),
            "identity_check": identity_check,
            "deterministic_replay": replay_info,
        },
    )

    manifest = build_manifest(
        work_dir=work_dir,
        repo_root=Path(args.repo_root),
        ninone_root=Path(args.ninone_root),
        plan=plan,
        adapter=identity,
        invocation={
            "command": run.command,
            "seeds_path": relative_display(seeds_path, work_dir),
            "output_path": relative_display(output_path, work_dir),
            "mode": mode,
            "decision_limit": decision_limit,
            "python": str(args.python),
            "worker": str(args.worker),
            "ninone_root": str(args.ninone_root),
            "driver": str(args.driver),
        },
        timing=timing,
        deterministic_replay=replay_info,
        argv=argv,
        artifacts={
            "pilot_seeds_path": "pilot/seeds.json",
            "pilot_seeds_sha256": sha256_file(work_dir / "pilot" / "seeds.json"),
            "pilot_results_path": "pilot/results.jsonl",
            "pilot_results_sha256": sha256_file(work_dir / "pilot" / "results.jsonl"),
            "main_seeds_path": relative_display(seeds_path, work_dir),
            "main_seeds_sha256": sha256_file(seeds_path),
            "main_results_path": relative_display(output_path, work_dir),
            "main_results_sha256": sha256_file(output_path),
            "main_run_path": relative_display(main_dir / "run.json", work_dir),
            "main_run_sha256": sha256_file(main_dir / "run.json"),
            "pre_run_evidence_path": relative_display(
                main_dir / "pre-run-evidence.json", work_dir
            ),
            "pre_run_evidence_sha256": sha256_file(
                main_dir / "pre-run-evidence.json"
            ),
        },
        captures=pre_evidence,
        identity_check=identity_check,
    )
    write_json(work_dir / "manifest.json", manifest)

    run_problems = list(evaluation.problems)
    stage = "main"
    if identity_check["status"] != "stable":
        stage = "main-identity-drift"
        run_problems.append(
            Problem(
                "identity_drifted",
                "driver/worker/heuristic/repository identity changed while main and "
                "replay ran; the manifest records the pre-run implementation, which is "
                "not the one that finished the run",
                context=identity_check,
            )
        )
    if evaluation.ok and replay_problems:
        if stage == "main":
            stage = "main-determinism-check"
        run_problems.extend(replay_problems)
    if run_problems:
        failed = RunEvaluation(
            ok=False,
            results=evaluation.results,
            problems=run_problems,
            by_seed=evaluation.by_seed,
        )
        write_json(
            main_dir / "failures.json",
            failures_payload(
                failed,
                stage=stage,
                extra={
                    "manifest": "manifest.json",
                    "deterministic_replay": replay_info,
                    "identity_check": identity_check,
                },
            ),
        )
        raise ExperimentError(
            f"main run invalid ({len(run_problems)} problem(s)); "
            f"see {main_dir / 'failures.json'}; no summary published"
        )

    integrity = verify_artifact_integrity(work_dir, plan, results_path=output_path)
    summary = build_summary(
        plan=plan,
        evaluation=evaluation,
        timing=timing,
        deterministic_replay=replay_info,
        integrity=integrity,
    )
    json_path, markdown_path = write_summary(work_dir, summary, results_path=output_path)
    print(f"summary: {json_path}")
    print(f"summary: {markdown_path}")
    return summary


def summarize_artifacts(work_dir: Path) -> dict[str, Any]:
    """Recompute the summary from stored per-game results (no replay).

    A recomputation failure is written to the derived
    ``main/summarize_failures.json``. The original ``main/failures.json`` from
    the main process is never overwritten, so its stderr tail and failure
    context remain preserved. No normal summary is left on disk.
    """
    work_dir = Path(work_dir)
    # Any stale summary must disappear before validation, so an invalid or
    # tampered artifact set can never leave a normal summary on disk.
    remove_if_exists(work_dir / "summary.json")
    remove_if_exists(work_dir / "summary.md")
    try:
        return _recompute_summary(work_dir)
    except (ExperimentError, SummaryError) as exc:
        write_json(
            work_dir / "main" / "summarize_failures.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "stage": "summarize",
                "generated_at_utc": utc_now(),
                "normal_summary_published": False,
                "error": str(exc),
                "problems": list(getattr(exc, "problems", [])),
                "note": (
                    "Derived diagnostics from re-summarizing stored artifacts. The "
                    "original main/failures.json, when present, is preserved unchanged."
                ),
            },
        )
        remove_if_exists(work_dir / "summary.json")
        remove_if_exists(work_dir / "summary.md")
        raise


def _recompute_summary(work_dir: Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    plan = load_plan(work_dir)
    seeds = resolve_plan_seeds(plan)
    output_path = work_dir / "main" / "results.jsonl"
    evaluation = load_output(seeds, output_path)
    if not evaluation.ok:
        raise SummarizeFailure(
            f"stored results are invalid ({len(evaluation.problems)} problem(s)); "
            "see main/summarize_failures.json",
            problems=[problem.to_json() for problem in evaluation.problems],
        )
    run_record: dict[str, Any] = {}
    run_path = work_dir / "main" / "run.json"
    if run_path.is_file():
        run_record = read_json(run_path)
    timing = {
        "elapsed_seconds": run_record.get("elapsed_seconds"),
        "mean_seconds_per_seed": run_record.get("mean_seconds_per_seed"),
        "pilot_mean_seconds_per_seed": (plan.get("pilot") or {}).get(
            "mean_seconds_per_seed"
        ),
    }
    integrity = verify_artifact_integrity(work_dir, plan, results_path=output_path)
    summary = build_summary(
        plan=plan,
        evaluation=evaluation,
        timing=timing,
        deterministic_replay=run_record.get("deterministic_replay"),
        integrity=integrity,
    )
    write_summary(work_dir, summary, results_path=output_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_work_dir(args: argparse.Namespace, *, require: bool = False) -> Path:
    if args.work_dir is not None:
        return Path(args.work_dir).resolve()
    if require:
        raise ExperimentError(
            "this step needs an explicit --work-dir: it must operate on the work "
            "directory that holds the pilot/plan/main artifacts"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (SCRIPT_DIR / "runs" / f"{stamp}-{getattr(args, 'mode', None) or 'mixed'}").resolve()


def _capture_evidence(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    stage: str,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return capture_run_evidence(
        work_dir=work_dir,
        stage=stage,
        repo_root=Path(args.repo_root),
        ninone_root=Path(args.ninone_root),
        identity=identity if identity is not None else _current_identity(args),
    )


def _evidence_provider(
    args: argparse.Namespace, *, work_dir: Path, stage: str
) -> Callable[[], dict[str, Any]]:
    return lambda: _capture_evidence(args, work_dir=work_dir, stage=stage)



def _print_plan(plan: dict[str, Any]) -> None:
    sample = plan["sample"]
    print(
        f"plan frozen: n_main={sample['n_main']} "
        f"predicted_half_width={sample['predicted_half_width']:.4f} "
        f"precision_target_met={sample['precision_target_met']}"
    )
    for limitation in plan.get("limitations") or []:
        print(f"  limitation: {limitation}")


def cmd_run(args: argparse.Namespace) -> int:
    work_dir = _resolve_work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work_dir}")
    procedure = seed_procedure(int(args.seed_procedure_seed))
    pilot_seeds = prepare_corpus(work_dir, procedure=procedure, pilot_size=int(args.pilot_size))
    print(f"pilot seeds: {len(pilot_seeds)}")
    identity = _current_identity(args)
    pre_evidence = _capture_evidence(
        args, work_dir=work_dir, stage="pilot-pre-run", identity=identity
    )
    pilot_report = run_pilot(
        work_dir,
        seeds=pilot_seeds,
        driver_args=_driver_args(args),
        adapter_identity=identity,
        pre_evidence=pre_evidence,
        post_evidence_provider=_evidence_provider(
            args, work_dir=work_dir, stage="pilot-post-run"
        ),
    )
    print(
        f"pilot done: block_sd={pilot_report['block_sd']:.4f} "
        f"mean_seconds_per_seed={pilot_report['mean_seconds_per_seed']:.4f}"
    )
    plan = build_plan(
        pilot_report=pilot_report,
        procedure=procedure,
        mode=args.mode,
        decision_limit=int(args.decision_limit),
        target_precision=args.target_precision,
        runtime_budget=args.runtime_budget,
        max_main_seeds=args.max_main_seeds,
        analysis=_analysis_settings(args),
        adapter_identity=_current_identity(args),
    )
    write_plan(work_dir, plan)
    _print_plan(plan)
    run_main(work_dir, plan=plan, args=args, argv=sys.argv[1:])
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    work_dir = _resolve_work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    procedure = seed_procedure(int(args.seed_procedure_seed))
    pilot_seeds = prepare_corpus(work_dir, procedure=procedure, pilot_size=int(args.pilot_size))
    identity = _current_identity(args)
    report = run_pilot(
        work_dir,
        seeds=pilot_seeds,
        driver_args=_driver_args(args),
        adapter_identity=identity,
        pre_evidence=_capture_evidence(
            args, work_dir=work_dir, stage="pilot-pre-run", identity=identity
        ),
        post_evidence_provider=_evidence_provider(
            args, work_dir=work_dir, stage="pilot-post-run"
        ),
    )
    print(
        f"pilot done: count={report['count']} block_sd={report['block_sd']:.4f} "
        f"mean_seconds_per_seed={report['mean_seconds_per_seed']:.4f}"
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    work_dir = _resolve_work_dir(args, require=True)
    report_path = work_dir / "pilot" / "report.json"
    pilot_report = read_json(report_path)
    procedure = plan_procedure_from(work_dir, pilot_report)
    plan = build_plan(
        pilot_report=pilot_report,
        procedure=procedure,
        mode=args.mode,
        decision_limit=args.decision_limit,
        target_precision=args.target_precision,
        runtime_budget=args.runtime_budget,
        max_main_seeds=args.max_main_seeds,
        analysis=_analysis_settings(args),
        adapter_identity=_current_identity(args),
    )
    write_plan(work_dir, plan)
    _print_plan(plan)
    return 0


def plan_procedure_from(
    work_dir: Path, pilot_report: dict[str, Any]
) -> dict[str, Any]:
    """Recover the recorded seed procedure; a missing corpus is an error.

    Planning must never silently fall back to a default procedure: the pilot
    corpus and main seeds have to come from the same recorded stream.
    """
    work_dir = Path(work_dir)
    corpus_path = work_dir / "corpus.json"
    if not corpus_path.is_file():
        raise ExperimentError(
            f"pilot corpus is missing: {corpus_path}; run the pilot in this work "
            "directory before planning"
        )
    corpus = read_json(corpus_path)
    procedure = corpus.get("procedure")
    if not isinstance(procedure, dict):
        raise ExperimentError(f"{corpus_path} has no usable seed procedure")
    validate_seed_procedure(procedure)
    corpus_seeds = [int(seed) for seed in corpus.get("pilot_seeds") or []]
    report_seeds = [int(seed) for seed in pilot_report.get("seeds") or []]
    if corpus_seeds != report_seeds:
        raise ExperimentError(
            "the pilot corpus seeds do not match the pilot report; the work "
            "directory mixes different pilot samples"
        )
    return procedure


def cmd_main(args: argparse.Namespace) -> int:
    work_dir = _resolve_work_dir(args, require=True)
    plan = load_plan(work_dir)
    run_main(work_dir, plan=plan, args=args, argv=sys.argv[1:])
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    work_dir = _resolve_work_dir(args, require=True)
    summary = summarize_artifacts(work_dir)
    means = summary["means_per_player_per_game"]
    uncertainty = summary["uncertainty"]
    print(
        f"delta={means['delta']:.4f} se={uncertainty['standard_error']:.4f} "
        f"ci=[{uncertainty['ci_low']:.4f}, {uncertainty['ci_high']:.4f}] "
        f"blocks={summary['seeds']['count']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment.py",
        description=(
            "Reproducible mixed-policy A/B experiment: pilot -> plan -> main -> "
            "summarize (see specs/active/reproducible-policy-ab/design.md)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(
        target: argparse.ArgumentParser,
        *,
        mode_default: str | None,
        limit_default: int | None,
    ) -> None:
        target.add_argument(
            "--work-dir",
            type=Path,
            default=None,
            help="work directory; required for plan, main, and summarize",
        )
        target.add_argument(
            "--repo-root",
            type=Path,
            default=DEFAULT_REPO_ROOT,
            help="simulator repository recorded in the manifest",
        )
        target.add_argument("--ninone-root", type=Path, default=DEFAULT_NINONE_ROOT)
        target.add_argument("--mode", choices=MODES, default=mode_default)
        target.add_argument("--decision-limit", type=int, default=limit_default)

    def add_seed_procedure(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--seed-procedure-seed", type=int, default=DEFAULT_SEED_PROCEDURE_SEED
        )

    def add_analysis(target: argparse.ArgumentParser, *, defaults: bool) -> None:
        # ``main`` records None so omitted settings inherit the frozen plan
        # instead of silently substituting a different CLI default.
        target.add_argument(
            "--bootstrap-seed",
            type=int,
            default=DEFAULT_BOOTSTRAP_SEED if defaults else None,
        )
        target.add_argument(
            "--bootstrap-resamples",
            type=int,
            default=DEFAULT_BOOTSTRAP_RESAMPLES if defaults else None,
        )

    def add_driver(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--driver",
            type=Path,
            required=True,
            help="path to the built policy_ab example binary",
        )
        target.add_argument("--python", type=Path, default=Path(sys.executable))
        target.add_argument("--worker", type=Path, default=DEFAULT_WORKER_PATH)
        target.add_argument("--driver-timeout", type=float, default=None)
        target.add_argument(
            "--allow-stale-driver",
            action="store_true",
            help=(
                "allow a driver executable older than the recorded Rust build inputs; "
                "the mismatch is recorded in the manifest"
            ),
        )

    def add_plan_options(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--target-precision",
            type=float,
            default=None,
            help="target 95% interval half-width in score units",
        )
        target.add_argument(
            "--runtime-budget",
            type=float,
            default=None,
            help="explicit runtime budget in seconds for the main sample",
        )
        target.add_argument("--max-main-seeds", type=int, default=None)

    run_parser = subparsers.add_parser("run", help="pilot -> plan -> main -> summarize")
    add_common(run_parser, mode_default="mixed", limit_default=DEFAULT_DECISION_LIMIT)
    add_seed_procedure(run_parser)
    add_analysis(run_parser, defaults=True)
    add_driver(run_parser)
    add_plan_options(run_parser)
    run_parser.add_argument("--pilot-size", type=int, default=8)
    run_parser.add_argument(
        "--determinism-check",
        type=int,
        default=2,
        help="number of main seeds to replay for a deterministic-replay check (0 disables)",
    )
    run_parser.set_defaults(func=cmd_run)

    pilot_parser = subparsers.add_parser("pilot", help="run the pilot sample only")
    add_common(pilot_parser, mode_default="mixed", limit_default=DEFAULT_DECISION_LIMIT)
    add_seed_procedure(pilot_parser)
    add_driver(pilot_parser)
    pilot_parser.add_argument("--pilot-size", type=int, default=8)
    pilot_parser.set_defaults(func=cmd_pilot)

    plan_parser = subparsers.add_parser("plan", help="freeze the main sample from the pilot")
    add_common(plan_parser, mode_default=None, limit_default=None)
    add_analysis(plan_parser, defaults=True)
    add_driver(plan_parser)
    add_plan_options(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)

    main_parser = subparsers.add_parser("main", help="run the frozen main sample")
    add_common(main_parser, mode_default=None, limit_default=None)
    add_analysis(main_parser, defaults=False)
    add_driver(main_parser)
    main_parser.add_argument("--determinism-check", type=int, default=2)
    main_parser.set_defaults(func=cmd_main)

    summarize_parser = subparsers.add_parser(
        "summarize", help="recompute the summary from stored results"
    )
    # Summarizing only reads the recorded work directory. Exposing ignored
    # settings here would suggest they affect the summary when they do not.
    summarize_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="work directory containing plan.json, manifest.json, and main results",
    )
    summarize_parser.set_defaults(func=cmd_summarize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
