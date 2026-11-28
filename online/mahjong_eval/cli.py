"""Command line interface for formal online policy comparisons.

``mahjong-eval compare`` loads a caller-supplied ``best`` or ``policy`` checkpoint
and validates the frozen Rust seed split. It publishes a plan before play,
streams terminal records, writes measured costs, and summarizes the persisted
files. Validation runs support diagnostics and test runs support the frozen
comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import records, scheduler

COSTS_FORMAT = "mahjong-eval-costs"
COSTS_FORMAT_VERSION = 1
COSTS_FILENAME = "costs.json"

_U64_MAX = 2**64 - 1


class CompareError(RuntimeError):
    """An invalid comparison configuration or a failed comparison run."""


def _offline_dependency_hint(exc: ImportError) -> str:
    return (
        "comparison requires the optional offline dependencies 'mahjong-bc' "
        "and PyTorch; install the offline package first with "
        "'python -m pip install -e ./python', then install the online "
        "package with its 'evaluation' extra "
        f"({type(exc).__name__}: {exc})"
    )


def _require_offline_evaluation() -> None:
    try:
        import mahjong_bc  # noqa: F401
    except ImportError as exc:
        raise CompareError(_offline_dependency_hint(exc)) from exc


def _handled_errors() -> tuple[type[BaseException], ...]:
    """Expected failures, including the offline package's error only if loaded."""

    errors: tuple[type[BaseException], ...] = (
        CompareError,
        records.RecordError,
        OSError,
    )
    try:
        from mahjong_bc.errors import MahjongBCError
    except ImportError:
        return errors
    return (*errors, MahjongBCError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mahjong-eval",
        description=(
            "Run frozen neural policies against the native Sichuan mahjong "
            "environment and persist reproducible comparison records"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser(
        "compare",
        help="compare a frozen checkpoint over six placements per seed",
    )
    compare.add_argument(
        "--checkpoint",
        required=True,
        help="compatible 'best' or 'policy' checkpoint; 'last' is reserved for recovery",
    )
    compare.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
        help="frozen Rust seed split; validation is diagnostic, test is final",
    )
    compare.add_argument(
        "--seeds",
        required=True,
        help="comma-separated seed list; every seed must belong to --split",
    )
    compare.add_argument("--out", required=True, help="fresh comparison run directory")
    compare.add_argument(
        "--batch-size",
        type=int,
        default=scheduler.DEFAULT_BATCH_SIZE,
        help="maximum neural decisions per inference batch",
    )
    compare.add_argument(
        "--concurrent-games",
        type=int,
        default=scheduler.DEFAULT_CONCURRENT_GAMES,
        help="simultaneously active environment handles",
    )
    compare.add_argument(
        "--decision-limit",
        type=int,
        default=None,
        help="per-game accepted-decision bound (native default when omitted)",
    )
    compare.add_argument(
        "--teacher-cache-bytes",
        type=int,
        default=None,
        help="shared teacher cache budget in bytes; 0 disables it",
    )
    compare.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda[:index] for frozen inference",
    )

    subparsers.add_parser("version", help="print package and dependency versions")
    return parser


def _parse_seeds(text: str) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise CompareError("--seeds must contain at least one seed")
    seeds: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise CompareError(f"invalid --seeds list {text!r}: empty entry")
        try:
            seed = int(token, 10)
        except ValueError as exc:
            raise CompareError(
                f"invalid --seeds entry {token!r}: expected a decimal integer"
            ) from exc
        if seed < 0 or seed > _U64_MAX:
            raise CompareError(f"seed {seed} must fit in an unsigned 64-bit integer")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise CompareError("--seeds entries must be unique")
    return sorted(seeds)


def _require_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CompareError(f"--{name.replace('_', '-')} must be a positive integer")
    return value


def _require_nonnegative(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CompareError(
            f"--{name.replace('_', '-')} must be a nonnegative integer"
        )
    return value


def _validate_split_seeds(seeds: Sequence[int], split: str) -> dict[str, int]:
    """Reject any seed the frozen Rust split does not assign to ``split``."""

    buckets: dict[str, int] = {}
    for seed in seeds:
        actual = scheduler_seed_split(seed)
        if actual != split:
            raise CompareError(
                f"seed {seed} belongs to the {actual!r} split, not {split!r}"
            )
        buckets[str(seed)] = scheduler_seed_bucket(seed)
    return buckets


def _native_extension(label: str) -> Any:
    import mahjong_eval

    try:
        return getattr(mahjong_eval, label)
    except ImportError as exc:
        raise CompareError(
            "the native environment extension is unavailable; build and install "
            "the online wheel or an editable install first "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def scheduler_seed_split(seed: int) -> str:
    return _native_extension("seed_split")(seed)


def scheduler_seed_bucket(seed: int) -> int:
    return int(_native_extension("split_bucket")(seed))


def _compiled_source() -> dict[str, Any]:
    import mahjong_eval

    try:
        source = mahjong_eval.compiled_source()
    except ImportError as exc:
        raise CompareError(
            "the native environment extension is unavailable; build and install "
            "the online wheel or an editable install first "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(source, dict):
        raise CompareError("compiled source identity is not a JSON object")
    return dict(source)


def _dependency_record() -> dict[str, Any]:
    from mahjong_bc.config import collect_dependency_versions

    value = collect_dependency_versions().to_dict()
    value["evaluator"] = {
        "package": "mahjong-eval",
        "package_version": _online_version(),
    }
    return value


def _online_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("mahjong-eval")
    except Exception:
        return None


def _costs_document(
    *,
    result: scheduler.SchedulingResult,
    split: str,
    seeds: Sequence[int],
    checkpoint_sha256: str,
    device: str,
) -> dict[str, Any]:
    return {
        "format": COSTS_FORMAT,
        "format_version": COSTS_FORMAT_VERSION,
        "split": split,
        "seeds": list(seeds),
        "games": result.counts["games"],
        "checkpoint_sha256": checkpoint_sha256,
        "device": device,
        "counts": dict(result.counts),
        "timings": dict(result.timings),
        "per_game": [
            {
                "seed": game.seed,
                "neural_seats": list(game.neural_seats),
                "decisions": game.decisions,
                "neural_decisions": game.neural_decisions,
                "neural_forced_decisions": game.neural_forced_decisions,
                "forced_decisions": game.forced_decisions,
                "inference_rows": game.inference_rows,
                "winners": list(game.winners),
                "end_reason": game.end_reason,
            }
            for game in result.games
        ],
    }


def write_costs(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish measured costs; callers run this before closing."""

    from mahjong_bc.checkpoint import atomic_write_json

    try:
        atomic_write_json(path, document)
    except (OSError, TypeError, ValueError) as exc:
        raise CompareError(f"failed to write costs {path}: {exc}") from exc


def _cmd_compare(args: argparse.Namespace) -> int:
    seeds = _parse_seeds(args.seeds)
    batch_size = _require_positive("batch_size", args.batch_size)
    concurrent_games = _require_positive("concurrent_games", args.concurrent_games)
    if args.decision_limit is not None:
        _require_positive("decision_limit", args.decision_limit)
    if args.teacher_cache_bytes is not None:
        _require_nonnegative("teacher_cache_bytes", args.teacher_cache_bytes)
    _require_offline_evaluation()
    buckets = _validate_split_seeds(seeds, args.split)

    from mahjong_bc.constants import SPLIT_ALGORITHM

    import mahjong_eval

    from .inference import FrozenPolicy

    decision_limit = (
        int(mahjong_eval.DEFAULT_DECISION_LIMIT)
        if args.decision_limit is None
        else args.decision_limit
    )
    teacher_cache_bytes = (
        int(mahjong_eval.DEFAULT_TEACHER_CACHE_BYTES)
        if args.teacher_cache_bytes is None
        else args.teacher_cache_bytes
    )

    loaded, identity = records.load_checkpoint_with_identity(
        args.checkpoint, device=args.device
    )
    policy = FrozenPolicy.from_loaded(loaded, device=args.device)
    device = str(policy.device)
    compiled = _compiled_source()
    runtime_encoding = int(compiled.get("encoding_version", 0))
    if runtime_encoding != int(identity["encoding_version"]):
        raise CompareError(
            f"the native encoder version {runtime_encoding} does not match the "
            f"checkpoint encoding version {identity['encoding_version']}"
        )

    cache = mahjong_eval.TeacherCache(budget_bytes=teacher_cache_bytes)
    metadata = records.build_plan_metadata(
        checkpoint=identity,
        rust_runtime=compiled,
        split_algorithm=SPLIT_ALGORITHM,
        selection={
            "mode": "greedy",
            "device": device,
            "batch_size": batch_size,
            "concurrent_games": concurrent_games,
        },
        teacher_cache={
            "scope": "shared",
            "budget_bytes": teacher_cache_bytes,
            "disabled": teacher_cache_bytes == 0,
        },
        dependencies=_dependency_record(),
        training_provenance=records.training_provenance(loaded),
        extra={
            "command": "compare",
            "split": args.split,
            "split_buckets": buckets,
            "concurrent_games": concurrent_games,
            "decision_limit": decision_limit,
        },
    )
    plan = records.EvaluationPlan(
        seeds=tuple(seeds), metadata=metadata, decision_limit=decision_limit
    )
    specs = [
        scheduler.GameSpec(seed=seed, neural_seats=placement)
        for seed in seeds
        for placement in records.PLACEMENTS
    ]

    writer = records.create_run(args.out, plan)
    costs_path = writer.paths.directory / COSTS_FILENAME
    try:
        with writer:
            result = scheduler.run_games(
                specs,
                neural=policy,
                teacher_cache=cache,
                batch_size=batch_size,
                concurrent_games=concurrent_games,
                decision_limit=decision_limit,
                on_terminal=lambda game: writer.append(game.to_record()),
            )
            persisted = records.read_records(
                writer.paths.records, plan, require_complete=True
            )
            _verify_persisted(persisted, result)
            write_costs(
                costs_path,
                _costs_document(
                    result=result,
                    split=args.split,
                    seeds=seeds,
                    checkpoint_sha256=identity["sha256"],
                    device=device,
                ),
            )
        summary = records.finalize_run(writer.paths.directory)
    except Exception as exc:
        raise CompareError(
            f"comparison stopped; completed records are retained in "
            f"{writer.paths.directory}: {type(exc).__name__}: {exc}"
        ) from exc

    _emit(
        {
            "command": "compare",
            "split": args.split,
            "seeds": seeds,
            "out": str(writer.paths.directory),
            "plan": str(writer.paths.plan),
            "records": str(writer.paths.records),
            "costs": str(costs_path),
            "summary": str(writer.paths.summary),
            "result": summary,
        }
    )
    return 0


def _verify_persisted(
    persisted: Sequence[records.TerminalRecord],
    result: scheduler.SchedulingResult,
) -> None:
    # read_records already established plan coverage and unique record keys.
    played = {game.key: game.to_record() for game in result.games}
    if len(persisted) != len(played):
        raise CompareError("persisted game count does not match the played games")
    for record in persisted:
        if record != played.get((record.seed, record.neural_seats)):
            raise CompareError(
                f"persisted terminal result differs from the played game "
                f"(seed {record.seed}, placement {list(record.neural_seats)})"
            )


def _cmd_version(_args: argparse.Namespace) -> int:
    document: dict[str, Any] = {
        "package": "mahjong-eval",
        "package_version": _online_version(),
    }
    try:
        from mahjong_bc.config import collect_dependency_versions

        document["dependencies"] = collect_dependency_versions().to_dict()
    except ImportError as exc:
        document["dependencies"] = None
        document["dependencies_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import mahjong_eval

        document["native"] = mahjong_eval.compiled_source()
    except ImportError as exc:
        document["native"] = None
        document["native_error"] = f"{type(exc).__name__}: {exc}"
    _emit(document)
    return 0


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expected = _handled_errors()
    try:
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "version":
            return _cmd_version(args)
    except expected as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
