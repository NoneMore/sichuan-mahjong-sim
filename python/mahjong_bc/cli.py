"""Command line interface for offline behavior-cloning training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .cache import load_prepared_cache
from .checkpoint import atomic_write_json, load_policy_checkpoint
from .config import (
    ACTIVATION_NAMES,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from .constants import PACKAGE_VERSION, SPLIT_NAMES
from .errors import ConfigurationError, MahjongBCError
from .train import TRAIN_OUTPUT_NAMES, Trainer, evaluate_checkpoint


def _parse_widths(text: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"widths must be comma-separated integers, got {text!r}"
        ) from exc
    if not widths:
        raise argparse.ArgumentTypeError("widths cannot be empty")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mahjong-bc",
        description=(
            "Validate prepared behavior-cloning caches, train a candidate "
            "policy, and evaluate frozen checkpoints"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="train a fresh policy and save its validation-selected best checkpoint",
    )
    train.add_argument("--dataset", required=True, help="prepared cache root")
    train.add_argument("--output", required=True, help="checkpoint output directory")
    train.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    train.add_argument("--seed", type=int, help="training seed")
    train.add_argument("--epochs", type=int, help="epoch count")
    train.add_argument("--batch-size", type=int, help="batch size")
    train.add_argument("--learning-rate", type=float, help="AdamW learning rate")
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--beta1", type=float)
    train.add_argument("--beta2", type=float)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--state-hidden", type=_parse_widths, help="e.g. 256,128")
    train.add_argument("--scorer-hidden", type=_parse_widths, help="e.g. 128")
    train.add_argument(
        "--activation", choices=ACTIVATION_NAMES, help="model activation"
    )
    train.add_argument(
        "--log-file", help="optional JSON summary destination"
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a frozen checkpoint on a validation/test split"
    )
    evaluate.add_argument("--dataset", required=True, help="prepared cache root")
    evaluate.add_argument("--checkpoint", required=True, help="policy checkpoint")
    evaluate.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
        help="split to evaluate",
    )
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--output", help="optional JSON metrics destination")
    evaluate.add_argument("--log-file", help="optional JSON summary destination")

    inspect = subparsers.add_parser(
        "inspect", help="print checkpoint metadata without training"
    )
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--device", default="cpu")

    cache_info = subparsers.add_parser(
        "cache-info",
        help="validate a prepared cache and print its inventory counts",
    )
    cache_info.add_argument("--dataset", required=True, help="prepared cache root")
    cache_info.add_argument("--output", help="optional JSON destination")

    subparsers.add_parser("version", help="print package and dependency versions")
    return parser


def _new_run_config(args: argparse.Namespace) -> tuple[ModelConfig, TrainingConfig]:
    missing = [
        name
        for name, value in (
            ("--seed", args.seed),
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--learning-rate", args.learning_rate),
        )
        if value is None
    ]
    if missing:
        raise MahjongBCError(
            f"a fresh training run requires {', '.join(missing)}"
        )
    training_values: dict[str, Any] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    for name in ("weight_decay", "beta1", "beta2", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            training_values[name] = value
    model_values: dict[str, Any] = {}
    for name in ("state_hidden", "scorer_hidden", "activation"):
        value = getattr(args, name)
        if value is not None:
            model_values[name] = value
    return ModelConfig(**model_values), TrainingConfig(**training_values)


def _cache_metadata(cache: Any) -> dict[str, Any]:
    return {
        "cache_id": cache.cache_id,
        "encoding_version": cache.encoding_version,
        "preparation_format_version": cache.preparation_format_version,
        "preparation_version": cache.preparation_version,
        "source_revision": cache.source_revision,
        "corpus_manifest_sha256": cache.corpus_manifest_sha256,
        "split_algorithm": cache.split_algorithm,
        "provenance": cache.provenance,
    }


def _canonical_path(path: Path | str) -> Path:
    """Resolve parents/symlinks and normalize filesystem case for comparison."""

    resolved = Path(path).expanduser().resolve(strict=False)
    return Path(os.path.normcase(str(resolved)))


def _same_path(left: Path | str, right: Path | str) -> bool:
    return _canonical_path(left) == _canonical_path(right)


def _is_within(path: Path | str, directory: Path | str) -> bool:
    """Whether ``path`` is the directory itself or one of its descendants."""

    candidate = _canonical_path(path)
    root = _canonical_path(directory)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _require_report_destination_absent(path: Path | str, label: str) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ConfigurationError(
            f"{label} destination {path} already exists; report outputs are "
            "never overwritten"
        )


def _validate_report_paths(
    destinations: Sequence[tuple[str, str | None]],
    cache_root: Path | str,
    *,
    checkpoint: Path | str | None = None,
) -> None:
    """Reject report writes that could damage a cache or checkpoint."""

    for label, path in destinations:
        if path is None:
            continue
        if checkpoint is not None and _same_path(path, checkpoint):
            raise ConfigurationError(
                f"{label} destination {path} aliases the checkpoint {checkpoint}"
            )
        if _is_within(path, cache_root):
            raise ConfigurationError(
                f"{label} destination {path} is inside prepared cache "
                f"{cache_root}; report files may not modify cache inventory"
            )
        _require_report_destination_absent(path, label)


def _validate_train_paths(
    output_dir: Path | str,
    log_file: str | None,
    cache_root: Path | str,
) -> None:
    if _is_within(output_dir, cache_root):
        raise ConfigurationError(
            f"training output directory {output_dir} is inside prepared cache "
            f"{cache_root}; choose an output directory outside the cache"
        )
    if log_file is None:
        return
    if _is_within(log_file, cache_root):
        raise ConfigurationError(
            f"--log-file destination {log_file} is inside prepared cache "
            f"{cache_root}; report files may not modify cache inventory"
        )
    protected = [Path(output_dir)]
    protected.extend(Path(output_dir) / name for name in TRAIN_OUTPUT_NAMES)
    if any(_same_path(log_file, path) for path in protected):
        raise ConfigurationError(
            f"--log-file destination {log_file} aliases the training output "
            f"directory or one of {', '.join(TRAIN_OUTPUT_NAMES)}"
        )
    _require_report_destination_absent(log_file, "--log-file")


def _emit(
    value: Any,
    *,
    output: str | None = None,
    log_file: str | None = None,
) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    written: set[Path] = set()
    for path in (output, log_file):
        if path is None:
            continue
        canonical = _canonical_path(path)
        if canonical in written:
            continue
        written.add(canonical)
        _require_report_destination_absent(path, "report")
        atomic_write_json(path, value)
    print(text)


def _cmd_train(args: argparse.Namespace) -> int:
    model_config, training_config = _new_run_config(args)
    cache = load_prepared_cache(args.dataset)
    _validate_train_paths(args.output, args.log_file, cache.root)
    train_dataset = cache.load_split("train")
    validation_dataset = cache.load_split("validation")
    dataset_metadata = _cache_metadata(cache)
    trainer = Trainer(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output,
        device=args.device,
        dataset_provenance=dataset_metadata,
    )
    summary = trainer.train()
    payload = trainer.run_payload(summary)
    result = {
        "command": "train",
        "package_version": payload["package_version"],
        "source": payload["source"],
        "dataset": dataset_metadata,
        "model_config": payload["model_config"],
        "training_config": payload["training_config"],
        "dependencies": payload["dependencies"],
        "device_type": payload["device_type"],
        "train_dataset_identity": payload["dataset"]["train_dataset_identity"],
        "validation_dataset_identity": payload["dataset"][
            "validation_dataset_identity"
        ],
        "summary": payload["summary"],
        "checkpoints": {"best": str(trainer.best_path)},
        "run_record": str(trainer.run_record_path),
    }
    _emit(result, log_file=args.log_file)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    cache = load_prepared_cache(args.dataset)
    _validate_report_paths(
        (("--output", args.output), ("--log-file", args.log_file)),
        cache.root,
        checkpoint=args.checkpoint,
    )
    dataset = cache.load_split(args.split)
    report = evaluate_checkpoint(
        args.checkpoint,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    result = {
        "command": "evaluate",
        "package_version": PACKAGE_VERSION,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "dataset": _cache_metadata(cache),
        "dataset_identity": dataset.identity.to_dict(),
        "metrics": report.to_dict(),
    }
    _emit(result, output=args.output, log_file=args.log_file)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    loaded = load_policy_checkpoint(args.checkpoint, device=args.device)
    _emit({"command": "inspect", **loaded.metadata()})
    return 0


def _cmd_cache_info(args: argparse.Namespace) -> int:
    cache = load_prepared_cache(args.dataset)
    _validate_report_paths((("--output", args.output),), cache.root)
    result = {
        "command": "cache-info",
        "package_version": PACKAGE_VERSION,
        "root": str(cache.root),
        "format": cache.manifest["format"],
        "cache_id": cache.cache_id,
        "encoding_version": cache.encoding_version,
        "preparation_format_version": cache.preparation_format_version,
        "preparation_version": cache.preparation_version,
        "source_revision": cache.source_revision,
        "corpus_manifest_sha256": cache.corpus_manifest_sha256,
        "split_algorithm": cache.split_algorithm,
        "counts": cache.counts.to_dict(),
        "splits": {
            name: {
                "counts": cache.splits[name].counts.to_dict(),
                "shards": len(cache.splits[name].shards),
            }
            for name in SPLIT_NAMES
        },
    }
    _emit(result, output=args.output)
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    _emit(
        {
            "package": "mahjong-bc",
            "package_version": PACKAGE_VERSION,
            "dependencies": collect_dependency_versions().to_dict(),
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "cache-info":
            return _cmd_cache_info(args)
        if args.command == "version":
            return _cmd_version(args)
    except MahjongBCError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
