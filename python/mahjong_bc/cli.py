"""Command line interface for offline behavior-cloning training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .cache import load_prepared_cache
from .checkpoint import atomic_write_json, load_checkpoint, load_policy_checkpoint
from .config import (
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from .constants import PACKAGE_VERSION, SPLIT_NAMES
from .errors import MahjongBCError
from .train import Trainer, evaluate_checkpoint


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
        description="Offline behavior-cloning policy training for Sichuan mahjong",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="train from a prepared cache, or resume a recoverable last checkpoint",
    )
    train.add_argument("--dataset", required=True, help="prepared cache root")
    train.add_argument("--output", required=True, help="checkpoint output directory")
    train.add_argument(
        "--resume",
        help="resume from a last checkpoint; model/training configuration comes "
        "from the checkpoint and output/log paths may change",
    )
    train.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    train.add_argument("--seed", type=int, help="new-run training seed")
    train.add_argument("--epochs", type=int, help="new-run epoch count")
    train.add_argument("--batch-size", type=int, help="new-run batch size")
    train.add_argument("--learning-rate", type=float, help="new-run AdamW learning rate")
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--beta1", type=float)
    train.add_argument("--beta2", type=float)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--state-hidden", type=_parse_widths, help="e.g. 256,128")
    train.add_argument("--scorer-hidden", type=_parse_widths, help="e.g. 128")
    train.add_argument(
        "--activation", choices=("relu", "gelu", "tanh"), help="new-run activation"
    )
    train.add_argument(
        "--log-file", help="optional JSON summary destination; may change on resume"
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a frozen checkpoint on a validation/test split"
    )
    evaluate.add_argument("--dataset", required=True, help="prepared cache root")
    evaluate.add_argument("--checkpoint", required=True, help="policy checkpoint")
    evaluate.add_argument(
        "--split", required=True, choices=("validation", "test"), help="split to evaluate"
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
        "cache-info", help="validate a prepared cache and print its inventory counts"
    )
    cache_info.add_argument("--dataset", required=True)
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
            f"a new training run requires {', '.join(missing)}; "
            "use --resume to continue from a last checkpoint"
        )
    training = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.0 if args.weight_decay is None else args.weight_decay,
        beta1=0.9 if args.beta1 is None else args.beta1,
        beta2=0.999 if args.beta2 is None else args.beta2,
        num_workers=0 if args.num_workers is None else args.num_workers,
    )
    model = ModelConfig(
        state_hidden=(256,) if args.state_hidden is None else args.state_hidden,
        scorer_hidden=(128,) if args.scorer_hidden is None else args.scorer_hidden,
        activation="relu" if args.activation is None else args.activation,
    )
    return model, training


def _resume_config(args: argparse.Namespace) -> tuple[ModelConfig, TrainingConfig]:
    conflicting = [
        name
        for name, value in (
            ("--seed", args.seed),
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--learning-rate", args.learning_rate),
            ("--weight-decay", args.weight_decay),
            ("--beta1", args.beta1),
            ("--beta2", args.beta2),
            ("--num-workers", args.num_workers),
            ("--state-hidden", args.state_hidden),
            ("--scorer-hidden", args.scorer_hidden),
            ("--activation", args.activation),
        )
        if value is not None
    ]
    if conflicting:
        raise MahjongBCError(
            "--resume uses the checkpoint's model/training configuration; "
            f"remove {', '.join(conflicting)}"
        )
    loaded = load_checkpoint(args.resume, device="cpu", require_training=True)
    state = loaded.training
    assert state is not None
    return loaded.model_config, state.training_config


def _emit(value: Any, *, output: str | None = None, log_file: str | None = None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    if output:
        atomic_write_json(output, value)
    if log_file:
        atomic_write_json(log_file, value)
    print(text)


def _cmd_train(args: argparse.Namespace) -> int:
    if args.resume:
        model_config, training_config = _resume_config(args)
    else:
        model_config, training_config = _new_run_config(args)
    cache = load_prepared_cache(args.dataset)
    train_dataset = cache.load_split("train")
    validation_dataset = cache.load_split("validation")
    trainer = Trainer(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output,
        device=args.device,
    )
    summary = trainer.train(resume_from=args.resume)
    result = {
        "command": "train",
        "package_version": PACKAGE_VERSION,
        "dataset": {
            "cache_id": cache.cache_id,
            "encoding_version": cache.encoding_version,
            "preparation_format_version": cache.preparation_format_version,
            "preparation_version": cache.preparation_version,
            "source_revision": cache.source_revision,
            "corpus_manifest_sha256": cache.corpus_manifest_sha256,
        },
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "dependencies": collect_dependency_versions().to_dict(),
        "device_type": trainer.device.type,
        "train_dataset_identity": train_dataset.identity.to_dict(),
        "validation_dataset_identity": validation_dataset.identity.to_dict(),
        "summary": summary.to_dict(),
        "checkpoints": {
            "best": str(trainer.best_path),
            "last": str(trainer.last_path),
        },
        "run_record": str(trainer.run_record_path),
    }
    _emit(result, log_file=args.log_file)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    cache = load_prepared_cache(args.dataset)
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
        "dataset": {
            "cache_id": cache.cache_id,
            "encoding_version": cache.encoding_version,
            "preparation_format_version": cache.preparation_format_version,
            "preparation_version": cache.preparation_version,
            "source_revision": cache.source_revision,
            "corpus_manifest_sha256": cache.corpus_manifest_sha256,
        },
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
    result = {
        "command": "cache-info",
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
