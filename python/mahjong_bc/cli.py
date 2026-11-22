"""Command line interface for offline behavior-cloning training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ._input_pipeline import BatchTransferConfig
from .cache import load_prepared_cache
from .checkpoint import (
    LoadedCheckpoint,
    atomic_write_json,
    load_policy_checkpoint,
)
from .config import (
    ACTIVATION_NAMES,
    MODEL_ARCHITECTURES,
    ModelConfig,
    TrainingConfig,
    collect_dependency_versions,
)
from .constants import PACKAGE_VERSION, SPLIT_NAMES
from .errors import ConfigurationError, MahjongBCError
from .train import (
    TRAIN_OUTPUT_NAMES,
    Trainer,
    evaluate_checkpoint,
    load_resume_checkpoint,
)

_REQUIRED_RUN_OPTIONS = ("seed", "epochs", "batch_size", "learning_rate")
_TRAINING_OPTION_NAMES = ("weight_decay", "beta1", "beta2", "num_workers")
_MODEL_OPTION_NAMES = ("state_hidden", "scorer_hidden", "activation", "architecture")
_RESUME_FORBIDDEN_OPTIONS = (
    *_REQUIRED_RUN_OPTIONS,
    *_TRAINING_OPTION_NAMES,
    *_MODEL_OPTION_NAMES,
)
_MIB = 2**20
_DATA_STORAGE_CHOICES = ("mmap", "ram")
_BATCH_TRANSFER_CHOICES = ("sync", "cuda-prefetch")


@dataclass(frozen=True)
class _SupplyOptions:
    """One invocation's cache-loading and batch-transfer choices."""

    storage: str
    ram_limit_bytes: int | None
    batch_transfer: BatchTransferConfig


def _positive_mib(text: str) -> int:
    """Parse an exact positive MiB budget."""

    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of MiB, got {text!r}"
        ) from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 MiB, got {value}")
    return value


def _add_supply_options(parser: argparse.ArgumentParser) -> None:
    """Add the independent storage and transfer choices shared by commands."""

    parser.add_argument(
        "--data-storage",
        choices=_DATA_STORAGE_CHOICES,
        default="mmap",
        help="prepared-cache split representation (default: mmap)",
    )
    parser.add_argument(
        "--ram-limit-mib",
        type=_positive_mib,
        help="RAM budget in MiB; required by, and only valid for, "
        "--data-storage ram",
    )
    parser.add_argument(
        "--batch-transfer",
        choices=_BATCH_TRANSFER_CHOICES,
        default="sync",
        help="device-batch transfer (default: sync)",
    )
    parser.add_argument(
        "--pinned-limit-mib",
        type=_positive_mib,
        help="pinned slot budget in MiB; required by, and only valid for, "
        "--batch-transfer cuda-prefetch",
    )


def _resolve_supply_options(args: argparse.Namespace) -> _SupplyOptions:
    """Validate supply options and convert their MiB budgets to bytes."""

    if args.data_storage == "ram":
        if args.ram_limit_mib is None:
            raise ConfigurationError(
                "--data-storage ram requires --ram-limit-mib"
            )
        ram_limit_bytes = args.ram_limit_mib * _MIB
    elif args.ram_limit_mib is not None:
        raise ConfigurationError(
            "--ram-limit-mib requires --data-storage ram"
        )
    else:
        ram_limit_bytes = None
    if args.batch_transfer == "cuda-prefetch":
        if args.pinned_limit_mib is None:
            raise ConfigurationError(
                "--batch-transfer cuda-prefetch requires --pinned-limit-mib"
            )
        pinned_limit_bytes = args.pinned_limit_mib * _MIB
    elif args.pinned_limit_mib is not None:
        raise ConfigurationError(
            "--pinned-limit-mib requires --batch-transfer cuda-prefetch"
        )
    else:
        pinned_limit_bytes = None
    return _SupplyOptions(
        storage=args.data_storage,
        ram_limit_bytes=ram_limit_bytes,
        batch_transfer=BatchTransferConfig(
            mode=args.batch_transfer, pinned_limit_bytes=pinned_limit_bytes
        ),
    )


def _require_supply_workers(supply: _SupplyOptions, num_workers: int) -> None:
    """Continuous RAM and CUDA prefetch both require in-process batching."""

    if num_workers == 0:
        return
    if supply.storage == "ram":
        raise ConfigurationError(
            "continuous RAM loading requires num_workers=0, "
            f"got {num_workers}"
        )
    if supply.batch_transfer.mode != "sync":
        raise ConfigurationError(
            f"CUDA batch prefetch requires num_workers=0, got {num_workers}"
        )


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
        help="train a policy, or resume a recoverable last checkpoint",
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
    train.add_argument(
        "--architecture", choices=MODEL_ARCHITECTURES, help="new-run state encoder"
    )
    train.add_argument("--state-hidden", type=_parse_widths, help="e.g. 256,128")
    train.add_argument("--scorer-hidden", type=_parse_widths, help="e.g. 128")
    train.add_argument(
        "--activation", choices=ACTIVATION_NAMES, help="new-run activation"
    )
    train.add_argument(
        "--log-file", help="optional JSON summary destination"
    )
    _add_supply_options(train)

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
    _add_supply_options(evaluate)

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
        f"--{name.replace('_', '-')}"
        for name in _REQUIRED_RUN_OPTIONS
        if getattr(args, name) is None
    ]
    if missing:
        raise MahjongBCError(
            f"a fresh training run requires {', '.join(missing)}; "
            "use --resume to continue from a last checkpoint"
        )
    training_values: dict[str, Any] = {
        name: getattr(args, name) for name in _REQUIRED_RUN_OPTIONS
    }
    for name in _TRAINING_OPTION_NAMES:
        value = getattr(args, name)
        if value is not None:
            training_values[name] = value
    model_values: dict[str, Any] = {}
    for name in _MODEL_OPTION_NAMES:
        value = getattr(args, name)
        if value is not None:
            model_values[name] = value
    return ModelConfig(**model_values), TrainingConfig(**training_values)


def _require_no_resume_overrides(args: argparse.Namespace) -> None:
    conflicting = [
        f"--{name.replace('_', '-')}"
        for name in _RESUME_FORBIDDEN_OPTIONS
        if getattr(args, name) is not None
    ]
    if conflicting:
        raise MahjongBCError(
            "--resume uses the checkpoint's model/training configuration; "
            f"remove {', '.join(conflicting)}"
        )


def _resume_config(
    loaded: LoadedCheckpoint,
) -> tuple[ModelConfig, TrainingConfig]:
    state = loaded.training
    assert state is not None
    return loaded.model_config, state.training_config


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
    supply = _resolve_supply_options(args)
    resume = None
    if args.resume:
        _require_no_resume_overrides(args)
        resume = load_resume_checkpoint(args.resume)
        model_config, training_config = _resume_config(resume)
    else:
        model_config, training_config = _new_run_config(args)
    _require_supply_workers(supply, training_config.num_workers)
    cache = load_prepared_cache(args.dataset)
    _validate_train_paths(args.output, args.log_file, cache.root)
    dataset_metadata = _cache_metadata(cache)
    # The loading context is the last resource to close: dataset refresh and
    # batch production both finish before training publishes a checkpoint.
    with cache.open_splits(
        ("train", "validation"),
        storage=supply.storage,
        ram_limit_bytes=supply.ram_limit_bytes,
    ) as inputs:
        provenance = dict(dataset_metadata)
        provenance["input_loading"] = inputs.loading_record
        trainer = Trainer(
            train_dataset=inputs.datasets["train"],
            validation_dataset=inputs.datasets["validation"],
            model_config=model_config,
            training_config=training_config,
            output_dir=args.output,
            device=args.device,
            dataset_provenance=provenance,
            batch_transfer=supply.batch_transfer,
        )
        summary = trainer.train(resume_from=resume)
        payload = trainer.run_payload(summary)
        result = {
            "command": "train",
            "package_version": payload["package_version"],
            "source": payload["source"],
            "dataset": dataset_metadata,
            "input_loading": inputs.loading_record,
            "batch_transfer": payload["batch_transfer"],
            "model_config": payload["model_config"],
            "training_config": payload["training_config"],
            "dependencies": payload["dependencies"],
            "device_type": payload["device_type"],
            "train_dataset_identity": payload["dataset"]["train_dataset_identity"],
            "validation_dataset_identity": payload["dataset"][
                "validation_dataset_identity"
            ],
            "summary": payload["summary"],
            "checkpoints": {
                "best": str(trainer.best_path),
                "last": str(trainer.last_path),
            },
            "run_record": str(trainer.run_record_path),
        }
    _emit(result, log_file=args.log_file)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    supply = _resolve_supply_options(args)
    _require_supply_workers(supply, args.num_workers)
    cache = load_prepared_cache(args.dataset)
    _validate_report_paths(
        (("--output", args.output), ("--log-file", args.log_file)),
        cache.root,
        checkpoint=args.checkpoint,
    )
    with cache.open_splits(
        (args.split,),
        storage=supply.storage,
        ram_limit_bytes=supply.ram_limit_bytes,
    ) as inputs:
        dataset = inputs.datasets[args.split]
        report = evaluate_checkpoint(
            args.checkpoint,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            batch_transfer=supply.batch_transfer,
        )
        result = {
            "command": "evaluate",
            "package_version": PACKAGE_VERSION,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "dataset": _cache_metadata(cache),
            "dataset_identity": dataset.identity.to_dict(),
            "input_loading": inputs.loading_record,
            "batch_transfer": supply.batch_transfer.to_dict(),
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
