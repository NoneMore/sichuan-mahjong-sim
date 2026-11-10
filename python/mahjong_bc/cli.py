"""Command line interface for prepared behavior-cloning datasets."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .cache import load_prepared_cache
from .config import collect_dependency_versions
from .constants import PACKAGE_VERSION, SPLIT_NAMES
from .errors import MahjongBCError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mahjong-bc",
        description=(
            "Validate prepared behavior-cloning caches and report package "
            "environment information"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_info = subparsers.add_parser(
        "cache-info",
        help="validate a prepared cache and print its inventory counts",
    )
    cache_info.add_argument("--dataset", required=True, help="prepared cache root")

    subparsers.add_parser("version", help="print package and dependency versions")
    return parser


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_cache_info(args: argparse.Namespace) -> int:
    cache = load_prepared_cache(args.dataset)
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
    _emit(result)
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
