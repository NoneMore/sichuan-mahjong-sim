"""Long-lived JSON Lines worker around the original ninone heuristic.

One-off component of the reproducible policy A/B experiment
(`specs/active/reproducible-policy-ab`), not a runtime dependency of the
simulator. The worker owns transport only: it rebuilds the heuristic's original
request from the local hand, the own missing suit and the adapter's complete
candidate list, imports
`ninone.policies.agent_heuristic.choose_action` from an explicit checkout, calls
it, and reports the returned candidate's stable id. Candidate generation,
ordering, ids and the mapping back to simulator actions stay in the Rust
adapter; this worker never filters or reorders candidates and never falls back
to another action.

Request line (stdin, one JSON object per line):
    {"hand": ["m1", "m2", ...],
     "missing_suit": null | "m" | "p" | "s",
     "candidates": [{"id": "discard:0", "type": "discard", ...}, ...]}
Response line (stdout, one JSON object per line):
    {"id": "<candidate id returned by the heuristic>"}

A protocol or heuristic failure writes one `{"error": "<message>"}` response and
exits nonzero without consuming further requests, because the stream can no
longer be trusted. Startup failures exit nonzero with a stderr message only.
Every diagnostic goes to stderr, so stdout stays a pure response stream; stdout
written by the imported module or the heuristic call is redirected to stderr.
Unknown top-level request keys are ignored: only `hand`, `missing_suit` and
`candidates` reach the heuristic.

One process loads one checkout; use a fresh process to switch `--ninone-root`.

Usage:
    python worker.py --ninone-root ../ninone-mahjong-agent
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import sys
from typing import Any, Callable, Iterable, TextIO

PolicyCall = Callable[[dict], dict]

_TILE_CODE = re.compile(r"[mps][1-9]\Z")
_SUITS = frozenset(("m", "p", "s"))
_HEURISTIC_RELATIVE_PATH = ("ninone", "policies", "agent_heuristic.py")


class WorkerError(ValueError):
    """A protocol or heuristic violation that must invalidate the run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerError(message)


def _error_message(exc: BaseException) -> str:
    return str(exc).strip() or type(exc).__name__


def _tile_codes(value: Any, field: str) -> list[str]:
    _require(isinstance(value, list), f"{field} must be a list")
    tiles: list[str] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, str) and _TILE_CODE.match(item) is not None,
            f"{field}[{index}] must be a tile code such as m1",
        )
        tiles.append(item)
    return tiles


def _validate_candidate(candidate: dict, index: int) -> None:
    """Check the fields the imported heuristic actually reads for known types.

    Validation never drops or rewrites a candidate: an offending known type fails
    the request, and unknown types stay untouched for forward compatibility.
    """
    action_type = candidate["type"]
    if action_type == "exchange_three":
        suit = candidate.get("suit")
        _require(
            isinstance(suit, str) and suit in _SUITS,
            f"candidates[{index}].suit must be 'm', 'p' or 's'",
        )
        kinds = candidate.get("tile_kinds")
        _require(
            isinstance(kinds, list) and len(kinds) == 3,
            f"candidates[{index}].tile_kinds must hold exactly three tile codes",
        )
        for offset, kind in enumerate(kinds):
            _require(
                isinstance(kind, str) and _TILE_CODE.match(kind) is not None,
                f"candidates[{index}].tile_kinds[{offset}] must be a tile code such as m1",
            )
        _require(
            all(kind[0] == suit for kind in kinds),
            f"candidates[{index}].tile_kinds must all belong to suit {suit!r}",
        )
    elif action_type == "choose_missing_suit":
        suit = candidate.get("suit")
        _require(
            isinstance(suit, str) and suit in _SUITS,
            f"candidates[{index}].suit must be 'm', 'p' or 's'",
        )
    elif action_type in ("discard", "pong", "kong", "concealed_kong", "added_kong"):
        code = candidate.get("code")
        _require(
            isinstance(code, str) and _TILE_CODE.match(code) is not None,
            f"candidates[{index}].code must be a tile code such as m1",
        )


def build_request(payload: Any) -> dict:
    """Return the heuristic's original request for one protocol payload.

    The candidate dictionaries and their order are forwarded untouched, because
    the adapter's ordering and ids are part of the experiment's identity and the
    imported heuristic performs its own tie-breaking over them.
    """
    _require(isinstance(payload, dict), "request must be a JSON object")
    _require("hand" in payload, "request is missing hand")
    hand = _tile_codes(payload["hand"], "hand")
    _require("missing_suit" in payload, "request is missing missing_suit")
    missing_suit = payload["missing_suit"]
    _require(
        missing_suit is None
        or (isinstance(missing_suit, str) and missing_suit in _SUITS),
        "missing_suit must be null, 'm', 'p' or 's'",
    )
    candidates = payload.get("candidates")
    _require(
        isinstance(candidates, list) and bool(candidates),
        "candidates must be a non-empty list",
    )
    identifiers: set[str] = set()
    for index, candidate in enumerate(candidates):
        _require(
            isinstance(candidate, dict),
            f"candidates[{index}] must be a JSON object",
        )
        identifier = candidate.get("id")
        _require(
            isinstance(identifier, str) and bool(identifier),
            f"candidates[{index}].id must be a non-empty string",
        )
        _require(
            identifier not in identifiers,
            f"duplicate candidate id {identifier!r}",
        )
        identifiers.add(identifier)
        action_type = candidate.get("type")
        _require(
            isinstance(action_type, str) and bool(action_type),
            f"candidates[{index}].type must be a non-empty string",
        )
        _validate_candidate(candidate, index)
    return {
        "legal_actions": candidates,
        "observation": {"own_hand": hand, "own_missing_suit": missing_suit},
    }


def decide(payload: Any, choose_action: PolicyCall) -> dict:
    """Validate one payload, call the heuristic, and map its choice to an id."""
    request = build_request(payload)
    candidates = request["legal_actions"]
    # Remember the adapter's ids before the call so a mutated candidate id can
    # never redirect the returned choice to another action.
    originals = [(candidate, candidate["id"]) for candidate in candidates]
    with contextlib.redirect_stdout(sys.stderr):
        try:
            selected = choose_action(request)
        except Exception as exc:  # any heuristic failure invalidates the run
            raise WorkerError(f"heuristic raised {type(exc).__name__}: {exc}") from exc
    _require(isinstance(selected, dict), "heuristic returned a non-object selection")
    original_id = next(
        (identifier for candidate, identifier in originals if candidate is selected),
        None,
    )
    _require(original_id is not None, "heuristic returned a selection that is not an original candidate")
    current_id = selected.get("id")
    _require(
        current_id == original_id,
        f"heuristic mutated candidate id {original_id!r} to {current_id!r}",
    )
    return {"id": original_id}


def load_policy(ninone_root: str) -> PolicyCall:
    """Import the original heuristic from an explicit ninone checkout path."""
    root = os.path.realpath(ninone_root)
    expected = os.path.realpath(os.path.join(root, *_HEURISTIC_RELATIVE_PATH))
    _require(os.path.isdir(root), f"ninone root is not a directory: {ninone_root}")
    _require(
        os.path.isfile(expected),
        f"ninone root has no original heuristic: {expected}",
    )
    if root not in sys.path:
        sys.path.insert(0, root)
    # Upstream module-level prints must never be mistaken for protocol lines.
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module("ninone.policies.agent_heuristic")
    actual = os.path.realpath(getattr(module, "__file__", "") or "")
    _require(
        actual == expected,
        f"imported heuristic {actual!r} is not the requested {expected!r}",
    )
    choose_action = getattr(module, "choose_action", None)
    _require(callable(choose_action), f"{expected!r} has no callable choose_action")
    return choose_action


def serve(choose_action: PolicyCall, stdin: Iterable[str], stdout: TextIO) -> int:
    """Serve JSON Lines requests until EOF; return the process exit status."""
    for line in stdin:
        if not line.strip():
            continue
        try:
            response = decide(json.loads(line), choose_action)
        except ValueError as exc:  # WorkerError and JSON decoding errors
            stdout.write(json.dumps({"error": _error_message(exc)}) + "\n")
            stdout.flush()
            return 1
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ninone-root",
        required=True,
        help="path to the ninone-mahjong-agent checkout that hosts the original heuristic",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        choose_action = load_policy(args.ninone_root)
    except Exception as exc:  # a broken checkout must fail loudly, never fall back
        print(f"worker: {_error_message(exc)}", file=sys.stderr, flush=True)
        return 2
    return serve(choose_action, sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
