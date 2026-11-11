"""Online environment binding for frozen-policy evaluation.

The native extension keeps authoritative game state in Rust. ``Env.reset(seed)``
returns the first pending decision with owned float32 NumPy snapshots produced
by the frozen v1 encoder, and ``Env.step`` advances one cached canonical action
at a time until a terminal settlement. Importing :mod:`mahjong_eval.inference`
does not load the extension, so numeric inference fixtures remain usable in an
offline installation.

Stage codes are the frozen v1 order: 0 exchange, 1 missing suit, 2 turn,
3 discard response, 4 added-kong response. Teacher queries, native teacher
replay, the frozen seed split, and compiled source identity are also exposed
without importing the extension until one of their symbols is used.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from typing import TYPE_CHECKING

_NATIVE_EXPORTS = frozenset(
    {
        "CANDIDATE_WIDTH",
        "DEFAULT_DECISION_LIMIT",
        "DEFAULT_TEACHER_CACHE_BYTES",
        "ENCODING_VERSION",
        "STATE_WIDTH",
        "CandidateIndexError",
        "Decision",
        "DecisionLimitError",
        "Env",
        "EnvironmentStateError",
        "GameRuleError",
        "MahjongEvalError",
        "StaleDecisionError",
        "TeacherCache",
        "Terminal",
        "compiled_source",
        "native_teacher_replay",
        "seed_split",
        "split_bucket",
    }
)

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from ._native import (  # noqa: F401
        CANDIDATE_WIDTH,
        DEFAULT_DECISION_LIMIT,
        DEFAULT_TEACHER_CACHE_BYTES,
        ENCODING_VERSION,
        STATE_WIDTH,
        CandidateIndexError,
        Decision,
        DecisionLimitError,
        Env,
        EnvironmentStateError,
        GameRuleError,
        MahjongEvalError,
        StaleDecisionError,
        TeacherCache,
        Terminal,
        compiled_source,
        native_teacher_replay,
        seed_split,
        split_bucket,
    )


def _native_module():
    """Import and cache the extension only when a native symbol is used."""

    module = importlib.import_module(f"{__name__}._native")
    globals()["_native"] = module
    return module


def __getattr__(name: str):
    if name == "_native":
        return _native_module()
    if name in _NATIVE_EXPORTS:
        return getattr(_native_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = _distribution_version("mahjong-eval")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0.1.0"

__all__ = [
    "CANDIDATE_WIDTH",
    "DEFAULT_DECISION_LIMIT",
    "DEFAULT_TEACHER_CACHE_BYTES",
    "ENCODING_VERSION",
    "STATE_WIDTH",
    "CandidateIndexError",
    "Decision",
    "DecisionLimitError",
    "Env",
    "EnvironmentStateError",
    "GameRuleError",
    "MahjongEvalError",
    "StaleDecisionError",
    "TeacherCache",
    "Terminal",
    "__version__",
    "compiled_source",
    "native_teacher_replay",
    "seed_split",
    "split_bucket",
]
