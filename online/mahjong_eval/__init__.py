"""Online environment binding for frozen-policy evaluation.

The native extension keeps authoritative game state in Rust. `Env.reset(seed)`
returns the first pending decision with owned float32 NumPy snapshots produced
by the frozen v1 encoder. `step` and terminal results are added by the full
environment-lifecycle slice.

Stage codes are the frozen v1 order: 0 exchange, 1 missing suit, 2 turn,
3 discard response, 4 added-kong response.
"""

from __future__ import annotations

from . import _native
from ._native import (
    CANDIDATE_WIDTH,
    DEFAULT_DECISION_LIMIT,
    ENCODING_VERSION,
    STATE_WIDTH,
    Decision,
    Env,
)

__version__ = _native.__version__

__all__ = [
    "CANDIDATE_WIDTH",
    "DEFAULT_DECISION_LIMIT",
    "Decision",
    "ENCODING_VERSION",
    "Env",
    "STATE_WIDTH",
    "__version__",
]
