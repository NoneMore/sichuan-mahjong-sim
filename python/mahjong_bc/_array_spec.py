"""Single source of truth for the prepared packed-array wire protocol.

The Rust cache writer fixes array names, dtypes, manifest layout dimensions and
numeric shapes. The Python cache reader and packed-array validator derive their
views from this specification instead of restating those protocol facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import CANDIDATE_WIDTH, STATE_WIDTH


@dataclass(frozen=True)
class ArraySpec:
    """One packed wire array and its manifest representation."""

    name: str
    dtype: str
    dimensions: tuple[str, ...]

    def shape(self, decisions: int, candidates: int) -> tuple[int, ...]:
        """Resolve the concrete shape for one decision and candidate count."""

        return tuple(
            _dimension_value(dimension, decisions, candidates)
            for dimension in self.dimensions
        )


def _dimension_value(dimension: str, decisions: int, candidates: int) -> int:
    if dimension == "decisions":
        return decisions
    if dimension == "decisions + 1":
        return decisions + 1
    if dimension == "candidates":
        return candidates
    if dimension == "state_features":
        return STATE_WIDTH
    if dimension == "candidate_features":
        return CANDIDATE_WIDTH
    raise AssertionError(f"unknown packed-array dimension {dimension!r}")


ARRAY_SPECS: tuple[ArraySpec, ...] = (
    ArraySpec("states", "float32", ("decisions", "state_features")),
    ArraySpec(
        "candidate_features",
        "float32",
        ("candidates", "candidate_features"),
    ),
    ArraySpec("candidate_offsets", "int64", ("decisions + 1",)),
    ArraySpec("targets", "int64", ("decisions",)),
    ArraySpec("episode_seeds", "uint64", ("decisions",)),
    ArraySpec("decision_indices", "uint64", ("decisions",)),
    ArraySpec("actor_seats", "uint8", ("decisions",)),
    ArraySpec("stages", "uint8", ("decisions",)),
    ArraySpec("target_kinds", "uint8", ("decisions",)),
)

ARRAY_NAMES = tuple(spec.name for spec in ARRAY_SPECS)
ARRAY_DTYPES = {spec.name: spec.dtype for spec in ARRAY_SPECS}
ARRAY_DIMENSIONS = {spec.name: spec.dimensions for spec in ARRAY_SPECS}
