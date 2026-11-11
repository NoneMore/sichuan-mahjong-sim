"""Shared structural contract for packed decision arrays.

Direct :class:`~mahjong_bc.data.PackedDecisionDataset` construction and
prepared-cache loading validate the same packed layout through
:func:`validate_packed_arrays`, so the two entry points cannot drift apart.
Cache-only manifest, inventory, checksum, episode, and target-kind one-hot
checks stay in :mod:`mahjong_bc.cache`.

Callers translate :class:`ArrayContractViolation` into their own error type:
``DataError`` for direct dataset construction and ``CacheContractError`` with
file context for a corrupt cache. The violation carries the offending wire
array name to keep that localization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._array_spec import ARRAY_SPECS
from .constants import (
    ACTION_KIND_COUNT,
    CANDIDATE_WIDTH,
    FORCED_CANDIDATE_COUNT,
    STAGE_COUNT,
    STATE_WIDTH,
)


class ArrayContractViolation(ValueError):
    """A packed-layout violation awaiting a caller-specific error type."""

    def __init__(self, message: str, *, array: str) -> None:
        super().__init__(message)
        self.array = array


@dataclass(frozen=True)
class PackedArrayLayout:
    """Counts and eligibility spans derived from validated packed arrays."""

    decisions: int
    candidates: int
    candidate_counts: np.ndarray
    eligible_mask: np.ndarray
    eligible_indices: np.ndarray

    @property
    def eligible_count(self) -> int:
        return int(self.eligible_indices.size)

    @property
    def forced_count(self) -> int:
        return self.decisions - self.eligible_count


def expected_shapes(decisions: int, candidates: int) -> dict[str, tuple[int, ...]]:
    """Return the wire-array shapes implied by decision and candidate counts."""

    return {spec.name: spec.shape(decisions, candidates) for spec in ARRAY_SPECS}


def _require_array(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ArrayContractViolation(
            f"{name} must be a NumPy array", array=name
        )
    return value


def _require_shape(
    value: np.ndarray, name: str, shape: tuple[int, ...]
) -> None:
    if value.shape != shape:
        raise ArrayContractViolation(
            f"{name} must have shape {shape}, got {value.shape}", array=name
        )


def _require_integer_array(
    value: object, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _require_array(value, name)
    _require_shape(array, name, shape)
    if not np.issubdtype(array.dtype, np.integer):
        raise ArrayContractViolation(
            f"{name} must have an integer dtype, got {array.dtype}", array=name
        )
    return array


def _require_float32_array(
    value: object, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _require_array(value, name)
    _require_shape(array, name, shape)
    if array.dtype != np.float32:
        raise ArrayContractViolation(
            f"{name} must be float32, got {array.dtype}", array=name
        )
    return array


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def validate_packed_arrays(
    *,
    states: object,
    candidate_features: object,
    candidate_offsets: object,
    targets: object,
    stages: object,
    actor_seats: object | None = None,
    decision_indices: object | None = None,
    target_kinds: object | None = None,
) -> PackedArrayLayout:
    """Validate one packed decision layout and derive its eligibility spans.

    Required arrays follow the prepared-cache wire names. Optional metadata
    arrays are validated when present; target-kind one-hot agreement and
    episode structure remain caller-specific checks.
    """

    states = _require_array(states, "states")
    if states.ndim != 2:
        raise ArrayContractViolation(
            "states must be a two-dimensional array", array="states"
        )
    decisions = int(states.shape[0])
    _require_float32_array(states, "states", (decisions, STATE_WIDTH))

    candidate_features = _require_array(candidate_features, "candidate_features")
    if candidate_features.ndim != 2:
        raise ArrayContractViolation(
            "candidate_features must be a two-dimensional array",
            array="candidate_features",
        )
    candidates = int(candidate_features.shape[0])
    _require_float32_array(
        candidate_features,
        "candidate_features",
        (candidates, CANDIDATE_WIDTH),
    )

    candidate_offsets = _require_integer_array(
        candidate_offsets, "candidate_offsets", (decisions + 1,)
    )
    targets = _require_integer_array(targets, "targets", (decisions,))
    stages = _require_integer_array(stages, "stages", (decisions,))

    if int(candidate_offsets[0]) != 0:
        raise ArrayContractViolation(
            "candidate_offsets must start at zero", array="candidate_offsets"
        )
    if int(candidate_offsets[-1]) != candidates:
        raise ArrayContractViolation(
            f"candidate_offsets must end at the candidate count {candidates}, "
            f"got {int(candidate_offsets[-1])}",
            array="candidate_offsets",
        )
    candidate_counts = np.diff(
        candidate_offsets.astype(np.int64, copy=False)
    )
    if np.any(candidate_counts <= 0):
        raise ArrayContractViolation(
            "every decision must have at least one candidate",
            array="candidate_offsets",
        )

    local_targets = targets.astype(np.int64, copy=False)
    if np.any(local_targets < 0) or np.any(local_targets >= candidate_counts):
        raise ArrayContractViolation(
            "every target must index its own candidate range", array="targets"
        )

    stage_values = stages.astype(np.int64, copy=False)
    if np.any(stage_values < 0) or np.any(stage_values >= STAGE_COUNT):
        raise ArrayContractViolation(
            f"stages must be in [0, {STAGE_COUNT})", array="stages"
        )

    if actor_seats is not None:
        actor_seats = _require_integer_array(
            actor_seats, "actor_seats", (decisions,)
        )
        seats = actor_seats.astype(np.int64, copy=False)
        if np.any(seats < 0) or np.any(seats > 3):
            raise ArrayContractViolation(
                "actor_seats must be in [0, 4)", array="actor_seats"
            )
    if decision_indices is not None:
        decision_indices = _require_integer_array(
            decision_indices, "decision_indices", (decisions,)
        )
        if np.any(decision_indices.astype(np.int64, copy=False) < 0):
            raise ArrayContractViolation(
                "decision_indices must be non-negative", array="decision_indices"
            )
    if target_kinds is not None:
        target_kinds = _require_integer_array(
            target_kinds, "target_kinds", (decisions,)
        )
        kinds = target_kinds.astype(np.int64, copy=False)
        if np.any(kinds < 0) or np.any(kinds >= ACTION_KIND_COUNT):
            raise ArrayContractViolation(
                f"target_kinds must be in [0, {ACTION_KIND_COUNT})",
                array="target_kinds",
            )

    eligible_mask = candidate_counts > FORCED_CANDIDATE_COUNT
    eligible_indices = np.flatnonzero(eligible_mask)
    return PackedArrayLayout(
        decisions=decisions,
        candidates=candidates,
        candidate_counts=_read_only(candidate_counts),
        eligible_mask=_read_only(eligible_mask),
        eligible_indices=_read_only(eligible_indices),
    )
