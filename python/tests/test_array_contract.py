"""Regression tests for the shared packed-array contract.

The validator itself is private, but both public entry points must translate
its violations into their own errors: direct packed construction raises
``DataError`` and a corrupt prepared cache raises ``CacheContractError`` with
the offending array file.
"""

from __future__ import annotations

import numpy as np
import pytest

from mahjong_bc._array_contract import (
    ArrayContractViolation,
    expected_shapes,
    validate_packed_arrays,
)
from mahjong_bc.cache import load_prepared_cache
from mahjong_bc.constants import (
    ACTION_KIND_COUNT,
    CANDIDATE_WIDTH,
    STAGE_COUNT,
    STATE_WIDTH,
)
from mahjong_bc.data import PackedDecisionDataset
from mahjong_bc.errors import CacheContractError, DataError

from .cache_fixture import read_manifest, rewrite_cache_array


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "states": np.zeros((2, STATE_WIDTH), dtype=np.float32),
        "candidate_features": np.zeros(
            (3, CANDIDATE_WIDTH), dtype=np.float32
        ),
        "candidate_offsets": np.array([0, 1, 3], dtype=np.int64),
        "targets": np.array([0, 1], dtype=np.int64),
        "stages": np.array([0, STAGE_COUNT - 1], dtype=np.uint8),
        "actor_seats": np.array([0, 3], dtype=np.uint8),
        "decision_indices": np.array([0, 1], dtype=np.uint64),
        "target_kinds": np.array(
            [0, ACTION_KIND_COUNT - 1], dtype=np.uint8
        ),
    }


def _empty_offset_span(values: np.ndarray) -> np.ndarray:
    values[1] = values[0]
    return values


def _negative_target(values: np.ndarray) -> np.ndarray:
    values[0] = -1
    return values


def _stage_out_of_range(values: np.ndarray) -> np.ndarray:
    values[0] = STAGE_COUNT
    return values


def _seat_out_of_range(values: np.ndarray) -> np.ndarray:
    values[0] = 4
    return values


def _kind_out_of_range(values: np.ndarray) -> np.ndarray:
    values[0] = ACTION_KIND_COUNT
    return values


_CORRUPTIONS = {
    "candidate_offsets": _empty_offset_span,
    "targets": _negative_target,
    "stages": _stage_out_of_range,
    "actor_seats": _seat_out_of_range,
    "target_kinds": _kind_out_of_range,
}


def _packed(arrays: dict[str, np.ndarray]) -> PackedDecisionDataset:
    return PackedDecisionDataset(
        states=arrays["states"],
        candidate_features=arrays["candidate_features"],
        offsets=arrays["candidate_offsets"],
        targets=arrays["targets"],
        stages=arrays["stages"],
        identity=None,
        actor_seats=arrays["actor_seats"],
        decision_indices=arrays["decision_indices"],
        target_kinds=arrays["target_kinds"],
    )


def test_expected_shapes_cover_the_prepared_wire_arrays():
    assert expected_shapes(3, 7) == {
        "states": (3, STATE_WIDTH),
        "candidate_features": (7, CANDIDATE_WIDTH),
        "candidate_offsets": (4,),
        "targets": (3,),
        "episode_seeds": (3,),
        "decision_indices": (3,),
        "actor_seats": (3,),
        "stages": (3,),
        "target_kinds": (3,),
    }


def test_validator_derives_eligibility_and_read_only_views():
    layout = validate_packed_arrays(**_valid_arrays())
    assert layout.decisions == 2
    assert layout.candidates == 3
    assert layout.candidate_counts.tolist() == [1, 2]
    assert layout.eligible_mask.tolist() == [False, True]
    assert layout.eligible_indices.tolist() == [1]
    assert layout.forced_count == 1
    assert layout.eligible_count == 1
    for values in (
        layout.candidate_counts,
        layout.eligible_mask,
        layout.eligible_indices,
    ):
        with pytest.raises(ValueError):
            values[0] = 0


def test_validator_allows_absent_optional_metadata():
    arrays = _valid_arrays()
    for name in ("actor_seats", "decision_indices", "target_kinds"):
        del arrays[name]
    layout = validate_packed_arrays(**arrays)
    assert layout.eligible_indices.tolist() == [1]


@pytest.mark.parametrize(
    "name", list(_CORRUPTIONS), ids=list(_CORRUPTIONS)
)
def test_validator_rejects_bad_offset_target_stage_seat_and_kind_values(name):
    arrays = _valid_arrays()
    arrays[name] = _CORRUPTIONS[name](arrays[name])
    with pytest.raises(ArrayContractViolation) as excinfo:
        validate_packed_arrays(**arrays)
    assert excinfo.value.array == name


@pytest.mark.parametrize(
    "name", list(_CORRUPTIONS), ids=list(_CORRUPTIONS)
)
def test_packed_construction_translates_violations_to_data_error(name):
    arrays = _valid_arrays()
    arrays[name] = _CORRUPTIONS[name](arrays[name])
    with pytest.raises(DataError):
        _packed(arrays)


@pytest.mark.parametrize(
    "name", list(_CORRUPTIONS), ids=list(_CORRUPTIONS)
)
def test_cache_keeps_file_context_for_shared_contract_violations(
    fresh_cache, name
):
    root = fresh_cache.root
    manifest = read_manifest(root)
    relative = manifest["splits"]["train"]["shards"][0]["arrays"][name]["file"]
    values = np.load(root / relative)
    values = _CORRUPTIONS[name](values)
    relative = rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name=name,
        values=values,
    )
    with pytest.raises(CacheContractError) as excinfo:
        load_prepared_cache(root)
    assert relative in str(excinfo.value)
