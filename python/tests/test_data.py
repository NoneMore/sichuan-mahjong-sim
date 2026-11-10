from __future__ import annotations

import numpy as np
import pytest
import torch

from mahjong_bc.constants import CANDIDATE_WIDTH, STATE_WIDTH
from mahjong_bc.data import (
    PackedDecisionDataset,
    collate_decisions,
    derive_target_kind,
    iter_batches,
)
from mahjong_bc.errors import DataError

from .helpers import identity, make_dataset


def test_packed_dataset_exposes_layout_and_metadata():
    dataset = make_dataset(
        split="validation",
        counts=[2, 1, 3, 2],
        seed=4,
        cache_id="data-cache",
    )
    assert len(dataset) == 4
    assert dataset.eligible_count == 3
    assert dataset.forced_count == 1
    assert np.array_equal(dataset.eligible_indices(), np.array([0, 2, 3]))
    assert dataset.forced_by_stage() == tuple(
        int(np.count_nonzero(dataset.stages[~dataset.eligible_mask()] == stage))
        for stage in range(5)
    )
    sample = dataset[2]
    assert sample.candidate_count == 3
    assert sample.eligible is True
    assert sample.target_kind == int(dataset.target_kinds[2])
    assert sample.state.shape == (STATE_WIDTH,)
    assert sample.candidate_features.shape == (3, CANDIDATE_WIDTH)
    assert dataset.identity.cache_id == "data-cache"


def test_eligibility_and_count_accessors_are_read_only():
    dataset = make_dataset(split="train", counts=[2, 1, 3], seed=5)
    for accessor in (
        dataset.eligible_indices,
        dataset.eligible_mask,
        dataset.candidate_counts,
    ):
        values = accessor()
        with pytest.raises(ValueError):
            values[0] = 0


def test_target_kind_is_derived_when_metadata_is_absent():
    dataset = make_dataset(split="train", counts=[2, 2], seed=5)
    derived = PackedDecisionDataset(
        states=dataset.states,
        candidate_features=dataset.candidate_features,
        offsets=dataset.offsets,
        targets=dataset.targets,
        stages=dataset.stages,
        identity=dataset.identity,
    )
    for index in range(len(dataset)):
        assert derived[index].target_kind == int(dataset.target_kinds[index])


def test_derive_target_kind_rejects_ambiguous_kind_features():
    features = np.zeros((2, CANDIDATE_WIDTH), dtype=np.float32)
    features[0, 0] = 1.0
    assert derive_target_kind(features, 0) == 0
    with pytest.raises(DataError):
        derive_target_kind(features, 1)
    features[1, 0] = 0.5
    with pytest.raises(DataError):
        derive_target_kind(features, 1)
    features[1, 0] = 1.0
    features[1, 1] = 1.0
    with pytest.raises(DataError):
        derive_target_kind(features, 1)


def test_collation_pads_to_batch_max_and_masks_padding():
    dataset = make_dataset(split="train", counts=[1, 3, 2], seed=6)
    samples = [dataset[index] for index in range(3)]
    batch = collate_decisions(samples)
    assert batch.states.shape == (3, STATE_WIDTH)
    assert batch.candidates.shape == (3, 3, CANDIDATE_WIDTH)
    assert batch.mask.tolist() == [[True, False, False], [True, True, True], [True, True, False]]
    assert torch.count_nonzero(batch.candidates[0, 1:]) == 0
    assert batch.eligible.tolist() == [False, True, True]
    assert batch.targets.tolist() == [int(value) for value in dataset.targets]
    assert batch.stages.tolist() == [int(value) for value in dataset.stages]
    assert batch.target_kinds.tolist() == [int(value) for value in dataset.target_kinds]
    moved = batch.to("cpu")
    assert torch.equal(moved.states, batch.states)


def test_iter_batches_preserves_order_and_final_partial_batch():
    dataset = make_dataset(split="train", counts=[2, 2, 2, 2, 2], seed=7)
    indices = np.array([4, 1, 3, 0, 2])
    batches = list(iter_batches(dataset, indices, batch_size=2))
    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    ordered_targets = [int(value) for batch in batches for value in batch.targets]
    expected = [int(dataset.targets[index]) for index in indices]
    assert ordered_targets == expected


def test_dataset_rejects_invalid_layout_targets_metadata_and_identity():
    dataset = make_dataset(split="train", counts=[2, 2, 3], seed=8)
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states[:, :100],
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
        )
    bad_offsets = dataset.offsets.copy()
    bad_offsets[0] = 1
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=bad_offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
        )
    empty_range = dataset.offsets.copy()
    empty_range[2] = empty_range[1]
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=empty_range,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
        )
    target_outside = dataset.targets.copy()
    target_outside[0] = 9
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=target_outside,
            stages=dataset.stages,
            identity=dataset.identity,
        )
    bad_stage = dataset.stages.copy()
    bad_stage[0] = 5
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=bad_stage,
            identity=dataset.identity,
        )
    bad_kind = dataset.target_kinds.copy()
    bad_kind[0] = 9
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
            target_kinds=bad_kind,
        )
    bad_seat = np.zeros(len(dataset), dtype=np.uint8)
    bad_seat[0] = 4
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
            actor_seats=bad_seat,
        )
    wrong_identity = identity(
        split="train",
        decisions=len(dataset) + 1,
        eligible=dataset.identity.eligible_decisions,
    )
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states,
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=wrong_identity,
        )
    with pytest.raises(DataError):
        PackedDecisionDataset(
            states=dataset.states.astype(np.float64),
            candidate_features=dataset.candidate_features,
            offsets=dataset.offsets,
            targets=dataset.targets,
            stages=dataset.stages,
            identity=dataset.identity,
        )
