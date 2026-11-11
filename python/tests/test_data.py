from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mahjong_bc.constants import CANDIDATE_WIDTH, STATE_WIDTH
from mahjong_bc.data import (
    DecisionDataset,
    PackedDecisionDataset,
    ShardedDecisionDataset,
    collate_decisions,
    contains_memory_mapped_arrays,
    derive_target_kind,
    iter_batches,
    pad_decision_inputs,
)
from mahjong_bc.errors import DataError

from .helpers import identity, make_dataset


class _DelegatingDataset(DecisionDataset):
    """Contract-only wrapper with no concrete packed arrays to inspect."""

    def __init__(self, inner: DecisionDataset) -> None:
        self.inner = inner
        self.identity = inner.identity

    @property
    def is_memory_mapped(self) -> bool:
        return self.inner.is_memory_mapped

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        return self.inner[index]

    def eligible_indices(self) -> np.ndarray:
        return self.inner.eligible_indices()

    def forced_by_stage(self) -> tuple[int, ...]:
        return self.inner.forced_by_stage()

    def candidate_counts(self) -> np.ndarray:
        return self.inner.candidate_counts()


class _MissingMmapContract(DecisionDataset):
    """A subclass that omits the required mmap contract."""

    identity = None

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(index)

    def eligible_indices(self) -> np.ndarray:
        return np.empty(0, dtype=np.int64)

    def forced_by_stage(self) -> tuple[int, ...]:
        return (0,) * 5

    def candidate_counts(self) -> np.ndarray:
        return np.empty(0, dtype=np.int64)


def _write_memmap(tmp_path, name: str, values: np.ndarray) -> np.ndarray:
    path = tmp_path / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    return np.load(path, mmap_mode="r")


def _memory_mapped_dataset(tmp_path) -> PackedDecisionDataset:
    source = make_dataset(split="train", counts=[2, 1, 3], seed=45)
    decisions = len(source)
    values = {
        "states": source.states,
        "candidate_features": source.candidate_features,
        "offsets": source.offsets,
        "targets": source.targets,
        "stages": source.stages,
        "actor_seats": np.zeros(decisions, dtype=np.uint8),
        "decision_indices": np.arange(decisions, dtype=np.uint64),
        "episode_ids": np.arange(decisions, dtype=np.uint64),
        "target_kinds": np.asarray(source.target_kinds, dtype=np.uint8),
    }
    mapped = {
        name: _write_memmap(tmp_path, name, value)
        for name, value in values.items()
    }
    return PackedDecisionDataset(
        states=mapped["states"],
        candidate_features=mapped["candidate_features"],
        offsets=mapped["offsets"],
        targets=mapped["targets"],
        stages=mapped["stages"],
        identity=source.identity,
        actor_seats=mapped["actor_seats"],
        decision_indices=mapped["decision_indices"],
        episode_ids=mapped["episode_ids"],
        target_kinds=mapped["target_kinds"],
    )


def test_pad_decision_inputs_matches_offline_collation_inputs():
    dataset = make_dataset(split="train", counts=[1, 3, 2], seed=61)
    samples = [dataset[index] for index in range(3)]
    inputs = pad_decision_inputs(samples)
    batch = collate_decisions(samples)
    assert torch.equal(inputs.states, batch.states)
    assert torch.equal(inputs.candidates, batch.candidates)
    assert torch.equal(inputs.mask, batch.mask)
    assert inputs.batch_size == batch.batch_size
    assert inputs.candidate_count == 3
    assert inputs.states.dtype == torch.float32
    assert inputs.candidates.dtype == torch.float32
    assert inputs.mask.dtype == torch.bool
    assert inputs.mask.tolist() == [
        [True, False, False],
        [True, True, True],
        [True, True, False],
    ]
    moved = inputs.to("cpu")
    assert torch.equal(moved.states, inputs.states)
    assert torch.equal(moved.candidates, inputs.candidates)
    assert torch.equal(moved.mask, inputs.mask)


def test_pad_decision_inputs_accepts_label_free_arrays():
    rng = np.random.default_rng(62)
    decisions = [
        SimpleNamespace(
            state=rng.normal(size=(STATE_WIDTH,)),
            candidate_features=rng.normal(size=(count, CANDIDATE_WIDTH)),
        )
        for count in (2, 1, 3)
    ]
    inputs = pad_decision_inputs(decisions)
    assert inputs.states.dtype == torch.float32
    assert inputs.states.shape == (3, STATE_WIDTH)
    assert inputs.candidates.shape == (3, 3, CANDIDATE_WIDTH)
    assert inputs.mask.tolist() == [
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ]
    for row, decision in enumerate(decisions):
        assert torch.equal(
            inputs.states[row],
            torch.tensor(decision.state, dtype=torch.float32),
        )
        count = decision.candidate_features.shape[0]
        assert torch.equal(
            inputs.candidates[row, :count],
            torch.tensor(decision.candidate_features, dtype=torch.float32),
        )
        assert torch.count_nonzero(inputs.candidates[row, count:]) == 0


def test_pad_decision_inputs_rejects_empty_or_malformed_decisions():
    with pytest.raises(DataError):
        pad_decision_inputs([])
    with pytest.raises(DataError, match="state and candidate_features"):
        pad_decision_inputs([object()])
    with pytest.raises(DataError, match="state must have shape"):
        pad_decision_inputs(
            [
                SimpleNamespace(
                    state=np.zeros(STATE_WIDTH - 1, dtype=np.float32),
                    candidate_features=np.zeros((2, CANDIDATE_WIDTH), dtype=np.float32),
                )
            ]
        )
    with pytest.raises(DataError, match="candidates must have shape"):
        pad_decision_inputs(
            [
                SimpleNamespace(
                    state=np.zeros(STATE_WIDTH, dtype=np.float32),
                    candidate_features=np.zeros((2, CANDIDATE_WIDTH - 1), dtype=np.float32),
                )
            ]
        )
    with pytest.raises(DataError, match="at least one candidate"):
        pad_decision_inputs(
            [
                SimpleNamespace(
                    state=np.zeros(STATE_WIDTH, dtype=np.float32),
                    candidate_features=np.zeros((0, CANDIDATE_WIDTH), dtype=np.float32),
                )
            ]
        )
    with pytest.raises(DataError, match="numeric arrays"):
        pad_decision_inputs(
            [
                SimpleNamespace(
                    state=np.zeros(STATE_WIDTH, dtype=np.float32),
                    candidate_features=np.full(
                        (1, CANDIDATE_WIDTH), "not-a-number", dtype=object
                    ),
                )
            ]
        )


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
def test_sharded_dataset_maps_global_indices_across_shards():
    first = make_dataset(split="train", counts=[2, 1], seed=41)
    second = make_dataset(split="train", counts=[1, 3, 2], seed=42)
    dataset = ShardedDecisionDataset(
        (first, second),
        identity(
            split="train",
            decisions=len(first) + len(second),
            eligible=first.eligible_count + second.eligible_count,
        ),
    )
    assert len(dataset) == 5
    assert dataset.decision_count == 5
    assert dataset.eligible_count == first.eligible_count + second.eligible_count
    assert dataset.forced_count == 2
    assert dataset.eligible_indices().tolist() == (
        first.eligible_indices().tolist()
        + (second.eligible_indices() + len(first)).tolist()
    )
    assert np.array_equal(
        dataset.candidate_counts(),
        np.concatenate([first.candidate_counts(), second.candidate_counts()]),
    )
    assert dataset.forced_by_stage() == tuple(
        left + right
        for left, right in zip(first.forced_by_stage(), second.forced_by_stage())
    )
    assert not contains_memory_mapped_arrays(dataset)
    for index in range(len(dataset)):
        expected = first[index] if index < len(first) else second[index - len(first)]
        assert np.array_equal(dataset[index].state, expected.state)
        assert dataset[index].target == expected.target
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    dataset.close()


def test_iter_batches_validates_arguments_and_handles_empty_indices():
    dataset = make_dataset(split="train", counts=[2, 2], seed=43)
    with pytest.raises(DataError):
        list(iter_batches(dataset, [0], batch_size=0))
    with pytest.raises(DataError):
        list(iter_batches(dataset, [0], batch_size=1, num_workers=-1))
    with pytest.raises(DataError):
        list(iter_batches(dataset, np.array([[0]]), batch_size=1))
    assert list(iter_batches(dataset, [], batch_size=4)) == []


def test_collate_rejects_empty_batches_and_invalid_samples():
    with pytest.raises(DataError):
        collate_decisions([])
    dataset = make_dataset(split="train", counts=[2], seed=44)
    sample = dataset[0]
    bad_target = SimpleNamespace(
        state=sample.state,
        candidate_features=sample.candidate_features,
        target=sample.candidate_count,
        stage=sample.stage,
        target_kind=sample.target_kind,
        candidate_count=sample.candidate_count,
    )
    with pytest.raises(DataError):
        collate_decisions([bad_target])


def test_shard_identity_is_optional_but_sharded_split_identity_is_required():
    source = make_dataset(split="train", counts=[2, 1, 3], seed=45)
    shard = PackedDecisionDataset(
        states=source.states,
        candidate_features=source.candidate_features,
        offsets=source.offsets,
        targets=source.targets,
        stages=source.stages,
        identity=None,
        target_kinds=source.target_kinds,
    )
    assert shard.identity is None
    assert len(shard) == len(source)
    complete = ShardedDecisionDataset((shard,), source.identity)
    assert complete.identity is source.identity
    with pytest.raises(DataError, match="DatasetIdentity"):
        ShardedDecisionDataset((shard,), None)


def test_memory_map_contract_delegates_and_blocks_workers(tmp_path):
    packed = _memory_mapped_dataset(tmp_path)
    dataset = ShardedDecisionDataset((packed,), packed.identity)
    try:
        assert packed.is_memory_mapped
        assert contains_memory_mapped_arrays(packed)
        assert dataset.is_memory_mapped
        assert contains_memory_mapped_arrays(dataset)
        wrapper = _DelegatingDataset(dataset)
        assert wrapper.is_memory_mapped
        assert contains_memory_mapped_arrays(wrapper)
        with pytest.raises(DataError, match="memory-mapped"):
            list(
                iter_batches(
                    wrapper, [0], batch_size=1, num_workers=1
                )
            )
        sample = wrapper[0]
        expected_state = sample.state.copy()
        dataset.close()
        assert np.array_equal(sample.state, expected_state)
        with pytest.raises(DataError, match="closed"):
            wrapper[0]
    finally:
        dataset.close()


def test_in_memory_wrapper_has_no_map_and_still_iterates_without_workers():
    dataset = make_dataset(split="train", counts=[2, 2], seed=46)
    wrapper = _DelegatingDataset(dataset)
    assert not wrapper.is_memory_mapped
    assert not contains_memory_mapped_arrays(wrapper)
    batches = list(iter_batches(wrapper, [0, 1], batch_size=1))
    assert [batch.batch_size for batch in batches] == [1, 1]


def test_unknown_dataset_implementations_cannot_be_assumed_map_free():
    with pytest.raises(TypeError, match="is_memory_mapped"):
        _MissingMmapContract()
    with pytest.raises(DataError, match="DecisionDataset"):
        contains_memory_mapped_arrays(object())  # type: ignore[arg-type]
