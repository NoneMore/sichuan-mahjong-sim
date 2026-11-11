from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest
import torch

import mahjong_bc.data as data_module
from mahjong_bc.constants import CANDIDATE_WIDTH, STATE_WIDTH
from mahjong_bc.data import (
    DecisionDataset,
    DecisionSample,
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


def _memory_mapped_dataset(
    tmp_path,
    *,
    counts=(2, 1, 3),
    seed: int = 45,
    prefix: str = "",
    with_identity: bool = True,
    target_kinds: bool = True,
) -> PackedDecisionDataset:
    source = make_dataset(split="train", counts=list(counts), seed=seed)
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
        name: _write_memmap(tmp_path, f"{prefix}{name}", value)
        for name, value in values.items()
    }
    return PackedDecisionDataset(
        states=mapped["states"],
        candidate_features=mapped["candidate_features"],
        offsets=mapped["offsets"],
        targets=mapped["targets"],
        stages=mapped["stages"],
        identity=source.identity if with_identity else None,
        actor_seats=mapped["actor_seats"],
        decision_indices=mapped["decision_indices"],
        episode_ids=mapped["episode_ids"],
        target_kinds=mapped["target_kinds"] if target_kinds else None,
    )


def _metadata_dataset(
    *,
    counts,
    seed: int,
    split: str = "train",
    actor_seats: bool = True,
    decision_indices: bool = True,
    episode_ids: str | None = "array",
    target_kinds: bool = True,
) -> PackedDecisionDataset:
    """Packed split with every optional metadata array configurable."""

    source = make_dataset(split=split, counts=list(counts), seed=seed)
    decisions = len(source)
    if episode_ids == "array":
        episodes: Sequence[object] | None = (
            np.arange(decisions, dtype=np.uint64) + 2**40
        )
    elif episode_ids == "list":
        episodes = [f"episode-{index}" for index in range(decisions)]
    elif episode_ids == "with-none":
        episodes = [None if index == 1 else f"episode-{index}" for index in range(decisions)]
    else:
        episodes = None
    return PackedDecisionDataset(
        states=source.states,
        candidate_features=source.candidate_features,
        offsets=source.offsets,
        targets=source.targets,
        stages=source.stages,
        identity=source.identity,
        actor_seats=(
            np.arange(decisions, dtype=np.uint8) % 4 if actor_seats else None
        ),
        decision_indices=(
            np.arange(decisions, dtype=np.uint64) + 11 if decision_indices else None
        ),
        episode_ids=episodes,
        target_kinds=source.target_kinds if target_kinds else None,
    )


def _empty_packed_dataset() -> PackedDecisionDataset:
    """A valid packed split with no decisions (empty splits have no shards)."""

    return PackedDecisionDataset(
        states=np.empty((0, STATE_WIDTH), dtype=np.float32),
        candidate_features=np.empty((0, CANDIDATE_WIDTH), dtype=np.float32),
        offsets=np.zeros(1, dtype=np.int64),
        targets=np.empty(0, dtype=np.int64),
        stages=np.empty(0, dtype=np.int64),
        identity=identity(split="train", decisions=0, eligible=0),
        actor_seats=np.empty(0, dtype=np.uint8),
        decision_indices=np.empty(0, dtype=np.uint64),
        episode_ids=np.empty(0, dtype=np.uint64),
        target_kinds=np.empty(0, dtype=np.uint8),
    )


def _reference_batches(dataset, indices, batch_size: int):
    """Build batches the documented way: one sample per index plus collation."""

    ordered = np.asarray(indices, dtype=np.int64)
    return [
        collate_decisions(
            [dataset[int(index)] for index in ordered[start : start + batch_size]]
        )
        for start in range(0, ordered.size, batch_size)
    ]


_BATCH_FIELDS = (
    "states",
    "candidates",
    "mask",
    "targets",
    "stages",
    "target_kinds",
    "eligible",
    "actor_seats",
    "decision_indices",
)


def _assert_batches_equal(actual, expected) -> None:
    assert actual.batch_size == expected.batch_size
    for name in _BATCH_FIELDS:
        got = getattr(actual, name)
        want = getattr(expected, name)
        if want is None:
            assert got is None, name
            continue
        assert got is not None, name
        assert got.dtype == want.dtype, name
        assert torch.equal(got, want), name
    if expected.episode_ids is None:
        assert actual.episode_ids is None
    else:
        assert actual.episode_ids == expected.episode_ids
        assert [type(value) for value in actual.episode_ids] == [
            type(value) for value in expected.episode_ids
        ]


def _assert_vectorized_equals_reference(monkeypatch, dataset, indices, batch_size: int):
    """Compare the batch iterator with the per-sample reference path.

    ``__getitem__`` is disabled while iterating so a silent fallback cannot
    make these equivalence checks pass without the vectorized path.
    """

    expected = _reference_batches(dataset, indices, batch_size)

    def forbidden(self, index):
        raise AssertionError(
            "the vectorized path must not read one sample per decision"
        )

    with monkeypatch.context() as patch:
        patch.setattr(PackedDecisionDataset, "__getitem__", forbidden)
        patch.setattr(ShardedDecisionDataset, "__getitem__", forbidden)
        actual = list(iter_batches(dataset, indices, batch_size=batch_size))
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        _assert_batches_equal(got, want)
    return actual


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


class _TrackingDataset(PackedDecisionDataset):
    """Subclass whose ``__getitem__`` override must stay authoritative."""

    def __init__(self, source: PackedDecisionDataset) -> None:
        super().__init__(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=source.identity,
            actor_seats=source.actor_seats,
            decision_indices=source.decision_indices,
            episode_ids=source.episode_ids,
            target_kinds=source.target_kinds,
        )
        self.visited: list[int] = []

    def __getitem__(self, index: int) -> DecisionSample:
        self.visited.append(int(index))
        sample = super().__getitem__(index)
        return DecisionSample(
            state=sample.state * 2.0,
            candidate_features=sample.candidate_features,
            target=sample.target,
            stage=sample.stage,
            target_kind=sample.target_kind,
            actor_seat=sample.actor_seat,
            decision_index=sample.decision_index,
            episode_id=sample.episode_id,
        )


def test_vectorized_packed_batches_match_reference_collation(monkeypatch):
    dataset = _metadata_dataset(counts=[1, 3, 2, 4, 1, 2, 3], seed=101)
    indices = np.array([6, 0, 3, 3, 1, 5, 2, 4])
    batches = _assert_vectorized_equals_reference(monkeypatch, dataset, indices, 3)
    assert [batch.batch_size for batch in batches] == [3, 3, 2]
    assert [int(value) for batch in batches for value in batch.targets] == [
        int(dataset.targets[index]) for index in indices
    ]
    # Repeated indices repeat the same decision in the batch, in order.
    assert torch.equal(batches[0].states[2], batches[1].states[0])
    # Candidate order and padding match the packed source rows exactly.
    start = int(dataset.offsets[6])
    first_count = int(dataset.candidate_counts()[6])
    assert torch.equal(
        batches[0].states[0], torch.from_numpy(dataset.states[6])
    )
    assert torch.equal(
        batches[0].candidates[0, :first_count],
        torch.from_numpy(dataset.candidate_features[start : start + first_count]),
    )
    assert torch.count_nonzero(batches[0].candidates[0, first_count:]) == 0
    assert batches[0].mask[0, :first_count].all()
    assert not batches[0].mask[0, first_count:].any()
    assert batches[0].eligible.tolist() == [
        bool(dataset.candidate_counts()[index] > 1) for index in indices[:3]
    ]


@pytest.mark.parametrize(
    "actor_seats, decision_indices, episode_ids",
    [
        (True, True, "array"),
        (True, True, "list"),
        (False, True, "array"),
        (True, False, "list"),
        (False, False, None),
    ],
)
def test_vectorized_packed_batches_match_reference_metadata_variants(
    monkeypatch, actor_seats, decision_indices, episode_ids
):
    dataset = _metadata_dataset(
        counts=[2, 1, 3, 2],
        seed=102,
        actor_seats=actor_seats,
        decision_indices=decision_indices,
        episode_ids=episode_ids,
    )
    batches = _assert_vectorized_equals_reference(monkeypatch, dataset, [3, 0, 2, 1, 2], 2)
    assert (batches[0].actor_seats is not None) is actor_seats
    assert (batches[0].decision_indices is not None) is decision_indices
    assert (batches[0].episode_ids is not None) is (episode_ids is not None)
    assert batches[0].stages.dtype == torch.int64
    assert batches[0].target_kinds.dtype == torch.int64
    assert batches[0].candidates.dtype == torch.float32
    assert batches[0].mask.dtype == torch.bool


def test_vectorized_packed_batches_drop_partial_episode_metadata(monkeypatch):
    dataset = _metadata_dataset(counts=[2, 2, 2], seed=103, episode_ids="with-none")
    # Position 1 carries no episode id, so only batches that avoid it keep ids.
    without = _assert_vectorized_equals_reference(monkeypatch, dataset, [0, 2], 2)
    assert without[0].episode_ids == ("episode-0", "episode-2")
    with_missing = _assert_vectorized_equals_reference(monkeypatch, dataset, [0, 1], 2)
    assert with_missing[0].episode_ids is None


def test_vectorized_sharded_batches_match_reference_across_shards(monkeypatch):
    first = _metadata_dataset(counts=[2, 1], seed=104)
    second = _metadata_dataset(counts=[1, 3, 2], seed=105)
    third = _metadata_dataset(counts=[3, 2], seed=106)
    shards = (first, second, third)
    dataset = ShardedDecisionDataset(
        shards,
        identity(
            split="train",
            decisions=7,
            eligible=sum(shard.eligible_count for shard in shards),
        ),
    )
    indices = np.array([6, 0, 2, 4, 1, 5, 3, 2])
    batches = _assert_vectorized_equals_reference(monkeypatch, dataset, indices, 4)
    assert [batch.batch_size for batch in batches] == [4, 4]
    assert [int(value) for batch in batches for value in batch.targets] == [
        int(dataset[index].target) for index in indices
    ]
    assert batches[0].candidates.shape[1] == max(
        int(dataset.candidate_counts()[index]) for index in indices[:4]
    )
    assert batches[0].episode_ids == tuple(
        dataset[index].episode_id for index in indices[:4]
    )
    assert batches[1].eligible.tolist() == [
        bool(dataset.candidate_counts()[index] > 1) for index in indices[4:]
    ]
    # A whole-split order and a forced shard-local decision both stay exact.
    _assert_vectorized_equals_reference(monkeypatch, dataset, np.arange(7), 7)


def test_vectorized_sharded_metadata_presence_follows_touched_shards(monkeypatch):
    rich = _metadata_dataset(counts=[2, 2], seed=107)
    bare = _metadata_dataset(
        counts=[2, 2],
        seed=108,
        actor_seats=False,
        decision_indices=False,
        episode_ids=None,
    )
    dataset = ShardedDecisionDataset(
        (rich, bare),
        identity(
            split="train",
            decisions=4,
            eligible=rich.eligible_count + bare.eligible_count,
        ),
    )
    only_rich = _assert_vectorized_equals_reference(monkeypatch, dataset, [0, 1], 2)
    assert only_rich[0].actor_seats is not None
    assert only_rich[0].decision_indices is not None
    assert only_rich[0].episode_ids is not None
    across = _assert_vectorized_equals_reference(monkeypatch, dataset, [0, 2, 3], 3)
    assert across[0].actor_seats is None
    assert across[0].decision_indices is None
    assert across[0].episode_ids is None


def test_vectorized_empty_shard_and_duplicate_heavy_order(monkeypatch):
    empty = _empty_packed_dataset()
    filled = _metadata_dataset(counts=[3, 2], seed=110)
    dataset = ShardedDecisionDataset(
        (empty, filled),
        identity(split="train", decisions=2, eligible=filled.eligible_count),
    )
    indices = np.array([1, 1, 0, 1, 0])
    batches = _assert_vectorized_equals_reference(monkeypatch, dataset, indices, 2)
    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    assert torch.equal(batches[0].targets[0], batches[0].targets[1])


def test_vectorized_batches_keep_memory_mapped_sources_independent(tmp_path, monkeypatch):
    first = _memory_mapped_dataset(
        tmp_path, counts=(2, 1), seed=45, prefix="a", with_identity=False
    )
    second = _memory_mapped_dataset(
        tmp_path, counts=(1, 3, 2), seed=46, prefix="b", with_identity=False
    )
    dataset = ShardedDecisionDataset(
        (first, second),
        identity(
            split="train",
            decisions=5,
            eligible=first.eligible_count + second.eligible_count,
        ),
    )
    try:
        assert not first.states.flags.writeable
        assert not second.candidate_features.flags.writeable
        indices = np.array([4, 1, 0, 3, 2, 4])
        batches = _assert_vectorized_equals_reference(monkeypatch, dataset, indices, 4)
        mapped_address = first.states.__array_interface__["data"][0]
        assert batches[0].states.data_ptr() != mapped_address
        states = batches[0].states.clone()
        dataset.close()
        assert torch.equal(batches[0].states, states)
    finally:
        dataset.close()


def test_vectorized_batches_report_getitem_index_errors():
    dataset = _metadata_dataset(counts=[2, 3, 1], seed=111)
    for bad in (-1, 3, 99):
        with pytest.raises(IndexError) as reference:
            dataset[bad]
        with pytest.raises(IndexError) as batched:
            list(iter_batches(dataset, [0, bad], batch_size=4))
        assert str(batched.value) == str(reference.value)
    with pytest.raises(IndexError) as batched:
        list(iter_batches(dataset, [99, -1], batch_size=4))
    assert str(batched.value) == "decision index 99 is out of range"
    dataset.close()
    with pytest.raises(DataError, match="closed"):
        list(iter_batches(dataset, [0], batch_size=1))
    assert list(iter_batches(dataset, [], batch_size=1)) == []


def test_vectorized_batches_reject_closed_shards_only_when_read(monkeypatch):
    first = _metadata_dataset(counts=[2, 1], seed=112)
    second = _metadata_dataset(counts=[2, 2], seed=113)
    dataset = ShardedDecisionDataset(
        (first, second),
        identity(
            split="train",
            decisions=4,
            eligible=first.eligible_count + second.eligible_count,
        ),
    )
    try:
        first.close()
        with pytest.raises(DataError, match="closed"):
            list(iter_batches(dataset, [0, 2], batch_size=2))
        # A directly closed shard only fails batches that read it; closing also
        # clears its metadata, so such a split keeps the per-sample path.
        batches = list(iter_batches(dataset, [2, 3], batch_size=2))
        for got, want in zip(batches, _reference_batches(dataset, [2, 3], 2)):
            _assert_batches_equal(got, want)
        assert [batch.batch_size for batch in batches] == [2]
    finally:
        dataset.close()
    with pytest.raises(DataError, match="closed"):
        list(iter_batches(dataset, [2], batch_size=1))


def test_vectorized_batches_revalidate_targets_mutated_after_construction():
    dataset = _metadata_dataset(counts=[2, 2], seed=114)
    dataset.targets[0] = 5
    with pytest.raises(DataError, match="target"):
        list(iter_batches(dataset, [0, 1], batch_size=2))
    with pytest.raises(DataError, match="target"):
        collate_decisions([dataset[0]])
    assert list(iter_batches(dataset, [1], batch_size=1))[0].batch_size == 1


def test_subclass_getitem_override_is_not_bypassed(monkeypatch):
    source = _metadata_dataset(counts=[2, 1, 3], seed=115)
    dataset = _TrackingDataset(source)
    expected = _reference_batches(dataset, [2, 0, 1], 2)
    dataset.visited.clear()
    batches = list(iter_batches(dataset, [2, 0, 1], batch_size=2))
    assert dataset.visited == [2, 0, 1]
    assert torch.equal(batches[0].states[0], torch.from_numpy(source.states[2]) * 2.0)
    for got, want in zip(batches, expected):
        _assert_batches_equal(got, want)

    wrapped = _TrackingDataset(_metadata_dataset(counts=[2, 2], seed=116))
    other = _metadata_dataset(counts=[3, 1], seed=117)
    sharded = ShardedDecisionDataset(
        (wrapped, other),
        identity(
            split="train",
            decisions=4,
            eligible=wrapped.eligible_count + other.eligible_count,
        ),
    )
    sharded_expected = _reference_batches(sharded, [0, 2, 1], 3)
    wrapped.visited.clear()
    sharded_batches = list(iter_batches(sharded, [0, 2, 1], batch_size=3))
    assert wrapped.visited == [0, 1]
    for got, want in zip(sharded_batches, sharded_expected):
        _assert_batches_equal(got, want)


def test_vectorized_counts_follow_current_offsets(monkeypatch):
    # A valid in-place offsets edit must stay visible to batching exactly as it
    # is to ``__getitem__``; the cached construction-time layout is stale.
    dataset = _metadata_dataset(counts=[2, 2, 3], seed=119)
    dataset.targets[0] = 0
    dataset.offsets[1] = 1
    current = [
        int(dataset.offsets[index + 1] - dataset.offsets[index])
        for index in range(len(dataset))
    ]
    assert current == [1, 3, 3]
    assert int(dataset._candidate_counts[0]) == 2
    batches = _assert_vectorized_equals_reference(monkeypatch, dataset, [0, 1, 2], 2)
    assert batches[0].mask.sum(dim=1).tolist() == current[:2]
    assert batches[0].eligible.tolist() == [count > 1 for count in current[:2]]
    assert batches[1].mask.sum(dim=1).tolist() == current[2:]
    assert batches[1].eligible.tolist() == [count > 1 for count in current[2:]]


def test_vectorized_unsigned_metadata_overflow_keeps_reference_error():
    dataset = _metadata_dataset(counts=[2, 2], seed=120)
    dataset.decision_indices[0] = np.uint64(2**63)
    with pytest.raises(ValueError) as reference:
        collate_decisions([dataset[0]])
    with pytest.raises(ValueError) as batched:
        list(iter_batches(dataset, [0, 1], batch_size=2))
    assert str(batched.value) == str(reference.value)
    # Decisions below the int64 range keep the vectorized batch.
    other = list(iter_batches(dataset, [1], batch_size=1))
    assert other[0].decision_indices.tolist() == [int(dataset.decision_indices[1])]


def test_vectorized_actor_overflow_keeps_reference_error():
    dataset = _metadata_dataset(counts=[2, 2], seed=121)
    dataset.actor_seats = dataset.actor_seats.astype(np.uint64)
    dataset.actor_seats[0] = np.uint64(2**63)
    with pytest.raises(ValueError) as reference:
        collate_decisions([dataset[0]])
    with pytest.raises(ValueError) as batched:
        list(iter_batches(dataset, [0, 1], batch_size=2))
    assert str(batched.value) == str(reference.value)
    # Only the selected rows determine whether a batch needs the fallback.
    other = list(iter_batches(dataset, [1], batch_size=1))
    assert other[0].actor_seats.tolist() == [int(dataset.actor_seats[1])]


def test_packed_shards_without_target_kinds_keep_the_per_sample_path(monkeypatch):
    dataset = _metadata_dataset(counts=[2, 1, 3], seed=118, target_kinds=False)
    expected = _reference_batches(dataset, [0, 1, 2, 2], 2)
    calls: list[int] = []
    real_collate = data_module.collate_decisions

    def spy(samples):
        calls.append(len(samples))
        return real_collate(samples)

    with monkeypatch.context() as patch:
        patch.setattr(data_module, "collate_decisions", spy)
        batches = list(iter_batches(dataset, [0, 1, 2, 2], batch_size=2))
    assert calls == [2, 2]
    for got, want in zip(batches, expected):
        _assert_batches_equal(got, want)
    assert batches[0].target_kinds.tolist() == [
        int(dataset[0].target_kind),
        int(dataset[1].target_kind),
    ]
