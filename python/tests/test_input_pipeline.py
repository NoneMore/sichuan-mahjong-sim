"""Tests for the shared device-batch pipeline and CUDA prefetch.

Prefetch tests run on real CUDA: they cover staging capacity, batch equality
with the synchronous path, producer/consumer lifecycle, error ordering and
copy/compute overlap. Validation-only tests also run without a device.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import replace
from typing import Callable, Iterator, Sequence

import numpy as np
import pytest
import torch

import mahjong_bc._input_pipeline as pipeline_module
from mahjong_bc._input_pipeline import (
    BatchTransferConfig,
    device_batches,
    slot_capacity_bytes,
    validate_batch_transfer,
)
from mahjong_bc.data import (
    PackedDecisionDataset,
    ShardedDecisionDataset,
    _PackedCandidateBatch,
    _iter_model_batches,
)
from mahjong_bc.errors import ConfigurationError, DataError
from mahjong_bc.policy import CandidatePolicy, _batch_logits, build_policy

from .helpers import DelegatingDataset, identity, make_dataset

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
TRAIN_FIELDS = ("states", "candidates", "mask", "targets", "candidate_positions")
EVAL_FIELDS = TRAIN_FIELDS + ("stages", "target_kinds")
PRODUCER_NAME = "mahjong-bc-prefetch"
#: Fields the producer copies per commit; used to align per-batch events.
TRAIN_COPIES = 5


def freeze(dataset: PackedDecisionDataset) -> PackedDecisionDataset:
    """Make a fixture's consumed inputs read-only, like a RAM-resident split."""

    for name in pipeline_module._STAGED_ARRAYS:
        array = getattr(dataset, name)
        if array is not None:
            array.setflags(write=False)
    return dataset


def sharded_split(
    *, split: str = "train", first: Sequence[int] = (2, 1, 3), second: Sequence[int] = (2, 4)
) -> ShardedDecisionDataset:
    left = freeze(make_dataset(split=split, counts=list(first), seed=21))
    right = freeze(make_dataset(split=split, counts=list(second), seed=22))
    return ShardedDecisionDataset(
        (left, right),
        identity(
            split=split,
            decisions=len(left) + len(right),
            eligible=left.eligible_count + right.eligible_count,
        ),
    )


def pinned_bytes(
    dataset: PackedDecisionDataset | ShardedDecisionDataset,
    indices: np.ndarray,
    batch_size: int,
    *,
    training: bool,
) -> int:
    """The published budget formula for this traversal."""

    counts = dataset.candidate_counts()
    rows = min(batch_size, int(indices.size))
    candidates = int(counts[indices].max()) if indices.size else 0
    return slot_capacity_bytes(
        batch_size=rows, candidates=candidates, training=training
    )


@pytest.fixture(scope="module")
def model() -> CandidatePolicy:
    return build_policy(seed=0)


@pytest.fixture(autouse=True)
def no_leaked_producer_threads():
    """Every test must leave no prefetch producer behind."""

    yield
    assert not live_producers()


def reference_batches(
    dataset,
    indices: Sequence[int] | np.ndarray,
    batch_size: int,
    *,
    training: bool,
) -> list[_PackedCandidateBatch]:
    """The synchronous path's batches, moved to the device for comparison."""

    return [
        batch.to("cuda")
        for batch in _iter_model_batches(
            dataset,
            indices,
            batch_size=batch_size,
            training=training,
            packed_candidates=True,
        )
    ]


def collect(
    dataset,
    indices,
    batch_size: int,
    *,
    training: bool,
    config: BatchTransferConfig,
    model: CandidatePolicy,
    device: str = "cuda",
    eligible_only: bool = False,
    batch_iterator: Callable[..., Iterator[object]] | None = None,
) -> list[_PackedCandidateBatch]:
    batches: list[_PackedCandidateBatch] = []
    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=training,
        packed_candidates=True,
        model=model,
        device=device,
        batch_transfer=config,
        eligible_only=eligible_only,
        batch_iterator=batch_iterator,
    ) as iterator:
        for batch in iterator:
            batches.append(batch)
    return batches


def assert_batches_equal(
    got: object, want: object, fields: Sequence[str]
) -> None:
    assert got.batch_size == want.batch_size
    for name in fields:
        actual = getattr(got, name)
        expected = getattr(want, name)
        assert isinstance(actual, torch.Tensor), name
        assert actual.dtype == expected.dtype, name
        assert tuple(actual.shape) == tuple(expected.shape), name
        assert actual.device.type == "cuda", name
        assert torch.equal(actual.cpu(), expected.cpu()), name


def wait_for(predicate: Callable[[], bool], *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def with_deadline(action: Callable[[], None], *, timeout: float = 30.0) -> None:
    """Fail instead of hanging when a lifecycle operation deadlocks."""

    errors: list[BaseException] = []

    def run() -> None:
        try:
            action()
        except BaseException as error:  # re-raised on the test thread
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "pipeline operation did not finish in time"
    if errors:
        raise errors[0]


def live_producers() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == PRODUCER_NAME]


# ---- configuration -----------------------------------------------------


def test_default_config_is_synchronous_and_records_both_fields():
    config = BatchTransferConfig()
    assert config.mode == "sync"
    assert config.pinned_limit_bytes is None
    assert config.to_dict() == {"mode": "sync", "pinned_limit_bytes": None}
    prefetch = BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=4096)
    assert prefetch.to_dict() == {
        "mode": "cuda-prefetch",
        "pinned_limit_bytes": 4096,
    }


def test_sync_config_rejects_a_pinned_budget():
    with pytest.raises(ConfigurationError, match="does not accept pinned_limit_bytes"):
        BatchTransferConfig(mode="sync", pinned_limit_bytes=1024)


@pytest.mark.parametrize(
    "budget", [None, 0, -1, True, False, 1.5, "1024", 2**40 + 1]
)
def test_prefetch_config_requires_a_positive_integer_budget(budget):
    if budget == 2**40 + 1:
        # A large int is a valid budget; only non-ints are rejected here.
        assert BatchTransferConfig("cuda-prefetch", budget).pinned_limit_bytes == budget
        return
    with pytest.raises(ConfigurationError, match="pinned_limit_bytes"):
        BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=budget)


def test_unknown_transfer_mode_is_rejected():
    with pytest.raises(ConfigurationError, match="unknown batch transfer mode"):
        BatchTransferConfig(mode="prefetch")


@pytest.mark.parametrize("training", [True, False])
def test_pinned_budget_matches_the_published_formula(training):
    # 4060 = states 4*1013 + targets 8; 165 = candidates 4*39 + positions 8 +
    # mask 1; evaluation adds stages and target kinds (2*8).
    expected_rows = 4076 if not training else 4060
    assert slot_capacity_bytes(batch_size=8, candidates=3, training=training) == (
        2 * (expected_rows * 8 + 165 * 8 * 3)
    )
    assert slot_capacity_bytes(batch_size=1, candidates=1, training=training) == (
        2 * (expected_rows + 165)
    )
    assert slot_capacity_bytes(batch_size=0, candidates=0, training=training) == 0


# ---- synchronous path --------------------------------------------------


def test_sync_transfer_reproduces_the_default_iterator():
    dataset = freeze(make_dataset(split="train", counts=[3, 1, 2, 4], seed=31))
    indices = np.array([3, 0, 2], dtype=np.int64)
    expected = list(
        _iter_model_batches(
            dataset, indices, batch_size=2, training=True, packed_candidates=True
        )
    )
    with device_batches(
        dataset,
        indices,
        batch_size=2,
        training=True,
        packed_candidates=True,
        model=build_policy(seed=0),
        device="cpu",
    ) as iterator:
        got = list(iterator)
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected):
        assert actual.__class__ is wanted.__class__
        for name in TRAIN_FIELDS:
            assert torch.equal(getattr(actual, name), getattr(wanted, name)), name
        assert actual.stages is None and actual.target_kinds is None


def test_sync_transfer_uses_the_supplied_factory_and_host_validator():
    dataset = freeze(make_dataset(split="train", counts=[2] * 4, seed=32))
    seen: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []

    def factory(dataset, indices, **kwargs):
        calls.append(dict(kwargs))
        return _iter_model_batches(dataset, indices, **kwargs)

    def validator(batch: object) -> None:
        assert batch.states.device.type == "cpu"
        seen.append((batch.batch_size, "cpu"))

    with device_batches(
        dataset,
        np.arange(len(dataset)),
        batch_size=2,
        training=True,
        packed_candidates=True,
        model=build_policy(seed=0),
        device="cpu",
        batch_iterator=factory,
        host_validator=validator,
    ) as iterator:
        batches = list(iterator)
    assert [size for size, _ in seen] == [2, 2]
    assert [batch.batch_size for batch in batches] == [2, 2]
    assert calls == [
        {
            "batch_size": 2,
            "num_workers": 0,
            "training": True,
            "packed_candidates": True,
        }
    ]


def test_sync_transfer_rejects_a_non_callable_batch_iterator():
    dataset = freeze(make_dataset(split="train", counts=[2] * 2, seed=32))
    with pytest.raises(ConfigurationError, match="batch_iterator must be a callable"):
        with device_batches(
            dataset,
            np.arange(len(dataset)),
            batch_size=2,
            training=True,
            model=build_policy(seed=0),
            device="cpu",
            batch_iterator=iter(()),
        ):
            pass


def test_sync_host_validator_error_stops_before_the_batch_is_moved():
    dataset = freeze(make_dataset(split="train", counts=[2] * 4, seed=33))
    calls: list[int] = []

    def validator(batch: object) -> None:
        calls.append(batch.batch_size)
        if len(calls) == 2:
            raise DataError("validation stopped at batch 2")

    with device_batches(
        dataset,
        np.arange(len(dataset)),
        batch_size=2,
        training=True,
        packed_candidates=True,
        model=build_policy(seed=0),
        device="cpu",
        host_validator=validator,
    ) as iterator:
        assert next(iterator).batch_size == 2
        with pytest.raises(DataError, match="validation stopped at batch 2"):
            next(iterator)
        assert calls == [2, 2]
        with pytest.raises(StopIteration):
            next(iterator)


def test_sync_keeps_custom_inputs_and_workers(model):
    """An explicit sync request keeps every historical dataset allowance."""

    packed = freeze(make_dataset(split="train", counts=[2, 3, 1], seed=34))
    wrapped = DelegatingDataset(packed)
    validate_batch_transfer(
        wrapped,
        model=DelegatingModel(),
        device="cpu",
        num_workers=4,
        config=BatchTransferConfig(),
        training=True,
        packed_candidates=False,
    )
    with device_batches(
        wrapped,
        np.arange(len(wrapped)),
        batch_size=2,
        training=True,
        model=build_policy(seed=0),
        device="cpu",
    ) as iterator:
        batches = list(iterator)
    assert [batch.batch_size for batch in batches] == [2, 1]
    # Custom datasets keep the full public batch with supervision metadata.
    assert batches[0].eligible.tolist() == [True, True]


class DelegatingModel(CandidatePolicy):
    """Custom policy subclass: keeps the padded forward interface."""


# ---- prefetch validation ----------------------------------------------


def prefetch_request(dataset, model, **overrides):
    request = dict(
        model=model,
        device="cuda",
        num_workers=0,
        config=BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=1 << 20),
        training=True,
        packed_candidates=True,
    )
    request.update(overrides)
    return validate_batch_transfer(dataset, **request)


def test_prefetch_requires_a_cuda_device(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 1], seed=35))
    with pytest.raises(ConfigurationError, match="requires a CUDA device"):
        prefetch_request(dataset, model, device="cpu")


@CUDA
def test_prefetch_rejects_workers(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 1], seed=36))
    with pytest.raises(ConfigurationError, match="num_workers=0"):
        prefetch_request(dataset, model, num_workers=2)


@CUDA
def test_prefetch_requires_flat_candidates(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 1], seed=37))
    with pytest.raises(ConfigurationError, match="flat candidates"):
        prefetch_request(dataset, model, packed_candidates=False)


@CUDA
def test_prefetch_requires_the_builtin_policy():
    dataset = freeze(make_dataset(split="train", counts=[2, 1], seed=38))
    with pytest.raises(ConfigurationError, match="CandidatePolicy"):
        prefetch_request(dataset, DelegatingModel())


@CUDA
def test_prefetch_evaluation_requires_an_eligible_only_traversal(model):
    dataset = freeze(make_dataset(split="validation", counts=[2, 1], seed=39))
    with pytest.raises(ConfigurationError, match="eligible-only"):
        prefetch_request(dataset, model, training=False, eligible_only=False)
    assert (
        prefetch_request(dataset, model, training=False, eligible_only=True) is None
    )


@CUDA
def test_prefetch_rejects_custom_dataset_types(model):
    packed = freeze(make_dataset(split="train", counts=[2, 1], seed=40))
    for dataset in (DelegatingDataset(packed), PackedSubclass.from_dataset(packed)):
        with pytest.raises(ConfigurationError, match="built-in PackedDecisionDataset"):
            prefetch_request(dataset, model)


class PackedSubclass(PackedDecisionDataset):
    """A subclass must not silently enter the strict packed producer."""

    @classmethod
    def from_dataset(cls, source: PackedDecisionDataset) -> "PackedSubclass":
        return cls(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=source.identity,
            target_kinds=source.target_kinds,
        )


@CUDA
def test_prefetch_rejects_anonymous_shards(model):
    source = make_dataset(split="train", counts=[2, 1], seed=41)
    anonymous = freeze(
        PackedDecisionDataset(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=None,
            target_kinds=source.target_kinds,
        )
    )
    with pytest.raises(ConfigurationError, match="split identity"):
        prefetch_request(anonymous, model)


@CUDA
def test_prefetch_rejects_missing_target_kinds(model):
    source = make_dataset(split="train", counts=[2, 1], seed=42)
    dataset = freeze(
        PackedDecisionDataset(
            states=source.states,
            candidate_features=source.candidate_features,
            offsets=source.offsets,
            targets=source.targets,
            stages=source.stages,
            identity=source.identity,
            target_kinds=None,
        )
    )
    with pytest.raises(ConfigurationError, match="target_kinds"):
        prefetch_request(dataset, model)


@CUDA
@pytest.mark.parametrize(
    "name", ["states", "candidate_features", "offsets", "targets", "stages", "target_kinds"]
)
def test_prefetch_rejects_writable_inputs(model, name):
    dataset = make_dataset(split="train", counts=[2, 1], seed=43)
    for array_name in pipeline_module._STAGED_ARRAYS:
        if array_name != name:
            getattr(dataset, array_name).setflags(write=False)
    with pytest.raises(ConfigurationError, match=f"read-only.*{name}"):
        prefetch_request(dataset, model)


@CUDA
def test_prefetch_rejects_offsets_edited_after_construction(model):
    dataset = make_dataset(split="train", counts=[2, 3, 2], seed=44)
    dataset.offsets[1] += 1
    freeze(dataset)
    with pytest.raises(ConfigurationError, match="offsets matching"):
        prefetch_request(dataset, model)


@CUDA
def test_prefetch_validation_accepts_a_valid_request(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 3, 1], seed=45))
    assert prefetch_request(dataset, model) is None
    assert prefetch_request(dataset, model, training=True, eligible_only=True) is None


# ---- prefetch data equivalence ----------------------------------------


@CUDA
def test_prefetch_training_batches_match_the_sync_path(model):
    dataset = freeze(
        make_dataset(split="train", counts=[2, 3, 1, 4, 2, 5, 2, 3], seed=46)
    )
    indices = np.array([3, 3, 7, 0, 5, 4, 6, 1], dtype=np.int64)
    batch_size = 3
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    got = collect(
        dataset,
        indices,
        batch_size,
        training=True,
        config=config,
        model=model,
    )
    assert [batch.batch_size for batch in got] == [3, 3, 2]
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)
        assert actual.stages is None and actual.target_kinds is None
        assert not isinstance(actual, tuple)
        for absent in ("eligible", "actor_seats", "decision_indices", "episode_ids"):
            assert not hasattr(actual, absent), absent
    # Delivered tensors stay valid after staging slots were released.
    assert got[0].targets.tolist() == expected[0].targets.tolist()
    assert not live_producers()


@CUDA
def test_prefetch_evaluation_batches_match_the_sync_path(model):
    dataset = freeze(make_dataset(split="validation", counts=[2, 1, 3, 4, 2], seed=47))
    indices = dataset.eligible_indices()
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=False),
    )
    expected = reference_batches(dataset, indices, batch_size, training=False)
    got = collect(
        dataset,
        indices,
        batch_size,
        training=False,
        config=config,
        model=model,
        eligible_only=True,
    )
    assert len(got) == len(expected) == 2
    for actual, wanted in zip(got, expected):
        assert_batches_equal(actual, wanted, EVAL_FIELDS)


@CUDA
def test_prefetch_sharded_split_matches_the_sync_path(model):
    dataset = sharded_split(first=(2, 1, 3), second=(2, 4))
    # Unsorted, duplicated and shard-spanning, with a tail batch.
    indices = np.array([3, 0, 4, 2, 4], dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    got = collect(
        dataset, indices, batch_size, training=True, config=config, model=model
    )
    assert [batch.batch_size for batch in got] == [2, 2, 1]
    for actual, wanted in zip(got, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)


@CUDA
def test_prefetch_uses_the_injected_batch_iterator_factory(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 3, 1, 4], seed=61))
    indices = dataset.eligible_indices()
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    calls: list[dict[str, object]] = []

    def factory(dataset, indices, **kwargs):
        calls.append(dict(kwargs))
        return _iter_model_batches(dataset, indices, **kwargs)

    expected = reference_batches(dataset, indices, batch_size, training=True)
    got = collect(
        dataset,
        indices,
        batch_size,
        training=True,
        config=config,
        model=model,
        batch_iterator=factory,
    )
    assert calls == [
        {
            "batch_size": batch_size,
            "num_workers": 0,
            "training": True,
            "packed_candidates": True,
        }
    ]
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)


# ---- pinned capacity ---------------------------------------------------


@CUDA
def test_pinned_budget_must_cover_the_two_slots_exactly(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 3, 4], seed=48))
    indices = dataset.eligible_indices()
    batch_size = 2
    required = pinned_bytes(dataset, indices, batch_size, training=True)
    assert required > 0
    too_small = BatchTransferConfig(
        mode="cuda-prefetch", pinned_limit_bytes=required - 1
    )
    with pytest.raises(ConfigurationError, match=f"need {required} bytes"):
        with device_batches(
            dataset,
            indices,
            batch_size=batch_size,
            training=True,
            packed_candidates=True,
            model=model,
            device="cuda",
            batch_transfer=too_small,
        ):
            pass
    assert not live_producers()

    exact = BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=required)
    got = collect(
        dataset, indices, batch_size, training=True, config=exact, model=model
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    for actual, wanted in zip(got, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)


@CUDA
def test_empty_traversal_creates_no_producer_and_no_slots(model):
    dataset = freeze(make_dataset(split="validation", counts=[1, 1, 2], seed=49))
    config = BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=1)
    with device_batches(
        dataset,
        np.empty(0, dtype=np.int64),
        batch_size=4,
        training=False,
        packed_candidates=True,
        eligible_only=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    ) as iterator:
        assert list(iterator) == []
        assert iterator._pipeline._slots == []
        assert iterator._pipeline._producer is None
    assert not live_producers()


@CUDA
@pytest.mark.parametrize("overflow", ["columns", "flat", "positions"])
def test_staging_rejects_batches_that_do_not_fit_the_slot(model, overflow):
    dataset = freeze(make_dataset(split="train", counts=[2, 2], seed=62))
    indices = np.array([0, 1], dtype=np.int64)
    batch_size = 1
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    base = next(
        iter(
            _iter_model_batches(
                dataset, [0], batch_size=1, training=True, packed_candidates=True
            )
        )
    )
    assert base.candidates.shape == (2, base.candidates.shape[1])
    if overflow == "columns":
        fabricated = replace(base, mask=torch.zeros(1, 3, dtype=torch.bool))
        match = "pinned slot capacity"
    elif overflow == "flat":
        fabricated = replace(
            base,
            candidates=torch.zeros(5, base.candidates.shape[1], dtype=torch.float32),
            candidate_positions=torch.zeros(5, dtype=torch.int64),
        )
        match = "pinned slot capacity"
    else:
        fabricated = replace(base, candidate_positions=torch.zeros(3, dtype=torch.int64))
        match = "positions do not match"

    def factory(dataset, indices, **kwargs):
        return iter([fabricated])

    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
        batch_iterator=factory,
    ) as iterator:
        with pytest.raises(DataError, match=match):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
    assert not live_producers()


# ---- lifecycle ---------------------------------------------------------


@CUDA
def test_producer_blocks_on_two_staged_slots_until_the_consumer_commits(model):
    dataset = freeze(make_dataset(split="train", counts=[2] * 6, seed=50))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 1
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    ) as iterator:
        pipeline = iterator._pipeline
        # Nothing is consumed, so no copy can free a slot: the producer must
        # stop after the two slots are staged instead of overwriting one.
        assert wait_for(lambda: len(pipeline._ready) == 2)
        time.sleep(0.05)
        assert [item.index for item in pipeline._ready] == [0, 1]
        assert [slot.state for slot in pipeline._slots] == ["staged", "staged"]
        batches = list(iterator)
    assert len(batches) == len(expected)
    for actual, wanted in zip(batches, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)
    assert not live_producers()


@CUDA
def test_exit_with_a_staged_lookahead_does_not_hang(model):
    dataset = freeze(make_dataset(split="train", counts=[2] * 5, seed=51))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 1
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    context = device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    )
    iterator = context.__enter__()
    first = next(iterator)
    assert wait_for(lambda: len(iterator._pipeline._ready) >= 1)
    with_deadline(lambda: context.__exit__(None, None, None))
    assert not live_producers()
    assert_batches_equal(first, expected[0], TRAIN_FIELDS)
    # The delivered batch remains valid after staging was released.
    assert torch.equal(first.states.cpu(), expected[0].states.cpu())


@CUDA
def test_exit_before_consuming_anything_does_not_hang(model):
    dataset = freeze(make_dataset(split="train", counts=[2] * 40, seed=52))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 4
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    context = device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    )
    iterator = context.__enter__()
    with_deadline(lambda: context.__exit__(None, None, None))
    assert not live_producers()
    with pytest.raises(StopIteration):
        next(iterator)


@CUDA
def test_released_slots_do_not_retain_pinned_or_device_memory(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 3] * 6, seed=53))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 3
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )

    def run_once() -> None:
        got = collect(
            dataset, indices, batch_size, training=True, config=config, model=model
        )
        expected = reference_batches(dataset, indices, batch_size, training=True)
        assert len(got) == len(expected)
        for actual, wanted in zip(got, expected):
            assert_batches_equal(actual, wanted, TRAIN_FIELDS)

    run_once()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    first = torch.cuda.memory_allocated()
    run_once()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() <= first
    assert not live_producers()


# ---- failures ----------------------------------------------------------


@CUDA
def test_host_failure_stops_the_traversal_in_batch_order(model):
    dataset = make_dataset(split="train", counts=[2] * 6, seed=55)
    dataset.targets[4] = 5
    freeze(dataset)
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices[:4], batch_size, training=True)
    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    ) as iterator:
        for wanted in expected:
            assert_batches_equal(next(iterator), wanted, TRAIN_FIELDS)
        with pytest.raises(DataError, match="outside its candidate range"):
            next(iterator)
        # A delivered failure ends the traversal instead of skipping ahead.
        with pytest.raises(StopIteration):
            next(iterator)
    assert not live_producers()


@CUDA
def test_device_failure_terminates_the_traversal(model, monkeypatch):
    dataset = freeze(make_dataset(split="train", counts=[2] * 6, seed=56))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    original = pipeline_module._copy_to_device
    calls = {"count": 0}

    def failing(tensor, device):
        calls["count"] += 1
        # The third submission (the lookahead for batch 1) fails, so batch 0 is
        # still delivered and nothing after it is.
        if calls["count"] == 2 * TRAIN_COPIES + 1:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return original(tensor, device)

    monkeypatch.setattr(pipeline_module, "_copy_to_device", failing)
    context = device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    )
    iterator = context.__enter__()
    delivered: list[_PackedCandidateBatch] = []
    failure: BaseException | None = None
    try:
        while True:
            delivered.append(next(iterator))
    except RuntimeError as error:
        failure = error
    except StopIteration:
        pass
    # The failing submission may be issued while delivering the lookahead, so
    # the two batches committed before it are still handed over. Nothing is
    # committed, staged or delivered after the device error.
    assert failure is not None and "CUDA error" in str(failure)
    assert 1 <= len(delivered) <= 2
    expected = reference_batches(dataset, indices, batch_size, training=True)
    for actual, wanted in zip(delivered, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)
    with pytest.raises(StopIteration):
        next(iterator)
    with_deadline(lambda: context.__exit__(None, None, None))
    assert not live_producers()


class FlakyCopyStream:
    """The real copy stream, synchronized for real, then reported as failed.

    ``close`` must release the staging it guards, so the wrapper always runs
    the underlying synchronization before raising.
    """

    def __init__(self, inner: torch.cuda.Stream) -> None:
        self.inner = inner
        self.synchronized = False

    def synchronize(self) -> None:
        self.inner.synchronize()
        self.synchronized = True
        raise RuntimeError("copy stream failed")


def leaky_pipeline_context(dataset, indices, batch_size, config, model):
    return device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    )


@CUDA
def test_clean_exit_reports_a_copy_stream_failure(model):
    dataset = freeze(make_dataset(split="train", counts=[2] * 4, seed=57))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    context = leaky_pipeline_context(dataset, indices, batch_size, config, model)
    iterator = context.__enter__()
    next(iterator)
    stream = FlakyCopyStream(iterator._pipeline._copy_stream)
    iterator._pipeline._copy_stream = stream
    # A clean exit must not pass a device error off as success: the completed
    # epoch would otherwise commit a checkpoint.
    with pytest.raises(RuntimeError, match="copy stream failed"):
        context.__exit__(None, None, None)
    assert stream.synchronized
    assert iterator._pipeline._slots == []
    assert not live_producers()


@CUDA
def test_body_error_wins_over_a_copy_stream_failure(model):
    dataset = freeze(make_dataset(split="train", counts=[2] * 4, seed=57))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    wrappers: list[FlakyCopyStream] = []
    with pytest.raises(ValueError, match="training stopped"):
        with leaky_pipeline_context(dataset, indices, batch_size, config, model) as (
            iterator
        ):
            next(iterator)
            wrapper = FlakyCopyStream(iterator._pipeline._copy_stream)
            wrappers.append(wrapper)
            iterator._pipeline._copy_stream = wrapper
            raise ValueError("training stopped")
    assert wrappers[0].synchronized
    assert wrappers[0].inner is not None
    assert not live_producers()


@CUDA
def test_close_error_after_real_teardown_does_not_mask_the_body_error(
    model, monkeypatch
):
    dataset = freeze(make_dataset(split="train", counts=[2] * 4, seed=63))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    real_close = pipeline_module._PrefetchPipeline.close

    def close_then_fail(self) -> None:
        try:
            real_close(self)
        finally:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(pipeline_module._PrefetchPipeline, "close", close_then_fail)
    with pytest.raises(ValueError, match="training stopped"):
        with leaky_pipeline_context(dataset, indices, batch_size, config, model) as (
            iterator
        ):
            next(iterator)
            raise ValueError("training stopped")
    # The real teardown ran before the injected failure, so nothing leaked.
    assert not live_producers()


@CUDA
def test_delivery_does_not_wait_for_a_producer_blocked_on_a_copy_event(
    model, monkeypatch
):
    dataset = freeze(make_dataset(split="train", counts=[2] * 8, seed=64))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 1
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    producer_waiting = threading.Event()
    release_producer = threading.Event()
    guarded_event = None
    real_synchronize = torch.cuda.Event.synchronize
    real_wait_event = torch.cuda.Stream.wait_event
    compute_waits = []

    def gated_synchronize(event):
        if event is guarded_event:
            # Keep the producer in its event wait independently of GPU clock
            # speed or host scheduling; still perform the real H2D wait.
            producer_waiting.set()
            if not release_producer.wait(20):
                raise TimeoutError("test did not release the producer event wait")
        return real_synchronize(event)

    def observe_wait(stream, event):
        compute_waits.append(event)
        return real_wait_event(stream, event)

    monkeypatch.setattr(torch.cuda.Event, "synchronize", gated_synchronize)
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", observe_wait)
    with leaky_pipeline_context(dataset, indices, batch_size, config, model) as iterator:
        pipeline = iterator._pipeline
        guarded_event = pipeline._slots[1].event
        try:
            assert wait_for(lambda: len(pipeline._ready) == 2)
            first = next(iterator)
            assert producer_waiting.wait(10)
            assert pipeline._lookahead is not None
            assert pipeline._slots[1].state == "in_flight"
            # Delivery enqueues exactly its own dependency, not a wait for
            # the lookahead copy or for the blocked producer to return.
            assert len(compute_waits) == 1
            assert compute_waits[0] is pipeline._slots[0].event
            assert len(pipeline._ready) <= 1
        finally:
            release_producer.set()
        batches = [first, *iterator]
    assert len(batches) == len(expected)
    for actual, wanted in zip(batches, expected):
        assert_batches_equal(actual, wanted, TRAIN_FIELDS)
    assert not live_producers()


@CUDA
def test_at_most_one_committed_lookahead_is_retained(model):
    dataset = freeze(make_dataset(split="train", counts=[2, 3] * 5, seed=65))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    with leaky_pipeline_context(dataset, indices, batch_size, config, model) as (
        iterator
    ):
        pipeline = iterator._pipeline
        for _ in iterator:
            # Host staging is bounded by the two slots, and at most one batch
            # beyond the delivered one is ever committed to the device.
            assert len(pipeline._ready) <= 2
            assert pipeline._current is None
            committed = (0 if pipeline._lookahead is None else 1) + (
                0 if pipeline._current is None else 1
            )
            assert committed <= 1
        assert pipeline._lookahead is None and pipeline._current is None
    assert not live_producers()


# ---- device placement and timing --------------------------------------


@CUDA
def test_delivered_batches_are_usable_on_a_non_default_compute_stream(model):
    dataset = freeze(make_dataset(split="train", counts=[3, 2, 4, 3], seed=58))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    device_model = model.to("cuda")
    expected = reference_batches(dataset, indices, batch_size, training=True)
    stream = torch.cuda.Stream()
    logits: list[torch.Tensor] = []
    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    ) as iterator:
        with torch.cuda.stream(stream):
            for batch in iterator:
                logits.append(_batch_logits(device_model, batch).detach().clone())
    stream.synchronize()
    with torch.cuda.stream(torch.cuda.default_stream()):
        reference = [_batch_logits(device_model, batch).detach() for batch in expected]
    for actual, wanted in zip(logits, reference):
        assert torch.equal(actual.cpu(), wanted.cpu())


@CUDA
def test_every_delivered_tensor_is_recorded_on_the_compute_stream(model, monkeypatch):
    dataset = freeze(make_dataset(split="train", counts=[2, 3, 4, 2], seed=67))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 2
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    recorded: list[tuple[int, int]] = []
    original = torch.Tensor.record_stream

    def spy(tensor, stream):
        recorded.append((id(tensor), stream.cuda_stream))
        return original(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)
    stream = torch.cuda.Stream()
    with leaky_pipeline_context(dataset, indices, batch_size, config, model) as (
        iterator
    ):
        with torch.cuda.stream(stream):
            for batch in iterator:
                assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream
                for name in TRAIN_FIELDS:
                    tensor = getattr(batch, name)
                    assert (id(tensor), stream.cuda_stream) in recorded, name
    assert not live_producers()


@CUDA
def test_freed_batches_stay_valid_for_pending_compute_kernels(model):
    """A delivered batch may be freed while its compute stream still reads it."""

    dataset = freeze(make_dataset(split="train", counts=[2, 3, 4, 3] * 4, seed=68))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 4
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    expected = reference_batches(dataset, indices, batch_size, training=True)
    compute = torch.cuda.Stream()
    pending: list[torch.Tensor] = []
    with leaky_pipeline_context(dataset, indices, batch_size, config, model) as (
        iterator
    ):
        copy_stream = iterator._pipeline._copy_stream
        with torch.cuda.stream(compute):
            for batch in iterator:
                # Queue a delay and then a kernel that reads the batch, so the
                # read is still pending when the host reference disappears.
                torch.cuda._sleep(30_000_000)
                pending.append(
                    (
                        batch.states.sum()
                        + batch.candidates.sum()
                        + batch.targets.to(torch.float32).sum()
                    ).detach()
                )
                state_shape = batch.states.shape
                del batch
                gc.collect()
                # Same-shape allocations on the allocation (copy) stream are
                # exactly what a missing record_stream would let reuse the block
                # while the pending kernel still reads it.
                with torch.cuda.stream(copy_stream):
                    scratch = torch.empty(
                        state_shape, dtype=torch.float32, device="cuda"
                    )
                    scratch.fill_(7.0)
                del scratch
                gc.collect()
    torch.cuda.synchronize()
    assert len(pending) == len(expected)
    for got, wanted in zip(pending, expected):
        reference = (
            wanted.states.sum()
            + wanted.candidates.sum()
            + wanted.targets.to(torch.float32).sum()
        )
        assert torch.allclose(got.cpu(), reference.cpu(), rtol=0, atol=1e-2)
    assert not live_producers()


@CUDA
def test_next_batch_copy_overlaps_the_current_batch_compute(model, monkeypatch):
    """Timeline evidence: batch n+1's copy is hidden behind batch n's compute."""

    counts = [2, 3, 4, 3] * 512
    dataset = freeze(make_dataset(split="train", counts=counts, seed=60))
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_size = 512
    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, batch_size, training=True),
    )
    copies: list[list[torch.cuda.Event | None]] = []
    original = pipeline_module._copy_to_device
    calls = {"count": 0}

    def spy(tensor, device):
        slot = calls["count"] % TRAIN_COPIES
        calls["count"] += 1
        if slot == 0:
            start = torch.cuda.Event(enable_timing=True)
            start.record()  # current stream inside a commit is the copy stream
            copies.append([start, None])
        result = original(tensor, device)
        if slot == TRAIN_COPIES - 1:
            # A queued device-side delay stands in for a longer transfer (a
            # bigger batch or a slower link) so the copy is still in flight
            # while the previous batch computes. The copy itself is unchanged.
            torch.cuda._sleep(2_000_000)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            copies[-1][1] = end
        return result

    monkeypatch.setattr(pipeline_module, "_copy_to_device", spy)
    computes: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with device_batches(
        dataset,
        indices,
        batch_size=batch_size,
        training=True,
        packed_candidates=True,
        model=model,
        device="cuda",
        batch_transfer=config,
    ) as iterator:
        # Let the producer fill both slots first so every later batch has a
        # staged lookahead: this measures steady-state supply, not startup.
        assert wait_for(lambda: len(iterator._pipeline._ready) == 2)
        for _ in iterator:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(5_000_000)  # a few ms of device work per batch
            end.record()
            computes.append((start, end))
    torch.cuda.synchronize()
    expected_batches = (int(indices.size) + batch_size - 1) // batch_size
    assert len(computes) == len(copies) == expected_batches > 2
    assert all(end is not None for _, end in copies)

    origin = computes[0][0]

    def span(start: torch.cuda.Event, end: torch.cuda.Event) -> tuple[float, float]:
        return origin.elapsed_time(start), origin.elapsed_time(end)

    # Both streams share one device clock, so every transfer and compute
    # interval can be placed on one timeline. The first batch's own copy is
    # excluded: only lookahead transfers can be concurrent with computation.
    copy_spans = [span(start, end) for start, end in copies[1:]]
    compute_spans = [span(start, end) for start, end in computes]
    overlap_ms = 0.0
    for copy_start, copy_end in copy_spans:
        for compute_start, compute_end in compute_spans:
            overlap_ms += max(
                0.0, min(copy_end, compute_end) - max(copy_start, compute_start)
            )
    copy_ms = sum(end - start for start, end in copy_spans)
    assert copy_ms > 0
    # A synchronous transfer runs entirely after the current batch's compute on
    # the same stream and cannot overlap at all; the pipelined transfers spend
    # a substantial share of their time running concurrently with computation.
    assert overlap_ms >= 0.3 * copy_ms, {
        "copy_spans_ms": [(round(a, 3), round(b, 3)) for a, b in copy_spans],
        "compute_spans_ms": [(round(a, 3), round(b, 3)) for a, b in compute_spans],
        "overlap_ms": round(overlap_ms, 3),
        "copy_ms": round(copy_ms, 3),
    }


@CUDA
def test_delivery_does_not_wait_for_an_unready_host_batch(model):
    dataset = freeze(make_dataset(counts=[2, 3, 2, 3], seed=61))
    indices = dataset.eligible_indices()
    waiting_for_second = threading.Event()
    release_second = threading.Event()

    def slow_factory(dataset, indices, **kwargs):
        for number, batch in enumerate(_iter_model_batches(dataset, indices, **kwargs)):
            if number == 1:
                waiting_for_second.set()
                if not release_second.wait(20):
                    raise TimeoutError("test did not release the second host batch")
            yield batch

    config = BatchTransferConfig(
        mode="cuda-prefetch",
        pinned_limit_bytes=pinned_bytes(dataset, indices, 2, training=True),
    )
    first = []
    with device_batches(
        dataset, indices, batch_size=2, training=True, packed_candidates=True,
        model=model, device="cuda", batch_transfer=config,
        batch_iterator=slow_factory,
    ) as iterator:
        try:
            assert waiting_for_second.wait(10)
            # The next host batch cannot become READY until explicitly released.
            # Delivering the current batch must not depend on that release.
            with_deadline(lambda: first.append(next(iterator)), timeout=5)
            assert iterator._pipeline._lookahead is None
        finally:
            release_second.set()
        rest = list(iterator)
    reference = reference_batches(dataset, indices, 2, training=True)
    for got, want in zip(first + rest, reference):
        assert_batches_equal(got, want, TRAIN_FIELDS)
    assert len(first + rest) == len(reference) == 2
