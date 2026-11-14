"""Owned multi-split loading contexts: mmap retention and contiguous RAM.

The RAM representation is checked against the manifest arrays, the legacy
sharded loader, and batch collation; the budget checks use manifest-derived
byte counts so the preflight, the recorded held bytes and the rejection
boundary stay equivalent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mahjong_bc import cache as cache_module
from mahjong_bc.cache import SplitInputs, load_prepared_cache
from mahjong_bc.constants import CANDIDATE_WIDTH, SPLIT_NAMES, STATE_WIDTH
from mahjong_bc.data import (
    PackedDecisionDataset,
    ShardedDecisionDataset,
    iter_batches,
)
from mahjong_bc.errors import CacheContractError, ConfigurationError

from .cache_fixture import read_manifest, write_cache

_I64 = np.dtype(np.int64).itemsize
_BOOL = np.dtype(np.bool_).itemsize
_METADATA_ARRAYS = (
    "targets",
    "episode_seeds",
    "decision_indices",
    "actor_seats",
    "stages",
    "target_kinds",
)


def _manifest_array_itemsize(manifest, name):
    return np.dtype(manifest["layout"]["arrays"][name]["dtype"]).itemsize


def _preflight_bytes(manifest, names):
    """Spec budget: destination arrays plus derived layout capacity.

    Derived capacity is one candidate count and one eligibility mask entry per
    decision plus a full-length eligible-index bound, so forced decisions only
    reduce the actual bytes, never the preflight.
    """

    total = 0
    for name in names:
        counts = manifest["splits"][name]["counts"]
        decisions = counts["decisions"]
        candidates = counts["candidates"]
        total += decisions * STATE_WIDTH * _manifest_array_itemsize(manifest, "states")
        total += (
            candidates
            * CANDIDATE_WIDTH
            * _manifest_array_itemsize(manifest, "candidate_features")
        )
        total += (decisions + 1) * _manifest_array_itemsize(
            manifest, "candidate_offsets"
        )
        total += decisions * sum(
            _manifest_array_itemsize(manifest, array) for array in _METADATA_ARRAYS
        )
        total += decisions * (_I64 + _BOOL + _I64)
    return total


def _held_bytes(dataset: PackedDecisionDataset) -> int:
    arrays = (
        dataset.states,
        dataset.candidate_features,
        dataset.offsets,
        dataset.targets,
        dataset.stages,
        dataset.actor_seats,
        dataset.decision_indices,
        dataset.episode_ids,
        dataset.target_kinds,
        dataset.candidate_counts(),
        dataset.eligible_mask(),
        dataset.eligible_indices(),
    )
    return sum(int(np.asarray(array).nbytes) for array in arrays)


def _manifest_array(root, manifest, split, name):
    """Concatenate one wire array over a split's shards in manifest order."""

    parts = [
        np.load(root / shard["arrays"][name]["file"], allow_pickle=False)
        for shard in manifest["splits"][split]["shards"]
    ]
    if not parts:
        return None
    return np.concatenate(parts)


def _assert_closed(dataset) -> None:
    assert dataset._closed is True
    assert dataset.identity is not None
    with pytest.raises(Exception, match="closed"):
        dataset.forced_by_stage()
    if isinstance(dataset, ShardedDecisionDataset):
        for shard in dataset.shards:
            assert shard.states is None
    else:
        assert dataset.states is None


def _batch_fields(batch):
    return {
        "states": batch.states,
        "candidates": batch.candidates,
        "mask": batch.mask,
        "targets": batch.targets,
        "stages": batch.stages,
        "target_kinds": batch.target_kinds,
        "eligible": batch.eligible,
        "actor_seats": batch.actor_seats,
        "decision_indices": batch.decision_indices,
        "episode_ids": batch.episode_ids,
    }


def _assert_batches_equal(actual, expected) -> None:
    fields = _batch_fields(actual)
    other = _batch_fields(expected)
    assert set(fields) == set(other)
    for name, value in fields.items():
        assert (value is None) == (other[name] is None), name
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other[name]), name
        else:
            assert value == other[name], name


class _ExplodingDataset:
    """Dataset whose cleanup fails; used to check error precedence."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = RuntimeError("close boom") if error is None else error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise self.error


def _open_inputs(cache, monkeypatch, *, storage, dataset):
    def fake_materialize(cache_, name, *, storage):
        return dataset, None

    monkeypatch.setattr(cache_module, "_materialize_split", fake_materialize)
    return cache.open_splits(["train"], storage=storage)


def test_mmap_context_retains_sharded_layout_and_closes_on_exit(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
    inputs = cache.open_splits(["train", "test"], storage="mmap")
    assert isinstance(inputs, SplitInputs)
    assert not inputs.closed
    assert list(inputs.datasets) == ["train", "test"]
    assert inputs.loading_record == {
        "storage": "mmap",
        "ram_limit_bytes": None,
        "ram_bytes": None,
        "splits": {
            "train": {"representation": "sharded", "ram_bytes": None},
            "test": {"representation": "sharded", "ram_bytes": None},
        },
    }
    datasets = dict(inputs.datasets)
    with inputs as entered:
        assert entered is inputs
        for name, dataset in datasets.items():
            assert type(dataset) is ShardedDecisionDataset
            assert dataset.identity == cache.split_identity(name)
            assert dataset.identity.split == name
            assert len(dataset) == cache.splits[name].counts.decisions
            assert dataset.is_memory_mapped
            assert not any(
                array is None
                for shard in dataset.shards
                for array in (shard.states, shard.candidate_features)
            )
            for shard in dataset.shards:
                assert shard.states.flags.writeable is False
                assert shard.candidate_features.flags.writeable is False
                assert isinstance(shard.states, np.memmap)
    assert inputs.closed
    for dataset in datasets.values():
        _assert_closed(dataset)
    # The record describes the invocation and stays readable after exit.
    assert inputs.loading_record["storage"] == "mmap"


@pytest.mark.parametrize("storage", ["mmap", "ram"])
def test_open_splits_without_names_opens_nothing(shared_cache, storage):
    cache = load_prepared_cache(shared_cache.root)
    kwargs = {} if storage == "mmap" else {"ram_limit_bytes": 1}
    with cache.open_splits([], storage=storage, **kwargs) as inputs:
        assert inputs.datasets == {}
        assert inputs.loading_record == {
            "storage": storage,
            "ram_limit_bytes": None if storage == "mmap" else 1,
            "ram_bytes": None if storage == "mmap" else 0,
            "splits": {},
        }


def test_ram_packing_matches_manifest_arrays_and_legacy_loader(shared_cache):
    root = shared_cache.root
    manifest = read_manifest(root)
    cache = load_prepared_cache(root)
    with cache.open_splits(["train"], storage="ram", ram_limit_bytes=1 << 26) as inputs:
        dataset = inputs.datasets["train"]
        assert type(dataset) is PackedDecisionDataset
        assert dataset.identity == cache.split_identity("train")
        assert not dataset.is_memory_mapped
        assert len(dataset) == cache.splits["train"].counts.decisions
        assert dataset.offsets.dtype == np.int64
        assert dataset.states.dtype == np.float32
        assert dataset.candidate_features.dtype == np.float32
        assert dataset.episode_ids.dtype == np.uint64
        assert dataset.stages.dtype == np.uint8
        # Shard-local offsets are shifted by prior candidates and de-duplicated.
        shards = manifest["splits"]["train"]["shards"]
        merged = [np.zeros(1, dtype=np.int64)]
        candidate_total = 0
        for shard in shards:
            local = np.load(
                root / shard["arrays"]["candidate_offsets"]["file"],
                allow_pickle=False,
            )
            merged.append(local[1:] + candidate_total)
            candidate_total += int(local[-1])
        np.testing.assert_array_equal(dataset.offsets, np.concatenate(merged))
        for name, attribute in (
            ("states", "states"),
            ("candidate_features", "candidate_features"),
            ("targets", "targets"),
            ("stages", "stages"),
            ("actor_seats", "actor_seats"),
            ("decision_indices", "decision_indices"),
            ("episode_seeds", "episode_ids"),
            ("target_kinds", "target_kinds"),
        ):
            np.testing.assert_array_equal(
                getattr(dataset, attribute),
                _manifest_array(root, manifest, "train", name),
            )
        legacy = cache.load_split("train", mmap=False)
        try:
            assert dataset.eligible_count == legacy.eligible_count
            assert dataset.forced_count == legacy.forced_count
            np.testing.assert_array_equal(
                dataset.candidate_counts(), legacy.candidate_counts()
            )
            np.testing.assert_array_equal(
                dataset.eligible_indices(), legacy.eligible_indices()
            )
            np.testing.assert_array_equal(
                dataset.eligible_mask(), legacy.eligible_mask()
            )
            assert dataset.forced_by_stage() == legacy.forced_by_stage()
            for index in range(len(dataset)):
                ram_sample = dataset[index]
                plain_sample = legacy[index]
                np.testing.assert_array_equal(ram_sample.state, plain_sample.state)
                np.testing.assert_array_equal(
                    ram_sample.candidate_features, plain_sample.candidate_features
                )
                assert ram_sample.target == plain_sample.target
                assert ram_sample.stage == plain_sample.stage
                assert ram_sample.target_kind == plain_sample.target_kind
                assert ram_sample.actor_seat == plain_sample.actor_seat
                assert ram_sample.decision_index == plain_sample.decision_index
                assert int(ram_sample.episode_id) == int(plain_sample.episode_id)
        finally:
            legacy.close()


def test_ram_arrays_and_derived_layout_are_read_only(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
    with cache.open_splits(["validation"], storage="ram", ram_limit_bytes=1 << 26) as inputs:
        dataset = inputs.datasets["validation"]
        arrays = (
            dataset.states,
            dataset.candidate_features,
            dataset.offsets,
            dataset.targets,
            dataset.stages,
            dataset.actor_seats,
            dataset.decision_indices,
            dataset.episode_ids,
            dataset.target_kinds,
            dataset.candidate_counts(),
            dataset.eligible_mask(),
            dataset.eligible_indices(),
        )
        for array in arrays:
            assert np.asarray(array).flags.writeable is False
        with pytest.raises(ValueError):
            dataset.states[0, 0] = 1.0
        with pytest.raises(ValueError):
            dataset.targets[0] = 0


def test_ram_batches_match_sharded_batches_for_unsorted_duplicates(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
    sharded = cache.load_split("train")
    try:
        with cache.open_splits(
            ["train"], storage="ram", ram_limit_bytes=1 << 26
        ) as inputs:
            packed = inputs.datasets["train"]
            assert type(packed) is PackedDecisionDataset
            indices = [3, 0, 2, 0, 5, len(packed) - 1]
            for batch_size in (1, 2, 4, len(indices) + 3):
                ram_batches = list(
                    iter_batches(packed, indices, batch_size=batch_size)
                )
                legacy_batches = list(
                    iter_batches(sharded, indices, batch_size=batch_size)
                )
                assert len(ram_batches) == len(legacy_batches)
                for actual, expected in zip(ram_batches, legacy_batches):
                    assert actual.batch_size == expected.batch_size
                    _assert_batches_equal(actual, expected)
            eligible = packed.eligible_indices()
            ram_batches = list(
                iter_batches(packed, eligible, batch_size=3)
            )
            legacy_batches = list(
                iter_batches(sharded, eligible, batch_size=3)
            )
            assert len(ram_batches) == len(legacy_batches)
            for actual, expected in zip(ram_batches, legacy_batches):
                _assert_batches_equal(actual, expected)
    finally:
        sharded.close()


def test_ram_empty_split_uses_legal_empty_arrays(tmp_path):
    fixture = write_cache(tmp_path, seed_start=0, games=3)
    assert fixture.seeds["test"] == ()
    cache = load_prepared_cache(tmp_path)
    budget = _preflight_bytes(read_manifest(tmp_path), ["test"])
    assert budget == _I64  # only the single offset boundary remains
    with cache.open_splits(["test"], storage="ram", ram_limit_bytes=budget) as inputs:
        dataset = inputs.datasets["test"]
        assert type(dataset) is PackedDecisionDataset
        assert len(dataset) == 0
        assert dataset.identity.split == "test"
        assert dataset.identity.decisions == 0
        assert dataset.identity.eligible_decisions == 0
        assert dataset.states.shape == (0, STATE_WIDTH)
        assert dataset.candidate_features.shape == (0, CANDIDATE_WIDTH)
        np.testing.assert_array_equal(dataset.offsets, np.zeros(1, dtype=np.int64))
        assert dataset.targets.size == 0
        assert dataset.candidate_counts().size == 0
        assert dataset.eligible_indices().size == 0
        assert dataset.forced_by_stage() == (0, 0, 0, 0, 0)
        assert list(iter_batches(dataset, dataset.eligible_indices(), batch_size=4)) == []
        with pytest.raises(IndexError):
            dataset[0]
        assert inputs.loading_record["splits"]["test"] == {
            "representation": "packed",
            "ram_bytes": _I64,
        }
        assert inputs.loading_record["ram_bytes"] == _I64


def test_record_reports_actual_held_bytes_not_mmap_sizes_or_preflight(shared_cache):
    root = shared_cache.root
    manifest = read_manifest(root)
    cache = load_prepared_cache(root)
    names = ["train", "validation"]
    budget = _preflight_bytes(manifest, names)
    with cache.open_splits(names, storage="ram", ram_limit_bytes=budget) as inputs:
        record = inputs.loading_record
        actual = sum(_held_bytes(inputs.datasets[name]) for name in names)
        assert record["ram_bytes"] == actual
        assert record["ram_bytes"] == sum(
            record["splits"][name]["ram_bytes"] for name in names
        )
        assert record["ram_limit_bytes"] == budget
        assert list(record["splits"]) == names
        for name in names:
            assert record["splits"][name]["representation"] == "packed"
        # Forced decisions shrink the derived eligible-index capacity, so the
        # held bytes are strictly below the conservative preflight.
        forced = sum(
            cache.splits[name].counts.decisions
            - cache.splits[name].counts.eligible_decisions
            for name in names
        )
        assert forced > 0
        assert budget - record["ram_bytes"] == forced * _I64
        assert record["ram_bytes"] > 0


@pytest.mark.parametrize("storage", ["mmap", "ram"])
def test_budget_exactly_equal_to_preflight_is_accepted(shared_cache, storage):
    root = shared_cache.root
    manifest = read_manifest(root)
    cache = load_prepared_cache(root)
    names = list(SPLIT_NAMES)
    required = _preflight_bytes(manifest, names)
    if storage == "mmap":
        with cache.open_splits(names, storage=storage) as inputs:
            assert inputs.loading_record["ram_bytes"] is None
            assert all(
                entry["ram_bytes"] is None
                for entry in inputs.loading_record["splits"].values()
            )
        return
    with cache.open_splits(names, storage=storage, ram_limit_bytes=required) as inputs:
        assert inputs.loading_record["ram_bytes"] <= required
        with pytest.raises(ConfigurationError, match="ram_limit_bytes"):
            cache.open_splits(names, storage=storage, ram_limit_bytes=required - 1)


def test_combined_budget_must_cover_every_requested_split(shared_cache):
    root = shared_cache.root
    manifest = read_manifest(root)
    cache = load_prepared_cache(root)
    train_only = _preflight_bytes(manifest, ["train"])
    with cache.open_splits(["train"], storage="ram", ram_limit_bytes=train_only):
        pass
    with pytest.raises(ConfigurationError, match="validation"):
        cache.open_splits(
            ["train", "validation"], storage="ram", ram_limit_bytes=train_only
        )
    both = _preflight_bytes(manifest, ["train", "validation"])
    with cache.open_splits(
        ["train", "validation"], storage="ram", ram_limit_bytes=both
    ):
        pass


def test_budget_rejection_happens_before_any_materialization(shared_cache, monkeypatch):
    root = shared_cache.root
    manifest = read_manifest(root)
    cache = load_prepared_cache(root)
    requested = []

    def refuse(*args, **kwargs):
        requested.append(args)
        raise AssertionError("materialization must not start")

    monkeypatch.setattr(cache_module, "_materialize_split", refuse)
    with pytest.raises(ConfigurationError, match="cannot hold"):
        cache.open_splits(
            ["train"],
            storage="ram",
            ram_limit_bytes=_preflight_bytes(manifest, ["train"]) - 1,
        )
    assert requested == []


@pytest.mark.parametrize(
    "budget", [None, 0, -1, True, False, 1.5, "1024", [1]]
)
def test_ram_storage_requires_a_positive_integer_budget(shared_cache, budget):
    cache = load_prepared_cache(shared_cache.root)
    with pytest.raises(ConfigurationError, match="ram_limit_bytes"):
        cache.open_splits(["train"], storage="ram", ram_limit_bytes=budget)


@pytest.mark.parametrize("budget", [0, 1, 1 << 26])
def test_mmap_storage_rejects_any_budget(shared_cache, budget):
    cache = load_prepared_cache(shared_cache.root)
    with pytest.raises(ConfigurationError, match="does not accept ram_limit_bytes"):
        cache.open_splits(["train"], storage="mmap", ram_limit_bytes=budget)


@pytest.mark.parametrize("storage", ["disk", "", "RAM", "mmap ", None, b"mmap", 1])
def test_storage_must_be_a_supported_mode(shared_cache, storage):
    cache = load_prepared_cache(shared_cache.root)
    with pytest.raises(ConfigurationError, match="storage"):
        cache.open_splits(["train"], storage=storage, ram_limit_bytes=1 << 20)


@pytest.mark.parametrize(
    "names",
    [["bogus"], ["train", "train"], "train", 5, ["train", 3], None],
    ids=["unknown", "duplicate", "string", "integer", "integer-name", "none"],
)
def test_names_must_be_distinct_known_splits(shared_cache, names):
    cache = load_prepared_cache(shared_cache.root)
    with pytest.raises(ConfigurationError):
        cache.open_splits(names, storage="mmap")


@pytest.mark.parametrize("storage", ["mmap", "ram"])
def test_build_failure_closes_already_materialized_datasets(
    fresh_cache, monkeypatch, storage
):
    root = fresh_cache.root
    cache = load_prepared_cache(root)
    manifest = read_manifest(root)
    victim = manifest["splits"]["validation"]["shards"][0]["arrays"]["states"]["file"]
    (root / victim).unlink()
    real = cache_module._materialize_split
    built = []

    def recording(cache_, name, *, storage):
        dataset, held = real(cache_, name, storage=storage)
        built.append(dataset)
        return dataset, held

    monkeypatch.setattr(cache_module, "_materialize_split", recording)
    kwargs = {} if storage == "mmap" else {"ram_limit_bytes": 1 << 26}
    with pytest.raises(CacheContractError, match="failed to load"):
        cache.open_splits(["train", "validation"], storage=storage, **kwargs)
    assert len(built) == 1
    _assert_closed(built[0])


def test_ram_shape_guard_rejects_arrays_changed_after_validation(fresh_cache):
    root = fresh_cache.root
    cache = load_prepared_cache(root)
    manifest = read_manifest(root)
    descriptor = manifest["splits"]["validation"]["shards"][0]["arrays"]["states"]
    np.save(root / descriptor["file"], np.zeros((0, STATE_WIDTH), dtype=np.float32))
    with pytest.raises(CacheContractError, match="manifest structure requires"):
        cache.open_splits(["validation"], storage="ram", ram_limit_bytes=1 << 26)


def test_build_failure_does_not_mask_cleanup_errors(shared_cache, monkeypatch):
    cache = load_prepared_cache(shared_cache.root)
    exploding = _ExplodingDataset()
    calls = []

    def fake_materialize(cache_, name, *, storage):
        calls.append(name)
        if name == "validation":
            raise CacheContractError("synthetic build failure")
        return exploding, None

    monkeypatch.setattr(cache_module, "_materialize_split", fake_materialize)
    with pytest.raises(CacheContractError, match="synthetic build failure"):
        cache.open_splits(["train", "validation"], storage="mmap")
    assert calls == ["train", "validation"]
    assert exploding.close_calls == 1


def test_exit_cleanup_error_does_not_mask_body_exception(shared_cache, monkeypatch):
    cache = load_prepared_cache(shared_cache.root)
    exploding = _ExplodingDataset()
    inputs = _open_inputs(cache, monkeypatch, storage="mmap", dataset=exploding)
    with pytest.raises(ValueError, match="body boom"):
        with inputs:
            raise ValueError("body boom")
    assert inputs.closed
    assert exploding.close_calls == 1


def test_exit_cleanup_error_propagates_without_body_exception(
    shared_cache, monkeypatch
):
    cache = load_prepared_cache(shared_cache.root)
    exploding = _ExplodingDataset()
    inputs = _open_inputs(cache, monkeypatch, storage="mmap", dataset=exploding)
    with pytest.raises(RuntimeError, match="close boom"):
        with inputs:
            pass
    assert inputs.closed
    assert exploding.close_calls == 1
    # Closing again is a no-op and does not repeat the cleanup error.
    inputs.close()
    assert exploding.close_calls == 1


def test_legacy_loading_stays_mutable_and_records_nothing(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
    assert not hasattr(cache, "loading_record")
    for mmap in (True, False):
        dataset = cache.load_split("train", mmap=mmap)
        try:
            assert not hasattr(dataset, "loading_record")
            assert not hasattr(dataset, "datasets")
            shard = dataset.shards[0]
            if mmap:
                assert shard.states.flags.writeable is False
            else:
                assert shard.states.flags.writeable is True
                assert shard.targets.flags.writeable is True
                shard.targets[0] = int(shard.targets[0])
                shard.states[0, 0] = float(shard.states[0, 0])
        finally:
            dataset.close()
