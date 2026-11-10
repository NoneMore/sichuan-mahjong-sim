from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

import mahjong_bc
from mahjong_bc.cache import load_prepared_cache
from mahjong_bc.config import ModelConfig
from mahjong_bc.data import iter_batches
from mahjong_bc.errors import CacheContractError, DataError
from mahjong_bc.metrics import evaluate_dataset

from .cache_fixture import read_manifest, write_cache, write_manifest
from .helpers import make_dataset

SMALL_MODEL = ModelConfig(state_hidden=(8,), scorer_hidden=(8,))


def _sources():
    train = make_dataset(
        split="train",
        counts=[2, 3, 1, 2, 3, 1, 2, 3, 1],
        seed=31,
        cache_id="source-cache",
        signal=True,
    )
    validation = make_dataset(
        split="validation",
        counts=[2, 1, 2, 1, 2],
        seed=32,
        cache_id="source-cache",
        signal=True,
    )
    test = make_dataset(
        split="test",
        counts=[2, 2, 1],
        seed=33,
        cache_id="source-cache",
        signal=True,
    )
    return train, validation, test


def _cache(root: Path, **kwargs):
    sources = _sources()
    write_cache(
        root,
        train=sources[0],
        validation=sources[1],
        test=sources[2],
        decisions_per_episode=2,
        episodes_per_shard=2,
        **kwargs,
    )
    return sources


def _rewrite_array(root: Path, manifest: dict, split: str, shard_index: int, name: str, array: np.ndarray) -> None:
    relative = f"arrays/{split}/{shard_index:05d}.{name}.npy"
    np.save(root / relative, array, allow_pickle=False)
    descriptor = manifest["splits"][split]["shards"][shard_index]["arrays"][name]
    descriptor["dtype"] = array.dtype.name
    descriptor["shape"] = list(array.shape)
    descriptor["bytes"] = (root / relative).stat().st_size
    descriptor["sha256"] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_manifest(root, manifest)


def test_loads_sharded_splits_and_preserves_decisions(tmp_path):
    sources = _sources()
    write_cache(
        tmp_path,
        train=sources[0],
        validation=sources[1],
        test=sources[2],
        decisions_per_episode=2,
        episodes_per_shard=2,
    )
    cache = load_prepared_cache(tmp_path)
    assert cache.cache_id == "synthetic-cache-id"
    assert cache.encoding_version == 1
    assert cache.preparation_format_version == 1
    assert cache.preparation_version == 1
    assert cache.split_algorithm == "seed-sha256-80-10-10-v1"
    assert len(cache.splits["train"].shards) == 3
    assert cache.counts.decisions == 17

    for split, source in zip(("train", "validation", "test"), sources):
        dataset = cache.load_split(split)
        try:
            assert len(dataset) == len(source)
            assert dataset.identity.cache_id == cache.cache_id
            assert dataset.identity.split == split
            assert dataset.identity.decisions == len(source)
            assert dataset.identity.eligible_decisions == source.identity.eligible_decisions
            for index in range(len(source)):
                sample = dataset[index]
                start = int(source.offsets[index])
                end = int(source.offsets[index + 1])
                assert np.array_equal(sample.state, source.states[index])
                assert np.array_equal(sample.candidate_features, source.candidate_features[start:end])
                assert sample.target == int(source.targets[index])
                assert sample.stage == int(source.stages[index])
                assert sample.target_kind == int(source.target_kinds[index])
                assert sample.actor_seat == index % 4
                assert sample.candidate_count == end - start
            assert dataset.eligible_count == source.identity.eligible_decisions
            assert dataset.forced_by_stage() == source.forced_by_stage()
        finally:
            dataset.close()


def test_empty_split_has_zero_counts_and_no_shards(tmp_path):
    sources = _sources()
    write_cache(tmp_path, train=sources[0], validation=sources[1], test=None)
    manifest = read_manifest(tmp_path)
    assert manifest["splits"]["test"]["shards"] == []
    cache = load_prepared_cache(tmp_path)
    dataset = cache.load_split("test")
    try:
        assert len(dataset) == 0
        assert dataset.eligible_count == 0
        assert dataset.forced_by_stage() == (0, 0, 0, 0, 0)
        assert cache.splits["test"].counts.to_dict() == {
            "episodes": 0,
            "decisions": 0,
            "candidates": 0,
            "forced_decisions": 0,
            "eligible_decisions": 0,
            "min_candidates": 0,
            "max_candidates": 0,
        }
        report = evaluate_dataset(
            mahjong_bc.build_policy(SMALL_MODEL),
            dataset,
            batch_size=4,
            device="cpu",
        )
        assert report.decisions == 0
        assert report.nll is None
        assert report.top1 is None
    finally:
        dataset.close()


def test_mmap_and_plain_loading_agree(tmp_path):
    _cache(tmp_path)
    cache = load_prepared_cache(tmp_path)
    mmapped = cache.load_split("train", mmap=True)
    plain = cache.load_split("train", mmap=False)
    try:
        assert isinstance(mmapped.shards[0].states, np.memmap)
        assert not isinstance(plain.shards[0].states, np.memmap)
        assert np.array_equal(mmapped[3].state, plain[3].state)
        assert np.array_equal(mmapped[3].candidate_features, plain[3].candidate_features)
    finally:
        mmapped.close()
        plain.close()


def test_memmap_datasets_reject_worker_processes(tmp_path):
    _cache(tmp_path)
    cache = load_prepared_cache(tmp_path)
    dataset = cache.load_split("train", mmap=True)
    try:
        with pytest.raises(DataError, match="memory-mapped"):
            list(
                iter_batches(
                    dataset,
                    dataset.eligible_indices()[:2],
                    batch_size=1,
                    num_workers=1,
                )
            )
    finally:
        dataset.close()


def test_inventory_is_complete_and_portable(tmp_path):
    root = tmp_path / "first"
    _cache(root)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    cache = load_prepared_cache(moved)
    assert cache.cache_id == "synthetic-cache-id"
    dataset = cache.load_split("test")
    try:
        assert len(dataset) == 3
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "wrong-format"),
        ("preparation_format_version", 2),
        ("preparation_version", 2),
        ("encoding_version", 2),
        ("split_algorithm", "other-split"),
        ("cache_id", ""),
        ("source_revision", ""),
        ("corpus_manifest_sha256", "not-a-hash"),
        ("layout", []),
        ("provenance", "no"),
        ("corpus_manifest", []),
    ],
)
def test_rejects_unsupported_or_malformed_manifest_fields(tmp_path, field, value):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest[field] = value
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)


def test_rejects_layout_that_differs_from_preparation_format_one(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["layout"]["state_width"] = 1000
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)

    other = tmp_path / "layout-array"
    _cache(other)
    manifest = read_manifest(other)
    manifest["layout"]["arrays"]["states"]["dimensions"] = ["decisions"]
    write_manifest(other, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(other)


def test_rejects_unknown_and_missing_manifest_keys(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["unexpected"] = 1
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)
    manifest = read_manifest(tmp_path)
    del manifest["unexpected"]
    del manifest["counts"]
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)


def test_rejects_missing_extra_and_duplicate_inventory(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)
    (tmp_path / "extra.txt").unlink()

    first = tmp_path / manifest["splits"]["train"]["shards"][0]["arrays"]["states"]["file"]
    first.unlink()
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)

    other = tmp_path / "duplicate"
    _cache(other)
    text = (other / "dataset.json").read_text(encoding="utf-8")
    text = text.replace(
        '"format": "mahjong-bc-dataset",',
        '"format": "mahjong-bc-dataset",\n  "format": "mahjong-bc-dataset",',
        1,
    )
    (other / "dataset.json").write_text(text, encoding="utf-8")
    with pytest.raises(CacheContractError):
        load_prepared_cache(other)


def test_checksum_verification_and_optional_skip(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    relative = manifest["splits"]["train"]["shards"][0]["arrays"]["states"]["file"]
    path = tmp_path / relative
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)
    cache = load_prepared_cache(tmp_path, verify_checksums=False)
    assert cache.cache_id == "synthetic-cache-id"


def test_rejects_array_layout_and_target_errors(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    shard = manifest["splits"]["train"]["shards"][0]
    states = np.load(tmp_path / shard["arrays"]["states"]["file"])
    _rewrite_array(tmp_path, manifest, "train", 0, "states", states.astype(np.float64))
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)

    root = tmp_path / "target-out-of-range"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    targets = np.load(root / shard["arrays"]["targets"]["file"])
    targets[0] = 999
    _rewrite_array(root, manifest, "train", 0, "targets", targets)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)

    root = tmp_path / "empty-offset"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    offsets = np.load(root / shard["arrays"]["candidate_offsets"]["file"])
    offsets[1] = offsets[0]
    _rewrite_array(root, manifest, "train", 0, "candidate_offsets", offsets)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)

    root = tmp_path / "kind-mismatch"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    kinds = np.load(root / shard["arrays"]["target_kinds"]["file"])
    kinds[0] = (int(kinds[0]) + 1) % 9
    _rewrite_array(root, manifest, "train", 0, "target_kinds", kinds)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)

    root = tmp_path / "fortran-order"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    states = np.load(root / shard["arrays"]["states"]["file"])
    relative = shard["arrays"]["states"]["file"]
    np.save(root / relative, np.asfortranarray(states), allow_pickle=False)
    descriptor = manifest["splits"]["train"]["shards"][0]["arrays"]["states"]
    descriptor["bytes"] = (root / relative).stat().st_size
    descriptor["sha256"] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def test_rejects_bad_counts_and_episode_metadata(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["splits"]["train"]["counts"]["decisions"] += 1
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)

    root = tmp_path / "duplicate-episode"
    _cache(root)
    manifest = read_manifest(root)
    first_seed = np.load(
        root / manifest["splits"]["train"]["shards"][0]["arrays"]["episode_seeds"]["file"]
    )[0]
    second = manifest["splits"]["train"]["shards"][1]
    seeds = np.load(root / second["arrays"]["episode_seeds"]["file"])
    seeds[0] = first_seed
    _rewrite_array(root, manifest, "train", 1, "episode_seeds", seeds)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def test_trainer_rejects_memmap_worker_processes_before_training(tmp_path):
    from mahjong_bc.config import ModelConfig, TrainingConfig
    from mahjong_bc.train import Trainer

    _cache(tmp_path)
    cache = load_prepared_cache(tmp_path)
    train = cache.load_split("train", mmap=True)
    validation = cache.load_split("validation", mmap=True)
    try:
        with pytest.raises(DataError, match="memory-mapped"):
            Trainer(
                train_dataset=train,
                validation_dataset=validation,
                model_config=ModelConfig(state_hidden=(8,), scorer_hidden=(8,)),
                training_config=TrainingConfig(
                    seed=1, epochs=1, batch_size=2, learning_rate=0.01, num_workers=1
                ),
                output_dir=tmp_path / "workers",
                device="cpu",
            )
    finally:
        train.close()
        validation.close()
