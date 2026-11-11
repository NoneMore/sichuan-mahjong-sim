from __future__ import annotations

import numpy as np

from mahjong_bc.cache import load_prepared_cache
from mahjong_bc.constants import SPLIT_NAMES

from .cache_fixture import (
    copy_cache,
    episode_seeds,
    read_manifest,
    write_cache,
    write_manifest,
)


def test_default_interval_uses_actual_frozen_split_coverage(shared_cache):
    fixture = shared_cache

    assert (fixture.seed_start, fixture.games) == (148, 10)
    assert fixture.seeds == {
        "train": (148, 149, 151, 152, 155),
        "validation": (153, 156, 157),
        "test": (150, 154),
    }
    assert {split: len(fixture.seeds[split]) for split in SPLIT_NAMES} == {
        "train": 5,
        "validation": 3,
        "test": 2,
    }
    manifest = read_manifest(fixture.root)
    assert manifest["corpus_manifest"]["seed_start"] == f"{148:016x}"
    assert manifest["counts"]["episodes"] == 10
    assert manifest["counts"]["decisions"] == 20
    for split in SPLIT_NAMES:
        source = fixture.sources[split]
        assert source is not None
        assert fixture.counts[split].episodes == len(fixture.seeds[split])
        assert fixture.counts[split].decisions == 2 * len(fixture.seeds[split])
        assert manifest["splits"][split]["counts"] == fixture.counts[split].to_dict()
        assert source.identity.split == split
        assert len(source) == fixture.counts[split].decisions


def test_private_copy_has_independent_manifest_and_container_fields(
    tmp_path, shared_cache
):
    copied = copy_cache(shared_cache, tmp_path / "private")

    assert copied.manifest == shared_cache.manifest
    assert copied.manifest is not shared_cache.manifest
    assert copied.seeds is not shared_cache.seeds
    assert copied.sources is not shared_cache.sources
    assert copied.counts is not shared_cache.counts
    copied.manifest["cache_id"] = "mutated-copy"
    copied.seeds["train"] = ()
    write_manifest(copied.root, copied.manifest)

    assert read_manifest(copied.root)["cache_id"] == "mutated-copy"
    assert shared_cache.manifest["cache_id"] == "synthetic-cache-id"
    assert shared_cache.seeds["train"] == (148, 149, 151, 152, 155)


def test_large_fixed_interval_generates_without_composition_search(tmp_path):
    fixture = write_cache(
        tmp_path / "large",
        seed_start=300,
        games=100,
        episodes_per_shard=8,
    )

    expected_episodes = {"train": 82, "validation": 10, "test": 8}
    assert {
        split: len(fixture.seeds[split]) for split in SPLIT_NAMES
    } == expected_episodes
    assert all(
        fixture.seeds[split] == tuple(sorted(fixture.seeds[split]))
        for split in SPLIT_NAMES
    )

    cache = load_prepared_cache(fixture.root)
    assert cache.counts.episodes == 100
    assert cache.counts.decisions == 200
    assert len(cache.splits["train"].shards) > 1
    for split in SPLIT_NAMES:
        source = fixture.sources[split]
        assert source is not None
        assert fixture.counts[split].episodes == expected_episodes[split]
        assert cache.splits[split].counts == fixture.counts[split]
        dataset = cache.load_split(split)
        try:
            assert len(dataset) == len(source)
            assert episode_seeds(dataset) == fixture.seeds[split]
        finally:
            dataset.close()


def test_same_fixed_interval_generates_identical_semantics(tmp_path):
    first = write_cache(tmp_path / "first")
    second = write_cache(tmp_path / "second")

    assert read_manifest(first.root) == read_manifest(second.root)
    for split in SPLIT_NAMES:
        assert first.counts[split] == second.counts[split]
        first_source = first.sources[split]
        second_source = second.sources[split]
        assert (first_source is None) == (second_source is None)
        if first_source is not None:
            assert second_source is not None
            np.testing.assert_array_equal(first_source.states, second_source.states)
            np.testing.assert_array_equal(
                first_source.candidate_features,
                second_source.candidate_features,
            )
            np.testing.assert_array_equal(first_source.targets, second_source.targets)

    first_cache = load_prepared_cache(first.root)
    second_cache = load_prepared_cache(second.root)
    for split in SPLIT_NAMES:
        first_dataset = first_cache.load_split(split)
        second_dataset = second_cache.load_split(split)
        try:
            assert first_dataset.identity == second_dataset.identity
            for index in range(len(first_dataset)):
                left = first_dataset[index]
                right = second_dataset[index]
                np.testing.assert_array_equal(left.state, right.state)
                np.testing.assert_array_equal(
                    left.candidate_features, right.candidate_features
                )
                assert left.target == right.target
                assert left.stage == right.stage
                assert left.target_kind == right.target_kind
                assert left.episode_id == right.episode_id
                assert left.decision_index == right.decision_index
        finally:
            first_dataset.close()
            second_dataset.close()


def test_fixed_interval_can_leave_a_split_empty(tmp_path):
    fixture = write_cache(tmp_path / "empty-test", seed_start=0, games=3)

    assert fixture.seeds == {
        "train": (0, 1),
        "validation": (2,),
        "test": (),
    }
    assert fixture.sources["test"] is None
    assert fixture.counts["test"].to_dict() == {
        "episodes": 0,
        "decisions": 0,
        "candidates": 0,
        "forced_decisions": 0,
        "eligible_decisions": 0,
        "min_candidates": 0,
        "max_candidates": 0,
    }
    manifest = read_manifest(fixture.root)
    assert manifest["splits"]["test"]["shards"] == []

    cache = load_prepared_cache(fixture.root)
    dataset = cache.load_split("test")
    try:
        assert len(dataset) == 0
        assert dataset.identity.decisions == 0
        assert cache.splits["test"].counts == fixture.counts["test"]
    finally:
        dataset.close()
