from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from mahjong_bc.cache import _seed_split, load_cache_split, load_prepared_cache
from mahjong_bc.constants import SPLIT_NAMES
from mahjong_bc.data import iter_batches
from mahjong_bc.errors import CacheContractError, DataError

from .cache_fixture import read_manifest, write_cache, write_manifest
from .helpers import make_dataset


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


@pytest.mark.parametrize("seed, expected", [(158, "train"), (167, "validation"), (214, "test")])
def test_seed_split_matches_frozen_rust_oracles(seed, expected):
    # Frozen by crates/neural/src/collection/seed.rs, not the fixture generator.
    assert _seed_split(seed) == expected


def test_close_preserves_outstanding_mmap_views_and_rejects_new_reads(tmp_path):
    _cache(tmp_path)
    # Isolate the probe so a future dangling-mmap regression reports a failed
    # subprocess instead of taking the entire pytest process down.
    code = textwrap.dedent("""
        import sys
        import numpy as np
        from mahjong_bc import DataError, iter_batches, load_prepared_cache

        dataset = load_prepared_cache(sys.argv[1]).load_split("train")
        sample = dataset[0]
        states = dataset.shards[0].states
        expected_state = sample.state.copy()
        expected_candidates = sample.candidate_features.copy()
        batches = iter_batches(dataset, range(len(dataset)), batch_size=1)
        first = next(batches)
        dataset.close()
        dataset.close()
        np.testing.assert_array_equal(sample.state, expected_state)
        np.testing.assert_array_equal(sample.candidate_features, expected_candidates)
        np.testing.assert_array_equal(states[0], expected_state)
        np.testing.assert_array_equal(first.states[0].numpy(), expected_state)
        for operation in (lambda: dataset[0], lambda: next(batches),
                          lambda: dataset.shards[0][0], dataset.forced_by_stage):
            try:
                operation()
            except DataError:
                pass
            else:
                raise AssertionError("closed dataset should reject reads")
    """)
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    manifest = read_manifest(tmp_path)

    for split, source in zip(SPLIT_NAMES, sources):
        dataset = cache.load_split(split)
        try:
            assert len(dataset) == len(source)
            assert dataset.identity.cache_id == cache.cache_id
            assert dataset.identity.split == split
            assert dataset.identity.decisions == len(source)
            assert dataset.identity.eligible_decisions == source.identity.eligible_decisions
            expected_index = 0
            for shard in manifest["splits"][split]["shards"]:
                seeds = np.load(tmp_path / shard["arrays"]["episode_seeds"]["file"])
                indices = np.load(
                    tmp_path / shard["arrays"]["decision_indices"]["file"]
                )
                for position in range(len(seeds)):
                    sample = dataset[expected_index]
                    start = int(source.offsets[expected_index])
                    end = int(source.offsets[expected_index + 1])
                    assert np.array_equal(sample.state, source.states[expected_index])
                    assert np.array_equal(
                        sample.candidate_features,
                        source.candidate_features[start:end],
                    )
                    assert sample.target == int(source.targets[expected_index])
                    assert sample.stage == int(source.stages[expected_index])
                    assert sample.target_kind == int(
                        source.target_kinds[expected_index]
                    )
                    assert sample.actor_seat == expected_index % 4
                    assert sample.decision_index == int(indices[position])
                    assert sample.episode_id == int(seeds[position])
                    assert sample.candidate_count == end - start
                    expected_index += 1
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
        assert dataset.decision_count == 0
        assert dataset.eligible_count == 0
        assert dataset.forced_count == 0
        assert dataset.eligible_indices().size == 0
        assert dataset.candidate_counts().size == 0
        assert dataset.forced_by_stage() == (0, 0, 0, 0, 0)
        assert list(iter_batches(dataset, dataset.eligible_indices(), batch_size=4)) == []
        with pytest.raises(IndexError):
            dataset[0]
        assert dataset.identity.split == "test"
        assert dataset.identity.decisions == 0
        assert cache.splits["test"].counts.to_dict() == {
            "episodes": 0,
            "decisions": 0,
            "candidates": 0,
            "forced_decisions": 0,
            "eligible_decisions": 0,
            "min_candidates": 0,
            "max_candidates": 0,
        }
    finally:
        dataset.close()
    with pytest.raises(CacheContractError, match="unknown split"):
        cache.load_split("bogus")


def test_load_cache_split_returns_validated_metadata(tmp_path):
    _cache(tmp_path)
    dataset, cache = load_cache_split(tmp_path, "validation")
    try:
        assert dataset.identity.cache_id == cache.cache_id
        assert dataset.identity.split == "validation"
        assert len(dataset) == cache.splits["validation"].counts.decisions
        assert dataset.eligible_count == cache.splits["validation"].counts.eligible_decisions
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


def test_layout_scalars_must_be_integers(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["layout"]["candidate_width"] = float(manifest["layout"]["candidate_width"])
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="candidate_width"):
        load_prepared_cache(tmp_path)


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


def test_rejects_non_contiguous_episode_seed_blocks(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    shard = manifest["splits"]["train"]["shards"][0]
    seeds = np.load(tmp_path / shard["arrays"]["episode_seeds"]["file"])
    assert seeds[0] == seeds[1] and seeds[2] == seeds[3]
    first, second = seeds[0], seeds[2]
    seeds[1], seeds[2] = second, first
    _rewrite_array(tmp_path, manifest, "train", 0, "episode_seeds", seeds)
    with pytest.raises(CacheContractError, match="non-contiguous"):
        load_prepared_cache(tmp_path)


def test_rejects_out_of_range_metadata_values(tmp_path):
    root = tmp_path / "stage-range"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    stages = np.load(root / shard["arrays"]["stages"]["file"])
    stages[0] = 5
    _rewrite_array(root, manifest, "train", 0, "stages", stages)
    with pytest.raises(CacheContractError, match="stages"):
        load_prepared_cache(root)

    root = tmp_path / "seat-range"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    seats = np.load(root / shard["arrays"]["actor_seats"]["file"])
    seats[0] = 4
    _rewrite_array(root, manifest, "train", 0, "actor_seats", seats)
    with pytest.raises(CacheContractError, match="actor_seats"):
        load_prepared_cache(root)

    root = tmp_path / "kind-range"
    _cache(root)
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    kinds = np.load(root / shard["arrays"]["target_kinds"]["file"])
    kinds[0] = 9
    _rewrite_array(root, manifest, "train", 0, "target_kinds", kinds)
    with pytest.raises(CacheContractError, match="target_kinds"):
        load_prepared_cache(root)


@pytest.mark.parametrize("payload", ["[]\n", '"manifest"\n', "null\n"])
def test_root_manifest_must_be_a_json_object(tmp_path, payload):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dataset.json").write_text(payload, encoding="utf-8")
    with pytest.raises(CacheContractError, match="JSON object"):
        load_prepared_cache(tmp_path)


@pytest.mark.parametrize(
    "field",
    ["preparation_format_version", "preparation_version", "encoding_version"],
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_dataset_versions_reject_booleans_and_floats(tmp_path, field, value):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest[field] = value
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match=field):
        load_prepared_cache(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "candidate_offsets",
        "targets",
        "episode_seeds",
        "decision_indices",
        "actor_seats",
        "stages",
        "target_kinds",
    ],
)
def test_rejects_metadata_descriptor_shape_mismatch(tmp_path, name):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    descriptor = manifest["splits"]["train"]["shards"][0]["arrays"][name]
    descriptor["shape"] = [int(descriptor["shape"][0]) + 1]
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="declares shape"):
        load_prepared_cache(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("format_version", True),
        ("encoding_version", 2),
        ("split_algorithm", "other-split"),
        ("source_revision", "other-revision"),
        ("seed_start", "zz"),
        ("seed_start", 16),
        ("decision_limit", 0),
        ("decision_limit", 1.0),
        ("shard_games", 0),
        ("shards", "no"),
        ("unexpected", 1),
    ],
)
def test_rejects_invalid_captured_corpus_manifest(tmp_path, field, value):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["corpus_manifest"][field] = value
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(tmp_path)


@pytest.mark.parametrize(
    "field",
    ["format_version", "source_revision", "seed_start", "decision_limit", "shards"],
)
def test_rejects_missing_captured_corpus_keys(tmp_path, field):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    del manifest["corpus_manifest"][field]
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="invalid keys"):
        load_prepared_cache(tmp_path)


def test_rejects_captured_corpus_shard_protocol_errors(tmp_path):
    cases = [
        lambda corpus: corpus["shards"][1].__setitem__(
            "first_seed", f"{int(corpus['shards'][1]['first_seed'], 16) + 1:016x}"
        ),
        lambda corpus: corpus["shards"][0].__setitem__(
            "games", corpus["shard_games"] + 1
        ),
        lambda corpus: corpus["shards"][0].__setitem__("decisions", 0),
        lambda corpus: corpus["shards"][0].__setitem__("sha256", "not-a-hash"),
        lambda corpus: corpus["shards"][0].__setitem__("unexpected", 1),
    ]
    for index, mutate in enumerate(cases):
        root = tmp_path / f"case-{index}"
        _cache(root)
        manifest = read_manifest(root)
        mutate(manifest["corpus_manifest"])
        write_manifest(root, manifest)
        with pytest.raises(CacheContractError):
            load_prepared_cache(root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["counts"].__setitem__(
            "episodes", manifest["counts"]["episodes"] + 1
        ),
        lambda manifest: manifest["counts"].update(
            {
                "decisions": manifest["counts"]["decisions"] + 1,
                "eligible_decisions": manifest["counts"]["eligible_decisions"] + 1,
            }
        ),
    ],
)
def test_rejects_dataset_counts_that_disagree_with_captured_totals(
    tmp_path, mutate
):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    mutate(manifest)
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="captured"):
        load_prepared_cache(tmp_path)


def test_rejects_episode_seeds_outside_captured_range(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    corpus = manifest["corpus_manifest"]
    shift = 1_000_000
    corpus["seed_start"] = f"{int(corpus['seed_start'], 16) + shift:016x}"
    for shard in corpus["shards"]:
        shard["first_seed"] = f"{int(shard['first_seed'], 16) + shift:016x}"
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="outside the captured"):
        load_prepared_cache(tmp_path)


def test_rejects_episode_seeds_assigned_to_the_wrong_split(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    validation_shard = manifest["splits"]["validation"]["shards"][0]
    validation_seed = int(
        np.load(tmp_path / validation_shard["arrays"]["episode_seeds"]["file"])[0]
    )
    seeds = np.load(
        tmp_path
        / manifest["splits"]["train"]["shards"][0]["arrays"]["episode_seeds"]["file"]
    )
    seeds[:2] = np.uint64(validation_seed)
    _rewrite_array(tmp_path, manifest, "train", 0, "episode_seeds", seeds)
    with pytest.raises(CacheContractError, match="belongs to validation"):
        load_prepared_cache(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_revision", "other-revision"),
        ("encoding_version", 2),
        ("split_algorithm", "other-split"),
    ],
)
def test_rejects_collection_provenance_identity_mismatch(tmp_path, field, value):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["provenance"]["collection"][field] = value
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="collection provenance"):
        load_prepared_cache(tmp_path)


def test_rejects_provenance_without_required_sections(tmp_path):
    _cache(tmp_path)
    manifest = read_manifest(tmp_path)
    del manifest["provenance"]["preparation"]
    write_manifest(tmp_path, manifest)
    with pytest.raises(CacheContractError, match="provenance"):
        load_prepared_cache(tmp_path)
