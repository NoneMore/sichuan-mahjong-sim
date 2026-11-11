from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mahjong_bc.cache import _seed_split, load_cache_split, load_prepared_cache
from mahjong_bc.constants import SPLIT_NAMES
from mahjong_bc.data import iter_batches
from mahjong_bc.errors import CacheContractError, DataError

from .cache_fixture import (
    episode_seeds,
    read_manifest,
    rewrite_cache_array,
    write_cache,
    write_manifest,
)


@pytest.mark.parametrize("seed, expected", [(158, "train"), (167, "validation"), (214, "test")])
def test_seed_split_matches_frozen_rust_oracles(seed, expected):
    # Frozen by crates/neural/src/collection/seed.rs, not the fixture generator.
    assert _seed_split(seed) == expected


def test_close_preserves_outstanding_mmap_views_and_rejects_new_reads(shared_cache):
    root = shared_cache.root
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
        [sys.executable, "-c", code, str(root)], capture_output=True, text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_loads_sharded_splits_and_preserves_decisions(shared_cache):
    root = shared_cache.root
    fixture = shared_cache
    cache = load_prepared_cache(root)
    assert cache.cache_id == "synthetic-cache-id"
    assert cache.encoding_version == 1
    assert cache.preparation_format_version == 1
    assert cache.preparation_version == 1
    assert cache.split_algorithm == "seed-sha256-80-10-10-v1"
    assert len(cache.splits["train"].shards) == 3
    assert cache.counts.decisions == sum(
        value.decisions for value in fixture.counts.values()
    )
    manifest = read_manifest(root)

    for split in SPLIT_NAMES:
        source = fixture.sources[split]
        assert source is not None
        dataset = cache.load_split(split)
        try:
            assert len(dataset) == len(source)
            assert dataset.identity.cache_id == cache.cache_id
            assert dataset.identity.split == split
            assert dataset.identity.decisions == len(source)
            assert dataset.identity.eligible_decisions == source.identity.eligible_decisions
            assert episode_seeds(dataset) == fixture.seeds[split]
            expected_index = 0
            for shard in manifest["splits"][split]["shards"]:
                seeds = np.load(root / shard["arrays"]["episode_seeds"]["file"])
                indices = np.load(
                    root / shard["arrays"]["decision_indices"]["file"]
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
    fixture = write_cache(tmp_path, seed_start=0, games=3)
    assert fixture.seeds["test"] == ()
    assert fixture.sources["test"] is None
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


def test_load_cache_split_returns_validated_metadata(shared_cache):
    dataset, cache = load_cache_split(shared_cache.root, "validation")
    try:
        assert dataset.identity.cache_id == cache.cache_id
        assert dataset.identity.split == "validation"
        assert len(dataset) == cache.splits["validation"].counts.decisions
        assert dataset.eligible_count == cache.splits["validation"].counts.eligible_decisions
    finally:
        dataset.close()


def test_mmap_and_plain_loading_agree(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
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


def test_memmap_datasets_reject_worker_processes(shared_cache):
    cache = load_prepared_cache(shared_cache.root)
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


def test_inventory_is_complete_and_portable(tmp_path, shared_cache):
    moved = tmp_path / "moved"
    shutil.copytree(shared_cache.root, moved)
    cache = load_prepared_cache(moved)
    assert cache.cache_id == "synthetic-cache-id"
    dataset = cache.load_split("test")
    try:
        assert len(dataset) == shared_cache.counts["test"].decisions
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
def test_rejects_unsupported_or_malformed_manifest_fields(fresh_cache, field, value):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest[field] = value
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def _state_width_layout(root, manifest):
    manifest["layout"]["state_width"] = 1000
    write_manifest(root, manifest)


def _array_dimensions_layout(root, manifest):
    manifest["layout"]["arrays"]["states"]["dimensions"] = ["decisions"]
    write_manifest(root, manifest)


@pytest.mark.parametrize(
    "mutate",
    [_state_width_layout, _array_dimensions_layout],
    ids=["state-width", "array-dimensions"],
)
def test_rejects_layout_that_differs_from_preparation_format_one(
    fresh_cache, mutate
):
    root = fresh_cache.root
    mutate(root, read_manifest(root))
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def test_layout_scalars_must_be_integers(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest["layout"]["candidate_width"] = float(manifest["layout"]["candidate_width"])
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="candidate_width"):
        load_prepared_cache(root)


def test_rejects_unknown_and_missing_manifest_keys(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest["unexpected"] = 1
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)
    manifest = read_manifest(root)
    del manifest["unexpected"]
    del manifest["counts"]
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def _extra_inventory_file(root, manifest):
    (root / "extra.txt").write_text("extra", encoding="utf-8")


def _missing_array_file(root, manifest):
    first = root / manifest["splits"]["train"]["shards"][0]["arrays"]["states"]["file"]
    first.unlink()


def _duplicate_manifest_key(root, manifest):
    text = (root / "dataset.json").read_text(encoding="utf-8")
    text = text.replace(
        '"format": "mahjong-bc-dataset",',
        '"format": "mahjong-bc-dataset",\n  "format": "mahjong-bc-dataset",',
        1,
    )
    (root / "dataset.json").write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    "mutate",
    [_extra_inventory_file, _missing_array_file, _duplicate_manifest_key],
    ids=["extra-file", "missing-array", "duplicate-key"],
)
def test_rejects_missing_extra_and_duplicate_inventory(fresh_cache, mutate):
    root = fresh_cache.root
    mutate(root, read_manifest(root))
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def test_checksum_verification_and_optional_skip(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    relative = manifest["splits"]["train"]["shards"][0]["arrays"]["states"]["file"]
    path = root / relative
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)
    cache = load_prepared_cache(root, verify_checksums=False)
    assert cache.cache_id == "synthetic-cache-id"


def _states_float64(root, manifest):
    shard = manifest["splits"]["train"]["shards"][0]
    states = np.load(root / shard["arrays"]["states"]["file"])
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="states",
        values=states.astype(np.float64),
    )


def _target_out_of_range(root, manifest):
    shard = manifest["splits"]["train"]["shards"][0]
    targets = np.load(root / shard["arrays"]["targets"]["file"])
    targets[0] = 999
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="targets",
        values=targets,
    )


def _empty_offset_span(root, manifest):
    shard = manifest["splits"]["train"]["shards"][0]
    offsets = np.load(root / shard["arrays"]["candidate_offsets"]["file"])
    offsets[1] = offsets[0]
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="candidate_offsets",
        values=offsets,
    )


def _kind_mismatch(root, manifest):
    shard = manifest["splits"]["train"]["shards"][0]
    kinds = np.load(root / shard["arrays"]["target_kinds"]["file"])
    kinds[0] = (int(kinds[0]) + 1) % 9
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="target_kinds",
        values=kinds,
    )


def _fortran_state_array(root, manifest):
    shard = manifest["splits"]["train"]["shards"][0]
    states = np.load(root / shard["arrays"]["states"]["file"])
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="states",
        values=np.asfortranarray(states),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _states_float64,
        _target_out_of_range,
        _empty_offset_span,
        _kind_mismatch,
        _fortran_state_array,
    ],
    ids=[
        "states-dtype",
        "target-range",
        "empty-offset",
        "kind-mismatch",
        "fortran-order",
    ],
)
def test_rejects_array_layout_and_target_errors(fresh_cache, mutate):
    root = fresh_cache.root
    mutate(root, read_manifest(root))
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def _bump_split_decisions(root, manifest):
    manifest["splits"]["train"]["counts"]["decisions"] += 1
    write_manifest(root, manifest)


def _duplicate_episode_seed(root, manifest):
    first_seed = np.load(
        root / manifest["splits"]["train"]["shards"][0]["arrays"]["episode_seeds"]["file"]
    )[0]
    second = manifest["splits"]["train"]["shards"][1]
    seeds = np.load(root / second["arrays"]["episode_seeds"]["file"])
    seeds[0] = first_seed
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=1,
        name="episode_seeds",
        values=seeds,
    )


@pytest.mark.parametrize(
    "mutate",
    [_bump_split_decisions, _duplicate_episode_seed],
    ids=["split-decisions", "duplicate-episode"],
)
def test_rejects_bad_counts_and_episode_metadata(fresh_cache, mutate):
    root = fresh_cache.root
    mutate(root, read_manifest(root))
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


def test_rejects_non_contiguous_episode_seed_blocks(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    seeds = np.load(root / shard["arrays"]["episode_seeds"]["file"])
    assert seeds[0] == seeds[1] and seeds[2] == seeds[3]
    first, second = seeds[0], seeds[2]
    seeds[1], seeds[2] = second, first
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="episode_seeds",
        values=seeds,
    )
    with pytest.raises(CacheContractError, match="non-contiguous"):
        load_prepared_cache(root)


@pytest.mark.parametrize(
    ("name", "value"),
    [("stages", 5), ("actor_seats", 4), ("target_kinds", 9)],
)
def test_rejects_out_of_range_metadata_values(fresh_cache, name, value):
    root = fresh_cache.root
    manifest = read_manifest(root)
    shard = manifest["splits"]["train"]["shards"][0]
    values = np.load(root / shard["arrays"][name]["file"])
    values[0] = value
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name=name,
        values=values,
    )
    with pytest.raises(CacheContractError, match=name):
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
def test_dataset_versions_reject_booleans_and_floats(fresh_cache, field, value):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest[field] = value
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match=field):
        load_prepared_cache(root)


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
def test_rejects_metadata_descriptor_shape_mismatch(fresh_cache, name):
    root = fresh_cache.root
    manifest = read_manifest(root)
    descriptor = manifest["splits"]["train"]["shards"][0]["arrays"][name]
    descriptor["shape"] = [int(descriptor["shape"][0]) + 1]
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="declares shape"):
        load_prepared_cache(root)


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
def test_rejects_invalid_captured_corpus_manifest(fresh_cache, field, value):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest["corpus_manifest"][field] = value
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError):
        load_prepared_cache(root)


@pytest.mark.parametrize(
    "field",
    ["format_version", "source_revision", "seed_start", "decision_limit", "shards"],
)
def test_rejects_missing_captured_corpus_keys(fresh_cache, field):
    root = fresh_cache.root
    manifest = read_manifest(root)
    del manifest["corpus_manifest"][field]
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="invalid keys"):
        load_prepared_cache(root)


def _shift_second_corpus_shard(corpus):
    shard = corpus["shards"][1]
    shard["first_seed"] = f"{int(shard['first_seed'], 16) + 1:016x}"


def _corpus_shard_games(corpus):
    corpus["shards"][0]["games"] = corpus["shard_games"] + 1


def _corpus_shard_decisions(corpus):
    corpus["shards"][0]["decisions"] = 0


def _corpus_shard_hash(corpus):
    corpus["shards"][0]["sha256"] = "not-a-hash"


def _corpus_shard_extra_key(corpus):
    corpus["shards"][0]["unexpected"] = 1


@pytest.mark.parametrize(
    "mutate",
    [
        _shift_second_corpus_shard,
        _corpus_shard_games,
        _corpus_shard_decisions,
        _corpus_shard_hash,
        _corpus_shard_extra_key,
    ],
    ids=[
        "shard-order",
        "shard-games",
        "shard-decisions",
        "shard-hash",
        "shard-extra-key",
    ],
)
def test_rejects_captured_corpus_shard_protocol_errors(fresh_cache, mutate):
    root = fresh_cache.root
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
    ids=["episodes", "decisions-and-eligible"],
)
def test_rejects_dataset_counts_that_disagree_with_captured_totals(
    fresh_cache, mutate
):
    root = fresh_cache.root
    manifest = read_manifest(root)
    mutate(manifest)
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="captured"):
        load_prepared_cache(root)


def test_rejects_episode_seeds_outside_captured_range(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    corpus = manifest["corpus_manifest"]
    shift = 1_000_000
    corpus["seed_start"] = f"{int(corpus['seed_start'], 16) + shift:016x}"
    for shard in corpus["shards"]:
        shard["first_seed"] = f"{int(shard['first_seed'], 16) + shift:016x}"
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="outside the captured"):
        load_prepared_cache(root)


def test_rejects_episode_seeds_assigned_to_the_wrong_split(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    validation_shard = manifest["splits"]["validation"]["shards"][0]
    validation_seed = int(
        np.load(root / validation_shard["arrays"]["episode_seeds"]["file"])[0]
    )
    seeds = np.load(
        root
        / manifest["splits"]["train"]["shards"][0]["arrays"]["episode_seeds"]["file"]
    )
    seeds[:2] = np.uint64(validation_seed)
    rewrite_cache_array(
        root,
        manifest,
        split="train",
        shard_index=0,
        name="episode_seeds",
        values=seeds,
    )
    with pytest.raises(CacheContractError, match="belongs to validation"):
        load_prepared_cache(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_revision", "other-revision"),
        ("encoding_version", 2),
        ("split_algorithm", "other-split"),
    ],
)
def test_rejects_collection_provenance_identity_mismatch(
    fresh_cache, field, value
):
    root = fresh_cache.root
    manifest = read_manifest(root)
    manifest["provenance"]["collection"][field] = value
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="collection provenance"):
        load_prepared_cache(root)


def test_rejects_provenance_without_required_sections(fresh_cache):
    root = fresh_cache.root
    manifest = read_manifest(root)
    del manifest["provenance"]["preparation"]
    write_manifest(root, manifest)
    with pytest.raises(CacheContractError, match="provenance"):
        load_prepared_cache(root)
