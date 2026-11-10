"""Compare a small Rust corpus/cache pair through the public Python loader.

Run from the repository root after installing python/:
    python python/tools/verify_rust_handoff.py --corpus ../bc-data --cache ../bc-cache

This acceptance probe reads JSONL only to independently check the handoff; the
runtime package consumes prepared arrays, not corpus records. Inputs should be
small smoke datasets, since this probe deliberately materializes their records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mahjong_bc import contains_memory_mapped_arrays, iter_batches, load_prepared_cache
from mahjong_bc.constants import SPLIT_NAMES


def verify(corpus: Path, cache_root: Path) -> dict:
    cache = load_prepared_cache(cache_root)
    manifest_bytes = (corpus / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == cache.corpus_manifest_sha256
    manifest = json.loads(manifest_bytes)
    expected: dict[str, list[tuple[int, dict]]] = {name: [] for name in SPLIT_NAMES}
    # Read exactly the captured inventory, not arbitrary files in the directory.
    for shard in manifest["shards"]:
        shard_file = corpus / "shards" / f"{shard['first_seed']}.jsonl"
        episode = None
        with shard_file.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["type"] == "episode":
                    episode = (record["split"], int(record["episode_id"][2], 16))
                elif record["type"] == "decision":
                    assert episode is not None
                    expected[episode[0]].append((episode[1], record))
    results = {}
    for split in SPLIT_NAMES:
        dataset = cache.load_split(split)
        try:
            records = expected[split]
            assert len(dataset) == len(records)
            if records:
                assert contains_memory_mapped_arrays(dataset)
            candidates = forced = 0
            for index, (seed, record) in enumerate(records):
                sample = dataset[index]
                state = np.asarray(record["state"], dtype=np.float32)
                features = np.asarray(record["candidate_features"], dtype=np.float32)
                np.testing.assert_array_equal(sample.state.view(np.uint32), state.view(np.uint32))
                np.testing.assert_array_equal(
                    sample.candidate_features.view(np.uint32), features.view(np.uint32)
                )
                assert sample.target == record["teacher_target_index"]
                assert sample.episode_id == seed
                assert sample.decision_index == record["decision_index"]
                assert sample.actor_seat == record["actor_seat"]
                assert sample.stage == record["stage"]
                assert sample.target_kind == int(np.argmax(features[sample.target, :9]))
                assert sample.eligible == (len(features) > 1)
                candidates += len(features)
                forced += len(features) == 1
            assert candidates == cache.splits[split].counts.candidates
            assert forced == dataset.forced_count == cache.splits[split].counts.forced_decisions
            assert dataset.eligible_count == len(records) - forced
            eligible = [i for i, (_, r) in enumerate(records) if len(r["candidate_features"]) > 1]
            np.testing.assert_array_equal(dataset.eligible_indices(), eligible)
            forced_by_stage = [0] * 5
            for _, record in records:
                if len(record["candidate_features"]) == 1:
                    forced_by_stage[record["stage"]] += 1
            assert dataset.forced_by_stage() == tuple(forced_by_stage)
            # Reverse order crosses shard boundaries and makes ordering observable.
            order = np.arange(len(dataset) - 1, -1, -1, dtype=np.int64)
            sizes = []
            position = 0
            for batch in iter_batches(dataset, order, batch_size=17, num_workers=0):
                sizes.append(batch.batch_size)
                for row in range(batch.batch_size):
                    sample = dataset[int(order[position])]
                    count = sample.candidate_count
                    np.testing.assert_array_equal(batch.states[row].numpy(), sample.state)
                    np.testing.assert_array_equal(
                        batch.candidates[row, :count].numpy(), sample.candidate_features
                    )
                    assert batch.mask[row, :count].all().item()
                    assert not batch.mask[row, count:].any().item()
                    assert not batch.candidates[row, count:].any().item()
                    assert batch.targets[row].item() == sample.target
                    assert batch.stages[row].item() == sample.stage
                    assert batch.target_kinds[row].item() == sample.target_kind
                    assert batch.actor_seats[row].item() == sample.actor_seat
                    assert batch.decision_indices[row].item() == sample.decision_index
                    assert batch.episode_ids[row] == sample.episode_id
                    assert batch.eligible[row].item() == sample.eligible
                    position += 1
            assert position == len(dataset)
            if sizes:
                assert sizes[-1] == (len(dataset) - 1) % 17 + 1
            results[split] = {
                "decisions": len(dataset), "candidates": candidates,
                "forced": forced, "eligible": dataset.eligible_count, "batch_sizes": sizes,
            }
        finally:
            dataset.close()
    return {"cache_id": cache.cache_id, "splits": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.corpus, args.cache), indent=2))


if __name__ == "__main__":
    main()
