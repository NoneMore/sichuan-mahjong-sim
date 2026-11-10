# mahjong-bc

Python dataset loader for offline behavior cloning in Sichuan mahjong. It
validates the Rust [prepared-cache protocol](../crates/neural/README.md#dataset-preparation),
exposes sharded memory-mapped decisions and collates padded PyTorch batches.
Neither Rust bindings nor the original corpus are needed. Model scoring,
training and checkpoints are not yet implemented.

## Setup and commands

Use Python 3.10+ and the repository's Git-ignored `.venv/` (see the
[development guide](../docs/development.md#python-environment)). From the
repository root, create the environment if absent and activate it:

```text
python -m venv .venv
# Windows Git Bash: source .venv/Scripts/activate
# POSIX:            source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e "./python[dev]"
python -m mahjong_bc version
python -m mahjong_bc cache-info --dataset ../bc-cache
```

For CUDA, install the matching build from the [PyTorch selector](https://pytorch.org/get-started/locally/)
instead of the CPU wheel. Package installation reuses the installed build.
Keep generated caches outside the source tree.

The installed `mahjong-bc` console script exposes the same commands as
`python -m mahjong_bc`:

- `cache-info --dataset CACHE`: validate the cache and print versions, identity,
  counts and per-split inventory as JSON. Redirect stdout to save it.
- `version`: print package, Python, NumPy, PyTorch and CUDA build versions.

## Loading and batching

```python
from mahjong_bc import iter_batches, load_prepared_cache

cache = load_prepared_cache("../bc-cache")
train = cache.load_split("train")
try:
    for batch in iter_batches(
        train, train.eligible_indices(), batch_size=256, num_workers=0
    ):
        # states: [B, 1013]; candidates: [B, max_candidates, 39], float32
        # mask: [B, max_candidates], bool; targets: [B], local indices
        ...
finally:
    train.close()
```

`load_prepared_cache` validates versions, file inventory/checksums, array
headers, counts and target/metadata alignment. It cross-checks captured-corpus
totals, seed range/coverage, split assignments and collection provenance.
Rust defines the opaque `cache_id`; Python preserves it in `DatasetIdentity`
without recomputing its hash.

`PreparedCache` exposes aggregate and per-split counts, provenance and
`split_identity(split)`. `load_cache_split(root, split)` returns a dataset and
its cache metadata in one call. Each split is a `ShardedDecisionDataset` of
`PackedDecisionDataset` shards, mapped read-only by default. Global indices
follow manifest shard order without concatenating feature arrays; empty splits
have no shards. `close()` releases dataset-owned arrays and rejects further
sample reads. Outstanding sample/array views keep their maps alive until released.

Forced decisions are retained. `eligible_indices()` selects decisions with
more than one candidate; `forced_by_stage()` reports forced counts. Collation
preserves candidate order and local targets, pads only to the batch maximum,
and marks valid candidates with `mask`.

`iter_batches` follows the supplied index order and retains the final partial
batch. Mmap datasets require `num_workers=0`; positive worker counts raise
`DataError`. `contains_memory_mapped_arrays(dataset)` detects mapped data.
For in-memory workers, use `load_split(..., mmap=False)` and guard scripts with
`if __name__ == "__main__":` on spawn-based platforms such as Windows.

## Verification

After installation, in the activated environment:

```text
python -m pytest python/tests
python python/tools/verify_rust_handoff.py --corpus ../bc-data --cache ../bc-cache
```

The [CPU workflow](../.github/workflows/python-ci.yml) runs loader, batching and
CLI tests. The handoff probe compares a small real Rust corpus/cache pair
bit-for-bit, including candidate order, targets, metadata and padded batches.
It materializes the probe's input records, unlike the runtime loader. See
[validation](../docs/validation.md#python-dataset-loader-smoke-validation) for
reproducible inputs and observed results.
