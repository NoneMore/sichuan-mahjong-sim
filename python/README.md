# mahjong-bc

Offline behavior cloning for Sichuan mahjong: Rust
[prepared caches](../crates/neural/README.md#dataset-preparation) → mmap datasets
→ padded PyTorch batches → candidate policy training and frozen evaluation.
No Rust bindings or original corpus are needed after preparation. Online play
and training recovery are not implemented.

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
instead of the CPU wheel. Training supports CPU or one local CUDA device. Keep
generated caches and run directories outside the source tree.

The installed `mahjong-bc` console script exposes the same commands as
`python -m mahjong_bc`:

- `cache-info --dataset CACHE`: validate the cache and print versions, identity,
  counts and per-split inventory as JSON.
- `train --dataset CACHE --output DIR --seed N --epochs N --batch-size N
  --learning-rate X`: train a fresh policy, select `best.pt` by validation mean
  NLL, and write `run.json`. `--device cpu|cuda[:index]`, `--log-file`, and model
  architecture options are optional.
- `evaluate --dataset CACHE --checkpoint FILE --split validation|test`: evaluate
  a frozen checkpoint on a complete split, optionally writing `--output` metrics.
- `inspect --checkpoint FILE`: print checkpoint metadata without training.
- `version`: print package, Python, NumPy, PyTorch and CUDA build versions.

Fresh training rejects existing `best.pt`, `last.pt` or `run.json`; `last.pt`
is reserved but not produced. CLI run/report destinations must stay outside
the cache. Report files must be new paths and cannot alias checkpoints or
reserved run artifacts; conflicts are rejected before execution. `evaluate`
may send both report flags to the same new path, writing it once.

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
have no shards. A packed shard may have `identity=None`; a loaded split always
has its complete `DatasetIdentity`. Custom `DecisionDataset` implementations
must implement `is_memory_mapped` so worker restrictions also apply to wrappers.
`close()` releases dataset-owned arrays and rejects further sample reads.
Outstanding sample/array views keep their maps alive until released.

Forced decisions are retained. `eligible_indices()` selects decisions with
more than one candidate; `forced_by_stage()` reports forced counts. Collation
preserves candidate order and local targets, pads only to the batch maximum,
and marks valid candidates with `mask`.

`iter_batches` follows the supplied index order and retains the final partial
batch. Mmap datasets require `num_workers=0`; positive worker counts raise
`DataError`. `contains_memory_mapped_arrays(dataset)` detects mapped data.
For in-memory workers, use `load_split(..., mmap=False)` and guard scripts with
`if __name__ == "__main__":` on spawn-based platforms such as Windows.

## Training, metrics, and checkpoints

`CandidatePolicy` shares a state embedding across candidates and scores each
state/candidate pair with a nonlinear MLP. It uses float32 without dropout or
batch-dependent normalization. Candidate permutation permutes logits; padding
is zeroed before scoring and masked before loss/selection. Greedy ties choose
canonical-first; categorical sampling uses the same masked logits.

Each seeded epoch visits every eligible decision (more than one candidate)
once in shuffled order, including the final partial batch. Masked cross-entropy
weights decisions equally; forced choices are counted separately.

`metrics.py` aggregates complete eligible validation/test sets by decision:
overall NLL/top-1, per-stage support/NLL/top-1, teacher action-kind support/top-1,
and separate forced counts. Zero-support groups report `null`. Validation mean
NLL alone selects `best.pt`; test evaluation loads the frozen checkpoint and
never influences selection or updates the file.

`run.json` records the resolved model/training configuration, training seed,
device type, dependencies, dataset identities and provenance, best-checkpoint
name, and per-epoch history including validation metrics. History `train_nll` is
the decision-weighted mean of pre-update batch losses, not an epoch-end
re-evaluation. Checkpoint publication and the run record use write-then-replace
atomic publication.

The run `source` record captures package/version and a SHA-256 `hash` of the
imported package's sorted `.py` paths and contents, excluding `__pycache__`.
Editable installs at `<git root>/python/mahjong_bc` also report the repository's
`revision` and `dirty` state; wheel installs or unavailable Git metadata use
`null`, without borrowing an outer repository's identity. Source identity is
independent of the process working directory and captured once per `Trainer`.

Checkpoint format **2** stores model configuration, weights and optional
metrics, with `training: null` reserving the recovery-state boundary. Supported
roles are `policy`/`best`, model format **1**, encoding **1**. The loader rejects
unknown keys, unsupported versions/roles and non-null recovery state, uses
`torch.load(..., weights_only=True)`, validates weights strictly and returns an
eval-mode model. Incompatible schema changes require an explicit version bump.

## Verification

After installation, in the activated environment:

```text
python -m pytest python/tests
python python/tools/verify_rust_handoff.py --corpus ../bc-data --cache ../bc-cache
```

The [CPU workflow](../.github/workflows/python-ci.yml) runs loader, batching,
policy, metrics, checkpoint, training and CLI tests. The handoff probe compares a
small real Rust corpus/cache pair bit-for-bit, including candidate order,
targets, metadata and padded batches. It materializes the probe's input records,
unlike the runtime loader. See
[loader validation](../docs/validation.md#python-dataset-loader-smoke-validation)
and [training validation](../docs/validation.md#offline-bc-training-smoke-validation)
for reproducible inputs and observed results.
