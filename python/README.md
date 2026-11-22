# mahjong-bc

Offline behavior cloning for Sichuan mahjong: Rust
[prepared caches](../crates/neural/README.md#dataset-preparation) → packed datasets
→ PyTorch batches → candidate policy training and frozen evaluation.
No Rust bindings or original corpus are needed after preparation. Online play
is outside this package.

For the complete collection-to-evaluation sequence, see the [BC pipeline guide](../docs/guides/bc-pipeline.md).

## Setup and commands

Use Python 3.10+ in the repository's `.venv/` (see
[Python environment setup](../docs/development.md#python-environment)).
From the repository root:

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e "./python[dev]"
python -m mahjong_bc version
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
- `train --dataset CACHE --output DIR --resume RUN/last.pt`: continue a run
  from its last completed epoch.
- `evaluate --dataset CACHE --checkpoint FILE --split validation|test`: evaluate
  a frozen checkpoint on a complete split, optionally writing `--output` metrics.
- `inspect --checkpoint FILE`: print checkpoint metadata without training.
- `version`: print package, Python, NumPy, PyTorch and CUDA build versions.

Both `train` and `evaluate` accept independent input options:

| Option | Default / requirement |
| --- | --- |
| `--data-storage mmap\|ram` | `mmap` |
| `--ram-limit-mib N` | Required only for `ram` |
| `--batch-transfer sync\|cuda-prefetch` | `sync` |
| `--pinned-limit-mib N` | Required only for `cuda-prefetch` |

Budgets are positive integers in MiB (`2**20` bytes); supplying a budget for
an inactive mode is an error. RAM and CUDA prefetch require `--num-workers 0`.

Fresh training requires `best.pt`, `last.pt` and `run.json` to be absent.
Resume accepts only a `last` checkpoint, loads its configuration without flag
overrides and replays any interrupted epoch. Dataset identities, model/training
configuration (including total epochs), device type and dependency versions must
match; output and log paths may change. Storage and transfer are execution
options: resume may reselect them, and omission uses mmap/sync.

In-place resume updates the source run. A different destination must contain
none of the three run artifacts. Migration republishes best/last before further
training, even when the run is complete or no new best is found. If migration
fails after publishing only some files, choose a clean destination.

CLI run/report destinations must stay outside the cache. Report files must be
new paths and cannot alias checkpoints or reserved run artifacts; conflicts
are rejected before execution. `evaluate` may send both report flags to the
same new path, writing it once.

## Loading and batching

```python
from mahjong_bc import iter_batches, load_prepared_cache

cache = load_prepared_cache("../bc-cache")
with cache.open_splits(["train"], storage="mmap") as inputs:
    train = inputs.datasets["train"]
    for batch in iter_batches(
        train, train.eligible_indices(), batch_size=256, num_workers=0
    ):
        # states: [B, 1013]; candidates: [B, max_candidates, 39], float32
        # mask: [B, max_candidates], bool; targets: [B], local indices
        ...
```

`load_prepared_cache` validates versions, file inventory/checksums, array
headers, counts and target/metadata alignment. It cross-checks captured-corpus
totals, seed range/coverage, split assignments and collection provenance.
Rust defines the opaque `cache_id`; Python preserves it in `DatasetIdentity`
without recomputing its hash.

`PreparedCache` exposes aggregate and per-split counts, provenance and
`split_identity(split)`. `load_cache_split(root, split)` returns a dataset and
its cache metadata in one call. `load_split` returns a `ShardedDecisionDataset`
of `PackedDecisionDataset` shards, mapped read-only by default. Global indices
follow manifest shard order without concatenating feature arrays; empty splits
have no shards. A packed shard may have `identity=None`; a loaded split always
has its complete `DatasetIdentity`. Custom `DecisionDataset` implementations
must implement `is_memory_mapped` so worker restrictions also apply to wrappers.
Training and evaluation entry points require a constructed `DatasetIdentity`;
an anonymous shard or duck-typed identity is rejected.

`cache.open_splits(names, storage=..., ram_limit_bytes=...)` owns the datasets
in `inputs.datasets` and closes them on exit, after batch consumers finish.
`storage="mmap"` retains read-only shards and accepts no RAM budget.
`storage="ram"` requires a positive byte budget and produces one read-only
`PackedDecisionDataset` per split, preserving rows, metadata, candidate order
and identity. The combined budget is checked before feature allocation and
covers destination arrays, derived counts, eligibility masks and a conservative
eligible-index capacity—not RSS, source mappings/page cache, validation
temporaries, epoch indices, batches or models. `inputs.loading_record` reports
actual held RAM array bytes; mmap sizes are not resident-memory measurements.
Construction failure releases partial results.

The older `load_split(..., mmap=False)` still returns mutable shards without a
loading record. `close()` rejects further reads; outstanding views keep their
maps alive until released.

Forced decisions are retained. `eligible_indices()` selects decisions with
more than one candidate; `forced_by_stage()` reports forced counts. Collation
preserves candidate order and local targets, pads only to the batch maximum,
and marks valid candidates with `mask`.

`pad_decision_inputs(decisions)` is the label-free primitive behind collation
and online inference. It accepts any objects exposing `state` and
`candidate_features`, pads to the batch maximum with zero-filled candidate rows
and a Boolean mask, and preserves valid scores and canonical order without
requiring targets or other supervision. `collate_decisions` adds the offline
labels and metadata on top of it.

`iter_batches` preserves index order and the final partial batch. Built-in
packed datasets support column-wise batch reads; custom datasets retain
per-sample collation. Mmap datasets require `num_workers=0`; positive worker
counts raise `DataError`. `contains_memory_mapped_arrays(dataset)` detects
mapped data.
For in-memory workers, use `load_split(..., mmap=False)` and guard scripts with
`if __name__ == "__main__":` on spawn-based platforms such as Windows.

`Trainer`, `evaluate_dataset` and `evaluate_checkpoint` accept an optional
`BatchTransferConfig`, exported by `mahjong_bc`. The default `sync` mode retains
normal batch validation and device moves, including custom-dataset and policy
fallbacks. `BatchTransferConfig(mode="cuda-prefetch", pinned_limit_bytes=N)`
requires CUDA, zero workers, the exact built-in policy and packed dataset types,
read-only input arrays, target-kind metadata and unchanged eligible layouts.
Explicitly unsupported requests raise `ConfigurationError`, without fallback.
Do not modify, replace or close inputs during asynchronous consumption.

The pinned budget covers two fixed-capacity slots. For
`B=min(batch_size, number_of_indices)` and maximum selected candidate count `C`,
their total capacity is `2*(4060*B + 165*B*C)` bytes for training and
`2*(4076*B + 165*B*C)` for evaluation. It excludes ordinary gathered batches,
temporary indices, Tensor objects and allocator caches—not an OS locked-memory
limit. Empty/all-forced evaluation creates no slots or producer. See
[transfer ownership and scheduling](../docs/architecture/neural.md#offline-python-data-and-training)
for the producer, stream and cleanup boundaries.

## Training, metrics, and checkpoints

`CandidatePolicy` shares a state embedding across candidates and scores each
state/candidate pair with a nonlinear MLP. It uses float32 without dropout or
batch-dependent normalization. Candidate permutation permutes logits; padding
is zeroed before scoring and masked before loss/selection. Greedy ties choose
canonical-first; categorical sampling uses the same masked logits.

Compatible built-in packed paths score only valid candidates. Public inputs
and batches retain their padded contracts; see
[batching and fallbacks](../docs/architecture/neural.md#offline-python-data-and-training).

Each seeded epoch visits every eligible decision (more than one candidate)
once in shuffled order, including the final partial batch. Masked cross-entropy
weights decisions equally; forced choices are counted separately.

Evaluation scores eligible decisions in built-in packed splits with unchanged
layouts and takes forced counts from metadata. Reports cover complete
validation/test splits: overall and per-stage support/NLL/top-1, teacher
action-kind support/top-1, and separate forced counts. Zero-support groups
report `null`. Validation mean NLL alone selects `best.pt`; test evaluation
never influences selection or updates the frozen checkpoint.

`run.json` records the resolved model/training configuration, training seed,
device type, dependencies, dataset identities and provenance, checkpoint names,
and per-epoch history including validation metrics. CLI loading records appear
at `dataset.provenance.input_loading`, transfer mode/budget at `batch_transfer`;
CLI train/evaluation JSON exposes `input_loading` and `batch_transfer` at the
top level. These describe the current invocation, not previous epochs.
Library callers may pass `inputs.loading_record` through
`Trainer(dataset_provenance={"input_loading": inputs.loading_record})`.
History `train_nll` is the decision-weighted mean of pre-update batch losses,
not an epoch-end re-evaluation. Checkpoint publication and the run record use
write-then-replace atomic publication per file. `last.pt` is the epoch commit
point: a failed write leaves its predecessor usable. The files are not one
multi-file transaction;
resuming from last republishes its embedded best state before continuing.

The run `source` record captures package/version and a SHA-256 `hash` of the
imported package's sorted `.py` paths and contents, excluding `__pycache__`.
Editable installs at `<git root>/python/mahjong_bc` also report the repository's
`revision` and `dirty` state; wheel installs or unavailable Git metadata use
`null`, without borrowing an outer repository's identity. Source identity is
independent of the process working directory and captured once per `Trainer`.

Checkpoint format **3** separates the policy payload (model configuration,
weights and metrics) from optional recovery state. `last.pt` includes optimizer,
completed epoch/step/sample counters, Torch CPU/CUDA random states, the
shuffle generator and embedded best weights/metrics. No scheduler is used.
Model format **1** and encoding **1** are unchanged. The loader validates the
schema and weights, uses `torch.load(..., weights_only=True)` and returns an
eval-mode model. Format **2** policy/best artifacts remain readable for
inspection/evaluation, but cannot be resumed. Incompatible schema changes
require an explicit version bump.

## Verification

After installation, in the activated environment:

```text
python -m pytest python/tests
# Optional, device-dependent CUDA scheduling diagnostic:
python -m pytest python/tests -m cuda_diagnostic
python python/tools/verify_rust_handoff.py --corpus ../bc-data --cache ../bc-cache
python python/tools/verify_recovery.py --dataset ../bc-cache --output ../bc-recovery
```

The default pytest selection excludes `cuda_diagnostic`: its synthetic delays
and overlap threshold depend on device scheduling, not functional correctness
or end-to-end throughput. The normal CUDA suite retains event/slot reuse,
non-default streams, pending-compute allocator lifetime, error/exit cleanup and
exact recovery checks. The diagnostic command above overrides the default
marker filter and requires CUDA.

The [CPU workflow](../.github/workflows/python-ci.yml) runs loader, batching,
policy, metrics, checkpoint, training and CLI tests. The handoff probe compares a
small real Rust corpus/cache pair bit-for-bit, including candidate order,
targets, metadata and padded batches. The recovery probe injects a mid-epoch
CPU failure, compares replay order/steps and model/optimizer states exactly with
a continuous run, migrates a completed run and evaluates the frozen best model.
It uses one CPU thread and requires a new output directory.

## Training profiling

Use [profile_training.py](tools/profile_training.py) with a prepared cache and
a new output directory. This example profiles training with RAM storage and
CUDA prefetch.

```text
python python/tools/profile_training.py --dataset ../bc-cache --output ../bc-profile --storage ram --ram-limit-mib 14336 --batch-transfer cuda-prefetch --pinned-limit-mib 64 --repeats 3
```

Timing mode warms up with one full epoch by default. Each measured trial starts
from the same seed and includes a full training epoch, validation and checkpoint
publication. The report gives split loading time separately, records the source
hash and run settings, and checks both checkpoints for exact payload equality
across measured trials. Use `--reference-run ../bc-profile/trial-1` to check
equality against a specific trial from another run.

Use `--mode host` for inclusive host spans including gathering, staging and
consumer waits. Use `--mode cuda-trace` for a Chrome trace and GPU activity
summary over the training window selected by `--trace-wait`, `--trace-warmup`
and `--trace-steps`. Use uninstrumented timing runs for throughput comparisons.
