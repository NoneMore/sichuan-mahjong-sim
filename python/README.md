# mahjong-bc

Offline behavior-cloning training and a shared PyTorch candidate-scoring policy.
Rust owns corpus validation and cache preparation; Python owns batching,
optimization, metrics and checkpoints. Neither training nor frozen-model
evaluation requires PyO3.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, install the
appropriate CPU or CUDA build from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install this package from the repository root:

```text
python -m venv .venv
# Activate .venv in your shell before installing dependencies.
python -m pip install -e "./python[dev]"
```

Prepare input with the Rust [dataset preparation commands](../crates/neural/README.md#dataset-preparation).
Keep corpus, cache and run outputs outside the source tree.

## Dataset

`load_prepared_cache` consumes `dataset.json` and independently memory-maps its
split-specific NumPy shards. The [Rust cache contract](../crates/neural/README.md#dataset-preparation)
defines array layouts, snapshot identity and provenance. Loading validates
supported versions, layout, inventory, checksums, shapes, offsets, targets,
metadata and counts; it does not read the original corpus. Python treats
`cache_id` as an opaque identity defined by Rust.

Forced decisions remain in the cache but are excluded from training and primary
metrics. Empty splits are supported; training requires eligible decisions in
both train and validation. Use `num_workers=0` with memory-mapped datasets:
worker spawning would otherwise copy the maps.

## Policy and training

`CandidatePolicy` returns float32 logits from a state MLP and a shared nonlinear
scorer over `[state_representation, candidate_features]`. It has no dropout,
normalization or candidate-position input. Batch padding is masked for loss and
selection; greedy ties choose canonical-first, and categorical sampling uses
the masked softmax.

Every epoch visits each eligible training decision once in seeded shuffled
order, including the final partial batch. Masked cross-entropy weights decisions
equally. Evaluation reports decision-mean NLL/top-1, per-stage support/NLL/top-1,
and support/top-1 by teacher action kind. Forced counts are separate and
zero-support metrics are undefined.

Validation mean NLL selects `best.pt`; `last.pt` records completed epochs for
recovery. Checkpoints include versioned model/configuration, optimizer and RNG
states, and the selected best weights. Publication is atomic. Resume replays an
interrupted epoch and requires the same dataset identity, model/training
configuration, device type and dependency versions. Output/log paths may change:
both best and last checkpoints are republished in the destination directory.
`run.json` records configuration, source/dependency provenance and metrics.

## Commands

Run from the repository root after installation:

```text
python -m mahjong_bc train --dataset CACHE --output RUN \
  --seed 0 --epochs 100 --batch-size 256 --learning-rate 1e-3 \
  --state-hidden 256,128 --scorer-hidden 128 --device cpu

python -m mahjong_bc train --dataset CACHE --output NEW_RUN \
  --resume RUN/last.pt --device cpu

python -m mahjong_bc evaluate --dataset CACHE --checkpoint RUN/best.pt \
  --split test --batch-size 512 --output metrics.json

python -m mahjong_bc inspect --checkpoint RUN/best.pt
python -m mahjong_bc cache-info --dataset CACHE
python -m mahjong_bc version
```

For a new single-GPU run, use `--device cuda`; resume keeps the original device
type. New runs require seed, epochs, batch size and learning rate, and refuse
output directories containing existing run artifacts. Resume uses the saved
configuration and rejects conflicting options. Frozen evaluation accepts only
validation or test splits and never writes a checkpoint.

## Verification

```text
python -m pytest python/tests
```

Tests use synthetic caches, covering policy invariance, metrics, complete epochs,
atomic checkpoints and recovery. The CUDA smoke test runs when CUDA is available;
CPU CI skips it. Real corpus handoff and CPU/CUDA acceptance observations are
recorded in [validation](../docs/validation.md#offline-behavior-cloning-smoke-validation).
