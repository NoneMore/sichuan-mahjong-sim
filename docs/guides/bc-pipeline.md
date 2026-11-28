# Behavior-cloning pipeline: collection to online evaluation

This guide connects the existing Rust and Python tools end to end. It explains
execution order and the artifacts passed between components, not their complete
CLI references or file schemas. Run commands from the repository root and
consult the linked module guides for supported flags and recovery behavior.
For versioned handoffs and contract owners, see the
[data contract map](../reference/data-contracts.md).

## Prerequisites and workspace

- Use the current Rust toolchain (minimum Rust 1.75). The `bc-collect` binary
  requires a **clean, committed source build** to attest teacher provenance.
- Use Python **3.12 or later** for this entire sequence: the offline package
  supports earlier Python releases, but the online PyO3 evaluation package
  requires 3.12+. Install packages in the repository's ignored `.venv/`.
- The sample uses sibling directories `../bc-data`, `../bc-cache`, `../bc-run`,
  and `../online-eval-validation` to keep generated artifacts outside the
  source tree. Choose new output destinations for a fresh run.

## 1. Collect teacher episodes in Rust

```sh
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-collect -- collect \
  --root ../bc-data --games 100 --seed-start 0 --shard-games 20 --cache-bytes 0

cargo run --locked -p sichuan-mahjong-neural --bin bc-collect -- verify --root ../bc-data
```

The collector records complete four-seat `HeuristicPolicy` episodes, including
forced choices. `../bc-data/manifest.json` is the committed inventory of
immutable JSONL shards. Source revision, encoding version, seeds, and split
membership are part of the corpus contract. To resume or append, use the
collector's supported append behavior rather than editing shards manually.

See [teacher data collection](../../crates/neural/README.md#teacher-data-collection)
for locking, retry, append, and reporting details.

## 2. Prepare and verify the NumPy cache

```sh
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-prepare -- prepare \
  --corpus ../bc-data --out ../bc-cache

cargo run --locked -p sichuan-mahjong-neural --bin bc-prepare -- verify --cache ../bc-cache
```

`../bc-cache/dataset.json` and its sharded `.npy` arrays form a self-contained
prepared snapshot, independent of later corpus appends. `--out` must not
already exist; after failed preparation, select a fresh output directory
or follow the preparation guide's cleanup instructions.

The [preparation guide](../../crates/neural/README.md#dataset-preparation)
defines the array protocol, checksums, cache identity, and publication rules.

## 3. Install and inspect the offline dataset

```sh
python -m venv .venv
source .venv/bin/activate  # POSIX; use .venv/Scripts/activate in Windows Git Bash
# The CPU example uses the PyTorch CPU wheel; select the correct wheel for CUDA.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './python[dev]'
python -m mahjong_bc cache-info --dataset ../bc-cache
```

The Python loader consumes the prepared cache, not the original Rust corpus.
If `.venv/` already exists, reuse it instead of recreating it. On Windows Git
Bash, activate with `source .venv/Scripts/activate` instead of the POSIX
command. See the [offline BC package guide](../../python/README.md) for dataset
access and cache validation details.

## 4. Train, select, and evaluate a frozen policy

For the 50,000-game configuration, see the [BC baseline recipe](bc-baseline.md).

```sh
python -m mahjong_bc train --dataset ../bc-cache --output ../bc-run \
  --seed 7 --epochs 2 --batch-size 64 --learning-rate 0.001 --device cpu

python -m mahjong_bc evaluate --dataset ../bc-cache \
  --checkpoint ../bc-run/best.pt --split validation
```

Training uses eligible decisions with more than one candidate and selects
`best.pt` by validation mean NLL. `last.pt` supports training recovery. Use
a small corpus for a smoke run.
Training requires the dataset to contain usable training and validation
examples; collect more games if a split has insufficient support.

See [training, metrics, and checkpoints](../../python/README.md#training-metrics-and-checkpoints)
for checkpoint compatibility, resume constraints, and evaluation semantics.

## 5. Compare frozen policies in the online environment

```sh
python -m pip install -e './online[evaluation]'
python -m mahjong_eval version

mahjong-eval compare --checkpoint ../bc-run/best.pt --split validation \
  --seeds 2,10 --out ../online-eval-validation \
  --batch-size 12 --concurrent-games 12 --device cpu
```

The online package builds its PyO3 extension outside the root Rust workspace.
Its comparison command runs fresh seeded games with balanced seat placements
and records a plan, terminal records, timings, and a recomputed summary.
Use seeds belonging to the requested split. This example uses validation
seeds `2,10` and the `best.pt` selected during training. Compare also accepts
compatible `policy` exports. The caller chooses the checkpoint for each run.

See [online install and run](../../online/README.md#install-and-run)
and [records and provenance](../../online/README.md#records-and-provenance)
for the full command contract and incomplete-run behavior.

For component ownership, see the [neural architecture guide](../architecture/neural.md).
