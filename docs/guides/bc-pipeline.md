# Behavior-cloning pipeline: collection to online evaluation

This guide connects the existing Rust and Python tools end to end. It explains
execution order and the artifacts passed between components, not their complete
CLI references or file schemas. Run commands from the repository root and
consult the linked module guides for supported flags and recovery behavior.

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
# Activate the environment, or use its Python executable for all commands.
# The CPU example uses the PyTorch CPU wheel; select the correct wheel for CUDA.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './python[dev]'
python -m mahjong_bc cache-info --dataset ../bc-cache
```

The Python loader consumes the prepared cache, not the original Rust corpus.
It checks dataset identities and mappings before exposing mmap shards and
padded candidate batches. The root `.venv/` is ignored by Git; if it already
exists, reuse it instead of recreating it. See the [offline BC package guide](../../python/README.md)
for shell-specific environment activation and dataset access.

## 4. Train, select, and evaluate a frozen policy

```sh
python -m mahjong_bc train --dataset ../bc-cache --output ../bc-run \
  --seed 7 --epochs 2 --batch-size 64 --learning-rate 0.001 --device cpu

python -m mahjong_bc evaluate --dataset ../bc-cache \
  --checkpoint ../bc-run/best.pt --split validation
```

Training uses eligible decisions with more than one candidate and selects
`best.pt` solely by validation mean NLL. `last.pt` is for training recovery;
it is not the checkpoint selected for formal online comparisons. A small
corpus is useful for a smoke run, not for claims about policy strength.
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
Use seeds belonging to the requested split; `2,10` are the validation-seed
example documented by the online package. Online scores do not select or
rewrite the offline checkpoint. A complete comparison requires the offline
`best.pt` role, not `last.pt`.

See [online install and run](../../online/README.md#install-and-run)
and [records and provenance](../../online/README.md#records-and-provenance)
for the full command contract and incomplete-run behavior.

## What each stage owns

| Stage | Output | Authoritative detail |
| --- | --- | --- |
| Rust collection | Committed JSONL corpus and manifest | [Neural crate](../../crates/neural/README.md#teacher-data-collection) |
| Rust preparation | Sharded `.npy` cache and `dataset.json` | [Neural crate](../../crates/neural/README.md#dataset-preparation) |
| Python offline BC | `best.pt`, `last.pt`, `run.json` | [Python package](../../python/README.md) |
| Python online comparison | Plan, terminal records, costs, summary | [Online package](../../online/README.md) |

For component ownership, see the [neural architecture guide](../architecture/neural.md).
For dated measurements and reproduction limits, see [historical validation](../validation.md).
