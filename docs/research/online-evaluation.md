# Online evaluation validation

This is historical closed-loop integration and measurement evidence,
not a current performance promise or a model-strength result.
See the [online package guide](../../online/README.md) for current operations
and the [validation index](../validation.md) for other research.

## Python online comparison smoke validation

The binding, batched scheduler and comparison records were checked with a real
Rust-teacher checkpoint. This is pipeline validation, not evidence of policy
strength: the model has hidden widths 16/8 and only one training epoch.

### Inputs and environment

Windows 10 19045 / Ryzen 7 3700X (16 logical processors), NVIDIA RTX 2070,
CPython 3.12.10, NumPy 2.5.4, PyTorch 2.14.1+cu126 (CUDA 12.6), and a Rust 1.75
release extension. The original corpus contained seeds 0–23, collected at
`145006344d0c5466fdffc794fdc679f4dc5e2f87` and prepared as cache
`a3144bbfda95c207b76c3f7232d260d16ac6e21519cd5420abd0fa582cef2872`.
Training used seed 31, batch 64, AdamW at 0.001, 1,228 train decisions (923
eligible) and one epoch; selected validation NLL was 1.7826. The resulting
`best.pt` SHA-256 was
`8cb26765f5aed61b5f896895137605359a37fbc3c7178cd61da5775ed6ef7880`.

The integrated wheel check at `ff6f227` recorded these actual content identities:

| Source | SHA-256 |
|---|---|
| Compiled binding/core/neural | `bac3278a85712cb0d2274b69e4dfb1ccab9ad73fe89548754a068dd5f85be2ab` |
| Installed online Python | `6568296e3cf26d0e1111c931b340a80b7b0f6abac8092b1bcb07480657161015` |
| Imported offline Python | `b2c40331c926ad7bd27c17f8eb41958aacb8f7400c1aed01ad176a3e7e90e16e` |

The preceding batching pilot used compiled-source hash
`19e1953ca2062b324a0673c3ddd2087215e2620386639938d9ea0d8b6308f7cd`
and online-source hash
`191605b9346fd90dc753afa115c40e56dda320e2a7332f1d8cdaabcb34a7dd16`.
Hashes identify actual bytes, including checkout line endings, not just commits.

### Recreate the workload

From a clean checkout with the packages installed, generate fresh inputs:

```text
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-collect -- collect \
  --root corpus --games 24 --seed-start 0 --shard-games 6
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-prepare -- prepare \
  --corpus corpus --out cache
python -m mahjong_bc train --dataset cache --output training --device cpu \
  --seed 31 --epochs 1 --batch-size 64 --learning-rate 0.001 \
  --state-hidden 16 --scorer-hidden 8
mahjong-eval compare --checkpoint training/best.pt --split validation \
  --seeds 2,10 --out validation-run --batch-size 12 --concurrent-games 12 \
  --decision-limit 10000 --teacher-cache-bytes 27787264 --device cpu
mahjong-eval compare --checkpoint training/best.pt --split test \
  --seeds 12 --out test-run --batch-size 6 --concurrent-games 6 \
  --decision-limit 10000 --teacher-cache-bytes 27787264 --device cpu
```

Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for the measured one-thread CPU
workload. Seeds 2/10 are validation buckets 85/83; seed 12 is test bucket 99
under `seed-sha256-80-10-10-v1`. Each seed plays all six neural-seat pairs.

### Measurements

The batching pilot used three fresh-process runs per setting, 12 games and 898
decisions per run (406 inference rows, 291 teacher queries). Seconds below are
play-only medians with observed ranges; they exclude imports, checkpoint loading,
plan writing and process startup. CUDA was synchronized around each batch.

| Device | Batch | Batches | Total seconds | Inference seconds |
|---|---|---|---|---|
| cpu | 1 | 406 | 0.230 [0.223–0.232] | 0.195 [0.189–0.198] |
| cpu | 4 | 158 | 0.133 [0.123–0.168] | 0.098 [0.093–0.127] |
| cpu | 12 | 82 | 0.092 [0.091–0.104] | 0.060 [0.059–0.067] |
| cuda:0 | 1 | 406 | 0.884 [0.861–0.964] | 0.844 [0.824–0.913] |
| cuda:0 | 4 | 158 | 0.497 [0.481–0.523] | 0.461 [0.445–0.486] |
| cuda:0 | 12 | 82 | 0.401 [0.368–0.413] | 0.369 [0.333–0.377] |

CPU with one batch per wave was faster for this small model/workload; this does
not establish a general device preference. All settings agreed on aggregate
scores and decision totals, not verified action-by-action equivalence.

The separate integrated-wheel run produced these CPU observations:

| Split | Games | Decisions (forced) | Inference rows/batches | Teacher queries | Total / reset / step / teacher / inference seconds |
|---|---|---|---|---|---|
| validation | 12 | 898 (201) | 406 / 82 | 291 | 0.0787 / 0.0003 / 0.0044 / 0.0126 / 0.0506 |
| test | 6 | 435 (96) | 219 / 74 | 120 | 0.0533 / 0.0001 / 0.0024 / 0.0071 / 0.0372 |

Validation neural mean was -2.8333 (seed means -3.1667 and -2.5000), delta
-5.6667, neural-mean SE 0.3333 and delta SE 0.6667. Test neural mean was -2.6667,
delta -5.3333, with SE unavailable for one seed. All games ended by wall
exhaustion. Online scores did not select the checkpoint.

### Verification and retention

The installed wheel passed **152 online tests**, including real-corpus handoff,
exception pickling, failure retention and persisted-score disagreement. A copied
checkpoint also completed both comparisons from an unrelated working directory
with an I/O audit hook blocking the corpus, cache and training directory. A
stdlib-only probe independently recomputed placement coverage, zero-sum scores,
checkpoint identity, decision totals, mean scores and seed-block SE from JSON.

An sdist unpacked inside an outer checkout rebuilt with Rust 1.75, imported from
its installed wheel and completed a 53-decision teacher game. Its content hash
was independently recomputed; Git identity was unavailable rather than borrowed
from the outer checkout. CI exercises both packaging and corpus handoff. Local
execution was Windows MSVC; WSL failed to start
(`HCS/ERROR_NO_SYSTEM_RESOURCES`), so Ubuntu verification is left to CI.

Disposable corpora, caches, checkpoints, raw runs, probes, build outputs and the
subagent worktree were removed after preserving this record. Git retains the
regression fixtures and source, not the exact measurement artifacts; the commands
above regenerate the workload, not the old bytes or timing samples.
