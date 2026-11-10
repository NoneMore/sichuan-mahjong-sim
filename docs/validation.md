# Historical validation and research outcomes

This page records **point-in-time** validation and experimental conclusions,
not the present test count, CI status, or performance/strategy-strength
guarantee. For current build, test, Rust 1.75 MSRV and supported experiments
see [development](development.md) and
[CI](../.github/workflows/rust-ci.yml). Code and tests are the authority on
implemented behavior; [architecture](architecture/overview.md) describes
the implemented subsystem boundaries.

Detailed original measurements and methods remain in the
[pre-retirement snapshot](https://github.com/NoneMore/sichuan-mahjong-sim/tree/f57e48fdd39a9571371194b46d61bfba2b20d795).
The snapshot also contains completed `specs/archive/` Changes and historical
`docs/studies/` reports. Neither is a maintained description of the current
implementation. Earlier locally deleted raw samples, binaries, traces,
prototypes and scripts are **not** restored by the snapshot.

## Game and runner baseline

Initial verification at the original 1.98 Windows toolchain milestone passed
**113 Debug and Release tests**, covering hand shapes, fixed fan/payment facts,
256 exhaustive-draw seat classifications, response/claim arbitration, illegal
submission atomicity, seeded replay and checkpoint continuation. A further
**256 seeded game** stress corpus checked physical tile ownership and zero-sum
score invariants after accepted transitions. These counts describe the
original milestone, not today's suite.

The later decision-hot-path change reused the Game's legal-action buffer and
avoided repeating a winning-shape check for passed-Hu handling without
changing action semantics. A fixed 512-game cross-version replay matched
byte-for-byte. Local paired 100,000-game batches observed median throughput
increases of **10.36%** (FirstLegal) and **8.88%** (Random), without a
cross-machine guarantee. The original
[validation record](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/validation.md#issue-1-decision-hot-path-buffer-and-legality-reuse)
preserves the exact environment, workload, measurements and coverage.

## Strategy comparisons

The early shanten/ukeire implementation milestone compared the heuristic
against FirstLegal and Random over just **four seeds / 24 games per contrast**.
The observed advantage was **+4.375** terminal net score units per player per
game in each contrast. Those small exploratory sets do not establish general
playing strength. [Original study](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/shanten-ukeire.md).

The subsequent balanced missing-suit ablation used **4,000 complete seed
blocks / 24,000 games per contrast**, 48,000 games altogether. With positive
values favoring the policy in B:

| Contrast (B over A) | Mean net score advantage per player/game | Seed-block SE | Descriptive 95% interval |
| --- | ---: | ---: | --- |
| MissingOnly over FirstLegal | **+4.026667** | 0.082964 | [3.864058, 4.189275] |
| Heuristic over MissingOnly | **+2.962542** | 0.028990 | [2.905720, 3.019363] |

The missing-suit choice contributed meaningfully in the tested mixtures,
but did not account for the full heuristic advantage. The two contrasts
have different opponent mixtures and **must not be added**; convenience
seeds and descriptive intervals do not demonstrate population
representativeness. Six seat placements sharing a seed are not independent
observations. [Original protocol, results and provenance](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/missing-only-ablation.md).

## Exact-analysis changes and profiling

The following milestones were **implemented and verified at their own
revisions**. Their correctness requirements and current behavior live in
source and permanent regression tests, not in old benchmark percentages.

- **Seven-pairs formula and A2 suit-table pruning:** adopted after
  independent exact-target/table comparisons, cache checks and seeded replay.
  Small release-batch performance differences were local observations only.
  [Original validation](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/validation.md#heuristic-hand-analysis-optimization).
- **One-pass ukeire:** replaced per-hypothetical-draw distance recomputation
  with an exact optimal-target needed mask, preserving all tied mask bits.
  Historical paired 2,000-game comparisons observed approximately 62%
  total/policy time reductions in both seed groups; no present-day speedup
  is implied. [Results and proof](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/one-pass-ukeire/results.md).
- **One-pass discard:** derived exact candidate deficits from original-hand
  surplus-mask analysis without changing action selection or checkpoint
  semantics. [Original verification and timing](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/one-pass-discard/results.md).
- **Bounded shared masked-DP cache:** adopted exact byte-bounded reuse across
  decisions with disabled and allocation-failure fallbacks. The selected
  default was 26.5 MiB. Large-batch improvements and cold-start slowdowns
  were both observed in the original acceptance protocol; neither is a
  general performance promise. [Original results](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/bounded-masked-dp-cache/results.md).
- **Bounded Needed-mask DP:** retained a strict `g > d` cutoff with
  equal-cost optimal-mask paths intact. [Original acceptance](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/specs/archive/bounded-exact-needed-dp/results.md).

The initial, intermediate and post-discard CPU attribution studies measure
**different historical implementations**; their old hotspot rankings
must not be projected onto today's code. In the post-discard `06fb919`
snapshot, masked per-suit DP represented **79.09%** of sampled worker
cost (needed **57.25%**, surplus **21.84%**) and plain distance DP **3.50%**.
Sample proportions are not speedup predictions. For the full methods,
revision boundaries and evidence limits, see the
[initial](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution.md),
[intermediate](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-refresh.md),
and [post-discard](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-discard.md)
reports and the committed
[measurement summary](https://github.com/NoneMore/sichuan-mahjong-sim/blob/f57e48fdd39a9571371194b46d61bfba2b20d795/docs/studies/heuristic-cost-attribution-refresh/measurements.json).

## Rust BC preparation smoke validation

A PR 1 working tree based on `2b49dd4`, extracting Rust preparation from
`298bbee`, passed acceptance on Windows x86-64 with Rust/Cargo 1.98.0,
Python 3.12.10 and NumPy 2.5.4. Collection used a clean build at
`2b49dd4e812bcc56e5844068b7594adbf8db6425`; preparation recorded its dirty
build separately. Reproduce the dataset from the respective checkouts with
fresh outputs outside the source trees:

```text
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-collect -- collect \
  --root ../phase1-corpus --games 4 --seed-start 0 --shard-games 2 \
  --cache-bytes 0 --report ../phase1-collection.json
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-prepare -- prepare \
  --corpus ../phase1-corpus --out ../phase1-cache \
  --shard-decisions 60 --shard-candidates 400
# Move phase1-corpus away before verifying the self-contained cache.
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-prepare -- verify \
  --cache ../phase1-cache
```

The cache retained **232 decisions, 1,374 candidates and 56 forced choices**.
Train/validation/test eligible support was **137 / 39 / 0**; the empty test
split had no shards. Independent `numpy.load(..., mmap_mode="r")` reads
matched every state/candidate float32 bit, candidate order, local target,
episode seed, decision index, actor, stage and target kind against the listed
JSONL records, and checked array checksums and inventory. Rust verification
passed after moving the original corpus away.

- Captured manifest SHA-256:
  `360e6716c586b9838eb5571b1112f18b0d2b1a3b306ab6c337cf44ccb86f72c4`.
- Cache ID:
  `8d5aa96394bd2dfa90218e19afd4778abf48bc8d2c6bfc7798c10533c629cf90`.

Rust formatting, warning-denied workspace Clippy, Debug/Release all-target
tests, doctests, warning-denied rustdoc, and the Rust 1.75 all-target check
passed on the integrated working tree. The MSRV check retained pre-existing
dead-code warnings. The preparation suite ran 12 tests on Windows; its
Unix-only dangling-symlink test was not exercised locally.

This checks the Rust/NumPy handoff, not a Python loader or training.
Temporary corpus, caches, reports, probe, and builds were removed after
verification. Permanent regression coverage is in
[`prepare_contract.rs`](../crates/neural/tests/prepare_contract.rs).

## Python dataset loader smoke validation

The PR 2 loader, extracted from `298bbee` onto `176f08d`, was checked on
Windows 10 x86-64 with Rust/Cargo 1.98.0, Python 3.12.10, NumPy 2.5.4 and
PyTorch 2.14.1+cu126, using CPU operations. Collection and preparation used a
clean Release build of `176f08d499f7186425eba27397dd30a5c56b040b`:

```text
git worktree add --detach ../bc-phase2-source 176f08d
cargo build --release --locked --manifest-path ../bc-phase2-source/Cargo.toml \
  -p sichuan-mahjong-neural --bins --target-dir ../bc-phase2-build
../bc-phase2-build/release/bc-collect.exe collect \
  --root ../bc-phase2-corpus --games 4 --seed-start 0 --shard-games 2 \
  --cache-bytes 0 --report ../bc-phase2-collection.json
../bc-phase2-build/release/bc-prepare.exe prepare \
  --corpus ../bc-phase2-corpus --out ../bc-phase2-cache \
  --shard-decisions 60 --shard-candidates 400
../bc-phase2-build/release/bc-prepare.exe verify --cache ../bc-phase2-cache
```

From the loader checkout, with dependencies installed in the root `.venv/`:

```text
.venv/Scripts/python.exe -m pip wheel --no-deps --no-build-isolation ./python \
  --wheel-dir ../bc-phase2-wheels
.venv/Scripts/python.exe -m pip install --no-deps --force-reinstall \
  ../bc-phase2-wheels/mahjong_bc-0.1.0-py3-none-any.whl
.venv/Scripts/python.exe -m pytest python/tests -q
.venv/Scripts/python.exe python/tools/verify_rust_handoff.py \
  --corpus ../bc-phase2-corpus --cache ../bc-phase2-cache
```

The installed wheel imported from `.venv/Lib/site-packages`. All **88 Python
tests** passed, including manifest rejection, Rust seed-split oracles and
live-view safety after dataset close. Formatting, warning-denied Clippy and
**386 Debug all-target Rust tests** passed. The probe matched every
state/candidate float32 bit, candidate order, local target and episode/actor/
stage/kind metadata against the captured JSONL records:

| Split | Shards | Decisions | Candidates | Forced | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3 | 179 | 1,047 | 42 | 137 |
| validation | 1 | 53 | 327 | 14 | 39 |
| test | 0 | 0 | 0 | 0 | 0 |

Reversed-index batches of size 17 preserved alignment and padding/masks across
shards, with final partial batches of 9 and 2; the empty test split yielded no
batches. Rust verify and Python `cache-info` passed with the corpus moved away.

- Captured manifest SHA-256:
  `a8fc07e3a7e530bb01ed1ae67b94aebe1db4180e1a1c5433f4510b4a52487492`.
- Opaque cache ID:
  `7d1d327313a71ea8ad85413a10583facf63922bad50586f31c67196b5f945802`.

Generated corpus, cache, reports, wheels and isolated builds were removed after
verification. The loader tests and [`verify_rust_handoff.py`](../python/tools/verify_rust_handoff.py)
preserve regression and reproduction coverage.

The maintenance follow-up based on `926be42` passed **114 Python tests** from
both source and an installed wheel, including delegated mmap checks and a
100-episode fixed-interval fixture. The same Rust checks passed again. Repeating
the four-game workload above with a clean Rust build at
`926be4270b7a14310a02dfd09dace254aedc475e` preserved all counts and bitwise
handoff checks, including corpus-independent loading. Its snapshot SHA-256 was
`de51e7b39a9997237332598a4969ffff74b4794a7e22d38437bf6ccc79e8f1fa`, and cache ID
`991af57a3eae6dda9311ff38b4e08e451072f33a9faae1e5ca687b5d822afc12`.
Follow-up artifacts were also removed after verification.

## Offline BC training smoke validation

PR 3 was extracted from `298bbee` onto `bce00a7` and checked on Windows 10
x86-64 with Rust/Cargo 1.98.0, Python 3.12.10, NumPy 2.5.4 and PyTorch
2.14.1+cu126. Editable and wheel installs each passed **178 Python tests**,
covering policy invariance, complete epochs, metric denominators,
validation-only selection, frozen evaluation, version rejection, atomic-write
failures and CLI output protection.

A clean Release Rust build at `bce00a729fb2a0a90ae90d8a479fc7b7899ec2e4`
produced all three splits. After [package installation](../python/README.md#setup-and-commands),
reproduce from the repository root with fresh sibling paths:

```text
git worktree add --detach ../bc-phase3-source bce00a7
cargo build --release --locked --manifest-path ../bc-phase3-source/Cargo.toml \
  -p sichuan-mahjong-neural --bins --target-dir ../bc-phase3-build
../bc-phase3-build/release/bc-collect.exe collect \
  --root ../bc-phase3-corpus --games 13 --seed-start 0 --shard-games 4 \
  --cache-bytes 0 --report ../bc-phase3-collection.json
../bc-phase3-build/release/bc-prepare.exe prepare \
  --corpus ../bc-phase3-corpus --out ../bc-phase3-cache \
  --shard-decisions 60 --shard-candidates 400
../bc-phase3-build/release/bc-prepare.exe verify --cache ../bc-phase3-cache
.venv/Scripts/python.exe python/tools/verify_rust_handoff.py \
  --corpus ../bc-phase3-corpus --cache ../bc-phase3-cache
.venv/Scripts/python.exe -m mahjong_bc train \
  --dataset ../bc-phase3-cache --output ../bc-phase3-run --seed 7 \
  --epochs 3 --batch-size 17 --learning-rate 0.01 \
  --state-hidden 16 --scorer-hidden 16 --device cpu \
  --log-file ../bc-phase3-train-summary.json
.venv/Scripts/python.exe -m mahjong_bc inspect \
  --checkpoint ../bc-phase3-run/best.pt
.venv/Scripts/python.exe -m mahjong_bc evaluate \
  --dataset ../bc-phase3-cache --checkpoint ../bc-phase3-run/best.pt \
  --split test --batch-size 17 --device cpu --output ../bc-phase3-test.json
```

On POSIX, use `.venv/bin/python` and omit executable `.exe` suffixes. The
handoff probe matched all state/candidate bits, local targets, candidate order
and episode/actor/stage/kind metadata. The cache contained **13 episodes,
773 decisions, 4,604 candidates and 195 forced decisions**:

| Split | Decisions | Eligible | Forced |
| --- | ---: | ---: | ---: |
| train | 610 | 460 | 150 |
| validation | 125 | 91 | 34 |
| test | 38 | 27 | 11 |

Three CPU epochs consumed **84 steps / 1,380 samples**, each visiting 460
eligible decisions in 28 batches (the last containing one decision). Validation
NLL was **1.736139 / 1.666645 / 1.637471**, selecting zero-based epoch 2.
Reloaded test evaluation reported NLL **1.431383**, top-1 **12/27**, and 11
forced decisions. Repeated evaluation and a rejected `--output` alias left the
checkpoint bytes unchanged. Module and installed console-script inspection
both passed. These smoke results do not establish playing strength.

- Captured manifest SHA-256:
  `99f1a21d0e41e527ea94794bc8d7669f3134cefb2c1192e00be5e1ae0298457a`.
- Cache ID:
  `11c4ce7353ad4872165d2cb695cb4d7194f5574201cc042079c302a4f3ff4b67`.
- Imported training-source SHA-256:
  `80e4b311a4032ddfc9feac03ef7762c28ded3bfbc76274a80efe4cabe187d8f7`.
  The wheel recorded null Git revision/dirty fields, not the outer repository.

The CUDA fixture smoke (`test_cuda_training_and_checkpoint_reload_smoke`) passed
on an **NVIDIA GeForce RTX 2070 / CUDA build 12.6**, covering training, CUDA
checkpoint reload and frozen evaluation. GPU CI and real-corpus GPU training
were not exercised. Rust formatting, warning-denied workspace Clippy and Debug
all-target tests passed. Recovery remains PR 4 work.

Temporary worktrees/builds, corpus/cache, wheel, checkpoints and reports were
removed; tests and the handoff probe preserve regression coverage. The project
`.venv` was restored to an editable install.

## Offline BC recovery validation

Recovery was verified on `9e57481` plus the recovery changes, on Windows 10
x86-64 with Rust/Cargo 1.98.0, Python 3.12.10, NumPy 2.5.4 and PyTorch
2.14.1+cu126. Editable and installed-wheel runs each passed **227 Python tests**,
including replay, RNG restoration, incompatible/corrupt recovery rejection,
atomic-write failures, and migration with no improvement or a completed run.
Rust formatting, warning-denied workspace Clippy and Debug all-target tests passed.

A clean Release build of `9e5748128699130a2bafdbb927b58cc165e7ecb0` collected
13 games at seeds 0–12, four games per shard, with `--cache-bytes 0`. Preparation
used `--shard-decisions 60 --shard-candidates 400`. Rust verify and the bitwise
Python handoff probe passed. The 13-shard cache held 773 decisions / 4,604
candidates; train/validation/test eligible counts were 460/91/27 and forced
counts 150/34/11. Reproduce inputs with the
[training smoke commands](#offline-bc-training-smoke-validation), substituting
`9e57481` and fresh `bc-phase4-*` paths.

The CPU recovery probe used one thread, seed 7, three epochs, batch size 17,
learning rate 0.01 and state/scorer widths 16. After one batch of the second
epoch it interrupted, resumed from the first completed epoch into a new directory,
and matched uninterrupted sample order, history, **84 steps / 1,380 samples**,
and model/optimizer states exactly (**`rtol=0, atol=0`**). A second migration
republished the completed run without further steps. Frozen evaluation preserved
checkpoint bytes and reported test NLL **1.431383**, top-1 **12/27**.

The same configuration passed on an **RTX 2070 (8 GiB), driver 591.74, CUDA build
12.6**: training, checkpoint reload, completed-run migration, module/console
inspection and frozen evaluation. Migrated model weights matched exactly and
the checkpoint retained the selected local device's RNG state. Validation NLL
was **1.735905 / 1.666422 / 1.637446**; test NLL **1.431555**, top-1 **12/27**.
Repeated evaluation preserved checkpoint bytes. These checks establish neither
cross-runtime/device equivalence nor online playing strength.

After [package installation](../python/README.md#setup-and-commands):

```text
python python/tools/verify_recovery.py --dataset ../bc-phase4-cache --output ../bc-phase4-recovery
python -m mahjong_bc train --dataset ../bc-phase4-cache --output ../bc-phase4-gpu \
  --seed 7 --epochs 3 --batch-size 17 --learning-rate 0.01 \
  --state-hidden 16 --scorer-hidden 16 --device cuda:0
python -m mahjong_bc train --dataset ../bc-phase4-cache --output ../bc-phase4-gpu-migrated \
  --resume ../bc-phase4-gpu/last.pt --device cuda:0
python -m mahjong_bc evaluate --dataset ../bc-phase4-cache \
  --checkpoint ../bc-phase4-gpu-migrated/best.pt --split test --batch-size 17 --device cuda:0
```

Provenance identifiers:

- Captured manifest SHA-256: `26411a72c29bedc7297f4227a052fca62d3e7172668e0c2dc9c6fc9640244de5`.
- Cache ID: `95147694b1d202759b9abfc2ed00fb11bc0b7992d863cf03923aa846d911f714`.
- Training-source SHA-256: `8f3024c465f8622fb5c64ebd094bb5be5c0193390469e2a25aa979c6058fbc10`.

Temporary corpus/cache, checkpoints, wheels and reports were removed after
verification. Tests and the handoff/recovery probes retain reproduction coverage;
raw artifacts are not archived. The repository `.venv` remains an editable install.

## Recovery and new measurements

The studies ran primarily on Windows 10 / Ryzen 7 3700X with Rust 1.98.
Many original samples, EXE/PDB files, local source snapshots and experimental
tools were intentionally removed before retirement. The committed records
document their limits; they do **not** let Git reconstruct missing raw data.
Other accepted, rejected and still-undecided experiment candidates and their
original branch SHA values are catalogued in
[historical retirement PR #23](https://github.com/NoneMore/sichuan-mahjong-sim/pull/23).
A correct-but-slow or inconclusive experiment must not be presented as an
accepted optimization.

To remeasure, check out the specified historical source revision, rebuild it
with the documented toolchain and workload, save new evidence and compare
controlled baseline/candidate runs. Current CI or a new benchmark cannot
retroactively validate the old numerical samples.
