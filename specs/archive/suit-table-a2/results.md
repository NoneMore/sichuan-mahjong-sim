# Suit-table A2 — results

Experiment protocol: [protocol.md](protocol.md). Raw records:
`measurements/01-microbench.json`, `02-formal.json`, `03-asm-static.json`,
`04-artifact-verification.json`, and the corrected counting record
`05-corrected-counts.json`; build/trace manifest: `builds.json`;
candidate source+test patch: `patches/A2.patch`. The standard format, Clippy and
debug/release test runs of the candidate are recorded by the implementation owner
in [verification.json](verification.json); this Change adds the variant-focused
oracle/counter/trace and performance evidence, not a duplicate test sweep.

## Outcome

**A2 is retained: identifiable end-to-end benefit in both seed groups**, with
behavior identical to the mainline and to the frozen corpus.

| Variant | Verdict | Evidence |
| --- | --- | --- |
| `A2` | identifiable benefit | A2 max < M min for seeds 7 and 10007 after three rounds |
| `MA` (archived full A) | identifiable regression | M max < MA min in both seed groups, ≈ +20% `total_time` |

`MA` reproducing the archived A regression under today's conditions is the
control that the harness is sensitive to real differences: the same three-round
evaluation that calls A a regression calls A2 a benefit.

## Variants and provenance

Mainline base is the current B-only checkout `5ad75404a5dea07971779d163d47a05da3b66e73`
(`origin/main` is still `e9a38a2`, which does not contain B). All three variants
were rebuilt serially in one fresh detached worktree in the same session; no
historical timing was reused.

| Variant | Patch SHA-256 | Binary SHA-256 | Source SHA-256 | Canonical trace SHA-256 |
| --- | --- | --- | --- | --- |
| `M` | — | `fbb49e7b2b3e2a00…` | `1f0dffb9f1a4b3e0…` | `e4534ee7968af2b6…` |
| `MA` | `03bfdd59d5b69b99…` | `90ee6b046677525b…` | `4e74a01778df1dad…` | `e4534ee7968af2b6…` |
| `A2` | `d2bff63b24743abd…` | `7f023d3400fe42d3…` | `db0935b19a05f885…` | `e4534ee7968af2b6…` |

First 16 hex digits shown; the exact values are in `builds.json` (`variants.<name>.{patch_sha256,binary_sha256,source_sha256,trace_sha256,trace_normalized_sha256}`)
and `measurements/04-artifact-verification.json`, which re-hashes every measured
patch, source snapshot, executable and microbenchmark binary after the runs and
found all of them matching the manifests recorded before measuring.
Source hashes describe the exact observed checkout bytes, not a portable
line-ending normalization: `analysis.rs` used CRLF, while the added test file
used LF. Another checkout's line-ending policy can change these byte hashes
without changing Rust behavior. The retained LF patch is the reconstruction
authority; the function-body comparison in `verification.json` normalizes line
endings through Python's text reader.

Build conditions: default Cargo release profile, no `RUSTFLAGS`/`CARGO_*`
overrides, `rustc 1.98.0 (88d9e12ae)`, `cargo 1.98.0 (797e8a9bc)`, LLVM 22.1.8,
`x86_64-pc-windows-msvc`, AMD Ryzen 7 3700X (8 cores / 16 threads), Windows 10
Pro 19045, 32,664 MiB visible memory, Python 3.12.10.

## Correctness

- **Frozen 64-game trace corpus**: all three variants reproduce
  `specs/archive/heuristic-analysis-optimization/baseline/trace-corpus.txt`
  exactly. Canonical trace SHA-256 after excluding `revision:` lines was
  `e4534ee7968a…` for `M`, `MA` and `A2` alike; raw trace bytes are recorded by
  hash rather than copied.
- **Counting invariants** (corrected record `05-corrected-counts.json`): the
  counting copy is a separate untimed instrumented copy of the extracted
  `suit_table`; `accepted` (every transition passing all validity checks) is
  identical for all three variants on all 12 inputs, every instrumented full
  table equals the other variants' tables and its own uninstrumented table
  digest from the timed runs, and `accepted <= considered` holds everywhere.
  `01-microbench.json` recorded the first two invariants with no failure; the
  leaf-superset check was added with the corrected definition, and its failure
  for `MA` under the old injection is exactly what exposed that definition
  error.
- **Residual difference**: none observed. `A2`'s table, transitions and action
  corpus are indistinguishable from `M` and from the archived baseline.

### Counting definition (corrected)

`considered` counts enumerated **leaf** transition candidates: the count is
injected at the innermost loop containing the accept site — the `pair` loop in
`M` and `A2` (`sequences -> triplets -> pair`) and the `triplets` loop
in archived full `A` (`pair -> sequences -> triplets`). Full-capacity
(`target_sets = 4`) values:

| Variant | accepted | considered (leaf) | counted at |
| --- | --- | --- | --- |
| `M` | 1183 | 3036 | `pair` loop |
| `MA` (full A) | 1183 | 1183 | `triplets` loop |
| `A2` | 1183 | 1938 | `pair` loop |

`MA`'s leaf count equals its accepted count because its loop bounds make every
enumerated `(pair, sequences, triplets)` triple legal, so no further rejection
remains at the accept site.

The first counting run, kept unchanged in `01-microbench.json`, injected
`considered` at the last `pair` loop before the accept site. That is the leaf
loop for `M` and `A2` but the **outer** loop for `MA`, whose recorded `573` is
its `pair`-layer candidate count, not a leaf count. `accepted = 1183 >
considered = 573` was the visible symptom of that mismatch, and `573` against
`3036`/`1938` compared different units — the earlier "A removes ~81% of the
candidates" reading was therefore wrong. `01`'s `considered` field is retained
as raw history of the superseded definition; all of `01`'s timing samples and
table digests are valid, were not re-run or discarded, and the A2 decision does
not depend on that field. Only the three corrected counting binaries were
compiled and run (`--prepare-counts` / `--count`, new file names, old binaries
and `micro/prepare.json` untouched), and `05-corrected-counts.json` is the valid
counting evidence.

With one consistent unit, `A` enumerates 1183 leaf candidates against `M`'s
3036 and `A2` 1938 for the full-capacity input (per-layer iteration counts are
in `05`). Candidate counts alone do not translate into speed: `MA` has fewer
leaf candidates than `A2` yet runs 20–22% slower end-to-end, and `A2`'s counts
are lower than `M`'s without a proportional time change. No speed conclusion is
derived from these counts; the end-to-end comparison below is the evidence.

## Formal benchmark

Workload: `heuristic_bench --games 2000 --seed {7,10007} --warmup 20`, three
rounds per seed with all three variants per round in a pre-fixed rotation, no
extension needed, 18/18 invocations successful, no retries, no discarded
samples. Each round's non-time batch fields were equal across the three
variants.

| Comparison | Seed 7 | Seed 10007 | Verdict |
| --- | --- | --- | --- |
| `A-vs-M` | 15.925 s → 19.278 s (+19.85…22.17%) | 16.298 s → 19.707 s (+20.64…21.30%) | regression |
| `A2-vs-M` | 15.925 s → 15.598 s (−0.91…−4.07%) | 16.298 s → 16.043 s (−0.48…−1.74%) | benefit |

Detailed medians, ranges, throughput and per-round changes:
[performance-summary.md](performance-summary.md). The A2 ranges do not overlap
the mainline ranges in either seed group (`M` 15.780–16.240 / 16.277–16.313,
`A2` 15.579–15.636 / 16.015–16.199), so the conservative engineering criterion
is satisfied. This is a small benefit for this machine and workload, not a
statistical or cross-machine claim.

## Static release assembly (supporting, not causal proof)

Generated before measurement from each variant with
`cargo rustc --release --lib -- --emit=asm`, so no build was needed afterwards.
The protocol required inspection on regression or uncertainty; it was also run
as a supplementary diagnostic despite A2's benefit. This added no timed runs
and did not alter the retention criterion.
`suit_table` is inlined into `DistanceEvaluator::deficit` in every variant; that
enclosing function is therefore the comparable unit and all of its branch
targets resolve.

| Variant | Instructions | Conditional branches | Loop edges | Multiplications | Divisions |
| --- | --- | --- | --- | --- | --- |
| `M` | 939 | 90 | 32 | 3 | 0 |
| `MA` | 879 | 74 | 28 | 4 | 0 |
| `A2` | 980 | 96 | 27 | 3 | 0 |

Locatable facts, not speculation:

- A2 emits 41 more instructions and 6 more conditional branches than `M` while
  having 5 fewer backward-jump loop edges. A backward-jump edge is only a static
  loop edge: it does not say how often that loop executes, so "5 fewer loop
  edges" is not evidence of fewer loop iterations or of a different loop shape
  at runtime.
- A2 also has three more `sub` sites and seven more `cmp` sites, consistent with
  the added guards (the `a + b` reservation, the capacity subtraction, the
  pair-free triplet stop), but they cannot all be attributed to those guards: no
  source-to-assembly mapping was retained, and both variants change the function
  prologue/stack frame (e.g. a different `subq $N, %rsp`) — A2 differs in 10 hunk
  regions and MA in 20 — so LLVM reallocated registers and frame slots across
  the whole inlined function. Static counts are compatible with the new guards
  being traded against extra instructions, but the dynamic cause of any timing
  difference is not established here.
- Full A's dynamic `remaining / 3` bound has a locatable constant-division
  sequence in `measurements/asm/MA-*.s` at lines 362–370:
  `movabsq $-6148914691236517205, %rdx; mulq %rdx; shrq %rdx`, followed by
  comparison/conditional selection of the triplet bound. This is reciprocal
  multiplication, **not** a hardware divide instruction. A2 avoids this bound;
  its monotone stop uses the pair-free minimum instead. It is an identifiable
  extra cost in A, not a quantitative explanation of A's entire regression:
  loop order, register allocation and dynamic execution also differ.
- No dynamic branch-hit or misprediction claim is made; no profiler was used.
  Extracts and unified diffs: `measurements/asm/`.

## Final cleanup

The two comment-only items noted during review (the A2 test fixture's generated
capacities 2 and 3, and the "four-copy capacity" phrasing now stated as per-rank
capacity) were applied by the implementation owner as comment-only edits after
the measured patch was frozen. They change no executable code, so the measured
binaries, patches and hashes remain the recorded evidence, and
`patches/A2.patch` is unchanged. The final function body is byte-identical to
the measured one. The final format/Clippy/debug/release sweep passes (223 tests
per profile), and a fresh final-release collector again reproduces the frozen
64-game, 3514-decision corpus; exact hashes and commands are in
[verification.json](verification.json). This completed Change is archived here.
After the final hash audit, the temporary detached worktree, source snapshots,
binaries and PDB files were removed. The source/test patch, manifests, complete
raw measurement records and relevant assembly extracts remain checked in;
the scripts below rebuild the removed artifacts in a fresh directory.

## Reproduction

```powershell
$change = 'specs/archive/suit-table-a2'
# Choose new paths; existing artifact directories and evidence files are refused.
$artifacts = "$env:TEMP/suit-table-a2-rerun-builds"
$out = "$env:TEMP/suit-table-a2-rerun-measurements"
New-Item -ItemType Directory $out -ErrorAction Stop
python -B "$change/build_variants.py" $artifacts          # M, MA, A2 + asm + traces
python -B "$change/microbench.py" --prepare $artifacts
# All build/test work must finish before the next two timed commands.
python -B "$change/microbench.py" --measure $artifacts "$out/01-microbench.json"
python -B "$change/performance.py" $artifacts "$out/02-formal.json"
python -B "$change/asm_report.py" $artifacts "$out/03-asm-static.json"
# Untimed corrected counters, after every timed run has finished:
python -B "$change/microbench.py" --prepare-counts $artifacts
python -B "$change/microbench.py" --count $artifacts `
  "$out/05-corrected-counts.json" "$out/01-microbench.json"
```

The measurement steps must remain serial and must not overlap any build or test;
the scripts refuse to overwrite existing evidence files (`--prepare-counts`
likewise refuses to overwrite its manifest). `--prepare-counts` and `--count`
produce no timing and do not rebuild or replace any measured binary.
