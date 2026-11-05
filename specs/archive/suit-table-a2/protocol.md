# Suit-table A2 — experiment protocol

A2 is a candidate tightening of `suit_table`'s transition enumeration in
`src/policy/analysis.rs`. This directory holds the minimal, reproducible
experiment material for it. The archived
[heuristic-analysis-optimization](../../archive/heuristic-analysis-optimization/)
Change remains historical and is **not** modified; its collision-free helper
material (A patch, frozen trace collector, frozen 64-game trace corpus) is
reused by path.

Nothing here changes production code, rules, public API, cache semantics or game
driving. A2's source and focused tests live in the implementation change; this
Change only builds, measures and records.

## Variants

All three variants are rebuilt in this session from one fixed mainline revision.
No historical timing replaces the baseline.

| Label | Source | Meaning |
| --- | --- | --- |
| `M` | current B-only checkout `5ad7540` unchanged | mainline: original `suit_table` + retained seven-pairs formula (B). `origin/main` is still `e9a38a2` without B, so this checkout — not `origin/main` — is the requested baseline. |
| `MA` | `5ad7540` + archived `patches/A.patch` | full A (all three loop bounds) + B |
| `A2` | `5ad7540` + this Change's `patches/A2.patch` | A2 (tighter sequence/triplet bounds, pair still innermost) + B |

The archived `A.patch` is applied verbatim from the mainline commit's blob
(a `+9` line offset; verified to apply). It is read from git rather than from
the CRLF working copy so the patch bytes are the committed LF bytes.

## Phase 1 — build (serial, no measurement)

Run once, before any measurement, with no benchmark or test running:

```powershell
python -B specs/archive/suit-table-a2/build_variants.py $env:TEMP/suit-table-a2-builds
```

`build_variants.py`:

1. creates one new detached worktree at the pinned mainline;
2. per variant: `git reset --hard`, apply that variant's patches in order,
   save `<variant>.patch`, snapshot `sources/<variant>.analysis.rs`, run
   `cargo build --release --example heuristic_bench`, copy the executable,
   record patch/binary/source hashes;
3. per variant: pre-generate the release library assembly with
   `cargo rustc --release --lib -- --emit=asm` and copy the emitted `.s` into
   `asm/<variant>/`. These static artifacts are generated **now** so a later
   regression can be inspected without any further build;
4. per variant: compile the reused frozen collector
   (`archive/.../tools/trace_corpus.rs`) against that variant's rlib, capture the
   frozen 64-game corpus and require it to equal
   `archive/.../baseline/trace-corpus.txt` (revision line excluded), so all three
   current variants match the same frozen behavioral baseline.

Then compile the microbenchmark binaries (still build phase):

```powershell
python -B specs/archive/suit-table-a2/microbench.py --prepare $env:TEMP/suit-table-a2-builds
```

`--prepare` extracts `fn suit_table` text from each
`sources/<variant>.analysis.rs`, generates the timed harness (function text
unmodified) and the counting harness (same text with counters injected), and
compiles both with `rustc --edition=2021 -O`. No production file is modified and
no production instrumentation is added; the extraction and injection are
asserted to hit exactly one site each (the counting injection at the leaf
transition loop, see 2a), and the extracted text hash is recorded.

Build conditions are identical for all variants: default Cargo release profile,
no `RUSTFLAGS`/`CARGO_*` overrides, same toolchain, same target directory
conditions, same worktree. Recorded per run: `rustc -Vv`, `cargo -V`, relevant
environment variables, CPU model, OS caption/version, physical memory, commit
and patch hashes. Each variant's trace hash is recorded twice: over the raw
bytes and over the canonical text after excluding `revision:` header lines, and
the canonical form must equal the frozen corpus.

## Phase 2 — measurement (serial, interleaved, nothing else running)

Preconditions: all builds, asm generation, trace capture, tests and any other
agent activity have stopped. Measurements are serial; the three variants are
interleaved in a pre-fixed order and no run is retried or discarded.

### 2a. Microbenchmark (supplementary, per-call `suit_table` cost)

```powershell
python -B specs/archive/suit-table-a2/microbench.py --measure $env:TEMP/suit-table-a2-builds specs/archive/suit-table-a2/measurements/01-microbench.json
```

Fixed inputs (held/capacity/sets triples; `capacity` is the per-rank target
capacity after missing suit and own melds, `target_sets = 4 - meld count`).
They are **synthetic local-function stress probes**: each triple is chosen to
exercise a loop-bound combination of the extracted function, and `capacity` and
`target_sets` are varied independently. Several rows are therefore not whole
hands reachable through declared melds (e.g. `peng-cap1-s4`'s `capacity`
`[1,4,...]` with `target_sets = 4`, `gang-cap0-front-s4`, `tail-pressure-s4`),
so these rows are not a call distribution, carry no frequency weights, and
support no weighted or end-to-end benefit claim. Only the per-input,
across-variant comparison of the same probe is used; the real workload is the
formal benchmark in 2b.

| name | hand | capacity | sets |
| --- | --- | --- | --- |
| `empty-cap4-s4` | `[0,0,0,0,0,0,0,0,0]` | all 4 | 4 |
| `typical-cap4-s4` | `[1,1,1,1,1,1,1,0,0]` | all 4 | 4 |
| `dense-cap4-s4` | `[4,4,4,0,0,0,0,0,0]` | all 4 | 4 |
| `wide-cap4-s4` | `[2,1,2,1,2,1,2,1,2]` | all 4 | 4 |
| `empty-cap4-s3` | `[0,0,0,0,0,0,0,0,0]` | all 4 | 3 |
| `typical-cap4-s2` | `[1,1,1,1,1,1,1,0,0]` | all 4 | 2 |
| `typical-cap4-s1` | `[1,1,1,1,1,1,1,0,0]` | all 4 | 1 |
| `typical-cap4-s0` | `[1,1,1,1,1,1,1,0,0]` | all 4 | 0 |
| `peng-cap1-s4` | `[1,1,1,1,0,0,0,0,0]` | `[1,4,4,4,4,4,4,4,4]` | 4 |
| `gang-cap0-front-s4` | `[0,1,1,1,1,1,1,0,0]` | `[0,4,4,4,4,4,4,4,4]` | 4 |
| `tail-pressure-s4` | `[1,1,1,0,0,0,0,0,0]` | `[0,0,0,4,4,4,4,4,4]` | 4 |
| `missing-suit-cap0-s4` | `[1,1,1,1,1,1,1,1,1]` | all 0 | 4 |

Timer protocol: timed binary calls the unmodified extracted `suit_table` with
`std::hint::black_box` on all three inputs and on the returned table; per input
`--warmup` discarded calls, then 5 rounds of `--iters` calls, printing each
round's ns/call. One invocation per variant per slot, 5 slots, pre-fixed
rotation `M,MA,A2` → `MA,A2,M` → `A2,M,MA` → repeat. Reported per (variant,
input): median and min over the 25 samples and the relative change against `M`.
Absolute ns are not comparable to the Cargo build; only the across-variant
comparison is meaningful.

Counting model: never timed, run once per variant after all timing. Counters are
injected at statement sites that exist verbatim in all three variants:

- `accepted` — immediately before `let extra = (count as u8)...`, i.e. every
  transition that passed every validity check. **Must be identical across
  variants for every input**; a difference invalidates the experiment.
- `considered` — at the **leaf** transition loop, i.e. the innermost `for` loop
  whose body contains the `accepted` site. `M` and `A2` enumerate
  `sequences -> triplets -> pair`, so their leaf loop is the `pair`
  loop; archived full `A` enumerates `pair -> sequences -> triplets`, so its
  leaf loop is the `triplets` loop. One unit is therefore one enumerated leaf
  transition candidate in every variant, the only comparable unit.
- per-layer tallies — `rank`, `active` state, `pair`, `sequences` and
  `triplets` loop iterations are recorded alongside, untimed, so the layer a
  count came from is always visible in the evidence.

A `considered` count taken at an enclosing loop measures a different unit and is
not comparable across variants. The structural check `accepted <= considered`
cannot hold for such a count whenever the enclosing loop iterates fewer times
than the accepted transitions, which is what the first counting run ran into
(2c).

The counting harness also emits the full `[sets][pair]` table, which must equal
that variant's uninstrumented table and every other variant's table for the same
input; this keeps the instrumented model from drifting away from the real
`SuitTable`. A failed invariant stops the experiment with the evidence saved.

### 2b. Formal end-to-end benchmark

```powershell
python -B specs/archive/suit-table-a2/performance.py $env:TEMP/suit-table-a2-builds specs/archive/suit-table-a2/measurements/02-formal.json
```

Reuses the existing `heuristic-benchmark-v1` executable and workload
exclusively: `heuristic_bench --games 2000 --seed {7,10007} --warmup 20`.
No new game driver, no trace collection during measurement, no outer
process-time substitution.

Per seed, one round runs each of the three variants once in a pre-fixed order;
rotations are fixed before any result is seen:

| round | seed 7 | seed 10007 |
| --- | --- | --- |
| 1 | `M, MA, A2` | `A2, MA, M` |
| 2 | `MA, A2, M` | `MA, M, A2` |
| 3 | `A2, M, MA` | `M, A2, MA` |
| 4 (extension) | `M, MA, A2` | `A2, MA, M` |
| 5 (extension) | `MA, A2, M` | `MA, M, A2` |

Seed 10007 is the reversed rotation of seed 7, so the two seed groups are
mirrored and the order keeps rotating. Rounds 1–3 are mandatory. If the
`A2`-vs-`M` verdict is `uncertain` after round 3, rounds 4 and 5 are added for
both seeds, and both final comparisons use all observations. The `MA` control
cannot trigger an extension itself. No other extension, no reordering by
observed results, no retry: a failed invocation is retained and aborts the
experiment.

The extension is not A2-only in its execution: rounds 4–5 run all three
variants, and **both** comparisons are then judged from all five rounds so no
observation is discarded. Only the trigger is tied to A2 (`M` vs `MA` cannot
add rounds after round 3, so its verdict rests on rounds 1–3 only when no
extension happened and on all five rounds when one did).

Comparisons: `A-vs-M` and `A2-vs-M`, both against the same-session mainline
binary. Verdicts reuse the archived conservative range criterion and helper
functions (`archive/.../performance.py`): per seed, candidate max `<` baseline
min is a benefit, baseline max `<` candidate min is a regression, anything else
(overlap, equality, opposite seed directions) is uncertain. Every raw
stdout/stderr, exit status, command, UTC timestamp, run order, binary/patch hash,
environment and the derived medians/ranges/per-round relative changes are saved;
the JSON is written after every run and refuses to overwrite existing evidence.

### 2c. Corrected counting run

The first counting run (kept in `01-microbench.json`) injected `considered` at
the last `pair` loop before the `accepted` site. That is the leaf loop for `M`
and `A2`, but for `MA` it is the **outer** loop: the recorded `573` is full A's
`pair`-layer candidate count, while its leaf `triplets` layer enumerates `1183`
transitions. `accepted = 1183 > considered = 573` was the visible symptom; `573`
against `3036`/`1938` compared different units.

The counting injection was corrected to the leaf-loop rule above (plus the
per-layer tallies) and the corrected counting evidence was produced separately:

```powershell
python -B specs/archive/suit-table-a2/microbench.py --prepare-counts $env:TEMP/suit-table-a2-builds
python -B specs/archive/suit-table-a2/microbench.py --count $env:TEMP/suit-table-a2-builds `
  specs/archive/suit-table-a2/measurements/05-corrected-counts.json `
  specs/archive/suit-table-a2/measurements/01-microbench.json
```

`--prepare-counts` compiles **only** three new counting binaries
(`micro-<variant>-count-v2`) from the already-built source snapshots; it does not
run `--prepare`, does not rebuild, re-time or replace any existing binary, and
refuses to overwrite its `micro/prepare-counts.json` manifest. `--count` runs
those three binaries once each, serially, untimed, and refuses to overwrite
existing evidence. `05-corrected-counts.json` records the three commands,
source/binary hashes, raw stdout/stderr and exit status, every input definition,
the full table per variant and input, `accepted` equality, `considered`
self-consistency with its leaf layer, `accepted <= considered`, a known-answer
check for the full-capacity input, every instrumented table's digest against the
**timed** (`01`) and the earlier instrumented model's digest, and the per-layer
iteration counts.

`01-microbench.json` is retained byte-for-byte. All of its timing samples are
valid and were neither re-run nor discarded; the whole A2 decision rests on
those timings and on `02-formal.json`, which are untouched by the corrected
counting run. Only `summary.counts[*][*].considered` in `01` is the record of
the superseded definition and is not used for comparison. `05-corrected-counts.json`
is the valid counting evidence. Its `counting_definition.considered` prose
retains a stale `sets ->` prefix; there is no such transition loop in the
source, and its separately recorded `sets` tally is zero. The real order is
`sequences -> triplets -> pair` for M/A2, as the measured source and counters
show. This prose-only erratum changes no raw values; the current script's
metadata has been corrected.

## Phase 3 — static assembly analysis (only if A2 regresses or stays uncertain)

No building here; the assembly was generated in phase 1.

```powershell
python -B specs/archive/suit-table-a2/asm_report.py $env:TEMP/suit-table-a2-builds specs/archive/suit-table-a2/measurements/03-asm-static.json
```

Reports, per variant, the extracted `suit_table` (or its enclosing caller when
LLVM inlined it) with static instruction, conditional/unconditional branch,
call and backward-jump (loop) counts, plus unified diffs `M` vs `MA` and `M` vs
`A2`. These are locatable static code differences only: nothing here observes
runtime branch behavior, no backward-jump edge is an execution count, and no
branch-misprediction explanation is claimed.

## Artifact inventory

Kept in the artifact directory (rebuildable, hashes recorded): per-variant
patch, executable, source snapshot, assembly, collector, trace, microbench
harness sources and binaries (`micro-<variant>-{timing,count,count-v2}`),
`builds.json`, `micro/prepare.json`, `micro/prepare-counts.json`.
These describe build outputs, not checked-in binaries. After the final hash
verification, this run's temporary artifact directory and detached worktree were
removed; the scripts reconstruct them from the retained patches.

Kept in this Change (small, reviewable): `builds.json` (build/trace manifest),
`measurements/*.json` (raw runs plus summaries; `05` is the corrected counting
evidence, `01`'s counting field is superseded raw history), asm extracts/diffs, `results.md`
and `performance-summary.md`, and the candidate's `verification.json` written by
the implementation owner for the standard format/Clippy/debug+release test
runs. The A2 source+test patch is kept here even if A2 is withdrawn, together
with the raw records, so A/A2 can be reconstructed. Large per-variant traces are
not copied: their raw and canonical hashes plus the frozen corpus reference are
recorded instead. Scripts run with `python -B`, so no `__pycache__` is part of
the archived material.

## Decision and closure

Standard checks (`cargo fmt --all -- --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`,
`cargo test --release --all-targets`) and the focused A2 tests are run by the
implementation change, not re-run here.

A2 is retained **only** if all three variants stay behaviorally identical, the
counting invariants in the corrected counting record (`05-corrected-counts.json`)
hold, and the formal comparison shows an identifiable
benefit (candidate max < baseline min in both seed groups). This standard is
stricter than the archived Change's: an `uncertain` result is withdrawn even
when A2 looks like a simplification, and no claim of equal performance or of
"no regression" is derived from uncertainty. An identifiable regression also
withdraws A2 and, like uncertainty, triggers the phase-3 static assembly report
so the observed difference can be located in code instead of speculated about
dynamically. Either way the A2 source/test patch and every raw record are kept
for reconstruction.

The completed experiment is archived at `specs/archive/suit-table-a2/`.
Paths embedded in immutable raw records retain their original active-directory
locations; hashes describe the measured artifacts, not a renamed or rebuilt
binary. For a fresh reproduction, use new artifact/output directories as shown
in [results.md](results.md), rather than overwriting the archived evidence.

## Execution outcome (recorded after the run)

The phase-1 build and phase-2 microbenchmark and formal benchmark completed:
18/18 formal invocations succeeded, the `A2`-vs-`M` verdict was identifiable
after the three mandatory rounds, so no extension was needed and no sample was
retried or discarded. The pre-generated phase-3 assembly was additionally
inspected despite the benefit verdict, as a supplementary diagnostic only;
this added no timed runs. A2 showed an identifiable benefit in both seed groups and
is retained; the archived full-A variant reproduced its historical regression.
The first counting run (`01`) used an enclosing-loop `considered` definition for
`MA`; corrected counting evidence was produced afterwards without touching any
timing or binary from the measured runs, as described in 2c. Findings, static
assembly facts and the final-cleanup note are in [results.md](results.md); the
numbers are in [performance-summary.md](performance-summary.md).
