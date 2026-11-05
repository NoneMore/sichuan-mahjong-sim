# Execution boundaries and pre-implementation plan

Baseline: `c9202e33ac78cfb819def7339c3986c58cc3d13f` (clean checkout before work).
The original executable was built in a detached worktree before integrating
any candidate: `cargo build --release --example heuristic_bench`.

## Work units

- **Baseline behavioral evidence**: freeze seeds `7..=38` and `10007..=10038`
  (64 games), use `Runner::run_new` and the same Runner across games, retain
  nonempty complete traces and non-time results. Independent of candidate work;
  baseline capture must precede candidate comparison.
- **A**: tighten only transition enumeration, retain every `SuitTable` entry;
  independent original-bound oracle and enumeration-count evidence accompany it.
- **B**: formula for seven pairs; independent target-cost oracle over all 241
  count classes plus boundary/gating cases accompanies it.
- **C**: skip missing suit; identity, missing-tile and cache-order equivalence
  tests accompany it. B and C are separate patches and separate experiments.
- **D experiment**: remove only ukeire's direct-win shortcut, audit caller
  contracts and compare original shortcut and zero-deficit/win equivalence.
  Implementation correctness does not depend on A/B/C; its incremental
  performance comparison does depend on selection of the mainline combination.
- **Integration and acceptance**: compare each candidate and combination against
  the frozen behavioral corpus, standard and release checks, then serial bounded
  performance runs. Do not overlap measurements with builds/tests or other
  benchmarks. Decide retention from correctness, measured verdict and actual
  simplification, not an assumed speedup. Update algorithm comments and report
  only relevant implemented changes.

No tracker publication, new game-driving logic, public API, production
instrumentation or rule changes are part of these units.

## Performance execution

`performance.py` invokes only preserved builds of the existing benchmark with
`--games 2000 --seed {7,10007} --warmup 20`. For each comparison it runs three
pairs per seed; baseline first on odd pairs for seed 7, candidate first on odd
pairs for seed 10007. If the range verdict is uncertain it extends each seed
with pairs 4 and 5, continuing that alternation. It saves every attempt and raw
stdout/stderr, stops on failure without retry, and calculates its final verdict
from all observations. Extension is chosen to inform retention of the candidate
or combination; no experiments exceed five pairs per seed. All binaries are
built before a measurement phase. Trace collection is separate and untimed.

Environment: Windows 10 build 19045, AMD Ryzen 7 3700X (8 physical/16 logical
cores), 33,448,124 KiB visible memory, x86_64-pc-windows-msvc;
`rustc 1.98.0 (88d9e12ae 2026-08-18)`,
`cargo 1.98.0 (797e8a9bc 2026-08-05)`, LLVM 22.1.8.
Default Cargo release profile, no Rust flags or CPU affinity changes. Other
system activity is not strictly controlled; range comparisons are conservative
engineering observations, not statistical significance claims.

Before candidate integration, the baseline also passed `cargo fmt --all -- --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test --all-targets`
(209 tests across all targets). These are baseline observations, not final
candidate validation. The baseline corpus was then captured against the unchanged
detached baseline worktree: 64 games, 3514 decisions, per-range digests
`0338b336d275ea05` and `583986f70bb9614c`. The saved file was independently
checked for nonempty game traces, contiguous indices and agreement of actual
trace lengths with decision counts before any candidate source integration.

Candidate source provenance (isolated agent commits; the retained production
patches, rather than availability of temporary branches, are the reconstruction
authority):

- A: `e42f55a0efc5a544890779992fbcdceec033fef3` followed by
  `c06876ca2f1aefaa3096129e7f7587dd3b1e6d00`.
- B: `d5e729d4ed9bb26f670207d8e1ccdff113c66156`.
- C: `65d7b30de0a1df397790df54a51791dc276eea0c` (relative to B).
- D: `ecaf52f6e93772d20e6c1748d840b864e34131ac`.

Parent integration only consolidated test-module declarations and clarified
comments (including that remaining seven-pair units cost two copies, not
necessarily a new tile kind). `patches/*.patch` contain the exact production
changes used for the measured variants; B and C patches apply independently to
the baseline. Test-only original references are not in production patches.

The integrated experimental ABCD source passed all four required commands
(format, warning-denied Clippy, debug all-targets, release all-targets),
with 230 tests in each test profile (151 library + 79 other-target tests).
All six rebuilt variants passed the frozen full-trace comparison (only the
revision label is excluded from comparison). The collector itself checks each
trace is nonempty and its length equals that game's decision count.
`builds.json` retains exact patch/binary/trace hashes, toolchain and build flags.
The formal baseline is rebuilt from the same fixed revision in the same
isolated source directory as candidates; the original pre-edit baseline
executable was preserved separately throughout measurement. No historical
measurement was reused.

## Rebuilding and collecting the same variants

Run from the repository root (PowerShell example, choose a fresh output path):

```powershell
$change = 'specs/archive/heuristic-analysis-optimization'
$artifacts = "$env:TEMP/heuristic-opt-builds"
python -B "$change/build_variants.py" $artifacts --variants baseline A B C ABC ABCD
# This builds baseline, A, B, C, ABC, ABCD serially and collects/compares traces.
# Inspect builds.json in the output directory; binaries and per-variant source
# patches remain there, while the retained patches under $change reconstruct
# each candidate without depending on temporary implementation branches.
```

Actual artifact directory for this session:
`C:/Users/HellG/AppData/Local/Temp/heuristic-opt-builds`.
After all build/test activity has stopped, each formal comparison uses:

```powershell
python -B "$change/performance.py" A-vs-baseline "$artifacts/baseline.exe" "$artifacts/A.exe" "$change/measurements/01-A.json"
python -B "$change/performance.py" B-vs-baseline "$artifacts/baseline.exe" "$artifacts/B.exe" "$change/measurements/02-B.json"
python -B "$change/performance.py" C-vs-baseline "$artifacts/baseline.exe" "$artifacts/C.exe" "$change/measurements/03-C.json"
```

The output directory `measurements` must exist. Source variant selection is
made before its comparison; the script refuses to overwrite previous evidence.
Every process invocation and its UTC start time/order/raw result are retained
in JSON. Use a different output name for a separately authorized new experiment,
not to silently discard or repeat one in this bounded budget.

## Decisions after the single-candidate phase

The formal sequence was A, B, C. All 44 invocations completed successfully;
there were no retries or discarded observations. A and B each stopped after
three pairs per seed with identifiable regression/benefit respectively. C
extended from three to five pairs per seed and remained uncertain.

- **A withdrawn**: both baseline maxima are less than both corresponding
  candidate minima. Fewer counted candidates did not yield less wall time.
- **B retained as mainline**: both candidate maxima are below their corresponding
  baseline minima, and the formula removes the per-kind target DP.
- **C declined**: ranges overlap and nine of ten pairs slow down. No speedup is
  established; avoiding an identity pass removes no existing merge algorithm,
  so its modest simplification is not sufficient to keep this special case.
  This is a conservative choice, not a claim of identifiable regression.

A/C production patches are retained unmodified. Their test-only algorithms are
isolated in `patches/A-tests.patch` / `patches/C-tests.patch` rather than retained
as production test modules for optimizations that are no longer shipped. Each
production+test pair was reapplied to a clean baseline worktree and its four
focused tests passed. A's counting model now explicitly checks original
transition legality, rather than asserting an unused rejection counter is zero.

The BD experimental integration (without A/C) passed all four standard commands,
222 tests in each profile. Before the next measurement phase, BD was built and
its full corpus compared against the original frozen baseline:

```powershell
$finalArtifacts = "$env:TEMP/heuristic-opt-final-builds"
python -B "$change/build_variants.py" $finalArtifacts --variants BD
python -B "$change/performance.py" D-increment-vs-B "$artifacts/B.exe" "$finalArtifacts/BD.exe" "$change/measurements/04-D-increment.json"
# Only if D is retained, separately compare the final combination:
python -B "$change/performance.py" BD-vs-baseline "$artifacts/baseline.exe" "$finalArtifacts/BD.exe" "$change/measurements/05-BD-final.json"
```

The exact BD build manifest is `builds-retained.json`; it does not overwrite
`builds.json`. Every measurement role is mapped by binary SHA-256 to one of
these manifests: 01 uses original/A, 02 original/B, 03 original/C, 04 B/BD,
and any 05 uses original/BD. The selected mainline B is identical to the single
B candidate, so `02-B.json` also supplies its combination-vs-original evidence;
no duplicate comparison is needed. No ABC/ABCD performance comparison is used.

For an archived A, C or D experiment, create a fresh detached worktree at the
baseline, apply `patches/<candidate>.patch` followed by
`patches/<candidate>-tests.patch`, then run
`cargo test --lib policy::analysis::optimization_a_tests` or the corresponding
`optimization_c_tests` / `optimization_d_tests` from that worktree. No permanent
second implementation is introduced in production. Final comment-only review
clarifies the zero-deficit size/meld premise and limits a one-suit test helper's
explanation to its actual fixture; these comments have no runtime effect.

## Final closure

D used all five pairs per seed (20 invocations), remained uncertain, and was
not retained. Its contract/zero-deficit proof is correct and removing the
branch simplifies code, but 8/10 paired totals increase and no incremental
benefit is established. We choose to keep the existing direct-win shortcut;
uncertainty is not relabeled as regression or equality. Consequently the
conditional `05-BD-final.json` command above was **not executed**, and no further
comparison is needed: final B is identical to the candidate in `02-B.json`.

The final B-only source passed format, warning-denied Clippy, debug all-targets
and release all-targets again, with 214 tests per profile. Its freshly collected
full action corpus also matches the frozen baseline. The cache-order test now
explicitly checks fresh-per-hand and repeated calls for both deficit and ukeire.
All three withdrawn production+test pairs were rebuilt independently (A 4, C 4,
D 8 focused tests). The final source, trace and audit outcomes are recorded in
`verification.json`; all 64 formal measurement invocations succeeded, with zero
retries and no discarded samples. Detailed observations and retention rationale
are in `performance-summary.md` and `results.md`.

After verification, temporary worktrees, candidate-only local branches and
rebuildable binaries were removed. The paths above describe the original
measurement environment, not checked-in executables. Source patches, complete
raw measurement JSON, build manifests and the frozen baseline trace remain
available for reconstruction. Four duplicate console summaries were removed
only after confirming their run order and summary exactly matched the JSON;
no raw run or measurement sample was removed. Change-local attributes preserve
LF in runnable scripts, the canonical trace and unified patches; patch context
markers are intentionally exempt from trailing-whitespace checks. Before
publication, the staged LF patches reconstructed every measured variant's exact
source-diff SHA-256, and the A/C/D focused tests passed again.

A single preliminary baseline smoke run (`--games 1 --seed 7 --warmup 0`)
completed before formal comparisons; its 0.007283 s batch time is not a formal
sample or evidence of improvement.
