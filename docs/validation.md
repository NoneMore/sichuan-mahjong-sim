# Validation and initial native CPU baseline

## Environment and scope

- Toolchain: `rustc 1.98.0 (88d9e12ae 2026-08-18)`, Cargo 1.98.0.
- Host target: `x86_64-pc-windows-msvc`, LLVM 22.1.8.
- One Rust crate, no third-party dependencies, no unsafe code.
- Measurements use the default Cargo release profile, one process and one thread.
- `rust-version = "1.75"` is the declared minimum; this session validated the installed 1.98 toolchain, not a separately installed 1.75 toolchain.

## Repeatable validation commands

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --all-targets
cargo doc --no-deps
```

Final verification passed formatting, all-target/all-feature Clippy with warnings denied, warning-free rustdoc, and **113 tests in both debug and release** (92 library, 6 CLI, and 15 integration tests).

The suite includes:

- Pure hand shapes, seven pairs, open-hand waits and own-meld four-copy limits.
- Fixed fan facts, payer matrices, and all 256 four-seat combinations of winner / flower pig / tenpai / non-tenpai for exhaustive-draw transfers.
- Atomic exchange, private missing-suit collection, isolated response collection, multiple winners, kongs and robbery, passed-Hu locks, and exhaustive-draw refunds.
- Public-API rejection atomicity and pending-read idempotence, checking full diagnostic state before and after rejected submissions.
- Seeded replay and continuation, including cloning both the Game and the separate randomized policy streams.
- Policy failure propagation, per-drive decision limits, and CLI parsing and replay.
- Kong-discard event scoring versus the same ordinary discard, Pass followed by OpenKong, repeated Pass termination, and diagnostic rejection of inconsistent winner/discard metadata.
- 256 seeded randomized games, validating physical tile conservation and score balance after every automatic progression and accepted submission; half of the seeds also voluntarily decline Hu when another legal action is available.

Independent read-only reviews covered the pure foundation, native execution boundary, and authoritative Game. Execution documentation was corrected to distinguish Game checkpoints from stateful policy checkpoints. The concealed-count view deliberately counts physical concealed tiles only: a discard winner references the winning discard rather than owning a duplicated tile.

These are basic rule and invariant tests, not exhaustive state-space verification. Weak baseline policies do not make random self-play a substitute for constructed rare-rule scenarios.

## Initial measured batches

After compiling with `cargo build --release`, the executable was run directly:

```text
target/release/sichuan-mahjong-sim.exe --games 20000 --seed 7 --policy first
target/release/sichuan-mahjong-sim.exe --games 20000 --seed 7 --policy random
```

| Metric | First legal | Random legal |
|---|---:|---:|
| Games | 20,000 | 20,000 |
| Decisions | 1,377,921 | 1,456,109 |
| Wall time | 1.330 s | 1.392 s |
| Rule-driving time | 1.034 s | 1.052 s |
| Policy time | 0.063 s | 0.090 s |
| Games/sec | 15,042 | 14,368 |
| Decisions/sec | 1,036,349 | 1,046,097 |
| Three-winner endings | 18 | 6 |
| Wall-exhaustion endings | 19,982 | 19,994 |
| Aggregate score sum | 0 | 0 |

Terminal score totals were `[9502, 3177, -2623, -10056]` and `[2391, 3560, -2168, -3783]`, respectively. These weak policies frequently exhaust the wall; the figures do not measure playing strength or predict the cost of an analytical policy.

`policy_time` measures `Policy::choose`; `rules_time` measures `Game::next_decision` plus `Game::submit`. Total wall time also includes game construction, aggregation, timer overhead, and loop overhead, so the two component times need not sum to total time. Per-decision timing itself has a cost. Batch execution does not record decision traces or save per-step snapshots. The CLI uses its seed for both the initial game seed and random-policy stream initialization; the library API can seed them separately.

Rates are observations from this local machine, not performance requirements or a cross-machine benchmark. Re-run after changes rather than treating historical rates as current proof. Clone cost and more complex policy cost have not been separately benchmarked.

## Issue #1: decision hot-path buffer and legality reuse

### Change and regression checks

The decision path now clears and refills `Game::legal_actions` for missing-suit,
turn, and response choices instead of replacing its allocation. `submit` reads
whether the validated pending list offers `Hu` before clearing it, then passes
that result to turn application for the passed-Hu lock. No persistent cache,
public API change, policy change, or rule/settlement change was introduced.
Reuse applies to a game advanced in place: cloning preserves the pending action
contents, but does not promise spare `Vec` capacity, so a cloned game may need
to grow its buffer when a later decision has more actions.

Five added regression tests cover declining a self-draw by discarding, rejection
of stale Hu in a follow-up response, isolation of different seats' response
options, exact action ordering and turn-buffer reuse, and missing-suit ordering
and buffer reuse. All **118 tests** pass in both debug and release (97 library,
6 CLI, and 15 integration tests). The following also pass:

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

An external replay harness was compiled against each version's release library.
For each policy it constructed one runner (`Runner::first_legal()` or
`Runner::random(7)`) and called `run_new(seed)` for seeds `7..263`, retaining the
policy streams between games. It printed each game's seed, scores, winners,
finish reason, decision count, and complete `RunReport.trace`, excluding timings.
The two 512-game outputs compared byte-for-byte equal (SHA-256
`6006f8a9392cc135af4c86eef22124ef70d28c901c58b2fa3c7559f562d9ff90`).

### Measurement conditions

- Baseline source: `7a5cf48`; comparison: the implementation in this change.
  The baseline executable was preserved before edits; the optimized executable
  was built with `cargo build --release` after the source changes.
- CPU: AMD Ryzen 7 3700X 8-Core Processor (16 logical CPUs).
- OS: Windows 10, build 19045, x64; target `x86_64-pc-windows-msvc`.
- Toolchain: rustc 1.98.0 (`88d9e12ae`), Cargo 1.98.0, LLVM 22.1.8.
- Default Cargo release profile; direct executable invocation, one single-threaded
  simulator process at a time; no CPU affinity pinning. No builds or test runs
  overlapped the measured batches.
- Each version/policy received one unmeasured 20,000-game warm-up process.
  Each measured process ran 100,000 games with seed 7; five repetitions per
  version/policy. Odd repetitions ran baseline then optimized, with `first`
  before `random`; even repetitions reversed both orders. Every invocation
  started fresh policy streams. The existing per-decision timing remained enabled.

For each saved version, the measured commands were:

```text
sichuan-mahjong-sim.exe --games 100000 --seed 7 --policy first
sichuan-mahjong-sim.exe --games 100000 --seed 7 --policy random
```

### Results

Rates below are medians, with min-max ranges over the five runs. Timings are
medians in seconds; they are independently aggregated and need not come from
the same repetition.

| Metric | First before | First after | Random before | Random after |
|---|---:|---:|---:|---:|
| Games/sec | 14,533.03 | 16,038.99 | 14,337.70 | 15,610.97 |
| Games/sec range | 13,810.05-14,596.54 | 14,983.90-16,185.03 | 13,692.05-14,526.75 | 14,188.36-15,776.09 |
| Decisions/sec | 1,001,496.96 | 1,105,275.25 | 1,043,581.10 | 1,136,257.06 |
| Decisions/sec range | 951,674.85-1,005,873.50 | 1,032,567.33-1,115,339.00 | 996,586.50-1,057,341.22 | 1,032,711.11-1,148,275.49 |
| Total time | 6.881 | 6.235 | 6.975 | 6.406 |
| Rule-driving time | 5.366 | 4.732 | 5.283 | 4.711 |
| Policy time | 0.322 | 0.320 | 0.452 | 0.454 |

The median throughput changes were **+10.36%** for first-legal and **+8.88%**
for random-legal; rule-driving time decreased by 11.82% and 10.83%, respectively.
These are local observations, not required speedups or cross-machine promises;
run-to-run variation is visible, especially for random-legal.

All measured runs matched these non-timing outputs exactly, before and after:

| Check | First legal | Random legal |
|---|---|---|
| Decisions | 6,891,178 | 7,278,579 |
| Three-winner / wall-exhaustion endings | 115 / 99,885 | 27 / 99,973 |
| Terminal score totals | `[51528, 14913, -15135, -51306]` | `[24503, 12683, -10925, -26261]` |
| Wins by seat | `[6505, 6401, 6285, 5961]` | `[3432, 3306, 3115, 3006]` |
| Aggregate score sum | 0 | 0 |

## Shanten-and-ukeire policy implementation milestone

See the [shanten/ukeire implementation study](studies/shanten-ukeire.md) for
validation coverage, exploratory 24-game A/B comparisons, historical release
timings, reproduction instructions, and limitations.

## Missing-suit ablation: 1000x A/B study (2026-11-02)

See the [missing-suit ablation study](studies/missing-only-ablation.md) for the
48,000-game experimental setup, per-seed uncertainty convention, scores,
reproduction commands, and artifact provenance.

## Heuristic cost attribution

See the [heuristic cost-attribution study](studies/heuristic-cost-attribution.md)
for historical pre-optimization measurements. The
[post-optimization refresh](studies/heuristic-cost-attribution-refresh.md) profiles
`6d3173b` after both optimizations below, separating distance-only and masked suit
DP, with per-run evidence, reproduction steps and limits. The
[one-pass discard refresh](studies/heuristic-cost-attribution-discard.md) summarizes
`06fb919`: masked suit DP dominates, chiefly the needed-mask path. Its temporary
artifacts were deleted at the user's request; the summary retains measurement
conditions and limits. These are historical observations, not performance guarantees.

## Heuristic hand-analysis optimization

Two private exact-analysis optimizations are retained: B replaces the seven-pairs
DP with a pair-unit formula (every declared meld still disables that route), and
A2 prunes impossible suit-table transitions without changing loop order or DP
states. Rules, policy ranking, information boundaries, cache keys/lifetime and
checkpoint behavior are unchanged. The full A enumeration rewrite and the C/D
shortcut experiments were not retained.

Formatting, warning-denied Clippy and all-target tests pass, with **223 tests in
both debug and release**. Permanent coverage includes independent 0/2/4 target
costs over all 241 seven-pairs count classes, formula boundaries and meld gating,
fresh/reused cache equivalence, and complete suit-table comparisons against the
original enumeration and an independent concrete-target oracle. Historical
cross-version replay also matched 64 games and 3514 decisions, including full
traces, scores, winners and finish reasons.

### Historical performance and rerunning

The measurements used Windows 10 build 19045, Ryzen 7 3700X,
`x86_64-pc-windows-msvc`, rustc/Cargo 1.98.0, LLVM 22.1.8 and the default release
profile, without custom Rust flags or CPU affinity. Each comparison used
`heuristic-benchmark-v1`, 2000 games, warmup 20, first seeds 7 and 10007, and
three interleaved rounds per seed. Median batch times in seconds were:

| Comparison | Seed 7, before → after | Seed 10007, before → after |
| --- | --- | --- |
| B: `c9202e3` → `5ad7540` | 16.632 → 16.205 (−2.57%) | 17.124 → 16.830 (−1.72%) |
| A2 on B: `5ad7540` → `d29d864` | 15.924593 → 15.598216 (−2.05%) | 16.297933 → 16.042941 (−1.56%) |

Candidate maxima were below baseline minima in both seed groups for each
comparison. These are small local observations, not statistical significance,
a cross-machine promise or a measured combined speedup; A2's isolated
microbenchmark was mixed. Raw samples, the frozen trace corpus, experimental
patches and one-off tooling are not maintained in the current tree.

To remeasure, build each source revision in a separate checkout before any timed
runs, then invoke its saved executable directly (`.exe` on Windows):

```text
cargo build --release --example heuristic_bench
target/release/examples/heuristic_bench --games 2000 --seed 7 --warmup 20
target/release/examples/heuristic_bench --games 2000 --seed 10007 --warmup 20
```

Run serially without overlapping builds/tests. Interleave baseline and candidate
runs for at least three rounds per seed, varying order, and retain all samples
and non-time outputs with source, toolchain and environment details for the new
comparison. Fresh measurements do not reproduce the original timing samples.

## One-pass ukeire (2026-11-05)

Ukeire now reads an exact optimal-target missing-kind mask instead of repeating
distance analysis per hypothetical draw. B/A2 distance-only analysis and separate
decision-local mask caches preserve rules, policy ranking, and checkpoint semantics.

Formatting, warning-denied Clippy, and **238 tests in debug and release** pass.
Permanent coverage includes a frozen compatibility reference, concrete-target
oracles, full suit tables, ownership, ties, and cache ordering. Historical replay
matched 64 games (3514 decisions) plus 2234 clone/fresh-policy continuation decisions.

The serial default-release `heuristic-benchmark-v1` experiment used 2000 games,
warmup 20, seeds 7/10007, B/B calibration, and five interleaved pairs per seed.
Paired median total/policy reductions were 62.44%/62.86% and 61.95%/62.35%,
respectively, with 5/5 improvements and identical non-timing results. Pure-distance
instrumented microbenchmarks regressed slightly (cold 1.3%, warm 4.8%).

See the [archived summary](../specs/archive/one-pass-ukeire/results.md) for the
proof, environment and limitations. Experiment tools and raw evidence were deleted
at the user's request; only permanent tests, proof and historical summaries remain.
These are historical observations, not performance or statistical guarantees.

## One-pass discard (2026-11-05)

Discard ranking now derives exact candidate deficits from one original-hand
surplus-mask analysis, without changing policy choices or checkpoint semantics.
See the [archived summary](../specs/archive/one-pass-discard/results.md) for the
258-test verification, cross-version replay, interleaved A/B results, and limits.
One-off tools and raw measurements were deleted at the user's request; the
reported gains are historical local observations, not performance guarantees.

## Bounded cross-decision masked-DP cache

An explicit, byte-bounded `HeuristicCache` now memoizes exact masked suit tables
across decisions, games, and explicitly shared heuristic seats. Decision-local
caches remain the first lookup layer; ordinary distance tables, policy ranking,
information boundaries, and checkpoint semantics are unchanged. The retained
four-way round-robin layout uses a selected 26.5 MiB default and a configurable
53 MiB comparison tier, plus zero-budget mode.

Formatting, warning-denied Clippy/rustdoc, and **316 Rust tests in both debug
and release** passed. An independently built original source `8332537` matched
all eight fixed-corpus comparisons: 40 full games (2,179 decisions) and 39
checkpoint continuations (1,221 decisions), using disabled, one-group
eviction-heavy, 26.5 MiB and 53 MiB contexts in forward and reverse order.
Permanent tests include complete table/oracle comparisons, shared/local cache
isolation, resource accounting, and failure fallback.

The `heuristic-benchmark-v2` tool makes cold/warm cache inheritance and
initialization costs explicit. A converged, predeclared 276-arm experiment
completed in 221 seconds without failures or degradation. At seeds 7/10007,
26.5 MiB reduced paired 500-game cold-batch times by 41.87%/43.93%, 2,000-game
accumulation times by 54.73%/55.78%, and 8/12/16-worker process makespans by
21.82–27.09%; every main group passed its own B/B noise gate with 5/5 positive
pairs. Doubling memory did not meet the predeclared extra-benefit threshold.
Short 20-game batches did not pass the throughput gate; one-game cold-process
times increased by 60.86–64.46%. The pressure fixture used 1 MiB and does not
establish fully saturated production-budget throughput.

See the [implementation and verification record](../specs/archive/bounded-masked-dp-cache/results.md)
for the historical environment, memory and page-fault observations, and
limitations. Temporary experiment tools and raw measurement data were removed at
the user's request, so these figures cannot be recomputed from repository data.
They are historical local observations, not a cross-machine or all-workload
speedup guarantee.

## Bounded exact Needed-mask DP

Ukeire now passes its already-known exact deficit to a strict `g > d` cutoff in
the Needed-mask suit DP, retaining every optimal mask bit. Ordinary distance,
Surplus analysis, policy ordering and checkpoint semantics are unchanged.
Formatting, warning-denied Clippy and **271 tests in debug and release** passed;
the fixed 40-game comparison and checkpoint continuations matched the baseline,
and both benchmark seeds met the pre-registered paired acceptance criteria.

See the [archived result summary](../specs/archive/bounded-exact-needed-dp/results.md)
for the pruning contract, correctness scope, and historical workload and timing
figures. Raw counts, one-off tooling and binary mirrors were deleted at the
user's request; the summary cannot be recomputed from repository samples, and
its local observations are not a cross-machine performance promise.
