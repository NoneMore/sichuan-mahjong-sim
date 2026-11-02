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
The harness and generated outputs were temporary validation artifacts, not new
project dependencies or APIs.

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
