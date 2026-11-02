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
