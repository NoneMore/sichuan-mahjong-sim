# Implementation and exploratory measurements

## Delivered behavior

`HeuristicPolicy` is implemented in `src/policy/heuristic.rs`, with private exact
winning-target distance and ukeire analysis in `src/policy/analysis.rs`. Both
baseline policies, the fixed rules, and the `Game`/`Policy`/`Runner` contracts
are preserved. Memoization is decision-local; no logical checkpoint state or
RNG is introduced.

The delivery choice is optional `--a first|random|heuristic` and
`--b first|random|heuristic` in `examples/policy_ab.rs`, defaulting to the original
first/random comparison. The main CLI is unchanged. The original
[specification](spec.md) and [design](design.md) are preserved as historical
proposal documents; implementation evidence is the checked-in Rust and tests.

## Verification

All **178 tests** pass in both debug and release: 125 library, 6 CLI,
17 integration, and 30 A/B-example tests. These checks passed:

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

New coverage includes independent standard/seven-pairs target enumeration,
own-meld copy limits, non-winning draws normalized by exhaustive subsequent
discards, cache-fill-order independence, exhausted structural waits, per-kind
physical ownership through hypothetical calls/discards, ranking and call gates,
added kongs, seeded replay, and cloned-game continuation with fresh policies.
Independent read-only review found no remaining concrete functional defect.
The A/B argument tests assert the intended error rather than potentially
failing on missing unrelated required inputs. Pre-PR cleanup removed a duplicate
Hu/Pass helper and redundant total-count assertions while retaining per-kind
ownership checks. After cleanup, both A/B raw records and summaries matched the
original runs byte-for-byte; all checks and measurements were rerun.

Coverage is bounded, not exhaustive game-space verification. Some kong
ordering/gate branches use synthetic progress ranks. Replacement draws,
robbery risk, and refunds remain outside the specified structural kong proxy.

## Source and environment

- Base revision: `50f40774ffb1997fba706553b85d25900dbc09f1`, with the uncommitted
  implementation present (the retained run metadata records a dirty tree).
- AMD Ryzen 7 3700X; Windows 10 build 19045 x64.
- rustc 1.98.0 (`88d9e12ae`), Cargo 1.98.0, LLVM 22.1.8;
  `x86_64-pc-windows-msvc`. The declared Rust 1.75 minimum was not separately
  tested.
- Default Cargo release profile; one single-threaded simulator process at a
  time, no affinity pinning, no builds or tests overlapping the timed runs.
  Correctness runs preceded timing. The timings below are one post-cleanup run
  per contrast, not a repeated performance study; no dedicated warm-up was used.

Raw-byte SHA-256 fingerprints identify the measured changed production sources
and the separate timing reproducer (line-ending conversion changes raw hashes):

```text
e9222d4f31332e4333dc763e09c88eddf26bfd76e54705acafdfbc25ca80fbac  src/policy.rs
60b6b5823168ec12cc2299adb29bd662ed628d6890694868117600085d04bb2a  src/policy/analysis.rs
b68e9f42b0100c1d981d136460ca6e7c806495229f25b3d3a2a9bdf68b76252c  src/policy/heuristic.rs
ae5072a410a10ecc2f1570b3b5cad0ffa6ecc730631ffee7ec89ce0aaf2fc39c  src/lib.rs
4b517505d10a400238558b56c8bef6c527c88e7bc60b1930ea9f222648abdd57  examples/policy_ab.rs
35401927c4b9426428c0923fc1d3b3c9d9accf89dea1ebe78194b9dbf67a15c2  specs/archive/shanten-ukeire-policy/timing.rs
```

## Exploratory A/B scores

Convenience seeds `1,2,3,4` were fixed before viewing outcomes. Each contrast
uses all six balanced 2A2B placements per seed, fresh independent policies per
game, policy seed 0, and a 10,000-decision limit. These are development seeds,
not a randomly sampled or independent confirmation corpus.

```text
cargo build --release --example policy_ab --lib
target/release/examples/policy_ab.exe --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out specs/archive/shanten-ukeire-policy/ab-first --a heuristic --b first
target/release/examples/policy_ab.exe --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out specs/archive/shanten-ukeire-policy/ab-random --a heuristic --b random
```

Use fresh output directories when rerunning. The original artifact schema is
unchanged: run metadata, per-game records, and recomputed summaries are retained
in [ab-first](ab-first/summary.txt) and [ab-random](ab-random/summary.txt).
A is heuristic; B is the named baseline. Means and Delta are terminal net score
units per player per game. The complete seed block, not a placement or player,
is the sampling unit for SE.

| Metric | B = FirstLegalPolicy | B = RandomPolicy |
|---|---:|---:|
| Seeds / completed games | 4 / 24 | 4 / 24 |
| A_mean | 2.187500 | 2.187500 |
| B_mean | -2.187500 | -2.187500 |
| Delta | 4.375000 | 4.375000 |
| Seed-block SE | 1.266329 | 0.800391 |
| Decisions | 1,705 | 1,724 |
| Three-winner / wall-exhaustion endings | 3 / 21 | 0 / 24 |
| Terminal score totals by seat | `[46, 5, -53, 2]` | `[-17, 4, 12, 1]` |
| Aggregate score sum | 0 | 0 |

Per-seed Delta values for seeds 1 through 4 were
`[4.333333, 7.666667, 1.500000, 4.000000]` against first and
`[2.833333, 3.333333, 6.333333, 5.000000]` against random. Both contrasts happen
to have the same overall mean, but their seed-level outcomes differ.

The favorable point estimates describe only these opponent populations and
these seeds. Four convenience blocks and their SE do **not** establish
population-level precision, a confirmatory strength claim, or a practically
meaningful improvement with uncertainty accounted for. No policy was tuned on
these outcomes. No win rates or settlement diagnostics are inferred from scores
or finish reasons. Practical thresholds, confirmation sampling/size/interval
methods, and acceptable numerical runtime remain future research decisions.

## Release runtime observations

The separate [one-off timing reproducer](timing.rs) uses the same seeds and
fresh mixed-policy games through the existing `Runner`. It measures whole-loop
wall time (including policy/game construction and trace collection, excluding
A/B metadata and file I/O), existing Runner policy/rule times, and inner
heuristic `choose` time using a timing-only wrapper per seat. Counters contain
no game information, are not shared between policies, and do not affect
choices. Extra timer/counter overhead is not subtracted. Decision totals and
score totals matched the persisted A/B records.

```text
rustc --edition 2021 -O specs/archive/shanten-ukeire-policy/timing.rs --extern sichuan_mahjong_sim=target/release/libsichuan_mahjong_sim.rlib -L dependency=target/release/deps -o target/shanten-policy-timing.exe
target/shanten-policy-timing.exe first
target/shanten-policy-timing.exe random
```

| Metric | Heuristic / first | Heuristic / random |
|---|---:|---:|
| Whole workload wall time (s) | 0.106246 | 0.101817 |
| Runner policy time, all seats (s) | 0.104619 | 0.100348 |
| Runner rules time (s) | 0.001266 | 0.001121 |
| Games/sec | 225.891 | 235.717 |
| Decisions/sec | 16,047.678 | 16,932.339 |
| Heuristic decisions | 730 | 701 |
| Inner heuristic choose time (s) | 0.104433 | 0.100144 |
| Mean heuristic choose cost (microseconds) | 143.059 | 142.859 |

Heuristic decision cost averages all its choices, including short-circuit
opening/forced choices, not just analyzed discards. These roughly 0.1-second,
single-run observations are noisy and are neither a numerical speed acceptance
claim nor a comparison to historical homogeneous baseline throughput. Larger,
repeated workloads and an explicit budget are needed to assess acceptable speed.
