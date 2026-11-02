# Shanten-and-ukeire policy implementation milestone

`HeuristicPolicy` and selectable policies in the existing A/B example preserve
fixed rules and the `Game`/`Policy`/`Runner` contracts. All **178 tests** pass in
debug and release (125 library, 6 CLI, 17 integration, 30 A/B example);
formatting, all-target Clippy with warnings denied, and warning-free rustdoc
also pass. The [archived proposal](../../specs/archive/shanten-ukeire-policy/spec.md)
and its design are historical intent, not a second implementation authority.

New coverage includes independent standard/seven-pairs target enumeration,
own-meld copy limits, exhaustive subsequent-discard normalization, cache-order
independence, exhausted structural waits, per-kind ownership through calls,
ranking/call gates, seeded replay, and cloned-game continuation with fresh
policies. Some kong ordering/gate branches use synthetic progress ranks;
replacement draws, robbery risk, and refunds remain outside the structural
kong proxy. Coverage is bounded, not exhaustive game-space verification.

## Conditions and reproduction

- Measured production code is recorded in commit `d27b96f`; the runs preceded
  that commit. After the behavior-preserving cleanup, both A/B raw records and
  summaries matched the original runs byte-for-byte.
- AMD Ryzen 7 3700X; Windows 10 build 19045 x64;
  `x86_64-pc-windows-msvc`; rustc 1.98.0 (`88d9e12ae`), Cargo 1.98.0,
  LLVM 22.1.8. The declared Rust 1.75 minimum was not separately tested.
- Default release profile, one single-threaded simulator process at a time,
  no affinity pinning, no overlapping builds/tests. Correctness runs preceded
  timing, with no dedicated warm-up. Each timing below is a single run.
- Convenience seeds `1,2,3,4` were fixed before viewing outcomes. Each contrast
  uses all six balanced 2A2B placements per seed, fresh independent policies per
  game, policy seed 0, and a 10,000-decision limit (24 games per contrast).

Use fresh local output directories; generated artifacts are not kept in the
specification archive:

```text
cargo build --release --example policy_ab --lib
target/release/examples/policy_ab.exe --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out target/shanten-ab-first --a heuristic --b first
target/release/examples/policy_ab.exe --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out target/shanten-ab-random --a heuristic --b random
```

## Exploratory scores

A is heuristic; B is the named baseline. Means and Delta are terminal net score
units per player per game. The complete seed block, not a placement or player,
is the sampling unit for SE.

| Metric | B = FirstLegalPolicy | B = RandomPolicy |
|---|---:|---:|
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
`[2.833333, 3.333333, 6.333333, 5.000000]` against random. No policy was tuned on
these outcomes. Four convenience seed blocks and their SE do not establish
population-level precision or a confirmatory strength claim. No win rates or
settlement diagnostics are inferred from scores or finish reasons.

## Historical release timing

A one-off local harness ran the same fresh mixed-policy games through `Runner`.
Whole-loop wall time included policy/game construction and trace collection,
but excluded A/B metadata and file I/O. Each seat had its own timing-only
wrapper around `choose` to measure heuristic cost; no game information or
counters were shared between policies. Timer/counter overhead was not
subtracted. Decision and score totals matched the A/B runs. The temporary
harness and raw outputs are not maintained as repository tooling.

| Metric | Heuristic / first | Heuristic / random |
|---|---:|---:|
| Whole workload wall time (s) | 0.106246 | 0.101817 |
| Runner policy / rules time (s) | 0.104619 / 0.001266 | 0.100348 / 0.001121 |
| Games/sec | 225.891 | 235.717 |
| Decisions/sec | 16,047.678 | 16,932.339 |
| Heuristic decisions / choose time (s) | 730 / 0.104433 | 701 / 0.100144 |
| Mean heuristic choose cost (microseconds) | 143.059 | 142.859 |

Cost averages all heuristic choices, including short-circuit actions, not just
analyzed discards. These roughly 0.1-second single-run observations are noisy,
not a speed acceptance claim or a comparison to historical homogeneous baseline
throughput. Reproducing decision cost requires separate instrumentation of
`Policy::choose`; the A/B example does not persist these timings. Larger repeated
workloads and an explicit budget are needed to assess acceptable speed. The
main CLI and A/B artifact schema remain unchanged.
