# Compare policies and benchmark heuristics

Run these examples from the repository root. Their experiment-specific
options are separate from the root self-play CLI.

## Balanced policy A/B comparison

`policy_ab` plays six balanced 2A2B placements for each seed, with
fresh policies and Runner per game. Options `--a` and `--b` accept
`first`, `random`, `heuristic` or `missing-only` (defaults:
`first` versus `random`). All heuristic seats share one explicit
cache context during the invocation. Random seat streams derive from
`random_seat_seed(game_seed ^ policy_seed, seat)`.

```sh
cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 \
  --a heuristic --b first --out ../ab-heuristic

cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 \
  --a missing-only --b heuristic --out ../ab-missing
```

| Option | Meaning |
| --- | --- |
| `--seeds LIST` | Nonempty comma-separated distinct game seeds |
| `--a STYLE`, `--b STYLE` | Policy on each side of the 2A2B comparison |
| `--policy-seed S` | Random policy stream seed; ignored by deterministic policies |
| `--decision-limit N` | Positive limit per game |
| `--cache-budget-bytes B` | Heuristic shared cache budget; 0 disables sharing |
| `--out DIR` | Fresh output directory; its parent must exist |

The directory contains `run.txt` (revision, options and seed plan),
`games.tsv` (one validated game per row), optional `cache.txt`
(context and allocation diagnostics), and `summary.txt` (published on
successful completion). Individual records remain after a failed game.

For each seed, average the difference between the two A-seat and B-seat
mean terminal scores over six placements. The summary reports the
mean Delta across seed blocks and the between-block standard error;
a positive Delta favors A. The independent sample unit is the seed
block, not an individual seat or game.

## Four-seat heuristic benchmark

`heuristic_bench` times complete batches of HeuristicPolicy self-play,
with all seats sharing one cache context. Run cold and warm cache
lifecycles separately.

```sh
cargo run --release --example heuristic_bench -- \
  --games 200 --seed 7 --warmup 20 --lifecycle cold

cargo run --release --example heuristic_bench -- \
  --games 200 --seed 7 --warmup 20 --lifecycle warm \
  --cache-budget-bytes 55574528
```

`--games N` and `--seed S` are required; `--warmup W`
defaults to 0; `--lifecycle cold|warm` defaults to cold.
Warm requires a positive warmup. The shared cache default is
27,787,264 bytes (26.5 MiB), and 0 disables it.

Cold uses a separate warmup runner/cache and measures a fresh context.
Warm reuses the warmup runner/cache, shifting initialization to the
warmup phase. Measured games use seeds `S + i`; warmup uses the
preceding W seeds.

The report includes workload and cache identity, measured games and
decisions, cache counters, total/policy/rules time, throughput and
terminal summaries. Total batch time includes game creation and
aggregation; process startup, warmup and report output are measured
separately. For optimization decisions, compare repeated runs using
the same toolchain, workload, seeds and cache lifecycle.

[Experiment ownership](../architecture/experiments.md) ·
[Rust library and caches](../library.md)
