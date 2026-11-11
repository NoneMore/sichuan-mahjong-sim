# Compare policies and benchmark heuristics

These two standalone Rust examples provide balanced policy comparison and
four-seat heuristic benchmarking without extending the main CLI. Run commands
from the repository root; use [development](../development.md) for workspace
build and verification checks. This is an operational guide, not a claim of
current policy strength or performance.

## Policy A/B comparison example

`examples/policy_ab.rs` runs a balanced 2A2B comparison. By default A is
`FirstLegalPolicy` and B is `RandomPolicy`, which reproduces the original
behavior; `--a` and `--b` select `first`, `random`, `heuristic`, or
`missing-only` for each side. Every game seed is played six times with A in
seats `[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]` and B in the remaining seats, each
placement using fresh policies and a fresh `Runner`. All heuristic seats on
both sides share one explicit cache context across the complete invocation
(seeds and placements); non-heuristic seats do not use it. The cache is
memoization only and does not change the comparison's decisions. Each seat running
`RandomPolicy` (on either side) starts from
`policy::random_seat_seed(game_seed ^ policy_seed, seat)`; deterministic styles
ignore that stream.

```text
cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out ab-demo

# Heuristic policy as A against the deterministic baseline:
cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out ab-heuristic \
  --a heuristic --b first

# Missing-suit-only ablation as B against the baseline, and as A against the
# heuristic:
cargo run --release --example policy_ab -- \
  --seeds 1,2 --policy-seed 0 --decision-limit 10000 --out ab-missing-vs-first \
  --a first --b missing-only
cargo run --release --example policy_ab -- \
  --seeds 1,2 --policy-seed 0 --decision-limit 10000 --out ab-missing-vs-heuristic \
  --a missing-only --b heuristic
```

| Argument | Meaning |
|---|---|
| `--seeds LIST` | Comma-separated, nonempty, distinct `u64` game seeds. |
| `--policy-seed S` | `u64` seed for the per-seat random policy streams. |
| `--decision-limit N` | Positive per-game decision limit passed to the runner. |
| `--out DIR` | Output directory; it must not already exist. |
| `--a STYLE` | Policy in the two A seats: `first`, `random`, `heuristic`, or `missing-only`. Default `first`. |
| `--b STYLE` | Policy in the two remaining seats: `first`, `random`, `heuristic`, or `missing-only`. Default `random`. |
| `--cache-budget-bytes B` | Shared heuristic table budget in bytes; `0` disables it. Only valid when A or B is `heuristic`; the default is the selected 26.5 MiB library budget. |
| `-h`, `--help` | Print usage and exit. |

`--a` and `--b` are optional and independent, so `--a heuristic --b random` or
`--a first --b heuristic` work as well. The selection is recorded in `run.txt`
and `summary.txt`. Deterministic styles ignore the policy seed but still receive
their seat's derived stream seed. The example validates this comparison
workflow; it is not a policy-strength result.

All contexts report their budget, allocation, and state in cache diagnostics;
a failed allocation emits one warning per context while play continues. A
cache-disabled comparison still keeps decision-local analysis memoization.

`--out` uses a single `create_dir`, so an existing directory is refused rather
than overwritten, and its parent directory must already exist (for example,
create `runs/` before using `--out runs/ab-demo`).

Artifacts are written to the fresh output directory:

- `run.txt`: simulator revision and dirty status, toolchain, the debug-quoted
  argv (one quoted element per argument, so paths with spaces stay
  unambiguous), the selected policy names for A and B, policy seed and stream
  derivation, game seeds, decision limit, and placements. Revision, dirty
  status, and toolchain are collected with `git`/`rustc` against the source tree
  that built the example (`CARGO_MANIFEST_DIR`), not the caller's working
  directory; unavailable metadata is recorded as `unknown` with a warning.
- `games.tsv`: one tab-separated record per completed game (`seed`,
  `a_seat0`/`a_seat1`, four zero-sum scores by seat, end reason, decision
  count). Records are flushed as games finish, so a failed run keeps its
  completed records.
- `cache.txt` (when heuristic seats are present): the invocation-wide sharing
  scope, budget source, initial/final cache status, fixed context overhead,
  table allocation and counters, plus any allocation-failure warning. It is
  retained on execution failure alongside completed game records.
- `summary.txt`: `A_mean`, `B_mean`, `Delta = A_mean - B_mean` (positive
  favors A), the sample standard error over seed blocks (`SE`, unavailable for
  one seed), and each seed's `D`. The summary is recomputed from the persisted
  `games.tsv` records after all games. In both missing-only examples above, a
  negative `Delta` favors the policy upgrade in B; the ablation itself appears
  on different sides of the two comparisons.

An illegal action, decision-limit exhaustion, missing terminal result, output
error, or failed consistency check exits non-zero without writing a summary;
completed records remain for diagnosis. `SE` uses the seed block as the
sampling unit and describes variation among the provided seeds, not
population-level precision.

## Heuristic benchmark example

`examples/heuristic_bench.rs` measures complete, single-threaded
four-seat `HeuristicPolicy` self-play through `Runner::run_batch`.
It does not change game rules, policy decisions, or the main CLI.

```text
# Cold measured context; warmup only warms process-level state:
cargo run --release --example heuristic_bench -- --games 200 --seed 7 --warmup 20

# Explicitly inherit the warmed context, using the 53 MiB candidate:
cargo run --release --example heuristic_bench -- --games 200 --seed 7 \
  --warmup 20 --lifecycle warm --cache-budget-bytes 55574528
```

`--games N` (positive) and `--seed S` are required. `--warmup W` defaults to
`0`; `--lifecycle cold|warm` defaults to `cold`, and `warm` requires `W >= 1`.
`--cache-budget-bytes B` selects the shared table budget (`0` disables the
cross-decision cache; the selected 26.5 MiB default is 27,787,264 bytes; the
53 MiB comparison tier is 55,574,528 bytes).
`--help` prints usage. The measured games use wrapping seeds `S + i`; warmup
uses the adjacent `W` seeds preceding `S`. The tool rejects `N + W > 2^64`
to prevent overlap.

All four seats share one context, reused across measured games. With `cold`,
warmup uses a separate runner/context which is dropped before measurement;
the measured context starts empty, and its lazy initialization is included in
the measured batch time. With `warm`, one runner/context survives both phases,
and initialization that occurs during warmup is charged to warmup instead.
Cold `--warmup 0` in a fresh process also exercises full cold startup; measure
external launch-to-exit wall time separately to include process startup.

The `heuristic-benchmark-v2` report uses one `key: value` per line. It records
the workload, lifecycle, sharing scope, budget, actual table allocation, cache
state/counters before and after measurement, initialization boundary, separate
warmup time, measured game/decision counts, batch/policy/rules times,
throughput, mean policy cost, and terminal-game statistics. Cache-status
snapshots are outside the timed batch. Field names carry units; `seat_order`
labels seat arrays and `completed_games` records actual completions separately
from requested `games`. Unavailable rates use `unavailable`.

Measured `total_time` includes game construction, driving and aggregation, not
runner construction, reporting, or warmup; policy and rules timings need not
sum to it. Do not mix these batch times with external process wall times in a
paired ratio. The mean policy cost covers every decision kind; aggregate game
results do not establish action-by-action equivalence. Warmup does not reset
process-level state or guarantee a speedup.

The command reports one observation, not a statistical conclusion. Record the
source revision, build/toolchain, environment and repeat-run conditions.
Allocation degradation remains visible in the report and warns once per
context; retain such runs but do not use them for performance adoption. Failed
input, execution, or report output makes a run invalid; no partial batch is
reported as a normal measurement.

Historical benchmark observations, artifact-retention limitations, and the
cache-adoption results belong in [validation](../validation.md) and the
[historical validation record](../validation.md#exact-analysis-changes-and-profiling).

## Related references

- [Experiment architecture](../architecture/experiments.md) explains
  responsibility and replay boundaries.
- [Rust library guide](../library.md) covers policy construction and
  shared heuristic cache contexts.
- [Historical validation](../validation.md) records point-in-time outcomes,
  not current speed or policy-strength guarantees.
