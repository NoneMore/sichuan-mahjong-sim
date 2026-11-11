# Run seeded self-play

Use the simulator's root CLI for reproducible, single-threaded games with
the `first` or `random` legal-choice baseline. Run commands from the
repository root. For Rust installation, build checks, and the minimum
supported toolchain, see [development](../development.md#running).

## Run a batch

Run a reproducible self-play batch:

```text
cargo run --release -- --games 1000 --seed 7 --policy first
cargo run --release -- --games 1000 --seed 7 --policy random
```

CLI arguments:

| Argument | Meaning |
|---|---|
| `--games N` | Number of games; must be at least 1. Default 1. |
| `--seed S` | Seeds the first game and, with `--policy random`, the seat policy streams. Game `i` uses wrapping `S + i`. Default 0. |
| `--policy first\|random` | Policy for every seat. Default `first`. |
| `-h`, `--help` | Print usage and exit; arguments after it are ignored. |

The CLI deliberately has a single `--seed` with two roles: it seeds the game
sequence and, for `--policy random`, the policy streams. The library API sets
them independently by choosing the game seed passed to `Game::new` (or
`run_new`/`run_batch`) and the policy seed passed to `Runner::random` or to
explicitly constructed `RandomPolicy` values.

Unknown arguments, missing values, malformed numbers, an unknown policy, and
`--games 0` are rejected with a non-zero exit status. `-h`/`--help` short-
circuits on the usual CLI convention: it prints usage and exits successfully,
ignoring every argument that follows it. Arguments before it are still
validated. The output reports finish reasons, terminal score totals, mean score
per seat, win counts, total decisions, separate rules/policy timings, and
`games/sec` plus `decisions/sec`.

Measurements are single-threaded and include game construction in the batch
wall time. When comparing builds, record the command, seed, policy, and observed
rates.

## Next steps

- For `Runner::run_new`, `run_batch`, individual policies, and report fields,
  see the [Rust library guide](../library.md#runner-api).
- For full four-seat heuristic self-play benchmarks and balanced policy
  comparisons, see [policy comparison](policy-comparison.md). The root CLI
  does not expose heuristic or missing-only styles.
- For stateful-policy checkpoints and deterministic continuation, see
  [checkpoint and replay](checkpoint-replay.md).
