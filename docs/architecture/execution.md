# Execution subsystem

[`Runner`](../../src/runner.rs) drives exactly four policies against a
[`Game`](game.md), but it does not implement Mahjong rules. The
[`main.rs` CLI](../../src/main.rs) is a thin batch-execution entry point.
For commands and public API examples, use [development](../development.md)
and the [README](../../README.md).

## Run boundary

At each step, Runner obtains the next pending `Decision` from Game, calls the
acting seat's `Policy::choose`, and submits that action to Game. An invalid
choice is returned as an error: Runner does not silently replace it with a
different move. It neither inspects concealed opponent information nor
reconstructs legality.

The three entry points differ in lifecycle, not game rules:

| Entry | State boundary |
| --- | --- |
| `run_new(seed)` | Create a new seeded Game, drive it and return a per-game report |
| `run_game(&mut game)` | Continue the supplied existing Game, including a checkpoint |
| `run_batch(first_seed, games)` | Create successive games with wrapping `first_seed + i`, retaining the runner's seat-policy state between games |

A decision limit protects each individual drive from infinite policy/game
loops; it is counted per call or batch game, not across resumed invocations.
On that failure, already accepted Game actions are not rolled back, so a
caller may continue with a higher limit and the necessary policy state.

## Reports and measurement

A per-game `RunReport` contains a decision trace for replay. A batch report
aggregates scores, winners, finish reasons, decisions, and timings without
retaining a trace for every game. `policy_time` measures policy choice;
`rules_time` measures Game driving and submission; batch wall time also
includes other work, so those measurements need not add up to `total_time`.
Performance observations must be compared with a recorded workload and
environment, not asserted from a single invocation.

The main CLI exposes first-legal and random legal self-play. Heuristic
comparisons and benchmarks live in the distinct
[experiment subsystem](experiments.md), not in a larger Runner that knows
about their research protocols.

## Continuation and seeds

Game seeding controls game state and wall order. Random policies have
independent per-seat streams; the library can seed them separately from Game.
`Game::clone` therefore does not by itself restore a stateful policy run.
Clone the policies with their RNG positions too; rebuilding
`Runner::random(seed)` starts fresh policy streams. Deterministic heuristic
and first-legal policies need no logical policy checkpoint, though shared
heuristic caches may be reused for efficiency.

## Verification boundary

Use [execution integration tests](../../tests/execution.rs) for runner and
CLI behavior, seeded replay and continuation, and
[decision contract tests](../../tests/decision_contract.rs) for the
Game-facing boundary. The [CI workflow](../../.github/workflows/rust-ci.yml)
runs tests for all targets in Debug and Release, but does not certify
cross-version replay or statistical performance.
