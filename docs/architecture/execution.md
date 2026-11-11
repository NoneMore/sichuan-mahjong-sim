# Execution subsystem

[`Runner`](../../src/runner.rs) drives exactly four policies against a
[`Game`](game.md), but it does not implement Mahjong rules. The
[`main.rs` CLI](../../src/main.rs) is a thin batch-execution entry point.
For commands and public API examples, use [development](../development.md)
and the [library guide](../library.md).

## Component composition

Runner owns four seat-policy objects and a decision limit; Game separately
owns hand state. This separation lets research tools decide when to construct
fresh runners or retain policy state without giving Runner rule authority.
The [CLI](../../src/main.rs) provides basic batch self-play.

## Run boundary

At each step, Runner obtains the next pending `Decision` from Game, calls the
acting seat's `Policy::choose`, and submits that action to Game. An invalid
choice is returned as an error: Runner does not silently replace it with a
different move. It neither inspects concealed opponent information nor
reconstructs legality.

One drive loop supports new games, existing checkpoints and batches.
A batch creates fresh Games while retaining the Runner's policies, including
their RNG positions. The decision limit applies independently to each drive;
already accepted actions remain after a limit error. The
[library Runner API](../library.md#runner-api) documents the public methods
and limit controls.

## Reports and measurement

Single-game runs keep a replay trace; batches aggregate results without
retaining one trace per game. Timing separates policy choice from Game work,
but neither alone accounts for the full batch duration. Report fields and
usage examples are in the [library guide](../library.md#runner-api).

The main CLI exposes first-legal and random legal self-play. Heuristic
comparisons and benchmarks live in the distinct
[experiment subsystem](experiments.md), not in a larger Runner that knows
about their research protocols.

## Continuation and seeds

Game and policy RNG state are separately seeded and owned. Cloning Game
alone cannot restore a stateful policy run: the policies' RNG positions must
also be preserved. Stateless policies need no separate logical checkpoint;
heuristic caches are disposable. See the
[library checkpoint examples](../library.md#continuation-and-checkpoints).

## Verification boundary

Use [execution integration tests](../../tests/execution.rs) for runner and
CLI behavior, seeded replay and continuation, and
[decision contract tests](../../tests/decision_contract.rs) for the
Game-facing boundary. The [CI workflow](../../.github/workflows/rust-ci.yml)
runs tests for all targets in Debug and Release, but does not certify
cross-version replay or statistical performance.
