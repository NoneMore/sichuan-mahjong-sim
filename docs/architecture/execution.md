# Execution subsystem

[Runner](../../src/runner.rs) drives four policies against one
[Game](game.md). It asks Game for the current Decision, requests an action
from the acting policy, and submits that action to Game. Invalid actions
return an error; Game retains all rule and transition authority.

## Run lifecycle

Runner supports fresh games, existing positions and batches. A batch
creates new Games while retaining its policies and their RNG state.
The per-drive decision limit leaves already accepted actions intact.
Single-game reports include traces; batches aggregate scores and
policy/rules timings without storing every trace.

The root [CLI](../../src/main.rs) offers first-legal and random self-play.
Heuristic comparisons use [research tools](experiments.md).

## Replay and checkpoints

Game seeds control the wall and initial state; policies own their own RNG
streams. A cloned Game can resume a stateless strategy immediately.
For RandomPolicy, resume from the cloned Game **and** each policy's
advanced RNG state. Shared heuristic caches are optional memoization and
can be rebuilt without changing actions.

[Checkpoint example](../guides/checkpoint-replay.md) ·
[Runner API](../library.md#runner-api) ·
[Execution tests](../../tests/execution.rs)
