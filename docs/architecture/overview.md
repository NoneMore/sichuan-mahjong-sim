# Current architecture

This is the **map of the current subsystems**, not a second specification of
the game's rules or a list of every source file. Each subsystem guide describes
its ownership, boundaries, and verification entry points. Read only the guide
relevant to the change. Implemented behavior is established by code and tests;
the [fixed rules](../baseline-rules.md) state intended gameplay.

## Subsystems

| Subsystem | Guide | Owns |
| --- | --- | --- |
| Rules and tile model | [Fixed gameplay rules](../baseline-rules.md) | Intended legality, winning shapes, and payments; pure implementation in [`eval.rs`](../../src/eval.rs) and [`rules.rs`](../../src/rules.rs), with tile/hand types |
| Authoritative game | [Game](game.md) | State, turn/response collection, transitions, settlement, and invariants |
| Policies and hand analysis | [Policy](policy.md) | Choices from player-visible decisions, exact private analysis, and optional cache contexts |
| Execution | [Runner and CLI](execution.md) | Driving policies against a game, tracing, batching, and reporting |
| Research tools | [Experiments](experiments.md) | Balanced A/B comparisons, heuristic benchmarks, reproducibility artifacts, and measured boundaries |

## System interaction

```text
Pure rules and shape evaluation ----> Game (authoritative state and legality)
                                       |
                                  Decision (acting seat's view)
                                       |
                                  Policy::choose
                                       |
                                     Action
                                       |
                                 Game::submit
                                       ^
Runner drives this loop; experiments invoke Runner and record its outcomes.
```

`Game` is the only mutable owner of the hand state. Policies cannot inspect
private game state, and neither Runner nor experiment tools implement or
override gameplay rules. Shared heuristic caching may save computation, but
cannot change chosen actions. A game checkpoint does not implicitly contain
stateful policy RNG state.

For building, testing, running, and CI, use the [development guide](../development.md).
For historical measurements, use [validation](../validation.md).
Consequential architectural decisions, when needed, belong in ADRs, not in
these descriptions of the current system.
