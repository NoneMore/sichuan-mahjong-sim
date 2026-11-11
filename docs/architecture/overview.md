# Current architecture

This is the index of the **implemented subsystem architecture**. The
individual guides summarize current high-level structure, responsibilities,
interactions, important implemented design outcomes, and code entry points.
They do not override source evidence or duplicate detailed algorithms.
The [fixed gameplay rules](../baseline-rules.md) independently define
intended Mahjong semantics.

## Subsystems

| Subsystem | Architecture guide | Role |
| --- | --- | --- |
| Rule foundation | [Foundation](foundation.md) | Tile/hand model and pure shape, legality and settlement facts |
| Authoritative game | [Game](game.md) | Sole mutable hand state, decisions, progression and payments |
| Policies | [Policy](policy.md) | Player-visible action selection, exact analysis and optional memoization |
| Neural adapters | [Neural](neural.md) | Rust encoding, teacher collection and NumPy preparation; Python datasets, offline training and policy checkpoints; PyO3 environment binding, online scheduling and frozen comparisons |
| Execution | [Runner and CLI](execution.md) | Drives games and policies without interpreting rules |
| Research tools | [Experiments](experiments.md) | Repeatable policy comparisons, timings and evidence records |

The gameplay rules specify *intent* for the foundation and Game; the
foundation architecture describes *how the rule facts are implemented*.
Keep these two distinct when checking a rules disagreement.

## Crate boundary

The workspace keeps `sichuan-mahjong-sim` at the repository root and
`sichuan-mahjong-neural` under `crates/neural/`. The neural crate depends on
the core’s public types; the core does not depend on neural code.

## Interactions and architectural outcomes

```text
shared tile / hand / action types
           |
      eval + rules (pure facts)
           |
           v
       Game state ---- Decision (one player's view) ---> Policy
           ^                                            |
           +--------------------- Action ---------------+
           ^
           | Runner drives the loop and records outcomes
           |
       Experiment tools run comparisons and benchmarks through Runner
```

This is logical dataflow, not a compile-time dependency graph. `Game`
commits transitions; policies select from player-scoped decisions and
Runner drives the loop. Neural teacher collection drives a separate public
Game/Policy loop to record accepted decisions without changing Runner. Offline
preparation captures that corpus and publishes a self-contained NumPy cache;
neither collection nor preparation changes Runner or the simulator core. The
Python loader consumes only the prepared-cache protocol, validates its arrays,
and builds padded batches without Rust bindings. Python trains a shared candidate
scorer, selects checkpoints using validation NLL and evaluates frozen policies
on held-out data. The online package adds a separate PyO3 workspace that exposes
owned decision snapshots and canonical-action submission, plus synchronous
multi-game scheduling and a comparison CLI that records plans, streamed terminal
records, costs and recomputed summaries. None of these paths changes the
authoritative `Game` loop or the offline build. The subsystem guides explain
ownership and cache constraints in detail.

These are current **implemented outcomes**, not prospective designs.
Use [development](../development.md) for setup, CI and operational commands,
and [validation](../validation.md) for historical performance evidence.
Past Changes or ADRs may explain decisions but are not prerequisites for
understanding the current system.
