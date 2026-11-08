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
| Execution | [Runner and CLI](execution.md) | Drives games and policies without interpreting rules |
| Research tools | [Experiments](experiments.md) | Repeatable policy comparisons, timings and evidence records |

The gameplay rules specify *intent* for the foundation and Game; the
foundation architecture describes *how the rule facts are implemented*.
Keep these two distinct when checking a rules disagreement.

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

This depicts architectural roles, not every compile-time dependency.
**Game is the only authority** for a hand's transitions and settlement.
A Policy sees a legal, player-scoped Decision rather than hidden game
state; Runner and the research tools never add new Mahjong rules. Shared
heuristic caches are derived memoization, never logical game or policy
state. Cloning Game does not checkpoint an independent random-policy RNG.

These are current **implemented outcomes**, not prospective designs.
Use [development](../development.md) for setup, CI and operational commands,
and [validation](../validation.md) for historical performance evidence.
Past Changes or ADRs may explain decisions but are not prerequisites for
understanding the current system.
