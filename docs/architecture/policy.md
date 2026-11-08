# Policy subsystem

Policies choose an action from a player-visible
[`Decision`](../../src/game.rs); they do not own game progression or legal
action generation. The [`Policy` trait](../../src/policy.rs) is a deliberately
narrow seam between the authoritative [Game](game.md) and the
[Runner](execution.md).

## Strategy responsibilities

The current crate provides four policy roles:

| Role | Responsibility |
| --- | --- |
| `FirstLegalPolicy` | Reproducible legal-action baseline |
| `RandomPolicy` | Seeded random legal-choice baseline that takes Hu whenever offered |
| `MissingOnlyPolicy` | Controlled ablation of the missing-suit declaration; delegates remaining choices to the first-legal baseline |
| `HeuristicPolicy` | Deterministic action ranking from exact hand analysis and conservative claim gates |

Policies use only the offered `Decision`; legality, scoring, turn timing
and privileged hidden state remain Game responsibilities.

## Analysis structure

[`policy/heuristic.rs`](../../src/policy/heuristic.rs) owns the **strategy**:
missing-suit and exchange choices, discard ranking, and peng/kong decisions.
Private [`analysis.rs`](../../src/policy/analysis.rs) supplies exact
target deficit, Needed/ukeire and Surplus data, not legal-action authority.
The [rule foundation](foundation.md) remains responsible for winning-shape
and scoring facts used by Game.

```text
Decision and legal actions
       |
heuristic.rs (choose and compare actions)
       |
analysis.rs (exact distance + improving/surplus masks)
       |
       +-- standard.rs -> suit_dp.rs (counted per-suit targets)
       +-- seven_pairs.rs (seven-pairs formula / DP)
       |
       +-- decision-local caches -> optional shared HeuristicCache
```

The evaluator compares exact standard and seven-pairs targets (the latter
only without melds). Needed masks identify improving draws for ukeire;
Surplus masks avoid re-solving each possible discard. Bounded pruning must
retain equal-cost optimal paths and their mask bits.

The separation between strategy selection, exact analysis and authoritative
rules is the architectural boundary. Recurrences and cache-key encodings stay
near the source and independent oracles rather than being duplicated here.

## Cache and checkpoint boundary

[`HeuristicCache`](../../src/policy/cache.rs) is an optional, explicitly
shared, byte-bounded cache of derived masked suit tables across decisions or
seats. It is not a game-state cache. Decision-local analysis continues to
work when cross-decision caching is disabled, unallocated, or fails to
allocate. There is no implicit global or thread-local context; caller-owned
clones can share a context deliberately.

Complete and bounded queries have distinct cache reuse conditions:
a truncated bounded suit result must not answer a wider or unbounded query.
The evaluator uses decision-local exact caches first, optionally reusing
masked per-suit results across decisions via the shared context. Allocation
failure falls back to exact analysis, not a new strategy.

Caching may change memory use and timing, never actions or replay.
`HeuristicPolicy` has no logical checkpoint state; `RandomPolicy` owns
an RNG stream. See [execution](execution.md#continuation-and-seeds)
for the cross-subsystem continuation contract.

## Verification boundary

Use the [policy integration tests](../../tests/heuristic_policy.rs) for
legality and deterministic continuation, the
[missing-only ablation tests](../../tests/missing_only_policy.rs) for its
controlled difference, [analysis oracles](../../src/policy/analysis/one_pass_tests.rs)
for exact distances and masks, and
[cache tests](../../src/policy/analysis/cache_tests.rs) for bounded,
failure, and context-order behavior. Comparing only game outcomes does not
prove exact analysis or action equivalence.
