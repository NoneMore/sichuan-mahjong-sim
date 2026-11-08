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
| `RandomPolicy` | Seeded legal-choice baseline with its own RNG |
| `MissingOnlyPolicy` | Controlled ablation of the missing-suit declaration; delegates remaining choices to the first-legal baseline |
| `HeuristicPolicy` | Deterministic action ranking from exact hand analysis and conservative claim gates |

Policy implementations must use the offered legal actions and the information
included in `Decision`. They never read another player's concealed tiles,
an unrevealed simultaneous choice, or the hidden wall. They do not adjust
legality, scoring, or turn timing: those remain Game responsibilities.

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

The standard route combines per-suit target tables and one pair. The
seven-pairs route is available only without declared melds. The evaluator
takes the best exact target deficit and unions masks for tied optimal routes.
The Surplus mask avoids re-solving every candidate removal; the Needed mask
identifies draws improving a post-discard hand and feeds ukeire.

This separation of **action selection, exact analysis and game-rule
enforcement** is an implemented design result. Strictly over-bound DP
states may be pruned, but equal-cost paths and optimal-mask bits must survive.
Algorithm recurrences and cache key encodings stay with the source and
independent test oracles rather than being copied here.

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

Caching may change memory use and timing, **never** the action chosen, the
replay trace, or continuation. `HeuristicPolicy` has no logical RNG/checkpoint
state; its cache is disposable. `RandomPolicy` owns an RNG stream, so clone
or otherwise preserve its policy state for exact continuation. The
[Runner guide](execution.md#continuation-and-seeds) covers the cross-subsystem
checkpoint rule.

## Verification boundary

Use the [policy integration tests](../../tests/heuristic_policy.rs) for
legality and deterministic continuation, the
[missing-only ablation tests](../../tests/missing_only_policy.rs) for its
controlled difference, [analysis oracles](../../src/policy/analysis/one_pass_tests.rs)
for exact distances and masks, and
[cache tests](../../src/policy/analysis/cache_tests.rs) for bounded,
failure, and context-order behavior. Comparing only game outcomes does not
prove exact analysis or action equivalence.
