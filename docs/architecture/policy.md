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

## Analysis versus action selection

[`policy/heuristic.rs`](../../src/policy/heuristic.rs) owns the decision policy
(exchange/missing-suit selection, discard ranking, and peng/kong choices).
[`policy/analysis.rs`](../../src/policy/analysis.rs) computes the exact
target deficit, needed/ukeire and surplus information used by that ranking.
The private kernels are split into suit DP, standard-hand combination, and
seven-pairs analysis. They are **not** game-rule authorities and must not
mutate game state.

Exactness matters across refactors: equal-cost winning targets all contribute
to the optimal needed/surplus masks; bounded analysis may prune strictly
over-bound paths but must preserve equality. Keep independent reference/oracle
tests separate from production kernels. Implementation formulas and cache key
encoding live with the analysis source rather than being copied here.

## Cache and checkpoint boundary

[`HeuristicCache`](../../src/policy/cache.rs) is an optional, explicitly
shared, byte-bounded cache of derived masked suit tables across decisions or
seats. It is not a game-state cache. Decision-local analysis continues to
work when cross-decision caching is disabled, unallocated, or fails to
allocate. There is no implicit global or thread-local context; caller-owned
clones can share a context deliberately.

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
