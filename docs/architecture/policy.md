# Policy subsystem

[Policy](../../src/policy.rs) chooses an action from one player's
[Decision](../../src/game.rs). Game creates legal choices and owns
progression, hidden state and settlement.

| Strategy | Use |
| --- | --- |
| FirstLegalPolicy | Deterministic legal-action baseline |
| RandomPolicy | Seeded legal-choice baseline, taking Hu when offered |
| MissingOnlyPolicy | Missing-suit ablation with first-legal actions elsewhere |
| HeuristicPolicy | Deterministic exchange, discard and claim ranking |

## Analysis structure

[Heuristic strategy](../../src/policy/heuristic.rs) ranks offered actions.
Private [exact analysis](../../src/policy/analysis.rs) computes standard
and seven-pairs target distance, Needed/ukeire and Surplus masks; pure
winning-shape checks remain in the [rule foundation](foundation.md).
Needed masks include *all* optimal improving tiles, and Surplus masks
avoid separate exact solves for each discard. Bounded DP pruning must
preserve equal-cost paths and their mask unions.

## Cache and checkpoint boundary

[HeuristicCache](../../src/policy/cache.rs) is a caller-owned, byte-bounded
memoization of derived suit tables. Sharing across seats or decisions is
explicit; allocation failure or eviction falls back to exact recomputation.
Truncated bounded results cannot answer wider or unbounded queries.
Cache contents affect cost, not actions or replay; HeuristicPolicy needs
no logical checkpoint. RandomPolicy does: its RNG state must accompany
the cloned Game.

[Policy tests](../../tests/heuristic_policy.rs) ·
[Exact-analysis oracles](../../src/policy/analysis/one_pass_tests.rs) ·
[Cache tests](../../src/policy/analysis/cache_tests.rs)
