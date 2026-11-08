# Current architecture

This page maps implemented component responsibilities, information boundaries,
and code locations. The [baseline rules](../baseline-rules.md) define intended
gameplay.

## Responsibilities and code navigation

The crate has one authoritative `Game` for legality, state transitions, and
settlement. `Policy` implementations choose from player-visible decisions;
`Runner` invokes them and drives games without implementing mahjong rules.
Pure `eval` and `rules` helpers supply facts, not competing state owners.

| Area | Code | Responsibility |
| --- | --- | --- |
| Types and actions | [`tile.rs`](../../src/tile.rs), [`hand.rs`](../../src/hand.rs), [`seat.rs`](../../src/seat.rs), [`action.rs`](../../src/action.rs) | Tiles, counted hands, seats, actions, and melds |
| Hand evaluation | [`eval.rs`](../../src/eval.rs) | Winning shapes and waiting tiles |
| Fixed-rule helpers | [`rules.rs`](../../src/rules.rs) | Legality and settlement facts used by `Game` |
| Authoritative game | [`game.rs`](../../src/game.rs), [`game/settlement.rs`](../../src/game/settlement.rs), [`game/tests.rs`](../../src/game/tests.rs) | Decisions, stage progression, claims, settlement, and invariants |
| Policies | [`policy.rs`](../../src/policy.rs), [`policy/heuristic.rs`](../../src/policy/heuristic.rs), [`policy/missing_only.rs`](../../src/policy/missing_only.rs), [`policy/analysis.rs`](../../src/policy/analysis.rs), [`policy/cache.rs`](../../src/policy/cache.rs) | `Policy` trait, legal baselines, deterministic shanten/ukeire `HeuristicPolicy`, and the `MissingOnlyPolicy` ablation that delegates to `FirstLegalPolicy` except for the shared missing-suit rule; private exact hand analysis with an explicit bounded masked-suit cache context |
| Execution | [`runner.rs`](../../src/runner.rs), [`main.rs`](../../src/main.rs) | Driving games, aggregating results, and the native CLI |
| Native policy comparison | [`examples/policy_ab.rs`](../../examples/policy_ab.rs) | Standalone balanced 2A2B comparison with selectable A/B policies, fresh policies and runner per game, and one explicit cache context shared by heuristic seats across the invocation; no new game rules |
| Heuristic benchmark | [`examples/heuristic_bench.rs`](../../examples/heuristic_bench.rs) | Standalone `heuristic-benchmark-v2` four-seat self-play benchmark: explicit shared cache context, default cold measured batch or warmup-inheriting warm batch, status snapshots and initialization boundaries; warmup seeds precede the measured range; no new game rules |

Public modules and exports are declared in [`src/lib.rs`](../../src/lib.rs).

## Decision and information boundaries

`Runner` requests `Game::next_decision()`, passes the borrowed `Decision` to
the acting seat's `Policy::choose`, and submits the returned action through
`Game::submit`. `Game` advances automatic transitions and owns all legality
and scoring; callers can also use its decision/submit interface directly.
See [README usage and continuation examples](../../README.md).

Exchange, missing-suit declarations, and eligible discard/added-kong responses
are collected before common resolution. Pending decision reads are idempotent;
rejected submissions leave authoritative state unchanged. See
[decision contract tests](../../tests/decision_contract.rs).

`Game` owns the ordered wall, concealed hands, staged private choices,
discards, melds, scores, and payment records. `Decision` exposes the acting
player's concealed hand, public observations, and current legal actions,
not another concealed hand, wall order, or unrevealed simultaneous choices.
Policies do not receive `Game` or its privileged diagnostics.

Physical tiles are counted once, including a discard claimed by multiple
winners: the single discard record owns that tile, while winning-shape
evaluation may use a virtual copy. `PlayerView::concealed_count` reports only
physical concealed tiles: a discard or robbing winner has `13 - 3 * melds`
tiles rather than the `14 - 3 * melds` of a self-draw winner. The winning tile
is recovered from the referenced discard record for scoring.
`Game::debug_validate` checks ownership and score invariants; it is not a
policy-facing API. See [the game implementation](../../src/game.rs).

## Reproducibility boundary

Heuristic ukeire bounds only the private Needed-mask suit DP with the exact
deficit its caller already knows: strictly over-bound successors are discarded,
while equal-cost paths still contribute their full optimal-mask union. Complete
suit tables and cutoff-tagged bounded tables stay isolated within the same
fixed-context evaluator; a bounded table is reused only when it covers the
requested cutoff and never pollutes the complete-table cache. Ordinary distance
and Surplus-mask analysis neither read nor allocate the lazily created bounded
cache, and the full unbounded mask query remains available internally. Policy
ranking, decisions, and checkpoint semantics are unchanged.

Heuristic masked-DP analysis retains decision-local tables as its first lookup
layer. An explicit `HeuristicCache` adds byte-bounded, cross-decision memoization
of exact nine-kind suit tables only; ordinary distance analysis keeps its
original path. Keys own the hand, capacity, target-set count, mask direction,
and exact bound (or unbounded tag), so shared history cannot affect results.
Cached values retain every equal-cost optimal mask bit. Four-way groups use
empty slots first and then group-local round-robin replacement; hits do not
change replacement order.

Policies own shared cache handles, not `Game`, the `Policy` trait, or `Runner`.
There is no automatic global/TLS sharing. Context clones share one lazy table,
budget and allocation-failure state, and the last holder releases it. The
public status snapshot reports configured budget, table allocation, capacity,
and fixed context overhead. A zero budget disables the shared table; failed
allocation permanently falls back to the original exact computation. CLI tools
own warnings, while the library remains silent.

The 26.5 MiB default was selected over 53 MiB in paired batch measurements;
short cold runs still pay initialization overhead, and the retained layout is
not a claim of global optimality among other cache designs.

Cloning `Game` preserves the authoritative position and RNG, **not** the
state held by policies. To continue exactly with stateful policies such as
`RandomPolicy`, checkpoint their state as well. `HeuristicPolicy` has no logical
history or RNG: its private analysis memoization may be discarded when resuming,
so fresh instances reproduce continuation from a cloned game. The
[README checkpoint examples](../../README.md) cover construction and replay.

## Test navigation

- [`tests/decision_contract.rs`](../../tests/decision_contract.rs): pending
  decisions, rejection atomicity, and cloned-game replay.
- [`tests/execution.rs`](../../tests/execution.rs): runner and CLI behavior,
  seeded replay, checkpoint continuation, and score balance.
- [`src/policy/cache/tests.rs`](../../src/policy/cache/tests.rs) and
  [`src/policy/analysis/cache_tests.rs`](../../src/policy/analysis/cache_tests.rs):
  byte accounting, lazy/shared/failure lifecycle, exact keys and replacement,
  complete cached/uncached tables, independent target oracles, bound ordering,
  local-first reuse, collisions and eviction.
- [`tests/heuristic_policy.rs`](../../tests/heuristic_policy.rs): deterministic
  heuristic legality, seeded replay, and continuation with fresh policies.
- [`tests/missing_only_policy.rs`](../../tests/missing_only_policy.rs): every
  decision of the missing-suit-only ablation against its reference policy
  (`HeuristicPolicy` for the declaration, `FirstLegalPolicy` otherwise),
  legality, seeded replay, and continuation with fresh policies.
- [`src/policy/analysis/discard_surplus_tests.rs`](../../src/policy/analysis/discard_surplus_tests.rs)
  and [`src/policy/heuristic/tests/one_pass_discard.rs`](../../src/policy/heuristic/tests/one_pass_discard.rs):
  exact per-candidate removed-tile deficit references, optimal surplus unions,
  standard/seven-pairs ties, physical sampled states, lazy/context-isolated
  caches, candidate order, ukeire/fan ranking, and peng/kong decision regressions.
- [`src/policy/analysis/one_pass_tests.rs`](../../src/policy/analysis/one_pass_tests.rs)
  and its submodules: optimal-target masks, frozen ukeire/distance reference,
  concrete target oracles, full suit tables, ownership, ties, and cache ordering.
  [`bounded_needed.rs`](../../src/policy/analysis/one_pass_tests/bounded_needed.rs)
  checks bounded tables and global masks against those oracles, including strict
  cutoff equality, zero-distance cases, complete/bounded cache isolation, and
  cutoff ordering.
- [`src/policy/analysis/tests.rs`](../../src/policy/analysis/tests.rs),
  [`src/policy/analysis/optimization_b_tests.rs`](../../src/policy/analysis/optimization_b_tests.rs),
  [`src/policy/analysis/optimization_a2_tests.rs`](../../src/policy/analysis/optimization_a2_tests.rs),
  [`src/policy/heuristic/tests.rs`](../../src/policy/heuristic/tests.rs), and
  [`src/policy/heuristic/tests/calls.rs`](../../src/policy/heuristic/tests/calls.rs):
  independent target oracle, all 241 seven-pairs count classes and formula
  boundaries, suit-table equivalence, draw/discard normalization, public
  ownership, discard ranking, and peng/kong gates.
- [`src/policy/missing_only/tests.rs`](../../src/policy/missing_only/tests.rs):
  focused missing-suit, fallback, exchange, turn, and `Hu` cases for the
  ablation.
- [`examples/policy_ab/tests.rs`](../../examples/policy_ab/tests.rs): A/B example
  tests for seed-block aggregation, balanced placements, replay, and failure handling.
- [`examples/heuristic_bench/tests.rs`](../../examples/heuristic_bench/tests.rs):
  benchmark example tests for argument validation, seed planning and wrapping,
  phase isolation/reuse, cold/warm context inheritance, cache configuration,
  report fields, warning deduplication, and failure propagation; included by
  `examples/heuristic_bench.rs` and run by `cargo test --all-targets`.
- [`tests/settlement_matrix.rs`](../../tests/settlement_matrix.rs):
  payment-category combinations.
- [`tests/state_stress.rs`](../../tests/state_stress.rs):
  seeded transitions, tile conservation, and zero-sum settlement.
- [`src/game/tests.rs`](../../src/game/tests.rs): focused state and rule cases.

Build and test commands live in the [README](../../README.md).
