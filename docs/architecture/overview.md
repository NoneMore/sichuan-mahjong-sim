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
| Authoritative game | [`game.rs`](../../src/game.rs), [`game/tests.rs`](../../src/game/tests.rs) | Decisions, stage progression, claims, settlement, and invariants |
| Policies | [`policy.rs`](../../src/policy.rs), [`policy/heuristic.rs`](../../src/policy/heuristic.rs), [`policy/missing_only.rs`](../../src/policy/missing_only.rs), [`policy/analysis.rs`](../../src/policy/analysis.rs) | `Policy` trait, legal baselines, deterministic shanten/ukeire `HeuristicPolicy`, and the `MissingOnlyPolicy` ablation that delegates to `FirstLegalPolicy` except for the shared missing-suit rule; exact target-deficit, optimal needed-mask ukeire, and one-original-hand surplus-mask discard analysis remain private |
| Execution | [`runner.rs`](../../src/runner.rs), [`main.rs`](../../src/main.rs) | Driving games, aggregating results, and the native CLI |
| Native policy comparison | [`examples/policy_ab.rs`](../../examples/policy_ab.rs) | Standalone balanced 2A2B comparison with selectable A/B policies and fresh policies and runner per game; no new game rules or library API |
| Heuristic benchmark | [`examples/heuristic_bench.rs`](../../examples/heuristic_bench.rs) | Standalone `heuristic-benchmark-v1` four-seat `HeuristicPolicy` self-play benchmark: one `Runner` and four fresh policies per phase, reused across games within a phase, with warmup seeds preceding the measured range; no new game rules or library API |

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
deficit its caller already knows: a state is skipped when its accumulated cost
plus the mandatory sequence-tail deficit exceeds the cutoff, and strictly
over-bound successors are discarded. Equal-cost paths still contribute their
full optimal-mask union. The additional state-level tail bound is a **research
candidate on this branch, not an accepted optimization**: its short/long
performance checks conflict; see the [active change results](../../specs/active/needed-suffix-bound/results.md).
Complete suit tables and cutoff-tagged bounded tables stay isolated within the same
fixed-context evaluator; a bounded table is reused only when it covers the
requested cutoff and never pollutes the complete-table cache. Ordinary distance
and Surplus-mask analysis neither read nor allocate the lazily created bounded
cache, and the full unbounded mask query remains available internally. Policy
ranking, decisions, and checkpoint semantics are unchanged.

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
  cutoff equality, mandatory-tail equality at ranks 7/8, zero-distance cases,
  complete/bounded cache isolation, and cutoff ordering.
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
  phase isolation and reuse, report fields, and failure propagation; included by
  `examples/heuristic_bench.rs` and run by `cargo test --all-targets`.
- [`tests/settlement_matrix.rs`](../../tests/settlement_matrix.rs):
  payment-category combinations.
- [`tests/state_stress.rs`](../../tests/state_stress.rs):
  seeded transitions, tile conservation, and zero-sum settlement.
- [`src/game/tests.rs`](../../src/game/tests.rs): focused state and rule cases.

Build and test commands live in the [README](../../README.md).
