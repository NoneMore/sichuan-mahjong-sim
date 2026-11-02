# Current architecture

This is a navigation map of the **implemented** architecture, not a parallel
API specification. Source code is evidence of current behavior; the
[governance guide](../governance.md) explains document authority, and
[baseline rules](../baseline-rules.md) define intended gameplay.

## Responsibilities and code navigation

The crate has one authoritative `Game` for legality, state transitions, and
settlement. `Policy` implementations choose from player-visible decisions;
`Runner` invokes them and drives games without implementing mahjong rules.
Pure `eval` and `rules` helpers supply facts, not competing state owners.

| Area | Code | Responsibility |
| --- | --- | --- |
| Types and actions | [`tile.rs`](../../src/tile.rs), [`hand.rs`](../../src/hand.rs), [`seat.rs`](../../src/seat.rs), [`action.rs`](../../src/action.rs) | Tiles, counted hands, seats, actions, and melds |
| Hand evaluation | [`eval.rs`](../../src/eval.rs) | Winning shapes and waiting tiles, not strategy search |
| Fixed-rule helpers | [`rules.rs`](../../src/rules.rs) | Legality and settlement facts used by `Game` |
| Authoritative game | [`game.rs`](../../src/game.rs), [`game/tests.rs`](../../src/game/tests.rs) | Decisions, stage progression, claims, settlement, and invariants |
| Policies | [`policy.rs`](../../src/policy.rs) | `Policy` trait and legal baseline choices |
| Execution | [`runner.rs`](../../src/runner.rs), [`main.rs`](../../src/main.rs) | Driving games, aggregating results, and the native CLI |
| Optional policy experiment | [`policy_ab.rs`](../../examples/policy_ab.rs), [`experiment guide`](../../experiments/policy_ab/README.md) | Fallible external-heuristic orchestration and seed-block analysis, separate from the library and native CLI |

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
evaluation may use a virtual copy. `Game::debug_validate` checks ownership
and score invariants; it is not a policy-facing API. See
[the game implementation](../../src/game.rs).

## Reproducibility boundary

Cloning `Game` preserves the authoritative position and RNG, **not** the
state held by policies. To continue exactly with stateful policies such as
`RandomPolicy`, checkpoint their state as well. The
[README checkpoint examples](../../README.md) cover construction and replay.

## Optional experiment boundary

The `policy_ab` Cargo example also uses `next_decision` / `submit` directly,
without changing `Game`, `Policy`, or `Runner`. It sends only the acting
player's hand, missing suit, and complete legal candidates to a long-lived
Python worker importing the original ninone heuristic. Stable candidate IDs
map responses back to Rust actions; `Game` remains authoritative for legality
and settlement. JSON support is a development-only dependency, and the
external Python repository is not a simulator runtime dependency.

The Python experiment tooling owns seed sampling, pilot/main separation,
reproducibility evidence, complete six-placement validation, and whole-seed
block bootstrap analysis. This is one experiment, not a general cross-language
policy interface. See the [experiment guide](../../experiments/policy_ab/README.md)
for the operational contract and verification commands.

## Test navigation

- [`tests/decision_contract.rs`](../../tests/decision_contract.rs): pending
  decisions, rejection atomicity, and cloned-game replay.
- [`tests/execution.rs`](../../tests/execution.rs): runner and CLI behavior,
  seeded replay, checkpoint continuation, and score balance.
- [`tests/settlement_matrix.rs`](../../tests/settlement_matrix.rs):
  payment-category combinations.
- [`tests/state_stress.rs`](../../tests/state_stress.rs):
  seeded transitions, tile conservation, and zero-sum settlement.
- [`src/game/tests.rs`](../../src/game/tests.rs): focused state and rule cases.

These are test entry points, not a claim of exhaustive correctness.
Build and test commands live in the [README](../../README.md).
