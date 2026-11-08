# Rule foundation: implementation architecture

The [fixed gameplay rules](../baseline-rules.md) define **intended** gameplay.
This page describes the *implemented* tile model and pure rule computation,
which [Game](game.md) uses to enact those rules. Neither document replaces
evidence from code and tests when checking actual behavior.

## Components and dependencies

| Component | Responsibility |
| --- | --- |
| [`tile.rs`](../../src/tile.rs) | `Tile`, `Suit`, the fixed 27 kinds/four copies each, and the counted `Hand` implementation |
| [`hand.rs`](../../src/hand.rs) | Stable re-export of `Hand` and `HandError` from `tile.rs`; not a separate hand implementation |
| [`seat.rs`](../../src/seat.rs) | Four seat identities and relative table positions |
| [`action.rs`](../../src/action.rs) | Shared action, exchange direction and meld representations |
| [`eval.rs`](../../src/eval.rs) | Pure winning shapes, waits and tenpai queries |
| [`rules.rs`](../../src/rules.rs) | Pure legality facts, fan breakdowns, payments and wall-exhaustion transfers |

Shared types are used by both pure helpers and `Game`. See the
[architecture map](overview.md#interactions-and-architectural-outcomes) for
the cross-subsystem flow. Public exports are in [`lib.rs`](../../src/lib.rs).

## Pure facts versus stateful authority

`eval` computes winning shapes and waits for supplied hands and melds,
respecting the owner's physical-copy limits. It has no turn state or payment
authority.

`rules` supplies eligibility, fan and payment facts, but cannot commit a win
or a transfer. `Game` determines when those facts apply, which seats are
active, and what state changes. The foundation has no separate turn scheduler
or mutable Game state.

The [Policy evaluator](policy.md#analysis-structure) computes target deficits
and improving-tile counts for strategy. It is deliberately separate from
authoritative legal winning-shape and settlement checks.

## Implemented architectural outcomes

- Shape evaluation, pure scoring facts and state transitions have distinct
  owners. A pure result alone never advances a hand.
- The crate implements one fixed ruleset, not a pluggable runtime rule
  configuration. A new variant needs an explicit rule-contract and design
  decision rather than a Policy/Runner override.

## Verification

Use the unit tests in [`tile.rs`](../../src/tile.rs),
[`eval.rs`](../../src/eval.rs), and [`rules.rs`](../../src/rules.rs)
for representation, shape, legality, fan and payment facts. The
[Game settlement tests](../../src/game/tests/settlement.rs) and
[payment matrix](../../tests/settlement_matrix.rs) check that the stateful
engine applies those facts correctly.
