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

```text
Tile / Hand / Seat / Action
          |
       eval.rs (winning shapes and waits)
          |
       rules.rs (legality and settlement facts)
          |
       Game (authoritative decision and payment timing)
          |
       Decision -> Policy -> Action -> Game
```

The diagram summarizes the logical layering rather than every module import:
Game and policies also consume the shared types directly. Public exports
are declared in [`lib.rs`](../../src/lib.rs).

## Pure facts versus stateful authority

`eval` examines a supplied hand and declared melds; it does not know whose
turn it is, see other players' concealed tiles, or make a payment. Standard
sets and seven-pairs are shape alternatives, and a kong occupies one declared
set for counting the concealed part. Waiting-tile checks respect physical
copies in the player's hand and own melds, not tiles hidden elsewhere.

`rules` consumes shape facts to answer rule-level eligibility and scoring
questions. For example, `win_fan` computes a fan breakdown and
`win_payment` computes payment facts, but neither can declare a win in a
running game. Game checks the current stage and legal action, determines
active payers, and commits transfers and kong refund records. The rule
foundation has **no separate mutable Game state or turn scheduler**.

The [Policy evaluator](policy.md#analysis-structure) computes target deficits
and improving-tile counts for strategy. It is deliberately separate from
authoritative legal winning-shape and settlement checks.

## Implemented architectural outcomes

- Fixed 108-tile, counted-hand representations keep the foundation compact;
  the stateful Game still accounts for each physical copy exactly once.
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
