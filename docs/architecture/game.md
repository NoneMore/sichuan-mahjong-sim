# Game subsystem

[Game](../../src/game.rs) owns one hand's authoritative state, legal
choices, progression, tile ownership and settlement. The
[fixed rules](../baseline-rules.md) define intended game semantics.

## Decision and authority

The private stage machine advances exchange → missing-suit declaration →
turn/response → terminal settlement. Game completes automatic draws and
checks before exposing a pending player decision. A Decision contains the
acting player's hand, public information and legal actions; it excludes
opponents' hidden tiles and the unseen wall.

`next_decision()` advances automatic work until a choice is needed;
`submit(seat, action)` validates before committing it. Repeated pending
reads are stable; rejected submissions leave state unchanged.

Exchange choices, missing-suit declarations and competing response choices
are gathered before joint resolution. Early submissions cannot alter a
later seat's offered choices. Game arbitrates priority and multiple Hu
claims independently of the calling Runner or Policy.

## Physical tiles and settlement

The 108 physical tiles occupy the wall, concealed hands, melds or
discard records without double ownership. Tiles claimed for peng/kong
move to melds; one discard may count virtually in multiple Hu hands
without creating physical copies.

[Settlement](../../src/game/settlement.rs) commits balanced score transfers
on the same Game state. Pure [rule helpers](../../src/rules.rs) provide
fan and payment facts; Game determines timing and eligible seats.
[Diagnostics](../../src/game/diagnostics.rs) validate conservation and
representation invariants and remain separate from Policy inputs.

## Replay and tests

Game cloning preserves the hand, pending decisions and wall order.
Stateful policy streams must be cloned separately; see
[checkpoint continuation](../guides/checkpoint-replay.md).

[Game tests](../../src/game/tests.rs) ·
[Decision contracts](../../tests/decision_contract.rs) ·
[Settlement matrix](../../tests/settlement_matrix.rs) ·
[Seeded stress tests](../../tests/state_stress.rs)
