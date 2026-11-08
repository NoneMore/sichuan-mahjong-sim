# Game subsystem

The [authoritative `Game`](../../src/game.rs) owns one hand's state,
legal decisions, progression, and settlement. The [fixed rules](../baseline-rules.md)
say what play should mean; this page describes the implemented **state and
information boundaries**. Game is not a policy or a match-level controller.

## Ownership

- [`game.rs`](../../src/game.rs) owns the wall and RNG, concealed hands, melds,
  public discard history, missing-suit declarations, pass locks, pending
  decisions, scores, winners, and staged choices.
- [`game/settlement.rs`](../../src/game/settlement.rs) implements payments,
  winner bookkeeping, and wall-exhaustion settlement on that **same** `Game`
  instance. It does not own an independent ledger or state machine.
- [`game/diagnostics.rs`](../../src/game/diagnostics.rs) exposes diagnostic
  state and checks; these methods must not become policy inputs.
- Pure [rule facts](../../src/rules.rs) and [hand-shape evaluation](../../src/eval.rs)
  answer queries; they do not decide when a transition occurs.

## Decision and transition boundary

`Game::next_decision()` advances automatic work until a player must choose.
`Game::submit(seat, action)` validates the choice against the pending legal
decision before committing it. Repeated pending reads are idempotent;
rejection leaves the authoritative state unchanged.

Exchange selections, missing-suit declarations, and eligible discard or
added-kong responses are **collected before common resolution**. One seat's
early choice cannot immediately change another seat's options or reveal
uncommitted private choices. Response arbitration and turn progression stay
inside Game, including wins by multiple seats on the same discard.

The borrowed `Decision` is the intended boundary to Policy: it contains the
acting player's concealed hand, public information, and legal actions, not the
other players' concealed hands or hidden wall sequence. A privileged
`Game` reference must not cross this boundary.

## Physical state and settlement

Game tracks each of the 108 physical tiles **once**: in the live wall,
concealed hands, melds, or an unclaimed/claimed-by-Hu discard record. A
discard used for a peng or open kong becomes part of that meld, not a second
owned tile. Multiple Hu winners can use the same discard **virtually for hand
evaluation**, but they do not each own an additional physical tile.

Settlement changes scores through balanced transfers; accepted transitions
must conserve tiles and total score. Terminal wall-exhaustion evaluation and
kong-income refunds use facts from the rule subsystem while Game retains
authority over the timing and payment ledger. `Game::debug_validate` also
checks winner/discard metadata consistency; these are representation
invariants, not additional game rules.

## Replay and verification

A cloned `Game` includes its own RNG, wall position, and pending decision.
It does **not** include independently owned stateful policies; see
[execution checkpoints](execution.md#continuation-and-seeds).
Do not move a transition into Runner or Policy as a way to simplify Game.

Start regression work in the [game scenario modules](../../src/game/tests.rs),
[public decision contracts](../../tests/decision_contract.rs),
[settlement matrix](../../tests/settlement_matrix.rs), and
[seeded invariant stress tests](../../tests/state_stress.rs).
Rule discrepancies should be checked against the
[intended rules](../baseline-rules.md), not explained away by this document.
