# Rule foundation

The [fixed rules](../baseline-rules.md) define intended gameplay. Foundation
types and pure computations implement those facts for [Game](game.md).

[Tile and Hand](../../src/tile.rs) represent 27 tile kinds with four
physical copies each. [Seat](../../src/seat.rs) and
[Action](../../src/action.rs) are shared identities and move types;
[hand.rs](../../src/hand.rs) re-exports Hand.
[eval.rs](../../src/eval.rs) checks winning shapes and waits;
[rules.rs](../../src/rules.rs) computes eligibility, fan and payments.

## Authority boundary

Shape and scoring functions operate on supplied values. Game alone decides
when a result applies, advances turns, commits transfers and manages tile
ownership. [Policy analysis](policy.md#analysis-structure) solves a separate
problem: target deficits and improving tiles for selecting actions.

The ruleset is fixed. Introducing a game variant requires a revised
[rules contract](../baseline-rules.md), corresponding pure computations and
Game transitions.

## Verification

[Foundation tests](../../src/rules.rs) check pure facts;
[settlement tests](../../tests/settlement_matrix.rs) check how Game applies
those facts.
