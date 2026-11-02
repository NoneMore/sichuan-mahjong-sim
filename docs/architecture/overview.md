# Current architecture

This document maps the architecture **implemented in the repository**, rather
than the initial proposal or a future research roadmap. Check the linked Rust
source for implementation details. The [project governance](../governance.md)
explains the evidence model; [baseline rules](../baseline-rules.md) describe
the fixed **intended** mahjong semantics.

## System shape

- A single native Rust crate, with a library and a command-line executable
  ([`Cargo.toml`](../../Cargo.toml), [`src/lib.rs`](../../src/lib.rs),
  [`src/main.rs`](../../src/main.rs)). The crate currently has no third-party
  dependencies and forbids unsafe Rust.
- One authoritative, mutable `Game` instance models a single four-seat,
  108-tile hand. A `Runner` drives individual or repeated games through
  in-process `Policy` implementations. The existing batch runner runs
  sequentially; there is no separate parallel engine.
- The library supplies fixed rules and reproducible legal baseline policies;
  it is not a configurable Sichuan-rules platform or a strategy/training
  framework.

## Code navigation

| Area | Where to look | Implemented responsibility |
| --- | --- | --- |
| Public crate surface | [`src/lib.rs`](../../src/lib.rs) | Module declarations, exports, and fixed-deck checks |
| Tile and hand types | [`tile.rs`](../../src/tile.rs), [`hand.rs`](../../src/hand.rs), [`seat.rs`](../../src/seat.rs), [`action.rs`](../../src/action.rs) | Checked values, concealed tile counts, seats, semantic actions, and melds |
| Pure hand evaluation | [`eval.rs`](../../src/eval.rs) | Standard/seven-pairs winning shapes and waiting-tile queries |
| Pure fixed-rule facts | [`rules.rs`](../../src/rules.rs) | Eligibility helpers, fan calculations, payment and wall-exhaustion computations |
| Authoritative game | [`game.rs`](../../src/game.rs), [`game/tests.rs`](../../src/game/tests.rs) | State machine, legal decisions, claim resolution, scoring, public observations, and diagnostics |
| Legal policy choices | [`policy.rs`](../../src/policy.rs) | `Policy::choose`, `FirstLegalPolicy`, and independent seeded `RandomPolicy` |
| Execution and reporting | [`runner.rs`](../../src/runner.rs) | Policy invocation, new/existing game driving, batch aggregates, timing, and failure reporting |
| Native executable | [`main.rs`](../../src/main.rs) | Reproducible self-play command-line entry point |

The `eval` and `rules` helpers supply facts to the game without taking over
authoritative state or decision timing. `Policy` is a consumer of restricted
decisions, not a rules authority. `Runner` orchestrates calls but does not
independently implement mahjong legality or settlement.

## Execution and decision boundary

The normal execution loop in [`Runner`](../../src/runner.rs) is:

1. Ask `Game::next_decision()` for the next player choice. The game advances
   automatic draws, turns, and settlement until a decision or terminal result.
2. Pass the borrowed `Decision` to the acting seat's `Policy::choose`.
3. Submit the returned semantic `Action` with `Game::submit(seat, action)`.
   An illegal policy action is reported as an error, not silently replaced.
4. Repeat until `Game::result()` is available. The runner reports terminal
   seat scores, winners, end reason, decisions, and timing; batch reports
   aggregate without retaining each game's decision trace.

Callers may also use the public `Game` decision/submit interface without
`Runner`. The [README](../../README.md) documents supported use and
continuation examples.

The internal [`Game` state machine](../../src/game.rs) distinguishes exchange,
missing-suit selection, player turns, responses to discards or added kongs,
and the terminal state. Exchange selections, missing-suit selections, and
applicable responses are gathered before the common resolution. While an
action is pending, repeated `next_decision` reads preserve that decision.
The game reuses current-decision legality when validating `submit`;
rejected submissions do not change authoritative state. See the
[public decision contract tests](../../tests/decision_contract.rs).

## State ownership and information boundaries

[`Game`](../../src/game.rs) owns the wall and draw order, concealed hands,
melds, public discard history, private staged choices, per-seat state,
randomness, score transfers, kong-payment records, and terminal outcome.
Its `Decision` carries an `Observation` and `LegalActions` for the pending
seat. `Observation` exposes that seat's concealed hand and public information
about other players; it does **not** expose another concealed hand, the ordered
wall, or unrevealed simultaneous choices. A policy must not use the privileged
`Game` or debug state as input.

Each physical tile has a single owner in the wall, a concealed hand, a meld,
or the relevant discard history representation. Multiple winners on one
discard refer to that **one** physical discard; scoring may use a virtual
winning tile without duplicating ownership. The diagnostic
`Game::debug_validate` checks conservation, related metadata, and balanced
scores. It is not a policy-facing observation API.

## Randomness, cloning, and reproducibility

- `Game::new(seed)` controls the authoritative game's shuffle and exchange
  direction. `Game::clone()` preserves the complete game position, including
  its RNG and pending decisions.
- `RandomPolicy` has a **separate** seeded RNG for each seat. A complete
  mid-game checkpoint with stateful policies must preserve both the game and
  those policy states. Building a new `Runner::random(seed)` restarts policy
  streams; it does not restore their previous positions.
- `Runner::run_batch(first_seed, games)` derives successive game seeds and
  reuses the runner's continuing policy streams across the batch. Reproduction
  depends on the same build, seed inputs, policies, and decision sequence.
- Runner metrics distinguish `Policy::choose` time from rules-driving time.
  Throughput observations depend on environment and workload; see the
  [historical validation record](../validation.md), not a performance target.

## Verification landmarks

- [`tests/decision_contract.rs`](../../tests/decision_contract.rs):
  pending-read idempotence, invalid-action atomicity, and cloned-game replay.
- [`tests/execution.rs`](../../tests/execution.rs): runner/CLI behavior,
  replay, continuation including stateful policy checkpointing, and score balance.
- [`tests/settlement_matrix.rs`](../../tests/settlement_matrix.rs):
  exhaustive small-state payment category combinations.
- [`tests/state_stress.rs`](../../tests/state_stress.rs):
  seeded transitions, tile conservation, and zero-sum settlement.
- [`src/game/tests.rs`](../../src/game/tests.rs): focused state and
  rule-interaction cases.

These locate existing verification; they do not imply exhaustive correctness.
The normal build/test commands remain in the [README](../../README.md).

## Scope boundaries and historical material

The current implementation has no independent strategy-oriented `Hand Analysis`
module or search engine; `eval.rs` provides pure hand-shape and wait queries.
Configurable local rules, training trajectories, Python bindings, GPU execution,
and parallel batch processing are not part of the implemented baseline.
Potential directions in the [initial design](../history/xzd-rust-simulator-design-revised.md)
must not be read as features that already exist.

The [historical implementation plan](../history/implementation-plan.md) and
[historical integration contract](../history/integration-contract.md) document
initial delivery and coordination, not up-to-date instructions. Consult
current code and this architecture overview for the implemented system;
use [governance](../governance.md) for when to update this document.
