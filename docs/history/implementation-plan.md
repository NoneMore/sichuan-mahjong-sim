# Implementation slices

The architecture baseline is [the design](xzd-rust-simulator-design-revised.md). This file records execution boundaries and completion checks, not additional product scope.

## 1. Rules and pure computation

Deliver one Rust crate, checked tile/action types, exact winning-shape computation, and a documented fixed scoring baseline. Resolve the explicitly open rule details here, without a generic rule framework.

Completion: focused tests cover ordinary hands, open hands, seven pairs, invalid counts, scoring facts, and fixed rule examples. Public types are ready for Game and Policy to share.

## 2. Playable authoritative Game

Depends on slice 1. Implement the full single-game lifecycle, current-decision legality, hidden-information-safe borrowed views, staged simultaneous choices, response arbitration, cloning, and settlement.

Completion checks:

- Exchange direction is visible before choices; exchanges are applied only after all four submissions.
- Missing-suit and response submissions do not change other participants' pre-resolution observations or legality.
- Repeated decision reads preserve the pending choice; invalid submissions leave all authoritative state unchanged.
- Post-peng action does not draw; successful kong draws exactly one replacement; a robbed added kong is not committed or paid.
- Multiple discard winners are paid before lower-priority claims; inactive winners never decide again.
- Every payment is zero-sum, including kong refunds and exhaustive-draw penalties.
- Clone and continuation preserve decisions, randomness, terminal scores, and public history.
- Tile ownership is conserved across walls, concealed hands, melds, claimed discards, and winning tiles, according to the representation documented by Game.

## 3. Native execution and measurement

Depends on slice 1 and a settled Game public API; implementation can run alongside the remaining internals of slice 2. Deliver Policy, a baseline policy, Runner for both new and supplied games, a CLI, and reproducible single-thread measurements.

Completion: the runner uses only legal views, reports terminal net scores and finish reasons, and demonstrates seeded replay and continuation. Report games/sec and decisions/sec, separating policy time from rules-driving time. No performance threshold is imposed before a measured baseline exists.

## 4. Integration and independent review

Depends on completed runnable slices 2 and 3. Review actual source against the design and the fixed rule document, fix confirmed defects, and run formatting, lint, unit/integration tests, and release smoke/stress runs.

Completion: validation commands and actual outcomes are recorded; unresolved correctness issues remain explicit blockers rather than completed work. Do not introduce Python bindings, a rules DSL, parallel execution, general search, or speculative optimizations.
