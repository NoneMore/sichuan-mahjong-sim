# Current architecture

This page explains the ownership boundaries that matter when changing the
simulator. It is **not** a second description of every source file, algorithm,
or test case. Checked-in code is evidence of implemented behavior; the
[fixed game rules](../baseline-rules.md) describe intended gameplay. See
[development](../development.md) for setup, tests, and executable examples.

## Ownership and control flow

`Runner` drives the following boundary for each decision:

```text
Game (authoritative state) -> Decision (player-visible data)
                           -> Policy::choose -> Action -> Game::submit
```

Only `Game` decides legality, transitions, and settlement. `Policy` chooses
from its legal actions; `Runner` schedules decisions and collects reports
without interpreting mahjong rules. External callers can also drive the
`Game::next_decision` / `Game::submit` interface directly.

| Responsibility | Current owner |
| --- | --- |
| Tiles, hands, legal shapes, and fixed scoring facts | [`tile.rs`](../../src/tile.rs), [`hand.rs`](../../src/hand.rs), [`eval.rs`](../../src/eval.rs), [`rules.rs`](../../src/rules.rs) |
| Authoritative game state, turn/response collection, and claims | [`game.rs`](../../src/game.rs) |
| Game payments and terminal settlement; invariant diagnostics | [`game/settlement.rs`](../../src/game/settlement.rs), [`game/diagnostics.rs`](../../src/game/diagnostics.rs) |
| Action selection and private exact hand analysis | [`policy.rs`](../../src/policy.rs), [`policy/analysis.rs`](../../src/policy/analysis.rs) |
| Optional bounded analysis memoization | [`policy/cache.rs`](../../src/policy/cache.rs) |
| Native game execution and command-line entry | [`runner.rs`](../../src/runner.rs), [`main.rs`](../../src/main.rs) |
| Policy A/B studies and heuristic performance measurement | [`policy_ab.rs`](../../examples/policy_ab.rs), [`heuristic_bench.rs`](../../examples/heuristic_bench.rs) |

Public module exports are defined in [`src/lib.rs`](../../src/lib.rs).
The exact hand-analysis kernels are private children of `policy/analysis.rs`;
its module comments and code explain their DP and cache contracts. Example
entry points similarly delegate parsing, execution, and reports to private
child modules.

## Boundaries that must remain intact

- **Information isolation:** A `Decision` carries the acting player's
  concealed tiles, public observations, and legal actions, never another
  player's concealed hand, the hidden wall order, or uncommitted simultaneous
  choices. Policy code does not access `Game` internals.
- **Single state owner:** Exchange, missing-suit declarations, discard
  responses, and added-kong robbery collect eligible choices before resolution.
  Re-reading a pending decision is stable; a rejected action must leave
  authoritative state unchanged.
- **Tile and score invariants:** A physical discarded or robbed-kong tile has
  one owner in the game's accounting, even when several players Hu on it.
  Scoring may consider a virtual winning tile without introducing an extra
  physical copy. Game diagnostics verify tile conservation and zero-sum scores.
  These are representation invariants, not additional mahjong rules.
- **Replay and checkpoints:** Cloning `Game` captures the game position and
  its RNG, not an independent policy's RNG or other logical state. Stateful
  policies such as `RandomPolicy` must be checkpointed separately.
  `HeuristicPolicy`'s cache is discardable memoization and must not change
  decisions or continuation.
- **Exact analysis:** The heuristic's exact DP and bounded caches are
  implementation details, not extra rule authorities. Any optimization must
  preserve the exact distances, all tied optimal masks, and cache-independent
  action choices; its local algorithm contracts live with the source.

## Where to verify a change

- Game semantics, transitions, settlement, and invariants:
  [game scenarios](../../src/game/tests.rs),
  [decision contracts](../../tests/decision_contract.rs),
  [settlement matrix](../../tests/settlement_matrix.rs), and
  [seeded stress](../../tests/state_stress.rs).
- Policy legality and analysis: [policy integration tests](../../tests/heuristic_policy.rs),
  [exact-analysis tests](../../src/policy/analysis/one_pass_tests.rs), and
  [independent reference](../../src/policy/analysis/one_pass_reference.rs).
- Reproducible experiment tools: the tests alongside
  [`policy_ab`](../../examples/policy_ab/tests.rs) and
  [`heuristic_bench`](../../examples/heuristic_bench/tests.rs).

The [development guide](../development.md) is the maintained entry point for
local and CI verification commands. Historical results belong in
[validation](../validation.md), not in this current architecture map.
