# Game / Runner integration contract

This narrow contract allows independent implementation of Game internals and native execution. The fixed baseline remains authoritative for rules.

The Game slice owns `src/game.rs` (or its submodules), Game exports in `src/lib.rs`, and `docs/baseline-rules.md`. The execution slice owns `src/policy.rs`, `src/runner.rs`, `src/main.rs`, execution integration tests, and `README.md`. The execution slice must not edit `src/lib.rs`; Game adds the policy/runner module declarations and root exports when those files exist. Do not reformat another slice's files while they are being edited.

## Public Game surface

```rust
// Defined in game, re-exported at crate root.
impl Game {
    pub fn new(seed: u64) -> Self;
    pub fn next_decision(&mut self) -> Option<Decision<'_>>;
    pub fn submit(&mut self, seat: Seat, action: Action) -> Result<(), GameError>;
    pub fn result(&self) -> Option<&GameResult>;
}
// Game implements Clone, Debug; GameError implements Display and Error.
pub struct Decision<'a> {
    pub seat: Seat,
    pub observation: Observation<'a>,
    pub legal_actions: LegalActions<'a>,
}
pub enum LegalActions<'a> {
    // Compact constraint, not an enumeration of all triples.
    Exchange { eligible_suits: [bool; 3] },
    Actions(&'a [Action]),
}
pub struct Observation<'a> {
    pub own_hand: &'a Hand,
    // Game may add public fields for own melds, public opponents, public
    // discards, missing-suit declarations, exchange direction, wall count,
    // scores, response source, and decision stage. Never expose a Game ref,
    // another concealed hand, the wall contents, or pending private choices.
}
pub enum EndReason { ThreeWinners, WallExhausted }
pub struct GameResult {
    pub scores: [i32; 4],
    pub winners: [bool; 4],
    pub end_reason: EndReason,
}
```

`next_decision` performs automatic progression but is idempotent when a choice is pending. `submit` consumes only the pending player's valid action; invalid input must leave authoritative state unchanged. The current decision's legality is reused rather than recomputed at submission.

## Execution slice surface

```rust
pub trait Policy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action;
}
```

Policy names, Runner signatures, and CLI flags are owned by the execution slice. A deterministic legal baseline is sufficient; it can prefer Hu and select suitable exchange tiles directly from `own_hand`. Runner must accept an existing Game as well as create seeded games, must report policy failures rather than loop indefinitely, and must not inspect privileged Game state. Count decisions and separate policy choose time from rule-driving time. Any randomized policy owns its RNG independently of Game.

## Explicit parent rule adjustment

The opening dealer action is **not discard-only**: it may discard, declare an ordinary concealed kong, or Hu if the baseline shape and missing-suit conditions permit. An opening Hu uses ordinary self-draw scoring, with no extra heaven-hand bonus. Not implementing a heaven-hand bonus must not silently remove an otherwise ordinary legal action. Update baseline-rules section 4.1 accordingly.
