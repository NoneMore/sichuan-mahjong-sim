# Checkpoint continuation and replay

Use this guide when resuming an in-memory game position or verifying that
continuation produces the same outcomes and trace. A cloned `Game` is **not**
a serialized, version-independent checkpoint; policy-owned logical state
must be considered separately. For `Runner` methods and report semantics,
see the [Rust library guide](../library.md#runner-api).

## Continuation and checkpoints

`Game::clone` restores the authoritative position only: wall and draw order,
pending decision, public history, and settlement state. Anything a stateful
policy keeps on its own, such as `RandomPolicy`'s RNG position, is *not* in the
game. A checkpoint is therefore complete for stateless policies
(`FirstLegalPolicy`, `HeuristicPolicy`, and custom stateless policies) or when
the policy state is cloned as well.

`HeuristicPolicy` is stateless in the checkpoint sense: its hand-analysis caches
are internal memoization with no logical effect, so they can be discarded and
rebuilt at any point, and a resumed game makes the same decisions with cold or
warm caches. Continuing a heuristic run therefore needs only `Game::clone` and
fresh `HeuristicPolicy::new()` values.

The decision surface is enough to advance a checkpoint with a stateless
policy:

```rust
use sichuan_mahjong_sim::policy::{FirstLegalPolicy, Policy};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::Game;

let mut game = Game::new(1_234);
let mut policy = FirstLegalPolicy::new();

for _ in 0..24 {
    let (seat, action) = match game.next_decision() {
        Some(decision) => {
            let seat = decision.seat;
            let action = policy.choose(&decision);
            (seat, action)
        }
        None => break,
    };
    game.submit(seat, action)?;
}

let mut checkpoint = game.clone();

let mut first_runner = Runner::first_legal();
let forward = first_runner.run_game(&mut game)?;

// A fresh stateless runner carries no continuation state of its own.
let mut second_runner = Runner::first_legal();
let resumed = second_runner.run_game(&mut checkpoint)?;

assert_eq!(forward.scores, resumed.scores);
assert_eq!(forward.trace, resumed.trace);
```

`RandomPolicy` is `Clone`, so the complete random-run checkpoint includes
the game **and all four policies after they have consumed decisions**. Restoring
a new `Runner::random(seed)` instead would restart the policy streams.

```rust
use sichuan_mahjong_sim::policy::{random_seat_seed, Policy, RandomPolicy};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::{Game, Seat};

let mut game = Game::new(1_234);
let mut policies: [RandomPolicy; 4] = std::array::from_fn(|index| {
    let seat = Seat::new(index).expect("seat index is in range");
    RandomPolicy::new(random_seat_seed(0xC0FF_EE00, seat))
});

// Consume real decisions before taking the checkpoint: the policy RNG streams
// have now advanced and cannot be restored by reseeding a fresh runner.
let mut applied = 0;
for _ in 0..24 {
    let (seat, action) = match game.next_decision() {
        Some(decision) => {
            let seat = decision.seat;
            let action = policies[seat.index()].choose(&decision);
            (seat, action)
        }
        None => break,
    };
    game.submit(seat, action)?;
    applied += 1;
}
assert!(applied > 0);

let mut checkpoint = game.clone();
let checkpoint_policies = policies.clone();

let into_runner = |policies: [RandomPolicy; 4]| {
    Runner::new(policies.map(|p| Box::new(p) as Box<dyn Policy>))
};

let mut first_runner = into_runner(policies);
let forward = first_runner.run_game(&mut game)?;

let mut second_runner = into_runner(checkpoint_policies);
let resumed = second_runner.run_game(&mut checkpoint)?;

assert_eq!(forward.scores, resumed.scores);
assert_eq!(forward.trace, resumed.trace);
```

This is an in-memory, same-build continuation check, not a serialized
cross-version checkpoint guarantee. The game seed and policy seed have
different roles; both stateful policy positions must be preserved for an
exact continuation.
