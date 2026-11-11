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

`RandomPolicy` is `Clone`, so no additional API is needed to checkpoint a
random run: clone the game and every seat policy, then build a runner from the
cloned policies. `Runner::random(seed)` is *not* a checkpoint restore; it
starts each stream at its seeded position, exactly like a fresh runner at the
start of a batch.

```rust
use sichuan_mahjong_sim::policy::{random_seat_seed, Policy, RandomPolicy};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::{Game, Seat};

let mut game = Game::new(1_234);
let mut policies: [RandomPolicy; 4] = std::array::from_fn(|index| {
    let seat = Seat::new(index).expect("seat index is in range");
    RandomPolicy::new(random_seat_seed(0xC0FF_EE00, seat))
});

// Drive the opening decisions with the public surface, as above.
// ...

let mut checkpoint = game.clone();
let checkpoint_policies = policies.clone();

let policies = checkpoint_policies.map(|policy| Box::new(policy) as Box<dyn Policy>);
let resumed = Runner::new(policies).run_game(&mut checkpoint)?;
```

Policy RNG state continues across games in the same runner; a fresh runner
starts from its initial seeds. Separate game and policy RNG state does not
establish statistical independence between their seed-derived sequences.
