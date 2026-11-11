# Checkpoint continuation and replay

An in-memory checkpoint combines a cloned `Game` with the logical state
of its policies. FirstLegalPolicy and HeuristicPolicy are stateless for
continuation; their caches may be rebuilt. RandomPolicy owns a separate
RNG stream for each seat, so those four policy objects must also be cloned.

This example advances real random decisions, checkpoints both Game and
policies, then verifies identical continuation traces and scores:

```rust
use sichuan_mahjong_sim::policy::{random_seat_seed, Policy, RandomPolicy};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::{Game, Seat};

let mut game = Game::new(4_242);
let mut policies: [RandomPolicy; 4] = std::array::from_fn(|index| {
    let seat = Seat::new(index).unwrap();
    RandomPolicy::new(random_seat_seed(0x00C0_FFEE, seat))
});

let mut advanced = 0;
while advanced < 40 {
    let Some(decision) = game.next_decision() else { break };
    let seat = decision.seat;
    let action = policies[seat.index()].choose(&decision);
    game.submit(seat, action)?;
    advanced += 1;
}
assert_eq!(advanced, 40);

let mut cloned_game = game.clone();
let cloned_policies = policies.clone();
let runner_from = |policies: [RandomPolicy; 4]| {
    Runner::new(policies.map(|p| Box::new(p) as Box<dyn Policy>))
};

let forward = runner_from(policies).run_game(&mut game)?;
let resumed = runner_from(cloned_policies).run_game(&mut cloned_game)?;
assert_eq!(forward.trace, resumed.trace);
assert_eq!(forward.scores, resumed.scores);
```

A stateless continuation needs only `Game::clone()` followed by a fresh
`Runner::first_legal()` or fresh heuristic policies.
The [execution tests](../../tests/execution.rs) cover both cases;
the [Runner API](../library.md#runner-api) describes continuation methods.
