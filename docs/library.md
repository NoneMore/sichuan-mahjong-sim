# Rust library API

This reference covers the simulator's public `Policy`, `Runner`, and cache
usage. For seeded CLI runs and experiments, see the [workflow guides](README.md);
for continuation examples, see [checkpoint and replay](guides/checkpoint-replay.md).
For component responsibilities,
see the [execution](architecture/execution.md) and
[policy](architecture/policy.md) architecture guides.

## Construct policies

The public `Policy` trait selects from the current `Decision`; available
strategies and their behavior are listed in [Policy architecture](architecture/policy.md).
Use `Runner::first_legal()` or `Runner::random(seed)` for the built-in
baselines. To run heuristic or missing-only policies, construct four policy
values and pass them to `Runner::new`. For heuristic self-play,
`HeuristicPolicy::shared_group()` shares a bounded cache across seats.

## Heuristic cache contexts

`HeuristicPolicy::new()` and `Default` create an independent, enabled cache
context. `HeuristicPolicy::shared_group()` creates four policies sharing one
default context. For a custom byte budget, disabled caching, or mixed policies,
bind the same explicit context to the heuristic seats:

```rust
use sichuan_mahjong_sim::policy::{HeuristicCache, HeuristicPolicy, Policy};
use sichuan_mahjong_sim::runner::Runner;

let cache = HeuristicCache::new(27_787_264)?; // selected 26.5 MiB budget
let policies = std::array::from_fn(|_| {
    Box::new(HeuristicPolicy::with_cache(cache.clone())) as Box<dyn Policy>
});
let mut runner = Runner::new(policies);
let report = runner.run_new(7)?;
let status = cache.status();
```

A zero budget disables only cross-decision memoization; failed allocation
also falls back to exact analysis without changing choices. Inspect
`cache.status()` for diagnostics, and use a new context for a different budget
or independent cold run. See [Policy architecture](architecture/policy.md#cache-and-checkpoint-boundary)
for the cache and checkpoint design constraints.

## Runner API

```rust
use sichuan_mahjong_sim::runner::Runner;

let mut runner = Runner::random(7);
let report = runner.run_new(42)?;
println!("scores: {:?}", report.scores);
println!("reason: {:?}", report.end_reason);
println!("decisions: {}", report.decisions);
println!("policy time: {:?}", report.policy_time);
println!("rules time: {:?}", report.rules_time);

let batch = runner.run_batch(100, 1_000)?;
println!("{:.2} games/sec", batch.games_per_sec());
println!("{:.2} decisions/sec", batch.decisions_per_sec());
```

`Runner::run_game(&mut game)` accepts an existing game, including a cloned
mid-game checkpoint; it does not reset or inspect the game. `Runner::run_new`
creates a seeded game. `Runner::run_batch(first_seed, games)` uses seed
`first_seed.wrapping_add(i)` for game `i` and rejects `games == 0`. The
remaining policy RNG state is not part of the game, so a resumed stateful run
also needs the policies from the checkpoint (see
[checkpoint and replay](guides/checkpoint-replay.md)).

The decision safety limit is counted per call: per `run_game`, per generated
game of `run_batch`, and per `run_new`. It is not accumulated across resumed
calls. A game interrupted by `RunnerError::DecisionLimit` keeps every action
already accepted; raise the limit with `Runner::set_decision_limit` or
`Runner::with_decision_limit` and call `run_game` again to continue.

Single-game reports include a decision trace for replay checks. Batch reports
omit traces and aggregate terminal score totals, winners by seat, finish
reasons, decisions, and separate timings. `policy_time` covers `Policy::choose`;
`rules_time` covers `next_decision` and `submit`; batch `total_time` is the
whole measured wall clock, including game creation.

