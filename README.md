# Sichuan Mahjong Simulator

A native-Rust simulator for four-player Sichuan mahjong ("blood battle",
血战到底). The project is a **fixed, self-consistent research baseline** for
single-threaded self-play, policy comparison, and reproducible measurement. It
is not a complete implementation of every local house rule, and it does not
claim score compatibility with them.

The fixed deck is 108 tiles: ranks 1-9 in Wan, Tong, and Tiao, four copies of
each tile kind. There are no honors, flowers, winds, or dragons.

## Baseline semantics in brief

The intended game rules are defined in
[`docs/baseline-rules.md`](docs/baseline-rules.md). In particular:

- Four players exchange three tiles simultaneously, then choose a missing
  suit. Missing-suit restrictions apply to discards, claims, and wins.
- Hu takes priority over kong and peng; multiple players may win on one
  discard. There is no chi. Kong draws and robbed added kongs have distinct
  settlement behavior.
- The dealer acts first without drawing and may make any legal opening
  action, including an ordinary self-draw Hu or concealed kong.
- Scoring uses additive fan capped at **5**, not exponential multipliers.
  Terminal net scores are zero-sum; wall exhaustion can trigger flower-pig,
  big-call, and kong-income refund payments.

See the rules document for exact conditions, fan items, and excluded variants.

## Architecture

The simulator core is a Rust crate with an authoritative `Game`, a `Runner`
that drives games, and `Policy` implementations that choose actions from
player-visible decisions. Pure tile, hand, rule, and winning-shape helpers
support the core. See the [current architecture overview](docs/architecture/overview.md)
for implemented component responsibilities, decision flow, information
boundaries, and code navigation. The [baseline rules](docs/baseline-rules.md)
define intended game semantics.

### Neural modules

[`sichuan-mahjong-neural`](crates/neural/README.md) is a separate workspace
crate for neural-policy representations, behavior-cloning teacher data
collection, and offline NumPy cache preparation. It depends on the simulator’s
public decision interface; the core has no neural dependency. See its
[architecture guide](docs/architecture/neural.md) for the boundary and the
[crate guide](crates/neural/README.md#dataset-preparation) for the prepared-cache
commands and layout. The separate [Python package](python/README.md) provides
cache validation, sharded mmap datasets, padded PyTorch batches and offline
behavior-cloning training with versioned policy checkpoints.

### Policies

- `FirstLegalPolicy`: deterministic baseline, preferring `Hu` and otherwise
  choosing a legal action.
- `RandomPolicy`: seeded legal-choice baseline, also preferring `Hu`;
  useful for coverage and replay, not playing strength.
- `HeuristicPolicy`: deterministic native policy that ranks post-discard hands
  by shanten, then ukeire, with a discard-win fan tie-break at tenpai, and
  applies conservative peng and kong gates. It reads only `Decision`, uses no
  RNG, and has no logical checkpoint state; any internal analysis cache is
  discardable without changing later decisions.
- `MissingOnlyPolicy`: stateless ablation that uses the heuristic
  fewest-tile missing-suit declaration and delegates all other choices to
  `FirstLegalPolicy`. It shares the heuristic exchange-selection behavior.

`Runner::random(seed)` initializes a separate RNG stream for each seat. There is
no heuristic or missing-only runner shortcut: build `HeuristicPolicy::new()` or
`MissingOnlyPolicy::new()` values and pass them to `Runner::new`. For heuristic
self-play, prefer `HeuristicPolicy::shared_group()` so all four seats share one
bounded analysis-cache context rather than four independent budgets. The main
simulation CLI still offers only `--policy first|random`. Both policies are
available through the library and the
[`policy_ab` example](docs/development.md#policy-ab-comparison-example); four-seat heuristic
self-play also has a dedicated
[`heuristic_bench` example](docs/development.md#heuristic-benchmark-example).

### Heuristic cache contexts

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
or independent cold run. See [Policy architecture](docs/architecture/policy.md#cache-and-checkpoint-boundary)
for the design constraints and the
[historical cache study](docs/validation.md#exact-analysis-changes-and-profiling)
for measured trade-offs.

### Runner API

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
also needs the policies from the checkpoint (see below).

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

### Continuation and checkpoints

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

## Running

Rust 1.75 or newer is required; use the current stable toolchain for
development. A quick verification and deterministic self-play run:

```sh
cargo test --workspace --all-targets
cargo run --release -- --games 1000 --seed 7 --policy first
```

The main CLI supports `--policy first|random`. Two standalone examples are
available: [balanced 2A2B policy comparisons](docs/development.md#policy-ab-comparison-example)
and [four-seat heuristic benchmarking](docs/development.md#heuristic-benchmark-example).
See the [development guide](docs/development.md) for CLI options, experiment
outputs, Debug/Release and Clippy checks, CI and Rust 1.75 compatibility.

## Documentation

- [`docs/governance.md`](docs/governance.md) - shared conventions for
  contributors and coding agents.
- [`docs/development.md`](docs/development.md) - commands, CLI/executable
  examples, CI and MSRV verification.
- [`docs/architecture/overview.md`](docs/architecture/overview.md) -
  navigation map of the current implemented architecture.
- [`docs/baseline-rules.md`](docs/baseline-rules.md) - intended fixed game rules.
- [`docs/validation.md`](docs/validation.md) - validation index and historical
  native CPU measurements, not a current performance promise.
