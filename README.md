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

The simulator is one Rust crate with an authoritative `Game`, a `Runner`
that drives games, and `Policy` implementations that choose actions from
player-visible decisions. Pure tile, hand, rule, and winning-shape helpers
support the core. See the [current architecture overview](docs/architecture/overview.md)
for implemented component responsibilities, decision flow, information
boundaries, and code navigation. The [baseline rules](docs/baseline-rules.md)
define intended game semantics.

### Policies

- `FirstLegalPolicy`: deterministic baseline, preferring `Hu` and otherwise
  choosing a legal action.
- `RandomPolicy`: seeded legal-choice baseline, also preferring `Hu`;
  useful for coverage and replay, not playing strength.

`Runner::random(seed)` initializes a separate RNG stream for each seat.
It replays from initial streams; it does not restore mid-game policy state
(see **Continuation and checkpoints**).

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
game. A checkpoint is therefore complete only for stateless policies
(`FirstLegalPolicy` and custom stateless policies) or when the policy state is
cloned as well.

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

Within one runner, policy streams continue across games; a fresh runner
restarts them. Game and policy randomness have separate states, which does
not imply statistical independence of the derived seeds.

## Running

The crate has no runtime dependencies. Build and test with the stable Rust
toolchain:

```text
cargo build --release
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

Run a reproducible self-play batch:

```text
cargo run --release -- --games 1000 --seed 7 --policy first
cargo run --release -- --games 1000 --seed 7 --policy random
```

CLI arguments:

| Argument | Meaning |
|---|---|
| `--games N` | Number of games; must be at least 1. Default 1. |
| `--seed S` | Seeds the first game and, with `--policy random`, the seat policy streams. Game `i` uses wrapping `S + i`. Default 0. |
| `--policy first\|random` | Policy for every seat. Default `first`. |
| `-h`, `--help` | Print usage and exit; arguments after it are ignored. |

The CLI deliberately has a single `--seed` with two roles: it seeds the game
sequence and, for `--policy random`, the policy streams. The library API sets
them independently by choosing the game seed passed to `Game::new` (or
`run_new`/`run_batch`) and the policy seed passed to `Runner::random` or to
explicitly constructed `RandomPolicy` values. There is no separate
`--policy-seed` flag.

Unknown arguments, missing values, malformed numbers, an unknown policy, and
`--games 0` are rejected with a non-zero exit status. `-h`/`--help` short-
circuits on the usual CLI convention: it prints usage and exits successfully,
ignoring every argument that follows it. Arguments before it are still
validated. The output reports finish reasons, terminal score totals, mean score
per seat, win counts, total decisions, separate rules/policy timings, and
`games/sec` plus `decisions/sec`.

Measurements are single-threaded and include game construction in the batch
wall time. There is intentionally no performance threshold in the project;
record the command, seed, policy, and observed rates when comparing builds.

## Limitations

This is a fixed research baseline, not a configurable local-rules engine.
The included policies are legality baselines, not strategic search; there
are no training, Python, GPU, or parallel batch components. See
[baseline exclusions](docs/baseline-rules.md) for gameplay scope.
Performance is measured rather than guaranteed; consult the
[validation record](docs/validation.md) for historical observations.

## Documentation

- [`docs/governance.md`](docs/governance.md) - shared conventions for
  contributors and coding agents.
- [`docs/architecture/overview.md`](docs/architecture/overview.md) -
  navigation map of the current implemented architecture.
- [`docs/baseline-rules.md`](docs/baseline-rules.md) - intended fixed game rules.
- [`docs/validation.md`](docs/validation.md) - recorded validation coverage
  and historical native CPU measurements, not a current performance promise.
