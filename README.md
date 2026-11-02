# Sichuan Mahjong Simulator

A native-Rust simulator for four-player Sichuan mahjong ("blood battle",
血战到底). The project is a **fixed, self-consistent research baseline** for
single-threaded self-play, policy comparison, and reproducible measurement. It
is not a complete implementation of every local house rule, and it does not
claim score compatibility with them.

The fixed deck is 108 tiles: ranks 1-9 in Wan, Tong, and Tiao, four copies of
each tile kind. There are no honors, flowers, winds, or dragons.

## Baseline semantics in brief

The normative rule document is [`docs/baseline-rules.md`](docs/baseline-rules.md).
The most important fixed choices are:

- **Exchange and missing suit.** The game publishes a random giving direction
  (clockwise, counter-clockwise, or opposite) before anyone selects. All four
  players submit three tiles of one suit; the game applies the exchange only
  after all four submissions. Each player then permanently chooses a missing
  suit. A player holding missing-suit tiles must discard one of them, and may
  not peng, kong, or win until the concealed hand is clear of that suit.
- **Claims.** Hu outranks kong, which outranks peng. Multiple Hu claims on one
  discard all win and are paid separately by the discarder. There is no chi.
- **Turn flow.** A normal turn draws, optionally declares kongs, discards, and
  resolves claims. A peng claimant discards without drawing. Kong replacement
  tiles come from the tail of the wall. A robbed added kong is cancelled and
  not paid.
- **Opening turn.** The dealer holds 14 tiles and acts first. The opening turn
  is **not discard-only**: the dealer may discard, declare an ordinary
  concealed kong, or Hu when the baseline shape and missing-suit conditions
  allow it. An opening Hu uses ordinary self-draw scoring; there is no
  heaven-hand bonus.
- **Ending.** A player who wins leaves the hand. The hand ends when only one
  active player remains or when a required draw finds an empty wall. Wall
  exhaustion applies flower-pig payments (16 points per non-pig player), big
  call payments based on maximum hypothetical discard-win fan for tenpai
  players, and refunds of kong income to the original payers where required.
- **Scoring.** Fan items are summed and then capped once at **5**. Points paid
  equal the capped fan total: **there are no exponential multipliers and no
  `2^n` scoring**. The fixed fan table is base win +1, pure one suit +2, seven
  pairs +2, all pungs +1, each root +1, and at most one event bonus +1
  (self-draw, after-kong, kong-discard, or robbing-kong). A win that satisfies
  both seven pairs and a standard decomposition is scored as seven pairs.
- **Not implemented.** Chi, call transfer, heaven/earth-hand bonuses,
  bottom-of-the-sea bonuses, 血流成河, honors/flowers, local fan variants,
  dealer rotation, wind rounds, match-level scoring, and configurable rules are
  outside this baseline. See section 8 of the rule document.

An execution result is the **terminal net score per seat**. The fixed rules are
zero-sum, including kong refunds and wall-exhaustion penalties.

## Architecture

```text
Runner ──drives──> Game
   │                 │
   └──calls──> Policy │
                      └──owns authoritative state, legality, settlement
```

- `src/game.rs` owns the authoritative game state, the pending decision, legal
  action computation, response arbitration, settlement, and cloning.
- `src/policy.rs` defines the `Policy` trait and the two legal baseline
  policies. A policy receives only `Decision`: the acting seat, the player's own
  hand and public observation fields, and the current legal-action constraint.
  It never receives `&Game`, another concealed hand, or wall order.
- `src/runner.rs` drives a new or supplied game, calls one policy per seat,
  records decisions, and separates policy time from rule-driving time. It
  treats a rejected action as an error; it never silently substitutes another
  action or retries forever.
- `src/main.rs` is a small reproducible self-play CLI.
- `src/tile.rs`, `src/action.rs`, `src/seat.rs`, `src/eval.rs`, and
  `src/rules.rs` are the pure foundation shared by Game and policies.

The exchange policy does not enumerate all triples. It selects one suit from
the game's eligible-suit constraint and collects three tiles from the player's
own hand. The random policy owns an independent SplitMix64 RNG and never reads
game randomness.

### Policies

- `FirstLegalPolicy`: deterministic legal baseline. It prefers `Hu`; otherwise
  it uses the first legal action supplied by the game. For exchange it chooses
  the eligible suit with the fewest own tiles (lowest suit index on a tie) and
  passes that suit's three lowest tiles.
- `RandomPolicy`: seeded random legal policy. It prefers `Hu` to increase win
  coverage, and otherwise samples uniformly from the legal actions, or from
  the eligible exchange tiles. It is intended for coverage and reproducibility,
  not playing strength.

`Runner::random(seed)` derives one independent policy stream per seat with
`policy::random_seat_seed`. "Independent" means each seat owns a separate RNG
object with its own state; it does not claim statistical independence of the
derived seeds. Constructing a fresh runner with the same seed and driving the
same game from the start reproduces the same decisions. Rebuilding a runner
from a seed is a replay from the beginning, not the restoration of a mid-game
policy checkpoint.

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

Within one runner, policy streams continue across games, so
`run_batch(first_seed, n)` is the same as `n` consecutive `run_new` calls on
that runner. A fresh runner resets the streams to their seeded positions.

## Running

The crate has no runtime dependencies. Build and test with the stable Rust
toolchain:

```text
cargo build --release
cargo test
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

### Reproducibility

For a fixed build, the same rules seed, policy seed, and policy sequence
reproduce the same terminal scores and decisions. A batch with the same first
seed and game count reproduces the same games in the same order. Random policy
streams continue across games in one runner, so a batch is the continuous
stream; a fresh runner restarts every stream from its seed. Reusing a runner
with a different game seed keeps the policy streams at their current position
instead of restarting them.

The policy RNG is independent of the game's shuffle and settlement randomness
because it is a separate RNG object with separate state. That is a statement
about state ownership, not a statistical claim that one shared CLI `--seed`
produces uncorrelated game and policy randomness.

## Limitations

- This is a fixed research baseline, not a universal Sichuan rules engine.
  Local rules that differ on fan, timing, claims, or settlement are not
  supported.
- Fan is summed and capped at 5. It is not an exponential multiplier system.
- The policies are legal baselines, not search or evaluation engines. No
  strategy budget, lookahead, neural network, GPU, Python binding, parallel
  batch execution, undo, or copy-on-write search is included.
- There is no code for training trajectories or reward shaping. Those belong to
  a research layer above the authoritative game.
- Performance is not promised before measurement. The repository provides the
  measurement path, not a benchmark target.

## Documentation

- [`docs/validation.md`](docs/validation.md) - validation coverage and measured native CPU baseline.
- [`docs/baseline-rules.md`](docs/baseline-rules.md) - normative fixed rules.
- [`docs/integration-contract.md`](docs/integration-contract.md) - Game/Runner
  boundary.
- [`docs/implementation-plan.md`](docs/implementation-plan.md) - implementation
  slices and completion checks.
- [`xzd-rust-simulator-design-revised.md`](xzd-rust-simulator-design-revised.md)
  - architecture decisions and scope.
