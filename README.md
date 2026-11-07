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
- `HeuristicPolicy`: deterministic native policy that ranks post-discard hands
  by shanten, then ukeire, with a discard-win fan tie-break at tenpai, and
  applies conservative peng and kong gates. It reads only `Decision`, uses no
  RNG, and has no logical checkpoint state; any internal analysis cache is
  discardable without changing later decisions.
- `MissingOnlyPolicy`: missing-suit-only ablation. It makes exactly the
  `FirstLegalPolicy` choice at every decision except the missing-suit
  declaration, where it uses the `HeuristicPolicy` fewest-tile rule (fewest own
  tiles after the exchange, lowest suit index on a tie). `MissingOnlyPolicy`
  and `HeuristicPolicy` share exchange selection and the missing-suit rule, so
  comparing the ablation with `FirstLegalPolicy` isolates the value of a
  non-arbitrary suit declaration, while comparing it with `HeuristicPolicy`
  measures the remaining
  heuristic play logic, including discard ranking and the peng/kong gates. It
  is stateless.

`Runner::random(seed)` initializes a separate RNG stream for each seat. There is
no heuristic or missing-only runner shortcut: build `HeuristicPolicy::new()` or
`MissingOnlyPolicy::new()` values and pass them to `Runner::new`. For heuristic
self-play, prefer `HeuristicPolicy::shared_group()` so all four seats share one
bounded analysis-cache context rather than four independent budgets. The main
simulation CLI still offers only `--policy first|random`. Both policies are
available through the library and the
[`policy_ab` example](#policy-ab-comparison-example); four-seat heuristic
self-play also has a dedicated
[`heuristic_bench` example](#heuristic-benchmark-example).

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

A zero budget disables the cross-decision table, not the existing decision-local
memoization. Invalid or too-small positive budgets return `CacheConfigError`.
The large table is allocated lazily on the first masked-DP local miss, never
exceeds its configured budget, and uses local replacement rather than growing
or clearing the whole table. Fixed context overhead is reported separately.
Allocation failure permanently disables new caching in that context without
changing actions; the library does not print diagnostics. Inspect `status()`
to distinguish disabled, unallocated, ready, and allocation-failed contexts.

Cloning a policy or context shares its cache and initialization/failure state;
the last holder releases it. Policies are `Clone` and `Send + Sync`, but no
longer `Copy` or constant-constructible. There is no global or thread-local
sharing. Reuse a context explicitly across decisions, games, or runners; use a
new context to change its budget or start a cold experiment. Cache contents
are discardable memoization and never part of a checkpoint.

The 26.5 MiB default was selected against 53 MiB in paired batch measurements
on the measured host; doubling memory did not reach the predeclared
extra-benefit threshold. Short cold work may not amortize initialization, so
isolated short tasks should consider budget `0`. See the
[result summary](specs/archive/bounded-masked-dp-cache/results.md) for the
historical workload, memory cost, and limitations.

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
explicitly constructed `RandomPolicy` values.

Unknown arguments, missing values, malformed numbers, an unknown policy, and
`--games 0` are rejected with a non-zero exit status. `-h`/`--help` short-
circuits on the usual CLI convention: it prints usage and exits successfully,
ignoring every argument that follows it. Arguments before it are still
validated. The output reports finish reasons, terminal score totals, mean score
per seat, win counts, total decisions, separate rules/policy timings, and
`games/sec` plus `decisions/sec`.

Measurements are single-threaded and include game construction in the batch
wall time. When comparing builds, record the command, seed, policy, and observed
rates.

### Policy A/B comparison example

`examples/policy_ab.rs` runs a balanced 2A2B comparison. By default A is
`FirstLegalPolicy` and B is `RandomPolicy`, which reproduces the original
behavior; `--a` and `--b` select `first`, `random`, `heuristic`, or
`missing-only` for each side. Every game seed is played six times with A in
seats `[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]` and B in the remaining seats, each
placement using fresh policies and a fresh `Runner`. All heuristic seats on
both sides share one explicit cache context across the complete invocation
(seeds and placements); non-heuristic seats do not use it. The cache is
memoization only and does not change the comparison's decisions. Each seat running
`RandomPolicy` (on either side) starts from
`policy::random_seat_seed(game_seed ^ policy_seed, seat)`; deterministic styles
ignore that stream.

```text
cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out ab-demo

# Heuristic policy as A against the deterministic baseline:
cargo run --release --example policy_ab -- \
  --seeds 1,2,3,4 --policy-seed 0 --decision-limit 10000 --out ab-heuristic \
  --a heuristic --b first

# Missing-suit-only ablation as B against the baseline, and as A against the
# heuristic:
cargo run --release --example policy_ab -- \
  --seeds 1,2 --policy-seed 0 --decision-limit 10000 --out ab-missing-vs-first \
  --a first --b missing-only
cargo run --release --example policy_ab -- \
  --seeds 1,2 --policy-seed 0 --decision-limit 10000 --out ab-missing-vs-heuristic \
  --a missing-only --b heuristic
```

| Argument | Meaning |
|---|---|
| `--seeds LIST` | Comma-separated, nonempty, distinct `u64` game seeds. |
| `--policy-seed S` | `u64` seed for the per-seat random policy streams. |
| `--decision-limit N` | Positive per-game decision limit passed to the runner. |
| `--out DIR` | Output directory; it must not already exist. |
| `--a STYLE` | Policy in the two A seats: `first`, `random`, `heuristic`, or `missing-only`. Default `first`. |
| `--b STYLE` | Policy in the two remaining seats: `first`, `random`, `heuristic`, or `missing-only`. Default `random`. |
| `--cache-budget-bytes B` | Shared heuristic table budget in bytes; `0` disables it. Only valid when A or B is `heuristic`; the default is the selected 26.5 MiB library budget. |
| `-h`, `--help` | Print usage and exit. |

`--a` and `--b` are optional and independent, so `--a heuristic --b random` or
`--a first --b heuristic` work as well. The selection is recorded in `run.txt`
and `summary.txt`. Deterministic styles ignore the policy seed but still receive
their seat's derived stream seed. The example validates this comparison
workflow; it is not a policy-strength result.

All contexts report their budget, allocation, and state in cache diagnostics;
a failed allocation emits one warning per context while play continues. A
cache-disabled comparison still keeps decision-local analysis memoization.

`--out` uses a single `create_dir`, so an existing directory is refused rather
than overwritten, and its parent directory must already exist (for example,
create `runs/` before using `--out runs/ab-demo`).

Artifacts are written to the fresh output directory:

- `run.txt`: simulator revision and dirty status, toolchain, the debug-quoted
  argv (one quoted element per argument, so paths with spaces stay
  unambiguous), the selected policy names for A and B, policy seed and stream
  derivation, game seeds, decision limit, and placements. Revision, dirty
  status, and toolchain are collected with `git`/`rustc` against the source tree
  that built the example (`CARGO_MANIFEST_DIR`), not the caller's working
  directory; unavailable metadata is recorded as `unknown` with a warning.
- `games.tsv`: one tab-separated record per completed game (`seed`,
  `a_seat0`/`a_seat1`, four zero-sum scores by seat, end reason, decision
  count). Records are flushed as games finish, so a failed run keeps its
  completed records.
- `cache.txt` (when heuristic seats are present): the invocation-wide sharing
  scope, budget source, initial/final cache status, fixed context overhead,
  table allocation and counters, plus any allocation-failure warning. It is
  retained on execution failure alongside completed game records.
- `summary.txt`: `A_mean`, `B_mean`, `Delta = A_mean - B_mean` (positive
  favors A), the sample standard error over seed blocks (`SE`, unavailable for
  one seed), and each seed's `D`. The summary is recomputed from the persisted
  `games.tsv` records after all games. In both missing-only examples above, a
  negative `Delta` favors the policy upgrade in B; the ablation itself appears
  on different sides of the two comparisons.

An illegal action, decision-limit exhaustion, missing terminal result, output
error, or failed consistency check exits non-zero without writing a summary;
completed records remain for diagnosis. `SE` uses the seed block as the
sampling unit and describes variation among the provided seeds, not
population-level precision.

### Heuristic benchmark example

`examples/heuristic_bench.rs` measures complete, single-threaded
four-seat `HeuristicPolicy` self-play through `Runner::run_batch`.
It does not change game rules, policy decisions, or the main CLI.

```text
# Cold measured context; warmup only warms process-level state:
cargo run --release --example heuristic_bench -- --games 200 --seed 7 --warmup 20

# Explicitly inherit the warmed context, using the 53 MiB candidate:
cargo run --release --example heuristic_bench -- --games 200 --seed 7 \
  --warmup 20 --lifecycle warm --cache-budget-bytes 55574528
```

`--games N` (positive) and `--seed S` are required. `--warmup W` defaults to
`0`; `--lifecycle cold|warm` defaults to `cold`, and `warm` requires `W >= 1`.
`--cache-budget-bytes B` selects the shared table budget (`0` disables the
cross-decision cache; the selected 26.5 MiB default is 27,787,264 bytes; the
53 MiB comparison tier is 55,574,528 bytes).
`--help` prints usage. The measured games use wrapping seeds `S + i`; warmup
uses the adjacent `W` seeds preceding `S`. The tool rejects `N + W > 2^64`
to prevent overlap.

All four seats share one context, reused across measured games. With `cold`,
warmup uses a separate runner/context which is dropped before measurement;
the measured context starts empty, and its lazy initialization is included in
the measured batch time. With `warm`, one runner/context survives both phases,
and initialization that occurs during warmup is charged to warmup instead.
Cold `--warmup 0` in a fresh process also exercises full cold startup; measure
external launch-to-exit wall time separately to include process startup.

The `heuristic-benchmark-v2` report uses one `key: value` per line. It records
the workload, lifecycle, sharing scope, budget, actual table allocation, cache
state/counters before and after measurement, initialization boundary, separate
warmup time, measured game/decision counts, batch/policy/rules times,
throughput, mean policy cost, and terminal-game statistics. Cache-status
snapshots are outside the timed batch. Field names carry units; `seat_order`
labels seat arrays and `completed_games` records actual completions separately
from requested `games`. Unavailable rates use `unavailable`.

Measured `total_time` includes game construction, driving and aggregation, not
runner construction, reporting, or warmup; policy and rules timings need not
sum to it. Do not mix these batch times with external process wall times in a
paired ratio. The mean policy cost covers every decision kind; aggregate game
results do not establish action-by-action equivalence. Warmup does not reset
process-level state or guarantee a speedup.

The command reports one observation, not a statistical conclusion. Record the
source revision, build/toolchain, environment and repeat-run conditions.
Allocation degradation remains visible in the report and warns once per
context; retain such runs but do not use them for performance adoption. Failed
input, execution, or report output makes a run invalid; no partial batch is
reported as a normal measurement.

The figures are historical. Temporary experiment drivers, replay collectors,
and raw measurement data were removed at the user's request, so the original
samples cannot be recomputed from repository files. The permanent Rust cache,
continuation, example, and benchmark tests remain in the tree; the
[result summary](specs/archive/bounded-masked-dp-cache/results.md) retains the
protocol, boundaries, and adoption results.

## Documentation

- [`docs/governance.md`](docs/governance.md) - shared conventions for
  contributors and coding agents.
- [`docs/architecture/overview.md`](docs/architecture/overview.md) -
  navigation map of the current implemented architecture.
- [`docs/baseline-rules.md`](docs/baseline-rules.md) - intended fixed game rules.
- [`docs/validation.md`](docs/validation.md) - validation index and historical
  native CPU measurements, not a current performance promise.
- [`docs/studies/shanten-ukeire.md`](docs/studies/shanten-ukeire.md) - heuristic
  implementation coverage, exploratory A/B results, and timing conditions.
- [`docs/studies/missing-only-ablation.md`](docs/studies/missing-only-ablation.md) -
  48,000-game missing-suit ablation with methods, results, and provenance.
