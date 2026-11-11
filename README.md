# Sichuan Mahjong Simulator

A native-Rust simulator for four-player Sichuan mahjong ("blood battle",
血战到底). The project is a **fixed, self-consistent research baseline** for
single-threaded self-play, policy comparison, and reproducible measurement.
It does not aim to implement every local house rule or claim score compatibility
with other variants.

## Baseline semantics in brief

The fixed deck contains 108 tiles (ranks 1–9 in Wan, Tong, and Tiao, four
copies each), with no honors or flowers. The [fixed gameplay rules](docs/baseline-rules.md)
are the authority for intended game semantics. In brief:

- Four players simultaneously exchange three tiles, then choose a missing suit.
- Missing-suit restrictions govern discards, claims, and wins; there is no chi.
- Hu takes priority over kong and peng, and multiple players may win on one discard.
- The dealer opens without drawing and may take any legal opening action.
- Additive fan is capped at **5**; terminal scores are zero-sum, with defined
  wall-exhaustion settlements.

See the [rules contract](docs/baseline-rules.md) for exact timing, settlements,
fan items, and excluded variants.

## Architecture

The Rust core has an authoritative `Game` state machine, a `Runner` that drives
games, and `Policy` implementations selecting actions from player-visible
`Decision` values. Pure tile, hand, and rule helpers support the core.

The separate [neural Rust crate](crates/neural/README.md) handles encoding,
teacher data collection, and NumPy cache preparation. The [Python BC package](python/README.md)
loads prepared datasets and trains policies; the [online evaluation package](online/README.md)
uses a separate PyO3 binding to compare frozen neural policies with the teacher.
See the [implemented architecture map](docs/architecture/overview.md) for
component ownership, data flow, and code entry points.

## Running

Use Rust 1.75 or newer (prefer current stable for development):

```sh
cargo test --workspace --all-targets
cargo run --release -- --games 1000 --seed 7 --policy first
```

The main CLI supports `--policy first|random`; heuristic and missing-only
policies are available through the Rust library and experiment examples.
See [development](docs/development.md) for CLI options, verification commands,
and comparison/benchmark workflows.

## Documentation

Use the [documentation map](docs/README.md) for task-specific guides,
API references, architecture, validation records, and contributor guidance.
