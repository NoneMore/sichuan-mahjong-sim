# Sichuan Mahjong Neural

Versioned decision encoding and behavior-cloning teacher data collection. This
workspace crate consumes `sichuan-mahjong-sim` through its public interfaces;
serialization, hashing, and filesystem dependencies remain on the neural side.

## Decision encoding

```rust
use sichuan_mahjong_neural::encoding::{encode, ENCODING_VERSION};
use sichuan_mahjong_sim::{Game, HeuristicPolicy, Policy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut game = Game::new(7);
    let decision = game.next_decision().expect("opening exchange");
    let encoded = encode(&decision);
    let state = encoded.state.flatten();
    let actions: Vec<_> = encoded.candidates.iter()
        .map(|candidate| candidate.features.flatten())
        .collect();

    // Obtain the label from Game's original offered order.
    let target = HeuristicPolicy::new().choose(&decision);
    let target_index = encoded.target_index(&target)?;
    println!("v{ENCODING_VERSION}: {} state values, {} candidates, target {target_index}",
        state.len(), actions.len());
    Ok(())
}
```

V1 produces **1013** state values and **39** values per candidate, all `f32`.
Candidates are canonical semantic actions; their positions are local indices,
not durable action identities. `target_index` normalizes a label and returns its
candidate index or `TargetNotCandidate`.

Persist `ENCODING_VERSION` with models or encoded data, and reject incompatible
versions even if their tensor shapes match. The
[`encoding` rustdoc](src/encoding/mod.rs) defines the exact layout, scales,
categories and ownership semantics. Consumers own batching, padding and selection.

## Teacher data collection

`bc-collect` records complete four-seat `HeuristicPolicy` self-play, including
forced choices, passes, and losing trajectories. Each label uses the teacher's
choice on the original decision; records are written only after Game accepts
its canonical action. Rules, teacher strategy, and encoding semantics are unchanged.

Reusable collection requires a **clean, committed build**. The binary embeds
build-time Git provenance and rejects dirty or unavailable source identity.
Keep corpus and report outputs outside the source tree, or in an ignored directory.

```text
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-collect -- collect \
  --root ../bc-data --games 4 --seed-start 10 --shard-games 2 \
  --cache-bytes 0 --report ../bc-outcome.json

# Append four additional games using the existing corpus settings.
cargo run --release --locked -p sichuan-mahjong-neural --bin bc-collect -- collect \
  --root ../bc-data --games 4

cargo run --locked -p sichuan-mahjong-neural --bin bc-collect -- verify --root ../bc-data
cargo run --locked -p sichuan-mahjong-neural --bin bc-collect -- report --root ../bc-data --json
```

`--games` counts additional games, not the desired corpus total. New-corpus
defaults are seed 0, 100 games per shard, and a 10,000-decision per-game limit
(`--decision-limit`). All four teachers share one invocation-wide cache;
`--cache-bytes 0` disables it, and omission uses the library default. On append,
explicit `--seed-start`, `--shard-games`, and `--decision-limit` must match the
manifest. Seeds never wrap.

### Storage and recovery

Each corpus fixes a source revision and encoding version. `manifest.json` lists
immutable `shards/<first_seed>.jsonl` files containing complete episode blocks;
seeds are 16-digit lowercase hex strings. The seed-only
`seed-sha256-80-10-10-v1` split keeps trajectories grouped across corpus growth
and source revisions. See the [collection module](src/collection/mod.rs) for
the JSONL contract and the [encoder](src/encoding/mod.rs) for feature semantics.

A nonblocking exclusive OS lock on `LOCK` serializes writers (`CorpusBusy` on
contention). Validated shards are published before atomic manifest replacement.
Retry replays the unfinished shard from the first uncommitted seed; unlisted
files are ignored, and a smaller final shard is never extended. Errors stop
collection without skipping games or substituting actions.

Append checks manifest metadata without reading old decisions. `verify` and
`report` validate every listed shard, including checksum, record completeness,
source, split, shapes, actions, and targets. They retain at most one episode;
`read_corpus` is the explicit whole-corpus in-memory API.

### Reports and provenance

Collection prints JSON and optionally writes a new file at `--report`;
existing paths are never overwritten.
`committed_stats` covers only games committed by that invocation; `report`
reconstructs whole-corpus statistics. Both include all stages and action kinds,
including zero support. Kind opportunities count once per decision;
`primary_*` choice metrics exclude forced decisions.

Pilot outcomes include throughput, rules/teacher/encoding/I/O/commit timings,
storage bytes, build/environment facts, seed range, and cache settings. Retain
the command and outcome when interpreting measurements.

Library users enter through `collection::Corpus` and the `read_*`,
`verify_corpus`, and `report_corpus` functions.
`SourceRevision::from_compiled_source()` enforces build provenance; parsing a
revision directly checks text form only and leaves source attestation to the caller.

## Development

Run these commands from the repository root:

```text
cargo test -p sichuan-mahjong-neural --locked
cargo clippy -p sichuan-mahjong-neural --all-targets --locked -- -D warnings
cargo doc -p sichuan-mahjong-neural --no-deps --locked
```

`cargo test -p sichuan-mahjong-sim` tests the core independently. See the
[development guide](../../docs/development.md) for workspace checks and CI, and
the [architecture guide](../../docs/architecture/neural.md) for subsystem boundaries.
