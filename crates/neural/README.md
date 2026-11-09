# Sichuan Mahjong Neural

Versioned decision encoding for neural policies. This workspace crate consumes
`sichuan-mahjong-sim` through its public interfaces.

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
