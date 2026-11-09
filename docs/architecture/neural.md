# Neural adapters

[`sichuan-mahjong-neural`](../../crates/neural/) is a separate workspace crate
that consumes the simulator's public decision surface. The simulator core at
the repository root has no dependency on it. Building or testing the core with
`-p sichuan-mahjong-sim` does not compile neural code.

```text
sichuan-mahjong-neural
        |
        | public Decision, Action, Hand, Meld, Seat, Tile
        v
sichuan-mahjong-sim
```

This is a compile-time dependency boundary. `Game` still owns legality,
transitions and settlement; the neural crate has no privileged access to hidden
hands, wall contents or internal game state. Its production encoder invokes
neither teacher analysis nor simulated transitions. Teacher policies appear in
consumer examples and acceptance tests only.

## Encoding responsibilities

The public API is `sichuan_mahjong_neural::encoding`. It represents one valid
pending `Decision` as structured state and a complete legal candidate list.
It is stateless, not a `Policy` implementation or a Game checkpoint.

| Entry point | Responsibility |
| --- | --- |
| [`encoding/mod.rs`](../../crates/neural/src/encoding/mod.rs) | Public API, encoding assembly and semantic label lookup |
| [`schema.rs`](../../crates/neural/src/encoding/schema.rs) | Feature types, frozen v1 mappings and flattening |
| [`state.rs`](../../crates/neural/src/encoding/state.rs) | Player-visible state conversion |
| [`candidates.rs`](../../crates/neural/src/encoding/candidates.rs) | Legal candidate enumeration, canonical order and action features |

The [crate guide](../../crates/neural/README.md) covers usage; encoding rustdoc
defines the exact feature and compatibility contract.

## Verification and extension

Tests live with the neural crate. The
[feature fixtures](../../crates/neural/tests/encoding_features.rs) freeze v1
layout and physical-ownership semantics. The
[contract tests](../../crates/neural/tests/encoding_contract.rs) drive real Game
decisions, submit candidates to cloned pending games, and reconstruct teacher
inputs from the encoding. Seeded collision checks retain both decisions on
conflicting labels; sampled agreement is not a proof over every reachable
position.

Only encoding is implemented. Future network and training integrations belong
on the neural side of the dependency boundary, without adding learning-framework
dependencies to the simulator core.
