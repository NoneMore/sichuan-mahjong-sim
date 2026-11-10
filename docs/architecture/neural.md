# Neural adapters

[`sichuan-mahjong-neural`](../../crates/neural/) consumes the simulator's public
interfaces. The dependency is one-way: building the core with
`-p sichuan-mahjong-sim` does not compile neural code or its dependencies.
`Game` retains sole ownership of legality, hidden state, transitions, and settlement.

## Decision encoding

`encoding` converts a player-scoped `Decision` into state features and canonical
legal candidates. It is deterministic and stateless, with no teacher analysis
or simulated transitions. Versioning covers feature meaning, layout, scales,
and candidate order, not just tensor dimensions.

| Entry point | Responsibility |
| --- | --- |
| [`encoding/mod.rs`](../../crates/neural/src/encoding/mod.rs) | Assembly and semantic target lookup |
| [`schema.rs`](../../crates/neural/src/encoding/schema.rs) | Feature types, frozen mappings and flattening |
| [`state.rs`](../../crates/neural/src/encoding/state.rs) | Player-visible state conversion |
| [`candidates.rs`](../../crates/neural/src/encoding/candidates.rs) | Candidate enumeration, canonical order and action features |

## Teacher collection

The neural-side collector drives `Game`, `HeuristicPolicy`, and the encoder
without extending Runner. It labels the original decision once and records the
pre-action encoding only after submission succeeds. All seats, stages, forced
actions, passes, and losing trajectories are retained.

Modules under [`collection/`](../../crates/neural/src/collection/) separate
execution from persistence:

| Module | Responsibility |
| --- | --- |
| `corpus.rs` | Seed ranges, shard boundaries, committed inventory and read coordination |
| `driver.rs` | One teacher episode into a fallible record sink; no filesystem knowledge |
| `storage.rs` | OS lock, pending shards and atomic publication |
| `format.rs` | JSONL protocol and validation, reusing encoder mappings/order/features |
| `manifest.rs`, `seed.rs` | Inventory, identity and deterministic split contracts |
| `source.rs`, `build.rs` | Compiled provenance and build/environment facts |
| `stats.rs`, `report.rs` | Coverage, choice metrics and measurement outputs |

A corpus fixes source and encoding versions. Complete episodes form immutable
JSONL shards; per-shard manifest entries keep inventory proportional to shard
count. The seed-only split remains stable across growth and source revisions.
An OS-locked writer validates and publishes each shard before atomically
committing the manifest. Retry ignores unlisted work and resumes from the first
uncommitted seed; append checks metadata without rescanning old decisions.
Strict readers validate listed shards one episode at a time; only `read_corpus`
materializes the entire corpus.

The [crate guide](../../crates/neural/README.md) covers commands and reporting;
module rustdoc defines the encoding and storage contracts.

## Offline behavior cloning

Rust [preparation](../../crates/neural/src/prepare/) reuses strict corpus
validation; the [Python package](../../python/README.md) owns batching,
optimization, metrics and recovery without environment bindings.

```text
captured manifest + immutable corpus shards
    -> Rust validation / packed NumPy cache
    -> Python mmap Dataset / padded batches
    -> shared PyTorch scorer / metrics / checkpoints
```

The snapshot fixes membership during collection appends. Bounded preparation
buffers and staging-directory publication produce a self-contained cache with
versioned layout, checksums and separate collection/preparation provenance.
Python does not parse teacher JSONL or reimplement encoding.

A pointwise state/candidate scorer keeps candidate permutation semantics and
shares its logits interface with future online consumers. Validation NLL selects
the best model; epoch-boundary checkpoints retain optimizer, RNG and best-model
state for recovery, including migration to another output directory. The core
simulator has no training dependency. Operational contracts belong in the
[crate guide](../../crates/neural/README.md#dataset-preparation) and Python guide.

## Verification

[Feature fixtures](../../crates/neural/tests/encoding_features.rs) freeze v1
layout and ownership semantics. [Encoding contract tests](../../crates/neural/tests/encoding_contract.rs)
check candidate legality, information isolation, replay and teacher projection.
Collector tests cover strict round trips, source/split/seed boundaries,
publication recovery, manifest-scale resume, subprocess locks, and record-sink
failures. Sampled collision checks are evidence over their fixtures, not a proof
for every reachable position.
