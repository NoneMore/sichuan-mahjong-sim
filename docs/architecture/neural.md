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

## Cache preparation

[`prepare`](../../crates/neural/src/prepare/) consumes a
[`CorpusSnapshot`](../../crates/neural/src/collection/corpus.rs): exact committed
manifest bytes and immutable shard membership, unchanged by later appends.
Its fallible episode visitor reuses strict collection validation, keeping
output provisional until the complete shard checksum validates.

| Module | Responsibility |
| --- | --- |
| `writer.rs` | Snapshot streaming, bounded split buffers and staging publication |
| `manifest.rs` | Versioned dataset manifest, layout, provenance and `cache_id` |
| `npy.rs` | Strict little-endian C-order `.npy` writer/reader |
| `verify.rs` | Corpus-independent cache validation |
| `error.rs` | Preparation and verification error types |

Each split retains one bounded output buffer (or one oversized episode), plus
one input episode in the reader. Shards rotate at episode boundaries; a staging
directory is published only after validation. `dataset.json` binds the array
inventory, versions, checksums, counts, and separate collection/preparation
provenance. `verify_dataset` validates the cache without its source corpus.

The [crate guide](../../crates/neural/README.md#dataset-preparation) covers CLI
operations and the nine-array handoff; [module rustdoc](../../crates/neural/src/prepare/mod.rs)
and [`compute_cache_id`](../../crates/neural/src/prepare/manifest.rs) define the protocol.

## Python datasets and batches

The separate [`mahjong-bc` package](../../python/README.md) consumes prepared
caches without Rust bindings or the original corpus. `cache.py` validates the
inventory, checksums, arrays and metadata against the captured corpus and
collection provenance. Rust owns the opaque `cache_id`; Python preserves it
in dataset identities.

`data.py` exposes read-only packed/sharded datasets with forced and eligible
decision queries. Collation preserves candidate order and local targets while
padding to the batch maximum with a Boolean mask. Iteration follows supplied
indices, includes partial batches and requires `num_workers=0` for mmap data.
The package currently provides data loading, not model scoring or training.

## Verification

[Feature fixtures](../../crates/neural/tests/encoding_features.rs) freeze v1
layout and ownership semantics. [Encoding contract tests](../../crates/neural/tests/encoding_contract.rs)
check candidate legality, information isolation, replay and teacher projection.
Collector tests cover strict round trips, source/split/seed boundaries,
publication recovery, manifest-scale resume, subprocess locks, and record-sink
failures. [Preparation contract tests](../../crates/neural/tests/prepare_contract.rs)
cover round-trip arrays, splits, forced decisions, snapshot append isolation,
corpus-independent verification, bounded/oversized episode shards, failed-run
retry, refused output, empty splits, version and checksum rejection,
derived-count/seed tampering, and CLI strictness.
Sampled collision checks are evidence over their fixtures, not a proof
for every reachable position.
