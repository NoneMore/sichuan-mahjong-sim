# Neural adapters

[`sichuan-mahjong-neural`](../../crates/neural/) consumes the simulator's public
interfaces. The dependency is one-way: building the core with
`-p sichuan-mahjong-sim` does not compile neural code or its dependencies.
`Game` retains sole ownership of legality, hidden state, transitions, and settlement.

For the runnable collection → preparation → training → evaluation sequence,
see the [BC pipeline guide](../guides/bc-pipeline.md).
For Rust/Python protocol ownership and compatibility, see the
[data contract map](../reference/data-contracts.md).

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
in dataset identities. `_array_contract.py` supplies the shared packed-array
invariants for cache validation and direct Dataset construction; protocol and
provenance checks remain in `cache.py`.

`data.py` exposes read-only packed/sharded datasets with forced and eligible
decision queries. Collation preserves candidate order and local targets while
padding to the batch maximum with a Boolean mask. Iteration follows supplied
indices, includes partial batches and requires `num_workers=0` for mmap data.

## Offline policy training

`policy.py` shares one state embedding across a decision's candidates and
scores state/candidate pairs with a pointwise nonlinear MLP. Candidate position
is not an input; padding is masked before loss and action selection.

`train.py` owns CPU or single-local-GPU optimization and epoch-boundary recovery.
Seeded epochs visit every eligible decision once with equal-weight cross-entropy,
including partial batches. `metrics.py` aggregates complete splits by decision,
stage and teacher action kind, reporting forced counts separately. Validation NLL
alone selects `best.pt`; frozen test evaluation is separate from selection. A
shared run-progress object feeds the epoch loop, checkpoints and summaries. The
CLI reuses one loaded checkpoint to resolve configuration and resume training.

`checkpoint.py` validates versioned policy/recovery payloads and publishes
individual files atomically. `last.pt` commits the last completed epoch, including
optimizer, random/traversal and best-model state; an interrupted epoch is replayed.
Resume validates dataset/configuration, device type and dependency identities
before continuation. Output-directory migration republishes both best and last
even when no further epoch or validation improvement occurs.

Run records bind configuration, dataset identities/provenance, dependencies and
metrics to the imported Python source hash; `_source.py` adds Git metadata only
for in-repository source installs.
See the [package guide](../../python/README.md#training-metrics-and-checkpoints)
for policy, checkpoint-format and CLI contracts.

## Online environment binding and comparison

[`crates/python`](../../crates/python/) is a standalone Cargo workspace exposing
`mahjong_eval._native`. Excluding it from the root workspace keeps core,
collector and offline builds independent of Python. Each `Env` owns one Game
and its canonical action cache. Reset/step return owned encoded snapshots;
identities include reset generation, and invalid requests are rejected before
mutation. Rules, automatic transitions and terminal settlement stay in Game.

Teacher queries choose on the original Rust Decision, map semantic actions to
canonical indices and leave progression to step. Explicit cache sharing preserves
choices. Native semantic replay checks the binding path; split queries reuse the
corpus contract. Build-time provenance covers binding/core/neural source bytes,
separate from neural collection provenance and the caller's working directory.

The [online Python package](../../online/README.md) separates execution concerns:

| Module | Responsibility |
|---|---|
| `inference` | Frozen checkpoint weights, shared label-free padding/masks, greedy selection and forced bypass |
| `scheduler` | Bounded concurrent handles, per-policy batches, handle-local routing, shared teacher cache and execution costs |
| `records` | Planned placements, exact checkpoint/imported-source identity, flushed terminal records and seed-blocked summaries |
| `cli` | Selected-best checkpoint and Rust-split validation, comparison setup and persisted-result checks |

Concurrent handles can reuse decision ids, so routing also retains the originating
handle. Every stage and forced choice remains explicit, including continued play
after a win. Failures stop execution without substitute actions. The writer stays
incomplete until records and costs succeed; only a clean close permits summary
publication from persisted files. Training provenance is recorded separately from
the executing Python packages and compiled Rust runtime.

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
