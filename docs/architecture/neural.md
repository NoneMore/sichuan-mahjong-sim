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

[`crates/python`](../../crates/python/) is a separate Cargo workspace and PyO3
extension exposed as `mahjong_eval._native`. It is excluded from the root
workspace, so core, neural, collector and offline training builds neither
resolve nor link Python. The binding owns no rules: `Env.reset(seed)` returns an
owned float32 snapshot of the first pending `Decision`, and `Env.step` resolves
a local index through the handle's cached canonical actions and commits one
`Action` to `Game`. Decision identities pack a per-handle reset generation and
sequence, so stale identities and out-of-range indices are rejected before
mutation; snapshots never alias authoritative state. Every stage, forced choice,
response and continued play after a win remains an explicit pending decision.
Terminal values report seat scores, Boolean win flags, end reason, and the
accepted decision count including forced choices.

`Env.teacher_index` evaluates `HeuristicPolicy` on the original Rust decision and
maps its semantic choice through the encoder's canonical target lookup; it does
not advance the game. Teacher cache contexts can be shared explicitly across
handles and calls. `native_teacher_replay` exposes the same native teacher as a
complete semantic trace for cross-language replay; split bucket/name and
compiled extension source identity are separate query entry points. The
extension's build script hashes its own crate, the simulator core and the neural
crate, so the compiled identity is independent of the caller's working
directory and of the neural crate's collection provenance.

The [online Python package](../../online/README.md) owns scheduling, inference,
and records. `inference.FrozenPolicy` adopts an already loaded `LoadedCheckpoint`
(or loads one), rejects `last`, keeps eval-mode float32 weights on one device,
and selects from the shared label-free `pad_decision_inputs` contract; single
candidates bypass inference. `scheduler.run_games` drives several independent
`Env` handles, caps concurrency, refills finished games, groups pending neural
decisions by assigned policy into bounded chunks, routes returned indices by
chunk position (not decision id, which concurrent handles reuse), queries the
teacher through one shared cache, and commits forced choices as index zero. It
reports per-game accounting, recomputable totals, scalar-only optional traces,
and separate environment/teacher/inference timings; CUDA is synchronized at
inference batch boundaries. Any game, inference, or terminal-callback failure
stops scheduling with seed/placement/decision context.

`records` defines the plan, run state, raw terminal record, and summary
protocols. A plan fixes the seeds and six placements and carries checkpoint
content identity, installed Python source hashes, compiled Rust source,
training provenance kept separate from evaluation identity, split algorithm,
dependencies, and the selected device/batching/cache configuration. `create_run`
publishes state `incomplete`, then the plan, then the record stream; each
terminal record is appended, flushed and fsynced as its game finishes. Only a
clean close with no failed append changes the state to `complete`, and
`finalize_run` recomputes the seed-blocked summary from the persisted files.
`cli.compare` validates seeds with the Rust split, requires a selected `best`
checkpoint, publishes the plan before play, cross-checks persisted records,
writes costs while the writer context is still open, and only then closes and
finalizes, so failures leave completed records without a summary.

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
