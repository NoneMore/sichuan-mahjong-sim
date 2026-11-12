# Neural adapters

The core Game owns rules, legal actions, hidden state and settlement.
Neural tooling consumes player-visible decisions through separate Rust and
Python packages. The dependencies are one-way: the core builds without the
neural crate or Python extension.

[Run the complete BC workflow](../guides/bc-pipeline.md) ·
[Cross-language contracts](../reference/data-contracts.md)

## Decision encoding

The Rust [encoder](../../crates/neural/src/encoding/mod.rs) converts a
player-scoped Decision into deterministic state features and canonically
ordered legal candidates. Feature meanings, scales, enumeration and
tie-breaks are versioned together; changing tensor width alone does not
define compatibility. The encoder neither simulates transitions nor exposes
teacher analysis or concealed opponent information.

## Teacher collection

The collector drives Game and HeuristicPolicy directly, capturing the
original decision and accepted teacher action. It records every seat,
stage, forced choice and losing trajectory.

An immutable corpus commits complete JSONL episode shards through an
OS-locked writer and atomic manifest updates. The manifest fixes source and
encoding identities; a seed-only split remains stable as the corpus grows.
Append resumes at the first uncommitted seed, and strict readers verify
shards while streaming episodes.

[Collection operations](../../crates/neural/README.md#teacher-data-collection)

## NumPy preparation

Preparation pins exact committed manifest bytes and shard membership in a
CorpusSnapshot. A later append cannot silently change an existing prepared
dataset. Streaming verification precedes atomic publication of bounded
episode-aligned NumPy shards. dataset.json binds the array inventory,
split membership, checksums and distinct collection/preparation provenance.
Prepared caches can be verified without retaining the source corpus.

[Preparation operations](../../crates/neural/README.md#dataset-preparation)

## Offline Python data and training

The independent [mahjong-bc package](../../python/README.md) validates the
Rust-produced cache and preserves its opaque cache_id. Packed candidates
remain in canonical order; batch collation pads each batch and marks
padding with a mask. Read-only memory-mapped loading uses num_workers=0.

Public samples, `DecisionBatch`, collation, `iter_batches()` and batch device
moves retain their padded layouts and complete contracts. Internal training
batches gather model inputs and targets; evaluation adds stages and target
kinds. Neither path gathers provenance or an eligible flag. Host gathering
checks targets and current-offset training eligibility before loss computation.

With `num_workers=0`, exact built-in packed datasets with target-kind metadata
and an unmodified `CandidatePolicy` use flat valid-candidate rows. Only those
rows reach the scorer; logits are restored to the padded layout for the
existing loss and metrics.

Custom datasets, dataset subclasses, missing target-kind metadata and workers
retain full batches. Policy subclasses, `forward`/`score` overrides, compiled
calls and policy/global hooks retain padded scoring so public calls are not
bypassed. Evaluation retains full traversal when current offsets no longer
match cached counts; all-forced built-in splits use metadata without scoring.

The policy embeds the state once and scores each legal candidate with a
shared nonlinear network; candidate position does not enter the model.
Training visits every eligible decision each seeded epoch, uses
equal-weight imitation loss and reports forced decisions separately.

Validation NLL selects best.pt. last.pt captures the last completed epoch,
optimizer, traversal and RNG states for recovery; partial epochs replay
after interruption. Resume checks dataset, configuration, device and
dependency identities. Checkpoints publish atomically and carry model,
encoding and source provenance.

[Training and checkpoint contract](../../python/README.md#training-metrics-and-checkpoints)

## Online environment and comparison

A separate [PyO3 workspace](../../crates/python/) provides an Env per
Game. Reset/step expose owned, player-scoped encoded snapshots; cached
canonical actions and generation-aware decision IDs reject stale choices.
Teacher projection uses the original Rust Decision and submits semantic
actions through Game.

The [online Python package](../../online/README.md) batches frozen inference
across games while keeping each decision associated with its originating
environment handle. Forced choices and post-Hu continuation use the same
decision protocol. A comparison publishes its summary only after all
terminal records and costs have been persisted successfully. Training
checkpoint identity, executing Python source and compiled Rust provenance
are recorded separately.

## Contract tests

Rust [encoding](../../crates/neural/tests/encoding_contract.rs),
[collection](../../crates/neural/tests/collection_contract.rs) and
[preparation](../../crates/neural/tests/prepare_contract.rs) tests cover
legality, isolation, versioned handoffs, immutable publication and recovery.
The [Python tests](../../python/tests/) and
[online tests](../../online/tests/) cover loading, checkpoints, binding,
replay, and comparison records.
