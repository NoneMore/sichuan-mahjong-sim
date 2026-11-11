# Cross-language data contract boundaries

This page helps developers trace contracts between the Rust simulator,
neural data pipeline, offline Python model, and online evaluator. It is a
**navigation and compatibility map**, not a second wire-schema definition.
Implementation, parser validation, and contract tests govern actual behavior;
[fixed game rules](../baseline-rules.md) separately define intended gameplay.

## Contract handoffs

| Boundary | Producer → consumer | Versions and identity | Canonical implementation |
| --- | --- | --- | --- |
| Player-visible decision encoding | Rust `Decision` → Rust collector, Python model, online PyO3 binding | Encoding v1; 1,013 state features and 39 per candidate; semantic ordering is versioned | [Rust encoding schema](../../crates/neural/src/encoding/schema.rs), [encoding contract tests](../../crates/neural/tests/encoding_contract.rs) |
| Teacher corpus | `bc-collect` → `bc-prepare` | JSONL/manifest format v1; fixed source revision, encoding version, seed split, immutable listed shards and checksums | [Collection protocol](../../crates/neural/src/collection/mod.rs), [manifest](../../crates/neural/src/collection/manifest.rs), [collection tests](../../crates/neural/tests/collection_contract.rs) |
| Prepared NumPy cache | `bc-prepare` → `mahjong-bc` | `dataset.json` and nine-array shards; preparation format v1, `cache_id`, split/shape/checksum inventory | [Rust preparation protocol](../../crates/neural/src/prepare/mod.rs), [cache manifest](../../crates/neural/src/prepare/manifest.rs), [Python constants](../../python/mahjong_bc/constants.py) |
| Offline checkpoint | `mahjong-bc` training → offline evaluation and `mahjong-eval` | Checkpoint container v3; model format v1, encoding v1, role (`best`, `last`, `policy`); format 2 remains readable but not resumable | [Checkpoint loader and writer](../../python/mahjong_bc/checkpoint.py), [constants](../../python/mahjong_bc/constants.py) |
| Online environment snapshot | Rust `Game`/encoder → PyO3 `Env` → frozen inference | Owned canonical candidate snapshots; decision IDs protect against stale submissions; encoder version must match model | [PyO3 boundary](../../crates/python/src/lib.rs), [online guide](../../online/README.md#environment-and-teacher-api) |
| Online comparison records | `mahjong-eval` → persisted review and summary | Plan, terminal records, costs, completion state; summaries recomputed from persisted records | [Record implementation](../../online/mahjong_eval/records.py), [online guide](../../online/README.md#records-and-provenance) |

The corpus split uses `seed-sha256-80-10-10-v1`. Encoding compatibility
includes feature meanings and candidate order; cache identity includes the
captured manifest; checkpoint roles distinguish model selection from recovery.

## When a protocol changes

1. Change the producer and consumer deliberately. Update version markers when
   semantics or compatibility require it; changing only dimensions or
   Markdown text cannot preserve wire compatibility.
2. Update the relevant [contract tests](../../crates/neural/tests/),
   [offline Python tests](../../python/tests/), or
   [online evaluation tests](../../online/tests/) and validate real handoffs.
3. Update the owning crate/package README and this navigation table if the
   contract boundary or compatibility policy changed.

For runnable commands use the [BC pipeline](../guides/bc-pipeline.md); for
implemented component ownership use the [neural architecture](../architecture/neural.md).
