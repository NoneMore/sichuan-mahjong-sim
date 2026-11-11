# Documentation

Start with the [project overview and quick start](../README.md). The references
below distinguish intended game rules, current implementation, operational
workflows, and research decisions still relevant to future work.

## Find the right document

| If you want to… | Start here |
| --- | --- |
| Run a seeded game | [Self-play guide](guides/self-play.md) |
| Build and check a checkout | [Development and verification](development.md#running) |
| Use the Rust `Runner` and policy caches | [Library API](library.md) |
| Resume or replay an in-memory game | [Checkpoint and replay](guides/checkpoint-replay.md) |
| Understand legal actions, scoring, or excluded variants | [Fixed gameplay rules](baseline-rules.md) |
| Trace `Game` / `Decision` / `Policy` boundaries | [Architecture overview](architecture/overview.md) |
| Compare policies or benchmark heuristics | [Policy comparison](guides/policy-comparison.md) |
| Follow collection → cache → training → evaluation | [BC pipeline](guides/bc-pipeline.md) |
| Trace Rust/Python protocol versions, owners, and compatibility | [Data contract boundaries](reference/data-contracts.md) |
| Check teacher collection or cache format details | [Neural crate guide](../crates/neural/README.md) |
| Load datasets and train an offline BC policy | [Python BC guide](../python/README.md) |
| Run frozen neural-vs-teacher comparisons | [Online evaluation guide](../online/README.md) |
| Review unimplemented research choices and how they were tested | [Experimental methods and decisions](research-decisions.md) |
| Contribute code or consult development conventions | [Developer governance](governance.md) |
