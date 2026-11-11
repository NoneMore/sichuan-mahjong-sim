# Documentation

Start with the [project overview and quick start](../README.md). The references
below distinguish intended game rules, current implementation, operational
workflows, and historical measurements.

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
| Inspect historical validations or measurements | [Historical validation index](validation.md) |
| Change the project or consult past decisions | [Governance](governance.md) and [archived changes](../specs/archive/) |

For evidence precedence, documentation responsibilities, and how to resolve
disagreements, follow [project governance](governance.md#evidence-and-intent).
