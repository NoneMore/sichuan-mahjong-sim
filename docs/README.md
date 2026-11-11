# Documentation

Start with the [project overview and quick start](../README.md). The references
below distinguish intended game rules, current implementation, operational
workflows, and historical measurements.

## Find the right document

| If you want to… | Start here |
| --- | --- |
| Run a seeded game or check a checkout | [Development: running](development.md#running) |
| Use a Rust `Runner`, policy cache, or checkpoint | [Library API and continuation](library.md) |
| Understand legal actions, scoring, or excluded variants | [Fixed gameplay rules](baseline-rules.md) |
| Trace `Game` / `Decision` / `Policy` boundaries | [Architecture overview](architecture/overview.md) |
| Compare policies or benchmark heuristics | [Development: policy experiments](development.md#policy-experiments) |
| Collect teacher data and prepare NumPy caches | [Neural crate guide](../crates/neural/README.md) |
| Load datasets and train an offline BC policy | [Python BC guide](../python/README.md) |
| Run frozen neural-vs-teacher comparisons | [Online evaluation guide](../online/README.md) |
| Inspect historical validations or measurements | [Historical validation](validation.md) |
| Change the project or consult past decisions | [Governance](governance.md) and [archived changes](../specs/archive/) |

For evidence precedence, documentation responsibilities, and how to resolve
disagreements, follow [project governance](governance.md#evidence-and-intent).
