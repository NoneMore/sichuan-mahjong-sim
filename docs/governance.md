# Project governance

This is the **developer entry point** for maintaining and changing the project.
For project features, usage and general documentation, start with the
[root README](../README.md) and [documentation map](README.md).

## Developer references

| Need | Reference |
| --- | --- |
| Build, test and configure the workspace | [Development](development.md) |
| Understand implemented components and responsibilities | [Architecture](architecture/overview.md) |
| Change game behavior | [Fixed rules](baseline-rules.md) and relevant tests |
| Change Rust/Python handoffs | [Data contract boundaries](reference/data-contracts.md) |
| Run research or revisit unresolved choices | [Research methods and decisions](research-decisions.md) |
| Review active proposals or previous decisions | `specs/active/<change>/`, [archived Changes](../specs/archive/), and `docs/adr/` |

## Authority and documentation

- **Implemented behavior:** Source code is the primary evidence. Tests and
  runtime observations verify it. The [fixed rules](baseline-rules.md) define
  intended gameplay; resolve disagreements between implementation and intent
  with code inspection and focused tests. Proposals describe future behavior.
- **Code comments:** Keep local invariants, rationale, constraints and interface
  semantics near the implementation. Remove explanations already obvious from
  code; resolve material disagreements between comments and behavior.
- **Architecture:** Document implemented structure, ownership, interactions,
  component boundaries and consequential outcomes in
  [architecture guides](architecture/overview.md). Keep algorithms and
  fine-grained behavior next to code, and historical alternatives in ADRs.
- **Current baseline:** `main` must support setup, verification, understanding
  and further development using current code and guidance. Past Changes and
  agent sessions are optional historical context, not prerequisites.
- **Organization:** Use meaningful paths and filenames; avoid redundant indexes,
  metadata, and competing specifications of implemented behavior.

## Change-centered work

A **Change** is one cohesive intent, which may span tasks or pull requests.
Small fixes can be made directly without a Change.

- Use `specs/active/<change>/` for ongoing investigation or implementation
  and `specs/archive/<change>/` when the work ends, whether completed,
  cancelled or superseded. Directory location communicates lifecycle only,
  not implementation status, correctness or approval.
- Keep only useful Change files: `spec.md` for intended behavior,
  `design.md` for trade-offs and `tasks.md` for breakdowns are optional.
  Revise active specifications in place; archived material records historical
  intent and is not synchronized with later code.
- Track owners, priorities and progress in the existing issue/PR tracker.
  Use task files for durable breakdowns or dependencies rather than duplicate
  status reports. Changes need not map one-to-one to PRs.
- When work ends, promote implemented rules, tests, contracts and architectural
  outcomes to their current owners. Retain consequential reasoning and
  actionable negative findings where they are not captured elsewhere.
  Check dependencies, recovery and any evidence-retention obligations before
  removing disposable outputs or completed Change files from the working tree.

## Architecture decisions

Write an ADR in `docs/adr/` when alternatives and rationale for a consequential
architectural decision would be costly to reconstruct. Architecture documents
describe the adopted outcome; ADRs preserve the reasoning.

When a new ADR fully or partly supersedes an older one, the **new ADR**
identifies the older decision and the affected scope. Keep older ADRs unchanged;
they may move to `docs/adr/archive/`.

## Research retention

For experiment-backed choices **not incorporated** into the project, maintain
the [research decisions](research-decisions.md) with their hypothesis, controlled
alternatives, reproducible method, statistical unit, preregistered acceptance
rule, limits and remaining decision. Preserve experiment design rather than
one-off measured results. Remove an entry once the decision is implemented or
no longer actionable. Any independently required reviewable evidence is retained
through the appropriate external or historical record.

## Verification

Update the owning rules, architecture, API guide or active Change when its
contract changes. For game changes, test legality, information isolation, tile
conservation, zero-sum settlement and replay as applicable. For Rust/Python
protocol changes, update producer/consumer tests and compatibility versions.

Use the [development checks](development.md#running) for build, format, Clippy
and test commands. For performance or strategy conclusions, fix revisions,
seeds, workload, measurement conditions and acceptance criteria before
collecting results.
