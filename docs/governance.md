# Project governance

Keep current guidance proportional to this fixed-rule research simulator.
[Documentation map](README.md) lists the maintained references.

## Sources of truth

- [Fixed rules](baseline-rules.md) define intended gameplay; code and tests
  establish implemented behavior. Resolve discrepancies with targeted tests.
- [Architecture](architecture/overview.md) records durable subsystem
  boundaries and implemented decisions. Keep function details near code.
- [Module guides](../crates/neural/README.md) and
  [data contracts](reference/data-contracts.md) own runtime workflows and
  Rust/Python compatibility rules.
- [Research decisions](research-decisions.md) retain experiment methods for
  supported choices that remain unimplemented.

## Changes and design rationale

Use `specs/active/<change>/` for substantial work requiring an explicit
scope or design. A Change may span multiple tasks or PRs. Write a spec,
design or task list only when it helps the work; progress belongs in the
project tracker or PR.

When work concludes, promote implemented rules, architecture and API
contracts into their current owners. Preserve enduring design rationale
when the trade-off cannot be reconstructed easily from code. Retire
completed plans and specs from the working tree once their durable
knowledge is represented in code, tests or current documentation.
Git history retains the original Change context.

Record consequential architectural trade-offs in `docs/adr/` when useful.
The current architecture guide describes the resulting system; an ADR
explains the alternatives and rationale.

## Research documentation

For an unimplemented research decision, keep:
- the hypothesis and controlled alternatives;
- the experimental method: workload, seeds, independence unit, control of
  variables, correctness checks and measurement procedure;
- the prespecified acceptance rule;
- the decision and any next experiment required.

Summarize experiments as methods and decisions, without preserving
one-off measured outputs in the maintained documentation. Remove entries
when the decision is implemented or no longer relevant.

## Verification

Update rule, architecture or protocol guidance with corresponding changes.
Verify relevant game invariants: legal actions, hidden information,
tile ownership, zero-sum settlement, deterministic replay and checkpoints.
For cross-language changes, update the corresponding producer/consumer
tests and version contracts.

Use the [development checks](development.md#running) for workspace builds,
formatting, Clippy and tests.

For new performance or strategy claims, fix the workload, build,
seed plan, measurement conditions and acceptance rule before experiments.
