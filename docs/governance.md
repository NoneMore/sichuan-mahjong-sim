# Project governance

This is the canonical, human-readable guide to how this repository records
implementation knowledge, architectural decisions, and proposed changes. It
applies to human contributors and coding agents alike. Keep the practice
proportionate to this small, fixed-baseline Rust research simulator; do not
introduce documents or approval steps merely for uniformity.

## Evidence, intent, and authority

- **Code first for implemented behavior.** The checked-in Rust implementation
  is primary evidence of what the repository implements. Tests, builds, and
  actual runtime measurements help verify behavior; documentation (including
  comments) does not override implementation evidence. A proposal can describe
  a future target rather than existing functionality.
- **The fixed rule baseline expresses intended behavior.**
  [`baseline-rules.md`](baseline-rules.md) specifies the chosen Sichuan
  mahjong rules, including legality and settlement. When it differs materially
  from the code, distinguish observed behavior from intended behavior. Check
  the relevant tests and decide whether the implementation is defective or a
  rule change is genuinely intended; do not silently treat either side as
  proof of correctness. A deliberate rule change needs an explicit decision,
  updated rule documentation, and focused regression tests.
- **Preserve research invariants.** Changes affecting game semantics or the
  decision boundary should explicitly consider legal actions, hidden-information
  isolation, tile conservation, zero-sum settlement, seed reproducibility, and
  clone/continuation behavior, as applicable. Do not promise performance or
  strategy quality without measurements.

Comments are code-adjacent documentation for non-obvious invariants, rationale,
constraints, and interface semantics. Avoid narrating obvious code or copying
high-level architecture into comments. If code and comments disagree, report or
resolve the discrepancy within the task's scope.

## Where to look

| Question | Start here |
| --- | --- |
| What is the project and how do I build, run, or test it? | [README](../README.md), especially **Running** |
| What rules are intended for the fixed baseline? | [Baseline rules](baseline-rules.md) |
| What does the simulator currently implement? | Relevant `src/` code, `tests/`, and executable evidence |
| Where are the major component boundaries? | The README **Architecture** section, then the relevant code |
| Why was the initial design chosen? | [Initial design](../xzd-rust-simulator-design-revised.md) |
| What was verified or measured previously? | [Validation record](validation.md); re-run checks for the current revision |
| What is a proposed, bounded change? | Applicable materials under `specs/active/`, if present |
| Why was a consequential architecture choice made? | Relevant decision under `docs/adr/`, if present |

The existing [implementation plan](implementation-plan.md) describes the
initial implementation slices; the [integration contract](integration-contract.md)
describes the initial Game/Runner coordination boundary. Read them when that
history is relevant, but do not mistake initial plans or historical agreements
for an automatically up-to-date description of current code.

Use the entry that fits the question. No contributor or agent needs to read
every document or every ADR before routine work. Reading order is not an
authority ranking.

## Architecture knowledge

The README currently provides the implemented system's high-level entry:
`Game` owns authoritative state, legality and settlement; `Policy` selects
actions from a restricted `Decision`; and `Runner` drives games and measures
execution. Consult code for details.

When a more substantial navigation guide is useful, maintain
`docs/architecture/` to describe the **current implemented** structure,
responsibilities, boundaries, and code pointers. Clearly label future-facing
design directions as such. Reflect the architectural outcome of implemented
decisions there (or in the README while it remains the entry point), without
duplicating decision rationale or fine-grained function behavior.

## Change-centered work

A **Change** is a cohesive, bounded intent to alter the project. It is not a
task, a pull request, a mandatory lifecycle phase, or a child of another
Change. One Change may span several tasks or PRs. Independent intentions may
use separate Changes; trivial edits do not need one.

When change-scoped working materials would clarify intent, use:

- `specs/active/<change>/` while the change is being advanced.
- `specs/archive/<change>/` when it is no longer being advanced, whether
  completed, cancelled, or superseded.

Use clear, descriptive names. Create directories only when there is real
material to put in them; this repository does not require empty directory
placeholders. A Change has **no mandatory files**. Depending on the work, it
may contain `spec.md` for intended behavior and acceptance criteria,
`design.md` for alternatives or constraints, or `tasks.md` for a useful
breakdown. These are **change-scoped proposals**, not permanent fine-grained
authorities for what the simulator currently does.

Revise an active specification in place as its requirements settle. Archived
Change material is preserved as history; it is not synchronized to later code.
Neither `active` nor `archive` implies approval, implementation, validity,
or ongoing applicability. Git history, tests, and PRs provide implementation
and verification evidence. If Issues or another tracker is used, keep owners,
priority, and progress there; do not duplicate volatile status in `tasks.md`.

Do not impose one-Change-per-PR, mandatory specs for every PR, or
archive-before-merge gates unless the project explicitly adopts that stricter
workflow separately. Direct pushes and PRs are not given new approval rules
by this document.

## Architecture decision records (ADRs)

Create an ADR under `docs/adr/` only for a consequential architectural
choice whose context, alternatives, and reasoning would be expensive to
reconstruct. Record the decision, its motivation, trade-offs, and consequences.
A routine bug fix or local refactor does not require an ADR. The existing
initial design document remains valuable historical context; do not
retroactively split it into ADRs without a concrete need.

ADRs preserve **why a decision was made**, not a second registry of the
system's current architecture. Once implemented, update the appropriate
architecture entry to reflect the resulting design, without duplicating
historical rationale. When a new ADR fully or partially supersedes an old one,
the **new** ADR names the older decision and the affected scope. Do not edit an
older ADR to add forward-pointing status fields. Older decisions may move to
`docs/adr/archive/` when useful, preserving history.

## Keeping documentation and verification aligned

- Update the relevant rules, architecture entry, public contract explanation,
  or active Change material **when a change actually affects it**. Unrelated
  implementation work does not mandate a documentation sweep.
- For rule or Game changes, add focused tests for affected legality,
  transitions, scoring, or invariants. For Policy/Runner changes, check the
  observation boundary, legal-action handling, seeded replay, and checkpoint
  semantics where relevant.
- The README documents the normal local workflow. Current checks include
  `cargo fmt --all -- --check`,
  `cargo clippy --all-targets -- -D warnings`, and
  `cargo test --all-targets`. Use additional release, stress, or performance
  checks when the risk of a change warrants them.
- Treat [past validation measurements](validation.md) as observations for a
  recorded environment, **not** guaranteed results for later revisions or a
  universal performance threshold. For comparative measurements, record the
  build, environment, seeds, policy, workload, and actual outcomes.

This governance guide organizes work; it does **not** change existing mahjong
rules, Rust APIs, toolchain support claims, tests, or repository branch
protection settings.
