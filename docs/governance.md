# Project governance

This is the canonical, human-readable guide to project governance for contributors
and coding agents. Keep it proportionate to this fixed-baseline Rust research
simulator; create documents and process only when they serve a concrete need.
If agent instructions become useful, keep `AGENTS.md` a short pointer here,
with genuinely agent-specific guidance kept separate.

## Evidence and intent

- **Implemented behavior:** Checked-in Rust code is primary evidence of what the
  repository implements. Tests and runtime observations help verify it; docs,
  including comments, cannot override implementation evidence. Proposals may
  describe future behavior rather than the current system.
- **Intended rules:** [Baseline rules](baseline-rules.md) defines the fixed
  Sichuan mahjong rules. If code and rules disagree, distinguish actual behavior
  from intended behavior, inspect tests, and resolve or report the discrepancy.
  Deliberate rule changes require an explicit decision, updated rules, and
  focused regression tests.
- **Comments:** Preserve non-obvious local invariants, rationale, constraints,
  and interface semantics near the code. Avoid restating obvious behavior or
  duplicating architecture documentation; resolve or report material conflicts.
- **Current baseline independence:** The `main` branch must support ordinary
  setup, verification, understanding, and follow-on development from current
  code, tests, and development or architecture guidance. Completed or
  in-flight Changes and past agent sessions are not prerequisites. Keep
  durable implemented constraints in their current authorities; consult
  archived Changes only for specific historical rationale or evidence.

## Where to look

| Question | Start here |
| --- | --- |
| Setup, build, run, test | [README quick start](../README.md#running), then [development guide](development.md) for checks, CLI and CI |
| Fixed game semantics | [Baseline rules](baseline-rules.md) |
| Current implementation | Relevant `src/` code, `tests/`, and executable evidence |
| Component boundaries | [Current architecture](architecture/overview.md), then code |
| Past test coverage and measurements | [Validation](validation.md) |
| Proposed change intent | `specs/active/<change>/`, when present |
| Consequential decision history | `docs/adr/`, when present |

## Architecture knowledge

[Current architecture](architecture/overview.md) is the subsystem index;
its linked guides summarize the **current implemented high-level structure**,
responsibilities, major component interactions and boundaries, useful code
entry points, and **architectural outcomes of implemented decisions**.
README **Architecture** is the short entry. Keep intended gameplay in
[baseline rules](baseline-rules.md), distinct from the rule foundation's
implementation architecture.

Consult code for current behavior and function-level detail. Keep the
architecture guides aligned with implemented changes, not proposals; label
future, non-prescriptive design directions separately. ADRs preserve
consequential alternatives and historical reasoning, while architecture
records only implemented outcomes. Change-scoped design may inform an
architecture update after implementation without becoming a second current
authority.

Choose meaningful paths and document names for their subject and role; do
not replicate that metadata in extra indexes or require uniform documents
for every module. Create or split subsystem guides only when useful.

## Change-centered work

A **Change** is a cohesive, bounded intent, not a task, PR, mandatory phase, or
sub-Change. It may span tasks or PRs; trivial work does not need one.

- Put useful working material in `specs/active/<change>/`; move it to
  `specs/archive/<change>/` when work is no longer advanced (including
  completion, cancellation, or supersession). Create no empty placeholders.
- Use descriptive paths, avoid redundant indexes or metadata, and keep only
  helpful files, such as `spec.md` for intended
  behavior, `design.md` for trade-offs, or `tasks.md` for breakdowns. None
  is mandatory. These are change-scoped materials, not permanent authorities
  for implemented behavior.
- Revise active specifications in place as requirements settle. Archived
  materials remain historical and need not track later code. Neither directory
  implies approval, implementation, validity, or continued applicability.
- Keep owners, priorities, and progress in an existing tracker when available;
  avoid duplicating volatile state in `tasks.md`. Git, tests, and PRs supply
  implementation and verification evidence.
- For substantial investigations, identify scratch outputs and likely retention
  needs early when useful. When work stops, consider only the process artifacts
  created or modified by that Change: promote durable code, tests, and current
  knowledge; preserve useful positive and negative findings, limitations, and
  provenance when not otherwise captured; retain required external evidence
  with appropriate access, integrity, and retention; and discard disposable
  outputs only after checking dependencies and obligations. Trivial work needs
  no special review or additional file.

Do not require one Change per PR, specifications for every PR, or
archive-before-merge gates without separately adopting that stricter workflow.
This guide adds no direct-push or PR approval rules.

## Architecture decision records

Use `docs/adr/` only when an architectural decision's alternatives and
rationale would be costly to reconstruct. Record context, choice, trade-offs,
and consequences; routine fixes need no ADR.

ADRs preserve **why**, not a second registry of current architecture. Reflect
implemented outcomes in [current architecture](architecture/overview.md)
without copying historical rationale. When an ADR fully or partially
supersedes an older one, the **new** ADR identifies the old decision and
its affected scope;
do not edit the old ADR to point forward. Older ADRs may move to
`docs/adr/archive/` while remaining available as history.

## Documentation and verification

- Update current rules, architecture, or public API explanations when the
  implementation affects them; update active Change materials when their
  intent changes. Unrelated work needs no documentation sweep.
- For Game or rule changes, test affected legality, transitions, settlement,
  and applicable invariants: hidden-information isolation, tile conservation,
  zero-sum scores, seeded replay, and clone/continuation. For Policy/Runner
  changes, check observation boundaries, legal actions, seeds, and policy
  checkpoint semantics as relevant.
- Standard checks: `cargo fmt --all -- --check`,
  `cargo clippy --all-targets -- -D warnings`, and
  `cargo test --all-targets`. Add release, stress, or performance checks
  when warranted.
- Treat [validation measurements](validation.md) as historical observations,
  not a current performance guarantee. Record build, environment, seeds,
  policy, workload, and results for comparisons; measure before claiming
  performance or strategy quality.
