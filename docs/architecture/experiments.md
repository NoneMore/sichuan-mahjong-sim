# Experiment-tool subsystem

These [standalone Rust examples](../../examples/) operate **outside** the
Game/Policy/Runner library boundary. They may choose seeds, policy placements,
cache lifecycles, and output formats; they may not redefine game rules or
alter a policy's legal-action contract. Operational commands and exact CLI
options belong in the [development guide](../development.md).

## Composition

Both tools use the production [Runner](execution.md) and the
[Policy subsystem](policy.md) rather than reaching into a privileged Game
state. Private `cli` modules validate workload inputs and `report`
modules format results; [shared cache tooling](../../examples/support/cache_tool.rs)
constructs explicit contexts and reports allocation problems.

The A/B tool separates [comparison execution](../../examples/policy_ab/comparison.rs)
from [validated persisted-record aggregation](../../examples/policy_ab/records.rs).
The benchmark separates [warmup and measured phases](../../examples/heuristic_bench/phases.rs)
from report generation. These are different evidence pipelines, not a
single mandatory framework or another source of game semantics.

## Balanced policy comparison

[`policy_ab`](../../examples/policy_ab.rs) runs a balanced 2A2B design:
each game seed is exercised with all six choices of the two A seats.
Every placement constructs fresh seat policies and a fresh Runner, so
stateful policy RNG streams do not leak between comparisons. Heuristic seats
can deliberately share one explicit analysis-cache context over the
invocation; that cache does not affect the action sequence.

Execution, persisted game records, validation, aggregation, and presentation
are separate private modules. The completed `games.tsv` rows are the
statistics input; the summary is **reconstructed from those persisted
records**, not from an independent in-memory counter. An invalid or incomplete
run must not be reported as a completed comparison.

## Heuristic benchmark

[`heuristic_bench`](../../examples/heuristic_bench.rs) measures four-seat
heuristic self-play through the production Runner. The measured games use
fixed, seed-derived input; warmup seeds precede the measured range so changing
warmup length does not move the measured seed sequence.

The tool makes cache initialization boundaries explicit: `cold` uses a
separate warmup context, while `warm` reuses the runner and cache across
warmup and measurement. Cache-status snapshots and report formatting are
outside the timed batch. Allocation degradation and phase failures must
remain visible rather than being mistaken for valid performance samples.
The precise `heuristic-benchmark-v2` protocol lives next to the example
source and its CLI options in [development](../development.md).

## Evidence and verification

Benchmark throughput on shared CI runners is not a stable performance gate.
Record revision, build/toolchain, workload, seeds, environment, and independent
repeat conditions when interpreting results. [Historical studies](../validation.md)
are evidence for their original revisions, not new measurements.

The [A/B tests](../../examples/policy_ab/tests.rs) cover placement, records,
aggregation, and failure artifacts; the
[benchmark tests](../../examples/heuristic_bench/tests.rs) cover seed and
cache lifecycle, reporting, and failure propagation. Both run under
`cargo test --all-targets`.
