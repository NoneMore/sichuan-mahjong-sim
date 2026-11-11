# Experiment-tool subsystem

These [standalone Rust examples](../../examples/) operate **outside** the
Game/Policy/Runner library boundary. They may choose seeds, policy placements,
cache lifecycles, and output formats; they may not redefine game rules or
alter a policy's legal-action contract. Operational commands and exact CLI
options belong in the [policy comparison guide](../guides/policy-comparison.md).

## Composition

Both tools use the production [Runner](execution.md) and
[Policy](policy.md) boundary, not privileged Game state. Shared support
constructs explicit cache contexts; the examples own experiment lifecycles,
record validation and reporting, not gameplay rules.

## Balanced policy comparison

[`policy_ab`](../../examples/policy_ab.rs) compares policies across
balanced seat placements. Every placement uses fresh policies and a new
Runner, preventing RNG state from carrying across comparisons; only explicit
heuristic memoization may outlive a placement.

The statistical summary is reconstructed from validated persisted
`games.tsv` records, not a separate in-memory counter. An incomplete run
cannot become a completed comparison. Exact placements, seeds and report
formats are specified in [policy comparison](../guides/policy-comparison.md).

## Heuristic benchmark

[`heuristic_bench`](../../examples/heuristic_bench.rs) measures
four-seat heuristic self-play with explicit cache lifecycles: cold uses a
separate warmup context; warm retains the runner and context. This determines
which initialization work is charged to the measured phase. Cache snapshots
are outside timing, and allocation degradation remains observable.
See [policy comparison](../guides/policy-comparison.md#heuristic-benchmark-example) for seeds,
CLI options and report fields.

## Evidence and verification

Benchmark throughput on shared CI runners is not a stable performance gate.
Record revision, build/toolchain, workload, seeds, environment, and independent
repeat conditions when interpreting results. Old benchmark numbers are not a
performance baseline for the current revision.

The [A/B tests](../../examples/policy_ab/tests.rs) cover placement, records,
aggregation, and failure artifacts; the
[benchmark tests](../../examples/heuristic_bench/tests.rs) cover seed and
cache lifecycle, reporting, and failure propagation. Both run under
`cargo test --all-targets`.
