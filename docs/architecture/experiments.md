# Experiment tools

The standalone [policy_ab](../../examples/policy_ab.rs) and
[heuristic_bench](../../examples/heuristic_bench.rs) examples use public
Game/Policy/Runner interfaces. They own experiment seeds, placement,
cache lifecycles and reporting, while Game and Policy retain their
normal responsibilities.

## Paired policy comparisons

policy_ab plays six balanced 2A2B placements per seed with fresh policy
instances for each game. It computes seed-block statistics from validated
games.tsv records and publishes a summary only for complete runs.
Heuristic seats share the explicitly configured cache context.

## Heuristic benchmarking

heuristic_bench runs four-seat heuristic self-play. A cold measurement
gets a fresh cache after warmup; a warm measurement retains the cache.
Reports distinguish game construction, policy work and rule work, with
cache counters collected outside the timing interval.

Run multiple controlled, paired trials when comparing performance.
Document build, environment, workload, seeds and cache lifecycle.

[Commands and output files](../guides/policy-comparison.md) ·
[Experiment tests](../../examples/policy_ab/tests.rs)
