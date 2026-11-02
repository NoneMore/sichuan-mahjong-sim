# Minimal native policy A/B comparison

## Intent and scope

Given two native Rust policies and a preselected set of game seeds, run balanced
2A2B games and report a reproducible mean terminal net-score difference. This
Change proposes the smallest useful comparison tool, not an experiment platform.
It is a design only; no implementation or policy-strength result is claimed.

Use the existing `FirstLegalPolicy` as A and `RandomPolicy` as B for the initial
implementation. Both are legality baselines: this Change validates the
native comparison workflow, **not policy strength**. A successful outcome is a
complete, reproducible comparison regardless of the sign or size of `Delta`.
Other native policies can later use the same orchestration through the existing
`Policy` trait; no plugin or policy registry is needed.

A game means one complete hand, including settlement under the
[baseline rules](../../../docs/baseline-rules.md). Players act individually;
policy groups neither cooperate nor share private information. Results describe
this 2A2B population, not general playing strength.

## Implementation boundary

Add a small standalone Cargo example, `examples/policy_ab.rs`, keeping this
comparison separate from the existing self-play CLI. Construct four seat policies
as `Box<dyn Policy>` and use `Runner::new(...).run_new(seed)` for each game.
A fresh runner per game avoids `run_batch`'s cross-game policy RNG continuation.
Reuse its legality checks, decision limit, and terminal reporting.
Do not change `Game`, `Policy`, `Runner`, or the fixed rules. Policies see only
`Decision`; the tool must not supply hidden hands or wall order to policies.

Inputs are an explicit nonempty list of distinct `u64` game seeds, a policy RNG
seed, a positive per-game decision limit, and a fresh output directory. Reject
invalid inputs before running. The actual seed list and run size are fixed
before execution; do not stop early because the result looks favorable.
Input syntax and output serialization should stay simple and local to the
example, with no configuration framework or new library API. Do not create
reusable experiment-management, reporting, persistence, or error-handling
subsystems for this Change. If the implementation grows beyond a small example
and focused tests, remove optional capabilities before introducing abstractions.

## Balanced games and policy state

For each game seed, start six fresh games with A in these seat pairs and B in
the remaining seats, in this order:

```text
[0,1]  [0,2]  [0,3]  [1,2]  [1,3]  [2,3]
```

This covers adjacent and opposite pairs and puts each policy in every seat
three times. The seed fixes the initial deal and exchange direction; changing
policies may change subsequent draws. Do not rotate tiles or change the dealer.

Construct fresh policies and a fresh runner for every placement. For the
initial random baseline, derive each B seat's RNG seed with the existing
`policy::random_seat_seed(game_seed ^ policy_seed, seat)`. A given seat starts
from the same random stream whenever it is assigned B within a seed block;
streams do not continue between games or depend on loop order. Record the
derivation and policy seed. This is deterministic coupling, not a claim of
independent RNG streams. Future stateful policies must likewise start from
explicit recorded initial state, rather than sharing mutable policy instances
across seats.

## Results and statistics

Write one raw record per completed game containing the game seed, A-seat pair,
four terminal net scores, end reason, and decision count. Write records as games
finish so failures retain partial diagnostic results. These records, together
with the recorded inputs, must suffice to recompute the summary without replay.
Keep aggregation and record validation in local functions, tested with fixture
records; a separate summarize command or workflow stage is not required.

For seed `s` and placement `c`, let `a[s,c]` and `b[s,c]` be the average terminal
net scores of the two A players and two B players respectively:

```text
A_mean = sum of a[s,c] over all seeds and placements / (6N)
B_mean = sum of b[s,c] over all seeds and placements / (6N)
d[s,c] = a[s,c] - b[s,c]
D[s]   = mean of d[s,c] over the six placements
Delta  = mean of D[s] over the N game seeds = A_mean - B_mean
SE     = sample_sd(D) / sqrt(N), for N >= 2
```

Report seed/game counts, both policy means, each seed's `D[s]`, `Delta`, and
`SE`, in score units per player per game. With one seed, report the
mean but mark `SE` unavailable, not zero. Give seeds and placements equal weight.
The sampling unit is the complete seed block: do not count its six games or
individual players as independent observations. Across-seed uncertainty is not
replay variability. For a fixed or hand-picked seed corpus, `SE` describes
variation among its seed-block contrasts; without a justified random-sampling
procedure it must not be presented as evidence of population-level precision,
significance, or general policy superiority. No confidence interval or
significance test is required here.

## Minimal reproducibility and failure handling

Record the simulator revision and whether the working tree is dirty, policy
names and relevant parameters, actual game seeds, policy seed/derivation, and
decision limit in a small run record before play. Record toolchain information
and the invocation for practical reruns. If revision, dirty status, or toolchain
metadata cannot be obtained, record `unknown` with a warning rather than adding
a hard environment prerequisite. A dirty marker is not a source snapshot:
exact source reproduction remains the caller's responsibility. No fingerprints,
source archives, or artifact authentication are required.

A successful run has exactly one result for every planned seed/placement and
zero-sum scores in each game. Summarization must reject missing, duplicate,
unexpected, or invalid records. Each record must identify a planned game seed
and one of the six A-seat pairs, contain exactly four signed integer terminal
scores whose sum is zero, a supported terminal end reason, and a positive integer
decision count no greater than the configured limit. Seed and decision-count
fields must fit `u64`, and scores must fit the runner's `i32` score type. Reject
malformed or out-of-range fields rather than coercing them into valid records.
An illegal action, decision-limit exhaustion,
missing terminal result, output error, or failed consistency check stops the run
with a nonzero exit and no normal summary. Report the failing seed/placement
when applicable; retain completed records for diagnosis. Never substitute an
action, score an incomplete game as zero, skip failed games, or silently resume
a partial run. Refuse to overwrite an existing run directory.

## Focused verification and acceptance

Use unit tests inside the example (`#[cfg(test)]`) for orchestration, aggregation,
and failure fixtures; no additional executable or library test API is needed.

- A small seed corpus produces exactly six placements per seed and valid
  zero-sum terminal scores; the summary recomputes from the raw records.
- Identical game seeds, policy seed, policies, and parameters reproduce scores
  and decision traces. Test traces may use `Runner` reports; persisted traces
  and trace hashes are not required.
- Running placements in a different order leaves their results unchanged.
- Using `FirstLegalPolicy` on all four seats yields `D[s] = 0` for each seed
  after averaging all six placements; each seat is labeled A exactly three
  times. Individual placement contrasts `d[s,c]` need not be zero. This checks
  balanced labeling and aggregation, not whether different policies should tie.
- Hand-calculated fixtures verify block means and sample standard errors,
  including unavailable `SE` for one seed.
- Targeted failure tests cover rejected actions, an unfinished game at the
  decision limit, incomplete/duplicate result coverage, and output failures.
  None may produce a normal summary.
- Implementation must pass the standard checks in
  [project governance](../../../docs/governance.md); this design-only Change
  needs link, scope, and diff checks rather than simulator execution.

## Explicitly deferred

No new strategic heuristic, old `ninone` integration, Python worker, cross-language
adapter, automatic pilot/precision/runtime planning, multi-stage run workflow,
bootstrap intervals, provenance/manifest audit system, artifact tamper detection,
trace hashing, parallel execution, or performance/strategy-quality claims.
Introduce any of these separately only when a concrete experiment needs it.
