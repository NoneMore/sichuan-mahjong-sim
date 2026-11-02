# Reproducible policy A/B experiment

This Change proposes the first mixed-policy experiment; it does not describe
implemented experiment tooling. The simulator's [fixed rules](../../../docs/baseline-rules.md)
and `Game` remain unchanged.

## Question and scope

Compare two A players with two B players at one table:

- **A:** import `ninone.policies.agent_heuristic.choose_action` from
  `../ninone-mahjong-agent`, covering all decision phases.
- **B:** reuse the simulator's `FirstLegalPolicy`.

The outcome is the difference in mean terminal net score per player per game,
including all settlement; a game here is one hand. Players act individually;
grouping by policy does not permit cooperation or shared private information.
Conclusions concern this
2A2B population under the simulator's rules, not general playing strength.
`FirstLegalPolicy` is a weak legality baseline.

The small heuristic makes the experiment itself the first variable to validate:
it has no randomness, model resources, or time-budget-dependent behavior. The
advanced ninone evaluator, random baselines, and other table populations are
outside this Change.

## Paired deals and balanced seats

For each preselected seed, create six fresh games, assigning A to every possible
pair of seats and B to the other two. This covers both adjacent and opposite
placements, unlike four cyclic rotations of AABB. Each policy occupies each
seat three times, balancing the fixed dealer seat within a seed block.

`Game::new(seed)` reproduces the initial wall, seat-specific hands, and exchange
direction. Rotation changes policy assignments, not the dealer mechanism;
subsequent draws and positions may diverge as actions differ. Start each game
with fresh logical policy state so execution order cannot affect its result.

Uniformly sample distinct game seeds with a seeded generator. Use a separate,
non-overlapping pilot sample to estimate both runtime and the variation of the
seed-block contrasts `D[s]` defined below. Before running the main experiment,
choose a target interval precision or an explicit runtime budget, use the pilot
to select a feasible main sample size, and fix the seed-generation procedure,
seed, and sample size. Do not include pilot results in the main estimate or
stop the main experiment when its interval first looks favorable. If the
available budget cannot meet the target precision, report that limitation.

## Experiment boundary

Do not modify `Game`, the existing `Policy` trait, or `Runner`. A thin Rust
driver uses the public `next_decision` / `submit` surface and calls B directly.
For A, it exchanges JSON Lines with a long-lived Python worker that
imports the original heuristic rather than reimplementing it in Rust.
This is a one-off external dependency of this experiment, **not** a runtime
dependency or stable cross-language policy interface of the simulator. Future
native policies should be usable without this worker or adapter.

A separate, fallible driver is preferable to forcing process failures through
`Policy::choose`, which returns only an `Action`. It owns orchestration and
error reporting, not legality, state transitions, or scoring. Those remain
exclusively authoritative in `Game`. No general policy-plugin framework is
needed for this experiment.

The A adapter sends only the heuristic's required inputs: own hand, own missing
suit, and legal candidates. Encode Wan/Tong/Tiao as `m/p/s` and tiles as suit
followed by rank (for example, `m1`), not the simulator's display strings. Use
ninone's candidate type
names, preserving each action's meaning. Keep all legal options: A's preference
for discarding over kongs and passing over peng/open kong belongs to the imported
heuristic, not an adapter restriction.

Ordinary candidates come from the pending decision; exchange candidates are
distinct same-suit tile triples satisfying its eligible suits and hand
multiplicities. Stable candidate IDs map the returned choice back to the original
Rust action. Candidate order and ID construction are part of the adapter's
identity because they can affect tie-breaking. The worker receives no opponent
concealed hands or wall order.

## Estimator and uncertainty

For seed `s` and placement `c`, let `a[s,c]` and `b[s,c]` be the mean terminal net
scores of that placement's two A players and two B players. Define:

```text
d[s,c] = a[s,c] - b[s,c]
D[s]   = mean of d[s,c] over the six placements
Delta  = mean of D[s] over the sampled seeds
```

Placements and seeds receive equal weight. Report both policy means and `Delta`
in score units per player per game. Zero-sum settlement implies the B mean is
the negative of the A mean, and `Delta` is twice the A mean; these are useful
consistency checks, not independent evidence.

The sampling unit is a seed block, not a game or a player. Report the standard
error `sample_sd(D) / sqrt(number of seeds)` and a 95% percentile bootstrap
interval by resampling whole seed blocks. Fix and record the bootstrap seed and resample
count separately from the game seeds. Do not use an independent two-sample test
or treat the six shared-deal placements as independent observations.

The interval estimates uncertainty across seeds sampled by the chosen procedure,
not replay variability: identical inputs should reproduce the same actions and
scores. A fixed corpus's measured mean itself has no replay uncertainty.

## Evidence, failures, and checks

Keep a run manifest, per-game results, and a derived summary. Record enough
to reproduce the experiment: both repositories' revisions and local changes,
adapter identity, actual seeds and their generation procedure, analysis settings,
and relevant environment details. Per-game results retain seed, placement,
terminal scores, end reason, and decision count so the summary can be recomputed
without replay. Report both policy means, `Delta`, standard error, and interval;
timing is diagnostic, not part of deterministic result identity.

Process or protocol failures, unknown selections, illegal actions, and decision
limit exhaustion invalidate the run. Preserve the failing seed, placement, and
decision context for diagnosis. Never silently substitute a baseline action,
score an incomplete game as zero, or skip a failure to publish a normal summary.
A successful result contains all six placements for every planned seed.

Before interpreting A/B results, check deterministic replay, zero-sum scores,
and complete placement coverage. Symmetric six-placement checks using
`FirstLegalPolicy` on all seats, and separately the imported Python heuristic
on all seats, should each yield zero block contrasts after seat averaging.
The first checks the Rust harness; the second exercises the worker and adapter
end to end. Neither replaces focused adapter checks for candidate completeness,
ordering/IDs, exchange triples, and the mapping of returned choices to Rust
actions.
