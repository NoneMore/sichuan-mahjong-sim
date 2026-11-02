# Shanten-and-ukeire policy: design rationale

The [specification](spec.md) defines this Change's intended behavior, scope,
acceptance conditions, and unresolved delivery decisions. This document explains
why that direction was chosen and how to keep its implementation proportionate.
Neither document describes an implemented policy or establishes a strength claim.

## Evidence and opportunity

The current [`Policy` boundary](../../../src/policy.rs) supplies a borrowed
`Decision`. `FirstLegalPolicy` exchanges low-ranked tiles from the eligible suit
with the fewest tiles, then otherwise takes Hu or the first legal action. With
[`Game`](../../../src/game.rs)'s current ordering, this means declaring Wan
missing regardless of the hand, discarding by tile encoding, taking any available
peng, and prioritizing concealed/added kongs. `RandomPolicy` also prefers Hu but
otherwise samples legal actions. Neither preserves useful hand structure.

[Historical validation](../../../docs/validation.md) reports predominantly
wall-exhaustion endings in their self-play. This motivates, but does not prove,
an opportunity to improve basic progress rather than begin with deep search.
Big-call payments and kong refunds make reaching tenpai useful even without a
win. The fixed fan system adds points directly and caps the result at five,
rather than applying exponential multipliers, weakening the case for sacrificing
speed to chase speculative high-fan hands.

## Why lexicographic shanten and ukeire

| Alternative | Consequence | Decision |
| --- | --- | --- |
| Local scores for pairs, partial sequences, and isolated tiles | Cheapest, but overlapping structures can be counted twice and feature weights require tuning | Reject as the primary evaluator |
| Shanten plus ukeire | Resolves competing decompositions and separates progress from estimated availability | Choose |
| Rollouts, opponent-hand inference, or multi-step lookahead | Potentially stronger, but adds runtime, state, and tuning beyond the selected budget | Defer |

Lexicographic ranking is interpretable and avoids tuning exchange rates between
progress, availability, and fan. It is a proxy for net score, not a full
expected-value model. The late fan tie-break captures some exhaustive-draw value
without encouraging speculative detours.

Keeping structural distance separate from availability is essential: an exhausted
wait can still qualify for big-call settlement. Likewise, unseen counts are not
wall probabilities because opponents, including already-won players, retain
unrevealed concealed tiles. These distinctions explain the separate quantities
and ownership rules in the specification.

## Why simple opening choices and conservative calls

Retaining baseline exchange avoids estimating unknown incoming tiles and reduces
the number of strategic changes in the initial comparison. Fewest-tile missing
suit addresses the baseline's constant declaration without introducing a second
structural evaluator. During forced clearing, all eligible discards preserve the
same non-missing-suit hand, so repeated analysis buys nothing.

A peng can destroy flexible structures or a seven-pairs route; requiring actual
distance improvement discourages calls for their own sake. Kongs have immediate
income, motivating a less restrictive progress gate, but that gate is not an
expected-value calculation. It omits replacement-draw lookahead and robbery
probabilities, and income may be refunded. It can reject profitable kongs or
accept unprofitable ones; refund exposure alone does not establish negative
expectation.

## Implementation leverage and cost

The existing policy seam is sufficient: keep analysis behind one independent
policy rather than introducing a framework or a second rule authority. Existing
pure helpers in [`eval`](../../../src/eval.rs) recognize wins and waits but do
not compute shanten or ukeire. Reuse those facts and existing fan helpers where
useful. Keep new analysis private unless actual reuse earns a shared interface.

A suitable small implementation uses counted hands and memoized decomposition
subproblems, sharing work among candidates within a decision. Analyze distance
first, expand improving tile kinds only for minimum-distance candidates, and
calculate the fan tie-break only when needed. Short-circuit the simple decisions
identified in the specification. Keep memoization local or bounded rather than
depending on a large precomputed table. Exact decomposition and cache layout are
implementation choices, not additional public interfaces. A reference project's
target-deficit evaluator may inspire an algorithm, but its distance values or
special-case compatibility behavior do not redefine the structural shanten
contract in this independent simulator.

The A/B runner constructs fresh policies per game, so cross-game warming cannot
be assumed. Cache-independent decisions avoid adding logical checkpoint state.
Numerical performance expectations must come from profiling: the legal baselines
are exceptionally cheap, and historical throughput is not a runtime budget.

## Why this comparison and what could change the direction

The existing [`A/B example`](../../../examples/policy_ab.rs) already provides
balanced placements and seed-block aggregation. Its block-level standard error
accounts for within-seed dependence. Reusing it avoids building an experiment
platform. Its local style factory can construct the new policy, but
its current run configuration hard-codes first versus random; the remaining
selection-surface decision is recorded in the specification.

A missing-suit-only ablation can later separate gains from improved suit
selection and gains from hand analysis; it is useful but not a prerequisite to
implementing the policy. In balanced zero-sum 2A2B games, `Delta = 2 * A_mean`,
so the group means are not independent samples; common seeds couple initial
conditions but not necessarily draw trajectories. The existing seed-block
statistics support exploratory comparison. A confirmatory improvement claim
requires an independently planned corpus and inference threshold, not merely a
favorable development result.

The existing A/B records contain scores, finish reasons, and decision counts;
they do not preserve per-seat winners or end-of-wall tenpai/flower-pig facts.
`RunReport` has winner flags, but collecting win or settlement diagnostics needs
an explicit narrow addition to the experiment output. These are optional until
a concrete hypothesis requires them. Release throughput and policy decision
cost should still be measured: shorter games can offset more expensive
decisions.

Revisit defense or richer kong decisions only if diagnostics show losses
materially canceling progress gains. A disappointing result is evidence for a
new design decision, not automatic permission to expand the current scope.
