# Shanten-and-ukeire policy: specification

## Outcome, scope, and status

Provide a deterministic native `HeuristicPolicy` intended to improve mean
terminal net score per player per game against the existing legal baselines,
using shanten and ukeire rather than simulation. Win rate is secondary. The
[design](design.md) records the evidence and trade-offs behind this direction.
This is a prospective contract, not an implemented feature or a measured result.

The scope includes the policy, private hand analysis, and comparison through the
existing native A/B example. Preserve both baseline policies and the existing
`Game`, `Policy`, and `Runner` contracts. Do not change the
[fixed rules](../../../docs/baseline-rules.md).

Do not include stochastic simulation, opponent-hand inference, multi-step search,
weighted feature tuning, sophisticated exchange optimization, high-fan route
planning, or a general experiment framework. The hypothetical draw/discard
calculation below is hand analysis, not permission for further search.

Core behavior is settled below. A/B policy selection remains an integration
choice, but numerical strength and speed thresholds are not prerequisites for
implementing a correct policy. They matter only when making later empirical
claims; see the final section.

## Policy boundary and reproducibility

- Observe only the information in `Decision`. Its existing observation provides
  the own hand, all public melds, discard associations, public missing suits,
  active-player status, and wall count needed here. Do not obtain other
  concealed hands or wall order through diagnostics, shared policy state,
  or a privileged interface.
- Choose only an action permitted by the supplied legal choices. An illegal
  action is a policy defect, not grounds for silent substitution or retry.
- Use no RNG or history-dependent logical state. The same decision must produce
  the same action, regardless of cache population, eviction, or prior games.
- Break explicit suit/tile ties by fixed order and remaining action ties by
  fixed action order, never by cache traversal order. Discarding a cache during
  continuation may change timing but must not change subsequent actions.

## Hand-analysis contract

### Distance and normalized positions

Define `S(H)` on a post-discard concealed hand `H` of `13 - 3m` tiles,
where `m` is the number of already declared melds, held fixed during analysis,
and the declared missing suit has been cleared. A *winning target* `T` has
`14 - 3m` concealed tiles and, together with those same melds, forms a legal
standard four-sets-and-a-pair or seven-pairs winning shape. A target cannot
contain the missing suit or exceed four physical copies of any tile kind
when its concealed tiles and the existing melds are counted together.

Let `D(H)` be the minimum, over winning targets, of the number of copies
that must be drawn to reach the target while discarding surplus tiles:

```text
D(H) = min over legal winning targets T of sum_t max(0, T[t] - H[t])
S(H) = D(H) - 1
```

Thus `S(H) = 0` means a single draw can complete the hand; larger values count
the minimum additional hypothetical draws beyond that final winning draw.
This is structural distance, not a prediction of wall availability: public
exhaustion does not exclude a target, but copies in the player's own melds
do count toward the four-copy limit. Declared melds complete standard sets and
disable seven pairs. Four identical tiles may supply two pairs under the
fixed seven-pairs rule. A legal complete `14 - 3m` hand has shanten `-1`.

At `S = 0`, the structural waits must agree with the existing
[`eval::waiting_tiles`](../../../src/eval.rs), filtered for the missing suit;
publicly exhausted waits remain structural waits. Ordinary candidates are
compared only after discarding. For a hypothetical draw, recognize a legal win
directly; otherwise normalize by the best legal subsequent discard. Do not
compare 14-tile pre-discard and 13-tile post-discard positions directly, or
remove missing-suit tiles and run ordinary shanten on the incomplete result.
Forced missing-suit handling follows the decision rules below.

### Availability

For a post-discard position `H`, define:

```text
U(H) = sum unseen(t), over permitted tile kinds t whose draw reduces S(H)
unseen(t) = 4 - own concealed copies(t) - public physical copies(t)
```

Apply candidate actions consistently to the hypothetical concealed hand, melds,
and public ownership before counting. Public copies comprise all declared melds
and discards not incorporated into a meld. Count the deciding player's melds
once, omit meld-claimed discard records from discard ownership, and count a
multi-winner discard once. A kind with zero unseen copies contributes zero.

`U` estimates unseen improving copies, not exact wall contents or draw
probabilities. Do not infer identities of concealed tiles held by opponents,
including players who have won. In particular, `S = 0, U = 0` is permitted and
must not be treated as non-tenpai.

### Candidate ranking

Rank post-discard hands lexicographically:

1. Lower `S`.
2. Higher `U`.
3. When tenpai candidates remain tied, higher maximum hypothetical plain
   discard-win fan, as used for big-call settlement.
4. Fixed tie order.

Fan cannot compensate for worse distance or availability. The fan maximum uses
legal structural waits, including publicly exhausted ones, and no event bonus.

## Decision rules

| Decision | Required behavior |
| --- | --- |
| Exchange | Match the deterministic baseline: eligible suit with fewest own tiles, lowest suit index on a tie, then its three lowest tiles |
| Missing suit | Choose the fewest-tile suit in the post-exchange hand; use fixed suit order on a tie |
| Legal Hu | Take Hu immediately, rather than waiting for self-draw or a larger hand |
| Sole legal action | Return it without hand analysis |
| Forced missing-suit discard | Choose a legal missing-suit tile by fixed tile order, without shanten/ukeire analysis |
| Ordinary discard | Choose the resulting hand with the best candidate ranking |

Only minimum-shanten candidates need availability analysis, and only tied tenpai
candidates need the fan tie-break.

### Peng

Compare passing with the position after the peng and its best legal forced
discard. Accept peng only if it strictly lowers shanten. An availability increase
at unchanged distance is insufficient. Both alternatives use the same evaluator,
including the loss of the seven-pairs route on declaring a meld.

### Kongs

Evaluate a kong after declaration and before the replacement draw, with
updated melds and the corresponding post-discard effective hand size. Treat
this as a structural proxy for the next draw-ready position, **not** as the
value of the actual kong transition: it does not model the immediate
replacement tile, robbing an added kong, changed turn timing, or possible
refunds. Compare against the best ordinary discard on one's turn, or passing
on a discard response.

A kong is eligible only when its `(S, -U)` rank is lexicographically no worse
than the alternative. Prefer an eligible kong to the ordinary alternative.
Among eligible kongs, prefer better progress rank, then larger nominal immediate
income under the fixed rules and active payers, then fixed order. If both peng
and open kong are eligible on a response, compare their resulting progress ranks
and prefer the kong on a tie. Do not add replacement-draw search or a robbery
probability model. The gate does not guarantee positive expected kong income.

## Comparison and evidence contract

Use [`examples/policy_ab.rs`](../../../examples/policy_ab.rs) with fresh policies
per game, six balanced 2A2B placements per seed, and the complete seed block as
the sampling unit. Players act independently, without sharing concealed data.
The primary comparison is against `FirstLegalPolicy`; `RandomPolicy` is secondary.
Report both mean terminal net scores and `Delta = A_mean - B_mean` in score units
per player per game. Do not treat placements or players within a seed block as
independent samples, or compare homogeneous self-play means as evidence of an
advantage. Claims apply to the specified opponent population.

For the implementation milestone, reuse the existing A/B records and summary:
terminal net scores, finish reasons, decisions, policy means, seed-block `Delta`,
and its standard error. Also record release-mode end-to-end throughput and
policy decision cost using existing timing facilities or a separate measurement.
Identify the source revision, workload, policies, seeds, and measurement
conditions. Do not enlarge the Game/Runner interfaces or the A/B artifact schema
solely to collect extra diagnostics.

Development comparisons are exploratory and may inform later policy research.
A missing-suit-only ablation can isolate the benefit of fixing the baseline's
constant declaration, but is not required to merge this first policy. A later
*confirmatory* strength claim needs a policy and primary contrast frozen before
using independent confirmation seeds, with a predeclared seed-sampling procedure,
sample size, confidence-interval method, and practical effect threshold. The
sampling unit remains the complete seed block. If claiming a practically
meaningful improvement *with statistical uncertainty accounted for*, require
the prespecified 95% interval's lower bound to exceed the practical threshold;
a point estimate above the threshold alone is descriptive, not that stronger
claim. Choose the threshold and sample size using development results, not the
confirmation outcomes. Hand-picked seeds support descriptive conclusions only.

`RunReport` exposes winner flags, but the current A/B records do not persist
them. Exhaustive-draw tenpai, flower-pig status, kong income/refunds, and
specific discard-win losses are also not available from those records. Win
rates and settlement diagnostics are optional follow-up measurements: add a
narrow explicit collection path only if an experiment needs them, and never
infer them from net scores or finish reasons.

## Acceptance conditions

### Core implementation

Verification must distinguish these observable cases:

- A lower-shanten discard wins even when another has more ukeire; at equal
  distance more ukeire wins; the fan tie-break applies only as specified.
- Standard and seven-pairs routes are both considered before calling; four
  identical tiles supply two pair units, while any meld disables seven pairs.
- The defined target-draw distance is respected for `S > 0`, with fixed melds,
  standard/seven-pairs targets, and own-meld four-copy restrictions. Zero
  shanten agrees with legal structural waits; publicly exhausted waits retain
  tenpai status with zero availability.
- Meld-claimed discards, own melds, and multi-winner discards are not double
  counted, including hypothetical call and discard positions.
- Opening and forced choices follow the decision table. A peng that only raises
  ukeire is rejected. Kong eligibility uses the lexicographic gate: a lower
  shanten may justify lower ukeire, but equal shanten with lower ukeire fails.
- Equal inputs produce equal actions with cold or populated caches, and cloned
  game continuation produces the same decisions even if caches are discarded.
- All selected actions are legal; identical seeded runs replay; complete A/B
  blocks contain all six placements and zero-sum terminal scores. Failures use
  existing runner/example error handling, not fabricated scores or skipped games.

Apply the relevant observation-boundary, replay, and standard checks from
[project governance](../../../docs/governance.md). Private analysis helpers do
not need to become public merely for verification.

### Experimental outcome

Core implementation is accepted for correct, deterministic decisions, legal
behavior, native A/B integration, and reported runtime observations. A positive
score advantage, a confirmatory confidence interval, and a numerical speed
target are **not** implementation acceptance gates. Report unfavorable results
without silently revising the policy or tuning on confirmation seeds.

## Unresolved delivery and research choices

| Question | What settles it | What it currently blocks |
| --- | --- | --- |
| How users select the new comparison: local fixed configuration or CLI selection | Choose the smallest useful integration surface; no CLI expansion is presumed | A/B wiring, not the policy's rules |
| Practical effect threshold, confirmation sample size, sampling procedure, and interval method | Fix these before an independent confirmation study, using development evidence | A confirmatory strength claim, not implementation |
| Acceptable numerical runtime cost | Interpret measured performance against a concrete research workload | A claim of acceptable speed, not implementation |

Actual superiority remains an experimental result, not an assumption. Decomposition
and bounded/local memoization layout are internal implementation choices and
do not alter the defined distance or policy behavior. No policy implementation
or experiment is part of this documentation-only revision.
