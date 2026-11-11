# Unimplemented research decisions and experiment methods

Retain research here only when a **specific, evidence-supported decision has
not been incorporated into the project**. The durable knowledge is **how the
conclusion was tested**, what it can and cannot establish, and what decision
remains—not a table of results from one execution. Do not copy old timings,
scores, test counts, source hashes, or machine logs. Current behavior and
correctness belong to [code and tests](../src/), the
[fixed rules](baseline-rules.md), and [architecture](architecture/overview.md).

Each entry gives its hypothesis/control, correctness checks, experimental
workload, comparison/acceptance procedure, and disposition. Numerical **inputs
and thresholds** below describe the *experiment design*, not measured outcomes.
Pinned links are supplementary provenance, not a substitute for the method.

## Shared method for performance-sensitive candidates

The DP experiments below used this controlled comparison pattern, with
experiment-specific exceptions noted in each entry:

1. **Freeze B and C before measurement.** B is the production baseline; C
   differs only by the candidate optimization. Build with the same toolchain,
   dependencies, flags and default Release profile. Record the actual build
   identities. Keep instrumentation, tracing, test execution and file/report
   generation outside the formal timing interval.
2. **Prove observational equivalence first.** Compare exact costs and complete
   optimal masks with an independent concrete-target oracle, not merely with
   another use of the candidate algorithm. Compare fixed-seed *per-action*
   traces, legality, terminal outcomes and clone/continuation behavior;
   identical aggregate scores alone do not establish equivalence. Record
   which situations the fixtures did not reach.
3. **Use complete-game paired measurements.** Run the uninstrumented
   `heuristic_bench` four-seat workload with the same seeds, warmup, cache
   lifecycle and game count for B and C. Execute runs serially, without
   overlapping compilation or profiling. Alternate B/C execution order to
   limit order bias. Measure `BatchReport.total_time` (game construction and
   batch execution, excluding warmup and process startup); record policy time
   separately where required.
4. **Calibrate noise instead of treating any speedup as decisive.** For each
   seed, collect three B/B pairs. Let
   `n = max(|1 - B_second / B_first|)` across those pairs. For each B/C
   pair compute `r = 1 - C / B` using paired measurements, not the ratio of
   independently aggregated medians. The usual five-pair rule requires at
   least four positive `r` values and `median(r) > n` for *each* seed.
   Exceptions and additional gates are stated below.
5. **Do not select a winner after inspecting results.** Fix seeds, repetitions,
   acceptance gates, failure handling and sample inclusion before formal
   measurement. Keep slow valid observations; do not append or exclude runs to
   manufacture success. Isolated microbenchmarks, DP state counts, allocation
   sizes and cache hits can explain mechanisms but **cannot substitute for
   end-to-end acceptance**. New candidate versions need a separately frozen
   protocol rather than extending a failed trial.

These are descriptions of the recorded experiments, **not universal
performance guarantees or automatic requirements for unrelated future work**.
A new investigation may improve the protocol, but must fix its changes before
seeing the new formal results.

## Open adoption decisions

### Exchange-three structural proxy

**Question / variants.** Does evaluating the post-exchange *remaining hand*
produce better game outcomes at an acceptable cost than the existing exchange
heuristic? Keep all non-exchange choices on the same `HeuristicPolicy` so the
exchange is the only intervention. Enumerate legal same-suit three-tile
multisets `T`, let `K = H - T`, predict the missing suit from `K`, and
compare:

| Variant | Exchange ranking |
| --- | --- |
| Legacy | Existing baseline exchange choice |
| D-only | Minimal exact target deficit `D`, then baseline tie-break `B` |
| D+U | `(D, -U, B)`, with `U` counting improving unseen tile copies |
| Full | `(D, -U, C, B)`, adding the residual missing-suit count `C` |

The *unseen* weight in `U` uses the **original** hand `H`, not `K`;
the prediction is not the actual post-receipt missing-suit declaration.
The `C` tie-break was also tested against `D+U` to validate its proven
ordering redundancy, not to assume it improved play.

**How it was tested.** First check legal canonical candidates, exact
standard/seven-pairs target calculations, tie handling, cache independence,
unchanged downstream policy actions and seeded replay. Then use 1,024
preregistered seeds `100000..=101023` and all six balanced two-versus-two
placements per seed. Run three distinct contrasts: Full/Legacy, D+U/D-only,
and Full/D+U. Fresh policies/Runner per game; shared cache configuration
remains fixed. Treat the **seed block**, not each of its six games, as the
independent unit: within a seed average the two A-seat net scores minus the
two B-seat net scores across six placements, then calculate the mean
difference `Δ` and between-seed standard error `SE`. Do not add the
contrasts as if they measured independent strategy gains.

In a separate cost study, compare full-game and exchange-only time for all
four variants over five repetitions with reversed execution order and cold
cache contexts shared across seats *within* each measured phase. Do not
charge diagnostic rescoring or output writes to the timing interval. Inspect
one initial decision per seed/seat to distinguish changed exchange *suit*
from merely changed *three-tile selection*.

**Predeclared adoption recommendation.** Require
`Δ - 1.96 × SE > 0` for the strategy contrasts intended to establish
benefit, together with no more than 50% extra mean full-game cost for
Full/Legacy and D+U/D-only. The interval is descriptive for the chosen
seed blocks, not a population guarantee. The Full/D+U structural-equality
check is not required to show positive gain.

**Conclusion / remaining decision.** The tested strategy and cost criteria
did not jointly support adoption. That is **not evidence of no possible
improvement**. The Full candidate remains on a historical research branch,
not in the current production exchange policy. Decide whether to discard
this direction or design and preregister another candidate and experiment;
do not change the original acceptance threshold retroactively. See the
[original specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/d21504988b394072bba6205fa3e6b85e9fc3a1b7/specs/active/heuristic-exchange-structure-proxy/spec.md)
and [recorded disposition](https://github.com/NoneMore/sichuan-mahjong-sim/blob/d21504988b394072bba6205fa3e6b85e9fc3a1b7/specs/active/heuristic-exchange-structure-proxy/results.md).

### Needed-mask DP suffix lower bound

**Question / variants.** Can a local lower bound prune bounded Needed-mask
suit DP without altering exact masks or slowing complete games? B uses the
previous `g > d` cutoff. C adds, before transition enumeration, a
rank-local bound for required current/next-rank tiles:
`L_r = max(0, a+b-hand[r]) + max(0, b-hand[r+1])`;
prune only when `g + L_r > d`. Keep the cutoff **strict** so equal-cost
optimal masks survive. `d` is the exact whole-hand distance for the same
context; non-bounded paths, seven pairs, policy selection and cache lifecycle
are unchanged.

**How it was tested.** First compare every table cost and optimal mask
against an independent complete-target oracle, including `<`, `=` and
`>` bound cases, rank 7/8, suit capacity, missing suit and 0–4 melds.
Compare B/C action-by-action for 40 game seeds
`7..=26` and `10007..=10026`, including clone and fresh-policy
continuations at attempted decisions 24 and 80 (report early finishes).
Run Debug/Release correctness checks before timing.

For each seed `7` and `10007`, run uninstrumented four-seat
`heuristic_bench --games 2000 --warmup 20`: three B/B pairs and five
B/C pairs in `BC, CB, BC, CB, BC` order. Require at least 4/5 positive
paired total-time improvements and median improvement above that seed's
B/B noise maximum. An *additional predeclared* long-batch check used three
B/C pairs per seed at `--games 10000 --warmup 20` in `BC, CB, BC`
order, requiring **3/3 positive** and median above the same short-run B/B
noise threshold. Both short- and long-batch gates had to pass. Separate
state/transition counters were used only to explain the computational effect.

**Conclusion / remaining decision.** The candidate's correctness held in
the checked domain, but the preregistered short- and long-run performance
gates conflicted; it was **not accepted into production**. Do not reclassify
the conflicting samples as success or add samples to that trial. A renewed
adoption attempt needs a newly fixed protocol and a fresh B/C comparison.
See the [experiment contract](https://github.com/NoneMore/sichuan-mahjong-sim/blob/5360ab6ce520569142341de8ceb814549c50fc07/specs/active/needed-suffix-bound/spec.md)
and [decision record](https://github.com/NoneMore/sichuan-mahjong-sim/blob/5360ab6ce520569142341de8ceb814549c50fc07/specs/active/needed-suffix-bound/results.md).

## Decisions not to adopt the tested candidates

The negative decision is scoped to the **tested implementation and its
predeclared criteria**, not to every possible future variant. It remains
useful because it identifies which hypothesis was tested and what would
have to differ before revisiting it.

### Batch Needed-mask bidirectional sensitivity

**Question / candidate.** Replace separate bounded Needed-mask calculations
for discard candidates by one joint co-occurrence analysis on the original
hand's optimal-target graph. The candidate saves per-rank forward cost/
surplus summaries, computes reverse optimal suffix contributions, and
transposes a co-occurrence matrix into candidate-specific Needed masks.
Apply only when `D > 0`, the hand has `14-3m` tiles and candidate
distance equals the original optimum; otherwise fall back to the baseline.
The actual baseline had already reused per-suit caches and computed
roughly `k+2` bounded tables, so the experiment must compare **actual
work**, not a hypothetical `3k` recalculation.

**How it was tested.** Independently compare complete
`(deficit, needed_mask)`/ukeire results, all tied targets and fallback
paths against the existing bounded computation and concrete-target oracle.
Replay 40 fixed games (seeds `7..=26`, `10007..=10026`), including
clone continuations and reporting unexercised claim stages. Measure request,
build, forward/backward scan, snapshot allocation and transposition costs
*outside* formal timing. For each of seeds `7` and `10007`, use
`heuristic_bench --games 2000 --warmup 20` with three B/B and five B/C
pairs in `BB, BC, CB, BB, BC, CB, BB, BC` round order. Require 4/5
positive paired **total-time** changes and median improvement above the
largest absolute B/B noise difference for **both seeds**.

**Conclusion.** The first candidate did not meet the end-to-end gate and
was rolled back. Work saved in one DP pass did not establish system-level
speedup when reverse scans and allocations were included. Revisit only a
materially different design with new, predeclared measurement conditions.
[Original protocol](https://github.com/NoneMore/sichuan-mahjong-sim/blob/cc1a32e30919d6775d952733328ca875080b5b86/specs/archive/batch-needed-sensitivity/spec.md).

### Suit-local masked DP `u16`

**Question / candidate.** Does storing nine-bit suit-local needed/surplus
masks as `u16` instead of `u32` improve actual throughput? C changes only
the local representation and widens to the unchanged `u32` 27-bit mask
before cross-suit combination. It must not change cost, masks, strategy,
cache behavior or public API.

**How it was tested.** Compare cost/needed/surplus tables against an
independent concrete-target oracle and frozen wide-mask reference, including
boundary capacities and tied optima. Replay 64 games, seeds
`7..=38` and `10007..=10038`, plus decision-20 clone continuations
with fresh/cold policies. Inspect memory layout and use separate
microbenchmarks for cold DP and suit-cache hits, but decide by serial,
uninstrumented `heuristic_bench --games 10000 --warmup 20` runs. Each
seed `7` and `10007` has three B/B calibration pairs, followed by five
B/C pairs ordered `BC, CB, BC, CB, BC`. For **both total and policy
time**, require at least 4/5 improved pairs and median B/C reduction
strictly above the maximum B/B absolute relative difference, independently
at both seeds.

**Conclusion.** The exactness checks were not enough to justify adoption:
the performance gate failed, so the local `u32` representation was kept.
A smaller structure or better isolated microbenchmark result is not a
substitute for a complete-game speedup.
[Original protocol](https://github.com/NoneMore/sichuan-mahjong-sim/blob/01b25dac1f975d60da5c78eb3f0a51cef1654297/specs/archive/masked-dp-u16/spec.md).

### Shared needed/surplus suit DP

**Question / candidate.** Can sharing a `(cost, needed_mask, surplus_mask)`
suit-level DP and cache avoid duplicate builds without making evaluation
slower? Keep the whole-hand needed/surplus caches separate, leave plain
`deficit()` lightweight, preserve mask semantics and do not expand
cache lifetimes.

**How it was tested.** First count actual needed/surplus cache requests,
hits, builds, and per-evaluator overlapping `(suit, counts)` keys in
real access order; simulate possible reuse without merging different
evaluator/meld/missing-suit/capacity contexts. Then compare both masks
and costs with a concrete-target oracle, individual discard decisions,
and 64 fixed-seed game traces including clone/continuation. Isolated
build/cache/memory probes explain trade-offs but are **not** the adoption
metric. For each of seeds `7` and `10007`, use uninstrumented four-seat
`heuristic_bench --games 2000 --warmup 20`, three B/B pairs followed by
five B/C pairs (`BC, CB, BC, CB, BC`). Adoption requires correctness
**and**, at each seed, at least four faster B/C total-time pairs and
median `1 - C/B` above the seed's largest absolute B/B difference.

**Conclusion.** Reduced repeated construction did not satisfy the
predeclared end-to-end speed gate, and the candidate was rolled back.
Do not infer that all shared-DP designs are unprofitable or that the
old measured CPU hotspots remain current.
[Original protocol](https://github.com/NoneMore/sichuan-mahjong-sim/blob/85e44a9570b0486ee724e5aa8af881cb07c7759a/specs/archive/dual-mask-suit-dp/spec.md).

## Retention and repeatability boundary

The recorded one-off outputs, traces and experimental binaries were not
preserved in the working tree, and some discarded candidate versions would
need reconstruction. **These protocols explain how the past conclusions
were reached; they do not regenerate the exact historical measurements.**
For a new adoption decision, freeze an actual candidate and baseline,
register its evidence requirements and new acceptance plan first, then
collect and validate new results. Remove an entry when the decision is
implemented or no longer affects development. The
[historical retirement record](https://github.com/NoneMore/sichuan-mahjong-sim/pull/23)
identifies retained source-history snapshots.
