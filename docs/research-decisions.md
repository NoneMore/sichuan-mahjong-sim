# Unimplemented research decisions

Keep research methods only for choices that remain unimplemented. Each entry
records the hypothesis, comparison, measurement method, acceptance rule, and
decision. Current contracts are in [code and tests](../src/) and
[architecture](architecture/overview.md).

## Common performance protocol

For the four DP candidates below:

1. Freeze baseline B and single-change candidate C; build both with the same
   toolchain, inputs and uninstrumented Release settings.
2. Check exact costs and optimal masks against an independent target oracle,
   followed by fixed-seed action-by-action replay, settlement and checkpoint
   continuation. Count actual coverage of special stages.
3. Run four-seat `heuristic_bench` serially, away from builds and profiling.
   Preserve warmup, cache lifecycle and game count across B and C.
4. For each seed, obtain three B/B pairs to set the noise bound
   `n = max |1 - B2/B1|`. Alternate the order of five B/C pairs; set each
   paired improvement to `r = 1 - C/B`. Standard acceptance: at least four
   positive pairs and `median(r) > n` **for each seed**.
5. Decide from complete-game timings. Use counters, memory measurements and
   microbenchmarks to explain costs. Fix the acceptance rule before testing.

The experiments used seeds 7 and 10007. Variant-specific workloads and
additional rules follow.

## Open decisions

### Exchange-three structural proxy

**Hypothesis.** Selecting the three exchanged tiles using the remaining hand's
exact distance D and improving-tile width U improves terminal scores at an
acceptable cost. The variants differ only in exchange ranking; all other
choices use HeuristicPolicy:

| Variant | Exchange choice |
| --- | --- |
| Legacy | Existing exchange rule |
| D-only | Minimize (D, baseline tie-break) |
| D+U | Minimize (D, −U, baseline tie-break) |
| Full | Add remaining missing-suit count as a tie-break |

**Method.** Verify legal candidate enumeration, exact distance/width,
tie-break equivalence, cache independence and replay. Compare Full/Legacy,
D+U/D-only and Full/D+U separately over seeds 100000–101023. For each seed,
play all six 2A2B seat placements with fresh policies. Average the difference
between the A-seat and B-seat mean scores over those six games; calculate
mean Δ and standard error SE **across seed blocks**, not individual games.
Run five alternating-order cost repetitions using cold shared cache contexts
(27,787,264-byte budget), with full-game and exchange-only timings. Track
whether the changed decision affects the suit or only the tile multiset.

**Acceptance.** Require `Δ − 1.96 × SE > 0` for benefit contrasts and at
most 50% additional mean end-to-end cost in Full/Legacy and D+U/D-only.
Full/D+U checks a predicted equality of chosen actions, not a score gain.

**Decision.** The candidate missed the combined acceptance rule. Choosing
whether to redesign the exchange proxy or abandon it remains open.
[Experiment specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/d21504988b394072bba6205fa3e6b85e9fc3a1b7/specs/active/heuristic-exchange-structure-proxy/spec.md).

### Needed-mask DP suffix bound

**Hypothesis.** A lower bound on the required current/next-rank tiles can
prune bounded Needed-mask DP without losing equal-cost masks. Candidate C
adds `L = max(0, a+b−hand[r]) + max(0, b−hand[r+1])` and skips states only
when `g+L > d`, where d is exact whole-hand distance; B uses the original
`g > d` cutoff.

**Method.** Use independent full-table oracles, rank 7/8 and equality
cases, all meld counts, and 40 fixed-game action/continuation comparisons
(seeds 7–26 and 10007–10026). For each seed, compare B/C with
`heuristic_bench --games 2000 --warmup 20`: three B/B and five alternating
B/C pairs. Also preregister three long B/C pairs with 10,000 games and warmup
20. Measure state/transition reductions separately from timing.

**Acceptance.** The common five-pair rule must pass for both seeds **and**
all three long-batch pairs must improve with median exceeding the short
B/B noise bound, again for both seeds.

**Decision.** Correctness checks passed, but the short and long performance
gates conflicted. Adoption requires a newly fixed complete performance study.
[Experiment specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/5360ab6ce520569142341de8ceb814549c50fc07/specs/active/needed-suffix-bound/spec.md).

## Decisions against the tested candidates

### Batch Needed-mask bidirectional sensitivity

**Hypothesis.** Joint forward/reverse co-occurrence analysis can replace
per-discard bounded Needed-mask queries. C computes forward and reverse
optimal-path summaries and transposes co-occurrence into candidate masks;
other inputs use B's existing per-candidate path.

**Method.** Compare exact masks/ukeire with target oracles and 40 replay/
continuation games. Count baseline builds, forward/reverse states, allocations
and fallback paths. Run 2,000 games per seed with warmup 20, three B/B and
five B/C pairs, in round order BB, BC, CB, BB, BC, CB, BB, BC.

**Acceptance / decision.** Require the common paired total-time gate at both
seeds. The tested C failed and was rolled back; any revised approach needs a
fresh comparison including reverse-scan and allocation costs.
[Experiment specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/cc1a32e30919d6775d952733328ca875080b5b86/specs/archive/batch-needed-sensitivity/spec.md).

### Suit-local masked DP u16

**Hypothesis.** Narrowing nine-bit suit-local masks from u32 to u16 reduces
whole-game cost; global masks and semantics stay u32.

**Method.** Compare complete needed/surplus tables with independent and
frozen wide-mask oracles. Replay 64 fixed games (seeds 7–38 and
10007–10038) and decision-20 continuations. Inspect layout and cold/cache-hit
costs separately. Use 10,000 games per seed, warmup 20 and the common
three B/B plus five alternating B/C pairs.

**Acceptance / decision.** Apply the paired gate independently to **both
total and policy time**, for each seed. The candidate failed; retain the u32
layout unless a new candidate meets a new preregistered gate.
[Experiment specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/01b25dac1f975d60da5c78eb3f0a51cef1654297/specs/archive/masked-dp-u16/spec.md).

### Shared needed/surplus suit DP

**Hypothesis.** One (cost, needed, surplus) suit table and shared cache can
avoid redundant construction. Whole-hand caches and ordinary deficit queries
retain their separate paths.

**Method.** Count per-evaluator cache requests, hits, builds and overlapping
keys in real request order. Check both masks against independent targets and
64 fixed-game traces with continuation. Measure isolated build, allocation
and cache overhead outside the formal timing. Use 2,000 games per seed,
warmup 20, three B/B and five alternating B/C pairs.

**Acceptance / decision.** Require the common paired total-time gate at both
seeds. The tested shared cache failed and was rolled back; a new design must
measure both saved builds and the cost of sharing.
[Experiment specification](https://github.com/NoneMore/sichuan-mahjong-sim/blob/85e44a9570b0486ee724e5aa8af881cb07c7759a/specs/archive/dual-mask-suit-dp/spec.md).
