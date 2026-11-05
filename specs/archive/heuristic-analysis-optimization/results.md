# Heuristic analysis optimization — implementation and evidence

## Final outcome

**Retain B only (seven-pairs formula).** Withdraw the tested A form for
identifiable regression; decline C and D after their bounded uncertain results.
All candidates passed correctness checks. No rule, strength, action-selection,
cache-key or checkpoint behavior was deliberately changed.

## Scope and correctness evidence

Baseline source is `c9202e33ac78cfb819def7339c3986c58cc3d13f`.
This work changes only private exact hand analysis; rules, policy ranking and
information boundaries, Runner, cache keys and cache lifetime remain unchanged.
See [execution and reconstruction](execution.md), [build provenance](builds.json)
and the [frozen behavioral corpus](baseline/README.md).

Before integration, seeds `7..=38` and `10007..=10038` were fixed as the 64-game
comparison corpus. The baseline was captured from its unchanged detached
worktree using `Runner::run_new`, one Runner reused across all games. It contains
3514 decisions; each full trace is nonempty and has exactly the report's decision
count. Baseline, A, B, C, ABC and ABCD rebuilt executables all reproduce the full
per-action corpus and scores/winners/end reasons, excluding only revision-label
lines. This is cross-version evidence, not merely candidate self-replay.

### A: tighter standard-target enumeration (withdrawn)

The experiment reserved mandatory pending sequence tails `a + b` before
enumeration; the free
rank capacity and existing set budget bound pair, sequence and triplet loops.
No terminal-only pruning, graph precomputation or buffer redesign was added.
All ten `SuitTable` entries are retained and compared with both original-bound
DP and independent concrete target enumeration for `target_sets = 0..=4`.
Zero/full capacities, peng/kong occupied ranks and sequence boundaries are
covered, together with deterministic generated hands.

The isolated count test records the following full-four-capacity results:

| target_sets | Original candidates | Tightened candidates |
| ---: | ---: | ---: |
| 0 | 34 | 26 |
| 1 | 180 | 128 |
| 2 | 578 | 375 |
| 3 | 1436 | 715 |
| 4 | 3036 | 1183 |

At four sets this removes 1853 rejected candidates (61.03%). Counts are
hand-independent; eight representative hands give the same counts. This is
work-reduction evidence only, not a wall-clock speedup estimate. The three-pair
formal comparison found identifiable regression in both seed groups: baseline
ranges 16.578–17.044 / 16.851–17.116 s versus A ranges
20.259–20.762 / 20.836–21.228 s. The production change is withdrawn; its exact
production patch and test-only counting model remain available for review.
No further A samples are requested to seek a more favorable result.

### B: seven-pairs formula

Use `K = 7.saturating_sub(P)` and `D_7 = K + K.saturating_sub(O)` over non-missing
kinds, where a quad provides two free units and a held triple is also an odd
kind. Every declared meld, including a concealed kong, still disables the route.
Independent per-kind 0/2/4 target-cost enumeration covers all 241 held-count
classes (18 eligible kinds, total at most 14), in two assignments to actual kinds;
separate tests cover quads, triples, missing-suit filtering, gating and `P > 7`.
Zero deficit on oversized test inputs is not interpreted as a legal win.

### C: omit the missing-suit identity pass (not retained)

The experiment skips slicing, table construction, cache access and merge for the missing suit.
Tests establish the table is the merge identity (`[0][0] = 0`, all others INF),
not an all-zero table. All three missing-suit choices and inputs with nonzero
missing-suit counts match the original three-suit reference, including meld
capacity cases. Missing-suit tiles remain free surplus, not an added clearing
cost. Cache cold/warm/repeated/reordered queries are compared value-by-value
against a fresh evaluator per hand for both deficit and ukeire. After five
pairs per seed, both ranges overlap: the formal result is **uncertain**, not
proven equal and not identifiable regression. Nine of ten paired total times
increase. We decline to keep this special-case branch: it removes no existing
algorithm or merge implementation, its practical simplification is modest, and
this budget has not established speedup or excluded slowdown. All samples,
including the 18.843794 s candidate observation, remain in the result.
Production continues to process the identity suit as before. Its exact
candidate and tests remain in `patches/C.patch` and `patches/C-tests.patch`.

### D: direct-win branch experiment (not retained)

The experimental implementation replaces ukeire's direct-win shortcut with
`D(H+t) < D(H)` only. Caller audit and tests establish that H has `13 - 3m`
tiles, its deficit comes from the same evaluator, melds are legal, own physical
copies respect the four-copy bound, and public counts include own melds.
The experimental local contract and debug-only hand-size guard were explicit;
the final production implementation retains the original direct-win shortcut.
`max_discard_fan` keeps its distinct win-legality check throughout.

Test-only original-shortcut comparison and explicit
`can_hu(H+t) == (D(H+t) == 0)` cover `m = 0..=4`, standard and seven-pairs hands,
multiple waits and nonwaiting draws, own kongs and publicly exhausted kinds.
Direct-win ukeire assertions are retained. Counterexamples demonstrate why hand
size and own four-copy premises are necessary. Draw/discard normalization adds
m=0/1 coverage to the existing m=2/3/4 oracle coverage. The experiment is
reconstructible using `patches/D.patch` and `patches/D-tests.patch`.

## Verification

Before measurement, integrated ABCD passed:

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --release --all-targets
```

Each experimental ABCD test profile passed 230 tests (151 library + 79 other
targets). After withdrawing A/C/D, **final B-only passed the same four commands
again: 214 tests in both debug and release** (135 library + 79 other targets).
The cache-order regression additionally compares fresh evaluators per hand and
repeated deficit/ukeire queries after hypothetical draws populate caches.
Existing seeded legality/zero-sum and clone continuation with fresh
HeuristicPolicy instances remain green. The final working-tree release library
was separately linked to the frozen collector and its 64-game/3514-decision full
trace matched the pre-change baseline again. See [verification.json](verification.json).
No existing expected actions, deficits, or ukeire values were relaxed.

A/C/D production+test patches were independently reapplied to clean baseline
worktrees and their 4/4/8 focused tests passed. Test-only references for these
withdrawn experiments remain isolated rather than retained on production paths.

## Performance and retention

A is rejected for identifiable end-to-end regression despite equivalent
behavior and fewer enumerated candidates. B is retained: correct, simpler and
with identifiable benefit in both seed groups. C is declined after an uncertain
five-pair result as explained above; uncertainty is not relabeled as regression.
The selected mainline is therefore **B**, not ABC or BC. Its mainline-vs-original
comparison is exactly `measurements/02-B.json`, reused rather than repeated.

| First seed | Baseline total median (range), s | B total median (range), s | Median reduction |
| --- | --- | --- | --- |
| 7 | 16.632 (16.625–16.714) | 16.205 (15.954–16.328) | 2.57% |
| 10007 | 17.124 (16.991–17.334) | 16.830 (16.828–16.935) | 1.72% |

Every B pair improves (−0.95% to −4.54% total time); the conservative
candidate-max < baseline-min criterion holds in each seed group. This is a
small local engineering benefit, not statistical significance, a minimum
speedup requirement, playing-strength improvement or cross-machine promise.
Policy time and throughput and every individual paired change are reported
in the detailed summary.

D's **BD vs B** incremental experiment reached five pairs per seed and remains
**uncertain**, with overlapping ranges in both groups. B/BD total-time medians
are 16.152/16.162 s (seed 7) and 16.513/16.602 s (seed 10007); eight of ten pairs
increase. Removing a branch does simplify code and its correctness contract was
verified, so the specification permits considering retention for simplification.
We nevertheless decline this cost exchange: there is no established incremental
speedup, no demonstrated need to remove the useful direct-win short circuit,
and the observations do not exclude slowdown. This is not an identifiable
regression claim or proof of equal performance.

BD was rebuilt and full-trace checked after selecting B, with separate
provenance in `builds-retained.json`. ABC/ABCD were only correctness experiments;
no combined speedup is used to conceal A's regression or D's uncertain increment.
Since D is not retained, there is no final BD-vs-original comparison. The final
combination is exactly B, already covered by `02-B.json`, as permitted by the
specification's identical-comparison reuse rule.

| Candidate | Formal verdict | Actual decision |
| --- | --- | --- |
| A | Identifiable regression, 3 pairs/seed | Withdraw this form; preserve its patch and tests |
| B | Identifiable benefit, 3 pairs/seed | Retain formula and permanent regression coverage |
| C | Uncertain, 5 pairs/seed | Decline special case; no proved speedup/no claim of equal performance |
| D relative to B | Uncertain, 5 pairs/seed | Decline branch removal; retain original win short circuit |

All **64 formal benchmark invocations** completed successfully (128,000 measured
games plus 1280 unmeasured warmup games), with zero retries or discarded samples.
The binary hashes, required workload fields and alternating pair order were
independently checked against both build manifests; range verdicts were
recomputed from all observations. Decision totals were 113,841 (seed 7) and
115,351 (seed 10007) in every corresponding batch. Aggregate non-time equality
supplements, rather than replaces, the full-trace and local equivalence tests.

Detailed bounded observations will be retained in
[performance-summary.md](performance-summary.md) with raw per-run JSON under
`measurements/`. Default release builds use the unchanged
`heuristic-benchmark-v1` benchmark, 2000 games, warmup 20, first seeds 7 and
10007. Measurements are serial, separate from trace capture, builds and tests;
three pairs per seed extend to five only on an uncertain initial verdict.
No historical timing is substituted for this session's paired baselines.
