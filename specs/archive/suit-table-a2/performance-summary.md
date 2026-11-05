# Suit-table A2 — measurement summary

Generated from `measurements/02-formal.json`, `measurements/01-microbench.json` and
the corrected counting record `measurements/05-corrected-counts.json`; every
number below is reproducible from those raw records plus `builds.json`.
Seconds are the benchmark's internal `total_time`, not outer process time.

## Formal benchmark (`heuristic-benchmark-v1`, `--games 2000 --warmup 20`)

Rounds 1–3 per seed, three variants per round in the pre-fixed rotation
(`M,MA,A2` for seed 7, its reverse for seed 10007). No extension was needed
because `A2-vs-M` was already identifiable after three rounds; all 18
invocations succeeded, with no retry and no discarded sample.

| Comparison | Verdict | Basis |
| --- | --- | --- |
| `A-vs-M` | **regression** | M max < MA min in both seed groups |
| `A2-vs-M` | **benefit** | A2 max < M min in both seed groups |

Medians with min–max ranges over the three rounds follow. For convenient raw
inspection, these are all 18 internal batch times in seconds (the JSON also
retains each invocation's order, stdout/stderr and other fields):

| Seed | Version | Round 1 | Round 2 | Round 3 |
| --- | --- | ---: | ---: | ---: |
| 7 | M | 16.239503 | 15.779704 | 15.924593 |
| 7 | MA | 19.507388 | 19.278280 | 19.085092 |
| 7 | A2 | 15.578590 | 15.636014 | 15.598216 |
| 10007 | M | 16.297933 | 16.277499 | 16.313252 |
| 10007 | MA | 19.707493 | 19.636781 | 19.788332 |
| 10007 | A2 | 16.014748 | 16.199218 | 16.042941 |

### A-vs-M

| Seed | Version | total_time s | policy_time s | games/s | decisions/s |
| --- | --- | --- | --- | --- | --- |
| 7 | M | 15.925 (15.780–16.240) | 15.829 (15.686–16.136) | 125.59 | 7148.8 |
| 7 | MA | 19.278 (19.085–19.507) | 19.183 (18.991–19.406) | 103.74 | 5905.1 |
| 10007 | M | 16.298 (16.277–16.313) | 16.201 (16.181–16.216) | 122.72 | 7077.6 |
| 10007 | MA | 19.707 (19.637–19.788) | 19.611 (19.540–19.691) | 101.48 | 5853.2 |

Per-round change (candidate / baseline − 1):

| Seed | Round | total_time change | policy_time change |
| --- | --- | --- | --- |
| 7 | 1 | +20.12% | +20.27% |
| 7 | 2 | +22.17% | +22.29% |
| 7 | 3 | +19.85% | +19.98% |
| 10007 | 1 | +20.92% | +21.04% |
| 10007 | 2 | +20.64% | +20.76% |
| 10007 | 3 | +21.30% | +21.43% |

### A2-vs-M

| Seed | Version | total_time s | policy_time s | games/s | decisions/s |
| --- | --- | --- | --- | --- | --- |
| 7 | M | 15.925 (15.780–16.240) | 15.829 (15.686–16.136) | 125.59 | 7148.8 |
| 7 | A2 | 15.598 (15.579–15.636) | 15.504 (15.482–15.541) | 128.22 | 7298.3 |
| 10007 | M | 16.298 (16.277–16.313) | 16.201 (16.181–16.216) | 122.72 | 7077.6 |
| 10007 | A2 | 16.043 (16.015–16.199) | 15.948 (15.919–16.102) | 124.67 | 7190.1 |

Per-round change (candidate / baseline − 1):

| Seed | Round | total_time change | policy_time change |
| --- | --- | --- | --- |
| 7 | 1 | -4.07% | -4.05% |
| 7 | 2 | -0.91% | -0.92% |
| 7 | 3 | -2.05% | -2.05% |
| 10007 | 1 | -1.74% | -1.74% |
| 10007 | 2 | -0.48% | -0.49% |
| 10007 | 3 | -1.66% | -1.66% |

Every A2 round improves, from −0.48% to −4.07%; the conservative range criterion
holds in both seed groups. This is a small engineering benefit for this machine
and workload, not a statistical significance claim.

## Microbenchmark (per-call `suit_table`, supplementary)

Medians over 25 samples per input (5 slots x 5 rounds), from an uninstrumented
standalone build of the extracted production function. These inputs are
synthetic local-function stress probes, not whole hands reachable through
declared melds: `capacity` and `target_sets` are varied independently, so the
rows are unweighted point probes and no aggregate or end-to-end benefit is
derived from them. Relative changes are
against `M`. Count columns come from the separate untimed instrumented model and
use the corrected **leaf** transition definition (`considered` at the innermost
loop holding the accept site: `pair` loop for `M`/`A2`, `triplets` loop for
`MA`), taken from `05-corrected-counts.json`.

| Input | M ns/call | MA rel. | A2 rel. | accepted | M considered | MA considered | A2 considered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `empty-cap4-s4` | 5564 | +30.51% | +5.30% | 1183 | 3036 | 1183 | 1938 |
| `typical-cap4-s4` | 5897 | +28.20% | -1.92% | 1183 | 3036 | 1183 | 1938 |
| `dense-cap4-s4` | 5797 | +32.17% | -1.87% | 1183 | 3036 | 1183 | 1938 |
| `wide-cap4-s4` | 6053 | +25.05% | -4.53% | 1183 | 3036 | 1183 | 1938 |
| `empty-cap4-s3` | 3255 | +43.96% | +2.75% | 715 | 1436 | 715 | 1106 |
| `typical-cap4-s2` | 1850 | +41.47% | +0.61% | 375 | 578 | 375 | 544 |
| `typical-cap4-s1` | 808 | +35.10% | +3.38% | 128 | 180 | 128 | 180 |
| `typical-cap4-s0` | 394 | +9.99% | +0.91% | 26 | 34 | 26 | 34 |
| `peng-cap1-s4` | 5105 | +28.37% | -3.55% | 1042 | 2708 | 1042 | 1704 |
| `gang-cap0-front-s4` | 5011 | +26.12% | -5.42% | 983 | 2562 | 983 | 1608 |
| `tail-pressure-s4` | 3009 | +37.64% | -3.27% | 583 | 1614 | 583 | 948 |
| `missing-suit-cap0-s4` | 441 | -20.16% | -24.25% | 9 | 230 | 9 | 18 |

`accepted` is identical for all three variants on every input (the counting
invariant) and every instrumented table equals the timed uninstrumented table.
`MA`'s leaf `considered` equals its `accepted` count because its loop bounds
leave no further rejection at the accept site; `M` and `A2` still reject
candidates there (`M` 3036 -> 1183, `A2` 1938 -> 1183 on the full-capacity
input). Per-layer `pair`/`sequences`/`triplets` iteration counts are in
`05-corrected-counts.json`.

These counts are enumeration counts, not speed: `MA` enumerates fewer leaf
candidates than `A2` yet its per-call time is 25–44% higher than `M`'s, and
`A2`'s count reduction is not proportional to its small timing change. The
`missing-suit-cap0-s4` row has zero per-rank capacity, so almost every candidate
is rejected by the capacity guards early (230 leaf candidates, 9 accepted in
`M`); the row reflects that high-rejection, zero-capacity pruning, and no
dynamic early-exit attribution is made from it.

`01-microbench.json` also contains the first counting run, which placed
`considered` at the last `pair` loop and therefore recorded full A's outer
`pair`-layer count (573) instead of a leaf count. That field is superseded raw
history and is not used in this table; all `01` timing samples and table digests
remain valid.

## Static release assembly (informative)

`measurements/03-asm-static.json` (extracts and diffs in `measurements/asm/`),
generated before measurement with `cargo rustc --release --lib -- --emit=asm`.
`suit_table` is inlined into `DistanceEvaluator::deficit` in all three variants,
so that enclosing function is the comparable unit; every branch target inside it
resolves (0 unresolved). Static code counts only: a backward-jump/loop edge is a
static edge, not an execution count, and no branch-hit, misprediction or
runtime-attribution claim is made.

| Variant | Function | Instructions | Conditional branches | Unconditional jumps | Loop edges | Multiplications | Divisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M | `deficit` (inlined `suit_table`) | 939 | 90 | 25 | 32 | 3 | 0 |
| MA | `deficit` (inlined `suit_table`) | 879 | 74 | 23 | 28 | 4 | 0 |
| A2 | `deficit` (inlined `suit_table`) | 980 | 96 | 25 | 27 | 3 | 0 |
