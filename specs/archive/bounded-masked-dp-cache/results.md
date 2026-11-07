# Bounded masked-DP cache: implementation and verification

Implementation, checks, and the revised selection experiment are complete; the
26.5 MiB default is retained for batched simulation, with the short and startup
regressions below and no claim of universal acceleration.

## Implementation

`HeuristicPolicy` keeps decision-local tables as its first lookup layer; an
explicit `HeuristicCache` adds byte-bounded, cross-decision memoization of
exact nine-kind masked suit tables, and ordinary distance analysis is
untouched. Keys own hand, capacity, target-set count, mask direction, and exact
bound (or unbounded tag); values retain every equal-cost optimal mask bit, so
shared history cannot change a result.

The table is allocated lazily on the first cross-decision lookup, never
exceeds its budget, and uses four-way groups with empty slots first then
group-local round-robin replacement; hits do not reorder. Zero budget disables
only the shared table; failed allocation permanently falls back to the
original exact computation. Contexts are explicit: clones share table, budget,
and failure state, and the last holder releases them; there is no global/TLS
sharing. Policies stay `Clone` and `Send + Sync`, without `Copy`. Fixed context
overhead is reported separately from table bytes. Cached data is discardable
memoization, never part of a checkpoint, so a fresh policy continues from a
cloned game exactly. No game rule, ranking, information boundary, runner
protocol, or checkpoint state changed.

## Correctness and resource checks

The original checked-in source `8332537` (spec-only HEAD before implementation)
was built separately with the default release profile and served as the
baseline, not the cache-disabled candidate. Its API replay fixed 40 seeds
(`7..27`, `10007..10027`) and a decision-24 checkpoint: **40 games / 2,179 decisions** plus **39 continuations
/ 1,221 resumed decisions**. All eight candidate outputs (budgets 0, 292,
27,787,264, and 55,574,528 bytes; forward and reversed) matched byte-for-byte,
including cold first use, accumulated history, and continuation.

`cargo fmt --all -- --check`, warning-denied Clippy, warning-free rustdoc, and
`cargo test --all-targets` / `--release --all-targets` passed with **316 tests
in both profiles** (213 library, 6 CLI, 25 integration, 33 heuristic-benchmark,
39 policy-A/B). Permanent coverage includes target oracles, complete ten-cell
tables, both mask directions, key dimensions, unreachable entries, equal-bound
and multiple-optimum masks, bound-order isolation, local-first reuse,
collision/replacement, budget validation, zero-budget allocation, lazy
initialization, shared ownership and release, `Send + Sync`, and injected
allocation failure with exact fallback.

## Performance protocol

Environment: Rust/Cargo 1.98.0, LLVM 22.1.8, `x86_64-pc-windows-msvc`; default
release profile, no custom flags or affinity pinning. Windows 10 build 19045,
AMD Ryzen 7 3700X (8 cores / 16 logical processors), about 32 GiB RAM.

The redesign was frozen before formal runs with seeds 7 and 10007. `N` is the
maximum absolute B/B relative difference over 3 B/B pairs; `R` is the median
`1 - C/B` over 5 interleaved B/C pairs against separately built baseline
`8332537`. A group passes with at least 4/5 `R > 0` and `median(R) > N`; all
figures are elapsed-time reductions, not throughput increases.

The 276 planned arms / 279,472 measured games completed in 220.938 seconds with
no failure, timeout, missing resource evidence, or degradation; all six phase
invocations exited zero, and no sample was added, deleted, or rerun. A
500-game cold screen compared both budgets. Accumulation (2,000 games) and
8/12/16 workers x 150 games were then measured only for the selected budget.
These phases used a separate 20-game warmup. Worker `i` used seed
`S + 100000*i` modulo `2^64`; group makespan spans first launch to last exit.
Memory was sampled every 50 ms, with a 25 ms inter-arm gap.
Warm 500 with warmup 500, cold 20 with separate warmup 20, and a fresh
one-game process without warmup are separate diagnostics, not averaged in.

Both budgets passed the screen; 53 MiB reached R 43.40% (seed 7) and 41.32%
(seed 10007). Doubling memory then had to add a predeclared 5% normalized-time
improvement in both seeds and did not, so the **26.5 MiB** four-way
empty-first/local-round-robin layout is retained; every main group had
`median(R) > N` for both seeds.

| Workload / primary boundary | N (seed 7 / 10007) | Median R | Positive pairs |
| --- | ---: | ---: | ---: |
| Cold 500 games / batch | 1.91% / 1.66% | 41.87% / 43.93% | 5/5 each |
| Accumulation 2,000 games / batch | 3.14% / 1.85% | 54.73% / 55.78% | 5/5 each |
| 8 workers x 150 games / makespan | 4.56% / 1.47% | 27.09% / 25.58% | 5/5 each |
| 12 workers x 150 games / makespan | 3.09% / 7.59% | 22.81% / 23.07% | 5/5 each |
| 16 workers x 150 games / makespan | 13.67% / 5.60% | 21.82% / 22.55% | 5/5 each |

### Diagnostics

| Workload / primary boundary | N (7 / 10007) | Median R | Positive pairs | Gate |
| --- | ---: | ---: | ---: | --- |
| Warm 500, warmup 500 / batch | 3.20% / 5.25% | 55.96% / 55.84% | 5/5 each | pass |
| Cold 20, separate warmup 20 / batch | 11.30% / 10.40% | -15.65% / 5.02% | 2/5, 5/5 | fail |
| Fresh one-game process / process wall | 2.49% / 10.42% | -64.46% / -60.86% | 0/5 each | fail |

There is no stable 20-game speedup, and one-game cold processes regress clearly
(about 10.8 vs 18.1 ms at seed 7 and 11.5 vs 18.3 ms at seed 10007).
Initialization is not free, so budget `0` suits isolated short tasks;
main-batch gains do not offset these diagnostics.

### Memory and pressure coverage

Each measured process owns one context shared by its four heuristic seats;
there is no cross-process sharing. Peak process commit and API peak working
set at 26.5 MiB were:

| Scope | Commit MiB, baseline -> candidate | Working set MiB, baseline -> candidate |
| --- | ---: | ---: |
| Single process, accumulation | 0.85 -> 27.25 | 4.02 -> 29.96 |
| 8 workers | 6.37 -> 217.75 | 31.65 -> 239.20 |
| 12 workers | 9.52 -> 326.58 | 47.47 -> 358.80 |
| 16 workers | 12.60 -> 435.32 | 63.26 -> 478.25 |

Multi-process values sum individual peaks, not necessarily simultaneous group
peaks. Median summed page faults were 1,072 -> 14,671 (accumulation) and
12,693.5 -> 175,895.5 (concurrency): allocation and soft faults, **not** hard
page-in counts.

Table allocation was 27,787,012 bytes for 26.5 MiB (27,787,264 bytes) and
55,574,316 bytes for 53 MiB (55,574,528 bytes); fixed context layout was
104 bytes here, excluding handles, storage, or allocator bookkeeping. The
1 MiB / 500-game pressure fixture had 14,364 slots, 52,564/54,186 misses, and
38,212/39,824 evictions without degradation; it exercises capacity-exceeding
replacement, not steady-state throughput; at 2,000 games the production table
replaced 1,442/1,789 entries and left some groups unfilled.

## Provenance and limits

An earlier, much larger all-combinations plan was stopped at the user's
request; its completed arms are not mixed into the final experiment.

Temporary experiment drivers, the replay collector, paired data, and raw
output, including baseline worktrees and target evidence, were removed at the
user's request. Historical numbers cannot be recomputed from repository data;
permanent Rust tests remain. These are one host's observations, not a
universal speedup or significance/optimality claim.
