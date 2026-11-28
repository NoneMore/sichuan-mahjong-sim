# Online evaluation refactor verification

The fixed greedy comparison used validation seeds `2,10` in all six placements,
batch size 3, six concurrent handles, a 10,000-decision limit and a disabled
shared teacher cache. The checkpoint was generated once with
[the checkpoint fixture builder](../../../online/tests/_checkpoint_fixtures.py)
using role `best`, hidden width 8, model seed 5 and cache ID `refactor-baseline`.
Its SHA-256 was
`6721806e8017c55329560c0e59618ba556724f39c7a9992bc9350bca28412a5a`.

Baseline Python source came from
`f47801ccc4e03bd667a5ec1693e0b5524546f8fc`. Both versions used the same
release wheel's native extension, with compiled source SHA-256
`c128f7a91cc4eedcaaffbe4b7b57be052e979dbac9fa1a1766bb04e9f9458ae7`.
On CPU and on an RTX 2070 using CUDA, baseline and candidate matched
every greedy index, scalar trace, per-batch row order, terminal record,
call count and persisted summary. A batch row was identified by native
decision ID, seat, stage, candidate count and SHA-256 of the state bytes.
The candidate also reread the baseline's format v1 run and recomputed its
summary. All 12 games accounted for 883 decisions, including 203 forced,
366 eligible inference rows across 187 batches and 314 teacher queries.

## Matched CPU timing

Measured with Python 3.12.10, NumPy 2.5.4 and PyTorch 2.14.1+cu126 on
Windows x86-64 with a Ryzen 7 3700X. PyTorch intra-op and inter-op counts were
both one. After three warmups per version, 20 pairs alternated execution order
within one process. Each run measured the 12-game scheduler workload with
tracing disabled, weights already loaded and a fresh disabled teacher cache.
The scheduler timer includes handle creation, reset, selection and stepping.
The comparison CLI's checkpoint loading and file writes were outside this
diagnostic.

| Median per 12 games | Baseline | Candidate |
| --- | ---: | ---: |
| Total | 116.402 ms | 117.208 ms |
| Inference | 91.655 ms | 92.067 ms |
| Teacher queries | 15.039 ms | 15.111 ms |
| Environment step | 4.912 ms | 4.864 ms |
| Unaccounted | 4.545 ms | 5.019 ms |
| Throughput based on median total | 103.091 games/s | 102.382 games/s |

The median of paired total differences was +0.235 ms. Individual totals
ranged from 107.888 to 140.350 ms for baseline and 107.978 to 129.433 ms for
candidate. Earlier measurements showed roughly 0.95 ms of additional time
outside the native and inference calls. Removing repeated per-decision checks
and temporary group allocations brought that difference to about 0.47 ms.
The kernel builds requests, gathers per-seat counts and holds choices until the
round validates. Inference accounted for most of the variation in this workload.

## Verification coverage

- The release wheel passed 224 online tests, including the seed-0 corpus
  handoff. Native smoke tests exercised 2N2T, 4N, 4T and 2A2B. The 4T
  semantic wire trace matched native teacher replay and the seed-61 fixture
  continued after early Hu through the expected settlement.
- A v3 policy exported from the fixture's BC weights completed all six
  placements for validation seed 2 with its actual role, null training
  provenance and the same summary as the source weights. `last` and corrupted
  checkpoints failed before run publication.
- The standard BC suite passed 520 tests. Its opt-in CUDA overlap
  diagnostic was deselected by the package test configuration.
- Root workspace format, release build, all-target tests and warning-denied
  Clippy passed. The standalone extension passed format, tests,
  warning-denied Clippy and the Rust 1.75 check. A wheel rebuilt from the
  source distribution completed a native-only teacher plan.
