# Historical validation: evidence and present relevance

This index answers two questions for each recorded result: **what did the
evidence actually establish?** and **has that conclusion already been
incorporated into the project, or made obsolete by later changes?** Follow the
linked records for the original revision, environment, commands, measurements,
and limitations. These are point-in-time findings, **not** today's test counts,
CI status, throughput, policy strength, or cross-version replay guarantees.

A conclusion marked **incorporated** has a corresponding current implementation
and/or regression boundary. That does *not* carry an old numerical result
forward to the current revision. An **historical-only** observation is useful
for explaining a past decision or reconstructing a comparison, not for choosing
today's implementation without fresh evidence. Current build and verification
instructions live in [development](development.md); current behavior is
established by code and tests, while [architecture](architecture/overview.md)
explains implemented boundaries.

## Game and runner baseline

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Initial 113 Debug/Release tests and 256 seeded invariant games | The then-current fixed-rule engine passed the recorded game, settlement, ownership, zero-sum, replay, and continuation checks. | **Incorporated:** current [Game](architecture/game.md) and [runner](architecture/execution.md) tests guard these invariants. The old test count is obsolete; it does not prove every current path. |
| 512-game cross-version replay after the decision hot-path change | The two *tested* revisions produced identical traces for that fixed corpus. | **Incorporated:** legality-buffer reuse is part of the implementation; current replay tests remain relevant. The old comparison does not establish equivalence with today's revision. |
| Paired 100,000-game hot-path measurements | Local medians improved by 10.36% (FirstLegal) and 8.88% (Random) under the original workload. | **Historical-only performance:** the optimization was adopted, but those percentages cannot rank current bottlenecks or promise current throughput. |

See the [native validation record](research/native-validation.md#game-and-runner-baseline).
For current behavior, run [workspace checks](development.md#running) and the
relevant scenario and invariant tests.

## Strategy comparisons

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Four-seed / 24-game exploratory contrasts | The early heuristic outscored the tested baselines in those small samples; no reliable general strength estimate follows. | **Historical-only:** the current heuristic exists, but those observations are too small and too old for selecting today's strategy. |
| 4,000-seed-block missing-suit ablation per contrast | In the recorded opponent mixtures, MissingOnly exceeded FirstLegal by 4.026667 mean points per player/game and Heuristic exceeded MissingOnly by 2.962542. The contrasts cannot be added. | **Incorporated:** [MissingOnlyPolicy](architecture/policy.md) preserves the controlled ablation. The measured magnitudes are historical, not a current or population-wide strength guarantee. |

See [native strategy comparisons](research/native-validation.md#strategy-comparisons)
and the current [balanced comparison procedure](guides/policy-comparison.md).
New strategy claims need fresh balanced seed-block measurements.

## Exact-analysis changes and profiling

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Seven-pairs formula and A2 pruning checks | The tested optimized computations agreed with independent reference results and replay checks. | **Incorporated:** exact analysis and regression oracles now express the correctness requirement; old microbenchmarks are not a live speed claim. |
| One-pass ukeire tests and paired runs | Optimal-target Needed masks preserved the tested semantics; paired historical runs showed roughly 62% policy/total-time reductions. | **Incorporated:** the algorithm and oracle coverage remain, but the speedup is obsolete for present planning. |
| One-pass discard validation | Surplus-mask derivation preserved the tested discard decisions and continuation behavior. | **Incorporated:** current analysis/tests are the correctness authority; historical timings are only decision context. |
| Bounded shared masked-DP cache acceptance | Cache enabled/disabled/failure paths preserved tested decisions; the adopted 26.5 MiB default traded large-batch gains against cold-start costs. | **Incorporated:** [cache semantics](architecture/policy.md#cache-and-checkpoint-boundary) remain important; past cost curves do not choose a current budget without remeasurement. |
| Bounded Needed-mask DP acceptance | The strict cutoff preserved equal-cost optimal masks in the tested reference comparisons. | **Incorporated:** correctness is protected by current analysis oracles; old execution-cost observations are not current evidence. |
| CPU attribution samples from multiple revisions | Each profile identified where CPU time went *in that specific revision* (e.g. 79.09% masked suit DP in the post-discard snapshot). | **Historical-only:** those different implementations cannot be combined into a current hotspot ranking or converted into predicted speedups. |

See the [original decisions, probes, and revision boundaries](research/native-validation.md#exact-analysis-changes-and-profiling).
Retain these as explanations of adopted designs, not as a prioritized
optimization backlog.

## Rust BC preparation smoke validation

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Four-game preparation, independent NumPy bitwise comparison, and verification after removing the corpus | The recorded snapshot's 232 decisions / 1,374 candidates retained feature bits, order, targets and metadata; the prepared cache could be verified without its source corpus. | **Incorporated:** the [preparation contract tests](../crates/neural/tests/prepare_contract.rs) and [cache protocol](../crates/neural/README.md#dataset-preparation) now cover this boundary. The sample counts and cache hash identify historical inputs, not an expected current output. |

See [preparation evidence and environmental limits](research/offline-bc-validation.md#rust-bc-preparation-smoke-validation).

## Python dataset loader smoke validation

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Installed-wheel loader, Rust handoff probe, multi-shard/partial-batch and empty-split checks | The recorded loader correctly consumed that Rust cache with bitwise feature and metadata agreement and valid batches; the 88 passing tests describe that version only. | **Incorporated:** the [loader](../python/README.md#loading-and-batching), Python tests, and [handoff probe](../python/tools/verify_rust_handoff.py) now own the checks. |
| Later loader maintenance and 114-test rerun | That follow-up retained the stated loader and provenance behavior at its revision. | **Incorporated:** coverage persists in current tests; the old test count, hashes, and fixture-specific totals are obsolete as current acceptance metrics. |

See the [loader validation record](research/offline-bc-validation.md#python-dataset-loader-smoke-validation).

## Offline BC training smoke validation

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Thirteen-game corpus, three-epoch CPU run, frozen evaluation, wheel/editable tests, and local CUDA smoke | At the tested revisions, training traversed eligible decisions, selected `best.pt` by validation NLL, reloaded a frozen model, and completed a small GPU smoke. The reported NLL/accuracy are measurements of a tiny fixture, **not playing strength**. | **Incorporated:** [training/checkpoint contracts](../python/README.md#training-metrics-and-checkpoints) and current tests cover the behavior. Old loss values and 178-test counts are not benchmarks or current CI claims; local GPU smoke is not continuous GPU coverage. |

See the [training validation record](research/offline-bc-validation.md#offline-bc-training-smoke-validation).

## Offline BC recovery validation

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Interrupted CPU training versus uninterrupted training | Under the recorded fixed runtime/configuration, replay matched sample order, steps, model and optimizer exactly (`rtol=0, atol=0`). | **Incorporated:** checkpoint recovery and [recovery verification](../python/tools/verify_recovery.py) now cover this invariant. Exact reproducibility is conditional on the matched runtime, not universal across machines/devices. |
| CUDA recovery, completed-run migration, and later maintenance rerun | Those particular GPU and migration workflows worked at their source/dependency revisions; the later follow-up retained checkpoint schemas and reran compatibility checks. | **Incorporated:** these are present recovery behaviors and tests. Historic 227/228-test counts, device versions, hashes and small-run NLLs are not current acceptance results. |

See the [recovery validation record](research/offline-bc-validation.md#offline-bc-recovery-validation).

## Recovery and new measurements

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Retirement inventory, recorded branch revisions and deleted raw artifacts | The retained reports identify some historical accepted, rejected, and inconclusive candidates, but Git cannot reconstruct locally discarded samples, binaries or trace outputs. | **Still relevant evidence limit:** this is not an implemented feature or a current speed result. It constrains what can responsibly be claimed from old experiments. Rebuild the specified revision and collect *new* evidence for new comparisons. |

See [recovery and new measurements](research/offline-bc-validation.md#recovery-and-new-measurements)
and the linked historical retirement inventory.

## Python online comparison smoke validation

| Historical evidence | What it supports | Present relevance |
| --- | --- | --- |
| Native/Python replay, installed wheel, source-distribution and failure-record tests | The recorded binding, frozen inference, record integrity and packaging paths worked under the tested Windows/Linux-CI conditions. | **Incorporated:** current [online tests](../online/tests/) and [evaluation interface](../online/README.md) own the contracts; 152 tests at the historical revision is not today's suite count. |
| Small-model CPU/CUDA batching pilot | In this specific one-thread, small-network workload, CPU batches were faster than GPU; summary totals agreed across tested settings, but individual actions were not proven identical. | **Historical-only performance:** batching and scheduling remain implemented. This is not a general CPU recommendation or current throughput target. |
| 12-game validation and six-game test comparisons | The pipeline completed the six balanced placements per seed and produced the reported negative neural scores for that tiny one-epoch checkpoint. | **Pipeline smoke only:** the current comparison workflow remains useful, but those scores cannot establish learned-policy strength or select a checkpoint. The original ephemeral raw artifacts are no longer available. |

See the [online evaluation record](research/online-evaluation.md#python-online-comparison-smoke-validation).

### Inputs and environment

Historical identities and tested platform: [inputs and environment](research/online-evaluation.md#inputs-and-environment).
They identify a *past* experiment, not supported dependency pins.

### Recreate the workload

Use the [historical workload recipe](research/online-evaluation.md#recreate-the-workload)
with a fresh, clean build and output directories outside the checkout.
This regenerates a comparable workload, not the original checkpoint bytes.

### Measurements

Read the [timings and terminal-score observations](research/online-evaluation.md#measurements)
only with their batch size, model, device, and seed constraints.

### Verification and retention

See [verification and retention](research/online-evaluation.md#verification-and-retention).
The checked-in fixtures preserve regression boundaries, **not** the missing raw
measurements or a guarantee of action-by-action equivalence.
