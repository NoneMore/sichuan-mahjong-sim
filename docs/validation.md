# Historical validation and research outcomes

This index links **point-in-time** validation and experimental conclusions,
not the present test count, CI status, or performance/strategy-strength
guarantee. For current build, test, Rust 1.75 MSRV and supported experiments
see [development](development.md) and
[CI](../.github/workflows/rust-ci.yml). Code and tests are the authority on
implemented behavior; [architecture](architecture/overview.md) describes
the implemented subsystem boundaries.

Detailed original measurements and methods remain in the
[pre-retirement snapshot](https://github.com/NoneMore/sichuan-mahjong-sim/tree/f57e48fdd39a9571371194b46d61bfba2b20d795).
The snapshot also contains completed `specs/archive/` Changes and historical
`docs/studies/` reports. Neither is a maintained description of the current
implementation. Earlier locally deleted raw samples, binaries, traces,
prototypes and scripts are **not** restored by the snapshot.

Detailed writeups are grouped by subsystem below. The original Git snapshots
linked above remain historical primary evidence; these files do not
replace their recorded revisions or recreate unretained raw artifacts.

## Game and runner baseline

Historical evidence: [read this section](research/native-validation.md#game-and-runner-baseline).

## Strategy comparisons

Historical evidence: [read this section](research/native-validation.md#strategy-comparisons).

## Exact-analysis changes and profiling

Historical evidence: [read this section](research/native-validation.md#exact-analysis-changes-and-profiling).

## Rust BC preparation smoke validation

Historical evidence: [read this section](research/offline-bc-validation.md#rust-bc-preparation-smoke-validation).

## Python dataset loader smoke validation

Historical evidence: [read this section](research/offline-bc-validation.md#python-dataset-loader-smoke-validation).

## Offline BC training smoke validation

Historical evidence: [read this section](research/offline-bc-validation.md#offline-bc-training-smoke-validation).

## Offline BC recovery validation

Historical evidence: [read this section](research/offline-bc-validation.md#offline-bc-recovery-validation).

## Recovery and new measurements

Historical evidence: [read this section](research/offline-bc-validation.md#recovery-and-new-measurements).

## Python online comparison smoke validation

Historical evidence: [read this section](research/online-evaluation.md#python-online-comparison-smoke-validation).

### Inputs and environment

Historical detail: [online evaluation inputs and environment](research/online-evaluation.md#inputs-and-environment).

### Recreate the workload

Historical detail: [online evaluation recreate the workload](research/online-evaluation.md#recreate-the-workload).

### Measurements

Historical detail: [online evaluation measurements](research/online-evaluation.md#measurements).

### Verification and retention

Historical detail: [online evaluation verification and retention](research/online-evaluation.md#verification-and-retention).
