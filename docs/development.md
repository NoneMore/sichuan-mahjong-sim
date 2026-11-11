# Development and verification

Use this guide for workspace setup, build checks, and current CI. For daily
workflows, follow the task guides linked below; for module-specific CLI flags
and file contracts, use the corresponding crate or package README.
The [documentation map](README.md) is the main navigation entry.

## On this page

| Task | Reference |
| --- | --- |
| Build and verify the workspace | [Running](#running) and [Continuous integration](#continuous-integration) |
| Run seeded simulation batches | [Self-play](guides/self-play.md) |
| Collect, prepare, train, and evaluate BC models | [BC pipeline](guides/bc-pipeline.md) |
| Compare policies or benchmark heuristics | [Policy comparison](guides/policy-comparison.md) |
| Resume games or check deterministic replay | [Checkpoint and replay](guides/checkpoint-replay.md) |

## Running


The workspace contains the dependency-free simulator core and a
[neural adapter crate](../crates/neural/README.md) that consumes the public core
API and owns its serialization, hashing, and filesystem dependencies.
Both declare Rust 1.75 as their minimum; use a recent stable toolchain locally
and leave minimum-version verification to CI:

```text
cargo build --workspace --release
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The root CLI's seeded runs, flags, seed behavior, and report fields are in
[Run seeded self-play](guides/self-play.md). The Rust library's `Runner` and
policy API remain in the [library guide](library.md).

## Neural and Python workflows

End-to-end execution order and artifact handoffs are in the
[behavior-cloning pipeline guide](guides/bc-pipeline.md). Each module's
README remains the authority for its commands and format details.

### Python environment

Use the Git-ignored `.venv/` at the repository root for Python tooling. Reuse it
when present; otherwise create an isolated environment with `python -m venv .venv`.
In Git Bash on Windows, activate it with `source .venv/Scripts/activate`, or invoke
`.venv/Scripts/python.exe` directly. Install dependencies into this environment,
not the global Python or user site-packages.

### Behavior-cloning collection

Use [teacher collection](../crates/neural/README.md#teacher-data-collection)
for `bc-collect` options, provenance, locking, and recovery. See the
[pipeline](guides/bc-pipeline.md#1-collect-teacher-episodes-in-rust) for
its place in the end-to-end workflow.

### Cache preparation

Use [dataset preparation](../crates/neural/README.md#dataset-preparation)
for `bc-prepare` options and the NumPy cache contract. The
[pipeline](guides/bc-pipeline.md#2-prepare-and-verify-the-numpy-cache) covers the
handoff to Python.

### Python offline behavior cloning

Use the [offline package guide](../python/README.md) for training, resuming,
checkpoint inspection, and frozen evaluation. Run
`python -m pytest python/tests` after installation; see the
[BC pipeline](guides/bc-pipeline.md#4-train-select-and-evaluate-a-frozen-policy)
for the sequence from cache to checkpoint.

### Python online evaluation

Use the [online package guide](../online/README.md) for installation,
PyO3 bindings, frozen comparison, and records. The binding is not part
of the root Rust workspace. The [pipeline](guides/bc-pipeline.md#5-compare-frozen-policies-in-the-online-environment)
covers its checkpoint handoff and seed-split constraints.

## Policy experiments

The standalone tools for balanced A/B comparisons and four-seat heuristic
benchmarks are documented in [policy comparison](guides/policy-comparison.md).
They do not add heuristic styles to the root simulator CLI.

### Policy A/B comparison example

See [balanced policy comparison](guides/policy-comparison.md#policy-ab-comparison-example)
for placements, flags, output files, and statistical interpretation.

### Heuristic benchmark example

See [heuristic benchmark](guides/policy-comparison.md#heuristic-benchmark-example)
for cache lifecycle, seed selection, reports, and measurement boundaries.

## Continuous integration

[The Rust CI workflow](../.github/workflows/rust-ci.yml) runs on pull requests
targeting `main`, pushes to `main`, and manual dispatch. Its stable job checks
formatting, warning-denied Clippy, Debug and Release tests across all workspace
targets (including executable examples and their tests), doctests, and
warning-denied rustdoc. The separate MSRV job checks all workspace targets with
Rust 1.75.0, the minimum declared by both packages. Builds and tests use the
committed root lockfile.

Run the relevant checks locally before submitting changes. A successful CI run
verifies that specific source revision; it does not establish a performance
guarantee or replay equivalence to earlier revisions. For behavior-changing
work, use focused invariant and deterministic-replay tests; for performance
claims, use paired, controlled measurements outside shared CI runners.
[Historical validation results](validation.md) document earlier environments
and observations rather than the present test count.
