# Development and verification

Use this guide for workspace setup and verification. For runnable workflows,
start from the [documentation map](README.md); module READMEs own detailed
CLI options and artifact contracts.

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

## Neural and Python workflows

The [BC pipeline](guides/bc-pipeline.md) provides the runnable
collection → preparation → training → evaluation sequence. Each stage links to
its module README for flags, format contracts, and recovery procedures.

### Python environment

Use the Git-ignored `.venv/` at the repository root for Python tooling. Reuse it
when present; otherwise create an isolated environment with `python -m venv .venv`.
In Git Bash on Windows, activate it with `source .venv/Scripts/activate`, or invoke
`.venv/Scripts/python.exe` directly. On POSIX, use
`source .venv/bin/activate`. Install dependencies into this environment, not the
global Python or user site-packages.

## Policy experiments

### Policy A/B comparison example

See [balanced policy comparison](guides/policy-comparison.md#policy-ab-comparison-example)
for commands, placements, and statistical interpretation.

### Heuristic benchmark example

See [heuristic benchmark](guides/policy-comparison.md#heuristic-benchmark-example)
for commands, cache lifecycle, and measurement boundaries.

## Continuous integration

Markdown links, anchors, and fenced-code balance are checked by the
[docs CI workflow](../.github/workflows/docs-ci.yml) using
`python3 scripts/check_docs.py` from the repository root. This check is
independent of Rust/Python dependencies and does not probe external URLs.


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
