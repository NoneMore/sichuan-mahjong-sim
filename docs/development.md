# Development and verification

This guide covers workspace setup and verification. The
[developer governance](governance.md) defines contribution conventions;
the [documentation map](README.md) links to runnable workflows. Module
READMEs own detailed CLI options and artifact contracts.

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

[The Linux prebuilt workflow](../.github/workflows/linux-prebuilt.yml) runs on
pushes to `main` and manual dispatch, publishing clean-provenance Linux x86_64
`bc-collect`/`bc-prepare` binaries and `mahjong_bc`/`mahjong_eval` (`cp312-abi3`)
wheels to the `prebuilt-latest` (or manually specified) GitHub Release.

Run the relevant checks locally before submitting changes. A successful CI run
verifies that specific source revision; it does not establish a performance
guarantee or replay equivalence to earlier revisions. For behavior-changing
work, use focused invariant and deterministic-replay tests; for performance
claims, use paired, controlled measurements outside shared CI runners.
