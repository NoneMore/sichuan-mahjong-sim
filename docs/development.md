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

[The Linux prebuilt workflow](../.github/workflows/linux-prebuilt.yml) builds
and smoke-tests Linux x86_64 `bc-collect`/`bc-prepare` binaries and
`mahjong_bc`/`mahjong_eval` (`cp312-abi3`) wheels using a read-only token.
A separate publish job obtains write permission only after the verified bundle
is uploaded. Pushes to `main` and manual dispatch from `main` update the
**mutable** `prebuilt-latest` prerelease. Pushing a `vMAJOR.MINOR.PATCH` tag
creates a versioned Release without replacing an existing Release or tag.
Each bundle includes a commit manifest and SHA-256 checksum.

[Secret History Scan](../.github/workflows/secret-scan.yml) runs on PRs,
`main` pushes, weekly, and on demand. It fetches Git history and uses a pinned
Gitleaks binary to scan all reachable refs with secret values redacted. The
scan supplements, but does not replace, manual review of old branches, Actions
logs and release assets. See the [security guide](../SECURITY.md).

Run the relevant checks locally before submitting changes. A successful CI run
verifies that specific source revision; it does not establish a performance
guarantee or replay equivalence to earlier revisions. For behavior-changing
work, use focused invariant and deterministic-replay tests; for performance
claims, use paired, controlled measurements outside shared CI runners.
