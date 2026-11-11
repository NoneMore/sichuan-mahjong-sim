# mahjong-eval

Online environment binding for running frozen neural policies against the
Sichuan mahjong simulator. The Rust core still owns rules, candidates, legal
actions, and settlement; this package owns scheduling and policy inference.
Offline training and dataset tooling remain in the separate
[`mahjong-bc`](../../python/README.md) package.

## Status

The extension exposes complete single-game lifecycle control:

- `Env(decision_limit=10000)` creates one handle. The limit bounds accepted
  decisions per game; it is not cumulative across resets.
- `Env.reset(seed)` starts a fresh `Game` and returns its opening `Decision`.
  Every reset advances the packed decision generation, so identities from an
  earlier game are stale.
- `Env.step(decision_id, candidate_index)` resolves the local index through the
  cached canonical action list, commits exactly one action, and returns either
  the next `Decision` or a `Terminal`.
- `Decision` exposes `decision_id`, `actor_seat`, `stage`, `candidate_count`,
  `state`, and `candidate_features`. `state` is a NumPy-owned writable
  C-contiguous `float32` array of shape `(1013,)`; `candidate_features` is a
  NumPy-owned writable C-contiguous `float32` array of shape
  `(candidate_count, 39)`, in canonical candidate order.
- Arrays are independent snapshots; Python mutation cannot change authoritative
  game state, cached actions, or later decisions.
- `stage` is the frozen encoding stage code: `0` exchange, `1` missing suit,
  `2` turn, `3` discard response, `4` added-kong response. Every stage and every
  forced single-candidate choice is an explicit pending decision.
- `decision_id` packs the reset generation in the high 32 bits and the decision
  sequence in the low 32 bits. The first decision after a reset has sequence 1.
- `Terminal` exposes seat-ordered `scores`, `winners`, `end_reason`
  (`three_winners` or `wall_exhausted`), the seed, and the number of accepted
  decisions including forced choices. A win does not terminate the game early;
  reset after termination is explicit.
- Stale identities and out-of-range indices raise
  `StaleDecisionError`/`CandidateIndexError` before mutation. Decision-limit
  exhaustion raises `DecisionLimitError` with seed and accepted-count context,
  distinct from a normal `Terminal`. Unstarted, terminal, or otherwise invalid
  handle use raises `EnvironmentStateError`.

`import mahjong_eval` does not load the extension. Native symbols are imported
lazily, so `mahjong_eval.inference` and its numeric fixtures remain usable in an
installation that does not build or load `mahjong_eval._native`.

## Teacher, split, and compiled identity

`Env.teacher_index(cache=None)` asks the native `HeuristicPolicy` to choose on
the original Rust pending `Decision`, resolves the semantic choice through the
encoder's canonical candidate lookup, and returns the local index without
advancing the game. `step` then commits exactly that index. Repeated queries do
not change the pending decision or the returned choice.

`TeacherCache(budget_bytes=...)` is an explicit `HeuristicCache` context. Pass
it as `Env.teacher_index(cache=...)` or `native_teacher_replay(seed, cache=...)`
to share derived masked-suit tables across handles and calls. Zero disables the
cache. Caching never changes a teacher action; `status()` reports availability,
allocation, hit/miss, and eviction counters.

`native_teacher_replay(seed, cache=None, decision_limit=...)` drives one game
with the native teacher and returns a JSON-compatible mapping. Its `trace`
entries include the decision identity, actor, stage, candidate count, canonical
`teacher_index`, and the semantic `action` (`kind`, `kind_id`, and v1 wire
parameters, not only a local index). The settlement fields are `decisions`,
`scores`, `winners`, and `end_reason`. Python teacher drivers can compare their
own trace and settlement against this execution under the same Rust runtime.

`split_bucket(seed)` and `seed_split(seed)` expose the frozen Rust collection
algorithm `seed-sha256-80-10-10-v1`.

`compiled_source()` reports the executing extension's compiled identity in two
parts:

- `extension` is the complete extension-scoped identity. It records the full
  Git commit and dirty flag when Git was available at build time, plus
  `source_sha256` over the sorted relative paths and bytes of every
  `crates/python`, `src`, and `crates/neural` source, manifest, and lockfile
  listed in `source_files`. For each sorted relative path, the hash updates the
  path bytes, a NUL byte, the file length as little-endian `u64`, and the file
  content. `source_complete` is false when a source input was missing at build
  time.
- `neural` is the neural crate's separate collection build provenance. It does
  not cover the binding crate and is not a substitute for `extension`.

The content identity is computed by the extension build script from the build
source tree, never from the caller's working directory. The build script watches
the three scopes and Git metadata, so a source or commit change refreshes the
recorded values. Wheels built from a checkout or an sdist contain this compiled
metadata; pair it with the loaded `mahjong_eval._native.__file__` when recording
an installation.

## Layout and independent builds

The PyO3 crate lives in [`crates/python`](../../crates/python/Cargo.toml) as its
own Cargo workspace and is excluded from the root workspace. Ordinary core,
neural, collector, and offline training builds therefore neither resolve nor
link Python. The package metadata here builds the extension as
`mahjong_eval._native` and adds this Python package.

Use the repository `.venv` (Python 3.12 in the verified environment):

```text
# From the repository root:
python -m pip install "maturin>=1.16,<2"
cd online
python -m maturin build --release --locked --out dist
python -m maturin sdist --out dist
python -m pip install --force-reinstall --no-deps <produced-wheel>
cd ..
python -m pytest online/tests
```

Keep native build output in the extension's ignored `crates/python/target/` (the
maturin default) or pass `--target-dir` so it cannot contend with the root Rust
target directory.

## Verification

The extension is its own workspace, so root `cargo ... --workspace` commands do
not cover it. Run its checks explicitly; use an independent target directory if
the root workspace is building concurrently:

```text
cargo +1.75.0 check --manifest-path crates/python/Cargo.toml --all-targets --locked
cargo clippy --manifest-path crates/python/Cargo.toml --all-targets --locked -- -D warnings
cargo +1.75.0 test --manifest-path crates/python/Cargo.toml --all-targets --locked
cargo fmt --manifest-path crates/python/Cargo.toml -- --check
python -m pytest online/tests
```

## Verified build matrix

| Component | Verified value |
| --- | --- |
| Python | CPython 3.12.10 x86-64, `abi3-py312` wheel |
| NumPy | 2.5.4 |
| PyO3 | 0.23.5 |
| rust-numpy | 0.23.0 |
| maturin | 1.16.0 |
| Rust | 1.75.0 for the extension and the root workspace |

The extension build uses a committed independent lockfile. `rustc-hash` is
pinned to 2.1.1 there because later releases declare an MSRV above 1.75.
The verified Python minimum is 3.12, matching the checked-in environment and
the `abi3-py312` wheel tag; other Python versions are not claimed.

## Fixture

`crates/python/tests/fixtures/first_decision_seed_0.json` is generated from the
public Rust core and encoder APIs and stores exact little-endian float32 bytes.
The Rust test and `online/tests/test_environment.py` both compare against it, so
the binding, candidate order, and feature values cannot drift independently.

`crates/python/tests/fixtures/environment_trace_seed_61.json` stores one
complete scripted game: every pending decision across all five stages,
including an added-kong response, a win followed by continued play, and the
terminal settlement. The Python test replays the recorded canonical indices and
compares every state, candidate row, order, identity, and terminal field. The
Rust test independently regenerates the same trace.

`crates/python/tests/fixtures/teacher_replay_seed_0.json` records the same
seed-0 teacher game generated directly from the core and teacher APIs. Each
entry stores the teacher's canonical index and semantic action plus exact
state/candidate SHA-256 hashes; the terminal block stores scores, winners, and
reason. The Python test reproduces the trajectory through `Env.teacher_index`
and `Env.step`, compares the hashes, and separately compares
`native_teacher_replay`'s semantic trace and settlement.

Regenerate a fixture deliberately with:

```text
UPDATE_FIRST_DECISION_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test first_decision
UPDATE_ENVIRONMENT_TRACE_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test environment_trace
UPDATE_TEACHER_REPLAY_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test teacher_replay
```
