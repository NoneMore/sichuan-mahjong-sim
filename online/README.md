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

## Multi-game scheduling

`mahjong_eval.scheduler.run_games(specs, neural=..., teacher_cache=...)` plays
several games in one process. Each `GameSpec(seed, neural_seats)` assigns the
frozen policy to two seats; the other seats use the native `HeuristicPolicy`
through one explicit cache shared by every handle and query. Pending neural
decisions are grouped by policy and scored in `batch_size` chunks; indices are
routed back by chunk position, which matters because concurrent handles reuse
decision ids. Forced single-candidate decisions are committed as index zero
without inference or a teacher query, while still counting toward decisions.
Up to `concurrent_games` handles are active and refilled as games finish, so
games can sit at different stages and decision counts. A win for one seat never
ends a game early; only the environment terminal result does.

`on_terminal(game)` is called as each game finishes, before the next game is
started; a callback or write failure stops scheduling instead of substituting
an action. `trace=False` (the default) retains no per-decision data;
`trace=True` retains only small scalar entries (identity, actor, stage,
candidate count, assigned policy, forced flag, index), never state or candidate
arrays.

The result exposes completed games in plan order with per-game decision,
forced-decision, and inference-row counts, plus totals recomputable from those
games. Timings separate environment reset, environment step, teacher, and
inference seconds and include call counts. CUDA is synchronized before and
after each inference batch, so inference time covers queued GPU work while the
other timers exclude it. Any game, teacher, inference, or terminal-callback
failure raises `SchedulingError` carrying the seed, placement, and decision
context of the affected pending decision.

## Formal comparison

`mahjong-eval compare` (same as `python -m mahjong_eval compare`) runs the
six placement games for every requested seed and publishes a reproducible run
directory:

```text
mahjong-eval compare --checkpoint ../bc-run/best.pt --split validation \
  --seeds 2,10 --out ../online-eval-validation \
  --batch-size 64 --concurrent-games 6 \
  --decision-limit 10000 --teacher-cache-bytes 27787264 --device cpu
```

| Argument | Meaning |
|---|---|
| `--checkpoint FILE` | Selected `best` checkpoint. `last` and plain policy exports are rejected. |
| `--split validation or test` | Frozen Rust seed split. Validation runs are diagnostics; test runs are the frozen comparison. Neither selects a checkpoint. |
| `--seeds LIST` | Comma-separated unique `u64` seeds; every seed must belong to `--split` under the Rust algorithm. |
| `--out DIR` | Run directory. It may exist but must not contain run artifacts. |
| `--batch-size N` | Maximum neural decisions per inference batch. Default 64. |
| `--concurrent-games N` | Simultaneously active environment handles. Default 6. |
| `--decision-limit N` | Per-game accepted-decision bound. Default: native 10000. |
| `--teacher-cache-bytes N` | One shared teacher cache budget; `0` disables it. Default: native 27,787,264. |
| `--device cpu or cuda[:index]` | Frozen inference device. Default `cpu`. |

The checkpoint file is read once, hashed, and loaded from those exact bytes, so
the recorded content identity cannot change even if the path is replaced.
Frozen inference adopts the already loaded policy; the path is not read again.
The plan is published before the first game and records checkpoint identity and
role, encoding/model versions, installed `mahjong-eval`/`mahjong-bc` source
hashes, the executing extension's compiled source plus environment, training
provenance kept separate from evaluation identity, the split algorithm and each
seed bucket, dependencies, selected device, batching/concurrency, decision
limit, and teacher cache configuration.

Each game appends one `games.jsonl` record as it finishes. The CLI re-reads and
cross-checks all planned records, then writes `costs.json` atomically and only
afterwards leaves the writer context. A clean close marks `state.json`
`complete`; `summary.json` is then recomputed from the persisted plan and
records by `finalize_run`. Costs include recomputable decision, forced,
inference-row, and teacher-query counts, per-game accounting, and separate
environment reset/step, teacher, and inference timings.

Any game, inference, record, costs, or consistency failure stops the run:
completed records and the plan remain, `state.json` stays `incomplete`, and no
costs or summary is published. `records.finalize_run` independently refuses
such a run. If only the final summary write fails after a clean close, the run
stays `complete`, no summary is published, and a later `finalize_run`
recomputes it from the same plan and records.

### Installation order

Frozen inference and comparison need the offline `mahjong-bc` package. It is a
repository-local distribution, not a PyPI name, so install it first:

```text
python -m pip install -e ./python
python -m pip install -e ./online          # or ./online[evaluation]
```

`[project.optional-dependencies] evaluation = ["mahjong-bc"]` documents that
dependency; because the local package is already installed, the extra adds no
download. A native-only install (wheel without the extra) still imports and
drives `Env`; only `mahjong_eval.inference`, `scheduler`, and `cli` need the
offline package and PyTorch.

## Layout and independent builds

The PyO3 crate lives in [`crates/python`](../../crates/python/Cargo.toml) as its
own Cargo workspace and is excluded from the root workspace. Ordinary core,
neural, collector, and offline training builds therefore neither resolve nor
link Python. The package metadata here builds the extension as
`mahjong_eval._native` and adds this Python package.

The crate keeps PyO3's `extension-module` link mode behind its own
`extension-module` feature, so plain `cargo check` and `cargo test` stay
linkable without a Python host. Only the maturin build below passes
`features = ["extension-module"]`; Python imports and the online tests therefore
use the built wheel or editable install, not a raw `cargo build` output.

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
# Optional read-only probe against an external teacher corpus:
MAHJONG_EVAL_ACCEPTANCE_CORPUS=/path/to/corpus python -m pytest online/tests
```

`online/tests` uses the installed or editable extension; a bare `cargo build`
output is not importable because it does not enable the `extension-module`
feature.

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
