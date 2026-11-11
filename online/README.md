# mahjong-eval

Run frozen PyTorch policies against the native `HeuristicPolicy`. Rust owns game
rules, legal candidates, encoding and settlement; Python owns synchronous
multi-game scheduling, inference and comparison records. Offline training stays
in the independent [`mahjong-bc`](../python/README.md) package.

For the upstream data and checkpoint handoff, see the [BC pipeline guide](../docs/guides/bc-pipeline.md).

## Install and run

Use the repository `.venv` with Python 3.12 or later. Install the repository-local
offline package first; it supplies PyTorch and the shared checkpoint contract:

```text
python -m pip install -e "./online[evaluation]"
python -m mahjong_eval version

```

Run a [complete online comparison](../docs/guides/bc-pipeline.md#5-compare-frozen-policies-in-the-online-environment)
with the BC pipeline's example, or use the options below.

Formal comparison requires the offline-selected `best` checkpoint, not `last`
or a plain policy export. Only the checkpoint is needed, not its training data
or run directory. Each seed plays six games with neural seats in pairs
`[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]` and heuristic policies elsewhere.
Validation seeds support repeated diagnostics; test seeds support frozen final
comparison. Online scores never select the checkpoint.

| Argument | Meaning |
|---|---|
| `--checkpoint FILE` | Selected `best` checkpoint. |
| `--split validation\|test` | Required Rust corpus split; every seed must belong to it. |
| `--seeds LIST` | Required comma-separated unique `u64` seeds. |
| `--out DIR` | Required directory with no existing run artifacts. |
| `--batch-size N` | Maximum neural decisions per inference batch; default 64. |
| `--concurrent-games N` | Maximum active handles; default 6. |
| `--decision-limit N` | Accepted decisions per game; default 10,000. |
| `--teacher-cache-bytes N` | Shared cache budget; default 27,787,264 bytes, `0` disables it. |
| `--device cpu\|cuda[:index]` | Inference device; default `cpu`. |

`python -m mahjong_eval compare` is the equivalent module entry point. A native-only
installation can omit the `evaluation` extra; inference and comparison require
`mahjong-bc`. Importing `mahjong_eval` loads native symbols lazily.

## Environment and teacher API

```python
from mahjong_eval import Env, TeacherCache, Terminal

env = Env(decision_limit=10000)
cache = TeacherCache(budget_bytes=0)
pending = env.reset(7)
while not isinstance(pending, Terminal):
    index = env.teacher_index(cache=cache)  # query only; does not advance
    pending = env.step(pending.decision_id, index)
print(pending.scores)
```

Each handle owns one game. `reset(seed)` starts a fresh game and returns a
`Decision`; `step(id, index)` commits one cached canonical action and returns
the next decision or terminal result. All stages and forced choices are explicit;
one seat winning does not end play. Reset after termination is explicit.

A `Decision` exposes `decision_id`, `actor_seat`, `stage`, `candidate_count`,
`state` and `candidate_features`. State shape is `(1013,)`; candidates have shape
`(candidate_count, 39)`. Both arrays are owned, writable, C-contiguous float32
snapshots in canonical order; mutation cannot change the game or cached actions.
Stage codes are `0` exchange, `1` missing suit, `2` turn, `3` discard response,
`4` added-kong response. Identities pack reset generation and decision sequence
in the high and low 32 bits, respectively; sequence starts at 1 after reset.

`Terminal` contains `seed`, accepted `decisions` including forced choices,
seat-ordered `scores`, four Boolean `winners`, and `end_reason`
(`three_winners` or `wall_exhausted`). Stale identities and invalid indices raise
`StaleDecisionError` / `CandidateIndexError` before mutation. Limit exhaustion
raises `DecisionLimitError` with seed/count context rather than a terminal;
unstarted or terminal handle misuse raises `EnvironmentStateError`.

`teacher_index(cache=None)` chooses on the original Rust decision and maps its
semantic action to a canonical index. `TeacherCache` shares memoization across
handles without changing choices; `status()` reports allocation and counters.
`native_teacher_replay(seed, cache=None, decision_limit=...)` returns a complete
semantic action trace and terminal result for cross-language comparison.
`split_bucket(seed)` and `seed_split(seed)` expose `seed-sha256-80-10-10-v1`.

## Frozen inference and scheduling

`inference.FrozenPolicy` loads a compatible `best` or `policy` checkpoint, or
adopts an existing `LoadedCheckpoint` through `from_loaded`. It rejects `last`,
freezes eval-mode weights, and uses shared label-free padding with float32 inputs
and Boolean masks. Greedy ties choose canonical-first; non-finite valid logits
stop evaluation. Single-candidate decisions return index zero without inference.

`scheduler.run_games(specs, neural=..., teacher_cache=...)` runs `GameSpec(seed,
neural_seats)` games, refills finished handles, groups pending decisions by policy
and scores bounded chunks. Returned indices route by originating handle and
chunk position, not decision id alone. Forced choices bypass inference and
teacher queries but retain counts. `on_terminal(game)` runs as each game finishes;
a callback failure stops scheduling before another game starts.

Results include per-game accounting, recomputable totals and environment reset,
step, teacher and inference timings. CUDA is synchronized around inference batches.
Optional `trace=True` retains scalar decision metadata, never feature arrays.
Execution failures raise `SchedulingError` with seed/placement/decision context.

## Records and provenance

The CLI reads checkpoint bytes once for both loading and SHA-256 identity, then
adopts those weights without rereading the path. It writes:

- `state.json`: initially `incomplete`; a clean record-stream close marks it
  `complete`.
- `plan.json`: published before play; fixes seeds/placements, checkpoint role,
  versions/configuration, Python and Rust source identities, separate training
  provenance, split buckets, dependencies, device, batching, limit and cache.
- `games.jsonl`: one terminal record flushed and fsynced as each game finishes.
- `costs.json`: atomic timing/count report after persisted results are checked
  against the games actually played, while the writer is still open.
- `summary.json`: recomputed from persisted plan/records after successful close.
  Primary metric is mean neural-seat terminal net score; uncertainty uses seed
  blocks, with SE unavailable for one seed.

Game, inference, record or costs failures retain completed records and leave state
incomplete without publishing a summary. `records.finalize_run(directory)` refuses
such runs. If only summary publication fails after a clean close, records remain
complete and that function can retry. `records.summarize_run(directory)` provides
read-only recomputation; it does not authorize publication of an incomplete run.

Python identities hash the imported package sources, not the working directory.
`compiled_source()` reports the extension's build-time content hash over sorted
relative source paths, a NUL byte, little-endian `u64` content length and bytes for
`crates/python`, `src` and `crates/neural`. `source_complete` flags missing inputs.
Git metadata is attached only for an actual source checkout; an sdist does not
borrow an outer repository. The neural crate's collection provenance is separate
and does not identify the full extension.

## Build and verify

[`crates/python`](../crates/python/Cargo.toml) is a standalone workspace excluded
from the root workspace, preserving independent core/collector/offline builds.
Maturin enables its `extension-module` feature; ordinary Cargo test executables
link Python normally. Use the package build, not raw Cargo output, for imports:

```text
python -m pip install "maturin>=1.16,<2"
cd online
python -m maturin build --release --locked --out dist
python -m maturin sdist --out dist
cd ..
python -m pip install --force-reinstall --no-deps <produced-wheel>

cargo fmt --manifest-path crates/python/Cargo.toml -- --check
cargo clippy --manifest-path crates/python/Cargo.toml --all-targets --locked -- -D warnings
cargo +1.75.0 check --manifest-path crates/python/Cargo.toml --all-targets --locked
cargo test --manifest-path crates/python/Cargo.toml --all-targets --locked
python -m pytest online/tests
```

Root workspace checks do not cover the extension. Its independent lockfile pins
`rustc-hash` to 2.1.1 for Rust 1.75. The package requires Python 3.12+
and NumPy 2+, while supported build/toolchain behavior must be checked against
the current CI and lockfile. Verify the selected platform and dependency combination using current CI
and local tests rather than older platform measurements.

Checked-in Rust/Python fixtures cover the first decision, all-stage scripted
replay including continued play after a win, and semantic teacher replay. Rust
regeneration tests use `UPDATE_FIRST_DECISION_FIXTURE`,
`UPDATE_ENVIRONMENT_TRACE_FIXTURE` or `UPDATE_TEACHER_REPLAY_FIXTURE` set to `1`
with the corresponding `first_decision`, `environment_trace` or `teacher_replay`
test target; regenerate only for deliberate changes. To also check an external
corpus, set `MAHJONG_EVAL_ACCEPTANCE_CORPUS=/path/to/corpus` when running pytest.
