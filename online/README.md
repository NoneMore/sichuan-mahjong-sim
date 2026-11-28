# mahjong-eval

Run synchronous plans of Sichuan mahjong games with caller-assigned policies.
Rust owns game rules, legal candidates, encoding and settlement. Python
coordinates game execution, policy selection and 2N2T comparison records.
Frozen inference uses the independent [`mahjong-bc`](../python/README.md) package.

For the upstream data and checkpoint handoff, see the [BC pipeline guide](../docs/guides/bc-pipeline.md).

## Install and run

Use the repository `.venv` with Python 3.12 or later. Install the repository-local
offline package first; it supplies PyTorch and the shared checkpoint contract:

```text
python -m pip install -e ./python
python -m pip install -e "./online[evaluation]"
python -m mahjong_eval version
```

Run a [complete online comparison](../docs/guides/bc-pipeline.md#5-compare-frozen-policies-in-the-online-environment)
with the BC pipeline's example, or use the options below.

Each seed plays six games with greedy neural seats in pairs
`[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]` and heuristic policies elsewhere.
Validation seeds support repeated diagnostics, and test seeds support the
frozen final comparison.

| Argument | Meaning |
|---|---|
| `--checkpoint FILE` | Compatible `best` or `policy` checkpoint supplied by the caller. The `last` role is reserved for training recovery. |
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

## Execution API

```python
from mahjong_eval import TeacherCache
from mahjong_eval.execution import Choice, GameSpec, NativeTeacher, execute_games


class FirstCandidate:
    def select(self, requests):
        return [Choice(0, metadata=request.seat) for request in requests]


accepted_indices = []
result = execute_games(
    [
        GameSpec(game_id, 7, ("first", "teacher", "first", "teacher"))
        for game_id in ("deal-7-a", "deal-7-b")
    ],
    policies={"first": FirstCandidate(), "teacher": NativeTeacher()},
    teacher_cache=TeacherCache(budget_bytes=0),
    on_decision=lambda event: accepted_indices.append((event.request.game_id, event.index)),
)
assert [game.game_id for game in result.games] == ["deal-7-a", "deal-7-b"]
```

Supply a finite, nonempty plan with unique string game IDs, unsigned 64-bit
seeds and four registered policy IDs in seat order. A deal and assignment may
repeat under different game IDs. Keep each registered policy's semantics stable
through the run. All native teacher registrations share the supplied cache.
Omitting it creates a disabled shared cache.

Ordinary selectors implement `select(requests)`. Each request includes the
game ID, seed, seat, policy ID and owned player-visible `Decision` from
Encoding v1. Return one local integer candidate index or `Choice(index,
metadata)` per request in batch order. A single-candidate decision uses
index zero and bypasses selection, teacher queries and sampling.

The kernel fills and refills handles in plan order. Each round visits teacher
groups first, then ordinary policy groups. Within each tier the first pending
request in active game order determines group order. Ordinary chunks follow
that order and filter forced rows after chunking. The kernel validates every
choice before stepping each originating handle in active game order.

`on_decision(event)` receives the accepted index, forced flag, optional choice
metadata and pre-action request after a successful step. The request carries
the same owned snapshot seen by the selector. Forced events have empty
metadata. Observers may retain snapshots and metadata. `on_terminal(game)`
runs after the last accepted event and before refill. Callbacks are synchronous
and must leave execution and policy state alone. Results contain validated
settlement and per-seat counts in plan order. `trace=True` adds scalar decision
entries. `diagnostics=True` adds accepted and forced counts by policy and seat,
eligible selector rows and batches, teacher queries and timings. Selector counts
measure calls at the request boundary.

`ExecutionError` stops a run with game and decision context where available.
If a game still has a pending decision at its limit, execution stops before
another round of selection. Already accepted steps and callback effects remain
in effect. Callers own recovery and persistence around their callbacks.

## Frozen inference and scheduling

`inference.FrozenPolicy` loads a compatible `best` or `policy` checkpoint, or
adopts an existing `LoadedCheckpoint` through `from_loaded`. Register it
directly as a greedy selector, or call `select_indices(decisions)` on native
snapshots. `inference.SamplingSelector(policy, generator=generator)` samples the
legal-mask softmax at temperature 1 with a caller-supplied `torch.Generator`
on the policy device. Frozen inference runs in eval mode without gradients and
rejects nonfinite valid logits. Greedy ties use canonical candidate order.
Sampling replay requires identical policy and generator states, plan, batching
and deterministic execution conditions, including device and dependencies.

`scheduler.run_games(specs, neural=..., teacher_cache=...)` adapts the same
kernel to `scheduler.GameSpec(seed, neural_seats)` for 2N2T comparison. It
projects the legacy results, scalar traces and inference counts and synchronizes
CUDA around measured inference batches.

## Records and provenance

The checkpoint SHA-256 identifies the bytes loaded for the run.

- `state.json` starts as `incomplete`. A clean record-stream close marks it
  `complete`.
- `plan.json` fixes seeds and placements, actual checkpoint role and identity,
  Python and Rust source identities, split buckets, dependencies, device,
  batching, limit and cache before play. Training provenance can be null for
  policy exports.
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

Python identities hash the imported package sources. `compiled_source()` reports
the extension's build-time source hash for `crates/python`, `src` and
`crates/neural`. `source_complete` flags missing inputs. Git metadata requires
an actual source checkout. The neural crate records collection provenance separately.

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

Checked-in Rust/Python fixtures cover the first decision, all-stage scripted
replay including continued play after a win, and semantic teacher replay. Rust
regeneration tests use `UPDATE_FIRST_DECISION_FIXTURE`,
`UPDATE_ENVIRONMENT_TRACE_FIXTURE` or `UPDATE_TEACHER_REPLAY_FIXTURE` set to `1`
with the corresponding `first_decision`, `environment_trace` or `teacher_replay`
test target; regenerate only for deliberate changes. To also check an external
corpus, set `MAHJONG_EVAL_ACCEPTANCE_CORPUS=/path/to/corpus` when running pytest.
