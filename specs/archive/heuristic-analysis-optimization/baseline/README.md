# Baseline behavioral trace corpus

Working material for the [heuristic analysis optimization Change](../spec.md).
This directory holds one behavioral evidence artifact and no performance
measurement; benchmark runs stay separate and are owned elsewhere.

## Fixed baseline

- Source: `c9202e33ac78cfb819def7339c3986c58cc3d13f`, a clean checkout of the
  commit the Change specification was written against.
- Policy: `HeuristicPolicy` in all four seats, built as
  `Runner::new([HeuristicPolicy::new(); 4])`.
- One `Runner` (and its four seat policies) is built once and reused for every
  game of a run.
- Games are driven only by `Runner::run_new`, so `Game::next_decision` and
  `Game::submit` remain the production runner's responsibility. The collector
  contains no game-driving, rule, timing, or benchmark logic.

## Frozen corpus

| Range | Seeds | Games |
| --- | --- | --- |
| A | `7..=38` | 32 |
| B | `10007..=10038` | 32 |

Both ranges are fixed inside `../tools/trace_corpus.rs` (`CORPUS`); they are not
command-line input. Changing them invalidates this artifact, exactly like
changing the comparison corpus.

## Artifact

`trace-corpus.txt` is the baseline corpus: 64 games and 3514 decisions
(1766 in range A, 1748 in range B). It is produced by
`../tools/trace_corpus.rs` and is time-free, so the same library and revision
reproduce it byte for byte.

## Collector interface

`../tools/trace_corpus.rs` is a change-scoped, isolated tool rather than a
production example. It takes no required arguments and writes both frozen
ranges to stdout:

```text
trace_corpus [--revision REV]
```

`--revision` labels the source version in each section header. When omitted the
label comes from `HEURISTIC_TRACE_REVISION`, and defaults to `unrecorded`. The
label never affects the digests, so corpora from different builds compare
directly apart from that header line.

## Reconstruction

`../tools/collect_trace.sh [REPO] [OUT_FILE]` performs the whole capture against
an arbitrary checkout (baseline or candidate worktree). It refuses to overwrite
an existing output corpus; pass a fresh `OUT_FILE` for every comparison. The
collector source is resolved relative to the script itself, not the baseline
checkout (which intentionally contains no optimization tools). Its steps are:

```bash
cargo build --release --lib --locked
rustc --edition 2021 -O -D warnings \
  --extern sichuan_mahjong_sim=target/release/deps/libsichuan_mahjong_sim-<hash>.rlib \
  -L dependency=target/release/deps \
  -o target/release/trace_corpus \
  specs/archive/heuristic-analysis-optimization/tools/trace_corpus.rs
HEURISTIC_TRACE_REVISION=$(git rev-parse HEAD) \
  target/release/trace_corpus > trace-corpus.txt
```

The baseline artifact was captured from a detached baseline worktree with the
same commands. Candidate corpora use the identical collector source and
invocation, only linked against the candidate release library; the variant
builder in the Change directory already compiles and runs it that way.

## Output format (`heuristic-trace-v1`)

The file contains two sections, one per seed range, separated by a blank line.
Each section is:

```text
protocol: heuristic-trace-v1
revision: <label>
policy: heuristic
seats: 4
seat_order: [0, 1, 2, 3]
first_seed: <seed>
games: <count>
digest_algorithm: fnv1a64
game_seed: <seed>
game_decisions: <count>
game_trace_len: <count>
game_end_reason: ThreeWinners | WallExhausted
game_scores: [a, b, c, d]
game_winners: [bool, bool, bool, bool]
game_trace_digest: <16 hex digits>
game_trace:
  <decision_index> <seat_index> <action>
  ...
corpus_games: <count>
corpus_decisions: <count>
corpus_digest: <16 hex digits>
```

Only non-time result fields are recorded; `policy_time`, `rules_time`, and
`total_time` are deliberately absent.

Canonical trace lines use single spaces and an explicit, `Debug`-independent
action encoding: `exchange:W1,T9,B2`, `missing:W`, `discard:T3`, `peng`,
`open-kong`, `concealed-kong:W5`, `added-kong:T7`, `hu`, `pass`. Tiles are a
suit letter (`W`, `T`, `B`) plus the one-based rank.

`game_trace_digest` is FNV-1a 64 over the game's canonical trace lines, each
followed by a newline. `corpus_digest` is FNV-1a 64 over the section's canonical
game records, meaning every line of a game before its `game_trace:` block
(including that game's trace digest), each followed by a newline. The revision
header is excluded from both.

## Captured checks

- Every game is validated before it is recorded: the trace must be non-empty
  and its length must equal the reported decision count. A violation is an
  error exit, never a shorter corpus.
- All 64 baseline games satisfy that check; trace lengths match decision
  counts, which range from 22 to 78 decisions per game (`37` for seed `7`,
  `53` for seed `10038`).
- The recorded digests were independently recomputed from the printed text with
  a separate FNV-1a 64 implementation, and every per-game and corpus digest
  matched.
- Two consecutive captures from the same baseline library were byte-identical,
  and a no-argument capture with the `unrecorded` label produced identical
  digests.
- `../tools/collect_trace.sh` run against the baseline worktree reproduced the
  committed artifact byte for byte.

## Comparing a candidate

Diff the candidate corpus against `trace-corpus.txt`; only the `revision:`
header lines may differ when the label is not set. Equal per-game
`game_decisions`, `game_trace_len`, `game_scores`, `game_winners`,
`game_end_reason`, `game_trace_digest`, and `corpus_digest` values mean the
candidate reproduced the baseline action sequence, results, and end reasons for
all 64 games. This corpus is correctness evidence only; it says nothing about
performance, which the Change measures separately.
