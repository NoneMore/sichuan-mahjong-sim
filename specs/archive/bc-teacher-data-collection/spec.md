# Behavior-cloning teacher data collection specification

Historical scope for the implemented collector; [design](design.md) records the
rationale. Current usage and contracts live in the [crate guide](../../../crates/neural/README.md)
and [collection module](../../../crates/neural/src/collection/mod.rs).

## Collection

The neural-side collector uses public Game, Policy, and
[encoding](../../../crates/neural/src/encoding/mod.rs) interfaces to run four-seat
`HeuristicPolicy` self-play. Collect every seat and decision stage
without changing rules, teacher strategy, encoding, or core Runner. Training,
exploration, augmentation, and parallel collection are outside this Change.

For each original `Decision`, encode the pre-action state, call the teacher once,
resolve its choice with `target_index`, and submit that action. Record the sample
after acceptance. Preserve all legal candidates and the teacher's tie-breaks;
retain forced choices, `Pass`, and losing games without filtering or rebalancing.

## Identity and split

Episode identity is `(source_revision, encoding_version, seed)`. The revision is
the full lowercase Git commit ID of the compiled simulator, teacher, and collector
source; reusable collection uses committed source. Each Corpus fixes one revision
and encoding version. Foreign-source append fails with `SourceMismatch`.

Collect consecutive unsigned 64-bit seeds from `seed_start`, without wrapping.
Seeds are stored as 16 lowercase hex digits. Split algorithm
`seed-sha256-80-10-10-v1` is fixed:

```text
digest = SHA256(ASCII("bc-split-v1") || byte(0) || u64_big_endian(seed))
bucket = unsigned_big_endian(digest[0:8]) % 100
train: 0..79; validation: 80..89; test: 90..99
```

All decisions and derived trajectories of a seed share its split across revisions
and Corpus growth. Recollection must not duplicate committed episodes.

## Storage

Stream uncompressed UTF-8 JSONL into shards of a configured positive
`shard_games` complete episodes. Normal completion may publish a smaller final
shard. Boundaries never split a game.

`manifest.json` records `format_version=1`, `source_revision`,
`encoding_version=1`, `split_algorithm`, `seed_start`, positive `decision_limit`,
`shard_games`, and a `shards` list. Each entry contains `first_seed`, `games`,
`decisions`, and the file's `sha256`; its path is `shards/<first_seed>.jsonl`.
Seed ranges form one contiguous, nonoverlapping prefix. The manifest stores
per-shard metadata, not a per-episode index. Counts and seed ranges must match
the shard contents.

Each shard concatenates episode blocks, using the following JSON object types:

- `type="episode"`: `episode_id` as `[revision, encoding_version, seed]`, and `split`.
- `type="decision"`: `decision_index`, `actor_seat`, `stage`, `state`,
  `candidate_features`, `candidate_actions`, and `teacher_target_index`.
- `type="result"`: `decisions`, four seat-ordered `scores` and `winners`, plus
  `end_reason` (`"three_winners"` or `"wall_exhausted"`).

Decision indices start at zero within each game; seats are `0..3` and stages use
encoding v1 IDs. State has 1013 finite `f32` values; candidates have `N × 39`,
with positive `N` and target in `[0, N)`. Preserve values and canonical order.
Actions use `[kind_id, ...parameters]` with v1 IDs: three sorted tiles for Exchange,
one suit for ChooseMissing, one tile for Discard/ConcealedKong/AddedKong, otherwise
no parameters. Tile/suit IDs follow the encoder. Only state and candidate features
are model inputs; batching owns padding.

## Commit and recovery

Initialization, append, and recovery hold a nonblocking exclusive OS file lock on
`LOCK`; a second committer returns `CorpusBusy` without modifying the Corpus.
Ownership is released on process exit.

Validate each completed shard, move it to its final path, then atomically replace
the manifest to commit it. Readers consume only listed shards; committed files
remain immutable. Verify source, splits, shapes, target bounds, episode/count
completeness, and checksum when writing or reading a shard. Unsupported versions
and corrupt data are errors, not grounds to silently omit records.

Append/resume validates manifest metadata without rescanning historical decisions.
Ignore unlisted files and replay from the first uncommitted seed. A crash may
require replaying the unfinished shard, not the whole Corpus. Enforce
`decision_limit` per game; errors stop with episode and failure context. Never
substitute actions or skip games.
For fixed source and seed, inputs, labels, and terminal results must reproduce.

## Reporting and acceptance

Report games, decisions, candidate counts, per-stage total/multi-candidate counts,
and per-kind opportunity/selection counts, including zero support. Count each
kind once per decision. Exclude forced choices from primary choice metrics.
Pilot measurements record throughput, teacher/encoding/I/O cost, storage size,
build, environment, seeds, and cache settings.

Acceptance covers:

- Seeded replay and round-trip records, including exchange canonicalization,
  forced choices, losses, source isolation, and fixed seed splits.
- Interrupted shard/manifest publication and retry: only complete committed games
  remain visible, with no duplicates; append after a smaller final shard starts
  at the next seed.
- Two committers: the second fails; collection resumes after the holder exits.
- Scale metadata: with 1,000 games per shard, one million games need 1,000 shard
  entries, and resume does not read all prior decision records.
