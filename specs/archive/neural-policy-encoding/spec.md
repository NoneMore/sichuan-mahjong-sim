# Neural policy encoding specification

Proposed v1 contract for single-decision encoding and semantic label mapping;
[design.md](design.md) explains the choices. Network implementation, training,
self-play orchestration, and dataset formats are outside this scope.

## Interface

- Input: a valid pending [`Decision`](../../../src/game.rs) from Game, covering
  all five decision stages. Only its player-visible fields may affect encoding.
  Validation of fabricated inconsistent decisions is outside the contract.
- Output: structured state features and a nonempty ordered candidate list. Each
  candidate contains its semantic [`Action`](../../../src/action.rs) and features.
- Encoding is deterministic and stateless, leaves Game unchanged, and uses no
  teacher analysis or simulated transitions.
- Single-decision output contains real candidates only. Tensor conversion,
  padding, and action selection belong to consumers.

## State

All seat references use clockwise distance from the deciding player:
self = 0, next clockwise = 1, opposite = 2, previous clockwise = 3. Retain fixed
positions after players win. Tile axes follow [`Tile::index`](../../../src/tile.rs):
Wan 1–9, Tong 1–9, Tiao 1–9; suit axes follow Wan, Tong, Tiao.

All numeric features use `f32`: per-kind counts divide by 4, concealed totals by
14, wall count by 108, and scores by 16 without clipping.
Categories use one-hot vectors and flags use 0/1. Meld-slot and response tile
fields are one-hot tile identities, not scaled physical counts; meld kinds
supply their physical multiplicities.

The following category orders are fixed v1 mappings, independent of future enum
reordering (indices start at zero):

| Category | Order |
| --- | --- |
| Missing suit | Wan, Tong, Tiao, Unpublished |
| Meld kind | Peng, OpenKong, AddedKong, ConcealedKong |
| Meld source | Self, NextClockwise, Opposite, PreviousClockwise, None |
| Exchange direction | Clockwise, CounterClockwise, Opposite |
| Stage | Exchange, MissingSuit, Turn, DiscardResponse, AddedKongResponse |
| Action kind | Exchange, ChooseMissing, Discard, Peng, OpenKong, ConcealedKong, AddedKong, Hu, Pass |
| Discard category | Unclaimed, MeldClaimed, HuClaimed |

Flatten state in the following row order; players follow relative-seat order and
the last array axis varies fastest. Total width is **1013**.

| Group | Width | Field order |
| --- | --- | --- |
| Own hand | 27 | Concealed tile counts |
| Players | 4 × 155 | Per player: concealed total (1), missing suit (4), score (1), has-won (1), four meld slots (4 × 37) |
| Discards | 4 × 3 × 27 | Counts indexed by source player, category, tile |
| Global context | 9 | Wall count (1), exchange direction (3), stage (5) |
| Response | 33 | Presence (1), tile (27), relative source (4), robbing-added-kong (1) |

Each meld slot is validity (1), tile (27), kind (4), source (5), in that order.
Own melds appear once in the player groups. Concealed totals describe physical
ownership, excluding virtual discard-winning tiles. Sort melds by ascending tile
index; zero-fill unused slots. A concealed kong has validity 1 and the None source
category. When no response exists, all 33 response fields are zero.

Classify every discard record once: **unclaimed** (no meld claim or Hu),
**meld-claimed**, or **Hu-claimed** (one or more winners). Include robbed-kong
records. In unscaled per-tile counts:

```text
public_owned = meld tiles + unclaimed records + Hu-claimed records
known_to_actor = own concealed tiles + public_owned + pending_added_tile
```

A multi-Hu record contributes one tile. `pending_added_tile` contributes one at
the response tile only during AddedKongResponse: it is publicly known but still
owned by the declarer's concealed hand, with the meld still a Peng. After
resolution it moves into the AddedKong meld or a robbed-kong record. Ordinary
response tiles are already in records. This derives known counts from existing
features; it neither adds a feature nor changes physical ownership.

Omit event order, exchanged-out tile memory, per-record winner associations,
historical robbed-kong flags, and payment history. Include neither a redundant
unseen-count vector nor exact shanten, ukeire, fan, or teacher rankings.

## Candidates and labels

- Ordinary candidate membership equals the offered action set, regardless of
  teacher preference.
- Exchange candidates are all unordered three-tile multisets from one eligible
  suit, with multiplicities bounded by the own hand. Canonicalize each triple
  into ascending tile-index order.
- Each legal semantic choice appears exactly once and can be submitted to the
  same pending Game by the deciding player.
- Sort lexicographically by `(kind_id, tile_ids, suit_id)` using unscaled integer
  parameters: `tile_ids` is the ascending sequence of three exchange tiles, one
  specified tile, or an empty sequence; `suit_id` is the chosen suit index or zero
  when unused. Ordering is independent of the offered list order. Candidate
  position is local, not a network feature or durable action identity.
- Obtain teacher labels from the original Game decision; teacher tie-breaking
  can depend on its offered order. Normalize the chosen action and match by
  semantic identity. The encoded result's `target_index(action)` returns
  `Result<usize, TargetNotCandidate>`; an absent target returns `Err`.

Within the supported Game domain, identical encoded states and canonical
candidates must imply identical normalized teacher choices.

Each candidate has 39 features, concatenated as:

| Group | Width | Meaning |
| --- | --- | --- |
| Kind | 9 | One-hot semantic action kind |
| Tile parameters | 27 | Parameter multiplicities divided by four |
| Suit parameter | 3 | One-hot missing-suit choice, otherwise zero |

| Action | Unscaled tile parameters | Suit parameter |
| --- | --- | --- |
| Exchange | Selected multiset, total three | Zero |
| ChooseMissing | Zero | Selected suit |
| Discard / ConcealedKong / AddedKong | One at the specified tile | Zero |
| Peng / OpenKong / Hu / Pass | Zero | Zero |

Kong parameters identify a tile kind, not four consumed copies. State context
supplies the opportunity for parameterless actions.

## Consumers and compatibility

BC uses cross-entropy over real candidates. Batch adapters may produce states
`[B, 1013]`, actions `[B, Kmax, 39]`, and padding masks `[B, Kmax]`; padding is
excluded from softmax. A single-candidate choice has probability one and zero BC
loss.

Models and encoded datasets identify an encoding version covering meanings,
scales, category and field order, layout, canonicalization, and candidate order.
Changes to these require a new version, even when tensor shapes are unchanged.
Consumers must not silently reinterpret incompatible encodings.

## Acceptance

Verify:

1. Determinism, input isolation, and unchanged Game state across all stages and
   calls interleaved between players or hands.
2. Relative-seat conversion, won-player positions, canonical meld slots, absence
   handling, and numeric fixtures for each feature group.
3. Physical ownership and actor-known counts for meld claims, multi-Hu, ordinary
   responses, and pending/successful/robbed added kongs.
4. Complete, unique candidates against ordinary offered sets and an independent
   exchange multiset enumeration; every candidate is accepted by a pending Game
   clone. Cover repeated tiles and multiple eligible suits.
5. Stable ordering under offered-list permutations and equivalent exchange
   permutations; teacher choices map uniquely and absent targets report failure.
6. All nine action feature forms, including exchange multiplicities and kong's
   single-tile parameter, plus frozen v1 schema/version fixtures.
7. Teacher projection equivalence: reconstruct teacher-relevant inputs from the
   encoding in tests, including Game's tie-breaking order, and compare teacher
   choices with the original decisions. Cover discard/kong ties, claim choices,
   and omitted record order and winner associations.
8. Seeded collision checks keyed by encoded state and canonical candidates;
   conflicting normalized teacher choices fail and retain both decisions as
   reproducible counterexamples. No sampled collisions is evidence, not a proof.
