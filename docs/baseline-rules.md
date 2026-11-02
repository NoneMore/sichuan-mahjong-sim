# Baseline rules

This document defines the **intended fixed gameplay and settlement rules**
for this Sichuan mahjong ("blood battle", 血战到底) baseline. It is normative
for game semantics, not evidence that a particular implementation or API
already behaves as described. See [governance](governance.md) for that
distinction. The baseline does not cover local variants or policy/runner
implementation details.

Normative key words: **must**, **may**, and **illegal** describe engine
behaviour. A "player" is one of four seats.

## 1. Deck, seats, and deal

- The deck has 108 tiles: ranks 1 through 9 in each of three suits, four copies
  of every tile kind.
- Suit names and encoding order are Wan, Tong, Tiao. Tile indices are
  `0..27`; the text form is `1W`, `5T`, `9B`.
- Seats are numbered `0..=3` in clockwise turn order. Seat 0 is the dealer.
- The dealer receives 14 tiles and each other seat receives 13 tiles. All
  tiles are dealt face down; only a player's own concealed tiles are visible to
  that player.
- There are no flower tiles, winds, dragons, or honors.

## 2. Exchange three and missing suit

1. Before any player selects exchange tiles, the game chooses one of three
   exchange directions uniformly at random: clockwise, counter-clockwise, or
   opposite. The chosen direction is public.
2. The direction is the **giving** direction. Every player passes exactly three
   tiles to the seat named by that direction. The same direction applies to all
   four players, so a player receives three tiles from the inverse seat.
   - Clockwise: pass to the next seat in clockwise order, receive from the
     previous seat.
   - Counter-clockwise: pass to the previous seat, receive from the next seat.
   - Opposite: pass across the table; opposite seats swap three tiles.
3. The three selected tiles must all belong to the same suit and must be
   present in the selecting player's hand, respecting duplicate copies.
   Selecting a tile the player does not hold is illegal.
4. Exchange selections are simultaneous. No player's selection may depend on
   another player's selected tiles. The game collects all four selections and
   then applies the exchange atomically.
5. Every 13- or 14-tile hand necessarily contains at least three tiles of some
   suit, so a legal exchange selection always exists. Exchange is mandatory;
   there is no pass.
6. After the exchange, each player privately and permanently chooses one of the
   three suits as their **missing suit** (定缺). The choice is not restricted to
   a suit the player currently holds. The chosen suit is fixed for the rest of
   the hand.

## 3. Missing-suit obligations

- While a player holds any tile of their missing suit, they must discard a
  missing-suit tile whenever they discard.
- A player may discard an arbitrary concealed tile only after holding zero
  missing-suit tiles.
- A player may not peng, kong, or win while holding any missing-suit tile.
- A meld may never contain a missing-suit tile. This applies even after the
  player's concealed missing-suit tiles have been cleared.

## 4. Turn structure and claims

### 4.1 Opening turn

After the exchange and missing-suit declaration, the dealer acts first. The
dealer already holds 14 tiles and therefore does not draw. The opening turn is
**not discard-only**: the dealer may discard, declare one or more ordinary
concealed kongs (each fully resolved before the next declaration, and each
requiring a replacement wall tile), or declare an ordinary self-draw win if the
winning-shape and missing-suit conditions permit. An opening win is scored as a
normal self-draw with the standard event bonus; there is no heaven-hand bonus
or special multiplier. An added kong cannot occur on the opening turn because
the dealer has no peng yet.

### 4.2 Normal turn

After the opening discard and its responses, play proceeds clockwise among
players who have not won. A normal turn consists of:

1. Draw one tile from the head of the wall.
2. Optionally declare one or more concealed/added kongs, resolving each fully
   before the next declaration.
3. Discard one tile.
4. Resolve claims on that discard.

If a player cannot draw because the wall is empty, the hand ends by wall
exhaustion after any pending response is resolved (see section 7).

### 4.3 Draws and the wall

- Normal draws take the first tile from the head of the wall.
- Every kong replacement draw takes the last tile from the tail of the wall.
- A kong may be declared only if at least one wall tile remains for its
  replacement draw. If the wall is empty, the kong is illegal and the player
  must choose another action.
- If a replacement draw empties the wall, the current player still discards and
  claims on that discard are resolved. The hand ends when the next draw is
  attempted and no tile is available.

### 4.4 Legal claims

Only the following claims exist:

- **Hu** on another player's discard.
- **Peng** on another player's discard.
- **Open kong** (直杠) on another player's discard.
- **Hu** by robbing an added kong (抢杠).
- **Hu** on one's own draw or kong replacement draw.

There is no chi.

Claim priority is:

```text
Hu > Kong > Peng
```

- If one or more players can and do claim Hu, all of them win
  ("一炮多响"). The discard cannot also be penged or konged. Each winner is
  scored independently. The discarder pays each winner separately.
- If no Hu is claimed, at most one kong claim applies. If no kong is claimed,
  at most one peng claim applies.
- If multiple players claim the same non-Hu priority tier, the claimant nearest
  to the discarder in clockwise order is selected. The tile-count limits make
  simultaneous legal peng claims, or simultaneous legal kong claims, on the
  same discard impossible in practice; the rule is retained for determinism.
- A player may decline Hu with **Pass** and still claim Peng or Kong on the same
  discard if those claims are otherwise legal.

After a discard with one or more Hu claims, all winners leave the hand and the
discard is consumed. The next action is a draw by the next active seat
clockwise after the discarder, unless the hand ends because only one active
player remains. The discarder does not draw again on that discard.

#### 4.4.1 Discard ownership and multiple winners

The public discard list is a global, ordered history of physical tiles. Every
discarded or robbed added-kong tile is counted exactly once:

- If the tile is claimed as a peng or open kong, it becomes part of the
  claimant's meld. The discard record keeps the public claim association and is
  not additional ownership.
- If the tile is claimed by Hu, one or more winners reference that single
  record. Each winner's scoring hand is evaluated with a virtual copy of the
  tile, but the physical tile is counted once, not once per winner. This is
  how a one-discard, three-winner hand (一炮多响) remains tile-conserving.
- An unclaimed discard and a robbed added-kong tile likewise remain one
  physical tile in the discard history. A self-draw winning tile stays in the
  winner's concealed hand.

### 4.5 Peng

A peng claim takes the discarded tile plus two matching tiles from the
claimant's hand. The claimant becomes the current player and must discard
immediately, without drawing. The meld is public.

### 4.6 Kongs

There are three kong declarations:

- **Open kong** (直杠): claim a discard with three matching tiles in hand. The
  discarder pays 2 points to the kong declarer. The declarer then draws a
  replacement tile from the wall tail and discards. An open kong cannot be
  robbed.
- **Added kong** (补杠): on the declarer's turn, add the fourth tile to an
  existing peng. Before the kong is finalized, any player who can win on that
  tile may rob it (抢杠). If robbed, the added kong is cancelled, no kong
  payment is made, the existing peng remains, and the robbed tile is treated as
  the declarer's discard for purposes of the win and turn order. If not robbed,
  the kong is finalized; every active opponent pays 1 point; the declarer draws
  a replacement tile from the wall tail and discards.
- **Concealed kong** (暗杠): declare four matching concealed tiles on the
  declarer's turn. A concealed kong cannot be robbed. Every active opponent
  pays 2 points. The declarer draws a replacement tile from the wall tail and
  discards.

A declared concealed kong is public: the tile is known to all players. Multiple
kongs may be declared consecutively if each is legal and enough wall tiles
remain for each replacement draw. Kong payments are recorded per payer because
they may be refunded at wall exhaustion.

### 4.7 Passing a win (过手胡)

- When a player can win and declines the win, that player may not declare Hu
  again until after their next draw. Declining includes choosing **Pass** and,
  when Hu is legal, choosing a lower-priority Peng or Kong claim instead. Any
  draw, whether a normal draw or a kong replacement draw, clears the lock.
- If the declined win was on the player's own draw, the lock lasts until the
  following draw.
- The lock applies to all win opportunities in between, including discards and
  added-kong robberies.
- Passing does not prevent the player from claiming Peng or Kong.

## 5. Winning shapes

A winning hand contains four sets and one pair, or seven pairs. Declared melds
count as sets.

- **Standard**: four sets plus one pair. A set is a triplet, a kong, or a
  sequence of three consecutive ranks in one suit. Melds are triplets/kongs;
  concealed tiles may form triplets or sequences.
- **Seven pairs**: exactly 14 concealed tiles, no declared melds, every tile
  count even, and seven pair units. Four identical tiles count as two pairs.
- Open hands can win with the standard shape. Seven pairs requires a fully
  concealed hand.
- When a hand satisfies both seven pairs and a standard decomposition, it is
  classified and scored as seven pairs. The all-pungs fan is not awarded to a
  seven-pairs win.
- Kongs count as one set for shape evaluation. A winning concealed part has
  `14 - 3 * meld_count` tiles; a kong's fourth tile is replaced by a draw and
  does not change that count.
- A win is legal only when the player holds no missing-suit tiles in the
  concealed hand or any meld.

## 6. Fan and payments

### 6.1 Fan table

| Item | Fan |
|---|---:|
| Base win | +1 |
| Pure one suit | +2 |
| Seven pairs | +2 |
| All pungs | +1 |
| Each root | +1 |
| Event bonus (one only) | +1 |
| **Total cap** | **5** |

- **Pure one suit** means every physical tile in the concealed hand and all
  melds belongs to one suit. There are no honors, so a one-suit hand is pure.
- **Seven pairs** is defined in section 5.
- **All pungs** means every set in a standard win is a triplet or kong. The pair
  is unrestricted. Seven pairs is not all pungs.
- A **root** is a tile kind for which the player owns all four physical copies
  across the concealed hand and declared melds. Each such tile kind gives +1,
  including a kong, and including a peng plus the fourth tile in hand. Each tile
  kind contributes at most one root.
- **Event bonus** is exactly one of:
  - Self-draw (自摸): win on a normal draw.
  - After-kong (杠上花): win on a kong replacement draw.
  - Kong-discard (杠上炮): win on a discard made by a player immediately after
    that player's kong replacement draw.
  - Robbing-kong (抢杠): win on the tile of an added kong.
  Event bonuses never stack. An after-kong win is not also scored as a normal
  self-draw; a robbing-kong win is not also scored as a plain discard.
- The total is summed first and then capped at 5. The cap applies once per win,
  not per fan item. Points paid equal the capped fan; there are no additional
  multipliers.

### 6.2 Win payments

Let `F` be the capped fan total of a winner.

- **Self-draw and after-kong**: every other active player pays `F` points to
  the winner. Players who already won do not pay.
- **Discard, kong-discard, and robbing-kong**: only the discarder, or the player
  whose added kong was robbed, pays `F` points. Other players pay nothing.
- **Multiple winners**: each winner receives the full `F` for their own hand
  from the discarder. The discarder pays each winner separately; there is no
  splitting or shared cap.

Win payments are settled when the Hu is resolved.

### 6.3 Kong payments

- Concealed kong: each active opponent pays 2.
- Added kong: each active opponent pays 1, only if the kong is not robbed.
- Open kong: the discarder pays 2.
- A peng has no payment.

"Active opponent" means a player who has not yet won. Kong income is recorded
per payer so that it can be refunded at wall exhaustion.

## 7. End of hand

A player who wins leaves the hand immediately and no longer acts. The hand ends
when either:

1. Only one active (not yet won) player remains; or
2. The wall is exhausted and the next required draw cannot be made.

When only one active player remains, the hand ends by that condition. The
end-of-wall checks and kong refunds in section 7.2 do **not** run in that case.

### 7.1 Wall exhaustion

Wall exhaustion occurs when a draw is required but no wall tile remains, after
resolving the discard and claims that preceded the draw attempt. All hands are
revealed for settlement. Settlement then applies, in any order that produces
the same net result:

1. Flower-pig payments.
2. Big-call payments.
3. Kong-income refunds.

### 7.2 Flower pig (查花猪)

A player is a **flower pig** if they have not won and still hold at least one
missing-suit tile. Each flower pig pays 16 points to every non-flower-pig
player, including players who already won. With multiple flower pigs, each pig
pays each non-pig separately.

### 7.3 Big call (查大叫)

Only players who have not won and are not flower pigs participate.

- A participant is **tenpai** if adding exactly one legal tile to their concealed
  hand produces a legal winning shape, with declared melds counted. The pure
  shape query respects the four-copy limit for the player's own concealed hand
  and melds; tiles held by other players are outside this query.
- Missing-suit restrictions are already satisfied by non-flower-pig
  participants; a missing-suit tile is never a legal winning tile.
- Each non-tenpai participant pays each tenpai participant that tenpai
  participant's **maximum hypothetical discard-win fan**: the maximum capped fan
  over all of that player's legal waits, scored as a plain discard win (no event
  bonus).
- Tenpai participants do not pay each other.

### 7.4 Kong-income refund (退税)

At wall exhaustion, a player who has not won and is either a flower pig or not
tenpai must return **all** kong income they received during the hand. The
refund is paid back to the original payers according to the recorded kong
ledger; it is not a payment to the other players in general. A player who won,
or who is a non-flower-pig tenpai player, keeps their kong income.

## 8. Explicitly not implemented

The baseline does not include:

- chi;
- call transfer (呼叫转移);
- heaven-hand or earth-hand (天胡/地胡) bonuses or special timing;
- bottom-of-the-sea/last-tile bonuses (海底);
- 血流成河 or multi-round continuations;
- honors, flowers, or bonus tiles;
- additional local fan such as 金钩钓, 门清, 独钓, 十八罗汉, or 杠上开花
  variants beyond the single event bonus above;
- dealer rotation, wind rounds, or match-level scoring;
- configurable rule sets.
