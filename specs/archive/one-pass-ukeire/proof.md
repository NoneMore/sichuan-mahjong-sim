# One-Pass Ukeire — Correctness proof

This proves [spec.md](spec.md) for [`src/policy/analysis.rs`](../../../src/policy/analysis.rs).
All deficits use exact integer arithmetic.

## 1. Context, targets, and distance

A context `C = (M,s)` fixes melds `M` (`m = |M| <= 4`), missing suit `s`, target
size `n = 14 - 3m`, and per-suit, per-rank capacities `cap[r] = 4 - (meld copies of rank r)`, saturated at zero,
with `cap[s] = 0`. A physical hand obeys `H[t] + meld copies[t] <= 4`. Held missing-suit
copies may be positive, but every target has `T[t] = 0` on the missing suit. Costs count only
positive deficits, so held missing-suit and other surplus copies are free: target counts, not
held counts, are constrained.

`T_C` contains standard targets: per-rank counts at most `cap`, exactly `4-m` sets
(triplets or same-suit consecutive sequences) and one pair in any suit, so `|T| = n`; when `M`
is empty it also contains seven-pairs targets: `T[t] in {0,2,4}` on non-missing
kinds, `T[t] = 0` on the missing suit, and seven pair units, so `|T| = 14 = n`.
`T_C` depends only on `C` and is non-empty for every `m <= 4` (with four melds, an unmelded
kind supplies a pair). Define
`c_H(T) = sum_t max(0,T[t]-H[t])`, `D_C(H) = min_{T in T_C} c_H(T)`,
`Opt_C(H) = {T in T_C : c_H(T) = D_C(H)}`, and
`Mask_C(H) = {t : exists T in Opt_C(H), T[t] > H[t]}`; the mask is the low 27 bits of a `u32`
indexed by `Tile::index`, a union over optimal targets.

## 2. Improvement theorem

**Lemma 2.1.** For any kind `t` and `T in T_C`,
`c_{H+t}(T) = c_H(T) - [T[t] > H[t]]` (1). *Proof.* Only the `t` summand
changes: if `T[t] > H[t]` its deficit drops by one, otherwise both are zero. The
identity also holds at `H[t] = 4` with a false indicator. ∎

**Lemma 2.2.** If `H[t] < 4`, then `D_C(H)-1 <= D_C(H+t) <= D_C(H)`. *Proof.* By (1) each
fixed-target cost drops by at most one, giving the lower bound. An optimal `T` has cost
`D_C(H)-1` if `T[t] > H[t]` and `D_C(H)` otherwise; hence `D_C(H+t) <= D_C(H)`. ∎

**Theorem 2.3.** For a legal draw of kind `t` (`H[t] < cap[t]`, in particular a
non-missing kind), `D_C(H+t) < D_C(H)` iff `t in Mask_C(H)`. *Proof.* If some optimal
`T` has `T[t] > H[t]`, (1) gives `c_{H+t}(T) = D_C(H)-1`, so the distance drops.
Conversely, let `T'` be optimal for `H+t`. By (1), `c_H(T') = D_C(H+t) + [T'[t] > H[t]]`.
If `T'[t] <= H[t]`, then `c_H(T') = D_C(H+t) < D_C(H)`, contradicting the definition of
`D_C(H)`. Thus `T'[t] > H[t]`, `c_H(T') = D_C(H+t)+1 <= D_C(H)`, so `T'` is optimal for
`H` and `t` is in the mask. ∎

The mask is structural, independent of `public`/`unseen`; `D_C(H)=0` implies
`Mask_C(H)=0` because a zero-cost target has `T[t]<=H[t]` everywhere. Availability is applied separately.

## 3. Masked per-suit dynamic program

For one suit let `h[r]` be held counts, `k[r]` target counts, suit cost
`sum_r max(0,k[r]-h[r])`, and needed set `{r : k[r] > h[r]}`.

**State and transition.** Before rank `r`, state `(a,b,sets,pair)` records
sequences started at `r-2`/`r-1`, sets already chosen (including sequences with pending tails),
and pair placement; it stores
`(cost,mask)`, the minimum partial cost and union of local needed ranks over all
minimum-cost partial targets reaching it. The start `(0,0,0,0)` is `(0,0)`, other
states `(INF,0)`. At rank `r`, with `reserved=a+b`, enumerate `sequences`,
`triplets`, and `pair`; then `k[r]=a+b+sequences+3*triplets+2*pair`,
`e=max(0,k[r]-h[r])`, bit `r` iff `k[r]>h[r]`, successor
`(b, sequences, sets+triplets+sequences, pair_used+pair)`; replace on strictly lower `cost+e`,
OR on equal cost, and expand only reachable states.

**Invariant.** For every reachable state, `cost` is the minimum and `mask` the needed-rank union over all
minimum-cost partial selections reaching it. *Proof.* The empty selection is the base. Every
in-capacity choice is enumerated, so every partial selection reaches a successor; paths for the
same partial target share cost and mask because both depend only on `k[0..=r]`. The added cost
and bit depend only on the carried state, choice, and `h[r]`, so successor pairs are the image
of prefix pairs. A cheaper arrival replaces the pair; an equal-cost arrival ORs its mask, all
such arrivals being minimum-cost; a costlier arrival is dropped. The union is exact because
every minimum-cost prefix plus the fixed choice is achievable. Since successors depend only on
`(a,b,sets,pair)` and later held counts, every minimum-cost prefix extends by any legal suffix
with the same added cost and mask; keeping only minimum-cost prefixes is lossless. ∎

**A2 is exact.** `reserved > cap[r]` is rejected soundly because earlier sequences already
commit those tiles. `sequences <= cap[r]-reserved` because each new sequence consumes one
tile at `r`; `sequences = 0` for `r > 6` because the sequence would need a nonexistent rank;
and `sequences + triplets <= target_sets-sets`. The triplet loop stops when
`3*triplets+sequences+reserved > cap[r]`; the pair only adds tiles, so no pair choice fits and
larger triplet counts cannot recover. The pair is tested per candidate and rejected only when
the full `k[r] > cap[r]`. Every legal target is reachable and no illegal one is enumerated;
costs equal the plain table.

**INF and full table.** Unreachable entries are `(INF,0)` and are never expanded; every written
cost is finite, bits attach only to finite costs, and the equal-cost OR needs a finite cost, so
`INF` never contributes a phantom bit. `saturating_add` never saturates: a per-suit target has
at most 14 tiles and three entries at most 42, below `INF=255`. **MaskTable**
`[[(u8,u32);2];5]` is read at `(a,b)=(0,0)` for every `sets in 0..=target_sets` and
`pair in 0..=1`; pending tails cannot finish after rank 8 and are never read. Local bits are
`1 << r`; bits `9..` are zero.

## 4. Cross-suit merge

`combine[sets][pair]` stores `(cost,mask)` for the cheapest combination of processed suits
using exactly `sets` sets and `pair` pairs, with `mask |= suit_mask << (9*suit_index)`.
*Invariant.* `cost` is the minimum combined cost and `mask` the union of needed kinds over all
combinations attaining it.

*Proof.* The empty prefix is `(0,0)` at `(0,0)`. A combination with one more suit is a prefix
combination plus a table entry: costs add and shifted masks OR. For fixed `(sets,pair)` and
split, the minimum is the sum of the two minima, and the union over optimal combinations is the
OR of the optimal unions, because cost is additive and a combination's mask is the OR of its
parts. Merging with replace-on-less and OR-on-equal while skipping `INF` preserves the invariant. ∎

With all three suits, `combine[4-m][1] = (D_standard, Mask_standard)`: the pair flag sums to one
exactly once, and suits occupy consecutive 9-kind blocks, so `suit_index*9 + rank` is exactly
`Tile::index`. Each kind belongs to one suit, so the OR neither loses nor duplicates a needed kind.

## 5. Seven-pairs union

Requires `M` empty, so `cap[t]=4` on non-missing kinds. For held `h in {0,1,2,3,4}` and target
count `2u` (`u in {0,1,2}`), cost `g_h(u)=max(0,2u-h)` has non-decreasing marginals
`h=0:2,2; h=1:1,2; h=2:0,2; h=3:0,1; h=4:0,0`. With `P=sum floor(h/2)`, `O=sum(h mod 2)`,
and `K=max(0,7-P)` over the 18 non-missing kinds; this covers zero held counts, odd singles,
triples, quads, and an uncleared missing suit.

**Lemma 5.1.** `D_7 = K + max(0,K-O)`. *Proof.* Minimize `sum g_{h[t]}(u_t)` with `sum u_t=7`,
`u_t in {0,1,2}`. Each assignment is a prefix of marginals. If `P>=7`, seven zero marginals can be
selected prefix-closed and `D_7=0=K+max(0,K-O)`. Otherwise `K>0` and the seven cheapest are
prefix-selectable: all `P` zeroes, then `min(K,O)` ones, then `max(0,K-O)` twos. No prefix-closed
7-set costs less, since its elements are at least the corresponding order statistics; cost
`= 0*P + 1*min(K,O) + 2*max(0,K-O) = K+max(0,K-O)`. ∎

**Lemma 5.2.** Assume `K>0` (`P<7`). Every optimal target (1) takes all `P` zero units
(`u_t>=floor(h[t]/2)`); (2) if `K<=O`, extends exactly `K` distinct odd kinds by one unit (`u=1`
for `h=1`, `u=2` for `h=3`) and nothing else; (3) if `O<K`, extends every odd kind and buys
`K-O` units at cost two from kinds with `h<4`. *Proof.* (1) Otherwise some positive marginal is
selected; replace a selected positive marginal that is last for its kind by an unselected zero
(use the first unit of an `h=4` kind if its second is the zero). The result is prefix-closed and
cheaper, contradiction. (2) The remaining units must be cheapest: `K` of the `O` unit-cost
marginals, exactly the first unit of `h=1` and the second of `h=3`, one per odd kind. (3) All `O`
unit marginals must be selected; remaining cost-two units are both units of `h=0`, the second of
`h=1`, and the second of `h=2`. There are `2*#{h=0}+#{h=1}+#{h=2}=36-P-O` and
`K-O<=36-P-O`, so enough remain; fill kinds one unit at a time. ∎

**Theorem 5.3 (mask).** `K=0`: empty. `0<K<=O`: exactly the odd kinds (`h in {1,3}`). `O<K`: every
non-missing kind with `h<4` (odd kinds plus `h in {0,2}`). *Proof.* `K=0`: cost zero forces
`T[t]<=h[t]` everywhere. For `0<K<=O`, Lemma 5.2(2) extends `K` distinct odd kinds; each odd kind
can be chosen (it plus `K-1` of the other `O-1`, as `O>=K`), while `h=0` has `u=0`, `h=2` has
`u=1` (`T=2=h`), and `h=4` has `T<=4=h`. For `O<K`, Lemma 5.2(3) extends every odd kind, and
each `h=0`/`h=2` kind can receive one of the `K-O>=1` cost-two units in some optimal target; the
inventory leaves enough. `h=4` cannot be exceeded. ∎

The masked route shares the cost formula, so distances agree; it is skipped when `M` is non-empty,
matching `eval::is_winning`. `O<K` is unreachable for a missing-cleared hand with `|H|>=13`
(`P+O=sum ceil(h/2)>=ceil(|H|/2)>=7`) but reachable uncleared: four Tiao 1, three Tiao 2, six
distinct Wan singles with Tiao missing give `P=0,O=6,K=7`; the theorem assumes neither a cleared
suit nor a hand size.

Routes merge as `Opt_C` requires: lower distance and its mask; exact tie ORs both masks because
both families are then optimal; otherwise only the better family's mask.

## 6. Assembly, public/unseen, caches

`deficit_with_mask` computes both routes, keeps the lower distance, ORs masks on an exact tie, and
caches `(distance,mask)` by the full 27-count hand; the context is fixed for the evaluator's
lifetime, so the key is unambiguous. The `suit_mask_cache`/`hand_mask_cache` are separate from
`suit_cache`/`hand_cache`: a plain hit never answers a mask query and `deficit()` pays no mask
cost. The value is a pure function of counts, capacities, target sets, and missing suit, so fill
order, repeated lookups, and fresh versus reused evaluators change only how often the DP runs,
never its result. Nothing is cached across decisions; evaluators are local to a `choose` call,
and a fresh policy after `Game::clone` reproduces the analysis. `ukeire` reads `(D_C(H),Mask_C(H))`
once, debug-asserts the caller's `deficit`, and counts `max(0,4-H[t]-public[t])` per set bit, skipping missing-suit kinds.

The mask is structural; `unseen` only limits availability. Since `T[t]<=cap[t]`, no target needs a
kind held at or above `cap[t]` (including held missing-suit tiles, `cap[s]=0`), so such a kind has
no bit. For non-missing kinds, production `public` includes the player's own meld copies, so
`H[t]=cap[t]` gives `unseen[t]<=4-cap[t]-meld copies[t]=0`. Public exhaustion of any other
structurally improving kind leaves its bit intact but contributes zero.

## 7. `can_hu` equivalence and shortcut removal

**Lemma 7.1.** Let `C` be legal (`m<=4`, valid melds, no missing-suit meld, physical ownership
`H[t]+meld copies[t]<=4`) and `|H|=n`. Then `can_hu(H,M,s)` iff `D_C(H)=0`. *Proof.* (`=>`)
`can_hu` means no missing-suit tile and `eval::is_winning(H,m)`. A standard win decomposes into
`4-m` sets and a pair, and physical ownership puts all concealed counts within `cap`, so `H` is a
standard target with `c_H(H)=0`. A seven-pairs win requires `m=0`, counts in `{0,2,4}`, and seven
units, so `H` is a seven-pairs target with `c_H(H)=0`.

(`<=`) If `D_C(H)=0`, some `T in T_C` has `T[t]<=H[t]` for all `t`. Since `sum T[t]=n=sum H[t]`,
all inequalities are equalities and `T=H`. A standard `T` makes `H` standard-winning with `m`
melds; a seven-pairs `T` makes `m=0` and `H` a seven-pairs hand. Every target count on the missing
suit is zero and melds contain none by legality, so `can_hu` holds. ∎

**Lemma 7.2.** For `|H|=n-1`, `D_C(H)>=1`: otherwise `T[t]<=H[t]` for all `t`
gives `n=sum T[t]<=sum H[t]=n-1`.

**Corollary 7.3.** In `ukeire`, `H+t` has size `n`. If `can_hu(H+t)`, then Lemmas 7.1 and 7.2 give
`D_C(H+t)=0<1<=D_C(H)`, so the draw is already a strict improvement and a mask bit by Theorem 2.3. The old `can_hu` branch
was a redundant subset of the improvement test; with the same `unseen > 0` gate, replacing the
per-kind loop by one mask lookup preserves the counted set. The new code constructs no drawn hand and calls no `can_hu`, `eval`, or
`Hand::add` in `ukeire`; it reads the mask and counts unseen copies.
