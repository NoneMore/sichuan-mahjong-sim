# One-Pass Ukeire — Correctness proof

This document proves the contract in [spec.md](spec.md) and ties every step to
the implementation in [`src/policy/analysis.rs`](../../../src/policy/analysis.rs).
It covers five claims:

1. the improvement theorem for a fixed context;
2. the masked per-suit dynamic program (state merge, exact tie handling,
   unreachable states);
3. the cross-suit merge and local/global bit mapping;
4. the seven-pairs optimal-target union;
5. `can_hu(H) <=> D_C(H) == 0` for legal size/meld premises, which lets the
   direct-win shortcut in `ukeire` be removed.

Terms in **bold** are the names used in the code.

## 1. Notation and context

A **context** `C = (M, s)` fixes

- the player's declared melds `M` with `m = |M| <= 4`,
- the declared missing suit `s`,
- the per-suit, per-rank target capacity
  `cap[r] = 4 - (copies of r in M)`, saturated at zero, with `cap[s] = 0`.

The **target size** is `n = 14 - 3m`. A physical hand `H` is a 27-vector of
concealed counts with `H[t] + meld copies[t] <= 4` for every kind `t`
(legal physical ownership). Target capacity is distinct: held missing-suit
copies may be positive even though every target has zero missing-suit copies.
Production `ukeire` uses `|H| = n - 1`, and its drawn hand `H + t` has size `n`;
the distance queries also support other sizes. Let `T_C` be the fixed family
of legal targets:

- **standard targets**: `T` with per-rank counts bounded by `cap`, exactly
  `4 - m` sets (triplets or same-suit consecutive sequences) and exactly one
  pair, so `|T| = n`; the pair may lie in any suit;
- **seven-pairs targets**, only when `M` is empty: `T[t] in {0, 2, 4}` for every
  non-missing `t`, `T[t] = 0` for the missing suit, and `sum_t T[t] / 2 = 7`
  units, so `|T| = 14 = n`.

`T_C` depends only on `C`, never on `H`, and it is non-empty for every `m <= 4`
(the existing `debug_assert` in `deficit` relies on this; e.g. with four melds
the target is a pair of an unmelded kind). Define

```text
c_H(T) = sum_t max(0, T[t] - H[t])          target deficit
D_C(H) = min_{T in T_C} c_H(T)              exact distance
Opt_C(H) = { T in T_C : c_H(T) = D_C(H) }    optimal targets
Mask_C(H) = { t : exists T in Opt_C(H) with T[t] > H[t] }
```

`Mask_C(H)` is encoded as the low 27 bits of a `u32`, indexed by `Tile::index`.
Surplus held copies are free, which is why the sum only counts positive
deficits and why `D_C` ignores held missing-suit tiles. The mask arguments
below constrain target counts, not held counts, so they apply unchanged to
uncleared missing-suit hands.

Every function below computes `D_C` exactly; no heuristic shanten estimate is
involved.

## 2. The improvement theorem

**Lemma 2.1 (single-copy effect on a fixed target).** For any kind `t` and any
target `T in T_C`,

```text
c_{H+t}(T) = c_H(T) - [T[t] > H[t]].                               (1)
```

*Proof.* Only the summand of kind `t` changes. If `T[t] > H[t]` then
`max(0, T[t] - (H[t]+1)) = T[t] - H[t] - 1`, otherwise `T[t] <= H[t]` implies
`T[t] <= H[t] + 1` and both deficits for that kind are zero. Additivity of the
sum gives (1). Note that `T[t] <= 4` always, so the identity also holds for
`H[t] = 4`; there the precondition is false. ∎

**Lemma 2.2 (bounded drop).** `D_C(H) - 1 <= D_C(H+t) <= D_C(H)` for every
kind `t` with `H[t] < 4` (so the drawn hand is a hand).

*Proof.* For any `T`, (1) gives `c_{H+t}(T) in {c_H(T), c_H(T) - 1}`, so
`c_{H+t}(T) >= c_H(T) - 1 >= D_C(H) - 1`. Taking the minimum over `T` gives the
lower bound. For the upper bound, let `T` be optimal for `H`. If `T[t] > H[t]`,
then `c_{H+t}(T) = D_C(H) - 1`; otherwise `c_{H+t}(T) = D_C(H)`. In both cases
`D_C(H+t) <= D_C(H)`. ∎

Lemma 2.2 also shows that `D_C(H+t)` can only equal `D_C(H)` or be exactly one
less, so the value is never ambiguous.

**Theorem 2.3 (improvement = mask bit).** Let `t` be a kind with a legal draw
(`H[t] < cap[t]`, in particular not the missing suit). Then

```text
D_C(H+t) < D_C(H)   <=>   t in Mask_C(H).
```

*Proof.*

(`<=`) If `t in Mask_C(H)`, some optimal `T` has `T[t] > H[t]`. By (1),
`c_{H+t}(T) = D_C(H) - 1`, and since `D_C(H+t) <= c_{H+t}(T)`, the distance
strictly drops.

(`=>`) Assume `D_C(H+t) < D_C(H)`. Let `T'` be optimal for `H+t`. By (1),
`c_H(T') = c_{H+t}(T') + [T'[t] > H[t]] = D_C(H+t) + [T'[t] > H[t]]`.
If `T'[t] <= H[t]`, then `c_H(T') = D_C(H+t) < D_C(H)`, contradicting the
definition of `D_C(H)`. Hence `T'[t] > H[t]`, and then
`c_H(T') = D_C(H+t) + 1 <= D_C(H)`, so `T' in Opt_C(H)` and `t` is in the
mask. ∎

Consequences used by production code:

- `Mask_C` is a *structural* set: `T_C` does not depend on `public`/`unseen`
  ownership, so publicly exhausted kinds keep their bit.
- If `D_C(H) = 0`, then no target count can exceed a held count (otherwise its
  cost would be positive and the minimum zero would be unattainable), so
  `Mask_C(H) = 0`; with `D_C(H+t) >= 0`, nothing improves, as expected.
- The mask is independent of the drawn tile's availability; `unseen` is applied
  separately in `ukeire`.

**Game-semantics corollary (post-draw discard).** For a non-winning drawn hand
`H'` of size `n`, `min_{d held in H'} D_C(H' - d) = D_C(H')`. Every removal can
only raise a fixed target's deficit, so the minimum is at least `D_C(H')`. Let
`T` be optimal for `H'`. If `T = H'` then `D_C(H') = 0` and `H'` is winning
(see section 7); otherwise, since `|T| = |H'|`, some kind has `T[d] < H'[d]`,
and discarding `d` leaves `c_{H'-d}(T) = c_{H'}(T) = D_C(H')`. Hence the best
post-draw discard realizes `D_C(H')`, which justifies comparing `D_C(H+t)` with
`D_C(H)` for the policy's ukeire. A winning drawn hand (`D_C(H') = 0`) is taken
as `Hu` instead.

## 3. Masked per-suit dynamic program (`suit_table_with_mask`)

A standard target decomposes by suit: each suit independently contributes a
multiset of triplets and sequences plus, in exactly one suit, one pair. For one
suit write `h[r]` for the held counts and `k[r]` for the target counts. The
per-suit cost is `sum_r max(0, k[r] - h[r])`, and the per-suit needed set is
`{ r : k[r] > h[r] }`.

**State.** After processing ranks `0..=r`, a state is `(a, b, sets, pair)`:
`a`/`b` sequences started at `r-1`/`r`, `sets` completed sets, `pair in {0,1}`
whether the pair is placed. Each state stores a pair `(cost, mask)`:
`cost` is the minimum partial cost over all partial targets reaching the state
(after ranks `0..=r`), and `mask` is the union of the local needed ranks of all
those minimum-cost partial targets. The initial state `(0,0,0,0)` holds
`(0, 0)`; every other state starts at `(INF, 0)`.

**Transition.** For rank `r` and a state, enumerate `sequences` new sequences,
`triplets` triplets and `pair` copies of the pair at `r`. The target count at
the rank is

```text
k[r] = a + b + sequences + 3 * triplets + 2 * pair,
```

which is the current `reserved = a + b` tails plus the tiles chosen now. The
increment is `e = max(0, k[r] - h[r])` and the local bit `r` is needed exactly
when `k[r] > h[r]`. The successor state is
`(b, sequences, sets + triplets + sequences, pair_used + pair)`. The update is

```text
cost' = cost + e
mask' = mask | (1 << r)      if k[r] > h[r]
        mask                 otherwise
```

written with replace-on-lower-cost and union-on-equal-cost.

**Invariant.** For every state after rank `r`, `cost` equals the minimum and
`mask` the union of needed ranks over all minimum-cost partial selections
reaching that state.

*Proof by induction over ranks.* The base state is empty, so `(0, 0)` is
`(min cost, union of masks)` trivially. Assume the invariant for all states
after rank `r-1`. A partial selection reaching a successor consists of a prefix
together with the choices `(sequences, triplets, pair)` at rank `r`, and the
enumeration below visits every in-capacity choice, so every partial selection is
realized by at least one path (distinct decompositions of the same rank counts
may both be visited; they produce the same `(cost, mask)` because both depend
only on the rank counts). Its cost is `cost_prefix + e` and its mask is
`mask_prefix | bit`, where `e` and the bit depend only on `a`, `b`, the choice at
rank `r`, and `h[r]`, not on the rest of the prefix. Hence the achievable pairs
for the successor are exactly the image of the prefix pairs, and

- a strictly cheaper arrival replaces the stored pair, since no costlier arrival
  can be part of a minimum;
- an equal-cost arrival unions its mask, because all such arrivals are
  minimum-cost partial selections and each realizes its own needed set;
- a costlier arrival cannot be part of any minimum and is dropped.

The invariant is preserved for every successor, which completes the induction.
The union is exact rather than an over-approximation: every minimum-cost prefix
combined with the fixed choice is itself an achievable partial selection, so
every mask bit in the union is realized. ∎

**Future equivalence.** The transitions available from a state depend only on
`(a, b, sets, pair)` and the held counts of later ranks, never on how the prefix
reached it (the per-rank increment and bit are functions of the carried tails
and the new choice). Therefore any minimum-cost prefix can be extended by any
legal suffix, and the cost and mask added by a suffix are the same for all
prefixes of the state. Keeping only minimum-cost prefixes is thus lossless:
replacing a costlier prefix by a minimum-cost one strictly lowers the total, so
a costlier prefix can never be part of a global minimum. This is precisely the
property used by the merge rule in the code.

**Enumeration bounds (A2) are exact.** The `reserved > cap[r]` rejection is
sound: `a + b` tiles for `r` are already committed by sequences started at
`r-2`/`r-1`, so every completion exceeds capacity. The bounds
`sequences <= cap[r] - reserved` and the triplet/tail test are sound because
`k[r] >= 3 * triplets + sequences + reserved`; if that minimum alone exceeded
the capacity no pair choice can fit, and the triplet loop may stop. The pair
branch is still tested per candidate and rejected only when `k[r] > cap[r]`.
Every legal target is therefore reachable and no illegal target is enumerated;
the masked table's costs are identical to the plain table's (also checked
against a full enumeration in the test modules).

**`INF` cannot produce a phantom mask.** Unreachable entries are `(INF, 0)`.
Every written cost is finite: it is `cost_prefix + e` with a finite prefix
(`INF` entries are not expanded because only active/reachable states are
iterated) and a finite increment. A mask bit is only ever attached to a finite
cost, and the union branch only runs for equal finite costs. Consequently an
entry's mask is zero whenever its cost is `INF`, and no `INF` entry contributes
bits to the cross-suit merge. `saturating_add` never saturates for legal inputs:
a per-suit target contains at most 14 tiles and every selected count is capped
by `cap`; held counts can only lower its cost. Even the loose bound of 42 for
three independently selected suit entries is below `INF`.

**Full table.** The final `MaskTable` is read at `(a, b) = (0, 0)` for every
`sets in 0..=target_sets` and `pair in 0..=1`; states with pending tails cannot
be completed after rank 9 and are never read. The type
`MaskTable = [[(u8, u32); 2]; 5]` therefore carries the complete masked table,
and bits are suit-local (rank `r` is `1 << r`, bits 9.. are zero).

## 4. Cross-suit merge (`standard_deficit_with_mask`)

`combine[sets][pair]` after processing a set of suits stores `(cost, mask)` for
the cheapest combination of those suits' table entries that uses exactly `sets`
sets and `pair` pairs. The mask is the union of the shifted suit masks,
`mask |= suit_mask << (9 * suit_index)`.

*Invariant.* `cost` is the minimum combined cost and `mask` is the union of
needed kinds over all combinations attaining it.

*Proof by induction over suits.* The empty prefix is `(0, 0)` at `(0, 0)`.
Suppose the invariant holds for the processed suits. Every combination with one
more suit consists of a combination of the prefix and one table entry, with cost
`cost_prefix + cost_suit` and mask `mask_prefix | (mask_suit << shift)`. For
fixed `(sets, pair)` and fixed split the minimum over combinations is the sum of
the two minima, and the union over optimal combinations is the union of the two
optimal mask unions, because cost is additive and the mask of a combination is
the OR of the parts. Merging the candidate states with replace-on-less and
union-on-equal (skipping `INF` operands) therefore maintains the invariant for
every combined state. ∎

Applying this with all three suits yields
`combine[4 - m][1] = (D_standard, Mask_standard)`: the pair flag sums to one
exactly once across suits, and shifted local bits coincide with the global
`Tile::index` because suits occupy consecutive 9-kind blocks
(`suit_index * 9 + rank`). A kind belongs to exactly one suit, so the OR neither
loses nor duplicates a needed kind.

The masked analysis is a separate function with separate caches. The plain
`standard_deficit` / `suit_table` path keeps its `u8`-only state, so a plain
`deficit()` call pays no mask cost. `deficit_with_mask` cross-checks its
distance against `deficit` with a `debug_assert`; the caches can never change a
value because the DP is a deterministic function of the counts and the fixed
context, and the caches are pure memoization.

## 5. Seven-pairs optimal-target union (`seven_pairs_deficit_with_mask`)

Only legal with `M` empty, so `cap[t] = 4` for non-missing kinds. For each kind
with held `h in {0,1,2,3,4}` the target count is `2u` with `u in {0,1,2}`, and
the cost is `g_h(u) = max(0, 2u - h)`. The marginal cost of the `u`-th unit is

```text
h = 0:  2, 2
h = 1:  1, 2
h = 2:  0, 2
h = 3:  0, 1
h = 4:  0, 0
```

so each kind's marginal sequence is non-decreasing. Let

```text
P = sum_t floor(h[t] / 2)     free pair units
O = sum_t (h[t] mod 2)        odd kinds, one unit each at cost one
K = max(0, 7 - P)             units still required
```

**Lemma 5.1 (cost).** `D_7 = K + max(0, K - O)`.

*Proof.* The task is `min sum_t g_{h[t]}(u_t)` subject to `sum u_t = 7`,
`u_t in {0,1,2}`. Every assignment corresponds to a per-kind prefix of the
marginal list, and the cost is the sum of the selected marginals. The 7 smallest
marginals of the whole multiset can be selected in a prefix-closed way: all `P`
zeroes come first, then `min(K, O)` of the `O` ones, then `max(0, K - O)`
twos. No prefix-closed 7-element set can cost less, because each of its
elements is at least the corresponding order statistic. The sum is
`0 * P + 1 * min(K, O) + 2 * max(0, K - O) = K + max(0, K - O)`. ∎

**Lemma 5.2 (structure of optimal targets).** Every optimal seven-pairs target

1. takes all `P` zero-cost units, i.e. `u_t >= floor(h[t]/2)` for every kind;
2. if `0 < K <= O`: extends exactly `K` distinct odd kinds by one unit
   (`u = 1` for `h = 1`, `u = 2` for `h = 3`) and no other kind;
3. if `O < K`: extends every odd kind and buys the remaining `K - O` units at
   marginal cost two from kinds with `h < 4`; the number of such two-cost
   marginals is `2 * #{h = 0} + #{h = 1} + #{h = 2} >= K - O`, so enough exist.

*Proof.*

(1) An optimal selection is a prefix-closed 7-element set of marginals. Suppose
some zero marginal is unselected. Then at least one selected marginal is
positive: selecting seven zeros would require `P >= 7`, i.e. `K = 0`. Choose a
selected positive marginal that is the last selected marginal of its kind (if a
later marginal of that kind is selected, it is also positive because marginals
are non-decreasing). Replace it with an unselected zero marginal; if the chosen
zero is the second unit of an `h = 4` kind, use its first unit when that first
unit is unselected, otherwise the second unit is already prefix-safe. The result
is still prefix-closed, has the same size, and costs strictly less, contradicting
optimality. Hence all zero marginals are selected. (Note `K > 0` is the only
case where a mask can be non-empty; the `K = 0` case is handled separately.)

(2) With all zeroes selected, the remaining `K` units must be the `K` cheapest
available marginals; if `K <= O` those are `K` of the `O` unit-cost marginals,
and any cost-two marginal would raise the cost. Cost-one marginals are exactly
the first unit of `h = 1` and the second unit of `h = 3`; each odd kind offers
exactly one, and distinct odd kinds are distinct kinds.

(3) If `K > O`, all `O` cost-one marginals must be selected; otherwise a
cost-two marginal replaces a cheaper one and the total rises. The remaining
`K - O` units come from cost-two marginals. Cost-two marginals are the second
unit of `h = 0`, the second unit of `h = 1`, and the second unit of `h = 2`;
for `h = 0` the first unit is also cost two. There are
`2 * #{h=0} + #{h=1} + #{h=2}` such units. Since
`P + O + (2*#{h=0} + #{h=1} + #{h=2}) = 2 * 18 = 36 >= 7`, and
`K = 7 - P`, the inequality `K - O <= 36 - P - O` holds. Any number of units up
to the inventory can be selected while respecting per-kind prefixes: fill kinds
one unit at a time, and for `h = 0` take the cheap first unit before the second
(both cost two). ∎

**Theorem 5.3 (mask).** With `D_7` and `K` as above,

- `D_7 = 0`: the mask is empty;
- `0 < K <= O`: the mask is exactly the odd kinds (`h` in `{1, 3}`);
- `O < K`: the mask is every non-missing kind with `h < 4` (odd kinds plus
  `h in {0, 2}`).

*Proof.*

- `D_7 = 0` means `K = 0` and every optimal target has cost zero, hence
  `T[t] <= h[t]` for all `t` by definition of cost; no kind is needed.
- If `0 < K <= O`, Lemma 5.2(2) says an optimal target extends `K` distinct odd
  kinds. Every odd kind can be among those `K` (choose any `K`-subset containing
  it, `K >= 1`), so every odd kind is needed. No other kind is needed: for
  `h = 0` the optimal `u` is `0` (`T = 0`), for `h = 2` it is `1` (`T = 2 = h`),
  and for `h = 4` the target count is at most `4 = h`; all are bounded by the
  held count.
- If `O < K`, Lemma 5.2(3) makes every odd kind needed (`h = 1` gets `T = 2`,
  `h = 3` gets `T = 4`). For `h = 0`, the optimal target uses at least one unit
  of that kind in some optimal target: `K - O >= 1` lets us allocate one
  required two-cost unit to it (Lemma 5.2(3) shows enough units exist for any
  distribution; one unit of an `h = 0` kind is available). For `h = 2`, its
  second unit costs two and can likewise be allocated one of the `K - O` units,
  so `T = 4 > 2` is needed in some optimal target. Kinds with `h = 4` can never
  be exceeded. ∎

**Implementation notes and a reachability remark.**

- `seven_pairs_units` computes `(P, O)`; `seven_pairs_cost` is shared by the
  plain and masked routes so the two cannot disagree on the distance.
- The masked route reuses the plain formula and only derives the union, so it is
  a full union rather than a "pick any optimal target" heuristic. Odd held sets
  are never used as the mask without the classification above.
- `O < K` is unreachable for a missing-suit-cleared hand with `|H| >= 13`:
  then the non-missing counts give
  `P + O = sum_t ceil(h[t]/2) >= ceil(|H|/2) >= 7`. It is reachable at size 13
  when the missing suit is uncleared: four Tiao 1, three Tiao 2, and six distinct
  Wan singles with Tiao missing give `P = 0`, `O = 6`, `K = 7`. Production
  policy gates avoid analyzing uncleared hands, but the distance API supports
  them and smaller hands. Theorem 5.3 does not assume a cleared suit or a
  particular hand size; its complete union is required in all these cases.
- The seven-pairs route and the standard route are merged exactly as in
  `deficit`: `min` for the distance, both masks ORed on an exact tie, and only
  the lower route's mask kept otherwise. This is the correct union over
  `Opt_C(H)` because `Opt_C(H)` is the union of the two families' optimal sets
  when they tie, and only the better family's set otherwise. Seven pairs is
  skipped entirely when `M` is non-empty, matching `eval::is_winning`.

## 6. `deficit_with_mask` assembly

`deficit_with_mask` computes the two routes, merges them, and caches
`(distance, mask)` in `hand_mask_cache`, keyed by the full 27-count hand.
The context (`M`, `s`, capacities) is fixed for the evaluator's lifetime, so the
key is unambiguous, and the new caches are separate from `suit_cache`/
`hand_cache`: a plain hit never answers a mask query, and the mask analysis adds
no cost to `deficit()`.

The cached value is a pure function of `(counts, capacity, target_sets,
missing)`; cache state, fill order, repeated lookups, and fresh versus reused
evaluators can only change how often the DP runs, never its result. Nothing is
cached across decisions.

## 7. `can_hu` equivalence and the removed direct-win shortcut

**Lemma 7.1.** Let `C` be a legal context (`m <= 4`, melds in order, no meld of
the missing suit, physical ownership so `H[t] + meld copies[t] <= 4`), and let
`|H| = n`. Then

```text
can_hu(H, M, s)   <=>   D_C(H) = 0.
```

*Proof.*

(`=>`) `can_hu` means `H` has no missing-suit tile (concealed or melded) and
`eval::is_winning(H, m)` holds. A standard win decomposes into `4 - m` sets and
one pair, all concealed counts within `cap` by physical ownership, so `H` itself
is a standard target and `c_H(H) = 0`. A seven-pairs win requires `m = 0`, every
count in `{0, 2, 4}` (even and at most four) and seven units, so `H` is a
seven-pairs target and again `c_H(H) = 0`.

(`<=`) If `D_C(H) = 0`, some `T in T_C` has `c_H(T) = 0`, i.e.
`T[t] <= H[t]` for every `t`. Since `sum T[t] = n = sum H[t]`, all inequalities
are equalities: `T = H`. If `T` is a standard target, `H` is standard-winning
with `m` melds; if it is a seven-pairs target, `m = 0` and `H` is a seven-pairs
hand. In either case `H` contains no missing-suit tile because every target
count there is zero, and the melds contain none by context legality, so
`can_hu` is true. ∎

**Lemma 7.2.** For `|H| = n - 1`, `D_C(H) >= 1`.

*Proof.* `T[t] <= H[t]` for all `t` would give `n = sum T[t] <= sum H[t] = n-1`,
a contradiction, so every target has a positive deficit. ∎

**Corollary 7.3 (shortcut removal).** In `ukeire`, the drawn hand `H + t` has
size `n`. The pre-change loop counted a drawable kind `t` when `can_hu(H+t)`
held, or else when `D_C(H+t) < D_C(H)`. By Lemma 7.1 and Lemma 7.2 the first
case implies `D_C(H+t) = 0 < 1 <= D_C(H)`, so it is a subset of the second case;
the shortcut was a redundant fast path under legal premises. By Theorem 2.3 the
second case is exactly "the bit is set", so replacing the whole loop by a mask
lookup preserves the counted set. The new code never constructs a drawn hand or
calls `can_hu`, `eval`, or `Hand::add` in `ukeire`; it only reads the mask and
counts unseen copies.

Note that the shortcut relied on physical `public` ownership in the caller's
`unseen` computation: `public` always includes the player's own meld copies
(`base_public`, and the claim/kong adjustments in `heuristic.rs`), so a kind at
its concealed capacity has `unseen <= 0`. The mask itself is capacity-respecting
independently of `public`: `T[t] <= cap[t]`, hence a kind held at capacity has
no bit and can never be counted.

## 8. Mapping to the implementation and behavioral equivalence

| Contract quantity | Implementation |
| --- | --- |
| `D_C(H)` plain path | `deficit`, `standard_deficit`, `suit_table`, `seven_pairs_deficit` (unchanged) |
| `(D_C(H), Mask_C(H))` | `deficit_with_mask`, `standard_deficit_with_mask`, `suit_table_with_mask`, `seven_pairs_units`, `seven_pairs_cost`, `seven_pairs_deficit_with_mask` |
| mask bit `t` | local rank `r` in `suit_table_with_mask`, shifted by `9 * suit_index` in `standard_deficit_with_mask`, global `1 << Tile::index` |
| tie handling | replace on strictly lower cost, `mask |=` on equal cost, skip `INF` |
| structural mask / `unseen` split | `deficit_with_mask` returns the mask; `ukeire` counts `max(0, 4 - H[t] - public[t])` per set bit |
| seven-pairs full union | `seven_pairs_deficit_with_mask` (Theorem 5.3) |
| route merge | `deficit_with_mask` (`min`; OR on exact tie) |
| caches | `suit_cache`/`hand_cache` untouched; `suit_mask_cache`/`hand_mask_cache` separate |

`ukeire` keeps its signature (including the caller-provided distance, which is
checked against the masked analysis with a `debug_assert`), so `heuristic.rs`
did not need to change: the same call sites, the same `Progress` ordering, the
same peng/kong gates, fan tie-break, income ordering, and fixed fallbacks. The
only production work removed is the per-kind `deficit()`/`can_hu()` loop inside
`ukeire`; the structural analysis now runs once per distinct queried hand and
the per-kind work is a 27-bit scan that never runs the distance DP.

Equivalence of the returned ukeire total follows because (a) the exact distance
is unchanged (the masked analysis asserts it, and the shared seven-pairs cost
keeps both routes aligned), (b) the counted kind set is unchanged (Theorem 2.3
and Corollary 7.3), and (c) the `unseen` formula is unchanged
(`4 - H[t] - public[t]`, skipped when non-positive). Missing-suit kinds are
skipped defensively in both paths and can never carry a bit (`cap[s] = 0`).
Over-capacity draws cannot carry a bit either, which is why the masked path
cannot produce the phantom counts the old `can_hu` branch could produce if a
caller passed `public` without its own meld copies; production callers always
include them, so behavior outside that documented input domain changed only for
inputs the policy never generates.

Determinism and lifecycle are untouched: the evaluator is created inside one
`choose` call, the mask caches live and die with it, the policy remains
stateless, and `Game::clone` continuation with a fresh policy therefore gets the
same analysis results from the same observations.

## 9. Checks aligned with this proof

- `suit_table_with_mask`: full `[sets][pair]` table compared with a brute-force
  enumeration of triplet/sequence multisets plus a pair, including capacity
  patterns that mirror melds; costs also compared with `suit_table`
  (`optimization_a2_tests` style table checks and the one-pass test module).
- `deficit_with_mask`: distance equal to `deficit` under the same context
  (production `debug_assert`, plus the frozen compatibility reference).
- Mask semantics: for every legal hand, bit `t` must equal
  `D_C(H+t) < D_C(H)`, and `Mask_C` must equal the union over the concrete
  target oracles (standard, seven pairs, and their tie).
- Seven pairs: the 241 held-count classes and the directed zero/triple/quad/
  missing-suit cases already used for `optimization_b_tests`, extended with the
  full optimal-target union and target enumeration.
- `can_hu(H) <=> D_C(H) = 0` and the removed shortcut are covered by the
  independent reference's per-tile loop and the differential game traces.
