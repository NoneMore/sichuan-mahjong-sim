//! Focused evidence for the one-pass discard surplus analysis.
//!
//! [`super::DistanceEvaluator::deficit_with_surplus`] is checked against an
//! independent ordinary-deficit path: for every held kind `t`, removing one
//! copy and calling the unchanged [`super::DistanceEvaluator::deficit`] (and,
//! for the hand-picked fixtures, the frozen `dc6ff5c` evaluator in
//! [`super::one_pass_reference`]) must return `D(H)` exactly for surplus bits
//! and `D(H) + 1` otherwise. The surplus analysis never builds its own
//! expectation and the needed mask is never complemented.
//!
//! Coverage:
//!
//! - Hand-picked fixtures with literal `(deficit, surplus, needed)` triples:
//!   complete standard hands, within-standard multi-target surplus with
//!   triplet/sequence competition, a seven-pairs-only surplus, a
//!   standard/seven-pairs tie, a peng capacity of one, four melds, an uncleared
//!   missing-suit tile, and seven-pairs hands with missing-suit tiles.
//! - Physical fixed-seed positions: [`super::test_fixtures::sampled_states`]
//!   documents seed base `0x5EED_1A55_0F0F_0001`, stride
//!   `0x9E37_79B9_7F4A_7C15`, and 75
//!   post-discard positions covering three missing suits and meld counts
//!   `0..=4` with peng/open-kong/concealed-kong/added-kong capacities. Each is
//!   extended by one deterministically chosen legal tile to the original
//!   `14 - 3m` hand, once preferring a non-missing kind and once a missing-suit
//!   kind, giving 150 original hands. The boundary fixtures contribute 30 more
//!   with the frozen per-candidate distance also checked.
//! - Cache behavior: lazy allocation, cold/warm/reordered queries, repeated
//!   queries, and per-evaluator context isolation.
//!
//! This is analysis-level differential evidence only. Heuristic `rank_discards`
//! integration, full-game trajectories, and performance measurements are
//! separate deliverables.

use super::test_fixtures::{
    assert_physical, boundary_states, meld_copies, sampled_states, tile, State,
};
use super::{one_pass_reference, DistanceEvaluator};
use crate::action::Meld;
use crate::hand::Hand;
use crate::seat::Seat;
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

fn bit(kind: Tile) -> u32 {
    1u32 << kind.index()
}

fn grouped(groups: &[(Suit, &[u8])]) -> Hand {
    let mut hand = Hand::new();
    for &(suit, ranks) in groups {
        for &rank in ranks {
            hand.add(tile(suit, rank)).expect("at most four copies");
        }
    }
    hand
}

fn candidates(hand: &Hand) -> Vec<Tile> {
    hand.distinct_tiles().collect()
}

/// One directed original-hand fixture with literal expectations.
struct Case {
    label: &'static str,
    hand: Hand,
    melds: Vec<Meld>,
    missing: Suit,
    deficit: u8,
    /// Kinds that some optimal target holds fewer of than the hand.
    surplus: u32,
    /// Kinds that some optimal target needs from a hand of the same size.
    needed: u32,
}

/// Hand-picked original hands with expectations derived from an independent
/// concrete-target enumeration over the same structural target definition.
///
/// `needed` is the [`DistanceEvaluator::deficit_with_mask`] mask for the
/// original-size hand, i.e. the union over optimal targets of kinds the target
/// holds more of. Overlaps with `surplus` are deliberate: they pin that the two
/// masks are not complements.
fn directed_cases() -> Vec<Case> {
    let wan = Suit::Wan;
    let tong = Suit::Tong;
    let tiao = Suit::Tiao;
    let w = |rank| tile(wan, rank);
    let t = |rank| tile(tong, rank);
    let b = |rank| tile(tiao, rank);
    vec![
        // Complete standard target: the zero-deficit optimal target is the
        // hand itself, so no kind can be surplus or needed and every discard
        // costs one.
        Case {
            label: "standard-complete",
            hand: grouped(&[
                (wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (tong, &[1, 1, 2, 3, 4]),
            ]),
            melds: vec![],
            missing: tiao,
            deficit: 0,
            surplus: 0,
            needed: 0,
        },
        // Distance one with several optimal standard targets. The 1T pair plus
        // 234T set is optimal (5T surplus), while 123T plus 55T is also
        // optimal and needs another 1T, so 1T is both surplus and needed.
        Case {
            label: "standard-surplus-union",
            hand: grouped(&[
                (wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (tong, &[1, 1, 2, 3, 5]),
            ]),
            melds: vec![],
            missing: tiao,
            deficit: 1,
            surplus: bit(t(1)) | bit(t(2)) | bit(t(5)),
            needed: bit(t(1)) | bit(t(4)) | bit(t(5)),
        },
        // Six pairs plus the two singles 3W and 6B: seven pairs is strictly
        // optimal and either singleton is extended, so both are surplus and
        // both are needed by the other optimal seven-pairs target.
        Case {
            label: "seven-pairs-surplus",
            hand: grouped(&[
                (wan, &[1, 1, 3, 4, 4, 7, 7]),
                (tiao, &[1, 1, 4, 4, 6, 7, 7]),
            ]),
            melds: vec![],
            missing: tong,
            deficit: 1,
            surplus: bit(w(3)) | bit(b(6)),
            needed: bit(w(3)) | bit(b(6)),
        },
        // Standard and seven pairs both reach deficit one. Every held kind is
        // surplus in some optimal target, while several kinds are also needed,
        // so a route-tie union must keep both masks.
        Case {
            label: "cross-route-tie",
            hand: grouped(&[(wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8])]),
            melds: vec![],
            missing: tiao,
            deficit: 1,
            surplus: (1u32 << 8) - 1,
            needed: bit(w(1)) | bit(w(2)) | bit(w(4)) | bit(w(5)) | bit(w(7)) | bit(w(8)),
        },
        // A peng leaves one concealed 5W target copy. One optimal target uses
        // that copy in 567W and leaves 8W surplus; the alternative uses 678W
        // and leaves 5W surplus.
        Case {
            label: "peng-capacity",
            hand: grouped(&[(wan, &[1, 2, 3, 5, 6, 7, 8]), (tong, &[1, 1, 9, 9])]),
            melds: vec![Meld::peng(w(5), Seat::DEALER)],
            missing: tiao,
            deficit: 1,
            surplus: bit(w(5)) | bit(w(8)),
            needed: bit(t(1)) | bit(t(9)),
        },
        // Four melds leave a two-tile pair-only target. The own peng caps 5W at
        // one copy, so the 5W/6W hand can only wait on 6W and 5W is surplus.
        Case {
            label: "four-meld-pair",
            hand: grouped(&[(wan, &[5, 6])]),
            melds: vec![
                Meld::peng(w(5), Seat::DEALER),
                Meld::peng(t(1), Seat::DEALER),
                Meld::open_kong(t(2), Seat::DEALER),
                Meld::concealed_kong(t(3)),
            ],
            missing: tiao,
            deficit: 1,
            surplus: bit(w(5)),
            needed: bit(w(6)),
        },
        // Six pairs plus the two singles 3T and 6B with Wan missing: the same
        // seven-pairs-only structure under the third missing suit.
        Case {
            label: "seven-pairs-surplus-missing-wan",
            hand: grouped(&[
                (tong, &[1, 1, 3, 4, 4, 7, 7]),
                (tiao, &[1, 1, 4, 4, 6, 7, 7]),
            ]),
            melds: vec![],
            missing: wan,
            deficit: 1,
            surplus: bit(t(3)) | bit(b(6)),
            needed: bit(t(3)) | bit(b(6)),
        },
        // Triplet and sequence competition inside the standard route: optimal
        // targets use the 333/666/999 triplets or a 345/789 sequence mix, and
        // they disagree on whether 8W or 9W is the unused copy. Seven pairs is
        // strictly worse, so only the standard surplus applies.
        Case {
            label: "standard-triplet-sequence",
            hand: grouped(&[(wan, &[1, 1, 3, 3, 3, 3, 4, 5, 6, 6, 6, 8, 9, 9])]),
            melds: vec![],
            missing: tiao,
            deficit: 1,
            surplus: bit(w(8)) | bit(w(9)),
            needed: bit(w(1)) | bit(w(7)) | bit(w(9)),
        },
        // Uncleared missing suit: the held 5B can never enter a target, so it
        // is surplus even though it is not a needed kind of any target.
        Case {
            label: "missing-held-surplus",
            hand: grouped(&[
                (wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (tong, &[1, 1, 2, 3]),
                (tiao, &[5]),
            ]),
            melds: vec![],
            missing: tiao,
            deficit: 1,
            surplus: bit(b(5)),
            needed: bit(t(1)) | bit(t(4)),
        },
        // Six pairs plus the two missing-suit singles 2T and 5T: seven pairs is
        // strictly optimal at distance two and the missing-held copies are the
        // only surplus kinds. A seven-pairs DP that skipped missing kinds would
        // lose both bits here.
        Case {
            label: "seven-pairs-missing-held",
            hand: grouped(&[
                (wan, &[1, 1, 4, 4, 7, 7]),
                (tiao, &[1, 1, 4, 4, 7, 7]),
                (tong, &[2, 5]),
            ]),
            melds: vec![],
            missing: tong,
            deficit: 2,
            surplus: bit(t(2)) | bit(t(5)),
            needed: non_missing_mask(tong),
        },
    ]
}

fn non_missing_mask(missing: Suit) -> u32 {
    let mut mask = 0u32;
    for kind in Tile::ALL {
        if kind.suit() != missing {
            mask |= bit(kind);
        }
    }
    mask
}

fn case(label: &str) -> Case {
    directed_cases()
        .into_iter()
        .find(|case| case.label == label)
        .unwrap_or_else(|| panic!("missing directed case {label}"))
}

/// Checks one original hand against the unchanged ordinary deficit.
///
/// Returns the tested `(deficit, surplus mask)`.
fn check_original(
    label: &str,
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    expected: Option<(u8, u32)>,
    frozen_candidates: bool,
) -> (u8, u32) {
    assert_eq!(
        hand.total(),
        14 - 3 * melds.len(),
        "{label}: an original hand has 14 - 3 * melds tiles"
    );
    let copies = meld_copies(melds);
    for meld in melds {
        assert_ne!(
            meld.tile().suit(),
            missing,
            "{label}: own melds stay outside the missing suit"
        );
    }
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        assert!(
            hand.count(kind) + copies[index] <= COPIES_PER_TILE,
            "{label}: {kind} is held {} times plus {} meld copies",
            hand.count(kind),
            copies[index]
        );
    }

    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("at most four melds");
    let actual = evaluator.deficit_with_surplus(hand);
    if let Some(expected) = expected {
        assert_eq!(actual, expected, "{label}: literal surplus expectation");
    }
    assert_eq!(
        actual.1 >> TILE_KINDS,
        0,
        "{label}: the surplus mask must use only the low {TILE_KINDS} bits, got {:#x}",
        actual.1
    );

    // The ordinary original-size distance, from a separate evaluator and from
    // the frozen pre-change analysis.
    let mut plain = DistanceEvaluator::new(melds, missing).expect("at most four melds");
    assert_eq!(
        plain.deficit(hand),
        actual.0,
        "{label}: surplus and plain original-hand distances differ"
    );
    let mut frozen =
        one_pass_reference::DistanceEvaluator::new(melds, missing).expect("at most four melds");
    assert_eq!(
        frozen.deficit(hand),
        actual.0,
        "{label}: frozen dc6ff5c original-hand distance differs"
    );
    assert_eq!(
        crate::rules::can_hu(hand, melds, missing),
        actual.0 == 0,
        "{label}: can_hu and zero original-hand deficit must agree at the legal size"
    );
    // The per-candidate reference: remove one copy, run the ordinary deficit.
    for kind in Tile::ALL {
        let held = hand.count(kind);
        let marked = actual.1 & bit(kind) != 0;
        if held == 0 {
            assert!(
                !marked,
                "{label}: {kind} is not held, so it cannot be surplus"
            );
            continue;
        }
        let mut after = *hand;
        after.remove(kind).expect("held copy");
        assert_eq!(
            after.total(),
            13 - 3 * melds.len(),
            "{label}: a candidate discard must have the post-discard size"
        );
        let ordinary = plain.deficit(&after);
        assert_eq!(
            marked,
            ordinary == actual.0,
            "{label}: bit {kind} must mean exactly D(H-t) = D(H) \
             (D={}, D(H-t)={ordinary})",
            actual.0
        );
        assert_eq!(
            ordinary,
            actual.0 + u8::from(!marked),
            "{label}: candidate {kind} must be D(H) or D(H) + 1"
        );
        if frozen_candidates {
            assert_eq!(
                frozen.deficit(&after),
                ordinary,
                "{label}: frozen dc6ff5c candidate {kind} distance differs"
            );
        }
    }

    // A warm repeated query is stable, and ordinary queries after the surplus
    // analysis still answer exactly.
    assert_eq!(
        evaluator.deficit_with_surplus(hand),
        actual,
        "{label}: repeated surplus query changed"
    );
    let mut removed = *hand;
    if let Some(kind) = candidates(hand).first().copied() {
        removed.remove(kind).expect("held");
        assert_eq!(
            evaluator.deficit(&removed),
            plain.deficit(&removed),
            "{label}: ordinary distance changed after a surplus query"
        );
        assert_eq!(
            evaluator.deficit_with_mask(&removed).0,
            plain.deficit(&removed),
            "{label}: needed-mask distance changed after a surplus query"
        );
    }
    actual
}

/// Extends a physical post-discard fixture by one legal tile.
///
/// `prefer_missing` chooses the first legal missing-suit kind instead of the
/// first legal non-missing kind, so uncleared missing-suit copies are covered.
/// The first legal kind in encoding order is used, which keeps the derived
/// original hand deterministic and physically consistent.
fn add_one_legal(state: &State, prefer_missing: bool) -> Option<Hand> {
    assert_physical(state);
    let copies = meld_copies(&state.melds);
    for kind in Tile::ALL {
        if (kind.suit() == state.missing) != prefer_missing {
            continue;
        }
        let index = kind.index() as usize;
        if state.hand.count(kind) + copies[index] >= COPIES_PER_TILE
            || state.hand.count(kind) + state.public[index] >= COPIES_PER_TILE
        {
            continue;
        }
        let mut hand = state.hand;
        hand.add(kind).expect("legal physical copy");
        for held in Tile::ALL {
            assert!(
                hand.count(held) + state.public[held.index() as usize] <= COPIES_PER_TILE,
                "{}: extended hand and public ownership exceed four copies of {held}",
                state.label
            );
        }
        return Some(hand);
    }
    None
}

#[test]
fn directed_hands_match_literals_ordinary_distance_and_needed_mask() {
    let cases = directed_cases();
    assert_eq!(cases.len(), 10, "directed fixture list changed");
    let mut meld_counts = [0usize; 5];
    let mut missing_counts = [0usize; 3];
    for case in &cases {
        let actual = check_original(
            case.label,
            &case.hand,
            &case.melds,
            case.missing,
            Some((case.deficit, case.surplus)),
            true,
        );
        assert_eq!(
            actual,
            (case.deficit, case.surplus),
            "{}: surplus literal",
            case.label
        );

        // The needed mask over the same original-size optimal targets must be
        // the independently expected one; this is the mask that must not be
        // complemented to obtain surplus.
        let mut masked = DistanceEvaluator::new(&case.melds, case.missing).expect("context");
        assert_eq!(
            masked.deficit_with_mask(&case.hand),
            (case.deficit, case.needed),
            "{}: needed mask over original-size optimal targets",
            case.label
        );

        meld_counts[case.melds.len()] += 1;
        missing_counts[case.missing.index()] += 1;
    }
    assert!(meld_counts[0] > 0, "need a no-meld fixture");
    assert!(meld_counts[1] > 0, "need a one-meld fixture");
    assert!(meld_counts[4] > 0, "need a four-meld fixture");
    assert!(
        missing_counts.iter().all(|&count| count > 0),
        "need all three missing suits"
    );
}

#[test]
fn surplus_is_not_the_needed_mask_complement() {
    for label in [
        "standard-surplus-union",
        "cross-route-tie",
        "seven-pairs-surplus",
    ] {
        let case = case(label);
        let mut surplus_evaluator =
            DistanceEvaluator::new(&case.melds, case.missing).expect("context");
        let (deficit, surplus) = surplus_evaluator.deficit_with_surplus(&case.hand);
        let mut needed_evaluator =
            DistanceEvaluator::new(&case.melds, case.missing).expect("context");
        let (needed_deficit, needed) = needed_evaluator.deficit_with_mask(&case.hand);
        assert_eq!(deficit, needed_deficit, "{label}: distances must agree");
        assert_ne!(
            surplus & needed,
            0,
            "{label}: at least one kind is both surplus and needed"
        );
        assert_ne!(
            surplus,
            !needed & ((1u32 << TILE_KINDS) - 1),
            "{label}: surplus must not be the complemented needed mask"
        );
        let mut held_mask = 0u32;
        for (index, &held) in case.hand.counts().iter().enumerate() {
            if held > 0 {
                held_mask |= 1u32 << index;
            }
        }
        assert_eq!(
            surplus & !held_mask,
            0,
            "{label}: surplus bits require held copies"
        );
    }
}

#[test]
fn route_ties_union_and_strictly_cheaper_routes_replace() {
    let tie = case("cross-route-tie");
    let mut evaluator = DistanceEvaluator::new(&[], tie.missing).expect("no melds");
    assert_eq!(
        route_deficits(&mut evaluator, &tie.hand),
        (tie.deficit, tie.deficit),
        "the tie fixture must tie the standard and seven-pairs routes"
    );
    assert_eq!(
        evaluator.deficit_with_surplus(&tie.hand),
        (tie.deficit, tie.surplus)
    );

    let seven_only = case("seven-pairs-surplus");
    let mut evaluator = DistanceEvaluator::new(&[], seven_only.missing).expect("no melds");
    let (standard, seven_pairs) = route_deficits(&mut evaluator, &seven_only.hand);
    assert!(
        seven_pairs < standard,
        "the seven-pairs-only fixture must separate the routes"
    );
    assert_eq!(
        evaluator.deficit_with_surplus(&seven_only.hand),
        (seven_pairs, seven_only.surplus)
    );
    let mut odd_kinds = 0u32;
    for kind in candidates(&seven_only.hand) {
        if seven_only.hand.count(kind) % 2 == 1 {
            odd_kinds |= bit(kind);
        }
    }
    assert_eq!(
        seven_only.surplus, odd_kinds,
        "every held odd kind is surplus in some optimal seven-pairs target"
    );

    let missing_held = case("seven-pairs-missing-held");
    let mut evaluator = DistanceEvaluator::new(&[], missing_held.missing).expect("no melds");
    assert_eq!(
        route_deficits(&mut evaluator, &missing_held.hand),
        (4, 2),
        "seven pairs must be strictly better than the standard route"
    );
    assert_eq!(
        evaluator.deficit_with_surplus(&missing_held.hand),
        (2, missing_held.surplus),
        "the missing-held copies must survive as seven-pairs surplus"
    );
}

fn route_deficits(evaluator: &mut DistanceEvaluator<'_>, hand: &Hand) -> (u8, u8) {
    let counts = *hand.counts();
    (
        evaluator.standard_deficit(&counts),
        evaluator.seven_pairs_deficit(&counts),
    )
}

#[test]
fn surplus_cache_is_lazy_and_keeps_ordinary_paths_unchanged() {
    let case = case("standard-surplus-union");
    let mut after = case.hand;
    after.remove(tile(Suit::Tong, 5)).expect("held");

    let mut evaluator = DistanceEvaluator::new(&case.melds, case.missing).expect("context");
    assert!(
        evaluator.surplus_cache.is_none(),
        "a fresh evaluator must not allocate surplus caches"
    );
    let plain = evaluator.deficit(&case.hand);
    assert_eq!(plain, case.deficit);
    assert!(
        evaluator.surplus_cache.is_none(),
        "ordinary deficit must not allocate surplus caches"
    );
    let (masked, _) = evaluator.deficit_with_mask(&case.hand);
    assert_eq!(masked, case.deficit);
    assert!(
        evaluator.surplus_cache.is_none(),
        "needed-mask queries must not allocate surplus caches"
    );

    let cold = evaluator.deficit_with_surplus(&case.hand);
    assert_eq!(cold, (case.deficit, case.surplus));
    let cache = evaluator
        .surplus_cache
        .as_ref()
        .expect("the first surplus query allocates the caches");
    assert!(
        !cache.suit.is_empty(),
        "the surplus suit tables must be cached"
    );
    assert_eq!(
        cache.hand.get(case.hand.counts()),
        Some(&cold),
        "the surplus hand result must be cached"
    );
    assert_eq!(
        evaluator.deficit_with_surplus(&case.hand),
        cold,
        "a warm surplus query must hit the cache and stay exact"
    );
    assert_eq!(
        evaluator.deficit(&case.hand),
        plain,
        "ordinary deficit must stay exact after a surplus query"
    );
    assert_eq!(
        evaluator.deficit(&after),
        1,
        "the post-discard hand one tile below tenpai keeps its own distance"
    );
    assert_eq!(
        evaluator.deficit_with_mask(&after).0,
        1,
        "needed-mask distance must stay exact after a surplus query"
    );
}

#[test]
fn surplus_queries_are_order_and_context_independent() {
    let same_context: Vec<Case> = directed_cases()
        .into_iter()
        .filter(|case| case.melds.is_empty() && case.missing == Suit::Tiao)
        .collect();
    assert!(
        same_context.len() >= 4,
        "need several original hands in one context"
    );

    // Fresh evaluator per hand.
    for case in &same_context {
        let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
        assert_eq!(
            evaluator.deficit_with_surplus(&case.hand),
            (case.deficit, case.surplus),
            "{}: fresh evaluator",
            case.label
        );
    }

    // One reused evaluator: forward, reversed, then forward again.
    let mut reused = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    for case in &same_context {
        assert_eq!(
            reused.deficit_with_surplus(&case.hand),
            (case.deficit, case.surplus),
            "{}: forward order",
            case.label
        );
    }
    for case in same_context.iter().rev() {
        assert_eq!(
            reused.deficit_with_surplus(&case.hand),
            (case.deficit, case.surplus),
            "{}: reversed order",
            case.label
        );
    }
    for case in &same_context {
        assert_eq!(
            reused.deficit_with_surplus(&case.hand),
            (case.deficit, case.surplus),
            "{}: repeated after reorder",
            case.label
        );
    }

    // A separate context (different missing suit) must be exact for the same
    // physical hand, and must not disturb the reused evaluator.
    let shared_hand = case("cross-route-tie").hand;
    for missing in Suit::ALL {
        check_original(
            &format!("cross-route-tie-missing-{}", missing.name()),
            &shared_hand,
            &[],
            missing,
            None,
            false,
        );
    }
    for case in &same_context {
        assert_eq!(
            reused.deficit_with_surplus(&case.hand),
            (case.deficit, case.surplus),
            "{}: another context must not leak into the reused evaluator",
            case.label
        );
    }
}

#[test]
fn candidate_subsets_and_complete_ties_follow_the_identity() {
    let tie = case("cross-route-tie");
    let mut evaluator = DistanceEvaluator::new(&[], tie.missing).expect("no melds");
    let (deficit, surplus) = evaluator.deficit_with_surplus(&tie.hand);
    assert_eq!(deficit, 1);
    assert_eq!(
        surplus,
        (1u32 << 8) - 1,
        "every held kind must be surplus in this complete tie"
    );
    assert_eq!(deficit, tie.deficit);

    let held = candidates(&tie.hand);
    assert_eq!(held.len(), 8);
    let mut plain = DistanceEvaluator::new(&[], tie.missing).expect("no melds");
    for kind in held.iter().rev() {
        let mut after = tie.hand;
        after.remove(*kind).expect("held");
        assert_eq!(
            plain.deficit(&after),
            deficit,
            "complete tie: removing {kind} must keep the deficit"
        );
        assert_eq!(
            evaluator.deficit(&after),
            deficit,
            "complete tie after a surplus query: removing {kind}"
        );
    }

    // A subset in different orders returns the same per-candidate values.
    let subset = &held[..3];
    let mut forward = Vec::new();
    for kind in subset {
        let mut after = tie.hand;
        after.remove(*kind).expect("held");
        forward.push(evaluator.deficit(&after));
    }
    let mut backward = Vec::new();
    for kind in subset.iter().rev() {
        let mut after = tie.hand;
        after.remove(*kind).expect("held");
        backward.push(evaluator.deficit(&after));
    }
    backward.reverse();
    assert_eq!(forward, backward);
    assert!(forward.iter().all(|&value| value == deficit));
}

#[test]
fn physical_sampled_original_hands_match_per_candidate_deficit() {
    let samples = sampled_states();
    assert_eq!(samples.len(), 75, "sample sweep size changed");
    let mut original_count = 0usize;
    let mut meld_counts = [0usize; 5];
    let mut missing_counts = [0usize; 3];
    let mut nonzero_surplus = 0usize;
    let mut missing_surplus = 0usize;

    for (index, state) in samples.iter().enumerate() {
        assert_physical(state);
        for (variant, prefer_missing) in [("cleared", false), ("missing-held", true)] {
            let Some(hand) = add_one_legal(state, prefer_missing) else {
                continue;
            };
            let label = format!("sampled-{index}-{variant}");
            let (_, surplus) =
                check_original(&label, &hand, &state.melds, state.missing, None, false);
            original_count += 1;
            meld_counts[state.melds.len()] += 1;
            missing_counts[state.missing.index()] += 1;
            if surplus != 0 {
                nonzero_surplus += 1;
            }
            if surplus & non_missing_mask(state.missing) != surplus {
                missing_surplus += 1;
            }
        }
    }

    assert_eq!(original_count, 150, "two variants per sampled position");
    for (meld_count, count) in meld_counts.iter().enumerate() {
        assert_eq!(
            *count, 30,
            "meld count {meld_count} needs two variants of fifteen samples"
        );
    }
    for (missing, count) in missing_counts.iter().enumerate() {
        assert_eq!(*count, 50, "missing suit {missing} needs fifty samples");
    }
    assert!(nonzero_surplus > 0, "the sweep must include surplus kinds");
    assert!(
        missing_surplus > 0,
        "the sweep must include held missing-suit surplus"
    );
}

#[test]
fn boundary_original_hands_match_frozen_per_candidate_distance() {
    let fixtures = boundary_states();
    assert_eq!(fixtures.len(), 15, "fixture list size changed");
    let mut original_count = 0usize;
    for state in &fixtures {
        for (variant, prefer_missing) in [("cleared", false), ("missing-held", true)] {
            let Some(hand) = add_one_legal(state, prefer_missing) else {
                continue;
            };
            let label = format!("{}-{variant}", state.label);
            check_original(&label, &hand, &state.melds, state.missing, None, true);
            original_count += 1;
        }
    }
    assert_eq!(
        original_count, 30,
        "both one-tile extensions of all fifteen boundary fixtures"
    );
}

#[test]
fn surplus_identity_holds_for_counted_sizes_beyond_the_original_hand() {
    // Eight complete pairs over Wan: both routes reach deficit zero, and the
    // seven-pairs route may drop any one pair, so all eight held kinds are
    // surplus. This also exercises the general seven-pairs unit DP beyond the
    // original-hand size used by the policy.
    let hand = grouped(&[(Suit::Wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])]);
    let mut evaluator = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    let (deficit, surplus) = evaluator.deficit_with_surplus(&hand);
    assert_eq!(deficit, 0);
    assert_eq!(surplus, (1u32 << 8) - 1);

    let mut plain = DistanceEvaluator::new(&[], Suit::Tiao).expect("no melds");
    for kind in candidates(&hand) {
        let mut after = hand;
        after.remove(kind).expect("held");
        assert_eq!(
            plain.deficit(&after),
            deficit,
            "removing {kind} from eight pairs must keep the zero deficit"
        );
    }
}
