//! Candidate-deficit evidence: per-kind candidate distances and the surplus
//! identity behind them.
//!
//! Every expectation is compared with the frozen removed-tile reference in
//! `reference.rs`; ukeire, fan, and the shared helpers are reused to check the
//! wiring, not to re-prove their own semantics.

use super::super::*;

use super::{
    all_candidates, assert_physical, assert_rank_differential, plain_deficit, single_deficit,
};

#[test]
fn candidate_deficits_come_from_the_original_hand_surplus() {
    let missing = Suit::Tiao;
    let hand = tenpai_plus_extra();
    let public = [0u8; TILE_KINDS];

    let one_wan = tile(Suit::Wan, 1);
    let two_wan = tile(Suit::Wan, 2);
    let nine_wan = tile(Suit::Wan, 9);
    let one_tong = tile(Suit::Tong, 1);

    assert_eq!(plain_deficit(&hand, &[], missing), 1);
    // The held Tong singleton and the 9W copy omitted by an optimal target are
    // surplus, so removing either keeps distance one. 1W and 2W are needed by
    // every optimal target, so removing one adds exactly one draw.
    assert_eq!(single_deficit(&hand, &[], missing, &public, one_tong), 1);
    assert_eq!(single_deficit(&hand, &[], missing, &public, nine_wan), 1);
    assert_eq!(single_deficit(&hand, &[], missing, &public, one_wan), 2);
    assert_eq!(single_deficit(&hand, &[], missing, &public, two_wan), 2);

    assert_rank_differential(
        "standard original-hand surplus",
        &hand,
        &[],
        missing,
        &public,
        &all_candidates(&hand),
        true,
    );
}

#[test]
fn tied_standard_targets_union_their_removable_kinds() {
    let missing = Suit::Tiao;
    // 111W 22W 33W 4W 5W 6W 7W 8W 99W: several optimal four-set-and-pair
    // targets differ in which held copy they omit.
    let hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 7),
        tile(Suit::Wan, 8),
        tile(Suit::Wan, 9),
        tile(Suit::Wan, 9),
    ]);
    let public = [0u8; TILE_KINDS];

    assert_eq!(plain_deficit(&hand, &[], missing), 1);
    for rank in [1, 2, 3, 5, 8, 9] {
        assert_eq!(
            single_deficit(&hand, &[], missing, &public, tile(Suit::Wan, rank)),
            1,
            "{rank}W is surplus in some tied optimal target"
        );
    }
    for rank in [4, 6, 7] {
        assert_eq!(
            single_deficit(&hand, &[], missing, &public, tile(Suit::Wan, rank)),
            2,
            "{rank}W is needed by every tied optimal target"
        );
    }

    assert_rank_differential(
        "tied standard target surplus union",
        &hand,
        &[],
        missing,
        &public,
        &all_candidates(&hand),
        true,
    );
}

#[test]
fn seven_pairs_and_standard_route_ties_union_their_surplus() {
    let missing = Suit::Tiao;
    let public = [0u8; TILE_KINDS];

    // Two different optimal seven-pairs targets: pair 11 plus 22..66 plus a
    // drawn 7W omits one 1W, while quad 1W plus 22..66 omits 7W.
    let seven_pairs = seven_pairs_plus_extra();
    assert_eq!(plain_deficit(&seven_pairs, &[], missing), 1);
    assert_eq!(
        single_deficit(&seven_pairs, &[], missing, &public, tile(Suit::Wan, 1)),
        1
    );
    assert_eq!(
        single_deficit(&seven_pairs, &[], missing, &public, tile(Suit::Wan, 7)),
        1
    );
    assert_eq!(
        single_deficit(&seven_pairs, &[], missing, &public, tile(Suit::Tong, 2)),
        2,
        "a held pair is not surplus in a distance-one seven-pairs target"
    );
    assert_rank_differential(
        "seven-pairs optimal-target union",
        &seven_pairs,
        &[],
        missing,
        &public,
        &all_candidates(&seven_pairs),
        true,
    );

    // A held quad counts as two pair units: 1111W 22W 33W 44W 55W 6W 7W is
    // one draw from seven pairs by extending either singleton.
    let quad = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 7),
    ]);
    assert_eq!(plain_deficit(&quad, &[], missing), 1);
    assert_eq!(
        single_deficit(&quad, &[], missing, &public, tile(Suit::Wan, 6)),
        1,
        "extending 6W leaves the quad's two pair units in the target"
    );
    assert_eq!(
        single_deficit(&quad, &[], missing, &public, tile(Suit::Wan, 7)),
        1
    );
    assert_eq!(
        single_deficit(&quad, &[], missing, &public, tile(Suit::Wan, 1)),
        1,
        "a tied standard target may omit one 1W copy, so the union keeps it"
    );
    assert_rank_differential(
        "quad counts as two seven-pairs units",
        &quad,
        &[],
        missing,
        &public,
        &all_candidates(&quad),
        true,
    );

    // 11W 22W 33W 44W 55W 6W 77W 9W ties a standard target (123,123,456,456,
    // 77) with a seven-pairs target at distance one; their removable kinds
    // differ and the union keeps all of them.
    let mut route_tie = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7], &[], &[]);
    route_tie
        .add(tile(Suit::Wan, 9))
        .expect("four copies at most");
    assert_eq!(plain_deficit(&route_tie, &[], missing), 1);
    for rank in [2, 5, 6, 9] {
        assert_eq!(
            single_deficit(&route_tie, &[], missing, &public, tile(Suit::Wan, rank)),
            1,
            "{rank}W is surplus in at least one tied route"
        );
    }
    assert_eq!(
        single_deficit(&route_tie, &[], missing, &public, tile(Suit::Wan, 1)),
        2,
        "1W is needed by every tied optimal target"
    );
    assert_rank_differential(
        "cross-route surplus union",
        &route_tie,
        &[],
        missing,
        &public,
        &all_candidates(&route_tie),
        true,
    );
}

#[test]
fn meld_capacity_and_missing_contexts_keep_their_surplus() {
    let source = Seat::new(1).expect("seat");
    let missing = Suit::Tiao;

    let one_peng = [Meld::peng(tile(Suit::Tong, 6), source)];
    let open_hand = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let open_public = public_of(&[], &one_peng);
    assert_physical(&open_hand, &one_peng, missing, &open_public);
    assert_eq!(plain_deficit(&open_hand, &one_peng, missing), 1);
    for rank in [1, 2, 5, 8, 9] {
        assert_eq!(
            single_deficit(
                &open_hand,
                &one_peng,
                missing,
                &open_public,
                tile(Suit::Wan, rank)
            ),
            1,
            "{rank}W is surplus with one peng"
        );
    }
    for rank in [3, 4, 6, 7] {
        assert_eq!(
            single_deficit(
                &open_hand,
                &one_peng,
                missing,
                &open_public,
                tile(Suit::Wan, rank)
            ),
            2,
            "{rank}W is needed with one peng"
        );
    }
    assert_rank_differential(
        "one peng",
        &open_hand,
        &one_peng,
        missing,
        &open_public,
        &all_candidates(&open_hand),
        true,
    );

    // An own kong caps 3W at zero concealed copies; only 2W is surplus.
    let kong_capacity = [Meld::concealed_kong(tile(Suit::Wan, 3))];
    let kong_hand = suit_hand(&[1, 1, 2, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let kong_public = public_of(&[], &kong_capacity);
    assert_physical(&kong_hand, &kong_capacity, missing, &kong_public);
    assert_eq!(plain_deficit(&kong_hand, &kong_capacity, missing), 1);
    assert_eq!(
        single_deficit(
            &kong_hand,
            &kong_capacity,
            missing,
            &kong_public,
            tile(Suit::Wan, 2)
        ),
        1
    );
    assert_eq!(
        single_deficit(
            &kong_hand,
            &kong_capacity,
            missing,
            &kong_public,
            tile(Suit::Wan, 1)
        ),
        2
    );
    assert_rank_differential(
        "own kong capacity",
        &kong_hand,
        &kong_capacity,
        missing,
        &kong_public,
        &all_candidates(&kong_hand),
        true,
    );

    // An existing peng is public ownership; its fourth copy is surplus.
    let added_kong = [Meld::peng(tile(Suit::Wan, 2), source)];
    let added_hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let added_public = public_of(&[], &added_kong);
    assert_physical(&added_hand, &added_kong, missing, &added_public);
    assert_eq!(plain_deficit(&added_hand, &added_kong, missing), 1);
    for candidate in [
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 8),
        tile(Suit::Wan, 9),
    ] {
        assert_eq!(
            single_deficit(&added_hand, &added_kong, missing, &added_public, candidate),
            1
        );
    }
    assert_eq!(
        single_deficit(
            &added_hand,
            &added_kong,
            missing,
            &added_public,
            tile(Suit::Wan, 3)
        ),
        2
    );
    assert_rank_differential(
        "peng before added kong",
        &added_hand,
        &added_kong,
        missing,
        &added_public,
        &all_candidates(&added_hand),
        true,
    );

    // Two pengs leave a target-size eight-tile hand.
    let two_peng = [
        Meld::peng(tile(Suit::Tong, 6), source),
        Meld::peng(tile(Suit::Tong, 7), source),
    ];
    let two_peng_hand = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 9], &[], &[]);
    let two_peng_public = public_of(&[], &two_peng);
    assert_physical(&two_peng_hand, &two_peng, missing, &two_peng_public);
    assert_rank_differential(
        "two pengs",
        &two_peng_hand,
        &two_peng,
        missing,
        &two_peng_public,
        &all_candidates(&two_peng_hand),
        true,
    );

    // Four melds leave only the pair; the target itself has no surplus.
    let four_melds = [
        Meld::peng(tile(Suit::Tong, 5), source),
        Meld::peng(tile(Suit::Tong, 6), source),
        Meld::concealed_kong(tile(Suit::Tong, 1)),
        Meld::open_kong(tile(Suit::Tong, 9), source),
    ];
    let pair = suit_hand(&[3, 3], &[], &[]);
    let pair_public = public_of(&[], &four_melds);
    assert_physical(&pair, &four_melds, missing, &pair_public);
    assert_eq!(plain_deficit(&pair, &four_melds, missing), 0);
    assert_rank_differential(
        "four melds leave the pair",
        &pair,
        &four_melds,
        missing,
        &pair_public,
        &all_candidates(&pair),
        true,
    );

    // A held missing-suit tile is free surplus even though no target needs it.
    let uncleared = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], &[], &[1, 9]);
    let uncleared_public = [0u8; TILE_KINDS];
    assert_physical(&uncleared, &[], missing, &uncleared_public);
    assert_eq!(plain_deficit(&uncleared, &[], missing), 2);
    assert_eq!(
        single_deficit(
            &uncleared,
            &[],
            missing,
            &uncleared_public,
            tile(Suit::Tiao, 1)
        ),
        2
    );
    assert_eq!(
        single_deficit(
            &uncleared,
            &[],
            missing,
            &uncleared_public,
            tile(Suit::Wan, 1)
        ),
        3
    );
    assert_rank_differential(
        "uncleared missing tile",
        &uncleared,
        &[],
        missing,
        &uncleared_public,
        &all_candidates(&uncleared),
        true,
    );
}

#[test]
fn nonproduction_hand_sizes_follow_the_removed_tile_identity() {
    // Outside ukeire's post-discard-size contract, these comparisons only
    // characterize unchanged defensive behavior; they do not assert that the
    // returned availability is a semantic count of improving draws.
    let missing = Suit::Tiao;
    let public = [0u8; TILE_KINDS];

    // Below the target size: the identity still holds, so every candidate adds
    // exactly one draw when the original hand has no surplus.
    let short = suit_hand(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9], &[1, 2], &[]);
    assert_eq!(plain_deficit(&short, &[], missing), 2);
    assert_eq!(
        single_deficit(&short, &[], missing, &public, tile(Suit::Wan, 5)),
        3
    );
    assert_eq!(
        single_deficit(&short, &[], missing, &public, tile(Suit::Tong, 1)),
        3
    );
    assert_rank_differential(
        "12-tile defensive input",
        &short,
        &[],
        missing,
        &public,
        &all_candidates(&short),
        true,
    );

    let shorter = suit_hand(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 2], &[]);
    assert_eq!(plain_deficit(&shorter, &[], missing), 4);
    assert_eq!(
        single_deficit(&shorter, &[], missing, &public, tile(Suit::Wan, 5)),
        5
    );
    assert_rank_differential(
        "10-tile defensive input",
        &shorter,
        &[],
        missing,
        &public,
        &all_candidates(&shorter),
        true,
    );

    // Above the target size: zero deficit with a non-empty surplus, so
    // removing a surplus copy keeps distance zero.
    let mut oversized = suit_hand(&[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5], &[], &[]);
    oversized
        .add(tile(Suit::Wan, 6))
        .expect("four copies at most");
    assert_eq!(plain_deficit(&oversized, &[], missing), 0);
    for rank in [2, 3, 4, 5, 6] {
        assert_eq!(
            single_deficit(&oversized, &[], missing, &public, tile(Suit::Wan, rank)),
            0,
            "{rank}W can be omitted from an optimal target"
        );
    }
    assert_eq!(
        single_deficit(&oversized, &[], missing, &public, tile(Suit::Wan, 1)),
        1,
        "1W is needed by every optimal target"
    );
    assert_rank_differential(
        "15-tile defensive input",
        &oversized,
        &[],
        missing,
        &public,
        &all_candidates(&oversized),
        true,
    );
}
