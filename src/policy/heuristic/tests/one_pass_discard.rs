//! Differential checks for the one-pass candidate-deficit ranking.
//!
//! The reference is the pre-change path: remove one candidate tile and ask for
//! the ordinary exact deficit of the remaining hand. It never asks for the new
//! surplus analysis. Candidate collection, the minimum-deficit filter, the
//! minimum-only ukeire stage, the fan tie-break, and the peng/kong gates stay
//! shared with production; only the candidate distance comes from a frozen
//! test-only ranking built on ordinary deficits.
//!
//! Every helper is test-only. Production deliberately has no instrumentation
//! for the "once per original hand" analysis count, so the tests compare
//! observable deficits, ukeire, selected tiles, and progress instead.
//! Only candidate deficits have an independent ordinary-distance reference;
//! ukeire, fan, and unchanged call helpers are intentionally reused to check
//! ranking/gate wiring, not to re-prove those helpers' semantics.

use super::*;

/// One scored candidate from the frozen removed-tile reference path.
struct ReferenceCandidate {
    tile: Tile,
    hand: Hand,
    deficit: u8,
    ukeire: u32,
    fan: u8,
}

/// Scores every removable candidate by removing it and using the ordinary
/// exact deficit, exactly as the pre-change ranking did.
///
/// Ukeire is computed for every retained candidate so single-candidate
/// differentials can compare it. The reference ranking below still consults
/// ukeire only for minimum-deficit candidates, so the selected tile and
/// progress match the old path; the extra values are test-only.
fn reference_candidates(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public_before_discard: &[u8; TILE_KINDS],
    candidates: &[Tile],
) -> Option<Vec<ReferenceCandidate>> {
    let mut evaluator = DistanceEvaluator::new(melds, missing)?;
    let mut scored = Vec::with_capacity(candidates.len());
    for &tile in candidates {
        let mut after = *hand;
        if after.remove(tile).is_err() {
            continue;
        }
        let deficit = evaluator.deficit(&after);
        let mut public = *public_before_discard;
        let index = tile.index() as usize;
        public[index] = public[index].saturating_add(1);
        let ukeire = analysis::ukeire(&mut evaluator, &after, &public, deficit);
        scored.push(ReferenceCandidate {
            tile,
            hand: after,
            deficit,
            ukeire,
            fan: 0,
        });
    }
    Some(scored)
}

/// Frozen ranking over already scored reference candidates.
///
/// This is the old selection logic: minimum deficit, minimum-only ukeire, then
/// the tenpai-only fan tie-break in candidate order. Empty scores select
/// nothing.
fn reference_ranking(
    scored: &mut [ReferenceCandidate],
    melds: &[Meld],
    missing: Suit,
    with_fan: bool,
) -> Option<(Tile, Progress)> {
    if scored.is_empty() {
        return None;
    }
    let best_deficit = scored
        .iter()
        .map(|candidate| candidate.deficit)
        .min()
        .expect("a scored discard exists");
    let mut best_ukeire = 0u32;
    for candidate in scored.iter().filter(|c| c.deficit == best_deficit) {
        best_ukeire = best_ukeire.max(candidate.ukeire);
    }

    let tenpai = best_deficit == 1;
    let finalists: Vec<usize> = (0..scored.len())
        .filter(|&index| {
            scored[index].deficit == best_deficit && scored[index].ukeire == best_ukeire
        })
        .collect();
    let best_index = if with_fan && tenpai && finalists.len() > 1 {
        let mut best = finalists[0];
        for &index in &finalists {
            scored[index].fan = analysis::max_discard_fan(&scored[index].hand, melds, missing);
            if scored[index].fan > scored[best].fan {
                best = index;
            }
        }
        best
    } else {
        finalists[0]
    };

    Some((
        scored[best_index].tile,
        Progress {
            shanten: best_deficit as i32 - 1,
            ukeire: best_ukeire,
        },
    ))
}

/// Frozen old path for one ranking call.
fn reference_rank_discards(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public_before_discard: &[u8; TILE_KINDS],
    candidates: &[Tile],
    with_fan: bool,
) -> Option<(Tile, Progress)> {
    let mut scored = reference_candidates(hand, melds, missing, public_before_discard, candidates)?;
    reference_ranking(&mut scored, melds, missing, with_fan)
}

/// Compares the production and removed-tile rankings for one context and
/// candidate list.
///
/// The full ranking must agree on the selected tile and `Progress`. Every
/// distinct candidate is then ranked alone, which exposes its exact deficit
/// (`shanten + 1`) and ukeire without adding production hooks; invalid
/// candidates must be skipped by both paths.
fn assert_rank_differential(
    label: &str,
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    candidates: &[Tile],
    with_fan: bool,
) {
    let actual = rank_discards(hand, melds, missing, public, candidates, with_fan);
    let mut reference = reference_candidates(hand, melds, missing, public, candidates)
        .expect("test contexts use a legal meld count");
    let expected = reference_ranking(&mut reference, melds, missing, with_fan);
    assert_eq!(actual, expected, "{label}: full candidate ranking");

    let mut unique = Vec::new();
    for &candidate in candidates {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    for candidate in unique {
        match reference.iter().find(|scored| scored.tile == candidate) {
            Some(scored) => {
                let (tile, progress) =
                    rank_discards(hand, melds, missing, public, &[candidate], false)
                        .expect("a held candidate is scored");
                assert_eq!(tile, candidate, "{label}: single candidate {candidate}");
                assert_eq!(
                    progress.shanten,
                    scored.deficit as i32 - 1,
                    "{label}: deficit after removing {candidate}"
                );
                assert_eq!(
                    progress.ukeire, scored.ukeire,
                    "{label}: ukeire after removing {candidate}"
                );
            }
            None => assert_eq!(
                rank_discards(hand, melds, missing, public, &[candidate], false),
                None,
                "{label}: invalid candidate {candidate} must be skipped"
            ),
        }
    }
}

/// Exact production deficit for one held candidate, recovered from a
/// single-candidate ranking where that candidate is necessarily minimal.
fn single_deficit(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    candidate: Tile,
) -> u8 {
    let (tile, progress) = rank_discards(hand, melds, missing, public, &[candidate], false)
        .unwrap_or_else(|| panic!("candidate {candidate} must be held"));
    assert_eq!(tile, candidate);
    u8::try_from(progress.shanten + 1).expect("deficit fits in u8")
}

fn plain_deficit(hand: &Hand, melds: &[Meld], missing: Suit) -> u8 {
    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid meld count");
    evaluator.deficit(hand)
}

fn all_candidates(hand: &Hand) -> Vec<Tile> {
    hand.distinct_tiles().collect()
}

fn assert_physical(hand: &Hand, melds: &[Meld], missing: Suit, public: &[u8; TILE_KINDS]) {
    assert!(melds.len() <= 4, "at most four declared melds");
    assert_eq!(
        hand.total(),
        14 - 3 * melds.len(),
        "ranking input holds the target-size concealed hand"
    );
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        assert!(
            hand.count(kind) + public[index] <= crate::tile::COPIES_PER_TILE,
            "{kind} exceeds four physical copies"
        );
    }
    for meld in melds {
        let index = meld.tile().index() as usize;
        assert_ne!(
            meld.tile().suit(),
            missing,
            "an own meld outside the missing suit"
        );
        assert!(
            public[index] >= meld.tile_count(),
            "public ownership includes the own meld copies"
        );
        assert!(
            hand.count(meld.tile()) + meld.tile_count() <= crate::tile::COPIES_PER_TILE,
            "concealed and meld copies stay within the physical deck"
        );
    }
}

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

#[test]
fn candidate_subsets_order_duplicates_invalid_and_empty_lists_match() {
    let missing = Suit::Tiao;
    let hand = tenpai_plus_extra();
    let public = [0u8; TILE_KINDS];
    let all = all_candidates(&hand);
    let one_wan = tile(Suit::Wan, 1);
    let nine_wan = tile(Suit::Wan, 9);
    let one_tong = tile(Suit::Tong, 1);
    let invalid = tile(Suit::Tong, 9);
    let another_invalid = tile(Suit::Tong, 8);

    assert_rank_differential("all candidates", &hand, &[], missing, &public, &all, true);

    let mut reversed = all.clone();
    reversed.reverse();
    assert_rank_differential(
        "reversed candidates",
        &hand,
        &[],
        missing,
        &public,
        &reversed,
        true,
    );
    assert_eq!(
        rank_discards(&hand, &[], missing, &public, &all, true),
        rank_discards(&hand, &[], missing, &public, &reversed, true),
        "candidate order must not change a strict winner"
    );

    assert_rank_differential(
        "candidate subset",
        &hand,
        &[],
        missing,
        &public,
        &[one_wan, one_tong],
        false,
    );
    // Ukeire is consulted only for minimum-deficit candidates: the worse
    // discard has more unseen copies, but the lower distance still wins.
    let scored = reference_candidates(&hand, &[], missing, &public, &[one_wan, one_tong])
        .expect("valid meld count");
    let more_ukeire = scored
        .iter()
        .find(|candidate| candidate.tile == one_wan)
        .expect("1W is scored");
    let lower_deficit = scored
        .iter()
        .find(|candidate| candidate.tile == one_tong)
        .expect("1T is scored");
    assert!(more_ukeire.deficit > lower_deficit.deficit);
    assert!(more_ukeire.ukeire > lower_deficit.ukeire);
    assert_eq!(
        rank_discards(&hand, &[], missing, &public, &[one_wan, one_tong], false),
        Some((
            one_tong,
            Progress {
                shanten: lower_deficit.deficit as i32 - 1,
                ukeire: lower_deficit.ukeire,
            }
        )),
        "minimum deficit precedes availability and progress reports it"
    );
    assert_rank_differential(
        "duplicate candidates",
        &hand,
        &[],
        missing,
        &public,
        &[one_tong, one_tong, one_wan, one_tong],
        false,
    );
    assert_rank_differential(
        "invalid entries mixed in",
        &hand,
        &[],
        missing,
        &public,
        &[one_wan, invalid, one_tong],
        false,
    );
    assert_rank_differential(
        "invalid entry between finalists",
        &hand,
        &[],
        missing,
        &public,
        &[nine_wan, invalid, one_tong],
        false,
    );

    assert_eq!(
        rank_discards(&hand, &[], missing, &public, &[], true),
        None,
        "an empty candidate list selects nothing"
    );
    assert_eq!(
        reference_rank_discards(&hand, &[], missing, &public, &[], true),
        None
    );
    assert_eq!(
        rank_discards(
            &hand,
            &[],
            missing,
            &public,
            &[invalid, another_invalid],
            true
        ),
        None,
        "an all-invalid candidate list selects nothing"
    );
    assert_eq!(
        reference_rank_discards(
            &hand,
            &[],
            missing,
            &public,
            &[invalid, another_invalid],
            true
        ),
        None
    );
}

#[test]
fn fan_tie_break_matches_the_reference_on_tenpai_ties() {
    let missing = Suit::Tiao;
    let hand = seven_pairs_plus_extra();
    let source = Seat::new(1).expect("seat");

    // Exhaust the standard waits so the two discards tie on deficit and
    // ukeire; fan then decides. This mirrors the existing decision-level fan
    // fixture.
    let mut discards = Vec::new();
    for kind in [
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 5),
    ] {
        for _ in 0..2 {
            discards.push(discard_info(source, kind, None, [false; 4]));
        }
    }
    let public = public_of(&discards, &[]);
    let lower_fan = tile(Suit::Wan, 1);
    let higher_fan = tile(Suit::Wan, 7);
    let candidates = [lower_fan, higher_fan];

    let without_fan = rank_discards(&hand, &[], missing, &public, &candidates, false)
        .expect("both discards are legal");
    let with_fan = rank_discards(&hand, &[], missing, &public, &candidates, true)
        .expect("both discards are legal");
    assert_eq!(
        without_fan.0, lower_fan,
        "candidate order decides without fan"
    );
    assert_eq!(with_fan.0, higher_fan, "fan decides the exact tie");
    assert_ne!(without_fan.0, with_fan.0);

    assert_rank_differential(
        "tenpai fan tie",
        &hand,
        &[],
        missing,
        &public,
        &candidates,
        true,
    );
    assert_rank_differential(
        "tenpai fan tie disabled",
        &hand,
        &[],
        missing,
        &public,
        &candidates,
        false,
    );

    // Availability still precedes fan: the higher-fan discard loses unseen
    // copies, so fan must not replace the better-ukeire finalist.
    let mut exhausted = Vec::new();
    for (kind, copies) in [
        (tile(Suit::Wan, 1), 1),
        (tile(Suit::Tong, 2), 2),
        (tile(Suit::Tong, 3), 2),
        (tile(Suit::Tong, 5), 2),
        (tile(Suit::Tong, 6), 2),
    ] {
        for _ in 0..copies {
            exhausted.push(discard_info(source, kind, None, [false; 4]));
        }
    }
    let exhausted_public = public_of(&exhausted, &[]);
    let availability_candidates = [higher_fan, lower_fan];
    assert_rank_differential(
        "ukeire precedes fan",
        &hand,
        &[],
        missing,
        &exhausted_public,
        &availability_candidates,
        true,
    );
    assert_eq!(
        rank_discards(
            &hand,
            &[],
            missing,
            &exhausted_public,
            &availability_candidates,
            true
        )
        .expect("both discards are legal")
        .0,
        lower_fan,
        "the higher-fan discard has no unseen wait left"
    );
}

#[test]
fn repeated_and_interleaved_contexts_stay_reference_equivalent() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");

    let standard = tenpai_plus_extra();
    let standard_public = [0u8; TILE_KINDS];
    let standard_candidates = all_candidates(&standard);

    let pairs = seven_pairs_plus_extra();
    let mut fan_discards = Vec::new();
    for kind in [
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 5),
    ] {
        for _ in 0..2 {
            fan_discards.push(discard_info(source, kind, None, [false; 4]));
        }
    }
    let fan_public = public_of(&fan_discards, &[]);
    let fan_candidates = [tile(Suit::Wan, 7), tile(Suit::Wan, 1)];

    let mut open_hand = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7], &[], &[]);
    open_hand
        .add(tile(Suit::Wan, 9))
        .expect("four copies at most");
    let open_public = [0u8; TILE_KINDS];
    let open_candidates = all_candidates(&open_hand);
    let missing_tong = Suit::Tong;

    let one_peng = [Meld::peng(tile(Suit::Tong, 6), source)];
    let peng_hand = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let peng_public = public_of(&[], &one_peng);

    let first_standard = rank_discards(
        &standard,
        &[],
        missing,
        &standard_public,
        &standard_candidates,
        true,
    );
    let first_fan = rank_discards(&pairs, &[], missing, &fan_public, &fan_candidates, true);
    let first_open = rank_discards(
        &open_hand,
        &[],
        missing_tong,
        &open_public,
        &open_candidates,
        true,
    );
    let first_peng = rank_discards(
        &peng_hand,
        &one_peng,
        missing,
        &peng_public,
        &all_candidates(&peng_hand),
        true,
    );

    // Interleaving different hands, meld sets, missing suits, and public
    // ownership must not leak cached analysis between contexts.
    for _ in 0..2 {
        assert_eq!(
            rank_discards(
                &standard,
                &[],
                missing,
                &standard_public,
                &standard_candidates,
                true
            ),
            first_standard
        );
        assert_eq!(
            rank_discards(&pairs, &[], missing, &fan_public, &fan_candidates, true),
            first_fan
        );
        assert_eq!(
            rank_discards(
                &open_hand,
                &[],
                missing_tong,
                &open_public,
                &open_candidates,
                true
            ),
            first_open
        );
        assert_eq!(
            rank_discards(
                &peng_hand,
                &one_peng,
                missing,
                &peng_public,
                &all_candidates(&peng_hand),
                true
            ),
            first_peng
        );
    }

    // Candidate order changes cache fill order inside one ranking call but
    // not the result.
    let mut reversed_standard = standard_candidates.clone();
    reversed_standard.reverse();
    assert_eq!(
        rank_discards(
            &standard,
            &[],
            missing,
            &standard_public,
            &reversed_standard,
            true
        ),
        first_standard
    );

    assert_rank_differential(
        "cold standard context",
        &standard,
        &[],
        missing,
        &standard_public,
        &standard_candidates,
        true,
    );
    assert_rank_differential(
        "warm fan context",
        &pairs,
        &[],
        missing,
        &fan_public,
        &fan_candidates,
        true,
    );
    assert_rank_differential(
        "different missing suit",
        &open_hand,
        &[],
        missing_tong,
        &open_public,
        &open_candidates,
        true,
    );
    assert_rank_differential(
        "melded context",
        &peng_hand,
        &one_peng,
        missing,
        &peng_public,
        &all_candidates(&peng_hand),
        true,
    );
}

/// Reference turn choice: only the ordinary-discard alternative comes from the
/// frozen removed-tile path. Kong positions, their ordinary deficits, income,
/// and the selection gate are the production helpers.
fn reference_turn_action(decision: &Decision<'_>, actions: &[Action], missing: Suit) -> Action {
    let observation = &decision.observation;
    let own_hand = observation.own_hand;
    let own_melds = observation.own_melds;
    let public = base_public(observation);
    let discard_tiles: Vec<Tile> = actions
        .iter()
        .filter_map(|action| match action {
            Action::Discard(tile) => Some(*tile),
            _ => None,
        })
        .collect();
    let (best_discard, alternative) =
        reference_rank_discards(own_hand, own_melds, missing, &public, &discard_tiles, true)
            .expect("fixture has a legal ordinary discard");

    let active_opponents = observation
        .players
        .iter()
        .filter(|player| player.seat != decision.seat && !player.has_won)
        .count();
    let mut choices = Vec::new();
    for action in actions {
        let (kind, hand, melds, adjusted_public) =
            match kong_position(*action, own_hand, own_melds, &public) {
                Some(position) => position,
                None => continue,
            };
        let Some(mut evaluator) = DistanceEvaluator::new(&melds, missing) else {
            continue;
        };
        let deficit = evaluator.deficit(&hand);
        choices.push(KongChoice {
            action: *action,
            progress: Progress {
                shanten: deficit as i32 - 1,
                ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted_public, deficit),
            },
            income: crate::rules::kong_payment(kind, active_opponents).total,
        });
    }

    match select_kong(alternative, &choices) {
        Some(action) => action,
        None => Action::Discard(best_discard),
    }
}

/// Reference peng position: only the mandatory post-peng discard ranking comes
/// from the frozen removed-tile path.
fn reference_evaluate_peng(
    own_hand: &Hand,
    own_melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    tile: Tile,
    source: Seat,
) -> Option<Progress> {
    let (_, hand, melds, public_after_claim) =
        claim_position(Action::Peng, own_hand, own_melds, public, tile, source)?;
    let candidates: Vec<Tile> = hand.distinct_tiles().collect();
    let (_, progress) = reference_rank_discards(
        &hand,
        &melds,
        missing,
        &public_after_claim,
        &candidates,
        false,
    )?;
    Some(progress)
}

/// Reference response choice: pass and open-kong progress are the production
/// ordinary-deficit helpers; only the peng ranking uses the frozen path. The
/// eligibility gates and response tie rule are shared with production.
fn reference_response_action(decision: &Decision<'_>, actions: &[Action], missing: Suit) -> Action {
    let observation = &decision.observation;
    let response = observation.response.expect("fixture is a discard response");
    let own_hand = observation.own_hand;
    let own_melds = observation.own_melds;
    let public = base_public(observation);

    let mut pass_evaluator =
        DistanceEvaluator::new(own_melds, missing).expect("fixture has legal melds");
    let pass_deficit = pass_evaluator.deficit(own_hand);
    let pass_progress = Progress {
        shanten: pass_deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut pass_evaluator, own_hand, &public, pass_deficit),
    };

    let peng_progress = if actions.contains(&Action::Peng) {
        reference_evaluate_peng(
            own_hand,
            own_melds,
            missing,
            &public,
            response.tile,
            response.source,
        )
    } else {
        None
    };
    let kong_progress = if actions.contains(&Action::OpenKong) {
        evaluate_open_kong(
            own_hand,
            own_melds,
            missing,
            &public,
            response.tile,
            response.source,
        )
    } else {
        None
    };

    let peng_eligible =
        peng_progress.is_some_and(|progress| progress.shanten < pass_progress.shanten);
    let kong_eligible = kong_progress.is_some_and(|progress| progress.no_worse_than(pass_progress));
    response_choice(peng_eligible, peng_progress, kong_eligible, kong_progress)
}

#[test]
fn kong_turn_decisions_match_the_removed_tile_reference() {
    let missing = Suit::Tiao;
    let mut policy = HeuristicPolicy::new();

    // Equal progress: the kong is preferred.
    let equal_progress_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 9], &[1, 2, 3, 7, 8, 9], &[]);
    let equal_actions = [
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Wan, 4)),
    ];
    let equal = decision(
        &equal_progress_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &equal_actions,
        &[],
    );
    assert_eq!(
        policy.choose(&equal),
        reference_turn_action(&equal, &equal_actions, missing)
    );
    assert_eq!(
        policy.choose(&equal),
        Action::ConcealedKong(tile(Suit::Wan, 1))
    );

    // Equal shanten but less ukeire: the kong gate rejects it.
    let less_ukeire_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let less_ukeire_actions = [
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Wan, 1)),
    ];
    let less_ukeire = decision(
        &less_ukeire_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &less_ukeire_actions,
        &[],
    );
    assert_eq!(
        policy.choose(&less_ukeire),
        reference_turn_action(&less_ukeire, &less_ukeire_actions, missing)
    );
    assert_eq!(
        policy.choose(&less_ukeire),
        Action::Discard(tile(Suit::Wan, 1))
    );

    // Added kong with an existing peng: equal progress, kong preferred.
    let added_melds = [Meld::peng(tile(Suit::Wan, 2), Seat::new(1).expect("seat"))];
    let added_hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let added_actions = [
        Action::AddedKong(tile(Suit::Wan, 2)),
        Action::Discard(tile(Suit::Wan, 2)),
    ];
    let added = decision(
        &added_hand,
        &added_melds,
        Some(missing),
        DecisionStage::Turn,
        None,
        &added_actions,
        &[],
    );
    assert_eq!(
        policy.choose(&added),
        reference_turn_action(&added, &added_actions, missing)
    );
    assert_eq!(policy.choose(&added), Action::AddedKong(tile(Suit::Wan, 2)));
}

#[test]
fn peng_and_open_kong_decisions_match_the_removed_tile_reference() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let mut policy = HeuristicPolicy::new();

    // Passing is already tenpai; the peng improves only ukeire and must lose.
    let tenpai = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let tenpai_claim = tile(Suit::Wan, 9);
    let tenpai_discards = [discard_info(source, tenpai_claim, None, [false; 4])];
    let tenpai_actions = [Action::Peng, Action::Pass];
    let tenpai_decision = decision(
        &tenpai,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: tenpai_claim,
            robbing_added_kong: false,
        }),
        &tenpai_actions,
        &tenpai_discards,
    );
    assert_eq!(
        policy.choose(&tenpai_decision),
        reference_response_action(&tenpai_decision, &tenpai_actions, missing)
    );
    assert_eq!(policy.choose(&tenpai_decision), Action::Pass);
    let tenpai_public = base_public(&tenpai_decision.observation);
    assert_eq!(
        evaluate_peng(&tenpai, &[], missing, &tenpai_public, tenpai_claim, source),
        reference_evaluate_peng(&tenpai, &[], missing, &tenpai_public, tenpai_claim, source),
        "peng progress uses the same candidate deficits as the old path"
    );

    // A peng that creates a set lowers shanten and wins.
    let open = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 9], &[7, 8, 9, 5, 5], &[]);
    let open_claim = tile(Suit::Wan, 1);
    let open_discards = [discard_info(source, open_claim, None, [false; 4])];
    let open_actions = [Action::Peng, Action::Pass];
    let open_decision = decision(
        &open,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: open_claim,
            robbing_added_kong: false,
        }),
        &open_actions,
        &open_discards,
    );
    assert_eq!(
        policy.choose(&open_decision),
        reference_response_action(&open_decision, &open_actions, missing)
    );
    assert_eq!(policy.choose(&open_decision), Action::Peng);
    let open_public = base_public(&open_decision.observation);
    assert_eq!(
        evaluate_peng(&open, &[], missing, &open_public, open_claim, source),
        reference_evaluate_peng(&open, &[], missing, &open_public, open_claim, source)
    );

    // Declaring a meld disables seven pairs, so this peng cannot strictly
    // improve.
    let six_pairs = suit_hand(&[1, 1, 7], &[2, 2, 3, 3, 4, 4, 5, 5, 6, 6], &[]);
    let six_pairs_claim = tile(Suit::Wan, 1);
    let six_pairs_discards = [discard_info(source, six_pairs_claim, None, [false; 4])];
    let six_pairs_actions = [Action::Peng, Action::Pass];
    let six_pairs_decision = decision(
        &six_pairs,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: six_pairs_claim,
            robbing_added_kong: false,
        }),
        &six_pairs_actions,
        &six_pairs_discards,
    );
    assert_eq!(
        policy.choose(&six_pairs_decision),
        reference_response_action(&six_pairs_decision, &six_pairs_actions, missing)
    );
    assert_eq!(policy.choose(&six_pairs_decision), Action::Pass);
    let six_pairs_public = base_public(&six_pairs_decision.observation);
    assert_eq!(
        evaluate_peng(
            &six_pairs,
            &[],
            missing,
            &six_pairs_public,
            six_pairs_claim,
            source
        ),
        reference_evaluate_peng(
            &six_pairs,
            &[],
            missing,
            &six_pairs_public,
            six_pairs_claim,
            source
        )
    );

    // Equal shanten with less ukeire: the open kong gate rejects it, and the
    // peng also fails its strict-shanten gate.
    let kong_hand = suit_hand(&[1, 1, 1, 2, 3, 4, 5, 6, 9, 9], &[1, 2, 3], &[]);
    let kong_claim = tile(Suit::Wan, 1);
    let kong_discards = [discard_info(source, kong_claim, None, [false; 4])];
    let kong_actions = [Action::Peng, Action::OpenKong, Action::Pass];
    let kong_decision = decision(
        &kong_hand,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: kong_claim,
            robbing_added_kong: false,
        }),
        &kong_actions,
        &kong_discards,
    );
    assert_eq!(
        policy.choose(&kong_decision),
        reference_response_action(&kong_decision, &kong_actions, missing)
    );
    assert_eq!(policy.choose(&kong_decision), Action::Pass);
}

/// Deterministic LCG, matching the analysis sampling style without adding a
/// test-only dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        ((self.next() >> 33) % bound as u64) as usize
    }
}

fn hand_from_counts(counts: &[u8; TILE_KINDS]) -> Hand {
    let mut hand = Hand::new();
    for kind in Tile::ALL {
        for _ in 0..counts[kind.index() as usize] {
            hand.add(kind).expect("at most four copies per kind");
        }
    }
    hand
}

struct SampledRankCase {
    hand: Hand,
    melds: Vec<Meld>,
    public: [u8; TILE_KINDS],
}

/// Builds one physical ranking input: a target-size concealed hand, distinct
/// meld kinds outside the missing suit, and public ownership that includes the
/// own melds.
fn sampled_rank_case(
    seed: u64,
    meld_count: usize,
    missing: Suit,
    include_missing_tiles: bool,
) -> SampledRankCase {
    let mut rng = Lcg::new(seed);

    let mut pool: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .collect();
    for index in (1..pool.len()).rev() {
        let swap = rng.below(index + 1);
        pool.swap(index, swap);
    }

    let mut melds = Vec::new();
    let mut meld_copies = [0u8; TILE_KINDS];
    for (index, &kind) in pool.iter().take(meld_count).enumerate() {
        let meld = match index % 4 {
            0 => Meld::peng(kind, Seat::DEALER),
            1 => Meld::open_kong(kind, Seat::DEALER),
            2 => Meld::concealed_kong(kind),
            _ => Meld::added_kong(kind, Seat::DEALER),
        };
        meld_copies[kind.index() as usize] = meld.tile_count();
        melds.push(meld);
    }

    let mut counts = [0u8; TILE_KINDS];
    let mut remaining = 14 - 3 * meld_count;
    if include_missing_tiles {
        for _ in 0..=rng.below(2) {
            if remaining == 0 {
                break;
            }
            let kind = tile(missing, 1 + rng.below(9) as u8);
            let index = kind.index() as usize;
            if counts[index] < crate::tile::COPIES_PER_TILE {
                counts[index] += 1;
                remaining -= 1;
            }
        }
    }
    while remaining > 0 {
        let kind = pool[rng.below(pool.len())];
        let index = kind.index() as usize;
        if counts[index] + meld_copies[index] >= crate::tile::COPIES_PER_TILE {
            continue;
        }
        counts[index] += 1;
        remaining -= 1;
    }
    let hand = hand_from_counts(&counts);

    let mut public = meld_copies;
    for _ in 0..rng.below(6) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        if public[index] + counts[index] >= crate::tile::COPIES_PER_TILE {
            continue;
        }
        public[index] += 1;
    }
    for _ in 0..rng.below(3) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        public[index] = crate::tile::COPIES_PER_TILE - counts[index];
    }

    SampledRankCase {
        hand,
        melds,
        public,
    }
}

fn sampled_candidates(hand: &Hand, variant: usize) -> Vec<Tile> {
    let mut distinct = all_candidates(hand);
    match variant {
        0 => distinct,
        1 => {
            distinct.reverse();
            distinct
        }
        2 => {
            distinct.truncate((distinct.len() / 2).max(1));
            distinct
        }
        _ => {
            let mut candidates = distinct.clone();
            if let Some(&first) = distinct.first() {
                candidates.push(first);
            }
            if let Some(&last) = distinct.last() {
                candidates.push(last);
            }
            if let Some(&unheld) = Tile::ALL.iter().find(|kind| hand.count(**kind) == 0) {
                candidates.push(unheld);
            }
            candidates
        }
    }
}

#[test]
fn sampled_rankings_match_the_removed_tile_reference() {
    // Seed base and stride follow the analysis sampling convention. The sweep
    // is 3 missing suits * 5 meld counts * 4 candidate variants = 60 physical
    // rankings; two variants keep missing-suit tiles in hand.
    const BASE: u64 = 0x5EED_0A11_0F0F_0001;
    const STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut index = 0u64;
    let mut missing_held = 0usize;
    let mut invalid_lists = 0usize;
    let mut surplus_hands = 0usize;
    for missing in Suit::ALL {
        for meld_count in 0..=4 {
            for variant in 0..4 {
                let seed = BASE.wrapping_add(STRIDE.wrapping_mul(index));
                index += 1;
                let case = sampled_rank_case(seed, meld_count, missing, variant % 2 == 1);
                assert_physical(&case.hand, &case.melds, missing, &case.public);
                let candidates = sampled_candidates(&case.hand, variant);
                missing_held += usize::from(case.hand.suit_count(missing) > 0);
                invalid_lists +=
                    usize::from(candidates.iter().any(|&kind| case.hand.count(kind) == 0));
                let mut evaluator = DistanceEvaluator::new(&case.melds, missing).unwrap();
                surplus_hands += usize::from(evaluator.deficit_with_surplus(&case.hand).1 != 0);
                let label = format!(
                    "sampled missing={missing} melds={meld_count} variant={variant} seed={seed:#x} counts={:?}",
                    case.hand.counts()
                );
                assert_rank_differential(
                    &label,
                    &case.hand,
                    &case.melds,
                    missing,
                    &case.public,
                    &candidates,
                    true,
                );
            }
        }
    }
    assert_eq!(index, 60, "sample sweep size changed");
    assert!(
        missing_held > 0,
        "sample sweep must retain missing-suit tiles"
    );
    assert!(
        invalid_lists > 0,
        "sample sweep must include invalid candidates"
    );
    assert!(
        surplus_hands > 0,
        "sample sweep must include optimal surplus"
    );
}
