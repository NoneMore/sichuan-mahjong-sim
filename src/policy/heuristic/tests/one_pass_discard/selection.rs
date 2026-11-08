//! Selection evidence: candidate subsets, duplicates, fan tie-breaks, and
//! interleaved contexts.
//!
//! The expectation comes from the frozen reference ranking in `reference.rs`;
//! production runs the unchanged `rank_discards` path.

use super::super::*;

use super::reference::{reference_candidates, reference_rank_discards};
use super::{all_candidates, assert_rank_differential};

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
        rank_discards(&hand, &[], missing, &public, &all, true, &uncached()),
        rank_discards(&hand, &[], missing, &public, &reversed, true, &uncached()),
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
        rank_discards(
            &hand,
            &[],
            missing,
            &public,
            &[one_wan, one_tong],
            false,
            &uncached()
        ),
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
        rank_discards(&hand, &[], missing, &public, &[], true, &uncached()),
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
            true,
            &uncached()
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

    let without_fan = rank_discards(
        &hand,
        &[],
        missing,
        &public,
        &candidates,
        false,
        &uncached(),
    )
    .expect("both discards are legal");
    let with_fan = rank_discards(&hand, &[], missing, &public, &candidates, true, &uncached())
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
            true,
            &uncached()
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
        &uncached(),
    );
    let first_fan = rank_discards(
        &pairs,
        &[],
        missing,
        &fan_public,
        &fan_candidates,
        true,
        &uncached(),
    );
    let first_open = rank_discards(
        &open_hand,
        &[],
        missing_tong,
        &open_public,
        &open_candidates,
        true,
        &uncached(),
    );
    let first_peng = rank_discards(
        &peng_hand,
        &one_peng,
        missing,
        &peng_public,
        &all_candidates(&peng_hand),
        true,
        &uncached(),
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
                true,
                &uncached()
            ),
            first_standard
        );
        assert_eq!(
            rank_discards(
                &pairs,
                &[],
                missing,
                &fan_public,
                &fan_candidates,
                true,
                &uncached()
            ),
            first_fan
        );
        assert_eq!(
            rank_discards(
                &open_hand,
                &[],
                missing_tong,
                &open_public,
                &open_candidates,
                true,
                &uncached()
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
                true,
                &uncached()
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
            true,
            &uncached()
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
