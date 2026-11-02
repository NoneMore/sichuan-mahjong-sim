use super::*;

#[test]
fn peng_must_strictly_lower_shanten() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");

    // Pass is already tenpai. The peng leaves the same distance but more
    // unseen improving copies, which is explicitly insufficient.
    let tenpai = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let claim = tile(Suit::Wan, 9);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);
    let pass = progress_of(&tenpai, &[], missing, &public);
    let peng = evaluate_peng(&tenpai, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert_eq!(peng.shanten, pass.shanten);
    assert!(
        peng.ukeire > pass.ukeire,
        "test premise: the peng raises availability"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &tenpai,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);

    // A peng that creates a set strictly lowers shanten and is accepted.
    let open = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 9], &[7, 8, 9, 5, 5], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);
    let pass = progress_of(&open, &[], missing, &public);
    let peng = evaluate_peng(&open, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert!(
        peng.shanten < pass.shanten,
        "test premise: the peng must lower shanten"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &open,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Peng);
}

#[test]
fn peng_cannot_use_the_seven_pairs_route_after_a_meld() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let six_pairs = suit_hand(&[1, 1, 7], &[2, 2, 3, 3, 4, 4, 5, 5, 6, 6], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);

    let pass = progress_of(&six_pairs, &[], missing, &public);
    assert_eq!(pass.shanten, 0, "the hand is a seven-pairs tenpai");
    let peng = evaluate_peng(&six_pairs, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert!(
        peng.shanten >= pass.shanten,
        "declaring a meld disables seven pairs, so it cannot strictly improve"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &six_pairs,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);
}

#[test]
fn concealed_kong_gate_rejects_equal_shanten_with_less_ukeire() {
    let missing = Suit::Tiao;
    let kong_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let kong_tile = tile(Suit::Wan, 1);
    let public = [0u8; TILE_KINDS];

    let discard = discard_progress(&kong_hand, kong_tile, &[], missing, &public);
    let (kind, hand, melds, adjusted) =
        kong_position(Action::ConcealedKong(kong_tile), &kong_hand, &[], &public)
            .expect("concealed kong position");
    assert_eq!(kind, MeldKind::ConcealedKong);
    let mut evaluator = DistanceEvaluator::new(&melds, missing).unwrap();
    let deficit = evaluator.deficit(&hand);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted, deficit),
    };
    assert_eq!(kong.shanten, discard.shanten);
    assert!(
        kong.ukeire < discard.ukeire,
        "test premise: the kong loses availability"
    );

    let actions = [Action::ConcealedKong(kong_tile), Action::Discard(kong_tile)];
    let choice = decision(
        &kong_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::Discard(kong_tile)
    );
}

#[test]
fn eligible_kong_is_preferred_on_equal_progress() {
    let missing = Suit::Tiao;
    let kong_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 9], &[1, 2, 3, 7, 8, 9], &[]);
    let kong_tile = tile(Suit::Wan, 1);
    let discard_tile = tile(Suit::Wan, 4);
    let public = [0u8; TILE_KINDS];

    let discard = discard_progress(&kong_hand, discard_tile, &[], missing, &public);
    let (_, hand, melds, adjusted) =
        kong_position(Action::ConcealedKong(kong_tile), &kong_hand, &[], &public)
            .expect("concealed kong position");
    let mut evaluator = DistanceEvaluator::new(&melds, missing).unwrap();
    let deficit = evaluator.deficit(&hand);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted, deficit),
    };
    assert_eq!(
        kong, discard,
        "test premise: the kong has the same progress rank"
    );

    let actions = [
        Action::ConcealedKong(kong_tile),
        Action::Discard(discard_tile),
    ];
    let choice = decision(
        &kong_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::ConcealedKong(kong_tile)
    );
}

#[test]
fn open_kong_gate_rejects_equal_shanten_with_less_ukeire() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let hand = suit_hand(&[1, 1, 1, 2, 3, 4, 5, 6, 9, 9], &[1, 2, 3], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);

    let pass = progress_of(&hand, &[], missing, &public);
    let kong = evaluate_open_kong(&hand, &[], missing, &public, claim, source)
        .expect("open kong position");
    assert_eq!(pass.shanten, 0);
    assert_eq!(kong.shanten, pass.shanten);
    assert!(
        kong.ukeire < pass.ukeire,
        "test premise: the open kong loses availability"
    );

    // Both claims are legal, but neither passes its gate against pass.
    let actions = [Action::Peng, Action::OpenKong, Action::Pass];
    let choice = decision(
        &hand,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);
}

#[test]
fn kong_gate_and_selection_rules_are_lexicographic() {
    let alternative = Progress {
        shanten: 1,
        ukeire: 10,
    };
    let lower_shanten = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 1)),
        progress: Progress {
            shanten: 0,
            ukeire: 1,
        },
        income: 2,
    };
    assert_eq!(
        select_kong(alternative, &[lower_shanten]),
        Some(lower_shanten.action),
        "a lower shanten may justify lower ukeire"
    );

    let equal_lower_ukeire = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 2)),
        progress: Progress {
            shanten: 1,
            ukeire: 9,
        },
        income: 100,
    };
    assert_eq!(
        select_kong(alternative, &[equal_lower_ukeire]),
        None,
        "equal shanten with lower ukeire fails regardless of income"
    );

    let equal_progress = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 3)),
        progress: alternative,
        income: 1,
    };
    assert_eq!(
        select_kong(alternative, &[equal_progress]),
        Some(equal_progress.action)
    );

    let better_progress = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 4)),
        progress: Progress {
            shanten: 1,
            ukeire: 11,
        },
        income: 1,
    };
    let richer = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 5)),
        progress: Progress {
            shanten: 1,
            ukeire: 10,
        },
        income: 100,
    };
    assert_eq!(
        select_kong(alternative, &[richer, better_progress]),
        Some(better_progress.action),
        "progress rank precedes nominal income"
    );

    let concealed = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 6)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 6,
    };
    let added = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 7)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 3,
    };
    assert_eq!(
        select_kong(alternative, &[added, concealed]),
        Some(concealed.action),
        "equal progress prefers the larger nominal income"
    );

    let earlier = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 8)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 6,
    };
    assert_eq!(
        select_kong(alternative, &[earlier, concealed]),
        Some(earlier.action),
        "an exact tie keeps the earlier fixed-order action"
    );
}

#[test]
fn open_kong_wins_a_response_tie_with_peng() {
    let equal = Progress {
        shanten: 0,
        ukeire: 5,
    };
    assert_eq!(
        response_choice(true, Some(equal), true, Some(equal)),
        Action::OpenKong,
        "the kong wins an exact progress tie"
    );

    let better_peng = Progress {
        shanten: 0,
        ukeire: 6,
    };
    assert_eq!(
        response_choice(true, Some(better_peng), true, Some(equal)),
        Action::Peng
    );
    assert_eq!(
        response_choice(true, Some(equal), true, Some(better_peng)),
        Action::OpenKong
    );
    assert_eq!(
        response_choice(true, Some(equal), false, None),
        Action::Peng
    );
    assert_eq!(
        response_choice(false, None, true, Some(equal)),
        Action::OpenKong
    );
    assert_eq!(response_choice(false, None, false, None), Action::Pass);
}

#[test]
fn hypothetical_meld_positions_conserve_physical_ownership() {
    let physical = |hand: &Hand, public: &[u8; TILE_KINDS]| -> [u8; TILE_KINDS] {
        let counts = std::array::from_fn(|index| hand.counts()[index] + public[index]);
        assert!(counts.iter().all(|&count| count <= 4));
        counts
    };
    let source = Seat::new(1).expect("seat");
    let discards = [
        discard_info(source, tile(Suit::Wan, 8), None, [false; 4]),
        discard_info(source, tile(Suit::Tong, 2), None, [false; 4]),
    ];
    let public = public_of(&discards, &[]);

    // Concealed kong on a turn.
    let turn_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 9], &[1, 2, 3, 7, 8, 9], &[]);
    let (_, concealed, _, concealed_public) = kong_position(
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        &turn_hand,
        &[],
        &public,
    )
    .expect("concealed kong position");
    assert_eq!(
        physical(&concealed, &concealed_public),
        physical(&turn_hand, &public)
    );

    // Added kong on a turn: the existing peng is already public ownership.
    let melds = [Meld::peng(tile(Suit::Wan, 2), source)];
    let public_with_peng = public_of(&discards, &melds);
    let added_hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let (_, after_added, _, added_public) = kong_position(
        Action::AddedKong(tile(Suit::Wan, 2)),
        &added_hand,
        &melds,
        &public_with_peng,
    )
    .expect("added kong position");
    assert_eq!(
        physical(&after_added, &added_public),
        physical(&added_hand, &public_with_peng)
    );

    // Peng and open kong on a response: the pending discard is in `public`.
    let claim = tile(Suit::Wan, 1);
    let response_discards = [discard_info(source, claim, None, [false; 4])];
    let response_public = public_of(&response_discards, &[]);
    let response_hand = suit_hand(&[1, 1, 1, 2, 3, 4, 5, 6, 9, 9], &[1, 2, 3], &[]);
    let (_, after_peng, _, peng_public) = super::super::claim_position(
        Action::Peng,
        &response_hand,
        &[],
        &response_public,
        claim,
        source,
    )
    .expect("peng position");
    assert_eq!(
        physical(&after_peng, &peng_public),
        physical(&response_hand, &response_public)
    );

    // A mandatory post-peng discard only moves ownership.
    let discard = after_peng
        .distinct_tiles()
        .next()
        .expect("peng hand is non-empty");
    let mut after_peng_discard = after_peng;
    after_peng_discard.remove(discard).unwrap();
    let mut peng_public_discard = peng_public;
    peng_public_discard[discard.index() as usize] =
        peng_public_discard[discard.index() as usize].saturating_add(1);
    assert_eq!(
        physical(&after_peng_discard, &peng_public_discard),
        physical(&response_hand, &response_public)
    );

    let (_, after_open_kong, _, kong_public) = super::super::claim_position(
        Action::OpenKong,
        &response_hand,
        &[],
        &response_public,
        claim,
        source,
    )
    .expect("open kong position");
    assert_eq!(
        physical(&after_open_kong, &kong_public),
        physical(&response_hand, &response_public)
    );
}

#[test]
fn added_kong_path_is_evaluated_and_can_be_chosen() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let melds = [Meld::peng(tile(Suit::Wan, 2), source)];
    let hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let kong_tile = tile(Suit::Wan, 2);
    let public = public_of(&[], &melds);

    let discard = discard_progress(&hand, kong_tile, &melds, missing, &public);
    let (kind, after, after_melds, after_public) =
        kong_position(Action::AddedKong(kong_tile), &hand, &melds, &public)
            .expect("added kong position");
    assert_eq!(kind, MeldKind::AddedKong);
    assert_eq!(after_melds.len(), melds.len());
    let mut evaluator = DistanceEvaluator::new(&after_melds, missing).unwrap();
    let deficit = evaluator.deficit(&after);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &after, &after_public, deficit),
    };
    assert_eq!(
        kong, discard,
        "test premise: the added kong and the fourth-tile discard have equal progress"
    );

    let actions = [Action::AddedKong(kong_tile), Action::Discard(kong_tile)];
    let choice = decision(
        &hand,
        &melds,
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::AddedKong(kong_tile)
    );
}

#[test]
fn zero_availability_tenpai_still_uses_the_fan_tie_break() {
    let missing = Suit::Tiao;
    let hand = seven_pairs_plus_extra();
    let source = Seat::new(1).expect("seat");

    // Exhaust every structural wait of both discards. Availability is zero,
    // but both hands remain tenpai, so fan is still the final tie-break.
    let mut discards = Vec::new();
    for (kind, copies) in [
        (tile(Suit::Wan, 1), 1),
        (tile(Suit::Tong, 2), 2),
        (tile(Suit::Tong, 3), 2),
        (tile(Suit::Tong, 5), 2),
        (tile(Suit::Tong, 6), 2),
        (tile(Suit::Wan, 7), 3),
    ] {
        for _ in 0..copies {
            discards.push(discard_info(source, kind, None, [false; 4]));
        }
    }
    let public = public_of(&discards, &[]);

    let lower_fan = tile(Suit::Wan, 1);
    let higher_fan = tile(Suit::Wan, 7);
    let low = discard_progress(&hand, lower_fan, &[], missing, &public);
    let high = discard_progress(&hand, higher_fan, &[], missing, &public);
    assert_eq!(low.shanten, 0);
    assert_eq!(high.shanten, 0);
    assert_eq!(low.ukeire, 0);
    assert_eq!(high.ukeire, 0);

    let mut low_after = hand;
    low_after.remove(lower_fan).unwrap();
    let mut high_after = hand;
    high_after.remove(higher_fan).unwrap();
    assert!(
        analysis::max_discard_fan(&high_after, &[], missing)
            > analysis::max_discard_fan(&low_after, &[], missing)
    );

    let actions = [Action::Discard(lower_fan), Action::Discard(higher_fan)];
    let choice = decision(
        &hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &discards,
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::Discard(higher_fan),
        "S = 0 and U = 0 is still tenpai and must use the fan tie-break"
    );
}

#[test]
fn repeated_decisions_are_cache_and_constructor_independent() {
    let missing = Suit::Tiao;
    let candidate_hand = tenpai_plus_extra();
    let actions = [
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Tong, 1)),
    ];
    let choice = decision(
        &candidate_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );

    let mut policy = HeuristicPolicy::new();
    let first = policy.choose(&choice);

    // A different decision cannot change the next result: the policy holds no
    // logical history or decision cache.
    let other_hand = suit_hand(&[1, 4, 7, 2, 9, 9], &[1, 4, 7, 2], &[1, 4, 7, 2]);
    let other_actions = [Action::Discard(tile(Suit::Wan, 7))];
    let other = decision(
        &other_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &other_actions,
        &[],
    );
    let _ = policy.choose(&other);

    assert_eq!(policy.choose(&choice), first);
    let mut default_policy: HeuristicPolicy = Default::default();
    assert_eq!(default_policy.choose(&choice), first);
}
