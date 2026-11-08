//! Decision-level evidence: only the ordinary-discard ranking comes from the
//! frozen removed-tile reference; kong and peng gates stay production helpers.

use super::super::*;

use super::reference::reference_rank_discards;

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
            &uncached(),
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
        evaluate_peng(
            &tenpai,
            &[],
            missing,
            &tenpai_public,
            tenpai_claim,
            source,
            &uncached()
        ),
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
        evaluate_peng(
            &open,
            &[],
            missing,
            &open_public,
            open_claim,
            source,
            &uncached()
        ),
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
            source,
            &uncached()
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
