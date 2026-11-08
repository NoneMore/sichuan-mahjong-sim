//! Turn responses, claim selection, and the passed-Hu lock scenarios.

use super::super::*;
use super::fixtures::*;

#[test]
fn pass_then_lower_claim_is_allowed_and_peng_does_not_draw() {
    let mut game = peng_and_hu_setup();
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));

    submit_specific(&mut game, seat(1), Action::Pass);
    assert!(game.pass_lock[1], "declining Hu locks the next Hu");
    assert!(game.winners.iter().all(|winner| !winner));
    let follow_up = {
        let decision = game.next_decision().unwrap();
        assert_eq!(decision.seat, seat(1));
        match decision.legal_actions {
            LegalActions::Actions(actions) => actions.to_vec(),
            LegalActions::Exchange { .. } => panic!("expected follow-up response"),
        }
    };
    assert!(!follow_up.contains(&Action::Hu));
    assert!(follow_up.contains(&Action::Peng));
    assert!(follow_up.contains(&Action::Pass));

    let wall_before = game.wall_remaining();
    submit_specific(&mut game, seat(1), Action::Peng);

    assert_eq!(game.melds[1][0].kind(), MeldKind::Peng);
    assert_eq!(game.hands[1].total(), 11);
    assert_eq!(game.wall_remaining(), wall_before, "peng does not draw");
    assert_eq!(game.discards.last().unwrap().meld_claim, Some(seat(1)));
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterPeng
        }
    ));
    let actions = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected turn actions"),
    };
    assert!(actions
        .iter()
        .all(|action| matches!(action, Action::Discard(_))));
    assert_valid(&game);
}

#[test]
fn a_second_pass_completes_a_two_stage_response_without_looping() {
    let mut game = peng_and_hu_setup();
    let wall_before = game.wall_remaining();
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));
    submit_specific(&mut game, seat(1), Action::Pass);
    assert!(game.pass_lock[1], "the first Pass declines the win");

    let follow_up = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected the staged follow-up"),
    };
    assert!(!follow_up.contains(&Action::Hu));
    assert!(follow_up.contains(&Action::Peng));
    assert!(follow_up.contains(&Action::Pass));

    submit_specific(&mut game, seat(1), Action::Pass);

    assert!(
        game.winners.iter().all(|&won| !won),
        "declining twice must not win"
    );
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterDraw(DrawKind::Normal)
        }
    ));
    assert_eq!(game.discards.len(), 1);
    assert_eq!(game.discards[0].meld_claim, None);
    assert_eq!(
        game.wall_remaining(),
        wall_before - 1,
        "the response resolved into the next normal draw"
    );
    assert_valid(&game);
}

/// Four copies of 5W: the claimant holds three and its Hu shape uses all four
/// as a triplet (555W) plus a sequence (567W). Hu and OpenKong are therefore
/// both legal on the same discard with globally legal (four-copy) tile counts.
fn hu_and_open_kong_setup() -> Game {
    let mut game = blank_game();
    game.hands[0] = hand(&[w(5)]);
    game.hands[1] = hand(&[
        w(5),
        w(5),
        w(5),
        w(6),
        w(7),
        w(1),
        w(2),
        w(3),
        t(1),
        t(2),
        t(3),
        t(9),
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

#[test]
fn passing_then_open_kong_claims_the_discard_after_declining_hu() {
    let mut game = hu_and_open_kong_setup();
    assert!(game.can_hu_on_tile(seat(1), w(5)));
    assert!(rules::can_open_kong(
        &game.hands[1],
        &game.melds[1],
        Suit::Tiao,
        w(5)
    ));
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));

    let first = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected response actions"),
    };
    assert!(first.contains(&Action::Hu));
    assert!(first.contains(&Action::OpenKong));

    submit_specific(&mut game, seat(1), Action::Pass);
    assert!(game.pass_lock[1], "declining Hu locks the next Hu");
    let follow_up = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected the staged follow-up"),
    };
    assert!(!follow_up.contains(&Action::Hu));
    assert!(follow_up.contains(&Action::OpenKong));
    assert!(follow_up.contains(&Action::Pass));

    submit_specific(&mut game, seat(1), Action::OpenKong);

    assert_eq!(game.melds[1][0].kind(), MeldKind::OpenKong);
    assert_eq!(
        game.hands[1].total(),
        11,
        "three hand tiles plus the discard become the kong, then one replacement is drawn"
    );
    assert_eq!(game.discards[0].meld_claim, Some(seat(1)));
    assert_eq!(game.scores, [-2, 2, 0, 0]);
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert_valid(&game);
}

#[test]
fn passing_a_win_is_cleared_by_the_next_normal_draw() {
    let mut game = single_waiter_setup();
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));
    let wall_before = game.wall_remaining();
    submit_specific(&mut game, seat(1), Action::Pass);

    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterDraw(DrawKind::Normal)
        }
    ));
    assert!(!game.pass_lock[1], "normal draw clears the passed-Hu lock");
    assert_eq!(game.wall_remaining(), wall_before - 1);
    assert_valid(&game);
}

#[test]
fn passing_a_self_draw_is_cleared_by_a_kong_replacement_draw() {
    let mut game = blank_game();
    game.hands[0] = hand(&[
        w(1),
        w(1),
        w(1),
        w(1),
        w(2),
        w(3),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(9),
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    submit_specific(&mut game, seat(0), Action::Pass);
    assert!(game.pass_lock[0]);
    let after_pass = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected turn actions"),
    };
    assert!(!after_pass.contains(&Action::Hu));
    assert!(after_pass.contains(&Action::ConcealedKong(w(1))));

    submit_specific(&mut game, seat(0), Action::ConcealedKong(w(1)));
    assert!(
        !game.pass_lock[0],
        "a kong replacement draw clears the passed-Hu lock"
    );
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S0,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert_valid(&game);
}

/// A self-draw winning hand that also leaves a waiting hand: the Discard
/// action must arm the passed-Hu lock, so the same seat cannot claim the tile
/// its own discard would have given it.
#[test]
fn discarding_a_winning_hand_arms_the_passed_hu_lock() {
    let mut game = blank_game();
    let mut winner = vec![w(1), w(2), w(3), w(7), w(7)];
    winner.extend(tong_run());
    game.hands[0] = hand(&winner);
    game.hands[1] = hand(&[
        w(4),
        w(5),
        w(6),
        w(8),
        w(9),
        w(9),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    let actions = pending_actions(&mut game);
    assert!(actions.contains(&Action::Hu), "the hand is already winning");
    assert_eq!(actions.last(), Some(&Action::Pass));

    submit_specific(&mut game, S0, Action::Discard(w(1)));
    assert!(
        game.pass_lock[0],
        "discarding while Hu is available locks the next Hu"
    );
    assert!(!game.pass_lock[1] && !game.pass_lock[2] && !game.pass_lock[3]);
    assert_eq!(game.pending.expect("seat 1 turn").seat, seat(1));
    assert_valid(&game);

    // Seat 1 draws, then discards the 4W that would complete seat 0's 234W plus
    // 77W pair. The lock turns that would-be win into an automatic Pass.
    let actions = pending_actions(&mut game);
    assert_eq!(game.pending.expect("seat 1 turn").seat, seat(1));
    assert!(actions.contains(&Action::Discard(w(4))));
    submit_specific(&mut game, seat(1), Action::Discard(w(4)));

    assert_eq!(
        game.pending.expect("seat 2 turn").seat,
        seat(2),
        "the locked seat must never re-enter the response stage"
    );
    assert!(game.winners.iter().all(|&won| !won));
    let discard = game.discards.last().expect("discard record");
    assert_eq!(discard.tile, w(4));
    assert_eq!(discard.winners, [false; PLAYER_COUNT]);
    assert_eq!(discard.meld_claim, None);
    assert_valid(&game);
}
