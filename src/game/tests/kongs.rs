//! Peng and kong claims with their immediate payments.

use super::super::*;
use super::fixtures::*;

#[test]
fn robbed_added_kong_is_not_committed_and_pays_no_kong_income() {
    let mut game = added_kong_setup();
    let wall_before = game.wall_remaining();
    let tail_before = game.wall_tail;
    submit_specific(&mut game, seat(0), Action::AddedKong(w(5)));

    assert!(matches!(game.stage, Stage::Response(_)));
    assert_eq!(game.scores, [0, 0, 0, 0]);
    assert!(game.kong_ledger.is_empty());
    assert_eq!(game.melds[0][0].kind(), MeldKind::Peng);
    assert_eq!(game.hands[0].count(w(5)), 1);
    assert_eq!(game.wall_remaining(), wall_before);

    submit_specific(&mut game, seat(1), Action::Hu);

    assert_eq!(game.scores, [-2, 2, 0, 0]);
    assert_eq!(game.melds[0][0].kind(), MeldKind::Peng);
    assert_eq!(game.hands[0].count(w(5)), 0);
    assert!(game.kong_ledger.is_empty());
    assert_eq!(game.wall_tail, tail_before, "robbing draws no replacement");
    assert_eq!(
        game.wall_remaining(),
        wall_before - 1,
        "only the following normal turn draws"
    );
    let discard = game.discards.last().unwrap();
    assert!(discard.robbed_kong);
    assert_eq!(discard.seat, seat(0));
    assert!(discard.winners[1]);
    assert_valid(&game);
}

#[test]
fn unrobbed_added_kong_pays_then_draws_one_replacement() {
    let mut game = added_kong_setup();
    let wall_before = game.wall_remaining();
    submit_specific(&mut game, seat(0), Action::AddedKong(w(5)));
    assert!(!game.pass_lock[1]);
    submit_specific(&mut game, seat(1), Action::Pass);

    assert_eq!(game.scores, [3, -1, -1, -1]);
    assert_eq!(game.kong_ledger.len(), 3);
    assert_eq!(game.melds[0][0].kind(), MeldKind::AddedKong);
    assert_eq!(game.hands[0].total(), 11);
    assert_eq!(game.wall_remaining(), wall_before - 1);
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S0,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert!(game.pass_lock[1], "the passer did not draw");
    assert_valid(&game);
}

/// Seat 0 owns a peng of 5W plus the fourth 5W and holds the 9W it will
/// discard. Seat 1 waits on 9W for a plain one-fan discard win, so the same
/// discard can be compared with and without a preceding kong.
fn kong_discard_setup() -> Game {
    let mut game = blank_game();
    game.melds[0] = vec![Meld::peng(w(5), seat(1))];
    game.hands[0] = added_kong_declarer_hand();
    game.hands[1] = nine_wan_waiter_hand();
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

/// Swaps `wanted` into the wall tail so a kong replacement draw is predictable.
/// `refresh_wall` fills the wall in tile order, so the tail otherwise holds
/// whichever remaining tile has the highest index.
fn put_tail_tile(game: &mut Game, wanted: Tile) {
    let tail = game.wall_tail - 1;
    if game.wall[tail] == wanted.index() {
        return;
    }
    let position = (game.wall_head..game.wall_tail)
        .find(|&index| game.wall[index] == wanted.index())
        .expect("wanted tail tile is still in the wall");
    game.wall.swap(tail, position);
}

#[test]
fn kong_discard_pays_one_more_than_the_same_plain_discard() {
    let mut plain = kong_discard_setup();
    let plain_before = plain.scores;
    submit_specific(&mut plain, seat(0), Action::Discard(w(9)));
    submit_specific(&mut plain, seat(1), Action::Hu);
    let plain_info = plain.win_info[1].expect("the waiter won the plain discard");
    assert_eq!(plain_info.method, WinMethod::Discard);
    assert_eq!(plain_info.payer, Some(seat(0)));
    let plain_paid = plain_before[0] - plain.scores[0];
    assert_eq!(plain_paid, i32::from(plain_info.fan));
    assert_valid(&plain);

    let mut kong = kong_discard_setup();
    put_tail_tile(&mut kong, t(9));
    submit_specific(&mut kong, seat(0), Action::AddedKong(w(5)));
    assert_eq!(
        kong.scores,
        [3, -1, -1, -1],
        "the three active opponents pay the added kong"
    );
    assert!(matches!(
        kong.stage,
        Stage::Action {
            seat: S0,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert_eq!(kong.last_draw_tile, Some(t(9)));

    let kong_before = kong.scores;
    submit_specific(&mut kong, seat(0), Action::Discard(w(9)));
    submit_specific(&mut kong, seat(1), Action::Hu);
    let kong_info = kong.win_info[1].expect("the waiter won the kong discard");
    assert_eq!(kong_info.method, WinMethod::KongDiscard);
    assert_eq!(kong_info.payer, Some(seat(0)));
    assert_eq!(kong_info.winning_tile, Some(w(9)));
    let kong_paid = kong_before[0] - kong.scores[0];
    assert_eq!(kong_paid, i32::from(kong_info.fan));
    assert_eq!(
        kong_info.fan,
        plain_info.fan + 1,
        "kong-discard adds exactly the single event fan"
    );
    assert_eq!(kong_paid, plain_paid + 1);
    assert!(
        kong_info.fan < rules::FAN_CAP,
        "the event bonus must not be clipped by the fan cap"
    );
    let discard = kong
        .discards
        .last()
        .expect("the post-kong discard is recorded");
    assert_eq!(discard.tile, w(9));
    assert_eq!(discard.seat, seat(0));
    assert!(!discard.robbed_kong);
    assert!(discard.winners[1]);
    assert_valid(&kong);
}

/// Four 1W plus a full Tong run and the 2W: a concealed kong leaves a
/// complete Tong run and the 2W to discard after the replacement draw.
fn concealed_kong_declarer_hand() -> Hand {
    let mut tiles = vec![w(1), w(1), w(1), w(1)];
    tiles.extend(tong_run());
    tiles.push(w(2));
    hand(&tiles)
}

#[test]
fn concealed_kong_pays_active_opponents_and_draws_from_the_tail() {
    let mut game = blank_game();
    game.hands[0] = concealed_kong_declarer_hand();
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    let head_before = game.wall_head;
    let tail_before = game.wall_tail;
    let wall_before = game.wall_remaining();

    submit_specific(&mut game, seat(0), Action::ConcealedKong(w(1)));

    assert_eq!(game.scores, [6, -2, -2, -2]);
    assert_eq!(game.kong_ledger.len(), 3);
    assert_eq!(game.melds[0][0].kind(), MeldKind::ConcealedKong);
    assert_eq!(game.hands[0].total(), 11);
    assert_eq!(
        game.wall_head, head_before,
        "replacement is not a normal draw"
    );
    assert_eq!(game.wall_tail, tail_before - 1);
    assert_eq!(game.wall_remaining(), wall_before - 1);
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S0,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert_valid(&game);
}

#[test]
fn inactive_winners_do_not_pay_kong_charges() {
    let mut game = blank_game();
    game.winners[1] = true;
    game.win_info[1] = Some(WinInfo {
        method: WinMethod::Discard,
        fan: 1,
        winning_tile: Some(w(5)),
        payer: Some(seat(0)),
    });
    game.discards.push(DiscardInfo::new(seat(0), w(5)));
    game.discards[0].winners[1] = true;
    game.hands[0] = concealed_kong_declarer_hand();
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    submit_specific(&mut game, seat(0), Action::ConcealedKong(w(1)));

    assert_eq!(game.scores, [4, 0, -2, -2]);
    assert_eq!(game.kong_ledger.len(), 2);
    assert!(game
        .kong_ledger
        .iter()
        .all(|record| record.payer != seat(1)));
    assert_valid(&game);
}

#[test]
fn open_kong_pays_the_discarder_and_draws_one_replacement() {
    let mut game = blank_game();
    game.hands[0] = five_wan_source_hand();
    let mut claimant = vec![w(5), w(5), w(5)];
    claimant.extend(tong_run());
    claimant.push(w(6));
    game.hands[1] = hand(&claimant);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    let tail_before = game.wall_tail;
    let wall_before = game.wall_remaining();

    submit_specific(&mut game, seat(0), Action::Discard(w(5)));
    submit_specific(&mut game, seat(1), Action::OpenKong);

    assert_eq!(game.scores, [-2, 2, 0, 0]);
    assert_eq!(game.kong_ledger.len(), 1);
    assert_eq!(game.melds[1][0].kind(), MeldKind::OpenKong);
    assert_eq!(game.hands[1].total(), 11);
    assert_eq!(game.wall_tail, tail_before - 1);
    assert_eq!(game.wall_remaining(), wall_before - 1);
    assert_eq!(game.discards.last().unwrap().meld_claim, Some(seat(1)));
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterDraw(DrawKind::Replacement)
        }
    ));
    assert_valid(&game);
}

/// Seat 0 discards 5W, seat 1 pengs it and then discards the 9W that seat 2
/// waits on. A discard after a peng is not a kong replacement.
fn peng_then_discard_setup() -> Game {
    let mut game = blank_game();
    game.hands[0] = hand(&[w(5)]);
    game.hands[1] = hand(&[
        w(5),
        w(5),
        w(9),
        w(1),
        w(2),
        w(3),
        w(4),
        w(6),
        w(7),
        w(8),
        t(1),
        t(2),
        t(3),
    ]);
    game.hands[2] = nine_wan_waiter_hand();
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

#[test]
fn a_discard_after_a_peng_is_a_plain_discard_win() {
    let mut game = peng_then_discard_setup();
    let wall_before = game.wall_remaining();
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));
    submit_specific(&mut game, seat(1), Action::Peng);
    assert_eq!(game.melds[1][0].kind(), MeldKind::Peng);
    assert_eq!(game.wall_remaining(), wall_before, "peng does not draw");
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: S1,
            kind: ActionKind::AfterPeng
        }
    ));

    submit_specific(&mut game, seat(1), Action::Discard(w(9)));
    submit_specific(&mut game, seat(2), Action::Hu);

    let info = game.win_info[2].expect("seat 2 won on the post-peng discard");
    assert_eq!(
        info.method,
        WinMethod::Discard,
        "a post-peng discard is not a kong discard"
    );
    assert_eq!(info.payer, Some(seat(1)));
    assert_eq!(game.scores, [0, -1, 1, 0]);
    let discard = game.discards.last().unwrap();
    assert_eq!(discard.tile, w(9));
    assert_eq!(discard.seat, seat(1));
    assert!(!discard.robbed_kong);
    assert!(discard.winners[2]);
    assert_valid(&game);
}

#[test]
fn empty_wall_blocks_kongs_but_not_peng() {
    let mut game = blank_game();
    game.hands[0] = hand(&[
        w(1),
        w(1),
        w(1),
        w(1),
        w(2),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(7),
        t(8),
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    drain_wall_to_discards(&mut game);
    assert_eq!(game.wall_remaining(), 0);
    assert_valid(&game);

    let actions = pending_actions(&mut game);
    assert!(actions
        .iter()
        .all(|action| !matches!(action, Action::ConcealedKong(_) | Action::AddedKong(_))));

    let mut response = blank_game();
    response.hands[0] = five_wan_source_hand();
    response.hands[1] = hand(&[
        w(5),
        w(5),
        w(1),
        w(2),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(7),
        t(8),
        t(9),
    ]);
    response.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut response);
    drain_wall_to_discards(&mut response);
    submit_specific(&mut response, seat(0), Action::Discard(w(5)));
    let response_actions = pending_actions(&mut response);
    assert!(response_actions.contains(&Action::Peng));
    assert!(!response_actions.contains(&Action::OpenKong));
    submit_specific(&mut response, seat(1), Action::Peng);
    assert_eq!(response.melds[1][0].kind(), MeldKind::Peng);
    assert_valid(&response);
}
