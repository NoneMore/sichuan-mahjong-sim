//! Multi-winner arbitration and wall-exhaustion settlement scenarios.

use super::super::*;
use super::fixtures::*;

const S3: Seat = match Seat::new(3) {
    Some(seat) => seat,
    None => panic!("seat 3"),
};

fn three_waiter_setup() -> Game {
    // All three hands wait on 5W: 345W plus three Tong sets, with a distinct
    // pair in each hand. The source holds the fourth 5W.
    let mut game = blank_game();
    game.hands[0] = hand(&[w(5)]);
    game.hands[1] = five_wan_waiter_hand(w(1));
    game.hands[2] = five_wan_waiter_hand(w(2));
    game.hands[3] = five_wan_waiter_hand(w(6));
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

fn discard_5w_to_three_waiters(game: &mut Game) {
    submit_specific(game, seat(0), Action::Discard(w(5)));
}

#[test]
fn one_discard_can_pay_three_winners() {
    let mut game = three_waiter_setup();
    discard_5w_to_three_waiters(&mut game);
    assert!(matches!(game.stage, Stage::Response(_)));

    for expected in 1..=3 {
        submit_specific(&mut game, seat(expected), Action::Hu);
        if expected < 3 {
            assert_eq!(
                game.scores,
                [0, 0, 0, 0],
                "Hu is not paid before arbitration"
            );
            assert!(!game.winners[1] && !game.winners[2] && !game.winners[3]);
        }
    }

    let result = *game.result().expect("three winners end the hand");
    assert_eq!(result.end_reason, EndReason::ThreeWinners);
    assert_eq!(result.winners, [false, true, true, true]);
    assert_eq!(result.scores, [-3, 1, 1, 1]);

    let discard = game.discards.last().unwrap();
    assert_eq!(discard.tile, w(5));
    assert_eq!(discard.winners, [false, true, true, true]);
    assert_eq!(discard.meld_claim, None);
    assert!(!discard.robbed_kong);
    assert_valid(&game);

    let terminal = format!("{game:?}");
    assert!(game.next_decision().is_none());
    assert!(game.next_decision().is_none());
    assert_eq!(terminal, format!("{game:?}"));
}

#[test]
fn response_collection_is_isolated_and_clone_preserves_the_future() {
    let mut game = three_waiter_setup();
    discard_5w_to_three_waiters(&mut game);

    submit_specific(&mut game, seat(1), Action::Hu);
    assert_eq!(game.scores, [0, 0, 0, 0]);
    assert_eq!(game.next_decision().expect("seat 2 response").seat, seat(2));
    let next_before = pending_actions(&mut game);
    let mut fork = game.clone();

    submit_specific(&mut game, seat(2), Action::Hu);
    submit_specific(&mut game, seat(3), Action::Hu);
    submit_specific(&mut fork, seat(2), Action::Hu);
    submit_specific(&mut fork, seat(3), Action::Hu);

    assert_eq!(format!("{game:?}"), format!("{fork:?}"));
    assert_eq!(game.result(), fork.result());
    assert_eq!(game.result().unwrap().scores, [-3, 1, 1, 1]);
    let after_first_hu = {
        let mut probe = three_waiter_setup();
        discard_5w_to_three_waiters(&mut probe);
        submit_specific(&mut probe, seat(1), Action::Hu);
        pending_actions(&mut probe)
    };
    assert_eq!(next_before, after_first_hu);
    assert_valid(&game);
}

#[test]
fn wall_exhaustion_applies_flower_pig_big_call_and_per_payment_refunds() {
    let mut game = blank_game();
    game.missing = [
        Some(Suit::Tiao),
        Some(Suit::Tiao),
        Some(Suit::Tong),
        Some(Suit::Tiao),
    ];
    game.winners[0] = true;
    game.win_info[0] = Some(WinInfo {
        method: WinMethod::Discard,
        fan: 1,
        winning_tile: Some(t(1)),
        payer: Some(seat(1)),
    });
    game.discards.push(DiscardInfo::new(seat(1), t(1)));
    game.discards[0].winners[0] = true;
    game.hands[1] = hand(&[b(1)]);
    game.hands[2] = hand(&[
        w(1),
        w(1),
        w(2),
        w(2),
        w(3),
        w(3),
        w(4),
        w(5),
        w(6),
        w(7),
        w(8),
        w(9),
        w(9),
    ]);
    game.hands[3] = hand(&[t(5)]);
    game.scores = [0, 3, -2, -1];
    game.kong_ledger = vec![
        KongRecord {
            declarer: seat(1),
            payer: seat(0),
            kind: MeldKind::ConcealedKong,
            points: 2,
        },
        KongRecord {
            declarer: seat(1),
            payer: seat(2),
            kind: MeldKind::ConcealedKong,
            points: 2,
        },
        KongRecord {
            declarer: seat(1),
            payer: seat(3),
            kind: MeldKind::ConcealedKong,
            points: 2,
        },
        KongRecord {
            declarer: seat(3),
            payer: seat(1),
            kind: MeldKind::AddedKong,
            points: 1,
        },
        KongRecord {
            declarer: seat(0),
            payer: seat(1),
            kind: MeldKind::ConcealedKong,
            points: 2,
        },
    ];
    game.stage = Stage::Action {
        seat: S3,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    drain_wall_to_discards(&mut game);
    assert_valid(&game);

    submit_specific(&mut game, seat(3), Action::Discard(t(5)));

    let result = *game.result().expect("empty wall ends the hand");
    assert_eq!(result.end_reason, EndReason::WallExhausted);
    assert_eq!(result.winners, [true, false, false, false]);
    assert_eq!(result.scores, [18, -50, 19, 13]);
    assert_eq!(result.scores.iter().sum::<i32>(), 0);
    assert_valid(&game);

    let terminal = format!("{game:?}");
    assert!(game.next_decision().is_none());
    assert_eq!(terminal, format!("{game:?}"));
}
