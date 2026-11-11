//! Opening exchange and missing-suit declaration scenarios.

use super::super::*;
use super::fixtures::*;

#[test]
fn exchange_is_collected_atomically_and_follows_the_public_direction() {
    let mut game = Game::new(7);
    let before: [Hand; PLAYER_COUNT] = game.hands;
    let direction = game.exchange_direction;
    let mut sent = [[w(1); 3]; PLAYER_COUNT];

    for index in 0..PLAYER_COUNT {
        let decision = game.next_decision().unwrap();
        assert_eq!(
            decision.observation.exchange_direction, direction,
            "direction is public before any choice"
        );
        let action = match &decision.legal_actions {
            LegalActions::Exchange { eligible_suits } => {
                exchange_action(decision.observation.own_hand, *eligible_suits)
            }
            LegalActions::Actions(_) => panic!("expected exchange"),
        };
        let Action::Exchange(tiles) = action else {
            unreachable!()
        };
        sent[decision.seat.index()] = tiles;

        let snapshot = game.hands;
        game.submit(seat(index), action).unwrap();
        if index < PLAYER_COUNT - 1 {
            assert_eq!(
                game.hands, snapshot,
                "a partial exchange must not move tiles"
            );
            assert!(matches!(game.stage, Stage::Exchange));
        }
    }

    assert!(matches!(game.stage, Stage::Missing));
    for seat in Seat::ALL {
        let mut expected = before[seat.index()];
        expected
            .remove_tiles(&sent[seat.index()])
            .expect("sent tiles were legal");
        let donor = direction.donor(seat);
        expected
            .add_tiles(&sent[donor.index()])
            .expect("received tiles were passed legally");
        assert_eq!(game.hands[seat.index()], expected);
    }
    assert_valid(&game);
}

#[test]
fn missing_suit_choices_are_private_until_all_four_are_collected() {
    let mut game = Game::new(9);
    complete_exchange(&mut game);

    for index in 0..PLAYER_COUNT {
        let decision = game.next_decision().unwrap();
        assert_eq!(decision.seat, seat(index));
        assert!(
            decision
                .observation
                .players
                .iter()
                .all(|player| player.missing_suit.is_none()),
            "pending missing choices must not be public"
        );
        let action = Action::ChooseMissing(Suit::Tong);
        game.submit(seat(index), action).unwrap();
        if index < PLAYER_COUNT - 1 {
            assert!(
                game.missing[index].is_none(),
                "a submitted missing choice is not public before collection"
            );
        }
    }

    assert_eq!(game.missing, [Some(Suit::Tong); PLAYER_COUNT]);
    assert!(matches!(
        game.stage,
        Stage::Action {
            seat: Seat::DEALER,
            kind: ActionKind::Opening
        }
    ));
    assert_valid(&game);
}

#[test]
fn missing_suit_tiles_force_discards_and_block_claims() {
    let mut game = blank_game();
    game.hands[0] = hand(&[b(1), b(2), b(3), w(1), w(2), w(3)]);
    game.missing[0] = Some(Suit::Tiao);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    let actions = pending_actions(&mut game);
    assert!(!actions.is_empty());
    for action in actions {
        let Action::Discard(tile) = action else {
            panic!("missing-suit hand must only discard: {action:?}");
        };
        assert_eq!(tile.suit(), Suit::Tiao);
    }

    game.hands[1] = hand(&[
        b(1),
        w(1),
        w(1),
        w(3),
        w(4),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(7),
        t(8),
    ]);
    game.missing[1] = Some(Suit::Tiao);
    refresh_wall(&mut game);
    assert!(
        !game.can_hu_on_tile(seat(1), w(5)),
        "a missing-suit tile blocks Hu even with a winning shape"
    );
    assert_valid(&game);
}
