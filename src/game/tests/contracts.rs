//! Decision atomicity, buffer reuse, cloning, and invariant diagnostics.

use super::super::*;
use super::fixtures::*;

#[test]
fn illegal_submissions_leave_the_entire_state_unchanged() {
    let mut game = Game::new(5);
    let decision = game.next_decision().unwrap();
    let pending_seat = decision.seat;
    let before = format!("{game:?}");

    assert!(game
        .submit(pending_seat.next_clockwise(), Action::Pass)
        .is_err());
    assert_eq!(before, format!("{game:?}"));

    assert!(game.submit(pending_seat, Action::Pass).is_err());
    assert_eq!(before, format!("{game:?}"));

    let valid = {
        let decision = game.next_decision().unwrap();
        match &decision.legal_actions {
            LegalActions::Exchange { eligible_suits } => {
                exchange_action(decision.observation.own_hand, *eligible_suits)
            }
            LegalActions::Actions(_) => panic!("expected exchange"),
        }
    };
    game.submit(pending_seat, valid).unwrap();
    assert_valid(&game);
}

#[test]
fn seeded_games_terminate_zero_sum_with_all_tiles_conserved() {
    for seed in 0..12 {
        let mut game = Game::new(seed);
        assert_valid(&game);
        let mut steps = 0u32;
        while let Some((seat, action)) = decide(&mut game) {
            game.submit(seat, action).unwrap();
            assert_valid(&game);
            steps += 1;
            assert!(steps < 3_000, "seed {seed} did not terminate");
        }
        let result = *game.result().expect("terminal result");
        assert_eq!(result.scores.iter().sum::<i32>(), 0, "seed {seed}");
        assert!(
            result.winners.iter().any(|&won| won) || result.end_reason == EndReason::WallExhausted
        );
    }
}

#[test]
fn clone_during_partial_exchange_replays_identically() {
    let mut original = Game::new(0xC10E);
    for _ in 0..2 {
        let (seat, action) = decide(&mut original).unwrap();
        original.submit(seat, action).unwrap();
    }
    assert!(matches!(original.stage, Stage::Exchange));
    let mut fork = original.clone();

    for _ in 0..2 {
        let (seat, action) = decide(&mut original).unwrap();
        original.submit(seat, action).unwrap();
        let (fork_seat, fork_action) = decide(&mut fork).unwrap();
        assert_eq!((seat, action), (fork_seat, fork_action));
        fork.submit(fork_seat, fork_action).unwrap();
    }

    assert_eq!(format!("{original:?}"), format!("{fork:?}"));
    assert!(matches!(original.stage, Stage::Missing));
    assert_valid(&original);
}

/// The follow-up of a two-stage response keeps the seat pending for its lower
/// claims only: the stale Hu action is rejected without changing any state.
#[test]
fn a_follow_up_response_rejects_the_stale_hu_action() {
    let mut game = peng_and_hu_setup();
    submit_specific(&mut game, S0, Action::Discard(w(5)));
    assert_eq!(
        pending_actions(&mut game),
        [Action::Hu, Action::Peng, Action::Pass]
    );
    submit_specific(&mut game, seat(1), Action::Pass);

    let follow_up = pending_actions(&mut game);
    assert_eq!(follow_up, [Action::Peng, Action::Pass]);
    assert_eq!(game.pending.expect("follow-up").seat, seat(1));

    let before = format!("{game:?}");
    let error = game.submit(seat(1), Action::Hu).expect_err("Hu is stale");
    assert_eq!(
        error,
        GameError::IllegalAction {
            seat: seat(1),
            action: Action::Hu,
        }
    );
    assert_eq!(before, format!("{game:?}"), "a rejected Hu changes nothing");
    assert!(game.pass_lock[1]);
    assert_eq!(game.discards.len(), 1);
    assert_eq!(game.discards[0].meld_claim, None);
    assert!(game.winners.iter().all(|&won| !won));

    // Re-reading the pending decision is idempotent and keeps the same options.
    assert_eq!(pending_actions(&mut game), follow_up);

    submit_specific(&mut game, seat(1), Action::Peng);
    assert_eq!(game.melds[1][0].kind(), MeldKind::Peng);
    assert_eq!(game.discards[0].meld_claim, Some(seat(1)));
    assert_valid(&game);
}

/// Two seats pending in the same discard response have different option sets:
/// seat 1 may only peng, seat 2 may only Hu. A stale action from the previous
/// seat must be rejected without panicking, leaking into the next seat's list,
/// or changing the collected state.
#[test]
fn response_options_do_not_leak_between_seats() {
    let mut game = blank_game();
    game.hands[0] = hand(&[
        w(5),
        w(7),
        w(8),
        w(1),
        w(2),
        w(3),
        t(1),
        t(2),
        t(3),
        t(4),
        t(5),
        t(6),
        t(8),
        t(9),
    ]);
    game.hands[1] = hand(&[
        w(5),
        w(5),
        w(6),
        w(7),
        w(8),
        w(9),
        w(2),
        t(5),
        t(6),
        t(7),
        t(8),
        t(9),
        t(9),
    ]);
    game.hands[2] = hand(&[
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
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    submit_specific(&mut game, S0, Action::Discard(w(5)));
    assert_eq!(
        pending_actions(&mut game),
        [Action::Peng, Action::Pass],
        "seat 1 can only peng the discard"
    );
    submit_specific(&mut game, seat(1), Action::Peng);
    assert_eq!(
        game.discards[0].meld_claim, None,
        "claims are applied only after collection"
    );

    assert_eq!(
        pending_actions(&mut game),
        [Action::Hu, Action::Pass],
        "seat 2 can only win on the discard"
    );
    assert_eq!(game.pending.expect("seat 2 response").seat, seat(2));

    let before = format!("{game:?}");
    let error = game
        .submit(seat(2), Action::Peng)
        .expect_err("seat 2 cannot peng");
    assert_eq!(
        error,
        GameError::IllegalAction {
            seat: seat(2),
            action: Action::Peng,
        }
    );
    assert_eq!(before, format!("{game:?}"), "a stale claim changes nothing");
    assert_eq!(pending_actions(&mut game), [Action::Hu, Action::Pass]);

    submit_specific(&mut game, seat(2), Action::Hu);
    assert!(game.winners[2]);
    assert_eq!(game.win_info[2].expect("winner metadata").payer, Some(S0));
    assert_eq!(
        game.melds[1].len(),
        0,
        "the earlier peng choice is not applied"
    );
    assert_eq!(
        game.hands[1].total(),
        14,
        "only the next normal draw touched seat 1"
    );
    let discard = game.discards.last().expect("discard record");
    assert_eq!(discard.winners, [false, false, true, false]);
    assert_eq!(discard.meld_claim, None);
    assert_valid(&game);
}

/// The pending action list keeps the documented pipeline order and reuses one
/// buffer: a Pass that removes the Hu slot repopulates the same allocation.
/// Pointer assertions intentionally bind this regression to allocation reuse.
#[test]
fn legal_action_buffer_is_reused_in_pipeline_order() {
    let mut game = blank_game();
    game.melds[0] = vec![Meld::peng(w(9), seat(1))];
    // 111W + 123W + 789W + 99T with the fourth 9W in hand for an added kong and
    // all four 1W for a concealed kong.
    game.hands[0] = hand(&[
        w(1),
        w(1),
        w(1),
        w(1),
        w(2),
        w(3),
        w(7),
        w(8),
        w(9),
        t(9),
        t(9),
    ]);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    let expected = vec![
        Action::Hu,
        Action::ConcealedKong(w(1)),
        Action::AddedKong(w(9)),
        Action::Discard(w(1)),
        Action::Discard(w(2)),
        Action::Discard(w(3)),
        Action::Discard(w(7)),
        Action::Discard(w(8)),
        Action::Discard(w(9)),
        Action::Discard(t(9)),
        Action::Pass,
    ];
    assert_eq!(pending_actions(&mut game), expected);

    let capacity = game.legal_actions.capacity();
    let buffer = game.legal_actions.as_ptr();
    assert!(capacity >= expected.len());

    submit_specific(&mut game, S0, Action::Pass);
    assert!(game.pass_lock[0]);
    let expected_reduced = expected[1..expected.len() - 1].to_vec();
    assert_eq!(pending_actions(&mut game), expected_reduced);
    assert_eq!(
        game.legal_actions.capacity(),
        capacity,
        "the decision buffer is reused across decisions"
    );
    assert_eq!(
        game.legal_actions.as_ptr(),
        buffer,
        "repopulating the list must not reallocate"
    );

    // A response collection round also runs through the same buffer.
    submit_specific(&mut game, S0, Action::Discard(w(1)));
    assert_eq!(game.pending.expect("seat 1 turn").seat, seat(1));
    assert_eq!(game.legal_actions.capacity(), capacity);
    assert_valid(&game);
}

#[test]
fn missing_suit_actions_keep_suit_order_in_the_shared_buffer() {
    let mut game = Game::new(7);
    let initial_capacity = game.legal_actions.capacity();
    complete_exchange(&mut game);
    assert!(matches!(game.stage, Stage::Missing));
    assert_eq!(game.missing, [None; PLAYER_COUNT]);
    assert_eq!(game.legal_actions.capacity(), initial_capacity);

    let expected = vec![
        Action::ChooseMissing(Suit::Wan),
        Action::ChooseMissing(Suit::Tong),
        Action::ChooseMissing(Suit::Tiao),
    ];
    assert_eq!(pending_actions(&mut game), expected);
    let capacity = game.legal_actions.capacity();
    let buffer = game.legal_actions.as_ptr();

    submit_specific(&mut game, S0, Action::ChooseMissing(Suit::Tong));
    assert_eq!(game.missing_choices[0], Some(Suit::Tong));
    assert_eq!(game.pending.expect("seat 1 missing choice").seat, seat(1));
    assert_eq!(pending_actions(&mut game), expected);
    assert_eq!(game.legal_actions.capacity(), capacity);
    assert_eq!(game.legal_actions.as_ptr(), buffer);
    assert_valid(&game);
}

/// Tampering tests below mutate metadata only (never tiles or scores), so each
/// rejected state differs from the valid one solely in winner/discard linkage.
#[test]
fn debug_validate_cross_checks_discard_winners_against_win_info() {
    let mut game = single_waiter_setup();
    submit_specific(&mut game, seat(0), Action::Discard(w(5)));
    submit_specific(&mut game, seat(1), Action::Hu);
    assert_valid(&game);

    let mut wrong_tile = game.clone();
    wrong_tile.win_info[1]
        .as_mut()
        .expect("winner metadata")
        .winning_tile = Some(t(1));
    let error = wrong_tile.debug_validate().unwrap_err();
    assert!(
        error.contains("records a different winning tile"),
        "unexpected error: {error}"
    );

    let mut wrong_payer = game.clone();
    wrong_payer.win_info[1]
        .as_mut()
        .expect("winner metadata")
        .payer = Some(seat(2));
    let error = wrong_payer.debug_validate().unwrap_err();
    assert!(
        error.contains("paid by the wrong discarder"),
        "unexpected error: {error}"
    );

    let mut unmarked = game.clone();
    unmarked.discards[0].winners[1] = false;
    let error = unmarked.debug_validate().unwrap_err();
    assert!(
        error.contains("instead of one"),
        "unexpected error: {error}"
    );

    let mut extra_winner = game.clone();
    extra_winner.discards[0].winners[2] = true;
    let error = extra_winner.debug_validate().unwrap_err();
    assert!(
        error.contains("without winner metadata"),
        "unexpected error: {error}"
    );
}

#[test]
fn debug_validate_rejects_a_self_draw_winner_marked_on_a_discard() {
    let mut game = blank_game();
    let mut winner = vec![w(1), w(2), w(3), w(7), w(7)];
    winner.extend(tong_run());
    game.hands[0] = hand(&winner);
    // A historic, unclaimed discard from an earlier turn of the same hand.
    game.discards.push(DiscardInfo::new(seat(2), b(9)));
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);

    submit_specific(&mut game, seat(0), Action::Hu);
    assert_eq!(
        game.win_info[0].expect("self-draw winner").method,
        WinMethod::SelfDraw
    );
    assert!(game
        .discards
        .iter()
        .all(|discard| !discard.winners.iter().any(|&won| won)));
    assert_valid(&game);

    let mut tampered = game.clone();
    tampered.discards[0].winners[0] = true;
    let error = tampered.debug_validate().unwrap_err();
    assert!(error.contains("self-draw"), "unexpected error: {error}");
}

#[test]
fn debug_validate_rejects_robbing_kong_metadata_mismatches() {
    let mut game = added_kong_setup();
    submit_specific(&mut game, seat(0), Action::AddedKong(w(5)));
    submit_specific(&mut game, seat(1), Action::Hu);
    assert_valid(&game);
    let robbed = game.discards.len() - 1;
    assert!(game.discards[robbed].robbed_kong);

    let mut wrong_method = game.clone();
    wrong_method.win_info[1]
        .as_mut()
        .expect("winner metadata")
        .method = WinMethod::Discard;
    let error = wrong_method.debug_validate().unwrap_err();
    assert!(
        error.contains("robbed-kong mismatch"),
        "unexpected error: {error}"
    );

    let mut unrobbed = game.clone();
    unrobbed.discards[robbed].robbed_kong = false;
    let error = unrobbed.debug_validate().unwrap_err();
    assert!(
        error.contains("robbed-kong mismatch"),
        "unexpected error: {error}"
    );

    let mut no_payer = game.clone();
    no_payer.win_info[1]
        .as_mut()
        .expect("winner metadata")
        .payer = None;
    let error = no_payer.debug_validate().unwrap_err();
    assert!(
        error.contains("records no payer"),
        "unexpected error: {error}"
    );

    let mut self_draw = game.clone();
    self_draw.win_info[1]
        .as_mut()
        .expect("winner metadata")
        .method = WinMethod::SelfDraw;
    let error = self_draw.debug_validate().unwrap_err();
    assert!(error.contains("self-draw"), "unexpected error: {error}");
}
