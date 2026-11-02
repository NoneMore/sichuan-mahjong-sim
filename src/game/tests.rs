//! Unit and scenario tests for the authoritative game state machine.
//!
//! The module is a child of `game`, so it can construct small synthetic states
//! with exact hands, melds, walls, and ledgers. Every scenario ends with
//! `debug_validate`, which checks physical tile conservation and zero-sum
//! scores across the documented ownership representation.

use super::*;

const S0: Seat = Seat::DEALER;
const S1: Seat = match Seat::new(1) {
    Some(seat) => seat,
    None => panic!("seat 1"),
};
const S3: Seat = match Seat::new(3) {
    Some(seat) => seat,
    None => panic!("seat 3"),
};

fn seat(index: usize) -> Seat {
    Seat::new(index).unwrap()
}

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).unwrap()
}

fn w(rank: u8) -> Tile {
    tile(Suit::Wan, rank)
}

fn t(rank: u8) -> Tile {
    tile(Suit::Tong, rank)
}

fn b(rank: u8) -> Tile {
    tile(Suit::Tiao, rank)
}

fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).unwrap()
}

fn tong_run() -> [Tile; 9] {
    [t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9)]
}

/// Builds a synthetic game with no hands, melds, discards, or scores. Callers
/// set the state they need and then call `refresh_wall`.
fn blank_game() -> Game {
    let mut game = Game::new(0);
    game.hands = [Hand::new(); PLAYER_COUNT];
    game.melds = std::array::from_fn(|_| Vec::new());
    game.discards.clear();
    game.missing = [Some(Suit::Tiao); PLAYER_COUNT];
    game.scores = [0; PLAYER_COUNT];
    game.winners = [false; PLAYER_COUNT];
    game.win_info = [None; PLAYER_COUNT];
    game.exchange_choices = [None; PLAYER_COUNT];
    game.missing_choices = [None; PLAYER_COUNT];
    game.pass_lock = [false; PLAYER_COUNT];
    game.kong_ledger.clear();
    game.last_draw_tile = None;
    game.stage = Stage::Action {
        seat: Seat::DEALER,
        kind: ActionKind::Opening,
    };
    game.pending = None;
    game.legal_actions.clear();
    game.result = None;
    game
}

/// Reconstructs the active wall from all remaining physical tiles in hands,
/// melds, and unincorporated discards. This keeps synthetic states conserving.
fn refresh_wall(game: &mut Game) {
    let mut counts = [COPIES_PER_TILE; TILE_KINDS];
    for seat in Seat::ALL {
        for tile in game.hands[seat.index()].tiles() {
            counts[tile.index() as usize] = counts[tile.index() as usize]
                .checked_sub(1)
                .expect("synthetic hand overuses a tile");
        }
        for meld in &game.melds[seat.index()] {
            let index = meld.tile().index() as usize;
            counts[index] = counts[index]
                .checked_sub(meld.tile_count())
                .expect("synthetic meld overuses a tile");
        }
    }
    for discard in &game.discards {
        if discard.meld_claim.is_none() {
            let index = discard.tile.index() as usize;
            counts[index] = counts[index]
                .checked_sub(1)
                .expect("synthetic discard overuses a tile");
        }
    }

    let mut cursor = 0;
    for tile in Tile::ALL {
        for _ in 0..counts[tile.index() as usize] {
            game.wall[cursor] = tile.index();
            cursor += 1;
        }
    }
    game.wall_head = 0;
    game.wall_tail = cursor;
}

/// Moves all active wall tiles into the public discard history and leaves an
/// empty wall. Used to reach wall-exhaustion states without losing tiles.
fn drain_wall_to_discards(game: &mut Game) {
    let tiles: Vec<Tile> = game.wall[game.wall_head..game.wall_tail]
        .iter()
        .map(|&raw| Tile::from_index(raw).unwrap())
        .collect();
    for tile in tiles {
        game.discards.push(DiscardInfo::new(Seat::DEALER, tile));
    }
    game.wall_head = game.wall_tail;
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

fn exchange_action(hand: &Hand, eligible: [bool; 3]) -> Action {
    let suit = Suit::ALL
        .into_iter()
        .find(|suit| eligible[suit.index()])
        .expect("a legal exchange suit exists");
    let tiles: Vec<Tile> = hand
        .tiles()
        .filter(|tile| tile.suit() == suit)
        .take(3)
        .collect();
    Action::Exchange([tiles[0], tiles[1], tiles[2]])
}

/// Chooses Hu when legal, otherwise the first legal action.
fn decide(game: &mut Game) -> Option<(Seat, Action)> {
    let decision = game.next_decision()?;
    let seat = decision.seat;
    let action = match &decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            exchange_action(decision.observation.own_hand, *eligible_suits)
        }
        LegalActions::Actions(actions) => actions
            .iter()
            .find(|action| matches!(action, Action::Hu))
            .copied()
            .unwrap_or_else(|| actions[0]),
    };
    Some((seat, action))
}

fn submit_specific(game: &mut Game, expected_seat: Seat, action: Action) {
    let decision = game.next_decision().expect("expected a pending decision");
    assert_eq!(decision.seat, expected_seat);
    match &decision.legal_actions {
        LegalActions::Actions(actions) => {
            assert!(
                actions.contains(&action),
                "expected {action:?} in {actions:?}"
            );
        }
        LegalActions::Exchange { .. } => panic!("expected an action list"),
    }
    game.submit(expected_seat, action).expect("action is legal");
}

fn complete_exchange(game: &mut Game) {
    for _ in 0..PLAYER_COUNT {
        let (seat, action) = decide(game).expect("exchange decision");
        game.submit(seat, action).expect("exchange is legal");
    }
    assert!(matches!(game.stage, Stage::Missing));
}

fn assert_valid(game: &Game) {
    game.debug_validate().expect("game invariants");
}

/// Reads (and thereby re-reads) the pending action list as an owned vector.
fn pending_actions(game: &mut Game) -> Vec<Action> {
    match game
        .next_decision()
        .expect("pending decision")
        .legal_actions
    {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected an action list"),
    }
}

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

fn three_waiter_setup() -> Game {
    // All three hands wait on 5W: 345W plus three Tong sets, with a distinct
    // pair in each hand. The source holds the fourth 5W.
    let mut game = blank_game();
    let mut a = vec![w(1), w(1), w(3), w(4)];
    a.extend(tong_run());
    let mut b = vec![w(2), w(2), w(3), w(4)];
    b.extend(tong_run());
    let mut c = vec![w(6), w(6), w(3), w(4)];
    c.extend(tong_run());
    game.hands[0] = hand(&[w(5)]);
    game.hands[1] = hand(&a);
    game.hands[2] = hand(&b);
    game.hands[3] = hand(&c);
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
    let next_before = {
        let decision = game.next_decision().unwrap();
        assert_eq!(decision.seat, seat(2));
        match decision.legal_actions {
            LegalActions::Actions(actions) => actions.to_vec(),
            LegalActions::Exchange { .. } => panic!("expected response"),
        }
    };
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
        let decision = probe.next_decision().unwrap();
        match decision.legal_actions {
            LegalActions::Actions(actions) => actions.to_vec(),
            LegalActions::Exchange { .. } => panic!("expected response"),
        }
    };
    assert_eq!(next_before, after_first_hu);
    assert_valid(&game);
}

fn added_kong_setup() -> Game {
    let mut game = blank_game();
    game.melds[0] = vec![Meld::peng(w(5), seat(1))];
    let declarer = vec![
        w(5),
        w(1),
        w(2),
        w(3),
        w(4),
        w(6),
        w(7),
        w(8),
        w(9),
        t(1),
        t(2),
    ];
    game.hands[0] = hand(&declarer);
    let mut waiter = vec![w(1), w(1), w(3), w(4)];
    waiter.extend(tong_run());
    game.hands[1] = hand(&waiter);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

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
    game.hands[0] = hand(&[
        w(5),
        w(1),
        w(2),
        w(3),
        w(4),
        w(6),
        w(7),
        w(8),
        w(9),
        t(1),
        t(2),
    ]);
    game.hands[1] = hand(&[
        w(7),
        w(8),
        w(1),
        w(1),
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
    game
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

#[test]
fn concealed_kong_pays_active_opponents_and_draws_from_the_tail() {
    let mut game = blank_game();
    let mut declarer = vec![w(1), w(1), w(1), w(1)];
    declarer.extend(tong_run());
    declarer.push(w(2));
    game.hands[0] = hand(&declarer);
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
    let mut declarer = vec![w(1), w(1), w(1), w(1)];
    declarer.extend(tong_run());
    declarer.push(w(2));
    game.hands[0] = hand(&declarer);
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
    let mut source = vec![w(5)];
    source.extend(tong_run());
    source.extend([w(1), w(2), w(3), w(4)]);
    game.hands[0] = hand(&source);
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

fn peng_and_hu_setup() -> Game {
    let mut game = blank_game();
    let mut source = vec![w(5)];
    source.extend(tong_run());
    source.extend([w(1), w(2), w(3), w(4)]);
    game.hands[0] = hand(&source);
    game.hands[1] = hand(&[
        w(5),
        w(5),
        w(1),
        w(1),
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
    game
}

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
    game.hands[2] = hand(&[
        w(7),
        w(8),
        w(1),
        w(1),
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

fn single_waiter_setup() -> Game {
    let mut game = blank_game();
    let mut source = vec![w(5)];
    source.extend(tong_run());
    source.extend([w(1), w(2), w(3), w(4)]);
    game.hands[0] = hand(&source);
    let mut waiter = vec![w(1), w(1), w(3), w(4)];
    waiter.extend(tong_run());
    game.hands[1] = hand(&waiter);
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
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

    let actions = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected turn actions"),
    };
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

    let actions = match game.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected turn actions"),
    };
    assert!(actions
        .iter()
        .all(|action| !matches!(action, Action::ConcealedKong(_) | Action::AddedKong(_))));

    let mut response = blank_game();
    let mut source = vec![w(5)];
    source.extend(tong_run());
    source.extend([w(1), w(2), w(3), w(4)]);
    response.hands[0] = hand(&source);
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
    let response_actions = match response.next_decision().unwrap().legal_actions {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected response actions"),
    };
    assert!(response_actions.contains(&Action::Peng));
    assert!(!response_actions.contains(&Action::OpenKong));
    submit_specific(&mut response, seat(1), Action::Peng);
    assert_eq!(response.melds[1][0].kind(), MeldKind::Peng);
    assert_valid(&response);
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
