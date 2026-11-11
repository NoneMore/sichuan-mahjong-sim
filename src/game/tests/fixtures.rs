//! Shared synthetic-state builders for the game scenario tests.
//!
//! Scenario modules construct exact hands, melds, walls, and ledgers through
//! these helpers so that every synthetic state stays tile-conserving. Only
//! builders used by more than one scenario module belong here; a setup used by
//! a single module stays in that module's file.

use super::super::*;

pub(super) const S0: Seat = Seat::DEALER;

pub(super) const S1: Seat = match Seat::new(1) {
    Some(seat) => seat,
    None => panic!("seat 1"),
};

pub(super) fn seat(index: usize) -> Seat {
    Seat::new(index).unwrap()
}

pub(super) fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).unwrap()
}

pub(super) fn w(rank: u8) -> Tile {
    tile(Suit::Wan, rank)
}

pub(super) fn t(rank: u8) -> Tile {
    tile(Suit::Tong, rank)
}

pub(super) fn b(rank: u8) -> Tile {
    tile(Suit::Tiao, rank)
}

pub(super) fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).unwrap()
}

pub(super) fn tong_run() -> [Tile; 9] {
    [t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9)]
}

/// Builds a synthetic game with no hands, melds, discards, or scores. Callers
/// set the state they need and then call `refresh_wall`.
pub(super) fn blank_game() -> Game {
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
pub(super) fn refresh_wall(game: &mut Game) {
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
pub(super) fn drain_wall_to_discards(game: &mut Game) {
    let tiles: Vec<Tile> = game.wall[game.wall_head..game.wall_tail]
        .iter()
        .map(|&raw| Tile::from_index(raw).unwrap())
        .collect();
    for tile in tiles {
        game.discards.push(DiscardInfo::new(Seat::DEALER, tile));
    }
    game.wall_head = game.wall_tail;
}

pub(super) fn exchange_action(hand: &Hand, eligible: [bool; 3]) -> Action {
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
pub(super) fn decide(game: &mut Game) -> Option<(Seat, Action)> {
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

pub(super) fn submit_specific(game: &mut Game, expected_seat: Seat, action: Action) {
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

pub(super) fn complete_exchange(game: &mut Game) {
    for _ in 0..PLAYER_COUNT {
        let (seat, action) = decide(game).expect("exchange decision");
        game.submit(seat, action).expect("exchange is legal");
    }
    assert!(matches!(game.stage, Stage::Missing));
}

pub(super) fn assert_valid(game: &Game) {
    game.debug_validate().expect("game invariants");
}

/// Reads (and thereby re-reads) the pending action list as an owned vector.
pub(super) fn pending_actions(game: &mut Game) -> Vec<Action> {
    match game
        .next_decision()
        .expect("pending decision")
        .legal_actions
    {
        LegalActions::Actions(actions) => actions.to_vec(),
        LegalActions::Exchange { .. } => panic!("expected an action list"),
    }
}

/// Fourteen-tile source hand for the 5W scenarios: the fourth 5W, a full Tong
/// run, and 1234W. The other copies of 5W are available to claimants.
pub(super) fn five_wan_source_hand() -> Hand {
    let mut tiles = vec![w(5)];
    tiles.extend(tong_run());
    tiles.extend([w(1), w(2), w(3), w(4)]);
    hand(&tiles)
}

/// Thirteen-tile hand waiting on 5W: 345W plus three Tong sets, with `pair`.
pub(super) fn five_wan_waiter_hand(pair: Tile) -> Hand {
    let mut tiles = vec![pair, pair, w(3), w(4)];
    tiles.extend(tong_run());
    hand(&tiles)
}

/// Thirteen-tile hand waiting on 5W through either a Peng or a Hu: 55W plus
/// the 1W pair and a full Tong run. The fourth 5W completes 555W for Hu.
fn peng_or_hu_waiter_hand() -> Hand {
    let mut tiles = vec![w(5), w(5), w(1), w(1)];
    tiles.extend(tong_run());
    hand(&tiles)
}

/// Eleven-tile declarer hand behind a Peng of 5W: the fourth 5W, 1234W/6789W,
/// and the 1T/2T tail. The 9W is the planned post-kong discard.
pub(super) fn added_kong_declarer_hand() -> Hand {
    hand(&[
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
    ])
}

/// Thirteen-tile hand waiting on 9W: 789W plus the 1W pair and a Tong run.
pub(super) fn nine_wan_waiter_hand() -> Hand {
    let mut tiles = vec![w(7), w(8), w(1), w(1)];
    tiles.extend(tong_run());
    hand(&tiles)
}

pub(super) fn added_kong_setup() -> Game {
    let mut game = blank_game();
    game.melds[0] = vec![Meld::peng(w(5), seat(1))];
    game.hands[0] = added_kong_declarer_hand();
    game.hands[1] = five_wan_waiter_hand(w(1));
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

pub(super) fn peng_and_hu_setup() -> Game {
    let mut game = blank_game();
    game.hands[0] = five_wan_source_hand();
    game.hands[1] = peng_or_hu_waiter_hand();
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}

pub(super) fn single_waiter_setup() -> Game {
    let mut game = blank_game();
    game.hands[0] = five_wan_source_hand();
    game.hands[1] = five_wan_waiter_hand(w(1));
    game.stage = Stage::Action {
        seat: S0,
        kind: ActionKind::AfterDraw(DrawKind::Normal),
    };
    refresh_wall(&mut game);
    game
}
