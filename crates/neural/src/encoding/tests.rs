//! Unit tests for the v1 encoding internals.

use super::schema::{
    ACTION_KIND_FEATURES, ACTION_SUIT_FEATURES, CONTEXT_FEATURES, DISCARDS_FEATURES,
    MELD_SLOT_FEATURES, PLAYERS_FEATURES, RESPONSE_FEATURES, TILE_FEATURES,
};
use super::*;
use sichuan_mahjong_sim::{
    Decision, DecisionStage, DiscardInfo, ExchangeDirection, Hand, LegalActions, Meld, Observation,
    PlayerView, ResponseView, Seat, Suit, Tile, PLAYER_COUNT,
};

const NO_MELDS: &[Meld] = &[];
const NO_DISCARDS: &[DiscardInfo] = &[];

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("valid test tile")
}

fn seat(index: usize) -> Seat {
    Seat::new(index).expect("valid test seat")
}

fn view<'a>(
    seat: Seat,
    concealed_count: usize,
    melds: &'a [Meld],
    missing_suit: Option<Suit>,
    score: i32,
    has_won: bool,
) -> PlayerView<'a> {
    PlayerView {
        seat,
        concealed_count,
        melds,
        missing_suit,
        score,
        has_won,
    }
}

fn quiet_views<'a>() -> [PlayerView<'a>; PLAYER_COUNT] {
    std::array::from_fn(|index| view(seat(index), 13 - index, NO_MELDS, None, 0, false))
}

/// An ordinary turn decision with no melds, discards, or response context.
fn turn_decision<'a>(decider: Seat, hand: &'a Hand, actions: &'a [Action]) -> Decision<'a> {
    Decision {
        seat: decider,
        observation: Observation {
            own_hand: hand,
            own_melds: NO_MELDS,
            players: quiet_views(),
            discards: NO_DISCARDS,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 40,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(actions),
    }
}

#[test]
fn frozen_v1_schema() {
    assert_eq!(ENCODING_VERSION, 1);
    assert_eq!(STATE_WIDTH, 1013);
    assert_eq!(ACTION_WIDTH, 39);
    assert_eq!(MELD_SLOTS, 4);
    assert_eq!(PLAYERS_FEATURES, 620);
    assert_eq!(MELD_SLOT_FEATURES, 37);
    assert_eq!(DISCARDS_FEATURES, 324);
    assert_eq!(CONTEXT_FEATURES, 9);
    assert_eq!(RESPONSE_FEATURES, 33);
    assert_eq!(
        TILE_FEATURES + PLAYERS_FEATURES + DISCARDS_FEATURES + CONTEXT_FEATURES + RESPONSE_FEATURES,
        STATE_WIDTH
    );
    assert_eq!(
        ACTION_KIND_FEATURES + TILE_FEATURES + ACTION_SUIT_FEATURES,
        ACTION_WIDTH
    );

    let hand = Hand::from_tiles(&[tile(Suit::Wan, 1)]).expect("hand");
    let decision = turn_decision(seat(0), &hand, &[]);
    let encoded = encode(&decision);
    assert_eq!(encoded.state.flatten().len(), STATE_WIDTH);
    assert_eq!(encoded.state.own_hand.len(), TILE_FEATURES);
    assert_eq!(
        ActionFeatures::new(Action::Pass).flatten().len(),
        ACTION_WIDTH
    );
}

#[test]
fn canonicalize_sorts_exchange_tiles_only() {
    let exchange = Action::Exchange([tile(Suit::Tiao, 3), tile(Suit::Wan, 9), tile(Suit::Tong, 1)]);
    let canonical = canonicalize(exchange);
    assert_eq!(
        canonical,
        Action::Exchange([tile(Suit::Wan, 9), tile(Suit::Tong, 1), tile(Suit::Tiao, 3)])
    );
    assert_eq!(canonicalize(canonical), canonical);

    let others = [
        Action::ChooseMissing(Suit::Tiao),
        Action::Discard(tile(Suit::Tong, 4)),
        Action::Peng,
        Action::OpenKong,
        Action::ConcealedKong(tile(Suit::Tiao, 9)),
        Action::AddedKong(tile(Suit::Wan, 2)),
        Action::Hu,
        Action::Pass,
    ];
    for action in others {
        assert_eq!(canonicalize(action), action);
    }
}

#[test]
fn action_features_use_kind_and_parameter_labels() {
    let cases: [(Action, usize); 9] = [
        (
            Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 1), tile(Suit::Tong, 2)]),
            0,
        ),
        (Action::ChooseMissing(Suit::Tiao), 1),
        (Action::Discard(tile(Suit::Wan, 5)), 2),
        (Action::Peng, 3),
        (Action::OpenKong, 4),
        (Action::ConcealedKong(tile(Suit::Tong, 7)), 5),
        (Action::AddedKong(tile(Suit::Tiao, 9)), 6),
        (Action::Hu, 7),
        (Action::Pass, 8),
    ];
    for (action, kind_id) in cases {
        let features = ActionFeatures::new(action);
        for (index, value) in features.kind.iter().enumerate() {
            assert_eq!(*value, if index == kind_id { 1.0 } else { 0.0 });
        }
        assert_eq!(features.flatten().len(), ACTION_WIDTH);
    }

    let exchange = ActionFeatures::new(Action::Exchange([
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Tong, 2),
    ]));
    assert_eq!(exchange.tiles[tile(Suit::Wan, 1).index() as usize], 0.5);
    assert_eq!(exchange.tiles[tile(Suit::Tong, 2).index() as usize], 0.25);
    assert_eq!(exchange.tiles.iter().sum::<f32>(), 0.75);
    assert_eq!(exchange.suit, [0.0; 3]);

    let missing = ActionFeatures::new(Action::ChooseMissing(Suit::Tong));
    assert_eq!(missing.suit, [0.0, 1.0, 0.0]);
    assert!(missing.tiles.iter().all(|&value| value == 0.0));

    for action in [
        Action::Discard(tile(Suit::Wan, 5)),
        Action::ConcealedKong(tile(Suit::Wan, 5)),
        Action::AddedKong(tile(Suit::Wan, 5)),
    ] {
        let features = ActionFeatures::new(action);
        assert_eq!(features.tiles[tile(Suit::Wan, 5).index() as usize], 0.25);
        assert_eq!(features.tiles.iter().sum::<f32>(), 0.25);
    }

    for action in [Action::Peng, Action::OpenKong, Action::Hu, Action::Pass] {
        let features = ActionFeatures::new(action);
        assert!(features.tiles.iter().all(|&value| value == 0.0));
        assert!(features.suit.iter().all(|&value| value == 0.0));
    }
}

#[test]
fn state_relativizes_seats_and_classifies_discards() {
    let decider = seat(1);
    let own_hand = Hand::from_tiles(&[tile(Suit::Wan, 1), tile(Suit::Tong, 5)]).expect("hand");
    let own_melds = [Meld::peng(tile(Suit::Wan, 7), seat(0))];
    let opponent_melds = [Meld::concealed_kong(tile(Suit::Tiao, 9))];
    let players = [
        view(seat(0), 13, &[], Some(Suit::Tong), -3, false),
        view(decider, 2, &own_melds, Some(Suit::Wan), 8, false),
        view(seat(2), 9, &opponent_melds, None, 0, false),
        view(seat(3), 10, &[], Some(Suit::Tiao), 1, true),
    ];

    let mut hu_winners = [false; PLAYER_COUNT];
    hu_winners[0] = true;
    let discards = [
        DiscardInfo {
            seat: seat(0),
            tile: tile(Suit::Wan, 1),
            meld_claim: None,
            winners: [false; PLAYER_COUNT],
            robbed_kong: false,
        },
        DiscardInfo {
            seat: decider,
            tile: tile(Suit::Wan, 2),
            meld_claim: Some(seat(2)),
            winners: [false; PLAYER_COUNT],
            robbed_kong: false,
        },
        DiscardInfo {
            seat: seat(2),
            tile: tile(Suit::Wan, 3),
            meld_claim: None,
            winners: hu_winners,
            robbed_kong: false,
        },
        DiscardInfo {
            seat: seat(3),
            tile: tile(Suit::Wan, 4),
            meld_claim: None,
            winners: hu_winners,
            robbed_kong: true,
        },
    ];

    let observation = Observation {
        own_hand: &own_hand,
        own_melds: &own_melds,
        players,
        discards: &discards,
        exchange_direction: ExchangeDirection::CounterClockwise,
        wall_count: 27,
        stage: DecisionStage::DiscardResponse,
        response: Some(ResponseView {
            source: seat(3),
            tile: tile(Suit::Tong, 5),
            robbing_added_kong: false,
        }),
    };
    let offered = [Action::Pass];
    let decision = Decision {
        seat: decider,
        observation,
        legal_actions: LegalActions::Actions(&offered),
    };
    let encoded = encode(&decision);
    assert_eq!(encoded, encode(&decision));

    let state = &encoded.state;
    assert_eq!(state.own_hand[tile(Suit::Wan, 1).index() as usize], 0.25);
    assert_eq!(state.own_hand[tile(Suit::Tong, 5).index() as usize], 0.25);

    // Relative seat 0 is the deciding player itself.
    let own = &state.players[0];
    assert_eq!(own.concealed_total, 2.0 / 14.0);
    assert_eq!(own.missing_suit, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(own.score, 0.5);
    assert_eq!(own.has_won, 0.0);
    assert_eq!(own.melds[0].valid, 1.0);
    assert_eq!(own.melds[0].tile[tile(Suit::Wan, 7).index() as usize], 1.0);
    assert_eq!(own.melds[0].kind, [1.0, 0.0, 0.0, 0.0]);
    // The peng source is seat 0, three clockwise steps from seat 1.
    assert_eq!(own.melds[0].source, [0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(own.melds[1..].iter().all(|slot| slot.valid == 0.0));
    assert!(own.melds[1..].iter().all(|slot| slot
        .tile
        .iter()
        .chain(&slot.kind)
        .chain(&slot.source)
        .all(|&value| value == 0.0)));

    // Relative seat 1 is the next clockwise seat and has no declaration.
    let next = &state.players[1];
    assert_eq!(next.missing_suit, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(next.concealed_total, 9.0 / 14.0);
    assert_eq!(next.melds[0].kind, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(next.melds[0].source, [0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        next.melds[0].tile[tile(Suit::Tiao, 9).index() as usize],
        1.0
    );

    let previous = &state.players[3];
    assert_eq!(previous.missing_suit, [0.0, 1.0, 0.0, 0.0]);
    let won = &state.players[2];
    assert_eq!(won.has_won, 1.0);
    assert_eq!(won.score, 1.0 / 16.0);
    assert_eq!(won.missing_suit, [0.0, 0.0, 1.0, 0.0]);

    // Discards are classified once, by relative source.
    assert_eq!(
        state.discards[3][0][tile(Suit::Wan, 1).index() as usize],
        0.25
    );
    assert_eq!(
        state.discards[0][1][tile(Suit::Wan, 2).index() as usize],
        0.25
    );
    assert_eq!(
        state.discards[1][2][tile(Suit::Wan, 3).index() as usize],
        0.25
    );
    assert_eq!(
        state.discards[2][2][tile(Suit::Wan, 4).index() as usize],
        0.25
    );
    assert_eq!(
        state.discards[0][2][tile(Suit::Wan, 2).index() as usize],
        0.0
    );
    assert_eq!(
        state.discards[2][1][tile(Suit::Wan, 4).index() as usize],
        0.0
    );

    assert_eq!(state.context.wall_count, 27.0 / 108.0);
    assert_eq!(state.context.exchange_direction, [0.0, 1.0, 0.0]);
    assert_eq!(state.context.stage, [0.0, 0.0, 0.0, 1.0, 0.0]);

    assert_eq!(state.response.present, 1.0);
    assert_eq!(
        state.response.tile[tile(Suit::Tong, 5).index() as usize],
        1.0
    );
    // The response source is seat 3, two clockwise steps from seat 1.
    assert_eq!(state.response.relative_source, [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(state.response.robbing_added_kong, 0.0);
}

#[test]
fn absent_response_and_unused_melds_are_zero() {
    let hand = Hand::from_tiles(&[tile(Suit::Tiao, 2)]).expect("hand");
    let offered = [Action::Discard(tile(Suit::Tiao, 2))];
    let mut decision = turn_decision(seat(2), &hand, &offered);
    decision.observation.exchange_direction = ExchangeDirection::Opposite;
    decision.observation.wall_count = 0;
    let encoded = encode(&decision);
    assert_eq!(encoded.state.context.exchange_direction, [0.0, 0.0, 1.0]);
    assert_eq!(encoded.state.context.stage, [0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(encoded.state.context.wall_count, 0.0);

    let response = &encoded.state.response;
    assert_eq!(response.present, 0.0);
    assert_eq!(response.robbing_added_kong, 0.0);
    assert!(response.tile.iter().all(|&value| value == 0.0));
    assert!(response.relative_source.iter().all(|&value| value == 0.0));

    for player in &encoded.state.players {
        assert_eq!(player.melds[0].valid, 0.0);
        assert!(player.melds.iter().all(|slot| slot.valid == 0.0));
        assert!(player.melds.iter().all(|slot| {
            slot.tile
                .iter()
                .chain(&slot.kind)
                .chain(&slot.source)
                .all(|&value| value == 0.0)
        }));
        assert_eq!(player.missing_suit, [0.0, 0.0, 0.0, 1.0]);
    }
}

#[test]
fn ordinary_candidates_are_canonical_unique_and_sorted() {
    let hand = Hand::from_tiles(&[tile(Suit::Wan, 1)]).expect("hand");
    let offered = [
        Action::Pass,
        Action::Discard(tile(Suit::Wan, 3)),
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Hu,
        Action::Discard(tile(Suit::Wan, 2)),
        Action::Discard(tile(Suit::Wan, 1)),
    ];
    let reversed: Vec<Action> = offered.iter().rev().copied().collect();

    let encoded = encode(&turn_decision(seat(0), &hand, &offered));
    let reordered = encode(&turn_decision(seat(0), &hand, &reversed));
    assert_eq!(encoded, reordered);

    let actions: Vec<Action> = encoded
        .candidates
        .iter()
        .map(|candidate| candidate.action)
        .collect();
    assert_eq!(
        actions,
        vec![
            Action::Discard(tile(Suit::Wan, 1)),
            Action::Discard(tile(Suit::Wan, 2)),
            Action::Discard(tile(Suit::Wan, 3)),
            Action::Hu,
            Action::Pass,
        ]
    );
    assert_eq!(
        encoded.target_index(&Action::Discard(tile(Suit::Wan, 2))),
        Ok(1)
    );

    let absent = Action::ConcealedKong(tile(Suit::Wan, 1));
    let error = encoded.target_index(&absent).unwrap_err();
    assert_eq!(error.action, absent);
    assert!(error.to_string().contains("ConcealedKong"));
}

#[test]
fn exchange_candidates_are_independent_multisets() {
    let hand = Hand::from_tiles(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 4),
        tile(Suit::Tiao, 5),
        tile(Suit::Tiao, 6),
    ])
    .expect("hand");
    let mut decision = turn_decision(seat(3), &hand, &[]);
    decision.observation.exchange_direction = ExchangeDirection::Opposite;
    decision.observation.wall_count = 55;
    decision.observation.stage = DecisionStage::Exchange;
    decision.legal_actions = LegalActions::Exchange {
        eligible_suits: [true, false, false],
    };
    let encoded = encode(&decision);

    // Only Wan (relative counts 2, 1, 1) is eligible: the three held
    // multisets are 1W-1W-2W, 1W-1W-3W, and 1W-2W-3W.
    assert_eq!(
        encoded
            .candidates
            .iter()
            .map(|candidate| candidate.action)
            .collect::<Vec<_>>(),
        vec![
            Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 1), tile(Suit::Wan, 2)]),
            Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 1), tile(Suit::Wan, 3)]),
            Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)]),
        ]
    );
    for candidate in &encoded.candidates {
        let Action::Exchange(tiles) = candidate.action else {
            panic!("exchange candidates only");
        };
        assert!(tiles.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(hand.contains_tiles(&tiles));
    }

    let unordered = Action::Exchange([tile(Suit::Wan, 2), tile(Suit::Wan, 3), tile(Suit::Wan, 1)]);
    assert_eq!(encoded.target_index(&unordered), Ok(2));
    assert_eq!(
        encoded.target_index(&Action::Exchange([
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 2)
        ])),
        Err(TargetNotCandidate {
            action: Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 2)])
        })
    );
}
