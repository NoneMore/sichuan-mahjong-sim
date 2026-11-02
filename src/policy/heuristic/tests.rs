//! Decision-level checks for [`HeuristicPolicy`].
//!
//! Synthetic decisions exercise the policy boundary directly: only the
//! supplied observation and legal action list are visible. Counted-ownership
//! checks verify that meld-claimed discards, own melds, and multi-winner
//! discards are not double counted.

use super::*;
use crate::action::{Action, ExchangeDirection, Meld};
use crate::game::{
    Decision, DecisionStage, DiscardInfo, LegalActions, Observation, PlayerView, ResponseView,
};
use crate::hand::Hand;
use crate::seat::{Seat, PLAYER_COUNT};
use crate::tile::{Suit, Tile};

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).expect("legal hand")
}

fn suit_hand(wan: &[u8], tong: &[u8], tiao: &[u8]) -> Hand {
    let mut tiles = Vec::new();
    for (suit, ranks) in [(Suit::Wan, wan), (Suit::Tong, tong), (Suit::Tiao, tiao)] {
        tiles.extend(ranks.iter().map(|&rank| tile(suit, rank)));
    }
    hand(&tiles)
}

fn view<'a>(
    seat: Seat,
    own_hand: &Hand,
    own_melds: &'a [Meld],
    missing: Option<Suit>,
) -> [PlayerView<'a>; PLAYER_COUNT] {
    std::array::from_fn(|index| {
        let view_seat = Seat::new(index).expect("seat index");
        PlayerView {
            seat: view_seat,
            concealed_count: if view_seat == seat {
                own_hand.total()
            } else {
                13
            },
            melds: if view_seat == seat { own_melds } else { &[] },
            missing_suit: if view_seat == seat {
                missing
            } else {
                Some(Suit::Tiao)
            },
            score: 0,
            has_won: false,
        }
    })
}

fn decision<'a>(
    hand: &'a Hand,
    melds: &'a [Meld],
    missing: Option<Suit>,
    stage: DecisionStage,
    response: Option<ResponseView>,
    actions: &'a [Action],
    discards: &'a [DiscardInfo],
) -> Decision<'a> {
    let seat = Seat::DEALER;
    Decision {
        seat,
        observation: Observation {
            own_hand: hand,
            own_melds: melds,
            players: view(seat, hand, melds, missing),
            discards,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage,
            response,
        },
        legal_actions: LegalActions::Actions(actions),
    }
}

fn discard_info(
    seat: Seat,
    tile: Tile,
    meld_claim: Option<Seat>,
    winners: [bool; PLAYER_COUNT],
) -> DiscardInfo {
    DiscardInfo {
        seat,
        tile,
        meld_claim,
        winners,
        robbed_kong: false,
    }
}

fn public_of(discards: &[DiscardInfo], melds: &[Meld]) -> [u8; TILE_KINDS] {
    let players = view(Seat::DEALER, &Hand::new(), melds, Some(Suit::Tiao));
    let observation = Observation {
        own_hand: &Hand::new(),
        own_melds: melds,
        players,
        discards,
        exchange_direction: ExchangeDirection::Clockwise,
        wall_count: 20,
        stage: DecisionStage::Turn,
        response: None,
    };
    base_public(&observation)
}

fn discard_progress(
    hand: &Hand,
    tile: Tile,
    melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
) -> Progress {
    let mut after = *hand;
    after.remove(tile).expect("candidate tile is held");
    let mut adjusted = *public;
    adjusted[tile.index() as usize] = adjusted[tile.index() as usize].saturating_add(1);
    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid melds");
    let deficit = evaluator.deficit(&after);
    Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &after, &adjusted, deficit),
    }
}

#[test]
fn public_ownership_counts_each_physical_tile_once() {
    let source = Seat::new(1).expect("seat");
    let own_melds = [Meld::peng(tile(Suit::Wan, 5), source)];
    let other_melds = [Meld::open_kong(
        tile(Suit::Tong, 3),
        Seat::new(2).expect("seat"),
    )];

    let own_hand = Hand::new();
    let mut players = view(Seat::DEALER, &own_hand, &own_melds, Some(Suit::Tiao));
    players[2].melds = &other_melds;

    let discards = [
        // Physically part of the deciding player's peng; the record is only a
        // public association and must not be counted again.
        discard_info(source, tile(Suit::Wan, 5), Some(Seat::DEALER), [false; 4]),
        // One physical tile referenced by two Hu winners; count it once.
        discard_info(
            source,
            tile(Suit::Tong, 1),
            None,
            [false, false, true, true],
        ),
        // An ordinary unclaimed discard.
        discard_info(
            Seat::new(2).expect("seat"),
            tile(Suit::Tiao, 2),
            None,
            [false; 4],
        ),
    ];

    let observation = Observation {
        own_hand: &own_hand,
        own_melds: &own_melds,
        players,
        discards: &discards,
        exchange_direction: ExchangeDirection::Clockwise,
        wall_count: 20,
        stage: DecisionStage::Turn,
        response: None,
    };
    let public = base_public(&observation);

    assert_eq!(
        public[tile(Suit::Wan, 5).index() as usize],
        3,
        "the claimed discard is the peng's own physical tile, counted once"
    );
    assert_eq!(public[tile(Suit::Tong, 3).index() as usize], 4);
    assert_eq!(
        public[tile(Suit::Tong, 1).index() as usize],
        1,
        "a multi-winner discard is one physical tile"
    );
    assert_eq!(public[tile(Suit::Tiao, 2).index() as usize], 1);
}

#[test]
fn exchange_matches_the_baseline_and_missing_suit_uses_fewest_tiles() {
    let exchange_hand = suit_hand(&[1, 2, 3], &[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]);
    let mut policy = HeuristicPolicy::new();
    let exchange = Decision {
        seat: Seat::DEALER,
        observation: Observation {
            own_hand: &exchange_hand,
            own_melds: &[],
            players: view(Seat::DEALER, &exchange_hand, &[], None),
            discards: &[],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Exchange,
            response: None,
        },
        legal_actions: LegalActions::Exchange {
            eligible_suits: [true, true, true],
        },
    };
    assert_eq!(
        policy.choose(&exchange),
        Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)])
    );

    // Fewest-tile suit: Tong (3 tiles) beats Wan (4) and Tiao (6).
    let actions = [
        Action::ChooseMissing(Suit::Wan),
        Action::ChooseMissing(Suit::Tong),
        Action::ChooseMissing(Suit::Tiao),
    ];
    let missing_hand = suit_hand(&[1, 2, 3, 4], &[1, 2, 3], &[1, 2, 3, 4, 5, 6]);
    let missing = decision(
        &missing_hand,
        &[],
        None,
        DecisionStage::MissingSuit,
        None,
        &actions,
        &[],
    );
    assert_eq!(policy.choose(&missing), Action::ChooseMissing(Suit::Tong));

    // Tie on count: Wan wins over Tong because of fixed suit order.
    let tie_hand = suit_hand(&[1, 2], &[1, 2], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tie = decision(
        &tie_hand,
        &[],
        None,
        DecisionStage::MissingSuit,
        None,
        &actions,
        &[],
    );
    assert_eq!(policy.choose(&tie), Action::ChooseMissing(Suit::Wan));
}

#[test]
fn hu_and_sole_actions_short_circuit_analysis() {
    let mut policy = HeuristicPolicy::new();
    let impossible = suit_hand(&[1, 4, 7, 2, 9, 9], &[1, 4, 7, 2], &[1, 4, 7, 2]);

    let hu_actions = [Action::Hu, Action::Discard(tile(Suit::Wan, 2))];
    let hu = decision(
        &impossible,
        &[],
        Some(Suit::Tiao),
        DecisionStage::Turn,
        None,
        &hu_actions,
        &[],
    );
    assert_eq!(policy.choose(&hu), Action::Hu);

    let sole_actions = [Action::Discard(tile(Suit::Wan, 7))];
    let sole = decision(
        &impossible,
        &[],
        Some(Suit::Tiao),
        DecisionStage::Turn,
        None,
        &sole_actions,
        &[],
    );
    assert_eq!(
        policy.choose(&sole),
        Action::Discard(tile(Suit::Wan, 7)),
        "a sole legal action bypasses hand analysis"
    );

    let response_actions = [Action::Hu, Action::Pass];
    let response = decision(
        &impossible,
        &[],
        Some(Suit::Tiao),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source: Seat::new(1).expect("seat"),
            tile: tile(Suit::Wan, 2),
            robbing_added_kong: false,
        }),
        &response_actions,
        &[],
    );
    assert_eq!(policy.choose(&response), Action::Hu);

    let pass_only = [Action::Pass];
    let added = decision(
        &impossible,
        &[],
        Some(Suit::Tiao),
        DecisionStage::AddedKongResponse,
        Some(ResponseView {
            source: Seat::new(1).expect("seat"),
            tile: tile(Suit::Wan, 2),
            robbing_added_kong: true,
        }),
        &pass_only,
        &[],
    );
    assert_eq!(policy.choose(&added), Action::Pass);
}

#[test]
fn forced_missing_suit_discard_uses_fixed_tile_order() {
    let hand = suit_hand(&[1, 2, 3, 4], &[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]);
    // Deliberately list the legal forced discards out of tile order; the
    // policy must use fixed tile order, not the supplied list order.
    let actions = [
        Action::Discard(tile(Suit::Tiao, 3)),
        Action::Discard(tile(Suit::Tiao, 1)),
        Action::Discard(tile(Suit::Tiao, 2)),
    ];
    let forced = decision(
        &hand,
        &[],
        Some(Suit::Tiao),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&forced),
        Action::Discard(tile(Suit::Tiao, 1))
    );
}

fn tenpai_plus_extra() -> Hand {
    hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 7),
        tile(Suit::Wan, 8),
        tile(Suit::Wan, 9),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 1),
    ])
}

fn seven_pairs_plus_extra() -> Hand {
    hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tong, 4),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
        tile(Suit::Tong, 6),
        tile(Suit::Wan, 7),
    ])
}

#[test]
fn lower_shanten_beats_higher_ukeire_even_when_listed_second() {
    let missing = Suit::Tiao;
    let hand = tenpai_plus_extra();
    let public = [0u8; TILE_KINDS];
    let more_ukeire = tile(Suit::Wan, 1);
    let lower_shanten = tile(Suit::Tong, 1);

    let high = discard_progress(&hand, more_ukeire, &[], missing, &public);
    let low = discard_progress(&hand, lower_shanten, &[], missing, &public);
    assert!(
        high.shanten > low.shanten,
        "test premise: the higher-ukeire discard must be worse in distance"
    );
    assert!(
        high.ukeire > low.ukeire,
        "test premise: the lower-shanten discard must have less ukeire"
    );

    let actions = [Action::Discard(more_ukeire), Action::Discard(lower_shanten)];
    let choice = decision(
        &hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::Discard(lower_shanten)
    );
}

#[test]
fn equal_shanten_prefers_higher_ukeire_even_when_listed_second() {
    let missing = Suit::Tiao;
    let hand = tenpai_plus_extra();
    let public = [0u8; TILE_KINDS];
    let lower_ukeire = tile(Suit::Wan, 9);
    let higher_ukeire = tile(Suit::Tong, 1);

    let low = discard_progress(&hand, lower_ukeire, &[], missing, &public);
    let high = discard_progress(&hand, higher_ukeire, &[], missing, &public);
    assert_eq!(low.shanten, high.shanten);
    assert!(high.ukeire > low.ukeire);

    let actions = [
        Action::Discard(lower_ukeire),
        Action::Discard(higher_ukeire),
    ];
    let choice = decision(
        &hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::Discard(higher_ukeire)
    );
}

#[test]
fn fan_breaks_only_tenpai_ties_on_distance_and_ukeire() {
    let missing = Suit::Tiao;
    let hand = seven_pairs_plus_extra();
    let source = Seat::new(1).expect("seat");

    // Exhaust the standard waits of the quad hand so both discards have the
    // same ukeire. One wait remains structural for fan purposes.
    let mut discards = Vec::new();
    for kind in [
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 5),
    ] {
        for _ in 0..2 {
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
    assert_eq!(low.ukeire, high.ukeire);

    let mut low_after = hand;
    low_after.remove(lower_fan).unwrap();
    let mut high_after = hand;
    high_after.remove(higher_fan).unwrap();
    let low_fan_value = analysis::max_discard_fan(&low_after, &[], missing);
    let high_fan_value = analysis::max_discard_fan(&high_after, &[], missing);
    assert!(high_fan_value > low_fan_value);

    // The higher-fan discard is listed second; fan must override fixed order.
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
        Action::Discard(higher_fan)
    );
}

#[test]
fn fan_cannot_compensate_for_lower_availability() {
    let missing = Suit::Tiao;
    let hand = seven_pairs_plus_extra();
    let source = Seat::new(1).expect("seat");

    // Fully exhaust every wait of the higher-fan discard, while the other
    // discard still has unseen copies. Both remain structurally tenpai.
    let mut discards = Vec::new();
    for kind in [
        tile(Suit::Wan, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
    ] {
        let copies = if kind == tile(Suit::Wan, 1) { 1 } else { 2 };
        for _ in 0..copies {
            discards.push(discard_info(source, kind, None, [false; 4]));
        }
    }
    let public = public_of(&discards, &[]);
    let higher_fan = tile(Suit::Wan, 7);
    let higher_ukeire = tile(Suit::Wan, 1);

    let fan_candidate = discard_progress(&hand, higher_fan, &[], missing, &public);
    let ukeire_candidate = discard_progress(&hand, higher_ukeire, &[], missing, &public);
    assert_eq!(fan_candidate.shanten, 0);
    assert_eq!(ukeire_candidate.shanten, 0);
    assert!(
        ukeire_candidate.ukeire > fan_candidate.ukeire,
        "test premise: the lower-fan candidate must have more unseen copies"
    );

    let mut fan_after = hand;
    fan_after.remove(higher_fan).unwrap();
    let mut ukeire_after = hand;
    ukeire_after.remove(higher_ukeire).unwrap();
    assert!(
        analysis::max_discard_fan(&fan_after, &[], missing)
            > analysis::max_discard_fan(&ukeire_after, &[], missing)
    );

    let actions = [Action::Discard(higher_fan), Action::Discard(higher_ukeire)];
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
        Action::Discard(higher_ukeire),
        "fan cannot compensate for worse availability"
    );
}

fn progress_of(hand: &Hand, melds: &[Meld], missing: Suit, public: &[u8; TILE_KINDS]) -> Progress {
    let mut evaluator = DistanceEvaluator::new(melds, missing).expect("valid melds");
    let deficit = evaluator.deficit(hand);
    Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, hand, public, deficit),
    }
}

#[path = "tests/calls.rs"]
mod calls;
