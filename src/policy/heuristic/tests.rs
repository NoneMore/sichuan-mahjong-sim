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

#[test]
fn peng_must_strictly_lower_shanten() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");

    // Pass is already tenpai. The peng leaves the same distance but more
    // unseen improving copies, which is explicitly insufficient.
    let tenpai = suit_hand(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9], &[], &[]);
    let claim = tile(Suit::Wan, 9);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);
    let pass = progress_of(&tenpai, &[], missing, &public);
    let peng = evaluate_peng(&tenpai, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert_eq!(peng.shanten, pass.shanten);
    assert!(
        peng.ukeire > pass.ukeire,
        "test premise: the peng raises availability"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &tenpai,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);

    // A peng that creates a set strictly lowers shanten and is accepted.
    let open = suit_hand(&[1, 1, 2, 3, 4, 5, 6, 9], &[7, 8, 9, 5, 5], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);
    let pass = progress_of(&open, &[], missing, &public);
    let peng = evaluate_peng(&open, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert!(
        peng.shanten < pass.shanten,
        "test premise: the peng must lower shanten"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &open,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Peng);
}

#[test]
fn peng_cannot_use_the_seven_pairs_route_after_a_meld() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let six_pairs = suit_hand(&[1, 1, 7], &[2, 2, 3, 3, 4, 4, 5, 5, 6, 6], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);

    let pass = progress_of(&six_pairs, &[], missing, &public);
    assert_eq!(pass.shanten, 0, "the hand is a seven-pairs tenpai");
    let peng = evaluate_peng(&six_pairs, &[], missing, &public, claim, source)
        .expect("peng then discard is legal");
    assert!(
        peng.shanten >= pass.shanten,
        "declaring a meld disables seven pairs, so it cannot strictly improve"
    );

    let actions = [Action::Peng, Action::Pass];
    let choice = decision(
        &six_pairs,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);
}

#[test]
fn concealed_kong_gate_rejects_equal_shanten_with_less_ukeire() {
    let missing = Suit::Tiao;
    let kong_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9], &[], &[]);
    let kong_tile = tile(Suit::Wan, 1);
    let public = [0u8; TILE_KINDS];

    let discard = discard_progress(&kong_hand, kong_tile, &[], missing, &public);
    let (kind, hand, melds, adjusted) =
        kong_position(Action::ConcealedKong(kong_tile), &kong_hand, &[], &public)
            .expect("concealed kong position");
    assert_eq!(kind, MeldKind::ConcealedKong);
    let mut evaluator = DistanceEvaluator::new(&melds, missing).unwrap();
    let deficit = evaluator.deficit(&hand);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted, deficit),
    };
    assert_eq!(kong.shanten, discard.shanten);
    assert!(
        kong.ukeire < discard.ukeire,
        "test premise: the kong loses availability"
    );

    let actions = [Action::ConcealedKong(kong_tile), Action::Discard(kong_tile)];
    let choice = decision(
        &kong_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::Discard(kong_tile)
    );
}

#[test]
fn eligible_kong_is_preferred_on_equal_progress() {
    let missing = Suit::Tiao;
    let kong_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 9], &[1, 2, 3, 7, 8, 9], &[]);
    let kong_tile = tile(Suit::Wan, 1);
    let discard_tile = tile(Suit::Wan, 4);
    let public = [0u8; TILE_KINDS];

    let discard = discard_progress(&kong_hand, discard_tile, &[], missing, &public);
    let (_, hand, melds, adjusted) =
        kong_position(Action::ConcealedKong(kong_tile), &kong_hand, &[], &public)
            .expect("concealed kong position");
    let mut evaluator = DistanceEvaluator::new(&melds, missing).unwrap();
    let deficit = evaluator.deficit(&hand);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted, deficit),
    };
    assert_eq!(
        kong, discard,
        "test premise: the kong has the same progress rank"
    );

    let actions = [
        Action::ConcealedKong(kong_tile),
        Action::Discard(discard_tile),
    ];
    let choice = decision(
        &kong_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::ConcealedKong(kong_tile)
    );
}

#[test]
fn open_kong_gate_rejects_equal_shanten_with_less_ukeire() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let hand = suit_hand(&[1, 1, 1, 2, 3, 4, 5, 6, 9, 9], &[1, 2, 3], &[]);
    let claim = tile(Suit::Wan, 1);
    let discards = [discard_info(source, claim, None, [false; 4])];
    let public = public_of(&discards, &[]);

    let pass = progress_of(&hand, &[], missing, &public);
    let kong = evaluate_open_kong(&hand, &[], missing, &public, claim, source)
        .expect("open kong position");
    assert_eq!(pass.shanten, 0);
    assert_eq!(kong.shanten, pass.shanten);
    assert!(
        kong.ukeire < pass.ukeire,
        "test premise: the open kong loses availability"
    );

    // Both claims are legal, but neither passes its gate against pass.
    let actions = [Action::Peng, Action::OpenKong, Action::Pass];
    let choice = decision(
        &hand,
        &[],
        Some(missing),
        DecisionStage::DiscardResponse,
        Some(ResponseView {
            source,
            tile: claim,
            robbing_added_kong: false,
        }),
        &actions,
        &discards,
    );
    assert_eq!(HeuristicPolicy::new().choose(&choice), Action::Pass);
}

#[test]
fn kong_gate_and_selection_rules_are_lexicographic() {
    let alternative = Progress {
        shanten: 1,
        ukeire: 10,
    };
    let lower_shanten = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 1)),
        progress: Progress {
            shanten: 0,
            ukeire: 1,
        },
        income: 2,
    };
    assert_eq!(
        select_kong(alternative, &[lower_shanten]),
        Some(lower_shanten.action),
        "a lower shanten may justify lower ukeire"
    );

    let equal_lower_ukeire = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 2)),
        progress: Progress {
            shanten: 1,
            ukeire: 9,
        },
        income: 100,
    };
    assert_eq!(
        select_kong(alternative, &[equal_lower_ukeire]),
        None,
        "equal shanten with lower ukeire fails regardless of income"
    );

    let equal_progress = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 3)),
        progress: alternative,
        income: 1,
    };
    assert_eq!(
        select_kong(alternative, &[equal_progress]),
        Some(equal_progress.action)
    );

    let better_progress = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 4)),
        progress: Progress {
            shanten: 1,
            ukeire: 11,
        },
        income: 1,
    };
    let richer = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 5)),
        progress: Progress {
            shanten: 1,
            ukeire: 10,
        },
        income: 100,
    };
    assert_eq!(
        select_kong(alternative, &[richer, better_progress]),
        Some(better_progress.action),
        "progress rank precedes nominal income"
    );

    let concealed = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 6)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 6,
    };
    let added = KongChoice {
        action: Action::AddedKong(tile(Suit::Wan, 7)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 3,
    };
    assert_eq!(
        select_kong(alternative, &[added, concealed]),
        Some(concealed.action),
        "equal progress prefers the larger nominal income"
    );

    let earlier = KongChoice {
        action: Action::ConcealedKong(tile(Suit::Wan, 8)),
        progress: Progress {
            shanten: 0,
            ukeire: 5,
        },
        income: 6,
    };
    assert_eq!(
        select_kong(alternative, &[earlier, concealed]),
        Some(earlier.action),
        "an exact tie keeps the earlier fixed-order action"
    );
}

#[test]
fn open_kong_wins_a_response_tie_with_peng() {
    let equal = Progress {
        shanten: 0,
        ukeire: 5,
    };
    assert_eq!(
        response_choice(true, Some(equal), true, Some(equal)),
        Action::OpenKong,
        "the kong wins an exact progress tie"
    );

    let better_peng = Progress {
        shanten: 0,
        ukeire: 6,
    };
    assert_eq!(
        response_choice(true, Some(better_peng), true, Some(equal)),
        Action::Peng
    );
    assert_eq!(
        response_choice(true, Some(equal), true, Some(better_peng)),
        Action::OpenKong
    );
    assert_eq!(
        response_choice(true, Some(equal), false, None),
        Action::Peng
    );
    assert_eq!(
        response_choice(false, None, true, Some(equal)),
        Action::OpenKong
    );
    assert_eq!(response_choice(false, None, false, None), Action::Pass);
}

#[test]
fn hypothetical_meld_positions_conserve_physical_ownership() {
    let physical = |hand: &Hand, public: &[u8; TILE_KINDS]| -> [u8; TILE_KINDS] {
        let counts = std::array::from_fn(|index| hand.counts()[index] + public[index]);
        assert!(counts.iter().all(|&count| count <= 4));
        counts
    };
    let source = Seat::new(1).expect("seat");
    let discards = [
        discard_info(source, tile(Suit::Wan, 8), None, [false; 4]),
        discard_info(source, tile(Suit::Tong, 2), None, [false; 4]),
    ];
    let public = public_of(&discards, &[]);

    // Concealed kong on a turn.
    let turn_hand = suit_hand(&[1, 1, 1, 1, 2, 3, 4, 9], &[1, 2, 3, 7, 8, 9], &[]);
    let (_, concealed, _, concealed_public) = kong_position(
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        &turn_hand,
        &[],
        &public,
    )
    .expect("concealed kong position");
    assert_eq!(
        physical(&concealed, &concealed_public),
        physical(&turn_hand, &public)
    );

    // Added kong on a turn: the existing peng is already public ownership.
    let melds = [Meld::peng(tile(Suit::Wan, 2), source)];
    let public_with_peng = public_of(&discards, &melds);
    let added_hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let (_, after_added, _, added_public) = kong_position(
        Action::AddedKong(tile(Suit::Wan, 2)),
        &added_hand,
        &melds,
        &public_with_peng,
    )
    .expect("added kong position");
    assert_eq!(
        physical(&after_added, &added_public),
        physical(&added_hand, &public_with_peng)
    );

    // Peng and open kong on a response: the pending discard is in `public`.
    let claim = tile(Suit::Wan, 1);
    let response_discards = [discard_info(source, claim, None, [false; 4])];
    let response_public = public_of(&response_discards, &[]);
    let response_hand = suit_hand(&[1, 1, 1, 2, 3, 4, 5, 6, 9, 9], &[1, 2, 3], &[]);
    let (_, after_peng, _, peng_public) = super::claim_position(
        Action::Peng,
        &response_hand,
        &[],
        &response_public,
        claim,
        source,
    )
    .expect("peng position");
    assert_eq!(
        physical(&after_peng, &peng_public),
        physical(&response_hand, &response_public)
    );

    // A mandatory post-peng discard only moves ownership.
    let discard = after_peng
        .distinct_tiles()
        .next()
        .expect("peng hand is non-empty");
    let mut after_peng_discard = after_peng;
    after_peng_discard.remove(discard).unwrap();
    let mut peng_public_discard = peng_public;
    peng_public_discard[discard.index() as usize] =
        peng_public_discard[discard.index() as usize].saturating_add(1);
    assert_eq!(
        physical(&after_peng_discard, &peng_public_discard),
        physical(&response_hand, &response_public)
    );

    let (_, after_open_kong, _, kong_public) = super::claim_position(
        Action::OpenKong,
        &response_hand,
        &[],
        &response_public,
        claim,
        source,
    )
    .expect("open kong position");
    assert_eq!(
        physical(&after_open_kong, &kong_public),
        physical(&response_hand, &response_public)
    );
}

#[test]
fn added_kong_path_is_evaluated_and_can_be_chosen() {
    let missing = Suit::Tiao;
    let source = Seat::new(1).expect("seat");
    let melds = [Meld::peng(tile(Suit::Wan, 2), source)];
    let hand = suit_hand(&[2, 3, 4, 5, 6, 7, 8, 9, 9], &[5, 5], &[]);
    let kong_tile = tile(Suit::Wan, 2);
    let public = public_of(&[], &melds);

    let discard = discard_progress(&hand, kong_tile, &melds, missing, &public);
    let (kind, after, after_melds, after_public) =
        kong_position(Action::AddedKong(kong_tile), &hand, &melds, &public)
            .expect("added kong position");
    assert_eq!(kind, MeldKind::AddedKong);
    assert_eq!(after_melds.len(), melds.len());
    let mut evaluator = DistanceEvaluator::new(&after_melds, missing).unwrap();
    let deficit = evaluator.deficit(&after);
    let kong = Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &after, &after_public, deficit),
    };
    assert_eq!(
        kong, discard,
        "test premise: the added kong and the fourth-tile discard have equal progress"
    );

    let actions = [Action::AddedKong(kong_tile), Action::Discard(kong_tile)];
    let choice = decision(
        &hand,
        &melds,
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );
    assert_eq!(
        HeuristicPolicy::new().choose(&choice),
        Action::AddedKong(kong_tile)
    );
}

#[test]
fn zero_availability_tenpai_still_uses_the_fan_tie_break() {
    let missing = Suit::Tiao;
    let hand = seven_pairs_plus_extra();
    let source = Seat::new(1).expect("seat");

    // Exhaust every structural wait of both discards. Availability is zero,
    // but both hands remain tenpai, so fan is still the final tie-break.
    let mut discards = Vec::new();
    for (kind, copies) in [
        (tile(Suit::Wan, 1), 1),
        (tile(Suit::Tong, 2), 2),
        (tile(Suit::Tong, 3), 2),
        (tile(Suit::Tong, 5), 2),
        (tile(Suit::Tong, 6), 2),
        (tile(Suit::Wan, 7), 3),
    ] {
        for _ in 0..copies {
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
    assert_eq!(low.ukeire, 0);
    assert_eq!(high.ukeire, 0);

    let mut low_after = hand;
    low_after.remove(lower_fan).unwrap();
    let mut high_after = hand;
    high_after.remove(higher_fan).unwrap();
    assert!(
        analysis::max_discard_fan(&high_after, &[], missing)
            > analysis::max_discard_fan(&low_after, &[], missing)
    );

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
        Action::Discard(higher_fan),
        "S = 0 and U = 0 is still tenpai and must use the fan tie-break"
    );
}

#[test]
fn repeated_decisions_are_cache_and_constructor_independent() {
    let missing = Suit::Tiao;
    let candidate_hand = tenpai_plus_extra();
    let actions = [
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Tong, 1)),
    ];
    let choice = decision(
        &candidate_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &actions,
        &[],
    );

    let mut policy = HeuristicPolicy::new();
    let first = policy.choose(&choice);

    // A different decision cannot change the next result: the policy holds no
    // logical history or decision cache.
    let other_hand = suit_hand(&[1, 4, 7, 2, 9, 9], &[1, 4, 7, 2], &[1, 4, 7, 2]);
    let other_actions = [Action::Discard(tile(Suit::Wan, 7))];
    let other = decision(
        &other_hand,
        &[],
        Some(missing),
        DecisionStage::Turn,
        None,
        &other_actions,
        &[],
    );
    let _ = policy.choose(&other);

    assert_eq!(policy.choose(&choice), first);
    let mut default_policy: HeuristicPolicy = Default::default();
    assert_eq!(default_policy.choose(&choice), first);
}
