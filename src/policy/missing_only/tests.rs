//! Decision-level checks for [`MissingOnlyPolicy`].
//!
//! The policy is a one-decision ablation, so each test pins one half of its
//! contract: the missing-suit declaration must equal the heuristic rule, and
//! every other decision must equal the deterministic baseline. Synthetic
//! decisions keep the boundary explicit: only the observation and the legal
//! action list are visible.

use super::*;
use crate::action::{Action, ExchangeDirection};
use crate::game::{Decision, DiscardInfo, LegalActions, Observation, PlayerView, ResponseView};
use crate::hand::Hand;
use crate::policy::HeuristicPolicy;
use crate::seat::{Seat, PLAYER_COUNT};
use crate::tile::{Suit, Tile};

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).expect("legal hand")
}

fn view<'a>(seat: Seat, own_hand: &Hand, missing: Option<Suit>) -> [PlayerView<'a>; PLAYER_COUNT] {
    std::array::from_fn(|index| {
        let view_seat = Seat::new(index).expect("seat index");
        PlayerView {
            seat: view_seat,
            concealed_count: if view_seat == seat {
                own_hand.total()
            } else {
                13
            },
            melds: &[],
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

fn decision<'a>(hand: &'a Hand, stage: DecisionStage, actions: &'a [Action]) -> Decision<'a> {
    let seat = Seat::DEALER;
    Decision {
        seat,
        observation: Observation {
            own_hand: hand,
            own_melds: &[],
            players: view(seat, hand, Some(Suit::Wan)),
            discards: &[] as &[DiscardInfo],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage,
            response: None as Option<ResponseView>,
        },
        legal_actions: LegalActions::Actions(actions),
    }
}

fn exchange_decision<'a>(hand: &'a Hand, eligible_suits: [bool; 3]) -> Decision<'a> {
    let seat = Seat::DEALER;
    Decision {
        seat,
        observation: Observation {
            own_hand: hand,
            own_melds: &[],
            players: view(seat, hand, Some(Suit::Wan)),
            discards: &[] as &[DiscardInfo],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Exchange,
            response: None as Option<ResponseView>,
        },
        legal_actions: LegalActions::Exchange { eligible_suits },
    }
}

/// Runs both reference policies on the same decision.
fn references(decision: &Decision<'_>) -> (Action, Action) {
    (
        FirstLegalPolicy::new().choose(decision),
        HeuristicPolicy::new().choose(decision),
    )
}

fn missing_suit_actions() -> [Action; 3] {
    [
        Action::ChooseMissing(Suit::Wan),
        Action::ChooseMissing(Suit::Tong),
        Action::ChooseMissing(Suit::Tiao),
    ]
}

#[test]
fn missing_suit_matches_the_heuristic_fewest_tile_rule() {
    // Tong holds the fewest tiles, so the heuristic rule must win even though
    // Wan is listed first and the baseline would take it.
    let hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 4),
        tile(Suit::Tiao, 5),
        tile(Suit::Tiao, 6),
    ]);
    let actions = missing_suit_actions();
    let decision = decision(&hand, DecisionStage::MissingSuit, &actions);

    let (first, heuristic) = references(&decision);
    assert_eq!(first, Action::ChooseMissing(Suit::Wan));
    assert_eq!(heuristic, Action::ChooseMissing(Suit::Tong));
    assert_eq!(MissingOnlyPolicy::new().choose(&decision), heuristic);
}

#[test]
fn missing_suit_tie_uses_the_lowest_suit_index_like_the_heuristic() {
    // Wan and Tong tie on two tiles; Tiao is larger. Both policies must pick
    // Wan, regardless of the legal list order. The baseline would instead
    // pick the first listed action, Tiao, from the reversed list below.
    let hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 4),
        tile(Suit::Tiao, 5),
        tile(Suit::Tiao, 6),
        tile(Suit::Tiao, 7),
        tile(Suit::Tiao, 8),
        tile(Suit::Tiao, 9),
    ]);
    // Reverse the legal list so the assertion cannot pass by list order alone.
    let actions = [
        Action::ChooseMissing(Suit::Tiao),
        Action::ChooseMissing(Suit::Tong),
        Action::ChooseMissing(Suit::Wan),
    ];
    let decision = decision(&hand, DecisionStage::MissingSuit, &actions);

    let (_, heuristic) = references(&decision);
    assert_eq!(heuristic, Action::ChooseMissing(Suit::Wan));
    assert_eq!(MissingOnlyPolicy::new().choose(&decision), heuristic);
}

#[test]
fn missing_suit_falls_back_to_the_first_legal_action_like_the_heuristic() {
    // Wan is the fewest-tile suit but is not offered, so both analysis policies
    // fall back to the first legal action.
    let hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 4),
        tile(Suit::Tiao, 5),
        tile(Suit::Tiao, 6),
        tile(Suit::Tiao, 7),
    ]);
    let actions = [
        Action::ChooseMissing(Suit::Tiao),
        Action::ChooseMissing(Suit::Tong),
    ];
    let decision = decision(&hand, DecisionStage::MissingSuit, &actions);

    let (first, heuristic) = references(&decision);
    assert_eq!(first, Action::ChooseMissing(Suit::Tiao));
    assert_eq!(heuristic, Action::ChooseMissing(Suit::Tiao));
    assert_eq!(MissingOnlyPolicy::new().choose(&decision), first);
}

#[test]
fn exchange_and_turn_decisions_keep_the_baseline_choice() {
    // Exchange: fewest-tile eligible suit, lowest three tiles, as the baseline.
    let exchange_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 4),
        tile(Suit::Tiao, 5),
        tile(Suit::Tiao, 6),
    ]);
    let exchange = exchange_decision(&exchange_hand, [true, true, true]);
    let (first_exchange, _) = references(&exchange);
    assert_eq!(
        first_exchange,
        Action::Exchange([tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)])
    );
    assert_eq!(
        MissingOnlyPolicy::new().choose(&exchange),
        first_exchange,
        "exchange must stay on the baseline choice"
    );

    // Turn: three complete sets, a Tiao 7/8 partial run, a Tiao 9 pair, and a
    // loose Tong 9. The baseline takes the first listed discard (the Tiao 9 that
    // keeps the hand 1-shanten); the heuristic keeps the tenpai Tong 9 discard.
    let turn_hand = hand(&[
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
        tile(Suit::Tong, 9),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Tiao, 7),
        tile(Suit::Tiao, 8),
        tile(Suit::Tiao, 9),
        tile(Suit::Tiao, 9),
    ]);
    let turn_actions = [
        Action::Discard(tile(Suit::Tiao, 9)),
        Action::Discard(tile(Suit::Tiao, 8)),
        Action::Discard(tile(Suit::Tiao, 7)),
        Action::Discard(tile(Suit::Tiao, 3)),
        Action::Discard(tile(Suit::Tiao, 2)),
        Action::Discard(tile(Suit::Tiao, 1)),
        Action::Discard(tile(Suit::Tong, 9)),
        Action::Discard(tile(Suit::Tong, 6)),
        Action::Discard(tile(Suit::Tong, 5)),
        Action::Discard(tile(Suit::Tong, 4)),
        Action::Discard(tile(Suit::Tong, 3)),
        Action::Discard(tile(Suit::Tong, 2)),
        Action::Discard(tile(Suit::Tong, 1)),
    ];
    let turn = decision(&turn_hand, DecisionStage::Turn, &turn_actions);
    let (first_turn, heuristic_turn) = references(&turn);
    assert_eq!(
        first_turn, turn_actions[0],
        "the baseline takes the first legal discard"
    );
    assert_ne!(
        heuristic_turn, first_turn,
        "the heuristic must diverge here for this test to be meaningful"
    );
    assert_eq!(MissingOnlyPolicy::new().choose(&turn), first_turn);
}

#[test]
fn a_legal_hu_is_preferred_before_any_analysis() {
    let hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 2),
        tile(Suit::Tiao, 3),
        tile(Suit::Wan, 7),
        tile(Suit::Wan, 7),
        tile(Suit::Tong, 7),
        tile(Suit::Tong, 7),
        tile(Suit::Tiao, 7),
    ]);
    let actions = [
        Action::Discard(tile(Suit::Tiao, 7)),
        Action::Hu,
        Action::Discard(tile(Suit::Wan, 7)),
    ];
    let decision = decision(&hand, DecisionStage::Turn, &actions);

    let (first, heuristic) = references(&decision);
    assert_eq!(first, Action::Hu);
    assert_eq!(heuristic, Action::Hu);
    assert_eq!(MissingOnlyPolicy::new().choose(&decision), Action::Hu);
}
