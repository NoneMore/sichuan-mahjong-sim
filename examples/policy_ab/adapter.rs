//! Decision adapter for the one-off Python policy experiment.
//!
//! The adapter is the only place that translates a borrowed simulator
//! [`Decision`] into the worker wire format and maps a returned candidate ID
//! back to the original [`Action`]. It exposes only the deciding player's own
//! concealed hand, that player's own missing suit, and the legal options of
//! the pending decision. Opponent concealed hands, wall order, and staged
//! private choices are never part of the request.
//!
//! # Candidate identity
//!
//! Candidate order and IDs are part of the experiment identity: the imported
//! Python heuristic breaks ties by input order and ID, so both must be stable.
//!
//! - Ordinary candidates follow the game's legal action order exactly (turn:
//!   `Hu`, concealed kongs ascending, added kongs ascending, discards
//!   ascending, `Pass`; response: `Hu`, peng, open kong, `Pass`).
//! - Exchange candidates are enumerated by suit in the game's suit order
//!   (Wan/Tong/Tiao = `m`/`p`/`s`), then by non-decreasing rank triple
//!   `i <= j <= k` in lexicographic order, respecting hand multiplicity.
//! - IDs are derived from semantic action parameters, never from a list
//!   position: `win`, `pass`, `discard:m1`, `pong:m1`, `kong:m1`,
//!   `concealed_kong:m1`, `added_kong:m1`, `choose_missing_suit:m`, and
//!   `exchange_three:m1-m2-m3`. IDs within one decision are unique.
//!
//! Wire `type` names follow ninone's action vocabulary: `win`,
//! `exchange_three`, `choose_missing_suit`, `discard`, `pong`, `kong`,
//! `concealed_kong`, `added_kong`, `pass`. `pong` is deliberately not `peng`.
//! Tile kinds are ninone kinds: suit letter `m`/`p`/`s` followed by rank,
//! for example `m1`.

use std::collections::HashSet;
use std::fmt;

use serde::Serialize;
use sichuan_mahjong_sim::{Action, Decision, Hand, LegalActions, Seat, Suit, Tile};

/// One worker request for one pending decision.
///
/// The Python worker translates this shape into the imported heuristic's
/// `{observation: {own_hand, own_missing_suit}, legal_actions}` request. The
/// wire deliberately has no handshake, request ID, channel, or retry field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DecisionRequest {
    /// Own concealed hand as sorted ninone tile kinds (`m1` ... `s9`).
    pub hand: Vec<String>,
    /// Own declared missing suit, or `None` before all declarations exist.
    pub missing_suit: Option<&'static str>,
    /// Legal candidates in adapter identity order.
    pub candidates: Vec<WireCandidate>,
}

/// One ninone-style candidate on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WireCandidate {
    /// Stable candidate ID used to map the worker's choice back to an action.
    pub id: String,
    /// Ninone action type name.
    #[serde(rename = "type")]
    pub action_type: &'static str,
    /// Tile kind for tile-bearing actions (`discard`, `pong`, `kong`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Suit letter for `exchange_three` and `choose_missing_suit`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suit: Option<&'static str>,
    /// Sorted ninone tile kinds for `exchange_three`, empty otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tile_kinds: Vec<String>,
}

/// A complete request plus the mapping from every candidate ID to its action.
#[derive(Debug, Clone)]
pub struct CandidateSet {
    /// Serialized worker request.
    pub request: DecisionRequest,
    actions: Vec<(String, Action)>,
}

impl CandidateSet {
    /// Number of legal candidates exposed to the worker.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Maps a worker-provided candidate ID back to its original action.
    pub fn action_for(&self, id: &str) -> Option<Action> {
        self.actions
            .iter()
            .find(|(candidate, _)| candidate == id)
            .map(|(_, action)| *action)
    }
}

/// Adapter failures. None of them is recoverable by substituting a baseline
/// action: the experiment fails instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// The pending decision exposed no candidates.
    NoCandidates {
        /// Deciding seat.
        seat: Seat,
    },
    /// Two different actions produced the same candidate ID.
    DuplicateId {
        /// Repeated ID.
        id: String,
    },
    /// A claim action is missing its response context tile.
    MissingClaimTile {
        /// Deciding seat.
        seat: Seat,
        /// Candidate type that needed the tile.
        action_type: &'static str,
    },
    /// A turn/response action list contained a phase-only action.
    UnsupportedAction {
        /// Deciding seat.
        seat: Seat,
        /// Offending action.
        action: Action,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterError::NoCandidates { seat } => {
                write!(f, "pending decision for {seat} exposes no candidates")
            }
            AdapterError::DuplicateId { id } => {
                write!(f, "candidate ID '{id}' is not unique")
            }
            AdapterError::MissingClaimTile { seat, action_type } => write!(
                f,
                "{action_type} candidate for {seat} has no claimed tile in the decision"
            ),
            AdapterError::UnsupportedAction { seat, action } => {
                write!(
                    f,
                    "unsupported action {action:?} in the action list of {seat}"
                )
            }
        }
    }
}

impl std::error::Error for AdapterError {}

/// Builds the request and the ID-to-action mapping for one decision.
pub fn build(decision: &Decision<'_>) -> Result<CandidateSet, AdapterError> {
    let seat = decision.seat;
    let mut candidates = Vec::new();
    let mut actions = Vec::new();

    match &decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            for triple in exchange_triples(decision.observation.own_hand, *eligible_suits) {
                let kinds: Vec<String> = triple.iter().map(|&tile| kind_string(tile)).collect();
                let id = format!("exchange_three:{}", kinds.join("-"));
                candidates.push(WireCandidate {
                    id: id.clone(),
                    action_type: "exchange_three",
                    code: None,
                    suit: Some(suit_letter(triple[0].suit())),
                    tile_kinds: kinds,
                });
                actions.push((id, Action::Exchange(triple)));
            }
        }
        LegalActions::Actions(legal) => {
            for &action in *legal {
                let (candidate, action) = ordinary_candidate(decision, action)?;
                let id = candidate.id.clone();
                candidates.push(candidate);
                actions.push((id, action));
            }
        }
    }

    if candidates.is_empty() {
        return Err(AdapterError::NoCandidates { seat });
    }

    let mut seen = HashSet::with_capacity(candidates.len());
    for candidate in &candidates {
        if !seen.insert(candidate.id.as_str()) {
            return Err(AdapterError::DuplicateId {
                id: candidate.id.clone(),
            });
        }
    }

    let hand = decision
        .observation
        .own_hand
        .tiles()
        .map(kind_string)
        .collect();
    let missing_suit = decision.observation.players[seat.index()]
        .missing_suit
        .map(suit_letter);
    Ok(CandidateSet {
        request: DecisionRequest {
            hand,
            missing_suit,
            candidates,
        },
        actions,
    })
}

/// Number of candidates [`build`] would expose. Used only for failure context,
/// so it skips ID and kind-string construction.
pub fn candidate_count(decision: &Decision<'_>) -> usize {
    match &decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            exchange_triples(decision.observation.own_hand, *eligible_suits).len()
        }
        LegalActions::Actions(legal) => legal.len(),
    }
}

/// Lexicographic, multiplicity-respecting exchange triples in suit order.
pub fn exchange_triples(hand: &Hand, eligible_suits: [bool; 3]) -> Vec<[Tile; 3]> {
    let mut triples = Vec::new();
    for suit in Suit::ALL {
        if eligible_suits[suit.index()] {
            collect_suit_triples(hand, suit, &mut triples);
        }
    }
    triples
}

/// Ninone tile kind: suit letter followed by rank, for example `m1`.
pub fn kind_string(tile: Tile) -> String {
    format!("{}{}", suit_letter(tile.suit()), tile.rank())
}

/// Ninone suit letter: Wan = `m`, Tong = `p`, Tiao = `s`.
pub const fn suit_letter(suit: Suit) -> &'static str {
    match suit {
        Suit::Wan => "m",
        Suit::Tong => "p",
        Suit::Tiao => "s",
    }
}

fn collect_suit_triples(hand: &Hand, suit: Suit, out: &mut Vec<[Tile; 3]>) {
    let mut present: Vec<(Tile, u8)> = Vec::with_capacity(9);
    for rank in 1..=9u8 {
        let tile = Tile::from_suit_rank(suit, rank).expect("rank is in 1..=9");
        let count = hand.count(tile);
        if count > 0 {
            present.push((tile, count));
        }
    }
    let mut chosen = Vec::with_capacity(3);
    collect_triples_from(&present, 0, &mut chosen, out);
}

fn collect_triples_from(
    present: &[(Tile, u8)],
    start: usize,
    chosen: &mut Vec<Tile>,
    out: &mut Vec<[Tile; 3]>,
) {
    if chosen.len() == 3 {
        out.push([chosen[0], chosen[1], chosen[2]]);
        return;
    }
    for index in start..present.len() {
        let (tile, count) = present[index];
        let used = chosen.iter().filter(|&&picked| picked == tile).count();
        if used >= count as usize {
            continue;
        }
        chosen.push(tile);
        collect_triples_from(present, index, chosen, out);
        chosen.pop();
    }
}

fn ordinary_candidate(
    decision: &Decision<'_>,
    action: Action,
) -> Result<(WireCandidate, Action), AdapterError> {
    let seat = decision.seat;
    let candidate = match action {
        Action::Hu => WireCandidate {
            id: "win".to_owned(),
            action_type: "win",
            code: None,
            suit: None,
            tile_kinds: Vec::new(),
        },
        Action::Pass => WireCandidate {
            id: "pass".to_owned(),
            action_type: "pass",
            code: None,
            suit: None,
            tile_kinds: Vec::new(),
        },
        Action::Discard(tile) => tile_candidate("discard", "discard", tile),
        Action::ConcealedKong(tile) => tile_candidate("concealed_kong", "concealed_kong", tile),
        Action::AddedKong(tile) => tile_candidate("added_kong", "added_kong", tile),
        Action::Peng => {
            let tile = claimed_tile(decision, seat, "pong")?;
            tile_candidate("pong", "pong", tile)
        }
        Action::OpenKong => {
            let tile = claimed_tile(decision, seat, "kong")?;
            tile_candidate("kong", "kong", tile)
        }
        Action::ChooseMissing(suit) => WireCandidate {
            id: format!("choose_missing_suit:{}", suit_letter(suit)),
            action_type: "choose_missing_suit",
            code: None,
            suit: Some(suit_letter(suit)),
            tile_kinds: Vec::new(),
        },
        Action::Exchange(_) => {
            return Err(AdapterError::UnsupportedAction { seat, action });
        }
    };
    Ok((candidate, action))
}

fn claimed_tile(
    decision: &Decision<'_>,
    seat: Seat,
    action_type: &'static str,
) -> Result<Tile, AdapterError> {
    decision
        .observation
        .response
        .map(|response| response.tile)
        .ok_or(AdapterError::MissingClaimTile { seat, action_type })
}

fn tile_candidate(id_prefix: &str, action_type: &'static str, tile: Tile) -> WireCandidate {
    let code = kind_string(tile);
    WireCandidate {
        id: format!("{id_prefix}:{code}"),
        action_type,
        code: Some(code),
        suit: None,
        tile_kinds: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sichuan_mahjong_sim::{
        DecisionStage, ExchangeDirection, FirstLegalPolicy, Game, Observation, PlayerView, Policy,
        ResponseView,
    };

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    fn hand(tiles: &[(Suit, u8)]) -> Hand {
        let mut hand = Hand::new();
        for &(suit, rank) in tiles {
            hand.add(tile(suit, rank)).unwrap();
        }
        hand
    }

    fn view<'a>(
        seat: Seat,
        own_hand: &'a Hand,
        actions: &'a [Action],
        missing: Option<Suit>,
        response: Option<ResponseView>,
        stage: DecisionStage,
    ) -> Decision<'a> {
        let players = std::array::from_fn(|index| PlayerView {
            seat: Seat::new(index).unwrap(),
            concealed_count: if index == seat.index() {
                own_hand.total()
            } else {
                13
            },
            melds: &[],
            missing_suit: missing,
            score: 0,
            has_won: false,
        });
        Decision {
            seat,
            observation: Observation {
                own_hand,
                own_melds: &[],
                players,
                discards: &[],
                exchange_direction: ExchangeDirection::Clockwise,
                wall_count: 50,
                stage,
                response,
            },
            legal_actions: LegalActions::Actions(actions),
        }
    }

    fn exchange_view<'a>(
        seat: Seat,
        own_hand: &'a Hand,
        eligible_suits: [bool; 3],
    ) -> Decision<'a> {
        let players = std::array::from_fn(|index| PlayerView {
            seat: Seat::new(index).unwrap(),
            concealed_count: if index == seat.index() {
                own_hand.total()
            } else {
                13
            },
            melds: &[],
            missing_suit: None,
            score: 0,
            has_won: false,
        });
        Decision {
            seat,
            observation: Observation {
                own_hand,
                own_melds: &[],
                players,
                discards: &[],
                exchange_direction: ExchangeDirection::Clockwise,
                wall_count: 100,
                stage: DecisionStage::Exchange,
                response: None,
            },
            legal_actions: LegalActions::Exchange { eligible_suits },
        }
    }

    #[test]
    fn exchange_triples_follow_lexicographic_multiplicity_order() {
        let hand = hand(&[
            (Suit::Wan, 1),
            (Suit::Wan, 1),
            (Suit::Wan, 2),
            (Suit::Wan, 3),
            (Suit::Tong, 5),
        ]);
        let triples = exchange_triples(&hand, [true, false, false]);
        assert_eq!(
            triples,
            vec![
                [tile(Suit::Wan, 1), tile(Suit::Wan, 1), tile(Suit::Wan, 2)],
                [tile(Suit::Wan, 1), tile(Suit::Wan, 1), tile(Suit::Wan, 3)],
                [tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)],
            ]
        );

        // A suit with fewer than three tiles exposes nothing even if asked.
        assert!(exchange_triples(&hand, [false, true, false]).is_empty());
        // Suit order is Wan, Tong, Tiao.
        let all = exchange_triples(&hand, [true, false, true]);
        assert_eq!(all, triples);
    }

    #[test]
    fn exchange_triples_cover_every_legal_multiset() {
        let hand = hand(&[
            (Suit::Wan, 1),
            (Suit::Wan, 2),
            (Suit::Wan, 2),
            (Suit::Wan, 2),
            (Suit::Wan, 2),
            (Suit::Wan, 9),
            (Suit::Tong, 3),
            (Suit::Tong, 3),
            (Suit::Tong, 3),
            (Suit::Tiao, 7),
        ]);
        let eligible = [true, true, false];
        let actual = exchange_triples(&hand, eligible);

        // Independent brute force over the sorted physical tile list: strict
        // index triples produce exactly the non-decreasing multisets.
        let physical: Vec<Tile> = hand.tiles().collect();
        let mut expected = Vec::new();
        for i in 0..physical.len() {
            for j in i + 1..physical.len() {
                for k in j + 1..physical.len() {
                    let tiles = [physical[i], physical[j], physical[k]];
                    let suit = tiles[0].suit();
                    if tiles.iter().all(|&t| t.suit() == suit) && eligible[suit.index()] {
                        expected.push(tiles);
                    }
                }
            }
        }
        expected.sort();
        expected.dedup();
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 5);
    }

    #[test]
    fn exchange_request_uses_ninone_fields_and_ids() {
        let hand = hand(&[(Suit::Wan, 1), (Suit::Wan, 2), (Suit::Wan, 2)]);
        let decision = exchange_view(Seat::new(1).unwrap(), &hand, [true, false, false]);
        let set = build(&decision).unwrap();
        let candidate = &set.request.candidates[0];
        assert_eq!(candidate.id, "exchange_three:m1-m2-m2");
        assert_eq!(candidate.action_type, "exchange_three");
        assert_eq!(candidate.suit, Some("m"));
        assert_eq!(candidate.tile_kinds, vec!["m1", "m2", "m2"]);
        assert_eq!(set.request.hand, vec!["m1", "m2", "m2"]);
        assert_eq!(set.request.missing_suit, None);
        assert_eq!(
            set.action_for("exchange_three:m1-m2-m2"),
            Some(Action::Exchange([
                tile(Suit::Wan, 1),
                tile(Suit::Wan, 2),
                tile(Suit::Wan, 2)
            ]))
        );
        assert_eq!(set.action_for("exchange_three:m1-m1-m2"), None);
    }

    #[test]
    fn ordinary_candidates_follow_the_game_order_and_map_back() {
        let hand = hand(&[
            (Suit::Wan, 1),
            (Suit::Wan, 2),
            (Suit::Wan, 3),
            (Suit::Tong, 4),
        ]);
        let actions = [
            Action::Hu,
            Action::ConcealedKong(tile(Suit::Wan, 1)),
            Action::AddedKong(tile(Suit::Wan, 2)),
            Action::Discard(tile(Suit::Wan, 3)),
            Action::Discard(tile(Suit::Tong, 4)),
            Action::Pass,
        ];
        let decision = view(
            Seat::new(2).unwrap(),
            &hand,
            &actions,
            Some(Suit::Tong),
            None,
            DecisionStage::Turn,
        );
        let set = build(&decision).unwrap();
        let ids: Vec<&str> = set
            .request
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec![
                "win",
                "concealed_kong:m1",
                "added_kong:m2",
                "discard:m3",
                "discard:p4",
                "pass"
            ]
        );
        let types: Vec<&str> = set
            .request
            .candidates
            .iter()
            .map(|candidate| candidate.action_type)
            .collect();
        assert_eq!(
            types,
            vec![
                "win",
                "concealed_kong",
                "added_kong",
                "discard",
                "discard",
                "pass"
            ]
        );
        assert_eq!(set.request.missing_suit, Some("p"));
        for (candidate, action) in set.request.candidates.iter().zip(actions) {
            assert_eq!(set.action_for(&candidate.id), Some(action));
        }
    }

    #[test]
    fn missing_suit_candidates_use_suit_letters() {
        let hand = hand(&[(Suit::Wan, 1)]);
        let actions = [
            Action::ChooseMissing(Suit::Wan),
            Action::ChooseMissing(Suit::Tong),
            Action::ChooseMissing(Suit::Tiao),
        ];
        let decision = view(
            Seat::DEALER,
            &hand,
            &actions,
            None,
            None,
            DecisionStage::MissingSuit,
        );
        let set = build(&decision).unwrap();
        assert_eq!(
            set.request.candidates,
            vec![
                WireCandidate {
                    id: "choose_missing_suit:m".to_owned(),
                    action_type: "choose_missing_suit",
                    code: None,
                    suit: Some("m"),
                    tile_kinds: Vec::new(),
                },
                WireCandidate {
                    id: "choose_missing_suit:p".to_owned(),
                    action_type: "choose_missing_suit",
                    code: None,
                    suit: Some("p"),
                    tile_kinds: Vec::new(),
                },
                WireCandidate {
                    id: "choose_missing_suit:s".to_owned(),
                    action_type: "choose_missing_suit",
                    code: None,
                    suit: Some("s"),
                    tile_kinds: Vec::new(),
                },
            ]
        );
        assert_eq!(
            set.action_for("choose_missing_suit:s"),
            Some(Action::ChooseMissing(Suit::Tiao))
        );
    }

    #[test]
    fn claims_use_the_claimed_tile_and_pong_is_not_peng() {
        let hand = hand(&[(Suit::Tong, 5), (Suit::Tong, 5), (Suit::Tong, 5)]);
        let actions = [Action::Hu, Action::Peng, Action::OpenKong, Action::Pass];
        let response = ResponseView {
            source: Seat::new(1).unwrap(),
            tile: tile(Suit::Tong, 5),
            robbing_added_kong: false,
        };
        let decision = view(
            Seat::new(2).unwrap(),
            &hand,
            &actions,
            Some(Suit::Wan),
            Some(response),
            DecisionStage::DiscardResponse,
        );
        let set = build(&decision).unwrap();
        let ids: Vec<&str> = set
            .request
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        assert_eq!(ids, vec!["win", "pong:p5", "kong:p5", "pass"]);
        assert_eq!(
            set.request.candidates[1].code.as_deref(),
            Some("p5"),
            "pong candidates carry the claimed tile kind"
        );
        assert_eq!(
            set.action_for("kong:p5"),
            Some(Action::OpenKong),
            "kong maps back to the open-kong claim"
        );
    }

    #[test]
    fn claim_without_response_context_is_an_error() {
        let hand = hand(&[(Suit::Tong, 5), (Suit::Tong, 5)]);
        let actions = [Action::Peng];
        let decision = view(
            Seat::new(2).unwrap(),
            &hand,
            &actions,
            Some(Suit::Wan),
            None,
            DecisionStage::DiscardResponse,
        );
        assert_eq!(
            build(&decision).unwrap_err(),
            AdapterError::MissingClaimTile {
                seat: Seat::new(2).unwrap(),
                action_type: "pong"
            }
        );
    }

    #[test]
    fn no_candidates_is_an_error() {
        let hand = hand(&[(Suit::Wan, 1)]);
        let decision = view(
            Seat::DEALER,
            &hand,
            &[],
            Some(Suit::Tong),
            None,
            DecisionStage::Turn,
        );
        assert!(matches!(
            build(&decision),
            Err(AdapterError::NoCandidates { .. })
        ));
    }

    #[test]
    fn wire_request_serialization_is_pinned() {
        let hand = hand(&[(Suit::Wan, 1), (Suit::Wan, 2), (Suit::Wan, 2)]);
        let decision = exchange_view(Seat::new(3).unwrap(), &hand, [true, false, false]);
        let set = build(&decision).unwrap();
        assert_eq!(
            serde_json::to_string(&set.request).unwrap(),
            concat!(
                r#"{"hand":["m1","m2","m2"],"missing_suit":null,"candidates":["#,
                r#"{"id":"exchange_three:m1-m2-m2","type":"exchange_three","suit":"m","tile_kinds":["m1","m2","m2"]}"#,
                "]}"
            )
        );
    }

    #[test]
    fn real_game_decisions_expose_complete_candidates() {
        let mut stages = HashSet::new();
        for seed in 0..4u64 {
            let mut game = Game::new(seed);
            let mut policy = FirstLegalPolicy::new();
            loop {
                let Some(decision) = game.next_decision() else {
                    break;
                };
                stages.insert(format!("{:?}", decision.observation.stage));
                let set = build(&decision).unwrap();
                let mut ids = HashSet::new();
                for candidate in &set.request.candidates {
                    assert!(ids.insert(candidate.id.clone()));
                    assert!(set.action_for(&candidate.id).is_some());
                }
                match &decision.legal_actions {
                    LegalActions::Actions(legal) => {
                        let mapped: Vec<Action> = set
                            .request
                            .candidates
                            .iter()
                            .map(|candidate| set.action_for(&candidate.id).unwrap())
                            .collect();
                        assert_eq!(mapped.as_slice(), *legal);
                        assert_eq!(candidate_count(&decision), legal.len());
                    }
                    LegalActions::Exchange { eligible_suits } => {
                        assert_eq!(
                            candidate_count(&decision),
                            exchange_triples(decision.observation.own_hand, *eligible_suits).len()
                        );
                        for candidate in &set.request.candidates {
                            let Some(Action::Exchange(tiles)) = set.action_for(&candidate.id)
                            else {
                                panic!("exchange candidate must map to an exchange action");
                            };
                            assert!(eligible_suits[tiles[0].suit_index()]);
                            assert!(sichuan_mahjong_sim::rules::is_valid_exchange(
                                decision.observation.own_hand,
                                tiles
                            ));
                        }
                    }
                }
                let action = policy.choose(&decision);
                let seat = decision.seat;
                game.submit(seat, action).unwrap();
            }
            assert!(game.result().is_some());
        }
        assert!(stages.contains("Exchange"));
        assert!(stages.contains("MissingSuit"));
        assert!(stages.contains("Turn"));
        assert!(
            stages.len() >= 3,
            "adapter coverage should include several decision stages"
        );
    }
}
