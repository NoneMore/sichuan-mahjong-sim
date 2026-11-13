//! Black-box acceptance checks for the v1 neural-policy encoding.
//!
//! The tests drive real seeded [`Game`] decisions through every public stage,
//! validate candidate membership and submission, reconstruct the
//! teacher-relevant decision surface from encoding output alone, and check
//! encoded collision keys. Numeric ownership snapshots and action-feature
//! golden fixtures belong to separate focused tests.
//!
//! The encoder accepts only the public [`Decision`]; an [`Observation`] carries
//! the deciding hand and public state, so another seat's concealed hand has no
//! interface through which it could enter the encoding.

use std::collections::{HashMap, HashSet};

use sichuan_mahjong_neural::encoding::{
    canonicalize, encode, Candidate, EncodedDecision, MeldSlot, StateFeatures, TargetNotCandidate,
    MELD_SLOTS,
};
use sichuan_mahjong_sim::{
    Action, Decision, DecisionStage, DiscardInfo, ExchangeDirection, Game, Hand, HeuristicPolicy,
    LegalActions, Meld, MeldKind, Observation, PlayerView, Policy, ResponseView, Seat, Suit, Tile,
    PLAYER_COUNT,
};

/// Seed whose aggressive claim/kong script reaches a pending AddedKongResponse
/// at decision 45. The search was exploratory; the fixed seed is the long-term
/// regression input.
const ADDED_KONG_SEED: u64 = 61;

fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in 1..=9")
}

fn seat(index: usize) -> Seat {
    Seat::new(index).expect("seat index in 0..=3")
}

fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).expect("test hand is physical")
}

fn decode_scaled(value: f32, divisor: f32, what: &str) -> usize {
    let scaled = value * divisor;
    let rounded = scaled.round();
    assert!(
        (scaled - rounded).abs() < 1e-3,
        "{what}: {value} is not an integer count over {divisor}"
    );
    assert!(rounded >= 0.0, "{what}: negative decoded count");
    rounded as usize
}

fn decode_signed(value: f32, divisor: f32, what: &str) -> i32 {
    let scaled = value * divisor;
    let rounded = scaled.round();
    assert!(
        (scaled - rounded).abs() < 1e-3,
        "{what}: {value} is not an integer score over {divisor}"
    );
    rounded as i32
}

fn decode_one_hot(values: &[f32], width: usize, what: &str) -> usize {
    let mut found = None;
    for (index, &value) in values[..width].iter().enumerate() {
        assert!(
            value == 0.0 || value == 1.0,
            "{what}: expected one-hot values, got {value}"
        );
        if value == 1.0 {
            assert!(found.is_none(), "{what}: multiple one-hot entries");
            found = Some(index);
        }
    }
    found.unwrap_or_else(|| panic!("{what}: no one-hot entry"))
}

fn decode_flag(value: f32, what: &str) -> bool {
    assert!(
        value == 0.0 || value == 1.0,
        "{what}: expected a 0/1 flag, got {value}"
    );
    value == 1.0
}

fn stage_index(stage: DecisionStage) -> usize {
    match stage {
        DecisionStage::Exchange => 0,
        DecisionStage::MissingSuit => 1,
        DecisionStage::Turn => 2,
        DecisionStage::DiscardResponse => 3,
        DecisionStage::AddedKongResponse => 4,
    }
}

fn stage_name(stage: DecisionStage) -> &'static str {
    match stage {
        DecisionStage::Exchange => "Exchange",
        DecisionStage::MissingSuit => "MissingSuit",
        DecisionStage::Turn => "Turn",
        DecisionStage::DiscardResponse => "DiscardResponse",
        DecisionStage::AddedKongResponse => "AddedKongResponse",
    }
}

#[derive(Default)]
struct StageCoverage {
    stages: [bool; 5],
    saw_won_player: bool,
    saw_meld: bool,
    saw_multiple_exchange_suits: bool,
    saw_repeated_exchange_tile: bool,
}

impl StageCoverage {
    fn mark(&mut self, stage: DecisionStage) {
        self.stages[stage_index(stage)] = true;
    }

    fn mark_state(&mut self, stage: DecisionStage, encoded: &EncodedDecision) {
        if encoded
            .state
            .players
            .iter()
            .any(|player| player.has_won == 1.0)
        {
            self.saw_won_player = true;
        }
        if encoded
            .state
            .players
            .iter()
            .any(|player| player.melds.iter().any(|meld| meld.valid == 1.0))
        {
            self.saw_meld = true;
        }
        if stage == DecisionStage::Exchange {
            let mut suits = [false; 3];
            for candidate in &encoded.candidates {
                let Action::Exchange(tiles) = candidate.action else {
                    continue;
                };
                suits[tiles[0].suit_index()] = true;
                if tiles[0] == tiles[1] || tiles[1] == tiles[2] {
                    self.saw_repeated_exchange_tile = true;
                }
            }
            if suits.into_iter().filter(|present| *present).count() >= 2 {
                self.saw_multiple_exchange_suits = true;
            }
        }
    }

    fn assert_all(&self, context: &str) {
        let missing: Vec<&str> = [
            DecisionStage::Exchange,
            DecisionStage::MissingSuit,
            DecisionStage::Turn,
            DecisionStage::DiscardResponse,
            DecisionStage::AddedKongResponse,
        ]
        .into_iter()
        .filter(|&stage| !self.stages[stage_index(stage)])
        .map(stage_name)
        .collect();
        assert!(missing.is_empty(), "{context}: missing stages {missing:?}");
        assert!(
            self.saw_won_player,
            "{context}: no decision exposed a won player"
        );
        assert!(
            self.saw_meld,
            "{context}: no decision exposed a declared meld"
        );
        assert!(
            self.saw_multiple_exchange_suits,
            "{context}: no exchange decision exposed multiple eligible suits"
        );
        assert!(
            self.saw_repeated_exchange_tile,
            "{context}: no exchange candidate repeated a tile"
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LegalSignature {
    Exchange { eligible_suits: [bool; 3] },
    Actions(Vec<Action>),
}

fn legal_signature(legal: &LegalActions<'_>) -> LegalSignature {
    match *legal {
        LegalActions::Exchange { eligible_suits } => LegalSignature::Exchange { eligible_suits },
        LegalActions::Actions(actions) => LegalSignature::Actions(actions.to_vec()),
    }
}

fn hand_from_encoded(counts: &[f32; 27]) -> Hand {
    let mut tiles = Vec::new();
    for (index, &scaled) in counts.iter().enumerate() {
        let count = decode_scaled(scaled, 4.0, "own hand count");
        assert!(count <= 4, "own hand has more than four copies");
        for _ in 0..count {
            tiles.push(Tile::from_index(index as u8).expect("tile index in range"));
        }
    }
    hand(&tiles)
}

fn decode_missing_suit(values: &[f32; 4]) -> Option<Suit> {
    match decode_one_hot(values, 4, "missing suit") {
        0 => Some(Suit::Wan),
        1 => Some(Suit::Tong),
        2 => Some(Suit::Tiao),
        3 => None,
        _ => unreachable!(),
    }
}

fn decode_meld_kind(values: &[f32; 4]) -> MeldKind {
    match decode_one_hot(values, 4, "meld kind") {
        0 => MeldKind::Peng,
        1 => MeldKind::OpenKong,
        2 => MeldKind::AddedKong,
        3 => MeldKind::ConcealedKong,
        _ => unreachable!(),
    }
}

fn decode_meld_source(values: &[f32; 5]) -> Option<Seat> {
    match decode_one_hot(values, 5, "meld source") {
        4 => None,
        index => Some(seat(index)),
    }
}

fn decode_melds(slots: &[MeldSlot; MELD_SLOTS]) -> Vec<Meld> {
    let mut melds = Vec::new();
    let mut previous_tile = None;
    for slot in slots {
        let valid = decode_flag(slot.valid, "meld validity");
        if !valid {
            assert!(
                slot.tile.iter().all(|&value| value == 0.0)
                    && slot.kind.iter().all(|&value| value == 0.0)
                    && slot.source.iter().all(|&value| value == 0.0),
                "unused meld slot must be all zero"
            );
            continue;
        }
        let tile_index = decode_one_hot(&slot.tile, 27, "meld tile");
        if let Some(previous) = previous_tile {
            assert!(
                previous < tile_index,
                "decoded meld slots are not sorted by tile index"
            );
        }
        previous_tile = Some(tile_index);
        let tile = Tile::from_index(tile_index as u8).expect("tile index in range");
        let kind = decode_meld_kind(&slot.kind);
        let source = decode_meld_source(&slot.source);
        melds.push(Meld::new(kind, tile, source).expect("decoded meld is consistent"));
    }
    melds
}

#[derive(Clone, Debug)]
struct ProjectedPlayer {
    concealed_count: usize,
    melds: Vec<Meld>,
    missing_suit: Option<Suit>,
    score: i32,
    has_won: bool,
}

/// A decision reconstructed only from [`EncodedDecision`].
///
/// All references are relative to the deciding player, so the projected seat is
/// always seat 0. No field is copied from the original decision.
struct ProjectedDecision {
    seat: Seat,
    own_hand: Hand,
    players: [ProjectedPlayer; PLAYER_COUNT],
    discards: Vec<DiscardInfo>,
    exchange_direction: ExchangeDirection,
    wall_count: usize,
    stage: DecisionStage,
    response: Option<ResponseView>,
    legal: LegalSignature,
}

impl ProjectedDecision {
    fn from_encoded(encoded: &EncodedDecision) -> Self {
        let state = &encoded.state;
        let stage = decode_stage(&state.context.stage);
        let exchange_direction = decode_exchange_direction(&state.context.exchange_direction);
        let wall_count = decode_scaled(state.context.wall_count, 108.0, "wall count");
        let own_hand = hand_from_encoded(&state.own_hand);
        let players: [ProjectedPlayer; PLAYER_COUNT] = std::array::from_fn(|relative| {
            let player = &state.players[relative];
            ProjectedPlayer {
                concealed_count: decode_scaled(player.concealed_total, 14.0, "concealed total"),
                melds: decode_melds(&player.melds),
                missing_suit: decode_missing_suit(&player.missing_suit),
                score: decode_signed(player.score, 16.0, "score"),
                has_won: decode_flag(player.has_won, "has_won"),
            }
        });
        let discards = decode_discards(state, &players);
        let response = decode_response(state, stage);
        let legal = decode_legal(&encoded.candidates, &own_hand, stage);
        Self {
            seat: seat(0),
            own_hand,
            players,
            discards,
            exchange_direction,
            wall_count,
            stage,
            response,
            legal,
        }
    }

    fn as_decision(&self) -> Decision<'_> {
        let players: [PlayerView<'_>; PLAYER_COUNT] = std::array::from_fn(|index| {
            let player = &self.players[index];
            PlayerView {
                seat: seat(index),
                concealed_count: player.concealed_count,
                melds: &player.melds,
                missing_suit: player.missing_suit,
                score: player.score,
                has_won: player.has_won,
            }
        });
        let legal_actions = match &self.legal {
            LegalSignature::Exchange { eligible_suits } => LegalActions::Exchange {
                eligible_suits: *eligible_suits,
            },
            LegalSignature::Actions(actions) => LegalActions::Actions(actions),
        };
        Decision {
            seat: self.seat,
            observation: Observation {
                own_hand: &self.own_hand,
                own_melds: &self.players[0].melds,
                players,
                discards: &self.discards,
                exchange_direction: self.exchange_direction,
                wall_count: self.wall_count,
                stage: self.stage,
                response: self.response,
            },
            legal_actions,
        }
    }
}

fn decode_stage(values: &[f32; 5]) -> DecisionStage {
    match decode_one_hot(values, 5, "stage") {
        0 => DecisionStage::Exchange,
        1 => DecisionStage::MissingSuit,
        2 => DecisionStage::Turn,
        3 => DecisionStage::DiscardResponse,
        4 => DecisionStage::AddedKongResponse,
        _ => unreachable!(),
    }
}

fn decode_exchange_direction(values: &[f32; 3]) -> ExchangeDirection {
    match decode_one_hot(values, 3, "exchange direction") {
        0 => ExchangeDirection::Clockwise,
        1 => ExchangeDirection::CounterClockwise,
        2 => ExchangeDirection::Opposite,
        _ => unreachable!(),
    }
}

fn decode_discards(
    state: &StateFeatures,
    players: &[ProjectedPlayer; PLAYER_COUNT],
) -> Vec<DiscardInfo> {
    let mut records = Vec::new();
    for (source, discards) in state.discards.iter().enumerate() {
        for (category, tiles) in discards.iter().enumerate() {
            for (tile_index, &value) in tiles.iter().enumerate() {
                let count = decode_scaled(value, 4.0, "discard count");
                for _ in 0..count {
                    let tile = Tile::from_index(tile_index as u8).expect("tile index in range");
                    let mut winners = [false; PLAYER_COUNT];
                    let meld_claim = match category {
                        0 => None,
                        1 => Some(claimed_by(players, seat(source), tile)),
                        2 => {
                            let winner = players
                                .iter()
                                .position(|player| player.has_won)
                                .unwrap_or(source);
                            winners[winner] = true;
                            None
                        }
                        _ => unreachable!(),
                    };
                    records.push(DiscardInfo {
                        seat: seat(source),
                        tile,
                        meld_claim,
                        winners,
                        robbed_kong: false,
                    });
                }
            }
        }
    }
    records
}

fn claimed_by(players: &[ProjectedPlayer; PLAYER_COUNT], source: Seat, tile: Tile) -> Seat {
    for (index, player) in players.iter().enumerate() {
        if player.melds.iter().any(|meld| {
            meld.tile() == tile
                && meld.from() == Some(source)
                && matches!(meld.kind(), MeldKind::Peng | MeldKind::OpenKong)
        }) {
            return seat(index);
        }
    }
    // The claimant association is intentionally omitted by the encoding. Any
    // valid claimant is enough for reconstructing player-visible physical
    // ownership; pick a deterministic adjacent seat when no meld identifies it.
    source.next_clockwise()
}

fn decode_response(state: &StateFeatures, stage: DecisionStage) -> Option<ResponseView> {
    let response_stage = matches!(
        stage,
        DecisionStage::DiscardResponse | DecisionStage::AddedKongResponse
    );
    let present = decode_flag(state.response.present, "response presence");
    if !response_stage {
        assert!(!present, "{stage:?}: response context must be absent");
        assert!(state.response.tile.iter().all(|&value| value == 0.0));
        assert!(state
            .response
            .relative_source
            .iter()
            .all(|&value| value == 0.0));
        assert_eq!(state.response.robbing_added_kong, 0.0);
        return None;
    }

    assert!(present, "{stage:?}: response context must be present");
    let tile_index = decode_one_hot(&state.response.tile, 27, "response tile");
    let source = decode_one_hot(&state.response.relative_source, 4, "response source");
    let robbing = decode_flag(state.response.robbing_added_kong, "robbing added kong flag");
    assert_eq!(
        robbing,
        stage == DecisionStage::AddedKongResponse,
        "response robbing flag must match the stage"
    );
    Some(ResponseView {
        source: seat(source),
        tile: Tile::from_index(tile_index as u8).expect("tile index in range"),
        robbing_added_kong: robbing,
    })
}

fn decode_legal(candidates: &[Candidate], own_hand: &Hand, stage: DecisionStage) -> LegalSignature {
    let actions: Vec<Action> = candidates
        .iter()
        .map(|candidate| candidate.action)
        .collect();
    let legal = match stage {
        DecisionStage::Exchange => {
            let mut eligible_suits = [false; 3];
            for action in &actions {
                let Action::Exchange(tiles) = *action else {
                    panic!("Exchange decision has a non-exchange candidate: {action:?}");
                };
                let suit = tiles[0].suit();
                assert!(tiles.iter().all(|value| value.suit() == suit));
                assert!(own_hand.contains_tiles(&tiles));
                eligible_suits[suit.index()] = true;
            }
            for suit in Suit::ALL {
                let hand_eligible = own_hand.suit_count(suit) >= 3;
                assert_eq!(
                    eligible_suits[suit.index()],
                    hand_eligible,
                    "exchange eligibility must follow hand and candidates"
                );
            }
            LegalSignature::Exchange { eligible_suits }
        }
        DecisionStage::MissingSuit => {
            let mut missing: Vec<Action> = actions
                .into_iter()
                .map(|action| match action {
                    Action::ChooseMissing(_) => action,
                    other => panic!("MissingSuit decision has unexpected candidate {other:?}"),
                })
                .collect();
            missing.sort_by_key(|action| match action {
                Action::ChooseMissing(suit) => suit.index(),
                _ => unreachable!(),
            });
            LegalSignature::Actions(missing)
        }
        DecisionStage::Turn => {
            let mut hu = false;
            let mut concealed_kongs = Vec::new();
            let mut added_kongs = Vec::new();
            let mut discards = Vec::new();
            let mut pass = false;
            for action in actions {
                match action {
                    Action::Hu => {
                        assert!(!hu, "duplicate Hu candidate");
                        hu = true;
                    }
                    Action::ConcealedKong(tile) => concealed_kongs.push(tile),
                    Action::AddedKong(tile) => added_kongs.push(tile),
                    Action::Discard(tile) => discards.push(tile),
                    Action::Pass => {
                        assert!(!pass, "duplicate Pass candidate");
                        pass = true;
                    }
                    other => panic!("Turn decision has unexpected candidate {other:?}"),
                }
            }
            concealed_kongs.sort_by_key(|tile| tile.index());
            added_kongs.sort_by_key(|tile| tile.index());
            discards.sort_by_key(|tile| tile.index());

            let mut offered = Vec::new();
            if hu {
                offered.push(Action::Hu);
            }
            offered.extend(concealed_kongs.into_iter().map(Action::ConcealedKong));
            offered.extend(added_kongs.into_iter().map(Action::AddedKong));
            offered.extend(discards.into_iter().map(Action::Discard));
            if pass {
                offered.push(Action::Pass);
            }
            LegalSignature::Actions(offered)
        }
        DecisionStage::DiscardResponse | DecisionStage::AddedKongResponse => {
            let mut hu = false;
            let mut peng = false;
            let mut open_kong = false;
            let mut pass = false;
            for action in actions {
                match action {
                    Action::Hu => {
                        assert!(!hu, "duplicate Hu candidate");
                        hu = true;
                    }
                    Action::Peng => {
                        assert!(!peng, "duplicate Peng candidate");
                        peng = true;
                    }
                    Action::OpenKong => {
                        assert!(!open_kong, "duplicate OpenKong candidate");
                        open_kong = true;
                    }
                    Action::Pass => {
                        assert!(!pass, "duplicate Pass candidate");
                        pass = true;
                    }
                    other => panic!("response decision has unexpected candidate {other:?}"),
                }
            }
            if stage == DecisionStage::AddedKongResponse {
                assert!(!peng && !open_kong, "AddedKongResponse offers only Hu/Pass");
            }
            let mut offered = Vec::new();
            if hu {
                offered.push(Action::Hu);
            }
            if peng {
                offered.push(Action::Peng);
            }
            if open_kong {
                offered.push(Action::OpenKong);
            }
            if pass {
                offered.push(Action::Pass);
            }
            LegalSignature::Actions(offered)
        }
    };
    let encoded_actions: Vec<Action> = match &legal {
        LegalSignature::Exchange { .. } => candidates
            .iter()
            .map(|candidate| candidate.action)
            .collect(),
        LegalSignature::Actions(actions) => actions.clone(),
    };
    assert_unique_actions(encoded_actions);
    legal
}

fn assert_unique_actions(actions: Vec<Action>) {
    let mut unique = HashSet::new();
    for action in &actions {
        assert!(
            unique.insert(*action),
            "duplicate semantic action {action:?}"
        );
    }
}

fn independent_exchange_oracle(hand: &Hand, eligible_suits: [bool; 3]) -> Vec<Action> {
    let mut seen = HashSet::new();
    let mut actions = Vec::new();
    for suit in Suit::ALL {
        if !eligible_suits[suit.index()] {
            continue;
        }
        let mut physical: Vec<Tile> = Vec::new();
        for tile in Tile::ALL {
            if tile.suit() != suit {
                continue;
            }
            for _ in 0..hand.count(tile) {
                physical.push(tile);
            }
        }
        for low in 0..physical.len() {
            for middle in low + 1..physical.len() {
                for high in middle + 1..physical.len() {
                    let action = canonicalize(Action::Exchange([
                        physical[low],
                        physical[middle],
                        physical[high],
                    ]));
                    if seen.insert(action) {
                        actions.push(action);
                    }
                }
            }
        }
    }
    actions.sort_by_key(|action| match *action {
        Action::Exchange(tiles) => (tiles[0].index(), tiles[1].index(), tiles[2].index()),
        _ => unreachable!("oracle emits only exchange actions"),
    });
    actions
}

fn candidate_actions(encoded: &EncodedDecision) -> Vec<Action> {
    encoded
        .candidates
        .iter()
        .map(|candidate| candidate.action)
        .collect()
}

fn canonical_action_set(actions: impl IntoIterator<Item = Action>) -> HashSet<Action> {
    actions.into_iter().map(canonicalize).collect()
}

fn assert_candidates_complete(decision: &Decision<'_>, encoded: &EncodedDecision, context: &str) {
    let candidates = candidate_actions(encoded);
    assert!(!candidates.is_empty(), "{context}: candidate list is empty");
    assert_unique_actions(candidates.clone());

    match decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            let expected =
                independent_exchange_oracle(decision.observation.own_hand, eligible_suits);
            assert_eq!(candidates, expected, "{context}: exchange oracle mismatch");
            assert_eq!(
                candidates.len(),
                canonical_action_set(candidates.clone()).len(),
                "{context}: exchange candidates are not unique"
            );
        }
        LegalActions::Actions(offered) => {
            let expected = canonical_action_set(offered.iter().copied());
            let actual = canonical_action_set(candidates.clone());
            assert_eq!(
                actual.len(),
                candidates.len(),
                "{context}: candidates are not unique"
            );
            assert_eq!(actual, expected, "{context}: ordinary membership mismatch");
        }
    }
}

fn assert_candidates_submit(
    decision: &Decision<'_>,
    game: &Game,
    encoded: &EncodedDecision,
    context: &str,
) {
    for (index, candidate) in encoded.candidates.iter().enumerate() {
        let mut fork = game.clone();
        assert!(
            fork.submit(decision.seat, candidate.action).is_ok(),
            "{context}: candidate {index} {:?} was rejected",
            candidate.action
        );
        assert_eq!(
            encoded.target_index(&candidate.action),
            Ok(index),
            "{context}: target_index does not resolve candidate {index}"
        );
    }
}

fn assert_absent_target_fails(encoded: &EncodedDecision, context: &str) {
    let absent = [
        Action::Pass,
        Action::Hu,
        Action::ChooseMissing(Suit::Wan),
        Action::Discard(Tile::ALL[0]),
        Action::Exchange([Tile::ALL[0]; 3]),
        Action::Peng,
        Action::OpenKong,
    ]
    .into_iter()
    .find(|action| encoded.target_index(action).is_err())
    .expect("a decision cannot offer every semantic action");
    match encoded.target_index(&absent) {
        Err(TargetNotCandidate { action }) => {
            assert_eq!(action, absent, "{context}: error changed the action")
        }
        Ok(index) => panic!("{context}: absent action {absent:?} resolved to {index}"),
    }
}

fn assert_candidates_stable_under_offered_permutation(
    decision: &Decision<'_>,
    encoded: &EncodedDecision,
    context: &str,
) {
    let candidates = candidate_actions(encoded);
    match decision.legal_actions {
        LegalActions::Exchange { .. } => {
            for (index, candidate) in encoded.candidates.iter().enumerate() {
                let Action::Exchange(tiles) = candidate.action else {
                    panic!("{context}: non-exchange candidate in exchange decision");
                };
                let permuted = Action::Exchange([tiles[2], tiles[0], tiles[1]]);
                assert_eq!(
                    encoded.target_index(&permuted),
                    Ok(index),
                    "{context}: exchange tile permutation is not canonical"
                );
            }
        }
        LegalActions::Actions(offered) => {
            let reversed: Vec<Action> = offered.iter().rev().copied().collect();
            let permuted = Decision {
                seat: decision.seat,
                observation: decision.observation,
                legal_actions: LegalActions::Actions(&reversed),
            };
            let reencoded = encode(&permuted);
            assert_eq!(
                reencoded.state, encoded.state,
                "{context}: permutation changed state"
            );
            assert_eq!(
                reencoded.candidates, encoded.candidates,
                "{context}: candidate order depends on offered order"
            );
        }
    }
    assert_eq!(candidates, candidate_actions(encoded));
}

fn script_choice(decision: &Decision<'_>) -> Action {
    match decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            let suit = Suit::ALL
                .into_iter()
                .find(|suit| {
                    eligible_suits[suit.index()]
                        && decision.observation.own_hand.suit_count(*suit) >= 3
                })
                .expect("an exchange triple exists");
            let mut tiles = decision
                .observation
                .own_hand
                .tiles()
                .filter(|tile| tile.suit() == suit);
            Action::Exchange([
                tiles.next().expect("three tiles"),
                tiles.next().expect("three tiles"),
                tiles.next().expect("three tiles"),
            ])
        }
        LegalActions::Actions(actions) => {
            if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
                return hu;
            }
            if decision.observation.stage == DecisionStage::Turn {
                if let Some(&added) = actions
                    .iter()
                    .find(|action| matches!(action, Action::AddedKong(_)))
                {
                    return added;
                }
                if let Some(&concealed) = actions
                    .iter()
                    .find(|action| matches!(action, Action::ConcealedKong(_)))
                {
                    return concealed;
                }
            } else if decision.observation.stage == DecisionStage::DiscardResponse {
                if let Some(&peng) = actions.iter().find(|action| matches!(action, Action::Peng)) {
                    return peng;
                }
                if let Some(&kong) = actions
                    .iter()
                    .find(|action| matches!(action, Action::OpenKong))
                {
                    return kong;
                }
            }
            actions.first().copied().unwrap_or(Action::Pass)
        }
    }
}

fn run_coverage_game(seed: u64) -> StageCoverage {
    let mut game = Game::new(seed);
    let mut coverage = StageCoverage::default();
    let mut decision_index = 0;
    // Establish the first pending decision so `next_decision` is idempotent in
    // the loop and a pre-read Debug snapshot isolates the encode call.
    let _ = game.next_decision();
    loop {
        let debug_before = format!("{game:?}");
        let fork_source = game.clone();
        let Some(decision) = game.next_decision() else {
            break;
        };
        let stage = decision.observation.stage;
        coverage.mark(stage);
        let context = format!("seed {seed} decision {decision_index} {stage:?}");
        let encoded = encode(&decision);
        coverage.mark_state(stage, &encoded);
        assert_eq!(
            encode(&decision),
            encoded,
            "{context}: repeated encode differs"
        );

        assert_candidates_complete(&decision, &encoded, &context);
        assert_candidates_submit(&decision, &fork_source, &encoded, &context);
        assert_absent_target_fails(&encoded, &context);
        assert_candidates_stable_under_offered_permutation(&decision, &encoded, &context);

        let seat = decision.seat;
        let action = script_choice(&decision);
        assert_eq!(
            debug_before,
            format!("{game:?}"),
            "{context}: encode mutated Game"
        );
        game.submit(seat, action)
            .unwrap_or_else(|error| panic!("{context}: script action rejected: {error}"));
        decision_index += 1;
        assert!(decision_index < 2_000, "{context}: game did not terminate");
    }
    assert!(game.result().is_some(), "seed {seed}: no terminal result");
    coverage
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Driver {
    Script,
    Teacher,
}

fn projected_equivalence_for_game(seed: u64, coverage: &mut StageCoverage, driver: Driver) {
    let mut game = Game::new(seed);
    let mut teacher = HeuristicPolicy::new();
    let mut decision_index = 0;
    let _ = game.next_decision();
    loop {
        let debug_before = format!("{game:?}");
        let Some(decision) = game.next_decision() else {
            break;
        };
        let stage = decision.observation.stage;
        coverage.mark(stage);
        let context = format!("seed {seed} decision {decision_index} {stage:?}");
        let original_choice = canonicalize(teacher.choose(&decision));
        let encoded = encode(&decision);
        coverage.mark_state(stage, &encoded);
        assert_eq!(
            canonicalize(teacher.choose(&decision)),
            original_choice,
            "{context}: encoding changed the teacher decision"
        );

        let projected = ProjectedDecision::from_encoded(&encoded);
        assert_eq!(
            projected.legal,
            legal_signature(&decision.legal_actions),
            "{context}: reconstructed offered pipeline differs"
        );
        let projected_decision = projected.as_decision();
        let projected_choice = canonicalize(teacher.choose(&projected_decision));
        assert_eq!(
            projected_choice, original_choice,
            "{context}: teacher projection changed the normalized choice"
        );

        match encoded.target_index(&original_choice) {
            Ok(index) => assert_eq!(encoded.candidates[index].action, original_choice),
            Err(_) => panic!("{context}: teacher chose a non-candidate {original_choice:?}"),
        }
        assert_absent_target_fails(&encoded, &context);

        let seat = decision.seat;
        let action = match driver {
            Driver::Script => script_choice(&decision),
            Driver::Teacher => original_choice,
        };
        assert_eq!(
            debug_before,
            format!("{game:?}"),
            "{context}: encode mutated Game"
        );
        game.submit(seat, action)
            .unwrap_or_else(|error| panic!("{context}: driver action rejected: {error}"));
        decision_index += 1;
        assert!(decision_index < 2_000, "{context}: game did not terminate");
    }
    assert!(game.result().is_some(), "seed {seed}: no terminal result");
}

type EncodedKey = (Vec<u32>, Vec<Action>);

#[derive(Debug)]
struct CollisionObservation {
    seed: u64,
    run: u8,
    decision_index: usize,
    debug: String,
    choice: Action,
}

#[derive(Default)]
struct CollisionTracker {
    seen: HashMap<EncodedKey, CollisionObservation>,
    /// Every recorded decision, including the deliberately repeated reads.
    records: usize,
    collisions: usize,
    /// Same seed, run, and decision index: forced by `collision_pass`
    /// recording each pending decision twice. Only reaches this bucket when
    /// the key was first seen at that same position; an earlier cross-position
    /// sighting classifies the duplicate as cross-position instead.
    repeated_encoding_collisions: usize,
    /// Same seed and decision index in a second run: deterministic replay.
    cross_run_collisions: usize,
    /// Same seed at a different decision index: one game's distinct positions
    /// share a key.
    cross_position_collisions: usize,
    /// Different seeds: independent games share a key.
    cross_seed_collisions: usize,
    /// Second reads issued by `collision_pass`, counted by the caller so the
    /// forced repetition is reported separately from discovered collisions.
    forced_repeats: usize,
}

impl CollisionTracker {
    fn record(
        &mut self,
        seed: u64,
        run: u8,
        decision_index: usize,
        debug: String,
        encoded: &EncodedDecision,
        choice: Action,
    ) {
        self.records += 1;
        let state = encoded.state.flatten();
        let key: EncodedKey = (
            state.iter().map(|value| value.to_bits()).collect(),
            encoded
                .candidates
                .iter()
                .map(|candidate| canonicalize(candidate.action))
                .collect(),
        );
        if let Some(previous) = self.seen.get(&key) {
            if previous.choice != choice {
                panic!(
                    "encoding collision with conflicting normalized teacher choices:\n\
                     first:  seed {} run {} decision {} choice {:?}\n\
                     second: seed {} run {} decision {} choice {:?}\n\
                     first decision:  {}\n\
                     second decision: {}",
                    previous.seed,
                    previous.run,
                    previous.decision_index,
                    previous.choice,
                    seed,
                    run,
                    decision_index,
                    choice,
                    previous.debug,
                    debug,
                );
            }
            self.collisions += 1;
            // The categories are mutually exclusive so the reported counts add
            // up to the collisions they describe.
            if previous.seed != seed {
                self.cross_seed_collisions += 1;
            } else if previous.decision_index != decision_index {
                self.cross_position_collisions += 1;
            } else if previous.run != run {
                self.cross_run_collisions += 1;
            } else {
                self.repeated_encoding_collisions += 1;
            }
        } else {
            self.seen.insert(
                key,
                CollisionObservation {
                    seed,
                    run,
                    decision_index,
                    debug,
                    choice,
                },
            );
        }
    }
}

fn collision_pass(seed: u64, run: u8, tracker: &mut CollisionTracker) {
    let mut game = Game::new(seed);
    let mut teacher = HeuristicPolicy::new();
    let mut decision_index = 0;
    while let Some(decision) = game.next_decision() {
        let choice = canonicalize(teacher.choose(&decision));
        let encoded = encode(&decision);
        let debug = format!("{decision:?}");
        tracker.record(seed, run, decision_index, debug.clone(), &encoded, choice);
        // Re-encoding the same pending decision is a forced repeated read; the
        // tracker counts it so the report does not present it as a finding.
        tracker.forced_repeats += 1;
        tracker.record(seed, run, decision_index, debug, &encoded, choice);

        let seat = decision.seat;
        let action = script_choice(&decision);
        game.submit(seat, action)
            .unwrap_or_else(|error| panic!("seed {seed} run {run}: {error}"));
        decision_index += 1;
        assert!(
            decision_index < 2_000,
            "seed {seed}: game did not terminate"
        );
    }
}

#[test]
fn seeded_game_covers_all_stages_and_candidates_are_submittable() {
    let coverage = run_coverage_game(ADDED_KONG_SEED);
    coverage.assert_all("coverage seed");
}

#[test]
fn exchange_oracle_covers_repeats_and_multiple_eligible_suits() {
    let own_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
        tile(Suit::Tong, 7),
        tile(Suit::Tong, 8),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 1),
    ]);
    let players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: own_hand.total(),
        melds: &[],
        missing_suit: None,
        score: 0,
        has_won: false,
    });
    let decision = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &own_hand,
            own_melds: &[],
            players,
            discards: &[],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 80,
            stage: DecisionStage::Exchange,
            response: None,
        },
        legal_actions: LegalActions::Exchange {
            eligible_suits: [true, true, true],
        },
    };
    let encoded = encode(&decision);
    let oracle = independent_exchange_oracle(&own_hand, [true, true, true]);
    assert_eq!(candidate_actions(&encoded), oracle);
    assert!(oracle.contains(&Action::Exchange([
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
    ])));
    assert!(oracle.contains(&Action::Exchange([
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
    ])));
    assert!(oracle.contains(&Action::Exchange([
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 1),
        tile(Suit::Tiao, 1),
    ])));
    assert_candidates_stable_under_offered_permutation(&decision, &encoded, "directed exchange");
}

#[test]
fn determinism_and_input_isolation_hold_across_calls_players_and_games() {
    let seed = 7;
    let other_seed = 13;
    let mut first = Game::new(seed);
    let mut second = Game::new(seed);
    let mut other = Game::new(other_seed);
    let mut decision_index = 0;
    let mut seats = [false; PLAYER_COUNT];
    let mut teacher = HeuristicPolicy::new();
    let _ = first.next_decision();
    let _ = second.next_decision();
    let _ = other.next_decision();

    loop {
        let first_before = format!("{first:?}");
        let second_before = format!("{second:?}");
        let first_decision = first.next_decision();
        let second_decision = second.next_decision();
        let other_decision = other.next_decision();

        let Some(first_decision) = first_decision else {
            assert!(
                second_decision.is_none(),
                "decision {decision_index}: repeated seeds terminated differently"
            );
            break;
        };
        let second_decision = second_decision.unwrap_or_else(|| {
            panic!("decision {decision_index}: repeated seeds terminated differently")
        });
        seats[first_decision.seat.index()] = true;
        let context = format!("seed {seed} decision {decision_index}");
        assert_eq!(
            first_decision.seat, second_decision.seat,
            "{context}: seat replay"
        );
        assert_eq!(
            first_decision.observation.stage, second_decision.observation.stage,
            "{context}: stage replay"
        );

        let encoded_first = encode(&first_decision);
        let encoded_again = encode(&first_decision);
        assert_eq!(encoded_first, encoded_again, "{context}: repeat encode");

        // Interleave an unrelated game between the two repeated reads.
        if let Some(other_decision) = other_decision {
            let other_encoded = encode(&other_decision);
            assert_eq!(other_encoded, encode(&other_decision));
        }

        let encoded_second = encode(&second_decision);
        assert_eq!(
            encoded_first, encoded_second,
            "{context}: repeated seed differs"
        );

        let teacher_choice = canonicalize(teacher.choose(&first_decision));
        let encoded_third = encode(&first_decision);
        assert_eq!(encoded_first, encoded_third);
        assert_eq!(
            canonicalize(teacher.choose(&first_decision)),
            teacher_choice,
            "{context}: teacher behavior changed"
        );

        let first_seat = first_decision.seat;
        let second_seat = second_decision.seat;
        let first_action = script_choice(&first_decision);
        let second_action = script_choice(&second_decision);
        assert_eq!(
            first_before,
            format!("{first:?}"),
            "{context}: encode mutated Game"
        );
        assert_eq!(
            second_before,
            format!("{second:?}"),
            "{context}: encode mutated Game"
        );
        first
            .submit(first_seat, first_action)
            .unwrap_or_else(|error| panic!("{context}: first {error}"));
        second
            .submit(second_seat, second_action)
            .unwrap_or_else(|error| panic!("{context}: second {error}"));
        assert_eq!(
            format!("{first:?}"),
            format!("{second:?}"),
            "{context}: replay diverged"
        );
        if let Some(other_decision) = other_decision {
            let other_seat = other_decision.seat;
            let other_action = script_choice(&other_decision);
            other
                .submit(other_seat, other_action)
                .unwrap_or_else(|error| panic!("{context}: other {error}"));
        }
        decision_index += 1;
        assert!(decision_index < 2_000, "games did not terminate");
    }

    assert!(first.result().is_some());
    assert!(second.result().is_some());
    assert_eq!(first.result(), second.result());
    assert!(
        seats.iter().all(|&seen| seen),
        "the interleaved replay did not observe every player"
    );
}

#[test]
fn teacher_projection_matches_original_choices_from_encoding_alone() {
    let mut coverage = StageCoverage::default();
    projected_equivalence_for_game(ADDED_KONG_SEED, &mut coverage, Driver::Script);
    projected_equivalence_for_game(7, &mut coverage, Driver::Script);
    // A teacher-driven trajectory reaches states selected by its own claims and
    // kongs, not just the aggressive coverage script's trajectory.
    projected_equivalence_for_game(7, &mut coverage, Driver::Teacher);
    coverage.assert_all("teacher projection seeds");
}

#[test]
fn directed_discard_kong_and_claim_choices_project_equivalently() {
    // Discard tie: both discards are tenpai with equal distance and ukeire, so
    // the fan tie-break must select the higher-fan tile regardless of the
    // offered order. The offered list stays in Game's Turn pipeline order.
    let seven_pairs = hand(&[
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
    ]);
    let source = seat(1);
    let mut discards = Vec::new();
    for kind in [
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 5),
    ] {
        for _ in 0..2 {
            discards.push(DiscardInfo {
                seat: source,
                tile: kind,
                meld_claim: None,
                winners: [false; PLAYER_COUNT],
                robbed_kong: false,
            });
        }
    }
    let players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if seat(index) == seat(0) {
            seven_pairs.total()
        } else {
            13
        },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let discard_actions = [
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Wan, 7)),
    ];
    let discard_decision = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &seven_pairs,
            own_melds: &[],
            players,
            discards: &discards,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&discard_actions),
    };
    assert_teacher_projection(
        &discard_decision,
        Action::Discard(tile(Suit::Wan, 7)),
        "discard tie",
    );

    // Pure discard tie: mirrored runs in Wan and Tong make distance, ukeire,
    // and fan exactly equal, so only the offered order decides. Game's Turn
    // pipeline lists discards by ascending tile index: Wan 9 before Tong 9.
    let mirror_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
        tile(Suit::Tong, 9),
    ]);
    let mirror_players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if index == 0 { mirror_hand.total() } else { 13 },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let pure_tie_actions = [
        Action::Discard(tile(Suit::Wan, 9)),
        Action::Discard(tile(Suit::Tong, 9)),
    ];
    let reversed_tie_actions = [
        Action::Discard(tile(Suit::Tong, 9)),
        Action::Discard(tile(Suit::Wan, 9)),
    ];
    let pure_tie = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &mirror_hand,
            own_melds: &[],
            players: mirror_players,
            discards: &[],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&pure_tie_actions),
    };
    let reversed_tie = Decision {
        legal_actions: LegalActions::Actions(&reversed_tie_actions),
        ..pure_tie
    };
    assert_offered_order_probe(
        &pure_tie,
        &reversed_tie,
        Action::Discard(tile(Suit::Wan, 9)),
        Action::Discard(tile(Suit::Tong, 9)),
        "pure discard tie",
    );

    // Kong tie: the added kong and the fourth-tile discard have equal progress,
    // so the kong must win. Game's Turn pipeline lists AddedKong before Discard.
    let melds = [Meld::peng(tile(Suit::Wan, 2), source)];
    let kong_hand = hand(&[
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 7),
        tile(Suit::Wan, 8),
        tile(Suit::Wan, 9),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
    ]);
    let kong_players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if seat(index) == seat(0) {
            kong_hand.total()
        } else {
            13
        },
        melds: if seat(index) == seat(0) { &melds } else { &[] },
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let kong_actions = [
        Action::AddedKong(tile(Suit::Wan, 2)),
        Action::Discard(tile(Suit::Wan, 2)),
    ];
    let kong_decision = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &kong_hand,
            own_melds: &melds,
            players: kong_players,
            discards: &discards,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&kong_actions),
    };
    assert_teacher_projection(
        &kong_decision,
        Action::AddedKong(tile(Suit::Wan, 2)),
        "kong tie",
    );

    // Pure kong tie: mirrored quad cores give the two concealed kongs exactly
    // equal progress and income, so the first offered kong wins. Game lists
    // concealed kongs before discards and in ascending tile order.
    let quad_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
    ]);
    let quad_players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if index == 0 { quad_hand.total() } else { 13 },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let quad_kong_actions = [
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        Action::ConcealedKong(tile(Suit::Tong, 1)),
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Wan, 2)),
        Action::Discard(tile(Suit::Wan, 3)),
        Action::Discard(tile(Suit::Wan, 4)),
        Action::Discard(tile(Suit::Tong, 1)),
        Action::Discard(tile(Suit::Tong, 2)),
        Action::Discard(tile(Suit::Tong, 3)),
        Action::Discard(tile(Suit::Tong, 4)),
    ];
    let reversed_quad_kong_actions: Vec<Action> = quad_kong_actions.iter().rev().copied().collect();
    let quad_tie = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &quad_hand,
            own_melds: &[],
            players: quad_players,
            discards: &[],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&quad_kong_actions),
    };
    let reversed_quad_tie = Decision {
        legal_actions: LegalActions::Actions(&reversed_quad_kong_actions),
        ..quad_tie
    };
    assert_offered_order_probe(
        &quad_tie,
        &reversed_quad_tie,
        Action::ConcealedKong(tile(Suit::Wan, 1)),
        Action::ConcealedKong(tile(Suit::Tong, 1)),
        "pure kong tie",
    );

    // Claim choice: the peng strictly lowers shanten and must be taken over
    // Pass. Game's response pipeline order is Hu, Peng, OpenKong, Pass.
    let claim_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 7),
        tile(Suit::Tong, 8),
        tile(Suit::Tong, 9),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 5),
    ]);
    let claim = tile(Suit::Wan, 1);
    let claim_discards = [DiscardInfo {
        seat: source,
        tile: claim,
        meld_claim: None,
        winners: [false; PLAYER_COUNT],
        robbed_kong: false,
    }];
    let claim_players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if seat(index) == seat(0) {
            claim_hand.total()
        } else {
            13
        },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let claim_actions = [Action::Peng, Action::Pass];
    let claim_decision = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &claim_hand,
            own_melds: &[],
            players: claim_players,
            discards: &claim_discards,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::DiscardResponse,
            response: Some(ResponseView {
                source,
                tile: claim,
                robbing_added_kong: false,
            }),
        },
        legal_actions: LegalActions::Actions(&claim_actions),
    };
    assert_teacher_projection(&claim_decision, Action::Peng, "claim choice");

    // Response tie: the open kong has exactly the pass hand's shanten and
    // ukeire (the concealed core keeps the same single wait), and a kong only
    // needs no-worse progress, so it wins the tie. The peng gate is stricter
    // and fails here because peng cannot strictly lower shanten.
    //
    // This fixture exercises the reachable kong-over-pass tie; the private
    // response-rule tests independently pin the peng/kong comparison helper.
    let open_kong_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 7),
        tile(Suit::Tong, 8),
        tile(Suit::Tong, 9),
    ]);
    let kong_claim = tile(Suit::Wan, 1);
    let kong_claim_discards = [DiscardInfo {
        seat: source,
        tile: kong_claim,
        meld_claim: None,
        winners: [false; PLAYER_COUNT],
        robbed_kong: false,
    }];
    let open_kong_players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if index == 0 {
            open_kong_hand.total()
        } else {
            13
        },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let open_kong_actions = [Action::Peng, Action::OpenKong, Action::Pass];
    let reversed_open_kong_actions = [Action::Pass, Action::OpenKong, Action::Peng];
    let open_kong_decision = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &open_kong_hand,
            own_melds: &[],
            players: open_kong_players,
            discards: &kong_claim_discards,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::DiscardResponse,
            response: Some(ResponseView {
                source,
                tile: kong_claim,
                robbing_added_kong: false,
            }),
        },
        legal_actions: LegalActions::Actions(&open_kong_actions),
    };
    let reversed_open_kong_decision = Decision {
        legal_actions: LegalActions::Actions(&reversed_open_kong_actions),
        ..open_kong_decision
    };
    assert_teacher_projection(&open_kong_decision, Action::OpenKong, "open kong tie");
    // The response rule is progress based, not offered-order based, so the
    // reversal probes order insensitivity instead of a second tie choice.
    assert_offered_order_probe(
        &open_kong_decision,
        &reversed_open_kong_decision,
        Action::OpenKong,
        Action::OpenKong,
        "open kong tie",
    );
}

fn assert_teacher_projection(decision: &Decision<'_>, expected: Action, context: &str) {
    let encoded = encode(decision);
    let projected = ProjectedDecision::from_encoded(&encoded);
    assert_eq!(
        projected.legal,
        legal_signature(&decision.legal_actions),
        "{context}: reconstructed pipeline differs"
    );
    let projected_decision = projected.as_decision();
    let mut teacher = HeuristicPolicy::new();
    assert_eq!(
        canonicalize(teacher.choose(decision)),
        expected,
        "{context}: fixture premise"
    );
    assert_eq!(
        canonicalize(teacher.choose(&projected_decision)),
        expected,
        "{context}: projected teacher choice differs"
    );
}

/// Asserts an offered-order probe on an otherwise identical decision.
///
/// `decision` keeps Game's pipeline order and must project equivalently. The
/// reversed list is a sensitivity probe only: if it is a pure tie, the
/// heuristic flips to `reversed_expected`, proving the original tie is real;
/// when the rule is order-insensitive both expectations coincide. In either
/// case the canonical encoding must not depend on the offered order, so the
/// reversed probe never enters the Game contract.
fn assert_offered_order_probe(
    decision: &Decision<'_>,
    reversed: &Decision<'_>,
    expected: Action,
    reversed_expected: Action,
    context: &str,
) {
    assert_teacher_projection(decision, expected, context);
    let mut teacher = HeuristicPolicy::new();
    assert_eq!(
        canonicalize(teacher.choose(reversed)),
        reversed_expected,
        "{context}: reversed offered order did not reach the probe choice"
    );

    let encoded = encode(decision);
    let reversed_encoded = encode(reversed);
    assert_eq!(
        encoded.state.flatten(),
        reversed_encoded.state.flatten(),
        "{context}: offered order leaked into state"
    );
    assert_eq!(
        encoded.candidates, reversed_encoded.candidates,
        "{context}: offered order leaked into candidates"
    );
}

#[test]
fn omitted_record_order_winners_and_meld_order_do_not_leak_into_encoding() {
    let source = seat(1);
    let own_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 7),
        tile(Suit::Wan, 8),
    ]);
    let melds_a = [
        Meld::peng(tile(Suit::Tong, 1), seat(2)),
        Meld::peng(tile(Suit::Tong, 9), source),
    ];
    let melds_b = [
        Meld::peng(tile(Suit::Tong, 9), source),
        Meld::peng(tile(Suit::Tong, 1), seat(2)),
    ];
    let mut winner_one = [false; PLAYER_COUNT];
    winner_one[2] = true;
    let mut winner_two = [false; PLAYER_COUNT];
    winner_two[3] = true;
    let records_a = [
        DiscardInfo {
            seat: source,
            tile: tile(Suit::Tiao, 1),
            meld_claim: None,
            winners: winner_one,
            robbed_kong: false,
        },
        DiscardInfo {
            seat: source,
            tile: tile(Suit::Tiao, 2),
            meld_claim: None,
            winners: winner_two,
            robbed_kong: false,
        },
        DiscardInfo {
            seat: source,
            tile: tile(Suit::Tong, 9),
            meld_claim: Some(seat(0)),
            winners: [false; PLAYER_COUNT],
            robbed_kong: false,
        },
    ];
    let mut records_b = [records_a[2], records_a[1], records_a[0]];
    // Keep the same Hu-claimed category while exchanging which omitted winner
    // association belongs to which record.
    records_b[1].winners = winner_one;
    records_b[2].winners = winner_two;

    let players_a = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if seat(index) == seat(0) {
            own_hand.total()
        } else {
            13
        },
        melds: if seat(index) == seat(0) {
            &melds_a
        } else {
            &[]
        },
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: matches!(index, 2 | 3),
    });
    let mut players_b = players_a;
    players_b[0].melds = &melds_b;
    let actions = [
        Action::Discard(tile(Suit::Wan, 1)),
        Action::Discard(tile(Suit::Wan, 2)),
        Action::Discard(tile(Suit::Wan, 3)),
        Action::Discard(tile(Suit::Wan, 4)),
        Action::Discard(tile(Suit::Wan, 5)),
        Action::Discard(tile(Suit::Wan, 6)),
        Action::Discard(tile(Suit::Wan, 7)),
        Action::Discard(tile(Suit::Wan, 8)),
    ];
    let first = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &own_hand,
            own_melds: &melds_a,
            players: players_a,
            discards: &records_a,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&actions),
    };
    let second = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &own_hand,
            own_melds: &melds_b,
            players: players_b,
            discards: &records_b,
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&actions),
    };

    assert_ne!(first.observation.discards, second.observation.discards);
    assert_ne!(first.observation.own_melds, second.observation.own_melds);

    let encoded_first = encode(&first);
    let encoded_second = encode(&second);
    assert_eq!(
        encoded_first.state.flatten(),
        encoded_second.state.flatten(),
        "record order, winner association, or meld order leaked into state"
    );
    assert_eq!(
        encoded_first.candidates, encoded_second.candidates,
        "omitted data changed candidates"
    );

    let mut teacher = HeuristicPolicy::new();
    // Compare the two original decisions first. Their omitted record order,
    // winner association, and meld order differ, so agreement is a real check
    // rather than a comparison of two projections of the same encoded state.
    let first_choice = canonicalize(teacher.choose(&first));
    let second_choice = canonicalize(teacher.choose(&second));
    assert_eq!(
        first_choice, second_choice,
        "omitted data changed the original teacher choice"
    );
    // Each original decision must also project to the same choice from its own
    // encoding; this is what the encoding contract promises downstream.
    assert_teacher_projection(&first, first_choice, "omitted-data first decision");
    assert_teacher_projection(&second, second_choice, "omitted-data second decision");
}

#[test]
fn seeded_collision_keys_report_repeated_encodings_without_conflicts() {
    let mut tracker = CollisionTracker::default();
    collision_pass(7, 0, &mut tracker);
    collision_pass(7, 1, &mut tracker);
    collision_pass(ADDED_KONG_SEED, 0, &mut tracker);
    collision_pass(ADDED_KONG_SEED, 1, &mut tracker);
    // Additional seeds sample cross-seed keys; a natural collision must still
    // agree on the normalized teacher choice.
    collision_pass(13, 0, &mut tracker);
    collision_pass(24, 0, &mut tracker);

    // Each pending decision is recorded twice. The forced second read is
    // counted separately and can never exceed the number of double records;
    // a same-position repeat is not evidence of anything stronger.
    assert_eq!(tracker.records % 2, 0, "each decision is recorded twice");
    assert_eq!(
        tracker.forced_repeats,
        tracker.records / 2,
        "the forced re-read count must match the double recording"
    );
    assert!(
        tracker.repeated_encoding_collisions <= tracker.forced_repeats,
        "a same-position repeat can only come from a forced re-read"
    );
    let classified = tracker.repeated_encoding_collisions
        + tracker.cross_run_collisions
        + tracker.cross_position_collisions
        + tracker.cross_seed_collisions;
    assert_eq!(
        classified, tracker.collisions,
        "collision categories must be mutually exclusive and exhaustive"
    );
    assert!(
        tracker.cross_run_collisions > 0,
        "a second run of the same seed must reproduce decision keys"
    );
    // Natural cross-position collisions are sampled, not guaranteed; a
    // conflict would already have panicked with both decisions and positions.
    // Report the observed categories without inflating the forced repeats.
    eprintln!(
        "collision sample: {} records ({} forced re-reads), {} duplicate keys = \
         {} same-position repeats + {} same-seed replay + {} same-seed cross-position + {} cross-seed",
        tracker.records,
        tracker.forced_repeats,
        tracker.collisions,
        tracker.repeated_encoding_collisions,
        tracker.cross_run_collisions,
        tracker.cross_position_collisions,
        tracker.cross_seed_collisions
    );
}

#[test]
fn conflicting_collision_reports_both_decisions_before_panicking() {
    // Two offered orders of the same decision encode identically, so the second
    // record is a real same-key collision. Its reversed offered order is an
    // out-of-domain probe: the differing teacher choice exercises the failure
    // path and checks that the panic carries the full counterexample.
    let own_hand = hand(&[
        tile(Suit::Wan, 1),
        tile(Suit::Wan, 2),
        tile(Suit::Wan, 3),
        tile(Suit::Wan, 4),
        tile(Suit::Wan, 5),
        tile(Suit::Wan, 6),
        tile(Suit::Wan, 9),
        tile(Suit::Tong, 1),
        tile(Suit::Tong, 2),
        tile(Suit::Tong, 3),
        tile(Suit::Tong, 4),
        tile(Suit::Tong, 5),
        tile(Suit::Tong, 6),
        tile(Suit::Tong, 9),
    ]);
    let players = std::array::from_fn(|index| PlayerView {
        seat: seat(index),
        concealed_count: if index == 0 { own_hand.total() } else { 13 },
        melds: &[],
        missing_suit: Some(Suit::Tiao),
        score: 0,
        has_won: false,
    });
    let actions = [
        Action::Discard(tile(Suit::Wan, 9)),
        Action::Discard(tile(Suit::Tong, 9)),
    ];
    let reversed_actions = [
        Action::Discard(tile(Suit::Tong, 9)),
        Action::Discard(tile(Suit::Wan, 9)),
    ];
    let first = Decision {
        seat: seat(0),
        observation: Observation {
            own_hand: &own_hand,
            own_melds: &[],
            players,
            discards: &[],
            exchange_direction: ExchangeDirection::Clockwise,
            wall_count: 20,
            stage: DecisionStage::Turn,
            response: None,
        },
        legal_actions: LegalActions::Actions(&actions),
    };
    let second = Decision {
        legal_actions: LegalActions::Actions(&reversed_actions),
        ..first
    };
    let first_encoded = encode(&first);
    let second_encoded = encode(&second);
    assert_eq!(
        first_encoded.state.flatten(),
        second_encoded.state.flatten(),
        "fixture premise: both decisions share encoded state"
    );
    assert_eq!(
        first_encoded.candidates, second_encoded.candidates,
        "fixture premise: both decisions share encoded candidates"
    );

    let first_choice = Action::Discard(tile(Suit::Wan, 9));
    let second_choice = Action::Discard(tile(Suit::Tong, 9));
    let mut teacher = HeuristicPolicy::new();
    assert_eq!(canonicalize(teacher.choose(&first)), first_choice);
    assert_eq!(canonicalize(teacher.choose(&second)), second_choice);
    assert_ne!(
        first_choice, second_choice,
        "fixture premise: the offered order must flip the pure tie"
    );
    let first_debug = format!("{first:?}");
    let second_debug = format!("{second:?}");
    let mut tracker = CollisionTracker::default();
    tracker.record(101, 0, 7, first_debug.clone(), &first_encoded, first_choice);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tracker.record(
            202,
            1,
            9,
            second_debug.clone(),
            &second_encoded,
            second_choice,
        );
    }))
    .expect_err("a same-key collision with different choices must panic");
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()))
        .expect("panic payload is a formatted string");

    for needle in [
        "seed 101 run 0 decision 7".to_string(),
        "seed 202 run 1 decision 9".to_string(),
        format!("{first_choice:?}"),
        format!("{second_choice:?}"),
        first_debug,
        second_debug,
    ] {
        assert!(
            message.contains(&needle),
            "collision report is missing {needle:?}:\n{message}"
        );
    }
}
