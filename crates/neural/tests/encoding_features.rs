//! Black-box v1 encoding contract: frozen layout and value fixtures, every
//! action form, relative-seat mapping, and actor-known ownership recovery.
//!
//! Expected flattened vectors are written at literal v1 offsets instead of
//! being derived from the encoder's own traversal, so a layout or scale change
//! breaks these checks.

use sichuan_mahjong_neural::encoding::{
    canonicalize, encode, EncodedDecision, StateFeatures, TargetNotCandidate, ACTION_WIDTH,
    ENCODING_VERSION, MELD_SLOTS, STATE_WIDTH,
};
use sichuan_mahjong_sim::{
    Action, Decision, DecisionStage, DiscardInfo, ExchangeDirection, Hand, LegalActions, Meld,
    Observation, PlayerView, ResponseView, Seat, Suit, Tile,
};

const TILE_COUNT: usize = 27;

// Frozen v1 state layout, written independently of the encoder constants.
const OWN_HAND_OFFSET: usize = 0;
const PLAYERS_OFFSET: usize = 27;
const DISCARDS_OFFSET: usize = 647;
const CONTEXT_OFFSET: usize = 971;
const RESPONSE_OFFSET: usize = 980;

const PLAYER_STRIDE: usize = 155;
const MELD_OFFSET: usize = 7;
const MELD_STRIDE: usize = 37;
const DISCARD_SOURCE_STRIDE: usize = 81;
const DISCARD_CATEGORY_STRIDE: usize = 27;

// Frozen v1 action layout.
const KIND_OFFSET: usize = 0;
const TILES_OFFSET: usize = 9;
const SUIT_OFFSET: usize = 36;

const fn player_base(relative: usize) -> usize {
    PLAYERS_OFFSET + relative * PLAYER_STRIDE
}

const fn meld_base(relative: usize, slot: usize) -> usize {
    player_base(relative) + MELD_OFFSET + slot * MELD_STRIDE
}

const fn discard_base(relative: usize, category: usize) -> usize {
    DISCARDS_OFFSET + relative * DISCARD_SOURCE_STRIDE + category * DISCARD_CATEGORY_STRIDE
}

/// Spec tile identity: Wan, Tong, Tiao; ranks 1-9 in that order.
fn tile(suit: usize, rank: u8) -> Tile {
    let suit = match suit {
        0 => Suit::Wan,
        1 => Suit::Tong,
        2 => Suit::Tiao,
        _ => panic!("suit index out of range"),
    };
    Tile::from_suit_rank(suit, rank).expect("valid tile")
}

const fn tile_index(suit: usize, rank: u8) -> usize {
    suit * 9 + (rank as usize - 1)
}

fn seat_at(index: usize) -> Seat {
    Seat::new(index).expect("valid seat")
}

fn missing_index(missing: Option<Suit>) -> usize {
    match missing {
        Some(Suit::Wan) => 0,
        Some(Suit::Tong) => 1,
        Some(Suit::Tiao) => 2,
        None => 3,
    }
}

fn one_hot_index(values: &[f32]) -> Option<usize> {
    let mut found = None;
    for (index, &value) in values.iter().enumerate() {
        if value == 1.0 {
            assert!(found.is_none(), "multiple one-hot entries in {values:?}");
            found = Some(index);
        } else {
            assert_eq!(value, 0.0, "one-hot entries must be 0 or 1 in {values:?}");
        }
    }
    found
}

/// One player-visible snapshot with a consistent public player array.
struct Snapshot {
    seat: Seat,
    hand: Hand,
    melds: [Vec<Meld>; 4],
    concealed: [usize; 4],
    missing: [Option<Suit>; 4],
    scores: [i32; 4],
    won: [bool; 4],
    discards: Vec<DiscardInfo>,
    direction: ExchangeDirection,
    wall: usize,
    stage: DecisionStage,
    response: Option<ResponseView>,
    offered: Vec<Action>,
    eligible_suits: Option<[bool; 3]>,
}

impl Snapshot {
    fn new(seat: usize) -> Self {
        Self {
            seat: seat_at(seat),
            hand: Hand::new(),
            melds: std::array::from_fn(|_| Vec::new()),
            concealed: [0; 4],
            missing: [None; 4],
            scores: [0; 4],
            won: [false; 4],
            discards: Vec::new(),
            direction: ExchangeDirection::Clockwise,
            wall: 40,
            stage: DecisionStage::Turn,
            response: None,
            offered: vec![Action::Pass],
            eligible_suits: None,
        }
    }

    fn with_hand(mut self, tiles: &[Tile]) -> Self {
        self.hand = Hand::from_tiles(tiles).expect("valid hand");
        self.concealed[self.seat.index()] = self.hand.total();
        self
    }

    fn with_melds(mut self, seat: usize, melds: &[Meld]) -> Self {
        self.melds[seat] = melds.to_vec();
        self
    }

    fn with_concealed(mut self, seat: usize, count: usize) -> Self {
        self.concealed[seat] = count;
        self
    }

    fn with_missing(mut self, seat: usize, missing: Option<Suit>) -> Self {
        self.missing[seat] = missing;
        self
    }

    fn with_score(mut self, seat: usize, score: i32) -> Self {
        self.scores[seat] = score;
        self
    }

    fn with_win(mut self, seat: usize, won: bool) -> Self {
        self.won[seat] = won;
        self
    }

    fn with_discards(mut self, discards: &[DiscardInfo]) -> Self {
        self.discards = discards.to_vec();
        self
    }

    fn with_direction(mut self, direction: ExchangeDirection) -> Self {
        self.direction = direction;
        self
    }

    fn with_wall(mut self, wall: usize) -> Self {
        self.wall = wall;
        self
    }

    fn with_stage(mut self, stage: DecisionStage) -> Self {
        self.stage = stage;
        self
    }

    fn with_response(mut self, response: ResponseView) -> Self {
        self.response = Some(response);
        self
    }

    fn with_offered(mut self, offered: &[Action]) -> Self {
        self.offered = offered.to_vec();
        self
    }

    fn with_eligible_suits(mut self, eligible_suits: [bool; 3]) -> Self {
        self.eligible_suits = Some(eligible_suits);
        self
    }

    fn decision(&self) -> Decision<'_> {
        let players = std::array::from_fn(|index| PlayerView {
            seat: seat_at(index),
            concealed_count: self.concealed[index],
            melds: &self.melds[index],
            missing_suit: self.missing[index],
            score: self.scores[index],
            has_won: self.won[index],
        });
        Decision {
            seat: self.seat,
            observation: Observation {
                own_hand: &self.hand,
                own_melds: &self.melds[self.seat.index()],
                players,
                discards: &self.discards,
                exchange_direction: self.direction,
                wall_count: self.wall,
                stage: self.stage,
                response: self.response,
            },
            legal_actions: match self.eligible_suits {
                Some(eligible_suits) => LegalActions::Exchange { eligible_suits },
                None => LegalActions::Actions(&self.offered),
            },
        }
    }

    fn encoded(&self) -> EncodedDecision {
        encode(&self.decision())
    }
}

fn record(seat: usize, tile: Tile) -> DiscardInfo {
    DiscardInfo {
        seat: seat_at(seat),
        tile,
        meld_claim: None,
        winners: [false; 4],
        robbed_kong: false,
    }
}

fn winners(seats: &[usize]) -> [bool; 4] {
    let mut flags = [false; 4];
    for &index in seats {
        flags[index] = true;
    }
    flags
}

fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut result = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let item = rest.remove(index);
        for permutation in permutations(&rest) {
            let mut sequence = vec![item.clone()];
            sequence.extend(permutation);
            result.push(sequence);
        }
    }
    result
}

/// Physical tile counts by recovering them from the encoded public features:
/// meld multiplicities plus unclaimed and Hu-claimed records. Meld-claimed
/// records are references into a meld and contribute nothing extra.
fn public_owned_from_state(state: &StateFeatures) -> [u8; TILE_COUNT] {
    let mut owned = [0u8; TILE_COUNT];
    for player in &state.players {
        for slot in &player.melds {
            if slot.valid != 1.0 {
                continue;
            }
            let tile = one_hot_index(&slot.tile).expect("valid meld tile");
            let kind = one_hot_index(&slot.kind).expect("valid meld kind");
            let copies = if kind == 0 { 3 } else { 4 };
            owned[tile] += copies;
        }
    }
    for by_source in &state.discards {
        for category in [0usize, 2] {
            for (tile, &value) in by_source[category].iter().enumerate() {
                owned[tile] += (value * 4.0).round() as u8;
            }
        }
    }
    owned
}

/// Actor-known counts: own concealed hand plus public ownership plus the
/// single pending added-kong tile, which stays concealed by its declarer.
fn known_to_actor_from_state(state: &StateFeatures) -> [u8; TILE_COUNT] {
    let mut known = public_owned_from_state(state);
    for (tile, &value) in state.own_hand.iter().enumerate() {
        known[tile] += (value * 4.0).round() as u8;
    }
    if state.context.stage[4] == 1.0 && state.response.present == 1.0 {
        let tile = one_hot_index(&state.response.tile).expect("pending response tile");
        known[tile] += 1;
    }
    known
}

/// Independent physical public ownership from the original decision fields.
fn public_owned_from_decision(decision: &Decision<'_>) -> [u8; TILE_COUNT] {
    let mut owned = [0u8; TILE_COUNT];
    for view in &decision.observation.players {
        for meld in view.melds {
            owned[meld.tile().index() as usize] += meld.tile_count();
        }
    }
    for record in decision.observation.discards {
        let hu_claimed = record.winners.iter().any(|&won| won);
        if hu_claimed || record.meld_claim.is_none() {
            owned[record.tile.index() as usize] += 1;
        }
    }
    owned
}

fn known_to_actor_from_decision(decision: &Decision<'_>) -> [u8; TILE_COUNT] {
    let mut known = public_owned_from_decision(decision);
    for (tile, &count) in decision.observation.own_hand.counts().iter().enumerate() {
        known[tile] += count;
    }
    if decision.observation.stage == DecisionStage::AddedKongResponse {
        if let Some(response) = decision.observation.response {
            known[response.tile.index() as usize] += 1;
        }
    }
    known
}

fn assert_ownership(snapshot: &Snapshot) -> EncodedDecision {
    let decision = snapshot.decision();
    let encoded = encode(&decision);
    assert_eq!(
        public_owned_from_state(&encoded.state),
        public_owned_from_decision(&decision),
        "public ownership recovered from the encoding differs from the decision"
    );
    assert_eq!(
        known_to_actor_from_state(&encoded.state),
        known_to_actor_from_decision(&decision),
        "actor-known counts recovered from the encoding differ from the decision"
    );
    encoded
}

#[test]
fn frozen_v1_layout_and_golden_state_vector() {
    assert_eq!(ENCODING_VERSION, 1);
    assert_eq!(STATE_WIDTH, 1013);
    assert_eq!(ACTION_WIDTH, 39);
    assert_eq!(MELD_SLOTS, 4);

    // Group boundaries and strides are frozen independently of the encoder.
    assert_eq!(OWN_HAND_OFFSET + 27, PLAYERS_OFFSET);
    assert_eq!(PLAYERS_OFFSET + 4 * PLAYER_STRIDE, DISCARDS_OFFSET);
    assert_eq!(DISCARDS_OFFSET + 4 * DISCARD_SOURCE_STRIDE, CONTEXT_OFFSET);
    assert_eq!(CONTEXT_OFFSET + 9, RESPONSE_OFFSET);
    assert_eq!(RESPONSE_OFFSET + 33, STATE_WIDTH);
    assert_eq!(MELD_OFFSET + 4 * MELD_STRIDE, PLAYER_STRIDE);
    assert_eq!(3 * DISCARD_CATEGORY_STRIDE, DISCARD_SOURCE_STRIDE);

    // Decider seat 2: relative seats wrap to 2, 3, 0, 1. Meld sources are
    // relative to the decider, not to the meld owner.
    let discards = [
        // Unclaimed and Hu-claimed records, including a robbed kong.
        record(3, tile(0, 9)),
        record(2, tile(0, 1)),
        record(1, tile(2, 6)),
        record(1, tile(2, 4)),
        record(1, tile(1, 2)),
        // Meld-claimed category coordinate probe; ownership has separate fixtures.
        DiscardInfo {
            meld_claim: Some(seat_at(2)),
            ..record(0, tile(1, 3))
        },
    ];
    let mut with_multi_hu = discards;
    with_multi_hu[2].winners = winners(&[0, 3]);
    with_multi_hu[3].winners = winners(&[0]);
    with_multi_hu[3].robbed_kong = true;
    with_multi_hu[4].meld_claim = Some(seat_at(3));

    let snapshot = Snapshot::new(2)
        .with_hand(&[tile(0, 1), tile(0, 1), tile(0, 9), tile(1, 1), tile(2, 5)])
        .with_concealed(0, 13)
        .with_concealed(1, 0)
        .with_concealed(3, 7)
        .with_missing(0, None)
        .with_missing(1, Some(Suit::Tiao))
        .with_missing(2, Some(Suit::Wan))
        .with_missing(3, Some(Suit::Tong))
        .with_score(0, 0)
        .with_score(1, 16)
        .with_score(2, 48)
        .with_score(3, -100)
        .with_win(1, true)
        .with_win(3, true)
        .with_melds(
            0,
            &[
                Meld::open_kong(tile(1, 9), seat_at(1)),
                Meld::peng(tile(1, 5), seat_at(2)),
            ],
        )
        .with_melds(1, &[Meld::added_kong(tile(2, 7), seat_at(0))])
        // Supplied in descending tile order to check canonical sorting.
        .with_melds(
            2,
            &[
                Meld::concealed_kong(tile(1, 4)),
                Meld::peng(tile(0, 2), seat_at(3)),
            ],
        )
        .with_melds(
            3,
            &[
                Meld::added_kong(tile(2, 9), seat_at(0)),
                Meld::peng(tile(1, 2), seat_at(1)),
            ],
        )
        .with_discards(&with_multi_hu)
        .with_direction(ExchangeDirection::CounterClockwise)
        .with_wall(54)
        .with_stage(DecisionStage::AddedKongResponse)
        .with_response(ResponseView {
            source: seat_at(3),
            tile: tile(1, 2),
            robbing_added_kong: true,
        })
        .with_offered(&[Action::Pass]);
    let encoded = snapshot.encoded();

    let mut expected = [0.0f32; STATE_WIDTH];

    // Own hand.
    expected[OWN_HAND_OFFSET + tile_index(0, 1)] = 0.5;
    expected[OWN_HAND_OFFSET + tile_index(0, 9)] = 0.25;
    expected[OWN_HAND_OFFSET + tile_index(1, 1)] = 0.25;
    expected[OWN_HAND_OFFSET + tile_index(2, 5)] = 0.25;

    // Relative seat 0 = seat 2, the decider.
    let base = player_base(0);
    expected[base] = 5.0 / 14.0;
    expected[base + 1] = 1.0; // missing Wan
    expected[base + 5] = 3.0; // 48 / 16
    expected[meld_base(0, 0)] = 1.0;
    expected[meld_base(0, 0) + 1 + tile_index(0, 2)] = 1.0; // Peng 2Wan
    expected[meld_base(0, 0) + 28] = 1.0; // Peng
    expected[meld_base(0, 0) + 32 + 1] = 1.0; // from seat 3: next clockwise
    expected[meld_base(0, 1)] = 1.0;
    expected[meld_base(0, 1) + 1 + tile_index(1, 4)] = 1.0; // ConcealedKong 4Tong
    expected[meld_base(0, 1) + 28 + 3] = 1.0; // ConcealedKong
    expected[meld_base(0, 1) + 32 + 4] = 1.0; // None

    // Relative seat 1 = seat 3.
    let base = player_base(1);
    expected[base] = 7.0 / 14.0;
    expected[base + 1 + 1] = 1.0; // missing Tong
    expected[base + 5] = -100.0 / 16.0; // unclipped
    expected[base + 6] = 1.0; // has won
    expected[meld_base(1, 0)] = 1.0;
    expected[meld_base(1, 0) + 1 + tile_index(1, 2)] = 1.0; // Peng 2Tong
    expected[meld_base(1, 0) + 28] = 1.0; // Peng
    expected[meld_base(1, 0) + 32 + 3] = 1.0; // from seat 1: previous clockwise
    expected[meld_base(1, 1)] = 1.0;
    expected[meld_base(1, 1) + 1 + tile_index(2, 9)] = 1.0; // AddedKong 9Tiao
    expected[meld_base(1, 1) + 28 + 2] = 1.0; // AddedKong
    expected[meld_base(1, 1) + 32 + 2] = 1.0; // from seat 0: opposite

    // Relative seat 2 = seat 0.
    let base = player_base(2);
    expected[base] = 13.0 / 14.0;
    expected[base + 1 + 3] = 1.0; // missing suit unpublished
    expected[meld_base(2, 0)] = 1.0;
    expected[meld_base(2, 0) + 1 + tile_index(1, 5)] = 1.0; // Peng 5Tong
    expected[meld_base(2, 0) + 28] = 1.0; // Peng
    expected[meld_base(2, 0) + 32] = 1.0; // from seat 2: self
    expected[meld_base(2, 1)] = 1.0;
    expected[meld_base(2, 1) + 1 + tile_index(1, 9)] = 1.0; // OpenKong 9Tong
    expected[meld_base(2, 1) + 28 + 1] = 1.0; // OpenKong
    expected[meld_base(2, 1) + 32 + 3] = 1.0; // from seat 1: previous clockwise

    // Relative seat 3 = seat 1.
    let base = player_base(3);
    expected[base + 1 + 2] = 1.0; // missing Tiao
    expected[base + 5] = 1.0; // 16 / 16
    expected[base + 6] = 1.0; // has won
    expected[meld_base(3, 0)] = 1.0;
    expected[meld_base(3, 0) + 1 + tile_index(2, 7)] = 1.0; // AddedKong 7Tiao
    expected[meld_base(3, 0) + 28 + 2] = 1.0; // AddedKong
    expected[meld_base(3, 0) + 32 + 2] = 1.0; // from seat 0: opposite

    // Discards by relative source and single classification.
    expected[discard_base(0, 0) + tile_index(0, 1)] = 0.25; // own unclaimed
    expected[discard_base(1, 0) + tile_index(0, 9)] = 0.25; // seat 3 unclaimed
    expected[discard_base(2, 1) + tile_index(1, 3)] = 0.25; // meld-claimed
    expected[discard_base(3, 1) + tile_index(1, 2)] = 0.25; // meld-claimed
    expected[discard_base(3, 2) + tile_index(2, 6)] = 0.25; // multi-Hu once
    expected[discard_base(3, 2) + tile_index(2, 4)] = 0.25; // robbed kong once

    // Context and pending added-kong response.
    expected[CONTEXT_OFFSET] = 54.0 / 108.0;
    expected[CONTEXT_OFFSET + 1 + 1] = 1.0; // CounterClockwise
    expected[CONTEXT_OFFSET + 4 + 4] = 1.0; // AddedKongResponse
    expected[RESPONSE_OFFSET] = 1.0;
    expected[RESPONSE_OFFSET + 1 + tile_index(1, 2)] = 1.0;
    expected[RESPONSE_OFFSET + 1 + 27 + 1] = 1.0; // source is next clockwise
    expected[RESPONSE_OFFSET + 32] = 1.0; // robbing flag at the last index

    let flattened = encoded.state.flatten();
    let mismatches: Vec<_> = flattened
        .iter()
        .zip(expected.iter())
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .map(|(index, (actual, expected))| (index, *actual, *expected))
        .collect();
    assert!(mismatches.is_empty(), "state mismatches: {mismatches:?}");
}

#[test]
fn action_vectors_cover_all_nine_kinds_and_kong_parameters() {
    assert_eq!(KIND_OFFSET + 9, TILES_OFFSET);
    assert_eq!(TILES_OFFSET + 27, SUIT_OFFSET);
    assert_eq!(SUIT_OFFSET + 3, ACTION_WIDTH);

    let offered = [
        Action::Pass,
        Action::ChooseMissing(Suit::Tiao),
        Action::AddedKong(tile(1, 3)),
        Action::OpenKong,
        Action::Discard(tile(2, 9)),
        Action::Hu,
        Action::ConcealedKong(tile(0, 5)),
        Action::Peng,
    ];
    let snapshot = Snapshot::new(0)
        .with_hand(&[tile(2, 9)])
        .with_offered(&offered);
    let encoded = snapshot.encoded();
    assert_eq!(encoded.candidates.len(), 8);
    assert_eq!(
        encoded
            .candidates
            .iter()
            .map(|candidate| candidate.action)
            .collect::<Vec<_>>(),
        vec![
            Action::ChooseMissing(Suit::Tiao),
            Action::Discard(tile(2, 9)),
            Action::Peng,
            Action::OpenKong,
            Action::ConcealedKong(tile(0, 5)),
            Action::AddedKong(tile(1, 3)),
            Action::Hu,
            Action::Pass,
        ]
    );

    for candidate in &encoded.candidates {
        let mut expected = [0.0f32; ACTION_WIDTH];
        match candidate.action {
            Action::ChooseMissing(Suit::Tiao) => {
                expected[KIND_OFFSET + 1] = 1.0;
                expected[SUIT_OFFSET + 2] = 1.0;
            }
            Action::ChooseMissing(_) => panic!("unexpected missing suit"),
            Action::Discard(tile) => {
                assert_eq!(tile, self::tile(2, 9));
                expected[KIND_OFFSET + 2] = 1.0;
                expected[TILES_OFFSET + tile_index(2, 9)] = 0.25;
            }
            Action::Peng => expected[KIND_OFFSET + 3] = 1.0,
            Action::OpenKong => expected[KIND_OFFSET + 4] = 1.0,
            Action::ConcealedKong(tile) => {
                assert_eq!(tile, self::tile(0, 5));
                expected[KIND_OFFSET + 5] = 1.0;
                expected[TILES_OFFSET + tile_index(0, 5)] = 0.25;
            }
            Action::AddedKong(tile) => {
                assert_eq!(tile, self::tile(1, 3));
                expected[KIND_OFFSET + 6] = 1.0;
                expected[TILES_OFFSET + tile_index(1, 3)] = 0.25;
            }
            Action::Hu => expected[KIND_OFFSET + 7] = 1.0,
            Action::Pass => expected[KIND_OFFSET + 8] = 1.0,
            Action::Exchange(_) => panic!("exchange candidates come from the exchange stage"),
        }
        assert_eq!(candidate.features.flatten(), expected);
        assert_eq!(candidate.features.kind.iter().sum::<f32>(), 1.0);

        if matches!(
            candidate.action,
            Action::ConcealedKong(_) | Action::AddedKong(_)
        ) {
            // A kong names one tile kind, never four consumed copies.
            assert_eq!(candidate.features.tiles.iter().sum::<f32>(), 0.25);
            let named: Vec<usize> = candidate
                .features
                .tiles
                .iter()
                .enumerate()
                .filter(|(_, &value)| value != 0.0)
                .map(|(index, _)| index)
                .collect();
            assert_eq!(
                named,
                vec![match candidate.action {
                    Action::ConcealedKong(tile) | Action::AddedKong(tile) => tile.index() as usize,
                    _ => unreachable!(),
                }]
            );
        }
    }

    // Exchange: one non-canonical multiset from the only eligible suit.
    let exchange = Snapshot::new(0)
        .with_hand(&[tile(0, 1), tile(0, 1), tile(0, 2), tile(2, 5)])
        .with_stage(DecisionStage::Exchange)
        .with_eligible_suits([true, false, false]);
    let encoded = exchange.encoded();
    assert_eq!(encoded.candidates.len(), 1);
    let mut expected = [0.0f32; ACTION_WIDTH];
    expected[KIND_OFFSET] = 1.0;
    expected[TILES_OFFSET + tile_index(0, 1)] = 0.5;
    expected[TILES_OFFSET + tile_index(0, 2)] = 0.25;
    assert_eq!(encoded.candidates[0].features.flatten(), expected);
    assert_eq!(encoded.candidates[0].features.suit, [0.0; 3]);
}

#[test]
fn context_axes_cover_three_directions_and_five_stages() {
    let directions = [
        ExchangeDirection::Clockwise,
        ExchangeDirection::CounterClockwise,
        ExchangeDirection::Opposite,
    ];
    let stages = [
        DecisionStage::Exchange,
        DecisionStage::MissingSuit,
        DecisionStage::Turn,
        DecisionStage::DiscardResponse,
        DecisionStage::AddedKongResponse,
    ];

    for (direction_index, &direction) in directions.iter().enumerate() {
        for (stage_index, &stage) in stages.iter().enumerate() {
            let mut snapshot = Snapshot::new(1)
                .with_hand(&[tile(0, 1), tile(0, 1), tile(0, 2)])
                .with_direction(direction)
                .with_wall(27)
                .with_stage(stage);
            if stage == DecisionStage::Exchange {
                snapshot = snapshot.with_eligible_suits([true, false, false]);
            } else if matches!(
                stage,
                DecisionStage::DiscardResponse | DecisionStage::AddedKongResponse
            ) {
                snapshot = snapshot.with_response(ResponseView {
                    source: seat_at(2),
                    tile: tile(1, 5),
                    robbing_added_kong: stage == DecisionStage::AddedKongResponse,
                });
            }

            let flattened = snapshot.encoded().state.flatten();
            let mut expected = [0.0f32; 9];
            expected[0] = 27.0 / 108.0;
            expected[1 + direction_index] = 1.0;
            expected[4 + stage_index] = 1.0;
            assert_eq!(&flattened[CONTEXT_OFFSET..RESPONSE_OFFSET], &expected[..]);

            if snapshot.response.is_none() {
                assert!(flattened[RESPONSE_OFFSET..]
                    .iter()
                    .all(|&value| value == 0.0));
            }
        }
    }
}

#[test]
fn relative_seats_wrap_and_winning_players_keep_their_slots() {
    let melds = [
        [Meld::peng(tile(1, 5), seat_at(3))],
        [Meld::concealed_kong(tile(0, 9))],
        [Meld::added_kong(tile(1, 4), seat_at(0))],
        [Meld::peng(tile(0, 2), seat_at(2))],
    ];
    let snapshot = Snapshot::new(3)
        .with_hand(&[tile(2, 1)])
        .with_concealed(0, 13)
        .with_concealed(1, 10)
        .with_concealed(2, 7)
        .with_missing(0, Some(Suit::Tong))
        .with_missing(1, None)
        .with_missing(2, Some(Suit::Wan))
        .with_missing(3, Some(Suit::Tiao))
        .with_score(0, 160)
        .with_score(1, -100)
        .with_score(2, 7)
        .with_score(3, -1)
        .with_win(1, true)
        .with_win(2, true)
        .with_melds(0, &melds[0])
        .with_melds(1, &melds[1])
        .with_melds(2, &melds[2])
        .with_melds(3, &melds[3]);
    let encoded = snapshot.encoded();
    let state = &encoded.state;

    let concealed = [13usize, 10, 7, 1];
    let missing = [Some(Suit::Tong), None, Some(Suit::Wan), Some(Suit::Tiao)];
    let scores = [160, -100, 7, -1];
    let won = [false, true, true, false];
    for (relative, player) in state.players.iter().enumerate() {
        let absolute = (3 + relative) % 4;
        assert_eq!(
            player.concealed_total,
            concealed[absolute] as f32 / 14.0,
            "relative seat {relative} must be absolute seat {absolute}"
        );
        let mut expected_missing = [0.0; 4];
        expected_missing[missing_index(missing[absolute])] = 1.0;
        assert_eq!(player.missing_suit, expected_missing);
        assert_eq!(player.score, scores[absolute] as f32 / 16.0);
        assert_eq!(player.has_won, if won[absolute] { 1.0 } else { 0.0 });
    }
    // Scores are not clipped and an unpublished missing suit has its own index.
    assert_eq!(state.players[1].score, 10.0);
    assert_eq!(state.players[2].score, -6.25);
    assert_eq!(state.players[2].missing_suit, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(state.own_hand[tile_index(2, 1)], 0.25);

    let expected_melds = [
        (0, tile_index(0, 2), 0, 3), // decider's peng from previous clockwise
        (1, tile_index(1, 5), 0, 0), // next player's peng claimed from decider
        (2, tile_index(0, 9), 3, 4), // opposite player's concealed kong
        (3, tile_index(1, 4), 2, 1), // previous player's added kong from next
    ];
    for (relative, tile, kind, source) in expected_melds {
        let slot = &state.players[relative].melds[0];
        assert_eq!(slot.valid, 1.0);
        assert_eq!(one_hot_index(&slot.tile), Some(tile));
        assert_eq!(one_hot_index(&slot.kind), Some(kind));
        assert_eq!(one_hot_index(&slot.source), Some(source));
        assert!(state.players[relative].melds[1..]
            .iter()
            .all(|slot| slot.valid == 0.0));
    }
    assert!(state
        .discards
        .iter()
        .flatten()
        .flatten()
        .all(|&value| value == 0.0));
}

#[test]
fn meld_slots_sort_by_tile_and_cover_every_kind_and_source() {
    let decider = 1;
    let snapshot = Snapshot::new(decider)
        .with_hand(&[tile(2, 1)])
        .with_melds(
            decider,
            &[
                Meld::added_kong(tile(2, 9), seat_at(0)),
                Meld::peng(tile(0, 1), seat_at(2)),
                Meld::concealed_kong(tile(1, 5)),
                Meld::open_kong(tile(0, 9), seat_at(3)),
            ],
        )
        .with_concealed(decider, 1);
    let encoded = snapshot.encoded();
    let slots = &encoded.state.players[0].melds;

    let expected = [
        (tile_index(0, 1), 0, 1), // Peng from seat 2: next clockwise
        (tile_index(0, 9), 1, 2), // OpenKong from seat 3: opposite
        (tile_index(1, 5), 3, 4), // ConcealedKong: None
        (tile_index(2, 9), 2, 3), // AddedKong from seat 0: previous
    ];
    for (slot, (tile, kind, source)) in slots.iter().zip(expected) {
        assert_eq!(slot.valid, 1.0);
        assert_eq!(one_hot_index(&slot.tile), Some(tile));
        assert_eq!(one_hot_index(&slot.kind), Some(kind));
        assert_eq!(one_hot_index(&slot.source), Some(source));
    }
}

#[test]
fn ownership_handles_meld_claims_multi_hu_and_ordinary_responses() {
    // A meld-claimed record is only a reference: the three physical tiles live
    // in the peng, so the same tile kind must not be counted twice.
    let mut claimed = record(1, tile(0, 1));
    claimed.meld_claim = Some(seat_at(0));
    let snapshot = Snapshot::new(0)
        .with_hand(&[tile(0, 1)])
        .with_melds(0, &[Meld::peng(tile(0, 1), seat_at(1))])
        .with_concealed(1, 10)
        .with_discards(&[claimed]);
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    let index = tile_index(0, 1);
    assert_eq!(public_owned_from_state(state)[index], 3);
    assert_eq!(known_to_actor_from_state(state)[index], 4);
    assert_eq!(state.discards[1][1][index], 0.25);
    assert_eq!(state.discards[1][0][index], 0.0);
    assert_eq!(state.discards[1][2][index], 0.0);

    // A record claimed by several Hu winners contributes one tile; winners'
    // concealed totals exclude the virtual winning copy.
    let mut multi_hu = record(0, tile(1, 3));
    multi_hu.winners = winners(&[1, 3]);
    let snapshot = Snapshot::new(2)
        .with_concealed(0, 13)
        .with_concealed(1, 13)
        .with_concealed(3, 10)
        .with_win(1, true)
        .with_win(3, true)
        .with_discards(&[multi_hu]);
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    let index = tile_index(1, 3);
    assert_eq!(public_owned_from_state(state)[index], 1);
    assert_eq!(known_to_actor_from_state(state)[index], 1);
    assert_eq!(state.discards[2][2][index], 0.25);
    assert_eq!(state.discards[2][0][index], 0.0);
    assert_eq!(state.players[3].concealed_total, 13.0 / 14.0);
    assert_eq!(state.players[1].concealed_total, 10.0 / 14.0);

    // An ordinary response tile is already in the discard record; it must not
    // be counted again as a pending added-kong tile.
    let snapshot = Snapshot::new(1)
        .with_hand(&[tile(2, 5)])
        .with_discards(&[record(0, tile(2, 5))])
        .with_stage(DecisionStage::DiscardResponse)
        .with_response(ResponseView {
            source: seat_at(0),
            tile: tile(2, 5),
            robbing_added_kong: false,
        });
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    assert_eq!(state.context.stage, [0.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(state.response.present, 1.0);
    assert_eq!(state.response.robbing_added_kong, 0.0);
    assert_eq!(known_to_actor_from_state(state)[tile_index(2, 5)], 2);
}

#[test]
fn ownership_handles_pending_upgrade_and_robbed_added_kongs() {
    // Pending added kong: the meld is still a Peng, the fourth tile is only in
    // the response and stays concealed by the declarer, yet is publicly known.
    let mut claimed = record(0, tile(1, 6));
    claimed.meld_claim = Some(seat_at(3));
    let snapshot = Snapshot::new(1)
        .with_melds(3, &[Meld::peng(tile(1, 6), seat_at(0))])
        .with_concealed(3, 4)
        .with_discards(&[claimed])
        .with_stage(DecisionStage::AddedKongResponse)
        .with_response(ResponseView {
            source: seat_at(3),
            tile: tile(1, 6),
            robbing_added_kong: true,
        });
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    let index = tile_index(1, 6);
    let declarer = &state.players[2];
    assert_eq!(one_hot_index(&declarer.melds[0].kind), Some(0));
    assert_eq!(one_hot_index(&declarer.melds[0].tile), Some(index));
    assert_eq!(one_hot_index(&declarer.melds[0].source), Some(3));
    assert_eq!(declarer.concealed_total, 4.0 / 14.0);
    assert_eq!(state.context.stage, [0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(one_hot_index(&state.response.tile), Some(index));
    assert_eq!(state.response.present, 1.0);
    assert_eq!(state.response.robbing_added_kong, 1.0);
    assert_eq!(public_owned_from_state(state)[index], 3);
    assert_eq!(known_to_actor_from_state(state)[index], 4);

    // Successful upgrade: the AddedKong meld owns all four copies and the old
    // claim record adds nothing.
    let mut claimed = record(1, tile(2, 7));
    claimed.meld_claim = Some(seat_at(2));
    let snapshot = Snapshot::new(0)
        .with_concealed(1, 10)
        .with_melds(2, &[Meld::added_kong(tile(2, 7), seat_at(1))])
        .with_concealed(2, 2)
        .with_discards(&[claimed]);
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    let index = tile_index(2, 7);
    let upgraded = &state.players[2].melds[0];
    assert_eq!(upgraded.valid, 1.0);
    assert_eq!(one_hot_index(&upgraded.tile), Some(index));
    assert_eq!(one_hot_index(&upgraded.kind), Some(2));
    assert_eq!(one_hot_index(&upgraded.source), Some(1));
    assert_eq!(public_owned_from_state(state)[index], 4);
    assert_eq!(known_to_actor_from_state(state)[index], 4);
    assert_eq!(state.response.present, 0.0);

    // Resolved robbed kong: the record is Hu-claimed once and the virtual
    // winning tile is not concealed by the winner.
    let mut robbed = record(1, tile(0, 8));
    robbed.winners = winners(&[3]);
    robbed.robbed_kong = true;
    let snapshot = Snapshot::new(0)
        .with_melds(1, &[Meld::peng(tile(0, 8), seat_at(2))])
        .with_concealed(1, 4)
        .with_concealed(3, 13)
        .with_win(3, true)
        .with_discards(&[robbed]);
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    let index = tile_index(0, 8);
    assert_eq!(public_owned_from_state(state)[index], 4);
    assert_eq!(known_to_actor_from_state(state)[index], 4);
    assert_eq!(state.discards[1][2][index], 0.25);
    assert_eq!(state.discards[1][0][index], 0.0);
    assert_eq!(state.players[3].concealed_total, 13.0 / 14.0);

    // Pending rob attempt: the response carries the robbing flag and the
    // pending tile is known without being public.
    let snapshot = Snapshot::new(0)
        .with_melds(1, &[Meld::peng(tile(0, 8), seat_at(2))])
        .with_concealed(1, 4)
        .with_stage(DecisionStage::AddedKongResponse)
        .with_response(ResponseView {
            source: seat_at(1),
            tile: tile(0, 8),
            robbing_added_kong: true,
        });
    let encoded = assert_ownership(&snapshot);
    let state = &encoded.state;
    assert_eq!(state.response.present, 1.0);
    assert_eq!(state.response.robbing_added_kong, 1.0);
    assert_eq!(state.response.relative_source, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(public_owned_from_state(state)[index], 3);
    assert_eq!(known_to_actor_from_state(state)[index], 4);
}

#[test]
fn ownership_recovers_four_physical_copies_for_every_kong_kind() {
    let mut open_claim = record(0, tile(1, 3));
    open_claim.meld_claim = Some(seat_at(1));
    let mut added_claim = record(1, tile(0, 2));
    added_claim.meld_claim = Some(seat_at(3));
    let snapshot = Snapshot::new(0)
        .with_hand(&[tile(0, 9)])
        .with_melds(1, &[Meld::open_kong(tile(1, 3), seat_at(0))])
        .with_melds(2, &[Meld::concealed_kong(tile(2, 5))])
        .with_melds(3, &[Meld::added_kong(tile(0, 2), seat_at(1))])
        .with_concealed(1, 10)
        .with_concealed(2, 10)
        .with_concealed(3, 10)
        .with_discards(&[open_claim, added_claim]);
    let encoded = assert_ownership(&snapshot);
    let public = public_owned_from_state(&encoded.state);
    let known = known_to_actor_from_state(&encoded.state);
    for index in [tile_index(1, 3), tile_index(2, 5), tile_index(0, 2)] {
        assert_eq!(public[index], 4);
        assert_eq!(known[index], 4);
    }
    assert_eq!(public.iter().sum::<u8>(), 12);
    assert_eq!(known[tile_index(0, 9)], 1);
}

#[test]
fn offered_permutations_yield_identical_candidates_and_targets() {
    let offered = [
        Action::ChooseMissing(Suit::Tiao),
        Action::Peng,
        Action::ChooseMissing(Suit::Wan),
        Action::Discard(tile(0, 2)),
        Action::ChooseMissing(Suit::Tong),
        Action::Discard(tile(0, 1)),
    ];
    let expected_actions = vec![
        Action::ChooseMissing(Suit::Wan),
        Action::ChooseMissing(Suit::Tong),
        Action::ChooseMissing(Suit::Tiao),
        Action::Discard(tile(0, 1)),
        Action::Discard(tile(0, 2)),
        Action::Peng,
    ];

    let mut baseline = None;
    for permutation in permutations(&offered) {
        let encoded = Snapshot::new(2)
            .with_hand(&[tile(0, 1), tile(0, 2)])
            .with_offered(&permutation)
            .encoded();
        match &baseline {
            None => baseline = Some(encoded),
            Some(base) => assert_eq!(&encoded, base),
        }
    }
    let baseline = baseline.expect("at least one permutation");
    let actions: Vec<Action> = baseline
        .candidates
        .iter()
        .map(|candidate| candidate.action)
        .collect();
    assert_eq!(actions, expected_actions);

    let mut targets = Vec::new();
    for (index, action) in expected_actions.iter().enumerate() {
        let target = baseline.target_index(action).expect("candidate present");
        assert_eq!(target, index);
        targets.push(target);
    }
    targets.sort_unstable();
    targets.dedup();
    assert_eq!(targets.len(), expected_actions.len());

    let absent = Action::ConcealedKong(tile(0, 1));
    assert_eq!(
        baseline.target_index(&absent),
        Err(TargetNotCandidate { action: absent })
    );
}

#[test]
fn exchange_triple_permutations_share_one_canonical_candidate() {
    let hand = [
        tile(0, 1),
        tile(0, 1),
        tile(0, 2),
        tile(0, 3),
        tile(1, 1),
        tile(1, 2),
        tile(1, 3),
        tile(2, 9),
    ];
    let encoded = Snapshot::new(0)
        .with_hand(&hand)
        .with_stage(DecisionStage::Exchange)
        .with_eligible_suits([true, true, false])
        .encoded();

    let expected_actions = vec![
        Action::Exchange([tile(0, 1), tile(0, 1), tile(0, 2)]),
        Action::Exchange([tile(0, 1), tile(0, 1), tile(0, 3)]),
        Action::Exchange([tile(0, 1), tile(0, 2), tile(0, 3)]),
        Action::Exchange([tile(1, 1), tile(1, 2), tile(1, 3)]),
    ];
    let actions: Vec<Action> = encoded
        .candidates
        .iter()
        .map(|candidate| candidate.action)
        .collect();
    assert_eq!(actions, expected_actions);

    for candidate in &encoded.candidates {
        let Action::Exchange(tiles) = candidate.action else {
            panic!("exchange candidates only");
        };
        assert_eq!(canonicalize(candidate.action), candidate.action);
        let expected_target = encoded
            .target_index(&candidate.action)
            .expect("candidate present");
        for permutation in permutations(&tiles) {
            let permuted = Action::Exchange([permutation[0], permutation[1], permutation[2]]);
            assert_eq!(canonicalize(permuted), candidate.action);
            assert_eq!(encoded.target_index(&permuted), Ok(expected_target));
        }
    }

    let absent = Action::Exchange([tile(2, 9), tile(2, 9), tile(2, 9)]);
    assert_eq!(
        encoded.target_index(&absent),
        Err(TargetNotCandidate { action: absent })
    );
}
