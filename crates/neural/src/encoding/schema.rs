//! Frozen v1 feature types, layout, scales, and label mappings.
//!
//! The items here define the tensor shapes, value scales, and category order
//! that [`super::encode`] fills. [`ENCODING_VERSION`] versions their meaning,
//! so changing one requires a new version even when the shapes are unchanged.

use sichuan_mahjong_sim::{
    Action, DecisionStage, DiscardInfo, ExchangeDirection, MeldKind, Seat, Suit, Tile,
    PLAYER_COUNT, TILE_KINDS, TOTAL_TILES,
};

/// Identifies the frozen v1 feature meanings, scales, category and field
/// order, layout, canonicalization, and candidate order.
///
/// Any change to those semantics requires a new version, even when the tensor
/// shapes are unchanged.
pub const ENCODING_VERSION: u32 = 1;

/// Width of [`StateFeatures::flatten`].
pub const STATE_WIDTH: usize = 1013;

/// Width of [`ActionFeatures::flatten`].
pub const ACTION_WIDTH: usize = 39;

/// Number of meld slots encoded per player.
pub const MELD_SLOTS: usize = 4;

pub(super) const TILE_FEATURES: usize = TILE_KINDS;
pub(super) const MISSING_SUIT_FEATURES: usize = 4;
pub(super) const MELD_KIND_FEATURES: usize = 4;
pub(super) const MELD_SOURCE_FEATURES: usize = 5;
pub(super) const MELD_SLOT_FEATURES: usize =
    1 + TILE_FEATURES + MELD_KIND_FEATURES + MELD_SOURCE_FEATURES;
pub(super) const PLAYER_FEATURES: usize =
    1 + MISSING_SUIT_FEATURES + 1 + 1 + MELD_SLOTS * MELD_SLOT_FEATURES;
pub(super) const PLAYERS_FEATURES: usize = PLAYER_COUNT * PLAYER_FEATURES;
pub(super) const DISCARD_CATEGORIES: usize = 3;
pub(super) const DISCARDS_FEATURES: usize = PLAYER_COUNT * DISCARD_CATEGORIES * TILE_FEATURES;
pub(super) const EXCHANGE_DIRECTION_FEATURES: usize = 3;
pub(super) const STAGE_FEATURES: usize = 5;
pub(super) const CONTEXT_FEATURES: usize = 1 + EXCHANGE_DIRECTION_FEATURES + STAGE_FEATURES;
pub(super) const RESPONSE_FEATURES: usize = 1 + TILE_FEATURES + PLAYER_COUNT + 1;
pub(super) const ACTION_KIND_FEATURES: usize = 9;
pub(super) const ACTION_SUIT_FEATURES: usize = 3;

/// Physical copies of one tile kind; the per-kind count divisor.
const TILE_COPIES: f32 = 4.0;
pub(super) const HAND_TOTAL_DIVISOR: f32 = 14.0;
pub(super) const SCORE_DIVISOR: f32 = 16.0;
pub(super) const WALL_DIVISOR: f32 = TOTAL_TILES as f32;

const _: () = {
    assert!(MELD_SLOTS == 4);
    assert!(TILE_FEATURES == 27);
    assert!(MELD_SLOT_FEATURES == 37);
    assert!(PLAYER_FEATURES == 155);
    assert!(PLAYERS_FEATURES == 620);
    assert!(DISCARDS_FEATURES == 324);
    assert!(CONTEXT_FEATURES == 9);
    assert!(RESPONSE_FEATURES == 33);
    assert!(
        TILE_FEATURES + PLAYERS_FEATURES + DISCARDS_FEATURES + CONTEXT_FEATURES + RESPONSE_FEATURES
            == STATE_WIDTH
    );
    assert!(ACTION_KIND_FEATURES + TILE_FEATURES + ACTION_SUIT_FEATURES == ACTION_WIDTH);
    assert!(TOTAL_TILES == 108);
};

/// Tile and meld features of one player, ordered by relative seat.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerFeatures {
    /// Concealed physical tile count divided by 14.
    pub concealed_total: f32,
    /// One-hot missing suit: Wan, Tong, Tiao, Unpublished.
    pub missing_suit: [f32; 4],
    /// Current net score divided by 16, unclipped.
    pub score: f32,
    /// 1 when the player has already won, otherwise 0.
    pub has_won: f32,
    /// Meld slots sorted by ascending tile index; unused slots are zero.
    pub melds: [MeldSlot; 4],
}

/// One declared meld, or an all-zero unused slot.
///
/// The tile vector identifies the meld's tile kind with a one-hot value. Its
/// physical tile count follows from the meld kind. The source is relative to
/// the deciding player, and `None` marks a concealed kong.
#[derive(Clone, Debug, PartialEq)]
pub struct MeldSlot {
    /// 1 when this slot holds a meld, otherwise 0.
    pub valid: f32,
    /// One-hot tile identity.
    pub tile: [f32; 27],
    /// One-hot meld kind: Peng, OpenKong, AddedKong, ConcealedKong.
    pub kind: [f32; 4],
    /// One-hot meld source relative to the deciding player: Self,
    /// NextClockwise, Opposite, PreviousClockwise, None.
    pub source: [f32; 5],
}

/// Global public context of the decision.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextFeatures {
    /// Tiles still in the wall divided by 108.
    pub wall_count: f32,
    /// One-hot exchange direction: Clockwise, CounterClockwise, Opposite.
    pub exchange_direction: [f32; 3],
    /// One-hot stage: Exchange, MissingSuit, Turn, DiscardResponse,
    /// AddedKongResponse.
    pub stage: [f32; 5],
}

/// Response context of a discard or added-kong claim.
///
/// All fields are zero when the decision has no response context. The tile
/// vector is a one-hot identity, and the source is relative to the deciding
/// player.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseFeatures {
    /// 1 when a claim response is pending, otherwise 0.
    pub present: f32,
    /// One-hot pending tile identity.
    pub tile: [f32; 27],
    /// One-hot relative source: Self, NextClockwise, Opposite,
    /// PreviousClockwise.
    pub relative_source: [f32; 4],
    /// 1 when the response robs an added kong, otherwise 0.
    pub robbing_added_kong: f32,
}

/// Player-visible state of one pending decision.
#[derive(Clone, Debug, PartialEq)]
pub struct StateFeatures {
    /// Own concealed tile counts divided by four.
    pub own_hand: [f32; 27],
    /// One entry per relative seat, self first.
    pub players: [PlayerFeatures; 4],
    /// Counts by relative source, discard category, and tile, divided by four.
    ///
    /// Every record is classified once as Unclaimed, MeldClaimed, or
    /// HuClaimed. A record claimed by one or more Hu winners is HuClaimed and
    /// still contributes a single tile, so the categories partition the
    /// discard history without double counting.
    pub discards: [[[f32; 27]; 3]; 4],
    /// Wall count, exchange direction, and stage.
    pub context: ContextFeatures,
    /// Pending claim response, zero when absent.
    pub response: ResponseFeatures,
}

/// Intent parameters of one candidate action.
///
/// Tile parameters are multiplicities divided by four: an exchange triple
/// sums to 0.75, and a discard or kong names one tile kind at 0.25. The suit
/// parameter is the chosen missing suit, otherwise zero.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionFeatures {
    /// One-hot action kind: Exchange, ChooseMissing, Discard, Peng, OpenKong,
    /// ConcealedKong, AddedKong, Hu, Pass.
    pub kind: [f32; 9],
    /// Tile parameters divided by four.
    pub tiles: [f32; 27],
    /// One-hot chosen suit for ChooseMissing, otherwise zero.
    pub suit: [f32; 3],
}

/// One semantic candidate with its features.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Canonical semantic action.
    pub action: Action,
    /// Intent parameters of the action.
    pub features: ActionFeatures,
}

impl StateFeatures {
    /// Flattens the state in the documented v1 row order.
    pub fn flatten(&self) -> [f32; STATE_WIDTH] {
        let mut out = [0.0f32; STATE_WIDTH];
        let mut cursor = 0;
        append(&mut out, &mut cursor, &self.own_hand);
        for player in &self.players {
            append(&mut out, &mut cursor, &[player.concealed_total]);
            append(&mut out, &mut cursor, &player.missing_suit);
            append(&mut out, &mut cursor, &[player.score, player.has_won]);
            for meld in &player.melds {
                append(&mut out, &mut cursor, &[meld.valid]);
                append(&mut out, &mut cursor, &meld.tile);
                append(&mut out, &mut cursor, &meld.kind);
                append(&mut out, &mut cursor, &meld.source);
            }
        }
        for by_source in &self.discards {
            for by_category in by_source {
                append(&mut out, &mut cursor, by_category);
            }
        }
        append(&mut out, &mut cursor, &[self.context.wall_count]);
        append(&mut out, &mut cursor, &self.context.exchange_direction);
        append(&mut out, &mut cursor, &self.context.stage);
        append(&mut out, &mut cursor, &[self.response.present]);
        append(&mut out, &mut cursor, &self.response.tile);
        append(&mut out, &mut cursor, &self.response.relative_source);
        append(&mut out, &mut cursor, &[self.response.robbing_added_kong]);
        debug_assert_eq!(cursor, STATE_WIDTH);
        out
    }
}

impl ActionFeatures {
    /// Flattens the action in the documented v1 row order.
    pub fn flatten(&self) -> [f32; ACTION_WIDTH] {
        let mut out = [0.0f32; ACTION_WIDTH];
        let mut cursor = 0;
        append(&mut out, &mut cursor, &self.kind);
        append(&mut out, &mut cursor, &self.tiles);
        append(&mut out, &mut cursor, &self.suit);
        debug_assert_eq!(cursor, ACTION_WIDTH);
        out
    }
}

/// One-hot index of a possibly unpublished missing suit.
pub(super) fn missing_suit_id(missing: Option<Suit>) -> usize {
    match missing {
        Some(Suit::Wan) => 0,
        Some(Suit::Tong) => 1,
        Some(Suit::Tiao) => 2,
        None => 3,
    }
}

/// One-hot index of a meld kind.
pub(super) fn meld_kind_id(kind: MeldKind) -> usize {
    match kind {
        MeldKind::Peng => 0,
        MeldKind::OpenKong => 1,
        MeldKind::AddedKong => 2,
        MeldKind::ConcealedKong => 3,
    }
}

/// One-hot index of a meld source relative to `decider`.
pub(super) fn meld_source_id(source: Option<Seat>, decider: Seat) -> usize {
    match source {
        Some(seat) => decider.clockwise_distance(seat) as usize,
        None => 4,
    }
}

/// One-hot index of an exchange direction.
pub(super) fn exchange_direction_id(direction: ExchangeDirection) -> usize {
    match direction {
        ExchangeDirection::Clockwise => 0,
        ExchangeDirection::CounterClockwise => 1,
        ExchangeDirection::Opposite => 2,
    }
}

/// Wire numeric ID of a decision stage.
///
/// Shared with the collection format, which exposes it through its stable
/// public wire API instead of owning a second mapping.
pub(crate) fn stage_code(stage: DecisionStage) -> u8 {
    match stage {
        DecisionStage::Exchange => 0,
        DecisionStage::MissingSuit => 1,
        DecisionStage::Turn => 2,
        DecisionStage::DiscardResponse => 3,
        DecisionStage::AddedKongResponse => 4,
    }
}

/// Reverses [`stage_code`].
pub(crate) fn stage_from_code(code: u8) -> Option<DecisionStage> {
    match code {
        0 => Some(DecisionStage::Exchange),
        1 => Some(DecisionStage::MissingSuit),
        2 => Some(DecisionStage::Turn),
        3 => Some(DecisionStage::DiscardResponse),
        4 => Some(DecisionStage::AddedKongResponse),
        _ => None,
    }
}

/// One-hot index of a decision stage.
pub(super) fn stage_id(stage: DecisionStage) -> usize {
    usize::from(stage_code(stage))
}

/// Discard category index. Hu claims win the classification of a record; a
/// record claimed by both is outside the valid game domain.
pub(super) fn discard_category_id(record: &DiscardInfo) -> usize {
    if record.winners.iter().any(|&won| won) {
        2
    } else if record.meld_claim.is_some() {
        1
    } else {
        0
    }
}

/// Wire numeric ID of an action kind.
///
/// Shared with the collection format, which exposes it through its stable
/// public wire API instead of owning a second mapping.
pub(crate) fn action_kind_code(action: &Action) -> u8 {
    match action {
        Action::Exchange(_) => 0,
        Action::ChooseMissing(_) => 1,
        Action::Discard(_) => 2,
        Action::Peng => 3,
        Action::OpenKong => 4,
        Action::ConcealedKong(_) => 5,
        Action::AddedKong(_) => 6,
        Action::Hu => 7,
        Action::Pass => 8,
    }
}

/// One-hot index of an action kind.
pub(super) fn action_kind_id(action: &Action) -> usize {
    usize::from(action_kind_code(action))
}

/// Chosen-suit index, or zero for actions without a suit parameter.
pub(super) fn action_suit_id(action: &Action) -> usize {
    match action {
        Action::ChooseMissing(suit) => suit.index(),
        _ => 0,
    }
}

/// Ascending exchange tiles, one specified tile, or no tile parameter.
pub(super) fn action_tile_parameters(action: &Action) -> ([Tile; 3], usize) {
    match action {
        Action::Exchange(tiles) => (*tiles, 3),
        Action::Discard(tile) | Action::ConcealedKong(tile) | Action::AddedKong(tile) => {
            ([*tile; 3], 1)
        }
        Action::ChooseMissing(_) | Action::Peng | Action::OpenKong | Action::Hu | Action::Pass => {
            ([Tile::ALL[0]; 3], 0)
        }
    }
}

/// Scales per-tile multiplicities by the physical copy count.
pub(super) fn scaled_counts(counts: &[u8; TILE_KINDS]) -> [f32; TILE_FEATURES] {
    std::array::from_fn(|index| f32::from(counts[index]) / TILE_COPIES)
}

fn append(out: &mut [f32], cursor: &mut usize, values: &[f32]) {
    let end = *cursor + values.len();
    out[*cursor..end].copy_from_slice(values);
    *cursor = end;
}
