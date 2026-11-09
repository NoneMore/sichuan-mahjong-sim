//! Canonical candidate enumeration and action features.

use std::cmp::Ordering;

use sichuan_mahjong_sim::{Action, Decision, Hand, LegalActions, Tile, TILE_KINDS};

use super::schema::{
    action_kind_id, action_suit_id, action_tile_parameters, scaled_counts, ActionFeatures,
    Candidate, ACTION_KIND_FEATURES, ACTION_SUIT_FEATURES,
};

/// Returns the canonical form used for candidate identity and label matching.
///
/// Exchange tiles are sorted by ascending tile index; every other action is
/// already canonical.
pub fn canonicalize(action: Action) -> Action {
    match action {
        Action::Exchange(mut tiles) => {
            tiles.sort_unstable();
            Action::Exchange(tiles)
        }
        other => other,
    }
}

impl ActionFeatures {
    /// One-hot kind, scaled tile parameters, and the optional chosen suit.
    ///
    /// This is the only v1 action-row construction; the collection reader
    /// validates recorded rows against it instead of owning a copy.
    pub(crate) fn new(action: Action) -> Self {
        let mut kind = [0.0f32; ACTION_KIND_FEATURES];
        kind[action_kind_id(&action)] = 1.0;

        let (parameters, len) = action_tile_parameters(&action);
        let mut copies = [0u8; TILE_KINDS];
        for &tile in &parameters[..len] {
            let slot = &mut copies[tile.index() as usize];
            *slot = slot.saturating_add(1);
        }

        let mut suit = [0.0f32; ACTION_SUIT_FEATURES];
        if let Action::ChooseMissing(chosen) = action {
            suit[chosen.index()] = 1.0;
        }

        Self {
            kind,
            tiles: scaled_counts(&copies),
            suit,
        }
    }
}

impl Candidate {
    pub(super) fn new(action: Action) -> Self {
        Self {
            action,
            features: ActionFeatures::new(action),
        }
    }
}

pub(super) fn candidate_actions(decision: &Decision<'_>) -> Vec<Action> {
    match decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            exchange_actions(decision.observation.own_hand, eligible_suits)
        }
        LegalActions::Actions(offered) => ordinary_actions(offered),
    }
}

/// Every unordered three-tile multiset of one eligible suit held by the hand,
/// in ascending tile order and therefore already canonical and sorted.
fn exchange_actions(own_hand: &Hand, eligible_suits: [bool; 3]) -> Vec<Action> {
    let mut actions = Vec::new();
    for (suit_index, &eligible) in eligible_suits.iter().enumerate() {
        if !eligible {
            continue;
        }
        let base = suit_index * 9;
        for low in 0..9 {
            for middle in low..9 {
                for high in middle..9 {
                    let tiles = [
                        Tile::ALL[base + low],
                        Tile::ALL[base + middle],
                        Tile::ALL[base + high],
                    ];
                    if own_hand.contains_tiles(&tiles) {
                        actions.push(Action::Exchange(tiles));
                    }
                }
            }
        }
    }
    actions
}

/// Canonical, deduplicated, and sorted offered actions.
fn ordinary_actions(offered: &[Action]) -> Vec<Action> {
    let mut actions: Vec<Action> = offered.iter().copied().map(canonicalize).collect();
    actions.sort_by(compare_actions);
    actions.dedup();
    actions
}

/// Candidate order: `(kind_id, tile_ids, suit_id)` with unscaled parameters.
///
/// The collection reader verifies recorded candidate lists with this same
/// comparison, so writer and reader cannot drift into different orders.
pub(crate) fn compare_actions(left: &Action, right: &Action) -> Ordering {
    let (left_tiles, left_len) = action_tile_parameters(left);
    let (right_tiles, right_len) = action_tile_parameters(right);
    action_kind_id(left)
        .cmp(&action_kind_id(right))
        .then_with(|| left_tiles[..left_len].cmp(&right_tiles[..right_len]))
        .then_with(|| action_suit_id(left).cmp(&action_suit_id(right)))
}
