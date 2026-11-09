//! Versioned v1 decision encoding for neural-policy learning.
//!
//! [`encode`] turns one valid pending [`Decision`] into a structured state and
//! a canonical, nonempty list of semantic [`Candidate`]s. The encoder is
//! deterministic and stateless: it reads player-visible fields only, never
//! mutates the game, and uses no rules analysis, teacher data, or simulated
//! transitions.
//!
//! # Example
//!
//! ```
//! use sichuan_mahjong_neural::encoding::{encode, ENCODING_VERSION, STATE_WIDTH};
//! use sichuan_mahjong_sim::{Game, HeuristicPolicy, Policy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut game = Game::new(7);
//! let decision = game.next_decision().expect("opening exchange");
//! let encoded = encode(&decision);
//! assert_eq!(encoded.state.flatten().len(), STATE_WIDTH);
//! assert_eq!(ENCODING_VERSION, 1);
//!
//! // Label the original decision, not a reordered candidate list.
//! let target = HeuristicPolicy::new().choose(&decision);
//! let index = encoded.target_index(&target)?;
//! let seat = decision.seat;
//! game.submit(seat, encoded.candidates[index].action)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Seats and layout
//!
//! Every seat reference is a clockwise distance from the deciding player:
//! self = 0, next clockwise = 1, opposite = 2, previous clockwise = 3. Won
//! players keep their fixed positions.
//!
//! [`StateFeatures::flatten`] concatenates the groups below, with the last
//! array axis varying fastest:
//!
//! | Group | Width | Fields |
//! | --- | --- | --- |
//! | Own hand | 27 | Concealed tile counts |
//! | Players | 4 × 155 | Per relative seat: concealed total, missing suit (4), score, has-won, four meld slots (4 × 37) |
//! | Discards | 4 × 3 × 27 | Counts by relative source, category, and tile |
//! | Context | 9 | Wall count, exchange direction (3), stage (5) |
//! | Response | 33 | Presence, tile, relative source (4), robbing-added-kong |
//!
//! A meld slot is validity, tile (27), kind (4), source (5). Meld slots are
//! sorted by ascending tile index and zero-filled when unused, so own melds
//! appear once, in the deciding player's group. When no response context
//! exists, all 33 response fields are zero.
//!
//! # Scales
//!
//! Values are `f32`. Per-kind counts (hand, discards, action parameters)
//! divide by four, concealed totals by 14, the wall count by 108, and scores
//! by 16 without clipping. Meld and response tile identities are one-hot,
//! as are categories; flags are 0 or 1.
//!
//! Physical visibility counts can be derived without additional features:
//! `public_owned` is the sum of meld multiplicities and unclaimed or Hu-claimed
//! records. Meld-claimed records add no ownership, and a multi-Hu record adds
//! one tile. `known_to_actor` adds the own concealed hand and, only during
//! AddedKongResponse, one pending response tile. That tile is public but still
//! in the declarer's concealed hand while the meld remains a peng; after
//! resolution it belongs to the added kong or a Hu-claimed robbed-kong record.
//!
//! # Category order
//!
//! These are fixed v1 mappings, independent of enum declaration order:
//!
//! | Category | Order |
//! | --- | --- |
//! | Missing suit | Wan, Tong, Tiao, Unpublished |
//! | Meld kind | Peng, OpenKong, AddedKong, ConcealedKong |
//! | Meld source | Self, NextClockwise, Opposite, PreviousClockwise, None |
//! | Exchange direction | Clockwise, CounterClockwise, Opposite |
//! | Stage | Exchange, MissingSuit, Turn, DiscardResponse, AddedKongResponse |
//! | Action kind | Exchange, ChooseMissing, Discard, Peng, OpenKong, ConcealedKong, AddedKong, Hu, Pass |
//! | Discard category | Unclaimed, MeldClaimed, HuClaimed |
//!
//! A meld source is relative to the deciding player, not to the meld owner.
//! Concealed kongs use the `None` source.
//!
//! # Candidates
//!
//! Ordinary candidates are exactly the offered actions. Exchange candidates
//! are every unordered three-tile multiset of an eligible suit that the own
//! hand holds. Candidates are canonicalized, deduplicated, and sorted by
//! `(kind_id, tile_ids, suit_id)` with unscaled integer parameters, so the
//! order is independent of the offered order. Position is a local index, not a
//! durable action identity; [`EncodedDecision::target_index`] resolves a
//! semantic action, for example a teacher choice, to its position.

use std::fmt;

use sichuan_mahjong_sim::{Action, Decision};

mod candidates;
mod schema;
mod state;

#[cfg(test)]
mod tests;

pub use candidates::canonicalize;
pub use schema::{
    ActionFeatures, Candidate, ContextFeatures, MeldSlot, PlayerFeatures, ResponseFeatures,
    StateFeatures, ACTION_WIDTH, ENCODING_VERSION, MELD_SLOTS, STATE_WIDTH,
};

/// Crate-internal v1 mappings and canonical action order shared with the
/// collection format.
///
/// The collection layer exposes these through its own stable public wire API;
/// they are not part of the encoder's public surface.
pub(crate) use candidates::compare_actions;
pub(crate) use schema::{action_kind_code, stage_code, stage_from_code};

use candidates::candidate_actions;
use state::encode_state;

/// Encoded state and legal candidates of one pending decision.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedDecision {
    /// Player-visible state features.
    pub state: StateFeatures,
    /// Nonempty canonical candidate list for a valid pending decision.
    pub candidates: Vec<Candidate>,
}

/// An action that is not a candidate of an encoded decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetNotCandidate {
    /// The queried action, unchanged.
    pub action: Action,
}

impl fmt::Display for TargetNotCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "action {:?} is not a candidate of the encoded decision",
            self.action
        )
    }
}

impl std::error::Error for TargetNotCandidate {}

impl EncodedDecision {
    /// Encodes a pending decision; the same result as [`encode`].
    pub fn encode(decision: &Decision<'_>) -> Self {
        encode(decision)
    }

    /// Position of the canonical form of `action` in [`Self::candidates`].
    ///
    /// Teacher choices are normalized through [`canonicalize`] and matched by
    /// semantic identity, so the order of the offered list does not matter.
    /// The error carries the queried action unchanged.
    pub fn target_index(&self, action: &Action) -> Result<usize, TargetNotCandidate> {
        let canonical = canonicalize(*action);
        self.candidates
            .iter()
            .position(|candidate| candidate.action == canonical)
            .ok_or(TargetNotCandidate { action: *action })
    }
}

/// Encodes one valid pending decision.
///
/// # Panics
///
/// Panics only if the observation does not cover all four seats, which cannot
/// happen for a decision produced by [`sichuan_mahjong_sim::Game`]. Fabricated
/// decisions with contradictory fields are outside the supported contract.
pub fn encode(decision: &Decision<'_>) -> EncodedDecision {
    EncodedDecision {
        state: encode_state(decision),
        candidates: candidate_actions(decision)
            .into_iter()
            .map(Candidate::new)
            .collect(),
    }
}
