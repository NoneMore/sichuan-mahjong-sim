//! Pure wall-exhaustion settlement facts and kong-income refund conditions.

use super::{FAN_CAP, FLOWER_PIG_PENALTY};
use crate::seat::Seat;

/// A single point transfer between two seats.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Transfer {
    /// Paying seat.
    pub from: Seat,
    /// Receiving seat.
    pub to: Seat,
    /// Points moved.
    pub points: u32,
}

/// Per-player facts required for end-of-wall settlement.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WallExhaustionEntry {
    /// Player seat.
    pub seat: Seat,
    /// Whether the player already won and left the hand.
    pub is_hu: bool,
    /// Whether the player still held missing-suit tiles when the wall emptied.
    pub is_flower_pig: bool,
    /// Whether the player is one tile from a winning shape.
    pub is_tenpai: bool,
    /// Maximum hypothetical discard-win fan for this player's ready hand.
    /// Ignored for players who are not tenpai.
    pub max_discard_fan: u8,
}

fn is_pig(entry: &WallExhaustionEntry) -> bool {
    !entry.is_hu && entry.is_flower_pig
}

/// Computes flower-pig and big-call transfers at wall exhaustion.
///
/// Flower pigs pay [`FLOWER_PIG_PENALTY`] to every non-flower-pig player,
/// including players who already won. The big-call check only involves players
/// who neither won nor are flower pigs: each non-tenpai player pays each tenpai
/// player that tenpai player's maximum hypothetical discard-win fan.
///
/// Kong refunds are not included because they depend on the per-payment kong
/// ledger; use [`must_refund_kong_income`] to identify refunding players.
/// Entries are expected to contain each seat at most once.
pub fn settle_wall_exhaustion(entries: &[WallExhaustionEntry]) -> Vec<Transfer> {
    let mut transfers = Vec::new();

    let non_pigs: Vec<&WallExhaustionEntry> =
        entries.iter().filter(|entry| !is_pig(entry)).collect();
    for pig in entries.iter().filter(|entry| is_pig(entry)) {
        for other in &non_pigs {
            if pig.seat != other.seat {
                transfers.push(Transfer {
                    from: pig.seat,
                    to: other.seat,
                    points: FLOWER_PIG_PENALTY,
                });
            }
        }
    }

    let call_players: Vec<&WallExhaustionEntry> = entries
        .iter()
        .filter(|entry| !entry.is_hu && !is_pig(entry))
        .collect();
    let ready: Vec<&WallExhaustionEntry> = call_players
        .iter()
        .copied()
        .filter(|entry| entry.is_tenpai && entry.max_discard_fan > 0)
        .collect();
    for waiting_loser in call_players
        .iter()
        .copied()
        .filter(|entry| !entry.is_tenpai)
    {
        for winner in &ready {
            if waiting_loser.seat != winner.seat {
                transfers.push(Transfer {
                    from: waiting_loser.seat,
                    to: winner.seat,
                    points: (winner.max_discard_fan.min(FAN_CAP)) as u32,
                });
            }
        }
    }

    transfers
}

/// Whether a player must return all kong income at wall exhaustion.
///
/// Refunds apply to a player who did not win and is either a flower pig or not
/// tenpai. The caller reverses the recorded kong ledger to the original payers.
pub fn must_refund_kong_income(is_hu: bool, is_flower_pig: bool, is_tenpai: bool) -> bool {
    !is_hu && (is_flower_pig || !is_tenpai)
}

