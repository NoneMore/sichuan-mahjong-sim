//! Legal-only policies for the fixed baseline.
//!
//! Policies observe only [`Decision`]: their own concealed hand, public
//! information exposed by the game, and the legality result for the current
//! choice. They never receive a `&Game`, another concealed hand, or the wall
//! order.
//!
//! Two legal baselines, an analysis policy, and a missing-suit ablation are
//! provided:
//!
//! - [`FirstLegalPolicy`] is deterministic. It always prefers `Hu` and
//!   otherwise takes the first action listed as legal by the game. Exchange
//!   tiles are collected directly from one eligible suit instead of
//!   enumerating triples.
//! - [`RandomPolicy`] owns its own seedable RNG. It always prefers `Hu` (to
//!   exercise wins), and otherwise samples uniformly from the listed legal
//!   actions or from the eligible exchange tiles.
//! - [`HeuristicPolicy`] is deterministic and ranks legal post-discard hands by
//!   exact target-deficit shanten, then unseen improving copies, then a tenpai
//!   fan tie-break. Its analysis is private and local to each decision.
//! - [`MissingOnlyPolicy`] is the missing-suit-only ablation: it uses exactly
//!   the [`FirstLegalPolicy`] choice everywhere except the missing-suit
//!   declaration, where it uses the [`HeuristicPolicy`] fewest-tile rule.
//!
//! No policy reads privileged game state; none is a search or evaluation
//! engine in the rollout sense.

mod analysis;
mod heuristic;
mod missing_only;

pub use heuristic::HeuristicPolicy;
pub use missing_only::MissingOnlyPolicy;

use crate::game::{Decision, LegalActions};
use crate::{Action, Hand, Seat, Suit, Tile};

/// Chooses one action for the pending decision.
///
/// Implementations may only use the borrowed decision. Returning an action
/// that is not legal is a policy bug; the runner reports it as an error and
/// never silently substitutes another action.
pub trait Policy {
    /// Chooses an action for `decision`.
    fn choose(&mut self, decision: &Decision<'_>) -> Action;
}

/// Deterministic legal baseline.
///
/// The policy is stateless:
///
/// - `Exchange`: choose the eligible suit with the fewest own tiles (lowest
///   suit index on a tie) and pass its three lowest tiles.
/// - Other actions: use `Hu` when legal, otherwise use the first legal action
///   in the order supplied by the game.
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstLegalPolicy;

impl FirstLegalPolicy {
    /// Creates the baseline policy.
    pub const fn new() -> Self {
        Self
    }
}

impl Policy for FirstLegalPolicy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action {
        match &decision.legal_actions {
            LegalActions::Exchange { eligible_suits } => {
                choose_exchange_tiles(decision.observation.own_hand, *eligible_suits)
            }
            LegalActions::Actions(actions) => preferred_action(actions),
        }
    }
}

/// Seeded random legal policy.
///
/// The RNG is independent of the game and is not a rule input. Two policies
/// built with the same seed and shown the same sequence of decisions make the
/// same choices, which makes self-play replayable.
///
/// The stream position lives in this policy, not in [`crate::Game`]. A
/// mid-game checkpoint therefore consists of the cloned game *and* a clone of
/// every stateful policy; see the README continuation example. The policy is
/// [`Clone`], so a continuation can be built with
/// `Runner::new(cloned_policies)` without additional API. Constructing
/// `Runner::random(seed)` instead starts each seat's stream from its seed and
/// is not equivalent to restoring a checkpoint.
#[derive(Clone, Debug)]
pub struct RandomPolicy {
    rng: SplitMix64,
}

impl RandomPolicy {
    /// Creates the policy from an explicit policy seed.
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    /// Current internal RNG state (for diagnostics and replay checks).
    ///
    /// A checkpoint clone preserves this value exactly, which is what makes a
    /// resumed random run reproduce the original continuation.
    pub const fn rng_state(&self) -> u64 {
        self.rng.state
    }
}

impl Policy for RandomPolicy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action {
        match &decision.legal_actions {
            LegalActions::Exchange { eligible_suits } => choose_random_exchange(
                decision.observation.own_hand,
                *eligible_suits,
                &mut self.rng,
            ),
            LegalActions::Actions(actions) => {
                // Prefer Hu so random self-play still covers wins.
                if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
                    return hu;
                }
                if actions.is_empty() {
                    // A pending decision with no legal action is a game bug.
                    // Return an action anyway; the runner will report rejection.
                    return Action::Pass;
                }
                actions[self.rng.below(actions.len())]
            }
        }
    }
}

/// Derives an independent RNG seed for one seat from a batch seed.
///
/// The derivation is intentionally simple and stable so a documented batch
/// seed fully determines every seat's random policy stream.
pub fn random_seat_seed(seed: u64, seat: Seat) -> u64 {
    SplitMix64::new(seed ^ (seat.index() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

/// 64-bit SplitMix64 generator, standard library only.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform-enough index in `0..upper` for policy sampling.
    ///
    /// The tiny modulo bias is irrelevant for policy variety, and keeping the
    /// code std-only and allocation-free is more valuable here.
    fn below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

/// Deterministic exchange selection: fewest-tile eligible suit, lowest tiles.
fn choose_exchange_tiles(hand: &Hand, eligible_suits: [bool; 3]) -> Action {
    let mut best: Option<(usize, Suit)> = None;
    for suit in Suit::ALL {
        if !eligible_suits[suit.index()] {
            continue;
        }
        let count = hand.suit_count(suit);
        if count < 3 {
            continue;
        }
        let replace = match best {
            None => true,
            Some((best_count, _)) => count < best_count,
        };
        if replace {
            best = Some((count, suit));
        }
    }

    let Some((_, suit)) = best else {
        return fallback_exchange(hand);
    };

    let mut picked = [Tile::ALL[0]; 3];
    let mut index = 0;
    for tile in hand.tiles() {
        if tile.suit() != suit {
            continue;
        }
        picked[index] = tile;
        index += 1;
        if index == 3 {
            break;
        }
    }
    Action::Exchange(picked)
}

/// Random exchange selection: random eligible suit, then three random copies.
fn choose_random_exchange(hand: &Hand, eligible_suits: [bool; 3], rng: &mut SplitMix64) -> Action {
    let mut suits = [Suit::Wan; 3];
    let mut suit_count = 0;
    for suit in Suit::ALL {
        if eligible_suits[suit.index()] && hand.suit_count(suit) >= 3 {
            suits[suit_count] = suit;
            suit_count += 1;
        }
    }

    // Defensive fallback if the game marks no suit eligible even though the
    // hand size guarantees an exchange triple exists.
    if suit_count == 0 {
        for suit in Suit::ALL {
            if hand.suit_count(suit) >= 3 {
                suits[suit_count] = suit;
                suit_count += 1;
            }
        }
    }
    if suit_count == 0 {
        return fallback_exchange(hand);
    }

    let suit = suits[rng.below(suit_count)];

    // A 14-tile hand never has more than 14 candidates, but the buffer is
    // sized for the full 108-tile domain so arbitrary `Hand` values are safe.
    let mut tiles = [Tile::ALL[0]; crate::TOTAL_TILES];
    let mut tile_count = 0;
    for tile in hand.tiles() {
        if tile.suit() == suit {
            tiles[tile_count] = tile;
            tile_count += 1;
        }
    }
    debug_assert!(tile_count >= 3);

    for index in 0..3 {
        let remaining = tile_count - index;
        let swap_with = index + rng.below(remaining);
        tiles.swap(index, swap_with);
    }
    Action::Exchange([tiles[0], tiles[1], tiles[2]])
}

/// Picks `Hu` when legal, otherwise the first listed legal action.
fn preferred_action(actions: &[Action]) -> Action {
    if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
        return hu;
    }
    actions.first().copied().unwrap_or(Action::Pass)
}

/// Last-resort exchange built from the first three hand tiles.
///
/// It is only reached if the game exposes an impossible exchange constraint;
/// the runner then reports the rejected action instead of hiding the problem.
fn fallback_exchange(hand: &Hand) -> Action {
    let mut picked = [Tile::ALL[0]; 3];
    let mut index = 0;
    for tile in hand.tiles() {
        picked[index] = tile;
        index += 1;
        if index == 3 {
            break;
        }
    }
    Action::Exchange(picked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    #[test]
    fn splitmix_is_reproducible() {
        let mut first = SplitMix64::new(42);
        let mut second = SplitMix64::new(42);
        for _ in 0..32 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn cloning_preserves_the_rng_stream_position() {
        let mut original = RandomPolicy::new(9);
        for _ in 0..5 {
            original.rng.next_u64();
        }

        let clone = original.clone();
        assert_eq!(clone.rng_state(), original.rng_state());
        assert_ne!(
            clone.rng_state(),
            RandomPolicy::new(9).rng_state(),
            "a rebuilt policy restarts the stream"
        );
    }

    #[test]
    fn seat_seeds_differ_between_seats() {
        let seeds: Vec<u64> = Seat::ALL
            .into_iter()
            .map(|seat| random_seat_seed(7, seat))
            .collect();
        for (index, seed) in seeds.iter().enumerate() {
            assert!(!seeds[..index].contains(seed));
        }
    }

    #[test]
    fn first_legal_exchange_uses_one_eligible_suit() {
        let hand = Hand::from_tiles(&[
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
        ])
        .unwrap();

        let Action::Exchange(tiles) = choose_exchange_tiles(&hand, [true, true, true]) else {
            panic!("expected an exchange action");
        };
        assert!(tiles.iter().all(|tile| tile.suit() == Suit::Wan));
        assert_eq!(
            tiles,
            [tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)]
        );
    }

    #[test]
    fn random_exchange_always_uses_an_eligible_suit() {
        let hand = Hand::from_tiles(&[
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
        ])
        .unwrap();

        for seed in 0..64 {
            let mut policy = RandomPolicy::new(seed);
            let Action::Exchange(tiles) =
                choose_random_exchange(&hand, [false, true, false], &mut policy.rng)
            else {
                panic!("expected an exchange action");
            };
            assert!(tiles.iter().all(|tile| tile.suit() == Suit::Tong));
            assert!(hand.contains_tiles(&tiles));
        }
    }
}
