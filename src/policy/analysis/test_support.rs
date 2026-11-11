//! Shared test-only construction and sampling helpers for the analysis tests.
//!
//! This module is compiled only for tests. Production code never imports it,
//! and the concrete-target reference in [`targets`] never calls a production
//! transition. Keeping these small helpers here removes drifting copies from
//! sibling test modules without turning them into a general test framework.

pub(super) mod targets;

use crate::hand::Hand;
use crate::tile::{Suit, Tile, TILE_KINDS};

/// Nine one-based ranks for one suit.
pub(super) fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

/// Builds a hand from exact tiles.
pub(super) fn hand(tiles: &[Tile]) -> Hand {
    Hand::from_tiles(tiles).expect("legal hand")
}

/// Builds a hand from one-based rank groups, one tile per listed rank.
pub(super) fn suit_hand(wan: &[u8], tong: &[u8], tiao: &[u8]) -> Hand {
    let mut hand = Hand::new();
    for (suit, ranks) in [(Suit::Wan, wan), (Suit::Tong, tong), (Suit::Tiao, tiao)] {
        for &rank in ranks {
            hand.add(tile(suit, rank)).expect("at most four copies");
        }
    }
    hand
}

/// Counts one tile per listed `(suit, one-based ranks)` group.
pub(super) fn counts_from_ranks(groups: &[(Suit, &[u8])]) -> [u8; TILE_KINDS] {
    let mut counts = [0u8; TILE_KINDS];
    for &(suit, ranks) in groups {
        for &rank in ranks {
            counts[tile(suit, rank).index() as usize] += 1;
        }
    }
    counts
}

/// Builds a hand from a counted vector.
pub(super) fn hand_from_counts(counts: &[u8; TILE_KINDS]) -> Hand {
    let mut hand = Hand::new();
    for kind in Tile::ALL {
        for _ in 0..counts[kind.index() as usize] {
            hand.add(kind).expect("at most four copies per kind");
        }
    }
    hand
}

/// The nine counted ranks of one suit.
pub(super) fn suit_counts(hand: &Hand, suit: Suit) -> [u8; 9] {
    let mut local = [0u8; 9];
    let start = suit.index() * 9;
    local.copy_from_slice(&hand.counts()[start..start + 9]);
    local
}

/// Small deterministic LCG used by the fixed-seed sweeps.
///
/// `new` preserves the historical `seed | 1` initialization. `raw` keeps a
/// complete caller-supplied state, which some fixed sweeps need to preserve
/// their original even-seeded stream byte for byte.
pub(super) struct Lcg(u64);

impl Lcg {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub(super) const fn raw(state: u64) -> Self {
        Self(state)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    pub(super) fn below(&mut self, bound: usize) -> usize {
        ((self.next() >> 33) % bound as u64) as usize
    }
}
