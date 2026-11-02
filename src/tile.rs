//! Tiles, suits, and compact counted hands.
//!
//! A tile is a value in `0..27`: nine ranks for each of the three suits.
//! The encoding order is Wan (`0..=8`), Tong (`9..=17`), Tiao (`18..=26`).
//! Each tile kind exists four times in the 108-tile deck.

use std::fmt;

/// Number of distinct tile kinds in the deck (3 suits x 9 ranks).
pub const TILE_KINDS: usize = 27;
/// Number of physical copies of each tile kind.
pub const COPIES_PER_TILE: u8 = 4;
/// Number of physical tiles in the full deck.
pub const TOTAL_TILES: usize = TILE_KINDS * COPIES_PER_TILE as usize;

/// The three suits used by Sichuan mahjong.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Suit {
    /// Characters / ten-thousands suit (`0..=8`).
    Wan,
    /// Dots / circles suit (`9..=17`).
    Tong,
    /// Bamboo suit (`18..=26`).
    Tiao,
}

impl Suit {
    /// All suits in encoding order.
    pub const ALL: [Suit; 3] = [Suit::Wan, Suit::Tong, Suit::Tiao];

    /// Zero-based suit index in `0..3`.
    pub const fn index(self) -> usize {
        match self {
            Suit::Wan => 0,
            Suit::Tong => 1,
            Suit::Tiao => 2,
        }
    }

    /// Builds a suit from a zero-based index.
    pub const fn from_index(index: usize) -> Option<Suit> {
        match index {
            0 => Some(Suit::Wan),
            1 => Some(Suit::Tong),
            2 => Some(Suit::Tiao),
            _ => None,
        }
    }

    /// One-letter suit code used by [`Tile`]'s text form.
    pub const fn short_name(self) -> char {
        match self {
            Suit::Wan => 'W',
            Suit::Tong => 'T',
            Suit::Tiao => 'B',
        }
    }

    /// Long English suit name.
    pub const fn name(self) -> &'static str {
        match self {
            Suit::Wan => "Wan",
            Suit::Tong => "Tong",
            Suit::Tiao => "Tiao",
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A single tile kind, stored as its index in `0..27`.
///
/// The field is private so that every `Tile` is guaranteed to be in range.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tile(u8);

const fn make_all_tiles() -> [Tile; TILE_KINDS] {
    let mut tiles = [Tile(0); TILE_KINDS];
    let mut i = 0;
    while i < TILE_KINDS {
        tiles[i] = Tile(i as u8);
        i += 1;
    }
    tiles
}

impl Tile {
    /// Number of distinct tile kinds.
    pub const COUNT: usize = TILE_KINDS;

    /// All tile kinds in ascending encoding order.
    pub const ALL: [Tile; TILE_KINDS] = make_all_tiles();

    /// Builds a tile from an encoded index in `0..27`.
    pub const fn from_index(index: u8) -> Option<Tile> {
        if index < TILE_KINDS as u8 {
            Some(Tile(index))
        } else {
            None
        }
    }

    /// Builds a tile from its suit and a rank in `1..=9`.
    pub const fn from_suit_rank(suit: Suit, rank: u8) -> Option<Tile> {
        if rank < 1 || rank > 9 {
            return None;
        }
        Some(Tile(suit.index() as u8 * 9 + rank - 1))
    }

    /// Encoded index in `0..27`.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Suit of this tile.
    pub const fn suit(self) -> Suit {
        match self.0 / 9 {
            0 => Suit::Wan,
            1 => Suit::Tong,
            _ => Suit::Tiao,
        }
    }

    /// One-based rank within the suit in `1..=9`.
    pub const fn rank(self) -> u8 {
        self.0 % 9 + 1
    }

    /// Zero-based suit index in `0..3`.
    pub const fn suit_index(self) -> usize {
        (self.0 / 9) as usize
    }

    /// Zero-based rank within the suit in `0..9`.
    pub const fn rank_index(self) -> usize {
        (self.0 % 9) as usize
    }

    /// Whether another tile belongs to the same suit and is exactly one rank
    /// above this tile.
    pub const fn is_immediate_successor(self, other: Tile) -> bool {
        self.suit_index() == other.suit_index() && other.rank_index() == self.rank_index() + 1
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit().short_name())
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Errors produced by mutating or constructing a [`Hand`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandError {
    /// Adding this tile would exceed the four physical copies in the deck.
    TooManyCopies(Tile),
    /// Removing this tile when the hand holds none of it.
    TileNotInHand(Tile),
}

impl fmt::Display for HandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandError::TooManyCopies(tile) => {
                write!(f, "hand already holds four copies of {tile}")
            }
            HandError::TileNotInHand(tile) => write!(f, "hand does not hold {tile}"),
        }
    }
}

impl std::error::Error for HandError {}

/// A counted hand over all 27 tile kinds.
///
/// `Hand` stores only concealed tiles; declared melds are modelled separately
/// by [`crate::Meld`]. A hand is a general container and does not enforce a
/// 14-tile limit, but it never allows more than four copies of one tile kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hand {
    counts: [u8; TILE_KINDS],
}

impl Hand {
    /// Creates an empty hand.
    pub const fn new() -> Self {
        Self {
            counts: [0; TILE_KINDS],
        }
    }

    /// Creates a hand from a list of tiles.
    pub fn from_tiles(tiles: &[Tile]) -> Result<Self, HandError> {
        let mut hand = Self::new();
        for &tile in tiles {
            hand.add(tile)?;
        }
        Ok(hand)
    }

    /// Read-only counts indexed by [`Tile::index`].
    pub const fn counts(&self) -> &[u8; TILE_KINDS] {
        &self.counts
    }

    /// Number of copies of `tile` in this hand.
    pub const fn count(&self, tile: Tile) -> u8 {
        self.counts[tile.index() as usize]
    }

    /// Total number of concealed tiles.
    pub fn total(&self) -> usize {
        self.counts.iter().map(|&count| count as usize).sum()
    }

    /// Whether this hand is empty.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Adds one copy of `tile`.
    pub fn add(&mut self, tile: Tile) -> Result<(), HandError> {
        let slot = &mut self.counts[tile.index() as usize];
        if *slot >= COPIES_PER_TILE {
            return Err(HandError::TooManyCopies(tile));
        }
        *slot += 1;
        Ok(())
    }

    /// Adds every tile in `tiles`, leaving the hand unchanged on error.
    pub fn add_tiles(&mut self, tiles: &[Tile]) -> Result<(), HandError> {
        let mut next = *self;
        for &tile in tiles {
            next.add(tile)?;
        }
        *self = next;
        Ok(())
    }

    /// Removes one copy of `tile`.
    pub fn remove(&mut self, tile: Tile) -> Result<(), HandError> {
        let slot = &mut self.counts[tile.index() as usize];
        if *slot == 0 {
            return Err(HandError::TileNotInHand(tile));
        }
        *slot -= 1;
        Ok(())
    }

    /// Removes every tile in `tiles`, leaving the hand unchanged on error.
    pub fn remove_tiles(&mut self, tiles: &[Tile]) -> Result<(), HandError> {
        let mut next = *self;
        for &tile in tiles {
            next.remove(tile)?;
        }
        *self = next;
        Ok(())
    }

    /// Whether this hand holds at least one copy of `tile`.
    pub const fn contains(&self, tile: Tile) -> bool {
        self.count(tile) > 0
    }

    /// Whether this hand holds every tile in `tiles`, respecting duplicates.
    pub fn contains_tiles(&self, tiles: &[Tile]) -> bool {
        let mut needed = [0u8; TILE_KINDS];
        for &tile in tiles {
            let slot = &mut needed[tile.index() as usize];
            if *slot >= self.count(tile) {
                return false;
            }
            *slot += 1;
        }
        true
    }

    /// Number of concealed tiles in `suit`.
    pub fn suit_count(&self, suit: Suit) -> usize {
        let start = suit.index() * 9;
        self.counts[start..start + 9]
            .iter()
            .map(|&count| count as usize)
            .sum()
    }

    /// Whether the hand holds at least one tile in `suit`.
    pub fn has_suit(&self, suit: Suit) -> bool {
        self.suit_count(suit) > 0
    }

    /// Whether every concealed tile belongs to `suit` (empty hands qualify).
    pub fn has_only_suit(&self, suit: Suit) -> bool {
        self.suits().all(|present| present == suit)
    }

    /// Suits present in this hand, in encoding order.
    pub fn suits(&self) -> impl Iterator<Item = Suit> + '_ {
        Suit::ALL
            .into_iter()
            .filter(move |&suit| self.has_suit(suit))
    }

    /// Every physical tile in this hand, sorted by tile index.
    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        Tile::ALL
            .into_iter()
            .flat_map(move |tile| std::iter::repeat(tile).take(self.count(tile) as usize))
    }

    /// Distinct tile kinds present in this hand, in encoding order.
    pub fn distinct_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        Tile::ALL
            .into_iter()
            .filter(move |&tile| self.contains(tile))
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, tile) in self.tiles().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{tile}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hand[{self}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    #[test]
    fn tile_index_round_trips_for_all_kinds() {
        for (index, tile) in Tile::ALL.into_iter().enumerate() {
            assert_eq!(tile.index() as usize, index);
            assert_eq!(Tile::from_index(tile.index()), Some(tile));
            assert_eq!(Tile::from_suit_rank(tile.suit(), tile.rank()), Some(tile));
        }
    }

    #[test]
    fn tile_encoding_matches_suit_layout() {
        assert_eq!(tile(Suit::Wan, 1).index(), 0);
        assert_eq!(tile(Suit::Wan, 9).index(), 8);
        assert_eq!(tile(Suit::Tong, 1).index(), 9);
        assert_eq!(tile(Suit::Tong, 9).index(), 17);
        assert_eq!(tile(Suit::Tiao, 1).index(), 18);
        assert_eq!(tile(Suit::Tiao, 9).index(), 26);
    }

    #[test]
    fn invalid_tiles_are_rejected() {
        assert_eq!(Tile::from_index(27), None);
        assert_eq!(Tile::from_index(255), None);
        assert_eq!(Tile::from_suit_rank(Suit::Wan, 0), None);
        assert_eq!(Tile::from_suit_rank(Suit::Wan, 10), None);
    }

    #[test]
    fn suit_queries_are_consistent() {
        for suit in Suit::ALL {
            for rank in 1..=9 {
                let tile = tile(suit, rank);
                assert_eq!(tile.suit(), suit);
                assert_eq!(tile.rank(), rank);
                assert_eq!(tile.suit_index(), suit.index());
                assert_eq!(tile.rank_index(), rank as usize - 1);
            }
        }
        assert_eq!(Suit::from_index(0), Some(Suit::Wan));
        assert_eq!(Suit::from_index(1), Some(Suit::Tong));
        assert_eq!(Suit::from_index(2), Some(Suit::Tiao));
        assert_eq!(Suit::from_index(3), None);
    }

    #[test]
    fn tile_display_uses_rank_and_suit_letter() {
        assert_eq!(tile(Suit::Wan, 3).to_string(), "3W");
        assert_eq!(tile(Suit::Tong, 7).to_string(), "7T");
        assert_eq!(tile(Suit::Tiao, 9).to_string(), "9B");
    }

    #[test]
    fn hand_add_remove_and_counts() {
        let mut hand = Hand::new();
        assert!(hand.is_empty());
        hand.add(tile(Suit::Wan, 1)).unwrap();
        hand.add(tile(Suit::Wan, 1)).unwrap();
        hand.add(tile(Suit::Tiao, 9)).unwrap();

        assert_eq!(hand.count(tile(Suit::Wan, 1)), 2);
        assert_eq!(hand.count(tile(Suit::Wan, 2)), 0);
        assert_eq!(hand.total(), 3);
        assert_eq!(hand.suit_count(Suit::Wan), 2);
        assert_eq!(hand.suit_count(Suit::Tiao), 1);
        assert!(hand.contains(tile(Suit::Wan, 1)));
        assert!(!hand.contains(tile(Suit::Tong, 1)));

        hand.remove(tile(Suit::Tong, 1)).unwrap_err();
        hand.remove_tiles(&[tile(Suit::Wan, 1), tile(Suit::Tiao, 9)])
            .unwrap();
        assert_eq!(hand.total(), 1);
        assert_eq!(hand.count(tile(Suit::Wan, 1)), 1);
    }

    #[test]
    fn hand_rejects_a_fifth_copy() {
        let mut hand = Hand::new();
        for _ in 0..COPIES_PER_TILE {
            hand.add(tile(Suit::Tong, 5)).unwrap();
        }
        assert_eq!(
            hand.add(tile(Suit::Tong, 5)),
            Err(HandError::TooManyCopies(tile(Suit::Tong, 5)))
        );
    }

    #[test]
    fn batch_mutations_are_atomic() {
        let mut hand = Hand::from_tiles(&[tile(Suit::Wan, 1)]).unwrap();
        let original = hand;
        assert!(hand
            .remove_tiles(&[tile(Suit::Wan, 1), tile(Suit::Wan, 2)])
            .is_err());
        assert_eq!(hand, original);

        assert!(hand
            .add_tiles(&[tile(Suit::Wan, 2), tile(Suit::Wan, 2), tile(Suit::Wan, 2)])
            .is_ok());
        let before = hand;
        let too_many = [tile(Suit::Wan, 2), tile(Suit::Wan, 2)];
        assert!(hand.add_tiles(&too_many).is_err());
        assert_eq!(hand, before);
    }

    #[test]
    fn hand_iteration_is_sorted_and_complete() {
        let hand = Hand::from_tiles(&[
            tile(Suit::Tiao, 2),
            tile(Suit::Wan, 5),
            tile(Suit::Tong, 1),
            tile(Suit::Wan, 5),
        ])
        .unwrap();

        let tiles: Vec<_> = hand.tiles().collect();
        assert_eq!(
            tiles,
            vec![
                tile(Suit::Wan, 5),
                tile(Suit::Wan, 5),
                tile(Suit::Tong, 1),
                tile(Suit::Tiao, 2)
            ]
        );
        let distinct: Vec<_> = hand.distinct_tiles().collect();
        assert_eq!(
            distinct,
            vec![tile(Suit::Wan, 5), tile(Suit::Tong, 1), tile(Suit::Tiao, 2)]
        );
        assert_eq!(hand.to_string(), "5W 5W 1T 2B");
    }

    #[test]
    fn contains_tiles_respects_duplicates() {
        let hand = Hand::from_tiles(&[tile(Suit::Wan, 3), tile(Suit::Wan, 3), tile(Suit::Wan, 4)])
            .unwrap();
        assert!(hand.contains_tiles(&[tile(Suit::Wan, 3), tile(Suit::Wan, 4)]));
        assert!(hand.contains_tiles(&[tile(Suit::Wan, 3), tile(Suit::Wan, 3)]));
        assert!(!hand.contains_tiles(&[
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 3)
        ]));
        assert!(!hand.contains_tiles(&[tile(Suit::Wan, 3), tile(Suit::Wan, 5)]));
    }

    #[test]
    fn contains_tiles_rejects_long_duplicate_runs_without_overflow() {
        let hand = Hand::from_tiles(&[tile(Suit::Wan, 3)]).unwrap();
        let many = [tile(Suit::Wan, 3); 300];
        assert!(!hand.contains_tiles(&many));
    }

    #[test]
    fn suit_membership_helpers() {
        let hand = Hand::from_tiles(&[tile(Suit::Wan, 1), tile(Suit::Wan, 4), tile(Suit::Tong, 9)])
            .unwrap();
        assert!(hand.has_suit(Suit::Wan));
        assert!(hand.has_suit(Suit::Tong));
        assert!(!hand.has_suit(Suit::Tiao));
        assert!(!hand.has_only_suit(Suit::Wan));
        assert!(Hand::new().has_only_suit(Suit::Tiao));

        let pure = Hand::from_tiles(&[tile(Suit::Tiao, 1), tile(Suit::Tiao, 2)]).unwrap();
        assert!(pure.has_only_suit(Suit::Tiao));
        let suits: Vec<_> = pure.suits().collect();
        assert_eq!(suits, vec![Suit::Tiao]);
    }

    #[test]
    fn from_tiles_rejects_impossible_hand() {
        let tiles = [tile(Suit::Tong, 9); 5];
        assert_eq!(
            Hand::from_tiles(&tiles),
            Err(HandError::TooManyCopies(tile(Suit::Tong, 9)))
        );
    }
}
