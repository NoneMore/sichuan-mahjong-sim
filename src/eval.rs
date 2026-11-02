//! Pure winning-shape evaluation and ready-hand (tenpai) queries.
//!
//! All functions in this module are pure: they inspect a concealed [`Hand`]
//! and the number of declared melds, and never mutate game state.

use crate::action::Meld;
use crate::hand::Hand;
use crate::tile::{Tile, COPIES_PER_TILE, TILE_KINDS};

/// A legal winning shape.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum WinShape {
    /// Four sets and one pair, with any number of declared melds.
    Standard,
    /// Seven pairs with no declared melds.
    SevenPairs,
}

/// Number of concealed tiles expected at a decision or win.
///
/// Each declared meld absorbs three tiles from the standard 14-tile shape.
/// Kongs still count as one meld for shape purposes; their fourth tile is
/// replaced by a wall draw. Returns `None` for more than four melds.
pub const fn expected_concealed_len(meld_count: usize) -> Option<usize> {
    if meld_count <= 4 {
        Some(14 - 3 * meld_count)
    } else {
        None
    }
}

fn can_form_sets(counts: &mut [u8; TILE_KINDS], start: usize) -> bool {
    let mut index = start;
    while index < TILE_KINDS && counts[index] == 0 {
        index += 1;
    }
    if index == TILE_KINDS {
        return true;
    }

    if counts[index] >= 3 {
        counts[index] -= 3;
        if can_form_sets(counts, index) {
            counts[index] += 3;
            return true;
        }
        counts[index] += 3;
    }

    let rank_index = index % 9;
    if rank_index <= 6 && counts[index + 1] > 0 && counts[index + 2] > 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        if can_form_sets(counts, index) {
            counts[index] += 1;
            counts[index + 1] += 1;
            counts[index + 2] += 1;
            return true;
        }
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }

    false
}

/// Whether the concealed tiles form four sets and one pair.
///
/// `meld_count` declared melds are assumed to already provide that many of the
/// four required sets. The concealed hand must therefore hold
/// `14 - 3 * meld_count` tiles.
pub fn is_standard_winning(hand: &Hand, meld_count: usize) -> bool {
    let Some(expected) = expected_concealed_len(meld_count) else {
        return false;
    };
    if hand.total() != expected {
        return false;
    }

    let counts = *hand.counts();
    for pair in 0..TILE_KINDS {
        if counts[pair] >= 2 {
            let mut rest = counts;
            rest[pair] -= 2;
            if can_form_sets(&mut rest, 0) {
                return true;
            }
        }
    }
    false
}

/// Whether the concealed tiles form seven pairs with no melds.
///
/// Four identical tiles count as two pairs, so every count must be even and
/// the total must be exactly 14. The caller is responsible for rejecting hands
/// with declared melds; [`is_winning`] and [`win_shape`] do so automatically.
pub fn is_seven_pairs(hand: &Hand) -> bool {
    if hand.total() != 14 {
        return false;
    }
    let mut pair_units = 0usize;
    for &count in hand.counts() {
        if count % 2 != 0 {
            return false;
        }
        pair_units += (count / 2) as usize;
    }
    pair_units == 7
}

/// Whether the hand is a legal win shape.
///
/// Seven pairs is only legal with no declared melds. Open hands can win with
/// the standard shape.
pub fn is_winning(hand: &Hand, meld_count: usize) -> bool {
    is_standard_winning(hand, meld_count) || (meld_count == 0 && is_seven_pairs(hand))
}

/// Classifies a legal winning shape, preferring seven pairs when both
/// interpretations are possible.
pub fn win_shape(hand: &Hand, meld_count: usize) -> Option<WinShape> {
    if meld_count == 0 && is_seven_pairs(hand) {
        Some(WinShape::SevenPairs)
    } else if is_standard_winning(hand, meld_count) {
        Some(WinShape::Standard)
    } else {
        None
    }
}

fn meld_copies_of(melds: &[Meld], tile: Tile) -> u8 {
    melds
        .iter()
        .filter(|meld| meld.tile() == tile)
        .map(|meld| meld.tile_count())
        .sum()
}

/// Tiles that would complete the hand if added to the concealed tiles.
///
/// The check respects the four-copy limit across the concealed hand and the
/// declared melds. It is a shape query only; it does not apply the missing-suit
/// restriction.
pub fn waiting_tiles(hand: &Hand, melds: &[Meld]) -> Vec<Tile> {
    let meld_count = melds.len();
    let mut waits = Vec::new();
    for tile in Tile::ALL {
        if hand.count(tile) + meld_copies_of(melds, tile) >= COPIES_PER_TILE {
            continue;
        }
        let mut candidate = *hand;
        if candidate.add(tile).is_err() {
            continue;
        }
        if is_winning(&candidate, meld_count) {
            waits.push(tile);
        }
    }
    waits
}

/// Whether the hand is one tile away from a legal win shape.
pub fn is_tenpai(hand: &Hand, melds: &[Meld]) -> bool {
    !waiting_tiles(hand, melds).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Meld;
    use crate::seat::Seat;
    use crate::tile::Suit;

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    fn hand(tiles: &[Tile]) -> Hand {
        Hand::from_tiles(tiles).unwrap()
    }

    fn wan(ranks: &[u8]) -> Vec<Tile> {
        ranks.iter().map(|&rank| tile(Suit::Wan, rank)).collect()
    }

    #[test]
    fn expected_concealed_lengths() {
        assert_eq!(expected_concealed_len(0), Some(14));
        assert_eq!(expected_concealed_len(1), Some(11));
        assert_eq!(expected_concealed_len(2), Some(8));
        assert_eq!(expected_concealed_len(3), Some(5));
        assert_eq!(expected_concealed_len(4), Some(2));
        assert_eq!(expected_concealed_len(5), None);
    }

    #[test]
    fn standard_hand_wins() {
        // 123 123 456 789 99
        let hand = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert!(is_standard_winning(&hand, 0));
        assert!(is_winning(&hand, 0));
        assert_eq!(win_shape(&hand, 0), Some(WinShape::Standard));
    }

    #[test]
    fn all_pungs_hand_wins() {
        // 111 222 333 444 55
        let hand = hand(&wan(&[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]));
        assert!(is_standard_winning(&hand, 0));
        assert!(is_winning(&hand, 0));
    }

    #[test]
    fn four_identical_tiles_can_split_across_sets() {
        // 111 + 123 + 456 + 789 + 99
        let hand = hand(&wan(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert_eq!(hand.total(), 14);
        assert!(is_standard_winning(&hand, 0));
    }

    #[test]
    fn seven_pairs_hand_wins() {
        let hand = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]));
        assert!(is_seven_pairs(&hand));
        assert!(is_winning(&hand, 0));
        assert_eq!(win_shape(&hand, 0), Some(WinShape::SevenPairs));
    }

    #[test]
    fn four_of_a_kind_counts_as_two_pairs() {
        // 1111 2222 3333 44: four pair units from the triplets plus one from 44.
        let hand = hand(&wan(&[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]));
        assert!(is_seven_pairs(&hand));
        assert!(is_winning(&hand, 0));
        assert_eq!(win_shape(&hand, 0), Some(WinShape::SevenPairs));
    }

    #[test]
    fn seven_pairs_requires_no_melds() {
        let hand = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]));
        assert!(!is_standard_winning(&hand, 1));
        assert!(!is_winning(&hand, 1));
        assert_eq!(win_shape(&hand, 1), None);
        assert!(is_seven_pairs(&hand));
    }

    #[test]
    fn seven_pairs_preferred_over_standard_interpretation() {
        // Also decomposes as 123 123 456 456 77.
        let hand = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]));
        assert!(is_standard_winning(&hand, 0));
        assert_eq!(win_shape(&hand, 0), Some(WinShape::SevenPairs));
    }

    #[test]
    fn open_standard_hand_can_win() {
        // Two declared melds leave 14 - 6 = 8 concealed tiles: 123 456 99.
        let hand = hand(&wan(&[1, 2, 3, 4, 5, 6, 9, 9]));
        assert!(is_standard_winning(&hand, 2));
        assert!(is_winning(&hand, 2));
        assert_eq!(win_shape(&hand, 2), Some(WinShape::Standard));
    }

    #[test]
    fn four_melds_require_only_a_pair() {
        let hand = hand(&wan(&[5, 5]));
        assert!(is_winning(&hand, 4));
        assert!(!is_winning(&hand, 4 + 1));
    }

    #[test]
    fn wrong_tile_count_is_not_a_win() {
        let thirteen = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9]));
        assert!(!is_winning(&thirteen, 0));

        let fourteen = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert!(!is_winning(&fourteen, 1));
    }

    #[test]
    fn disconnected_hand_does_not_win() {
        let hand = hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 7),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 7),
            tile(Suit::Tiao, 1),
            tile(Suit::Tiao, 4),
            tile(Suit::Tiao, 7),
            tile(Suit::Wan, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Tiao, 2),
            tile(Suit::Wan, 9),
            tile(Suit::Wan, 9),
        ]);
        assert!(!is_winning(&hand, 0));
    }

    #[test]
    fn waits_are_enumerated() {
        // The 13-tile shape 1122334567899 waits on 3W, 6W, and 9W.
        let hand = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9]));
        assert_eq!(hand.total(), 13);
        assert!(is_tenpai(&hand, &[]));
        assert_eq!(
            waiting_tiles(&hand, &[]),
            vec![tile(Suit::Wan, 3), tile(Suit::Wan, 6), tile(Suit::Wan, 9)]
        );
    }

    #[test]
    fn waits_include_multiple_tiles() {
        // 123 456 789 99 55 waits on either 5W or 9W.
        let hand = hand(&wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 5, 5]));
        assert_eq!(hand.total(), 13);
        let waits = waiting_tiles(&hand, &[]);
        assert!(waits.contains(&tile(Suit::Wan, 5)));
        assert!(waits.contains(&tile(Suit::Wan, 9)));
    }

    #[test]
    fn not_tenpai_when_no_tile_completes() {
        let hand = hand(&disconnected_tiles());
        assert!(!is_tenpai(&hand, &[]));
        assert!(waiting_tiles(&hand, &[]).is_empty());
    }

    fn disconnected_tiles() -> Vec<Tile> {
        vec![
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 7),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 4),
            tile(Suit::Tong, 7),
            tile(Suit::Tiao, 1),
            tile(Suit::Tiao, 4),
            tile(Suit::Tiao, 7),
            tile(Suit::Wan, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Tiao, 2),
            tile(Suit::Wan, 9),
        ]
    }

    #[test]
    fn waits_skip_tiles_exhausted_by_melds() {
        // A kong of 3W consumes all four copies, so the 3W wait cannot be drawn.
        let hand = hand(&wan(&[1, 2, 4, 5, 6, 7, 8, 9, 9, 9]));
        let melds = [Meld::concealed_kong(tile(Suit::Wan, 3))];
        assert!(!is_tenpai(&hand, &melds));
        assert!(!waiting_tiles(&hand, &melds).contains(&tile(Suit::Wan, 3)));
    }

    #[test]
    fn open_hand_waits_use_meld_count() {
        // With two melds, the 7-tile concealed part 123 456 7 waits on 7W.
        let hand = hand(&wan(&[1, 2, 3, 4, 5, 6, 7]));
        let source = Seat::new(1).unwrap();
        let melds = [
            Meld::peng(tile(Suit::Tong, 1), source),
            Meld::peng(tile(Suit::Tiao, 1), source),
        ];
        assert!(waiting_tiles(&hand, &melds).contains(&tile(Suit::Wan, 7)));
    }
}
