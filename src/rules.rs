//! Fixed baseline rules: legality helpers, fan facts, and settlement facts.
//!
//! Every function here is pure. The game engine owns state and timing; this
//! module only answers rule questions and computes point transfers.

use crate::action::{Meld, MeldKind};
use crate::eval::{self, WinShape};
use crate::hand::Hand;
use crate::seat::Seat;
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

/// Concealed tiles dealt to the dealer before the exchange.
pub const DEALER_INITIAL_TILES: usize = 14;
/// Concealed tiles dealt to each non-dealer before the exchange.
pub const NON_DEALER_INITIAL_TILES: usize = 13;

/// Fan for a plain win.
pub const BASE_WIN_FAN: u8 = 1;
/// Maximum total fan for one win, applied after adding all fan items.
pub const FAN_CAP: u8 = 5;

/// Points each flower pig pays to each non-flower-pig player at wall exhaustion.
pub const FLOWER_PIG_PENALTY: u32 = 16;
/// Points each active opponent pays for a concealed kong.
pub const CONCEALED_KONG_POINTS: u32 = 2;
/// Points each active opponent pays for an added kong.
pub const ADDED_KONG_POINTS: u32 = 1;
/// Points the discarder pays for an open (direct) kong.
pub const OPEN_KONG_POINTS: u32 = 2;

/// Whether the player still holds tiles of the declared missing suit.
///
/// Declared melds are included for robustness even though a legal meld can
/// never use the missing suit.
pub fn has_missing_suit_tiles(hand: &Hand, melds: &[Meld], missing: Suit) -> bool {
    hand.has_suit(missing) || melds.iter().any(|meld| meld.tile().suit() == missing)
}

/// Whether three selected tiles form a legal opening exchange.
///
/// The three tiles must share one suit and be present in the hand, respecting
/// duplicate copies. Every 13- or 14-tile hand necessarily has at least one
/// such triple, but the selected triple must itself be legal.
pub fn is_valid_exchange(hand: &Hand, tiles: [Tile; 3]) -> bool {
    let suit = tiles[0].suit();
    tiles.iter().all(|&tile| tile.suit() == suit) && hand.contains_tiles(&tiles)
}

/// Whether `suit` has at least three tiles available to exchange.
pub fn can_exchange_suit(hand: &Hand, suit: Suit) -> bool {
    hand.suit_count(suit) >= 3
}

/// Whether `tile` can be discarded under the missing-suit obligation.
///
/// If any missing-suit tile remains, only a missing-suit tile may be discarded.
/// Otherwise any concealed tile may be discarded.
pub fn can_discard(hand: &Hand, missing: Suit, tile: Tile) -> bool {
    hand.contains(tile) && (!hand.has_suit(missing) || tile.suit() == missing)
}

/// Whether the hand may claim `tile` as a peng.
///
/// Pengs are illegal while any missing-suit tile remains and may never use the
/// missing suit itself.
pub fn can_peng(hand: &Hand, melds: &[Meld], missing: Suit, tile: Tile) -> bool {
    tile.suit() != missing && !has_missing_suit_tiles(hand, melds, missing) && hand.count(tile) >= 2
}

/// Whether the hand may claim `tile` as an open kong.
pub fn can_open_kong(hand: &Hand, melds: &[Meld], missing: Suit, tile: Tile) -> bool {
    tile.suit() != missing && !has_missing_suit_tiles(hand, melds, missing) && hand.count(tile) >= 3
}

/// Whether the hand may declare a concealed kong of `tile`.
pub fn can_concealed_kong(hand: &Hand, melds: &[Meld], missing: Suit, tile: Tile) -> bool {
    tile.suit() != missing
        && !has_missing_suit_tiles(hand, melds, missing)
        && hand.count(tile) == COPIES_PER_TILE
}

/// Whether the hand may upgrade an existing peng of `tile` to an added kong.
pub fn can_added_kong(hand: &Hand, melds: &[Meld], missing: Suit, tile: Tile) -> bool {
    tile.suit() != missing
        && !has_missing_suit_tiles(hand, melds, missing)
        && hand.contains(tile)
        && melds
            .iter()
            .any(|meld| meld.kind() == MeldKind::Peng && meld.tile() == tile)
}

/// Whether the hand may declare a win.
///
/// The caller supplies the hand after adding the winning tile for a discard
/// claim. Winning requires a legal shape and no missing-suit tiles anywhere.
pub fn can_hu(hand: &Hand, melds: &[Meld], missing: Suit) -> bool {
    !has_missing_suit_tiles(hand, melds, missing) && eval::is_winning(hand, melds.len())
}

/// Whether every physical tile in the hand and melds belongs to one suit.
pub fn is_pure_suit(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suit = None;
    for tile in hand.tiles().chain(melds.iter().map(|meld| meld.tile())) {
        match suit {
            None => suit = Some(tile.suit()),
            Some(existing) if existing == tile.suit() => {}
            Some(_) => return false,
        }
    }
    suit.is_some()
}

/// Number of "root" fan items: tile kinds with all four copies held across the
/// concealed hand and declared melds.
///
/// A kong counts automatically. A peng plus the fourth tile in hand also
/// counts. Each tile kind contributes at most one root.
pub fn root_count(hand: &Hand, melds: &[Meld]) -> u8 {
    let mut roots = 0u8;
    for tile in Tile::ALL {
        let mut copies = hand.count(tile) as u32;
        for meld in melds {
            if meld.tile() == tile {
                copies += meld.tile_count() as u32;
            }
        }
        if copies == COPIES_PER_TILE as u32 {
            roots += 1;
        }
    }
    roots
}

/// Whether the standard win uses only triplets and kongs plus the pair.
///
/// All legal meld kinds already represent identical tiles, so only the
/// concealed part needs decomposition. Seven pairs is not all pungs.
pub fn is_all_pungs(hand: &Hand, meld_count: usize) -> bool {
    if !eval::is_standard_winning(hand, meld_count) {
        return false;
    }
    let counts = *hand.counts();
    for pair in 0..TILE_KINDS {
        if counts[pair] >= 2 {
            let mut rest = counts;
            rest[pair] -= 2;
            if rest.iter().all(|&count| count % 3 == 0) {
                return true;
            }
        }
    }
    false
}

/// How a win was completed.
///
/// The variants are mutually exclusive, which implements the non-stacking rule
/// for event fan: at most one event bonus applies.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum WinMethod {
    /// Win on another player's discard.
    Discard,
    /// Win on the player's own normal draw.
    SelfDraw,
    /// Win on the replacement tile after any kong (self-draw).
    AfterKong,
    /// Win on a discard produced immediately after that player's kong.
    KongDiscard,
    /// Win by robbing an added kong.
    RobbingKong,
}

impl WinMethod {
    /// Event fan for this method: one point for every win except a plain discard.
    pub const fn event_fan(self) -> u8 {
        match self {
            WinMethod::Discard => 0,
            WinMethod::SelfDraw
            | WinMethod::AfterKong
            | WinMethod::KongDiscard
            | WinMethod::RobbingKong => 1,
        }
    }

    /// Whether every other active player pays this win.
    pub const fn is_self_draw(self) -> bool {
        matches!(self, WinMethod::SelfDraw | WinMethod::AfterKong)
    }

    /// Number of paying players for this method.
    ///
    /// Self-draw methods use the supplied number of other active players.
    /// Discard-like methods always have exactly one payer (the discarder or the
    /// player whose added kong was robbed).
    pub const fn payer_count(self, active_opponents: usize) -> usize {
        if self.is_self_draw() {
            active_opponents
        } else {
            1
        }
    }
}

/// Itemized fan for one winning hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FanBreakdown {
    /// Winning shape used for scoring.
    pub shape: WinShape,
    /// Base win fan (always 1).
    pub base: u8,
    /// Pure one-suit fan (0 or 2).
    pub pure_suit: u8,
    /// Seven-pairs fan (0 or 2).
    pub seven_pairs: u8,
    /// All-pungs fan (0 or 1).
    pub all_pungs: u8,
    /// Root fan, one per four-of-a-kind tile kind.
    pub roots: u8,
    /// Event fan (0 or 1).
    pub event: u8,
    /// Sum before the cap.
    pub raw_total: u8,
    /// Final fan after applying [`FAN_CAP`].
    pub total: u8,
}

impl FanBreakdown {
    /// Whether the fan cap changed the raw total.
    pub const fn is_capped(&self) -> bool {
        self.raw_total > self.total
    }
}

/// Computes itemized fan for a legal winning hand.
///
/// Returns `None` when the hand does not have a legal shape. Missing-suit
/// legality is separate: callers must check [`can_hu`] first.
pub fn win_fan(hand: &Hand, melds: &[Meld], method: WinMethod) -> Option<FanBreakdown> {
    let shape = eval::win_shape(hand, melds.len())?;
    let pure_suit = if is_pure_suit(hand, melds) { 2 } else { 0 };
    let seven_pairs = if shape == WinShape::SevenPairs { 2 } else { 0 };
    let all_pungs = if shape == WinShape::Standard && is_all_pungs(hand, melds.len()) {
        1
    } else {
        0
    };
    let roots = root_count(hand, melds);
    let event = method.event_fan();
    let raw_total = BASE_WIN_FAN + pure_suit + seven_pairs + all_pungs + roots + event;
    Some(FanBreakdown {
        shape,
        base: BASE_WIN_FAN,
        pure_suit,
        seven_pairs,
        all_pungs,
        roots,
        event,
        raw_total,
        total: raw_total.min(FAN_CAP),
    })
}

/// Maximum hypothetical discard-win fan for a ready hand.
///
/// This is the value used by the end-of-wall big-call check. It returns `None`
/// when the hand is not tenpai. The caller must ensure the hand has no
/// missing-suit tiles; this is a shape-only query.
pub fn max_discard_win_fan(hand: &Hand, melds: &[Meld]) -> Option<u8> {
    let mut best = None;
    for tile in eval::waiting_tiles(hand, melds) {
        let mut winning_hand = *hand;
        if winning_hand.add(tile).is_err() {
            continue;
        }
        if let Some(fan) = win_fan(&winning_hand, melds, WinMethod::Discard) {
            best = Some(best.map_or(fan.total, |current: u8| current.max(fan.total)));
        }
    }
    best
}

/// Points transferred for one win.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WinPayment {
    /// Capped fan used as points per payer.
    pub fan: u8,
    /// Points paid by each payer.
    pub per_payer: u32,
    /// Number of payers.
    pub payer_count: usize,
    /// Total points received by the winner.
    pub total: u32,
}

/// Builds the payment fact for a win.
///
/// `fan_points` is capped defensively to [`FAN_CAP`]. For self-draw methods,
/// `active_opponents` is the number of other players who have not yet won.
/// Discard-like methods ignore it and have one payer.
pub fn win_payment(fan_points: u8, method: WinMethod, active_opponents: usize) -> WinPayment {
    let fan = fan_points.min(FAN_CAP);
    let payer_count = method.payer_count(active_opponents);
    WinPayment {
        fan,
        per_payer: fan as u32,
        payer_count,
        total: fan as u32 * payer_count as u32,
    }
}

/// Points transferred for one kong declaration.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KongPayment {
    /// Points paid by each payer.
    pub per_payer: u32,
    /// Number of payers.
    pub payer_count: usize,
    /// Total points received by the kong declarer.
    pub total: u32,
}

/// Builds the payment fact for a kong.
///
/// `Peng` is not a kong and yields a zero payment. For open kongs the single
/// payer is the discarder. For added and concealed kongs, every active
/// opponent (players who have not won) pays.
pub fn kong_payment(kind: MeldKind, active_opponents: usize) -> KongPayment {
    let (per_payer, payer_count) = match kind {
        MeldKind::Peng => (0, 0),
        MeldKind::OpenKong => (OPEN_KONG_POINTS, 1),
        MeldKind::AddedKong => (ADDED_KONG_POINTS, active_opponents),
        MeldKind::ConcealedKong => (CONCEALED_KONG_POINTS, active_opponents),
    };
    KongPayment {
        per_payer,
        payer_count,
        total: per_payer * payer_count as u32,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seat::Seat;
    use crate::tile::{Suit, Tile};

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank).unwrap()
    }

    fn hand(tiles: &[Tile]) -> Hand {
        Hand::from_tiles(tiles).unwrap()
    }

    fn wan(ranks: &[u8]) -> Vec<Tile> {
        ranks.iter().map(|&rank| tile(Suit::Wan, rank)).collect()
    }

    fn source(index: usize) -> Seat {
        Seat::new(index).unwrap()
    }

    fn contains_transfer(transfers: &[Transfer], from: usize, to: usize, points: u32) -> bool {
        let from = source(from);
        let to = source(to);
        transfers
            .iter()
            .any(|transfer| transfer.from == from && transfer.to == to && transfer.points == points)
    }

    #[test]
    fn exchange_requires_three_tiles_of_one_suit_in_hand() {
        let hand = hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Wan, 3),
            tile(Suit::Tong, 1),
        ]);
        assert!(is_valid_exchange(
            &hand,
            [tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 3)]
        ));
        assert!(is_valid_exchange(
            &hand,
            [tile(Suit::Wan, 3), tile(Suit::Wan, 3), tile(Suit::Wan, 1)]
        ));
        assert!(!is_valid_exchange(
            &hand,
            [tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Tong, 1)]
        ));
        assert!(!is_valid_exchange(
            &hand,
            [tile(Suit::Wan, 1), tile(Suit::Wan, 2), tile(Suit::Wan, 4)]
        ));
        assert!(!is_valid_exchange(
            &hand,
            [tile(Suit::Wan, 3), tile(Suit::Wan, 3), tile(Suit::Wan, 3)]
        ));
    }

    #[test]
    fn every_normal_deal_has_an_exchange_suit() {
        let mut hand = Hand::new();
        for rank in 1..=9 {
            hand.add(tile(Suit::Wan, rank)).unwrap();
        }
        for rank in 1..=4 {
            hand.add(tile(Suit::Tong, rank)).unwrap();
        }
        assert_eq!(hand.total(), 13);
        assert!(Suit::ALL
            .into_iter()
            .any(|suit| can_exchange_suit(&hand, suit)));
    }

    #[test]
    fn discard_must_clear_the_missing_suit_first() {
        let mixed = hand(&[tile(Suit::Wan, 1), tile(Suit::Tong, 2), tile(Suit::Tiao, 3)]);
        let missing = Suit::Tiao;
        assert!(can_discard(&mixed, missing, tile(Suit::Tiao, 3)));
        assert!(!can_discard(&mixed, missing, tile(Suit::Wan, 1)));
        assert!(!can_discard(&mixed, missing, tile(Suit::Tong, 2)));

        let cleared = hand(&[tile(Suit::Wan, 1), tile(Suit::Tong, 2)]);
        assert!(can_discard(&cleared, missing, tile(Suit::Wan, 1)));
        assert!(can_discard(&cleared, missing, tile(Suit::Tong, 2)));
        assert!(!can_discard(&cleared, missing, tile(Suit::Tiao, 3)));
    }

    #[test]
    fn no_missing_suit_tile_may_be_used_in_melds() {
        let missing = Suit::Tiao;
        let hand = hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Tong, 5),
        ]);
        assert!(can_peng(&hand, &[], missing, tile(Suit::Wan, 1)));
        assert!(can_open_kong(&hand, &[], missing, tile(Suit::Wan, 1)));
        assert!(can_concealed_kong(&hand, &[], missing, tile(Suit::Wan, 1)));
        assert!(!can_concealed_kong(&hand, &[], missing, tile(Suit::Wan, 2)));
        assert!(!can_peng(&hand, &[], missing, tile(Suit::Tiao, 1)));
    }

    #[test]
    fn melds_are_blocked_while_missing_suit_tiles_remain() {
        let missing = Suit::Tiao;
        let blocked = hand(&[
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 5),
            tile(Suit::Wan, 5),
            tile(Suit::Tiao, 1),
        ]);
        assert!(!can_peng(&blocked, &[], missing, tile(Suit::Wan, 5)));
        assert!(!can_open_kong(&blocked, &[], missing, tile(Suit::Wan, 5)));
        assert!(!can_added_kong(
            &blocked,
            &[Meld::peng(tile(Suit::Wan, 5), source(1))],
            missing,
            tile(Suit::Wan, 5)
        ));
    }

    #[test]
    fn added_kong_requires_an_existing_peng_and_the_fourth_tile() {
        let missing = Suit::Tiao;
        let hand = hand(&[tile(Suit::Wan, 5), tile(Suit::Wan, 6)]);
        let melds = [Meld::peng(tile(Suit::Wan, 5), source(2))];
        assert!(can_added_kong(&hand, &melds, missing, tile(Suit::Wan, 5)));
        assert!(!can_added_kong(&hand, &melds, missing, tile(Suit::Wan, 6)));
        assert!(!can_added_kong(
            &hand,
            &[Meld::open_kong(tile(Suit::Wan, 5), source(2))],
            missing,
            tile(Suit::Wan, 5)
        ));
    }

    #[test]
    fn hu_requires_no_missing_suit_tiles_and_a_valid_shape() {
        // 123W 456W 789W 111T 99T
        let mut tiles = wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        tiles.extend([
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 1),
            tile(Suit::Tong, 9),
            tile(Suit::Tong, 9),
        ]);
        let winning = hand(&tiles);
        assert!(can_hu(&winning, &[], Suit::Tiao));
        assert!(!can_hu(&winning, &[], Suit::Tong));

        let mut with_missing = winning;
        with_missing.remove(tile(Suit::Tong, 9)).unwrap();
        with_missing.add(tile(Suit::Tiao, 9)).unwrap();
        assert!(!can_hu(&with_missing, &[], Suit::Tiao));
    }

    #[test]
    fn pure_suit_checks_hand_and_melds() {
        let pure_hand = hand(&wan(&[1, 2, 3, 9, 9]));
        let melds = [Meld::peng(tile(Suit::Wan, 5), source(1))];
        assert!(is_pure_suit(&pure_hand, &melds));

        let mixed_hand = hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 2),
            tile(Suit::Wan, 3),
            tile(Suit::Tong, 9),
            tile(Suit::Tong, 9),
        ]);
        assert!(!is_pure_suit(&mixed_hand, &melds));

        let mixed_meld = [Meld::peng(tile(Suit::Tong, 5), source(1))];
        assert!(!is_pure_suit(&pure_hand, &mixed_meld));
        assert!(!is_pure_suit(&Hand::new(), &[]));
    }

    #[test]
    fn roots_count_four_identical_physical_tiles() {
        let no_root = hand(&wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert_eq!(root_count(&no_root, &[]), 0);

        let four_in_hand = hand(&wan(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert_eq!(root_count(&four_in_hand, &[]), 1);

        let no_root = hand(&wan(&[2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        let kong = [Meld::concealed_kong(tile(Suit::Wan, 1))];
        assert_eq!(root_count(&no_root, &kong), 1);

        let peng_plus_copy = [Meld::peng(tile(Suit::Wan, 1), source(1))];
        let hand_with_one = hand(&wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        assert_eq!(root_count(&hand_with_one, &peng_plus_copy), 1);
    }

    #[test]
    fn all_pungs_detects_triplet_only_standard_hands() {
        let pungs = hand(&wan(&[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]));
        assert!(is_all_pungs(&pungs, 0));

        let sequences = hand(&wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 9, 9]));
        assert!(!is_all_pungs(&sequences, 0));

        let pairs = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]));
        assert!(!is_all_pungs(&pairs, 0));
    }

    fn base_hand() -> Hand {
        // 123W 456W 789T 111B 99B: mixed suits, sequences, no roots.
        let mut tiles = wan(&[1, 2, 3, 4, 5, 6]);
        tiles.extend([
            tile(Suit::Tong, 7),
            tile(Suit::Tong, 8),
            tile(Suit::Tong, 9),
            tile(Suit::Tiao, 1),
            tile(Suit::Tiao, 1),
            tile(Suit::Tiao, 1),
            tile(Suit::Tiao, 9),
            tile(Suit::Tiao, 9),
        ]);
        hand(&tiles)
    }

    #[test]
    fn base_hand_scores_one_fan() {
        let fan = win_fan(&base_hand(), &[], WinMethod::Discard).unwrap();
        assert_eq!(fan.shape, WinShape::Standard);
        assert_eq!(fan.base, 1);
        assert_eq!(fan.pure_suit, 0);
        assert_eq!(fan.seven_pairs, 0);
        assert_eq!(fan.all_pungs, 0);
        assert_eq!(fan.roots, 0);
        assert_eq!(fan.event, 0);
        assert_eq!(fan.raw_total, 1);
        assert_eq!(fan.total, 1);
        assert!(!fan.is_capped());
    }

    #[test]
    fn pure_suit_scores_two_extra_fan() {
        let pure = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        let fan = win_fan(&pure, &[], WinMethod::Discard).unwrap();
        assert_eq!(fan.pure_suit, 2);
        assert_eq!(fan.roots, 0);
        assert_eq!(fan.all_pungs, 0);
        assert_eq!(fan.total, 3);
    }

    #[test]
    fn seven_pairs_scores_two_extra_fan() {
        let pairs = hand(&wan(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]));
        let fan = win_fan(&pairs, &[], WinMethod::Discard).unwrap();
        assert_eq!(fan.shape, WinShape::SevenPairs);
        assert_eq!(fan.seven_pairs, 2);
        assert_eq!(fan.pure_suit, 2);
        assert_eq!(fan.all_pungs, 0);
        assert_eq!(fan.total, 5);
    }

    #[test]
    fn all_pungs_scores_one_extra_fan() {
        let mixed = hand(&[
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Wan, 1),
            tile(Suit::Tong, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Tong, 2),
            tile(Suit::Tiao, 3),
            tile(Suit::Tiao, 3),
            tile(Suit::Tiao, 3),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 4),
            tile(Suit::Wan, 4),
            tile(Suit::Tong, 5),
            tile(Suit::Tong, 5),
        ]);
        let fan = win_fan(&mixed, &[], WinMethod::Discard).unwrap();
        assert_eq!(fan.all_pungs, 1);
        assert_eq!(fan.pure_suit, 0);
        assert_eq!(fan.total, 2);
    }

    #[test]
    fn roots_score_one_fan_each_and_apply_to_kongs() {
        let rooted = hand(&wan(&[1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]));
        let fan = win_fan(&rooted, &[], WinMethod::Discard).unwrap();
        assert_eq!(fan.roots, 1);
        assert_eq!(fan.pure_suit, 2);
        assert_eq!(fan.total, 4);

        let no_root = hand(&wan(&[2, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9]));
        let kong = [Meld::concealed_kong(tile(Suit::Wan, 1))];
        let kong_fan = win_fan(&no_root, &kong, WinMethod::Discard).unwrap();
        assert_eq!(kong_fan.roots, 1);
        assert_eq!(kong_fan.total, 4);
    }

    #[test]
    fn event_fan_is_one_for_every_win_method_except_discard() {
        let hand = base_hand();
        let expected = [
            (WinMethod::Discard, 0, 1),
            (WinMethod::SelfDraw, 1, 2),
            (WinMethod::AfterKong, 1, 2),
            (WinMethod::KongDiscard, 1, 2),
            (WinMethod::RobbingKong, 1, 2),
        ];
        for (method, event, total) in expected {
            let fan = win_fan(&hand, &[], method).unwrap();
            assert_eq!(fan.event, event);
            assert_eq!(fan.total, total);
        }
    }

    #[test]
    fn fan_is_capped_after_summing_items() {
        // 1111 2222 3333 44: seven pairs, pure, three roots, self draw.
        let hand = hand(&wan(&[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]));
        let fan = win_fan(&hand, &[], WinMethod::SelfDraw).unwrap();
        assert_eq!(fan.shape, WinShape::SevenPairs);
        assert_eq!(fan.roots, 3);
        assert_eq!(fan.raw_total, 9);
        assert_eq!(fan.total, FAN_CAP);
        assert!(fan.is_capped());
    }

    #[test]
    fn max_discard_win_fan_uses_the_best_wait() {
        let hand = hand(&wan(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 5, 5]));
        let fan = max_discard_win_fan(&hand, &[]).unwrap();
        assert_eq!(fan, 4);

        assert_eq!(max_discard_win_fan(&base_hand(), &[]), None);
    }

    #[test]
    fn kong_payments_follow_kind() {
        let mixed = kong_payment(MeldKind::Peng, 3);
        assert_eq!(mixed.total, 0);

        let concealed = kong_payment(MeldKind::ConcealedKong, 3);
        assert_eq!(concealed.per_payer, 2);
        assert_eq!(concealed.payer_count, 3);
        assert_eq!(concealed.total, 6);

        let added = kong_payment(MeldKind::AddedKong, 2);
        assert_eq!(added.per_payer, 1);
        assert_eq!(added.total, 2);

        let open = kong_payment(MeldKind::OpenKong, 3);
        assert_eq!(open.per_payer, 2);
        assert_eq!(open.payer_count, 1);
        assert_eq!(open.total, 2);
    }

    #[test]
    fn win_payments_depend_on_win_method() {
        let self_draw = win_payment(3, WinMethod::SelfDraw, 3);
        assert_eq!(self_draw.payer_count, 3);
        assert_eq!(self_draw.per_payer, 3);
        assert_eq!(self_draw.total, 9);

        let discard = win_payment(3, WinMethod::Discard, 3);
        assert_eq!(discard.payer_count, 1);
        assert_eq!(discard.total, 3);

        let after_kong = win_payment(3, WinMethod::AfterKong, 2);
        assert_eq!(after_kong.total, 6);

        let robbing = win_payment(3, WinMethod::RobbingKong, 3);
        assert_eq!(robbing.total, 3);

        assert_eq!(win_payment(9, WinMethod::Discard, 3).fan, FAN_CAP);
    }

    #[test]
    fn wall_exhaustion_settles_flower_pigs_then_big_calls() {
        let entries = [
            WallExhaustionEntry {
                seat: source(0),
                is_hu: true,
                is_flower_pig: false,
                is_tenpai: false,
                max_discard_fan: 0,
            },
            WallExhaustionEntry {
                seat: source(1),
                is_hu: false,
                is_flower_pig: true,
                is_tenpai: false,
                max_discard_fan: 0,
            },
            WallExhaustionEntry {
                seat: source(2),
                is_hu: false,
                is_flower_pig: false,
                is_tenpai: true,
                max_discard_fan: 3,
            },
            WallExhaustionEntry {
                seat: source(3),
                is_hu: false,
                is_flower_pig: false,
                is_tenpai: false,
                max_discard_fan: 0,
            },
        ];
        let transfers = settle_wall_exhaustion(&entries);
        assert_eq!(transfers.len(), 4);
        assert!(contains_transfer(&transfers, 1, 0, FLOWER_PIG_PENALTY));
        assert!(contains_transfer(&transfers, 1, 2, FLOWER_PIG_PENALTY));
        assert!(contains_transfer(&transfers, 1, 3, FLOWER_PIG_PENALTY));
        assert!(contains_transfer(&transfers, 3, 2, 3));
    }

    #[test]
    fn flower_pig_does_not_pay_big_call_and_hu_players_do_not_pay() {
        let entries = [
            WallExhaustionEntry {
                seat: source(0),
                is_hu: true,
                is_flower_pig: false,
                is_tenpai: false,
                max_discard_fan: 0,
            },
            WallExhaustionEntry {
                seat: source(1),
                is_hu: false,
                is_flower_pig: true,
                is_tenpai: false,
                max_discard_fan: 0,
            },
            WallExhaustionEntry {
                seat: source(2),
                is_hu: false,
                is_flower_pig: false,
                is_tenpai: true,
                max_discard_fan: 2,
            },
        ];
        let transfers = settle_wall_exhaustion(&entries);
        // The pig pays the hu player and the ready player only.
        assert!(contains_transfer(&transfers, 1, 0, FLOWER_PIG_PENALTY));
        assert!(contains_transfer(&transfers, 1, 2, FLOWER_PIG_PENALTY));
        // The hu player is not part of the big-call check.
        assert!(!transfers.iter().any(|transfer| transfer.from == source(0)));
        assert!(!transfers.iter().any(|transfer| transfer.to == source(1)));
    }

    #[test]
    fn kong_income_refund_conditions() {
        assert!(must_refund_kong_income(false, true, false));
        assert!(must_refund_kong_income(false, false, false));
        assert!(!must_refund_kong_income(false, false, true));
        assert!(!must_refund_kong_income(true, false, false));
        // A hu player can never be a flower pig in a legal game; hu wins.
        assert!(!must_refund_kong_income(true, true, false));
    }
}
