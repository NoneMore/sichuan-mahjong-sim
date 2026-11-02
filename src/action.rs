//! Semantic actions, exchange directions, and declared melds.

use std::fmt;

use crate::seat::Seat;
use crate::tile::{Suit, Tile};

/// Direction in which the three exchanged tiles are passed.
///
/// The direction is the *giving* direction. Every player passes three tiles to
/// the seat named by [`ExchangeDirection::recipient`], so each player receives
/// three tiles from the inverse seat ([`ExchangeDirection::donor`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ExchangeDirection {
    /// Pass to the next seat in clockwise turn order.
    Clockwise,
    /// Pass to the previous seat in clockwise turn order.
    CounterClockwise,
    /// Pass across the table to the opposite seat.
    Opposite,
}

impl ExchangeDirection {
    /// All exchange directions in declaration order.
    pub const ALL: [ExchangeDirection; 3] = [
        ExchangeDirection::Clockwise,
        ExchangeDirection::CounterClockwise,
        ExchangeDirection::Opposite,
    ];

    /// The seat that receives tiles passed from `from`.
    pub const fn recipient(self, from: Seat) -> Seat {
        match self {
            ExchangeDirection::Clockwise => from.next_clockwise(),
            ExchangeDirection::CounterClockwise => from.next_counter_clockwise(),
            ExchangeDirection::Opposite => from.opposite(),
        }
    }

    /// The seat whose tiles are received by `to`.
    ///
    /// This is the inverse of [`ExchangeDirection::recipient`].
    pub const fn donor(self, to: Seat) -> Seat {
        match self {
            ExchangeDirection::Clockwise => to.next_counter_clockwise(),
            ExchangeDirection::CounterClockwise => to.next_clockwise(),
            ExchangeDirection::Opposite => to.opposite(),
        }
    }
}

/// Kind of a declared meld.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MeldKind {
    /// Three identical tiles claimed from a discard.
    Peng,
    /// Four identical tiles claimed from a discard.
    OpenKong,
    /// A peng upgraded with the fourth tile during the owner's turn.
    AddedKong,
    /// A concealed kong declared from the owner's concealed hand.
    ConcealedKong,
}

impl MeldKind {
    /// Number of physical tiles in the meld.
    pub const fn tile_count(self) -> u8 {
        match self {
            MeldKind::Peng => 3,
            MeldKind::OpenKong | MeldKind::AddedKong | MeldKind::ConcealedKong => 4,
        }
    }

    /// Whether this meld is a kong.
    pub const fn is_kong(self) -> bool {
        !matches!(self, MeldKind::Peng)
    }

    /// Whether the meld was declared from concealed tiles only.
    pub const fn is_concealed(self) -> bool {
        matches!(self, MeldKind::ConcealedKong)
    }
}

/// Errors produced by [`Meld::new`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeldError {
    /// A claimed meld must record the seat whose discard was claimed.
    MissingSource,
    /// A concealed kong cannot have a source seat.
    UnexpectedSource,
}

impl fmt::Display for MeldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeldError::MissingSource => f.write_str("claimed meld requires a source seat"),
            MeldError::UnexpectedSource => f.write_str("concealed kong cannot have a source seat"),
        }
    }
}

impl std::error::Error for MeldError {}

/// A declared meld.
///
/// `from` records the player whose discard started the meld. For a peng
/// upgraded to an added kong, `from` remains the original peng source.
/// Concealed kongs have no source.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Meld {
    kind: MeldKind,
    tile: Tile,
    from: Option<Seat>,
}

impl Meld {
    /// Builds a meld, enforcing the source rule for concealed kongs.
    pub fn new(kind: MeldKind, tile: Tile, from: Option<Seat>) -> Result<Self, MeldError> {
        match (kind, from) {
            (MeldKind::ConcealedKong, Some(_)) => Err(MeldError::UnexpectedSource),
            (MeldKind::ConcealedKong, None) => Ok(Self {
                kind,
                tile,
                from: None,
            }),
            (_, None) => Err(MeldError::MissingSource),
            (_, Some(source)) => Ok(Self {
                kind,
                tile,
                from: Some(source),
            }),
        }
    }

    /// A peng claimed from `source`.
    pub fn peng(tile: Tile, source: Seat) -> Self {
        Self {
            kind: MeldKind::Peng,
            tile,
            from: Some(source),
        }
    }

    /// An open kong claimed from `source`.
    pub fn open_kong(tile: Tile, source: Seat) -> Self {
        Self {
            kind: MeldKind::OpenKong,
            tile,
            from: Some(source),
        }
    }

    /// A peng upgraded to an added kong. `source` is the original peng source.
    pub fn added_kong(tile: Tile, source: Seat) -> Self {
        Self {
            kind: MeldKind::AddedKong,
            tile,
            from: Some(source),
        }
    }

    /// A concealed kong with no source.
    pub fn concealed_kong(tile: Tile) -> Self {
        Self {
            kind: MeldKind::ConcealedKong,
            tile,
            from: None,
        }
    }

    /// Kind of this meld.
    pub const fn kind(self) -> MeldKind {
        self.kind
    }

    /// Tile repeated by this meld.
    pub const fn tile(self) -> Tile {
        self.tile
    }

    /// Seat whose discard was claimed, if any.
    pub const fn from(self) -> Option<Seat> {
        self.from
    }

    /// Number of physical tiles in this meld.
    pub const fn tile_count(self) -> u8 {
        self.kind.tile_count()
    }

    /// Whether this meld is a kong.
    pub const fn is_kong(self) -> bool {
        self.kind.is_kong()
    }
}

/// A semantic action submitted to the game engine.
///
/// The enum deliberately describes player intent only; it does not encode
/// policy action indices or derived analysis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Action {
    /// Exchange three tiles of one suit during the opening exchange.
    Exchange([Tile; 3]),
    /// Declare the permanent missing suit during the opening phase.
    ChooseMissing(Suit),
    /// Discard one concealed tile.
    Discard(Tile),
    /// Claim the pending discard as a peng.
    Peng,
    /// Claim the pending discard as an open kong.
    OpenKong,
    /// Declare a concealed kong from the owner's hand.
    ConcealedKong(Tile),
    /// Upgrade an existing peng with the fourth tile.
    AddedKong(Tile),
    /// Declare a win on the current opportunity.
    Hu,
    /// Decline the current claim opportunity.
    Pass,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Tile;

    fn w(rank: u8) -> Tile {
        Tile::from_suit_rank(Suit::Wan, rank).unwrap()
    }

    #[test]
    fn exchange_recipient_and_donor_are_inverse() {
        for direction in ExchangeDirection::ALL {
            for seat in Seat::ALL {
                let recipient = direction.recipient(seat);
                assert_eq!(direction.donor(recipient), seat);
                assert_ne!(recipient, seat);
            }
            assert_eq!(
                direction.recipient(Seat::DEALER),
                match direction {
                    ExchangeDirection::Clockwise => Seat::new(1).unwrap(),
                    ExchangeDirection::CounterClockwise => Seat::new(3).unwrap(),
                    ExchangeDirection::Opposite => Seat::new(2).unwrap(),
                }
            );
        }
    }

    #[test]
    fn meld_kind_tile_counts() {
        assert_eq!(MeldKind::Peng.tile_count(), 3);
        assert_eq!(MeldKind::OpenKong.tile_count(), 4);
        assert_eq!(MeldKind::AddedKong.tile_count(), 4);
        assert_eq!(MeldKind::ConcealedKong.tile_count(), 4);
        assert!(!MeldKind::Peng.is_kong());
        assert!(MeldKind::OpenKong.is_kong());
        assert!(MeldKind::AddedKong.is_kong());
        assert!(MeldKind::ConcealedKong.is_kong());
        assert!(MeldKind::ConcealedKong.is_concealed());
        assert!(!MeldKind::AddedKong.is_concealed());
    }

    #[test]
    fn meld_constructors_and_validation() {
        let source = Seat::new(2).unwrap();
        let peng = Meld::peng(w(5), source);
        assert_eq!(peng.kind(), MeldKind::Peng);
        assert_eq!(peng.tile(), w(5));
        assert_eq!(peng.from(), Some(source));
        assert_eq!(peng.tile_count(), 3);

        let open = Meld::open_kong(w(5), source);
        assert_eq!(open.kind(), MeldKind::OpenKong);
        assert_eq!(open.tile_count(), 4);

        let added = Meld::added_kong(w(5), source);
        assert_eq!(added.kind(), MeldKind::AddedKong);
        assert_eq!(added.from(), Some(source));

        let concealed = Meld::concealed_kong(w(5));
        assert_eq!(concealed.kind(), MeldKind::ConcealedKong);
        assert_eq!(concealed.from(), None);
        assert_eq!(concealed.tile_count(), 4);
    }

    #[test]
    fn meld_new_rejects_inconsistent_sources() {
        assert_eq!(
            Meld::new(MeldKind::Peng, w(1), None),
            Err(MeldError::MissingSource)
        );
        assert_eq!(
            Meld::new(MeldKind::ConcealedKong, w(1), Some(Seat::DEALER)),
            Err(MeldError::UnexpectedSource)
        );
        assert!(Meld::new(MeldKind::AddedKong, w(1), Some(Seat::DEALER)).is_ok());
    }

    #[test]
    fn action_variants_are_constructible() {
        let actions = [
            Action::Exchange([w(1), w(2), w(3)]),
            Action::ChooseMissing(Suit::Tiao),
            Action::Discard(w(9)),
            Action::Peng,
            Action::OpenKong,
            Action::ConcealedKong(w(1)),
            Action::AddedKong(w(2)),
            Action::Hu,
            Action::Pass,
        ];
        assert_eq!(actions.len(), 9);
    }
}
