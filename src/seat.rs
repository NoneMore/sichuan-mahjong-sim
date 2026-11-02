//! Seat identities and clockwise table geometry.
//!
//! Seats are numbered `0..=3` in clockwise turn order. Seat 0 is the dealer.
//! The next active seat is always found by moving clockwise.

use std::fmt;

/// Number of players at the table.
pub const PLAYER_COUNT: usize = 4;

/// A player seat in `0..=3`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Seat(u8);

impl Seat {
    /// Seat of the dealer.
    pub const DEALER: Seat = Seat(0);

    /// All seats in clockwise order.
    pub const ALL: [Seat; PLAYER_COUNT] = [Seat(0), Seat(1), Seat(2), Seat(3)];

    /// Builds a seat from an index, rejecting values outside `0..=3`.
    pub const fn new(index: usize) -> Option<Seat> {
        if index < PLAYER_COUNT {
            Some(Seat(index as u8))
        } else {
            None
        }
    }

    /// Zero-based seat index in `0..=3`.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The next seat in clockwise order.
    pub const fn next_clockwise(self) -> Seat {
        Seat((self.0 + 1) % PLAYER_COUNT as u8)
    }

    /// The next seat in counter-clockwise order.
    pub const fn next_counter_clockwise(self) -> Seat {
        Seat((self.0 + PLAYER_COUNT as u8 - 1) % PLAYER_COUNT as u8)
    }

    /// The opposite seat, two places away.
    pub const fn opposite(self) -> Seat {
        Seat((self.0 + 2) % PLAYER_COUNT as u8)
    }

    /// Number of clockwise steps from `self` to `other`, in `0..=3`.
    pub const fn clockwise_distance(self, other: Seat) -> u8 {
        (other.0 + PLAYER_COUNT as u8 - self.0) % PLAYER_COUNT as u8
    }

    /// Whether the two seats face each other.
    pub const fn is_opposite(self, other: Seat) -> bool {
        self.opposite().index() == other.index()
    }
}

impl fmt::Display for Seat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seat {}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seat_indices_are_checked() {
        for index in 0..PLAYER_COUNT {
            let seat = Seat::new(index).unwrap();
            assert_eq!(seat.index(), index);
        }
        assert_eq!(Seat::new(4), None);
        assert_eq!(Seat::new(usize::MAX), None);
        assert_eq!(Seat::DEALER, Seat::new(0).unwrap());
    }

    #[test]
    fn rotations_cycle_through_all_seats() {
        let mut seat = Seat::DEALER;
        let clockwise: Vec<_> = (0..PLAYER_COUNT)
            .map(|_| {
                let current = seat;
                seat = seat.next_clockwise();
                current
            })
            .collect();
        assert_eq!(clockwise, Seat::ALL.to_vec());

        let mut seat = Seat::DEALER;
        let counter: Vec<_> = (0..PLAYER_COUNT)
            .map(|_| {
                let current = seat;
                seat = seat.next_counter_clockwise();
                current
            })
            .collect();
        assert_eq!(counter, vec![Seat(0), Seat(3), Seat(2), Seat(1)]);
    }

    #[test]
    fn opposite_and_distances_are_consistent() {
        for seat in Seat::ALL {
            assert_eq!(seat.opposite().opposite(), seat);
            assert!(seat.is_opposite(seat.opposite()));
            assert!(!seat.is_opposite(seat));
            assert_eq!(seat.clockwise_distance(seat), 0);
            assert_eq!(seat.clockwise_distance(seat.next_clockwise()), 1);
            assert_eq!(seat.clockwise_distance(seat.opposite()), 2);
            assert_eq!(seat.clockwise_distance(seat.next_counter_clockwise()), 3);
        }
    }

    #[test]
    fn seat_display_includes_index() {
        assert_eq!(Seat::new(2).unwrap().to_string(), "seat 2");
    }
}
