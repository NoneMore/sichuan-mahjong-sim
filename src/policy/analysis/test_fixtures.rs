//! Physical fixtures and deterministic sampling for the analysis differential
//! tests.
//!
//! Loaded once as a `#[cfg(test)]` module of `analysis`, so the one-pass and
//! discard-surplus evidence share one fixture definition instead of copies
//! that could drift.
//!
//! A [`State`] is a post-discard physical position: `13 - 3 * melds.len()`
//! concealed tiles, the declared melds, the missing suit, and the public
//! ownership vector. Every fixture and every generated state satisfies the
//! same ownership premises the analysis assumes:
//!
//! - own melds stay outside the missing suit, so a standard target with
//!   `4 - melds.len()` concealed sets is compatible with `rules::can_hu`;
//! - one kind never exceeds four physical copies across the concealed hand and
//!   the own melds;
//! - `public` counts each physical tile once: it includes every own meld tile,
//!   and `hand[t] + public[t] <= 4`.
//!
//! Observation-level ownership (claimed discards, multi-winner discard
//! references) is covered by the existing `heuristic::tests::calls` gate tests
//! and is not re-derived here; this module only pins the analysis-level
//! ownership premises.

use crate::action::Meld;
use crate::hand::Hand;
use crate::seat::Seat;
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

/// One legal post-discard analysis position.
#[derive(Clone, Debug)]
pub(super) struct State {
    pub(super) label: &'static str,
    pub(super) hand: Hand,
    pub(super) melds: Vec<Meld>,
    pub(super) missing: Suit,
    /// Public copies per kind, including the own meld tiles.
    pub(super) public: [u8; TILE_KINDS],
}

pub(super) fn tile(suit: Suit, rank: u8) -> Tile {
    Tile::from_suit_rank(suit, rank).expect("rank in range")
}

pub(super) fn hand_from_counts(counts: &[u8; TILE_KINDS]) -> Hand {
    let mut hand = Hand::new();
    for kind in Tile::ALL {
        for _ in 0..counts[kind.index() as usize] {
            hand.add(kind).expect("at most four copies per kind");
        }
    }
    hand
}

pub(super) fn suit_counts(hand: &Hand, suit: Suit) -> [u8; 9] {
    let mut local = [0u8; 9];
    let start = suit.index() * 9;
    local.copy_from_slice(&hand.counts()[start..start + 9]);
    local
}

/// Copies of each kind owned by the declared melds.
pub(super) fn meld_copies(melds: &[Meld]) -> [u8; TILE_KINDS] {
    let mut copies = [0u8; TILE_KINDS];
    for meld in melds {
        copies[meld.tile().index() as usize] += meld.tile_count();
    }
    copies
}

/// Per-suit target capacity: four copies minus own meld copies.
pub(super) fn capacity_by_suit(melds: &[Meld], missing: Suit) -> [[u8; 9]; 3] {
    let mut capacity = [[COPIES_PER_TILE; 9]; 3];
    capacity[missing.index()].fill(0);
    for meld in melds {
        let suit = meld.tile().suit_index();
        let rank = meld.tile().rank_index();
        capacity[suit][rank] = capacity[suit][rank].saturating_sub(meld.tile_count());
    }
    capacity
}

/// Per-kind target capacity: the same rule as [`capacity_by_suit`].
pub(super) fn capacity_by_kind(melds: &[Meld], missing: Suit) -> [u8; TILE_KINDS] {
    let mut capacity = [COPIES_PER_TILE; TILE_KINDS];
    for rank in 1..=9 {
        capacity[tile(missing, rank).index() as usize] = 0;
    }
    for meld in melds {
        let index = meld.tile().index() as usize;
        capacity[index] = capacity[index].saturating_sub(meld.tile_count());
    }
    capacity
}

impl State {
    fn new(
        label: &'static str,
        hand: Hand,
        melds: Vec<Meld>,
        missing: Suit,
        extra_public: &[(Tile, u8)],
    ) -> Self {
        let copies = meld_copies(&melds);
        let mut public = copies;
        for &(kind, count) in extra_public {
            public[kind.index() as usize] += count;
        }
        let state = Self {
            label,
            hand,
            melds,
            missing,
            public,
        };
        assert_physical(&state);
        state
    }
}

/// Asserts the ownership and size premises above; never weakens them.
pub(super) fn assert_physical(state: &State) {
    assert!(
        state.melds.len() <= 4,
        "{}: more than four declared melds",
        state.label
    );
    assert_eq!(
        state.hand.total(),
        13 - 3 * state.melds.len(),
        "{}: a post-discard hand must hold 13 - 3 * melds tiles",
        state.label
    );
    let copies = meld_copies(&state.melds);
    for meld in &state.melds {
        assert_ne!(
            meld.tile().suit(),
            state.missing,
            "{}: an own meld in the missing suit has no legal winning target",
            state.label
        );
    }
    for kind in Tile::ALL {
        let index = kind.index() as usize;
        let held = state.hand.count(kind);
        assert!(
            held + copies[index] <= COPIES_PER_TILE,
            "{}: {} is held {} times plus {} meld copies",
            state.label,
            kind,
            held,
            copies[index]
        );
        assert!(
            state.public[index] >= copies[index],
            "{}: public ownership must include the own meld copies of {}",
            state.label,
            kind
        );
        assert!(
            held + state.public[index] <= COPIES_PER_TILE,
            "{}: {} is held {} times and owned {} times publicly",
            state.label,
            kind,
            held,
            state.public[index]
        );
    }
}

/// Deterministic fixtures with a known expected outcome.
///
/// These pin every declared boundary: meld counts `0..=4` (peng, open kong,
/// concealed kong, added kong), more than one missing suit, cleared and
/// uncleared missing suit, own-capacity exhaustion, a standard-only decode, a
/// seven-pairs-only decode, seven-pairs-only tie, and cross-route ties whose
/// masks must be unioned.
pub(super) fn boundary_states() -> Vec<State> {
    let wan = Suit::Wan;
    let tong = Suit::Tong;
    let tiao = Suit::Tiao;
    vec![
        // 1. Standard tenpai with three distinct waits (within-standard union).
        State::new(
            "standard-tenpai",
            hand_from_counts(&counts(&[(wan, &[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9])])),
            vec![],
            tiao,
            &[],
        ),
        // 2. Larger exact distance with two live suits.
        State::new(
            "scattered",
            hand_from_counts(&counts(&[
                (wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (tong, &[1, 4, 7, 9]),
            ])),
            vec![],
            tiao,
            &[],
        ),
        // 3. Six pairs plus a singleton: the seven-pairs route waits on 6W and
        // the standard route also waits on 3W; both routes tie at distance one.
        State::new(
            "route-tie-overlap",
            hand_from_counts(&counts(&[(wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7])])),
            vec![],
            tiao,
            &[],
        ),
        // 4. Route tie with disjoint masks: seven pairs waits on 8W, the
        // standard route waits on 7W, so the combined mask is exactly both.
        State::new(
            "route-tie-disjoint",
            hand_from_counts(&counts(&[(wan, &[3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 8, 9, 9])])),
            vec![],
            tiao,
            &[],
        ),
        // 5. Seven-pairs internal tie at distance two: any of the three odd
        // Tong singletons supplies one pair unit, and the standard route is
        // strictly worse.
        State::new(
            "seven-pairs-internal-tie",
            hand_from_counts(&counts(&[
                (wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
                (tong, &[1, 4, 7]),
            ])),
            vec![],
            tiao,
            &[],
        ),
        // 6. A held triple plus five pairs: seven pairs waits only on 1W and
        // the standard route is worse, so only the quadded kind is marked.
        State::new(
            "seven-pairs-triple",
            hand_from_counts(&counts(&[(wan, &[1, 1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 9, 9])])),
            vec![],
            tong,
            &[],
        ),
        // 7. Missed-clear fixture: one missing-suit tile stays in the hand and
        // is free surplus. No missing-suit bit may appear.
        State::new(
            "uncleared-missing-tile",
            hand_from_counts(&counts(&[
                (wan, &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
                (tiao, &[1]),
            ])),
            vec![],
            tiao,
            &[],
        ),
        // 8. Open hand: one peng leaves three concealed sets and a pair.
        State::new(
            "one-peng",
            hand_from_counts(&counts(&[(wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 9])])),
            vec![Meld::peng(tile(tong, 6), Seat::DEALER)],
            tiao,
            &[],
        ),
        // 9. Own peng caps its rank at one concealed copy: 5W is held once and
        // publicly owned three times, so it can never be marked or unseen.
        State::new(
            "peng-capacity",
            hand_from_counts(&counts(&[(wan, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 9])])),
            vec![Meld::peng(tile(wan, 5), Seat::DEALER)],
            tiao,
            &[],
        ),
        // 10. Own concealed kong exhausts 3W; the structural 3W wait is gone.
        State::new(
            "concealed-kong-capacity",
            hand_from_counts(&counts(&[(wan, &[1, 2, 4, 5, 6, 7, 8, 9, 9, 9])])),
            vec![Meld::concealed_kong(tile(wan, 3))],
            tiao,
            &[],
        ),
        // 11. Added kong consumes the fourth 8W copy; same capacity rule.
        State::new(
            "added-kong-capacity",
            hand_from_counts(&counts(&[(wan, &[1, 2, 3, 4, 5, 6, 7, 9, 9, 9])])),
            vec![Meld::added_kong(tile(wan, 8), Seat::DEALER)],
            tiao,
            &[],
        ),
        // 12. Two pengs leave two concealed sets plus the pair.
        State::new(
            "two-peng",
            hand_from_counts(&counts(&[(wan, &[1, 2, 3, 4, 5, 6, 9])])),
            vec![
                Meld::peng(tile(tong, 6), Seat::DEALER),
                Meld::peng(tile(tong, 7), Seat::DEALER),
            ],
            tiao,
            &[],
        ),
        // 13. Three melds leave one concealed set plus the pair.
        State::new(
            "three-meld",
            hand_from_counts(&counts(&[(wan, &[1, 1, 2, 2])])),
            vec![
                Meld::peng(tile(tong, 6), Seat::DEALER),
                Meld::peng(tile(tong, 7), Seat::DEALER),
                Meld::open_kong(tile(tong, 9), Seat::DEALER),
            ],
            tiao,
            &[],
        ),
        // 14. Four melds leave the pair only (zero concealed sets).
        State::new(
            "four-meld",
            hand_from_counts(&counts(&[(wan, &[3])])),
            vec![
                Meld::peng(tile(tong, 5), Seat::DEALER),
                Meld::peng(tile(tong, 6), Seat::DEALER),
                Meld::concealed_kong(tile(tong, 1)),
                Meld::open_kong(tile(tong, 9), Seat::DEALER),
            ],
            tiao,
            &[],
        ),
        // 15. Four melds under a different missing suit still leave a pair.
        State::new(
            "four-meld-missing-wan",
            hand_from_counts(&counts(&[(tong, &[3])])),
            vec![
                Meld::peng(tile(tong, 5), Seat::DEALER),
                Meld::added_kong(tile(tiao, 1), Seat::DEALER),
                Meld::concealed_kong(tile(tiao, 9)),
                Meld::open_kong(tile(tiao, 2), Seat::DEALER),
            ],
            wan,
            &[],
        ),
    ]
}

/// Builds a counted hand from `(suit, one-based ranks)` groups.
fn counts(groups: &[(Suit, &[u8])]) -> [u8; TILE_KINDS] {
    let mut counts = [0u8; TILE_KINDS];
    for &(suit, ranks) in groups {
        for &rank in ranks {
            counts[tile(suit, rank).index() as usize] += 1;
        }
    }
    counts
}

/// Small deterministic LCG; no test-only dependency is added.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        ((self.next() >> 33) % bound as u64) as usize
    }
}

/// Deterministic sample of legal physical positions.
///
/// Seed base `0x5EED_1A55_0F0F_0001`; state `index` uses base plus
/// `0x9E37_79B9_7F4A_7C15 * index`. Each (missing suit, meld count `0..=4`)
/// cell contributes five samples: three with a cleared missing suit and two
/// that keep a missing-suit tile in hand, so the sweep is
/// `3 suits * 5 meld counts * 5 = 75` states. Sampling is statistical coverage
/// of the combined parameter space, not a claim that every individual tile
/// pattern has been enumerated.
pub(super) fn sampled_states() -> Vec<State> {
    const BASE: u64 = 0x5EED_1A55_0F0F_0001;
    const STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut states = Vec::new();
    let mut index = 0u64;
    for missing in Suit::ALL {
        for meld_count in 0..=4 {
            for include_missing_tiles in [false, true, false, true, false] {
                let seed = BASE.wrapping_add(STRIDE.wrapping_mul(index));
                states.push(sampled_state(
                    seed,
                    meld_count,
                    missing,
                    include_missing_tiles,
                ));
                index += 1;
            }
        }
    }
    states
}

fn sampled_state(
    seed: u64,
    meld_count: usize,
    missing: Suit,
    include_missing_tiles: bool,
) -> State {
    let mut rng = Lcg::new(seed);

    // A shuffled pool of non-missing kinds gives distinct meld tiles, so the
    // four-copy budget stays explicit.
    let mut pool: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .collect();
    for index in (1..pool.len()).rev() {
        let swap = rng.below(index + 1);
        pool.swap(index, swap);
    }

    let mut melds = Vec::new();
    let mut copies = [0u8; TILE_KINDS];
    for (index, &kind) in pool.iter().take(meld_count).enumerate() {
        let meld = match index % 4 {
            0 => Meld::peng(kind, Seat::DEALER),
            1 => Meld::open_kong(kind, Seat::DEALER),
            2 => Meld::concealed_kong(kind),
            _ => Meld::added_kong(kind, Seat::DEALER),
        };
        copies[kind.index() as usize] = meld.tile_count();
        melds.push(meld);
    }

    let mut counts = [0u8; TILE_KINDS];
    let mut remaining = 13 - 3 * meld_count;
    if include_missing_tiles {
        for _ in 0..=rng.below(2) {
            if remaining == 0 {
                break;
            }
            let kind = tile(missing, 1 + rng.below(9) as u8);
            let index = kind.index() as usize;
            if counts[index] < COPIES_PER_TILE {
                counts[index] += 1;
                remaining -= 1;
            }
        }
    }
    while remaining > 0 {
        let kind = pool[rng.below(pool.len())];
        let index = kind.index() as usize;
        if counts[index] + copies[index] >= COPIES_PER_TILE {
            continue;
        }
        counts[index] += 1;
        remaining -= 1;
    }
    let hand = hand_from_counts(&counts);

    // Own melds are public; extra copies model other visible melds and
    // unclaimed discards. Some kinds are filled to their physical limit so the
    // sweep also includes publicly exhausted improving kinds.
    let mut public = copies;
    for _ in 0..rng.below(7) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        if public[index] + counts[index] >= COPIES_PER_TILE {
            continue;
        }
        public[index] += 1;
    }
    for _ in 0..rng.below(3) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        public[index] = COPIES_PER_TILE - counts[index];
    }

    let state = State {
        label: "sampled",
        hand,
        melds,
        missing,
        public,
    };
    assert_physical(&state);
    state
}

/// Rebuilds a state's public ownership with the meld copies only.
pub(super) fn own_public(state: &State) -> [u8; TILE_KINDS] {
    meld_copies(&state.melds)
}
