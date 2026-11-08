//! Fixed-seed sampled evidence for the candidate-ranking differential.

use super::super::*;

use super::{all_candidates, assert_physical, assert_rank_differential};

/// Deterministic LCG, matching the analysis sampling style without adding a
/// test-only dependency.
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

fn hand_from_counts(counts: &[u8; TILE_KINDS]) -> Hand {
    let mut hand = Hand::new();
    for kind in Tile::ALL {
        for _ in 0..counts[kind.index() as usize] {
            hand.add(kind).expect("at most four copies per kind");
        }
    }
    hand
}

struct SampledRankCase {
    hand: Hand,
    melds: Vec<Meld>,
    public: [u8; TILE_KINDS],
}

/// Builds one physical ranking input: a target-size concealed hand, distinct
/// meld kinds outside the missing suit, and public ownership that includes the
/// own melds.
fn sampled_rank_case(
    seed: u64,
    meld_count: usize,
    missing: Suit,
    include_missing_tiles: bool,
) -> SampledRankCase {
    let mut rng = Lcg::new(seed);

    let mut pool: Vec<Tile> = Tile::ALL
        .into_iter()
        .filter(|kind| kind.suit() != missing)
        .collect();
    for index in (1..pool.len()).rev() {
        let swap = rng.below(index + 1);
        pool.swap(index, swap);
    }

    let mut melds = Vec::new();
    let mut meld_copies = [0u8; TILE_KINDS];
    for (index, &kind) in pool.iter().take(meld_count).enumerate() {
        let meld = match index % 4 {
            0 => Meld::peng(kind, Seat::DEALER),
            1 => Meld::open_kong(kind, Seat::DEALER),
            2 => Meld::concealed_kong(kind),
            _ => Meld::added_kong(kind, Seat::DEALER),
        };
        meld_copies[kind.index() as usize] = meld.tile_count();
        melds.push(meld);
    }

    let mut counts = [0u8; TILE_KINDS];
    let mut remaining = 14 - 3 * meld_count;
    if include_missing_tiles {
        for _ in 0..=rng.below(2) {
            if remaining == 0 {
                break;
            }
            let kind = tile(missing, 1 + rng.below(9) as u8);
            let index = kind.index() as usize;
            if counts[index] < crate::tile::COPIES_PER_TILE {
                counts[index] += 1;
                remaining -= 1;
            }
        }
    }
    while remaining > 0 {
        let kind = pool[rng.below(pool.len())];
        let index = kind.index() as usize;
        if counts[index] + meld_copies[index] >= crate::tile::COPIES_PER_TILE {
            continue;
        }
        counts[index] += 1;
        remaining -= 1;
    }
    let hand = hand_from_counts(&counts);

    let mut public = meld_copies;
    for _ in 0..rng.below(6) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        if public[index] + counts[index] >= crate::tile::COPIES_PER_TILE {
            continue;
        }
        public[index] += 1;
    }
    for _ in 0..rng.below(3) {
        let kind = Tile::ALL[rng.below(Tile::ALL.len())];
        let index = kind.index() as usize;
        public[index] = crate::tile::COPIES_PER_TILE - counts[index];
    }

    SampledRankCase {
        hand,
        melds,
        public,
    }
}

fn sampled_candidates(hand: &Hand, variant: usize) -> Vec<Tile> {
    let mut distinct = all_candidates(hand);
    match variant {
        0 => distinct,
        1 => {
            distinct.reverse();
            distinct
        }
        2 => {
            distinct.truncate((distinct.len() / 2).max(1));
            distinct
        }
        _ => {
            let mut candidates = distinct.clone();
            if let Some(&first) = distinct.first() {
                candidates.push(first);
            }
            if let Some(&last) = distinct.last() {
                candidates.push(last);
            }
            if let Some(&unheld) = Tile::ALL.iter().find(|kind| hand.count(**kind) == 0) {
                candidates.push(unheld);
            }
            candidates
        }
    }
}

#[test]
fn sampled_rankings_match_the_removed_tile_reference() {
    // Seed base and stride follow the analysis sampling convention. The sweep
    // is 3 missing suits * 5 meld counts * 4 candidate variants = 60 physical
    // rankings; two variants keep missing-suit tiles in hand.
    const BASE: u64 = 0x5EED_0A11_0F0F_0001;
    const STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut index = 0u64;
    let mut missing_held = 0usize;
    let mut invalid_lists = 0usize;
    let mut surplus_hands = 0usize;
    for missing in Suit::ALL {
        for meld_count in 0..=4 {
            for variant in 0..4 {
                let seed = BASE.wrapping_add(STRIDE.wrapping_mul(index));
                index += 1;
                let case = sampled_rank_case(seed, meld_count, missing, variant % 2 == 1);
                assert_physical(&case.hand, &case.melds, missing, &case.public);
                let candidates = sampled_candidates(&case.hand, variant);
                missing_held += usize::from(case.hand.suit_count(missing) > 0);
                invalid_lists +=
                    usize::from(candidates.iter().any(|&kind| case.hand.count(kind) == 0));
                let mut evaluator = DistanceEvaluator::new(&case.melds, missing).unwrap();
                surplus_hands += usize::from(evaluator.deficit_with_surplus(&case.hand).1 != 0);
                let label = format!(
                    "sampled missing={missing} melds={meld_count} variant={variant} seed={seed:#x} counts={:?}",
                    case.hand.counts()
                );
                assert_rank_differential(
                    &label,
                    &case.hand,
                    &case.melds,
                    missing,
                    &case.public,
                    &candidates,
                    true,
                );
            }
        }
    }
    assert_eq!(index, 60, "sample sweep size changed");
    assert!(
        missing_held > 0,
        "sample sweep must retain missing-suit tiles"
    );
    assert!(
        invalid_lists > 0,
        "sample sweep must include invalid candidates"
    );
    assert!(
        surplus_hands > 0,
        "sample sweep must include optimal surplus"
    );
}
