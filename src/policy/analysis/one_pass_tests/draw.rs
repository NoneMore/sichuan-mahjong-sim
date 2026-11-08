//! Draw-level evidence: the per-kind improvement equivalence, direct wins,
//! and the unseen-copy ukeire sum.

use super::super::test_fixtures::{boundary_states, meld_copies, own_public, tile};
use super::super::{one_pass_reference as baseline, ukeire, DistanceEvaluator};
use super::{bit, check_state, spec_ukeire, unseen_copies, CheckDepth};
use crate::rules;
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS};

#[test]
fn improving_mask_matches_draw_distance_and_direct_wins() {
    let mut direct_wins = 0usize;
    let mut non_winning_draws = 0usize;
    let mut improving_draws = 0usize;
    let fixtures = boundary_states();
    for state in &fixtures {
        let mut evaluator =
            DistanceEvaluator::new(&state.melds, state.missing).expect("at most four melds");
        let (deficit, mask) = evaluator.deficit_with_mask(&state.hand);
        let target_size = 14 - 3 * state.melds.len();
        let copies = meld_copies(&state.melds);
        for kind in Tile::ALL {
            let index = kind.index() as usize;
            // Only physically available draws are in the equivalence's domain.
            if state.hand.count(kind) + copies[index] >= COPIES_PER_TILE {
                continue;
            }
            let mut drawn = state.hand;
            drawn.add(kind).expect("a physically available draw");
            assert_eq!(drawn.total(), target_size);
            let (drawn_deficit, _) = evaluator.deficit_with_mask(&drawn);
            let can_hu = rules::can_hu(&drawn, &state.melds, state.missing);
            assert_eq!(
                can_hu,
                drawn_deficit == 0,
                "{}: can_hu must equal zero deficit for the legal drawn size, \
                 drawn hand {:?}",
                state.label,
                drawn
            );
            if can_hu {
                direct_wins += 1;
            } else {
                non_winning_draws += 1;
            }
            if kind.suit() != state.missing {
                let marked = mask & bit(kind) != 0;
                assert_eq!(
                    marked,
                    drawn_deficit < deficit,
                    "{}: bit {kind} must mean exactly D(H+t) < D(H)",
                    state.label
                );
                if marked {
                    improving_draws += 1;
                }
            }
        }
    }
    assert!(direct_wins > 0, "no directly winning draw was exercised");
    assert!(non_winning_draws > 0, "no non-winning draw was exercised");
    assert!(improving_draws > 0, "no improving draw was exercised");
}

#[test]
fn unseen_and_ukeire_follow_public_ownership() {
    let fixtures = boundary_states();
    let tenpai = fixtures
        .iter()
        .find(|state| state.label == "standard-tenpai")
        .expect("tenpai fixture");
    let mut evaluator = DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    let (deficit, mask) = evaluator.deficit_with_mask(&tenpai.hand);
    assert_eq!((deficit, mask), (1, 0x124));
    let waits = [tile(Suit::Wan, 3), tile(Suit::Wan, 6), tile(Suit::Wan, 9)];

    // Fully available: 3W has two unseen copies, 6W three, 9W two.
    let empty = own_public(tenpai);
    assert_eq!(empty, [0u8; TILE_KINDS]);
    for (kind, expected) in [(waits[0], 2), (waits[1], 3), (waits[2], 2)] {
        assert_eq!(unseen_copies(&tenpai.hand, &empty, kind), expected);
    }
    assert_eq!(spec_ukeire(&tenpai.hand, &empty, mask), 7);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &empty, deficit),
        7
    );

    // Per-kind production contributions: exhaust every other kind so the
    // production total is exactly this improving kind's unseen copies.
    for &kind in &waits {
        let mut only = empty;
        for other in Tile::ALL {
            if other != kind {
                only[other.index() as usize] = COPIES_PER_TILE - tenpai.hand.count(other);
            }
        }
        let contribution = unseen_copies(&tenpai.hand, &only, kind);
        assert!(contribution > 0, "the wait {kind} must have unseen copies");
        assert_eq!(
            ukeire(&mut evaluator, &tenpai.hand, &only, deficit),
            contribution,
            "production ukeire must count each improving kind independently"
        );
        let mut frozen =
            baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
        assert_eq!(
            baseline::ukeire(&mut frozen, &tenpai.hand, &only, deficit),
            contribution,
            "frozen ukeire must count the same per-kind contribution"
        );
    }

    // One public 6W leaves two unseen copies of that wait.
    let mut partial = empty;
    partial[waits[1].index() as usize] = 1;
    assert_eq!(unseen_copies(&tenpai.hand, &partial, waits[1]), 2);
    assert_eq!(spec_ukeire(&tenpai.hand, &partial, mask), 6);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &partial, deficit),
        6
    );

    // A non-improving public copy must not change availability.
    let mut irrelevant = empty;
    irrelevant[tile(Suit::Wan, 1).index() as usize] = 1;
    assert_eq!(spec_ukeire(&tenpai.hand, &irrelevant, mask), 7);

    // Public exhaustion leaves the structural mask unchanged and contributes
    // zero unseen copies for every improving kind.
    let mut exhausted = empty;
    for kind in waits {
        exhausted[kind.index() as usize] = COPIES_PER_TILE - tenpai.hand.count(kind);
        assert_eq!(unseen_copies(&tenpai.hand, &exhausted, kind), 0);
    }
    let (exhausted_deficit, exhausted_mask) = evaluator.deficit_with_mask(&tenpai.hand);
    assert_eq!((exhausted_deficit, exhausted_mask), (deficit, mask));
    assert_eq!(
        exhausted_mask, 0x124,
        "all structural bits survive exhaustion"
    );
    assert_eq!(spec_ukeire(&tenpai.hand, &exhausted, mask), 0);
    let mut frozen =
        baseline::DistanceEvaluator::new(&tenpai.melds, tenpai.missing).expect("no melds");
    assert_eq!(
        baseline::ukeire(&mut frozen, &tenpai.hand, &exhausted, deficit),
        0
    );

    // Own meld copies are public: the peng fixture still totals eight unseen
    // improving copies, and the capped 5W contributes nothing.
    for label in ["one-peng", "peng-capacity"] {
        let state = fixtures
            .iter()
            .find(|state| state.label == label)
            .expect("open-hand fixture");
        let (deficit, mask) = check_state(state, CheckDepth::Full);
        assert_eq!((deficit, mask), (1, 0x124));
        assert_eq!(spec_ukeire(&state.hand, &state.public, mask), 8);
        let mut frozen =
            baseline::DistanceEvaluator::new(&state.melds, state.missing).expect("melds");
        assert_eq!(
            baseline::ukeire(&mut frozen, &state.hand, &state.public, deficit),
            8
        );
    }
    let capped = fixtures
        .iter()
        .find(|state| state.label == "peng-capacity")
        .expect("peng fixture");
    assert_eq!(
        unseen_copies(&capped.hand, &capped.public, tile(Suit::Wan, 5)),
        0,
        "own meld copy plus the held copy exhaust the physical copies"
    );

    // The kong fixture's structural distance-two mask still sums only unseen
    // copies: 3 copies each for 1W/2W/6W and one for 9W.
    let kong = fixtures
        .iter()
        .find(|state| state.label == "concealed-kong-capacity")
        .expect("kong fixture");
    let (deficit, mask) = check_state(kong, CheckDepth::Full);
    assert_eq!(deficit, 2);
    assert_eq!(spec_ukeire(&kong.hand, &kong.public, mask), 10);
}
