//! Exhaustive small-state oracle for the fixed exhaustive-draw payment matrix.
use sichuan_mahjong_sim::{
    must_refund_kong_income, settle_wall_exhaustion, Seat, WallExhaustionEntry, FLOWER_PIG_PENALTY,
};

#[test]
fn all_four_seat_settlement_categories_match_pairwise_payment_rules() {
    // Category 0: won; 1: flower pig; 2: tenpai; 3: non-tenpai.
    for encoded in 0usize..256 {
        let categories: [usize; 4] = std::array::from_fn(|i| (encoded >> (2 * i)) & 3);
        let entries: [WallExhaustionEntry; 4] = std::array::from_fn(|i| {
            let category = categories[i];
            WallExhaustionEntry {
                seat: Seat::new(i).unwrap(),
                is_hu: category == 0,
                is_flower_pig: category == 1,
                is_tenpai: category == 2,
                max_discard_fan: if category == 2 { i as u8 + 1 } else { 0 },
            }
        });
        let mut actual = [[0u32; 4]; 4];
        let mut scores = [0i32; 4];
        for transfer in settle_wall_exhaustion(&entries) {
            let from = transfer.from.index();
            let to = transfer.to.index();
            assert_ne!(from, to);
            assert!(transfer.points > 0);
            actual[from][to] += transfer.points;
            scores[from] -= transfer.points as i32;
            scores[to] += transfer.points as i32;
            assert_eq!(scores.iter().sum::<i32>(), 0);
        }
        for from in 0..4 {
            for to in 0..4 {
                let expected = if from == to {
                    0
                } else if categories[from] == 1 && categories[to] != 1 {
                    FLOWER_PIG_PENALTY
                } else if categories[from] == 3 && categories[to] == 2 {
                    to as u32 + 1
                } else {
                    0
                };
                assert_eq!(
                    actual[from][to], expected,
                    "categories={categories:?}, payer={from}, recipient={to}"
                );
            }
            assert_eq!(
                must_refund_kong_income(
                    entries[from].is_hu,
                    entries[from].is_flower_pig,
                    entries[from].is_tenpai
                ),
                matches!(categories[from], 1 | 3)
            );
        }
    }
}
