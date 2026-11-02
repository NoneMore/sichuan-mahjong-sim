//! Seeded legal-action stress tests of tile/accounting conservation.
use sichuan_mahjong_sim::{Action, Game, LegalActions, Policy, RandomPolicy};

#[test]
fn randomized_games_conserve_tiles_and_scores_after_every_transition() {
    for seed in 0..256u64 {
        let mut game = Game::new(seed);
        let mut policies: [RandomPolicy; 4] =
            std::array::from_fn(|i| RandomPolicy::new(seed ^ ((i as u64 + 1) * 0x1234567)));
        let mut decisions = 0;
        loop {
            let choice = game.next_decision().map(|decision| {
                let mut action = policies[decision.seat.index()].choose(&decision);
                // Exercise voluntarily declining Hu as well as accepting it.
                if seed % 2 == 0 && action == Action::Hu {
                    if let LegalActions::Actions(actions) = decision.legal_actions {
                        if let Some(&alternative) = actions.iter().rev().find(|&&a| a != Action::Hu)
                        {
                            action = alternative;
                        }
                    }
                }
                (decision.seat, action)
            });
            game.debug_validate()
                .unwrap_or_else(|why| panic!("seed {seed}, before decision {decisions}: {why}"));
            let Some((seat, action)) = choice else {
                break;
            };
            game.submit(seat, action).unwrap();
            game.debug_validate()
                .unwrap_or_else(|why| panic!("seed {seed}, after decision {decisions}: {why}"));
            decisions += 1;
            assert!(decisions < 2_000, "seed {seed} failed to terminate");
        }
        assert_eq!(game.result().unwrap().scores.iter().sum::<i32>(), 0);
    }
}
