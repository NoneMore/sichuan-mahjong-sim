//! Public Game/Runner checks for [`HeuristicPolicy`].
//!
//! These tests use only the public decision surface. They cover legal play,
//! seeded replay, and continuation from a cloned game with fresh policies,
//! which is equivalent to discarding every decision-local cache.

use sichuan_mahjong_sim::policy::{HeuristicPolicy, Policy};
use sichuan_mahjong_sim::runner::{Runner, RunnerError};
use sichuan_mahjong_sim::Game;

fn heuristic_runner() -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(HeuristicPolicy::new()) as Box<dyn Policy>
    }))
}

fn score_sum(scores: &[i32; 4]) -> i32 {
    scores.iter().sum()
}

#[test]
fn heuristic_games_are_legal_zero_sum_and_seed_replayable() {
    for seed in [1, 2, 7] {
        let first = heuristic_runner().run_new(seed).unwrap();
        let second = heuristic_runner().run_new(seed).unwrap();

        assert!(first.decisions > 0, "seed {seed} made no decisions");
        assert_eq!(first.trace, second.trace, "seed {seed} did not replay");
        assert_eq!(first.scores, second.scores);
        assert_eq!(first.winners, second.winners);
        assert_eq!(first.end_reason, second.end_reason);
        assert_eq!(score_sum(&first.scores), 0);
    }
}

#[test]
fn continuation_from_a_cloned_game_matches_with_fresh_policies() {
    let mut game = Game::new(31);
    let mut limited = heuristic_runner().with_decision_limit(40);

    match limited.run_game(&mut game) {
        Err(RunnerError::DecisionLimit { limit }) => assert_eq!(limit, 40),
        other => panic!("expected a mid-game checkpoint, got {other:?}"),
    }
    assert!(game.result().is_none(), "checkpoint should be mid-game");

    let mut fork = game.clone();
    let original = heuristic_runner().run_game(&mut game).unwrap();
    // Fresh policies hold no decision cache from before the checkpoint.
    let resumed = heuristic_runner().run_game(&mut fork).unwrap();

    assert!(original.decisions > 0);
    assert_eq!(original.trace, resumed.trace);
    assert_eq!(original.scores, resumed.scores);
    assert_eq!(original.winners, resumed.winners);
    assert_eq!(original.end_reason, resumed.end_reason);
    assert_eq!(score_sum(&original.scores), 0);
}
