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

/// Multi-seed checkpoint continuation with fresh policy instances.
///
/// Seeds cover both groups of the fixed `bounded-exact-trace-v1` corpus. A
/// game that ends before decision 24 is skipped exactly like the collector's
/// checkpoint rule, but at least one seed per group must reach the checkpoint.
#[test]
fn multi_seed_checkpoint_resume_matches_with_fresh_policies() {
    const CHECKPOINT: u64 = 24;
    let mut resumed_seeds = [0usize; 2];

    for seed in [7u64, 13, 24, 10_007, 10_026] {
        let full = heuristic_runner().run_new(seed).unwrap();
        let mut game = Game::new(seed);
        let mut limited = heuristic_runner().with_decision_limit(CHECKPOINT);

        match limited.run_game(&mut game) {
            Err(RunnerError::DecisionLimit { limit }) => assert_eq!(limit, CHECKPOINT),
            Ok(_) => {
                assert!(game.result().is_some(), "seed {seed}: early game result");
                continue;
            }
            Err(error) => panic!("seed {seed}: unexpected runner error {error:?}"),
        }
        assert!(
            game.result().is_none(),
            "seed {seed}: checkpoint should be mid-game"
        );

        let mut fork = game.clone();
        limited.set_decision_limit(10_000);
        let original = limited.run_game(&mut game).unwrap();
        let resumed = heuristic_runner().run_game(&mut fork).unwrap();

        assert_eq!(
            original.trace,
            full.trace[CHECKPOINT as usize..],
            "seed {seed}: continuation must replay the uninterrupted trace"
        );
        assert_eq!(original.decisions + CHECKPOINT, full.decisions);
        assert_eq!(original.trace, resumed.trace, "seed {seed}: fresh resume");
        assert_eq!(original.scores, resumed.scores, "seed {seed}: scores");
        assert_eq!(original.winners, resumed.winners, "seed {seed}: winners");
        assert_eq!(
            original.end_reason, resumed.end_reason,
            "seed {seed}: end reason"
        );
        assert_eq!(score_sum(&original.scores), 0, "seed {seed}: zero-sum");
        resumed_seeds[usize::from(seed >= 10_000)] += 1;
    }

    assert!(
        resumed_seeds.iter().all(|&count| count > 0),
        "both seed groups must reach the checkpoint"
    );
}
