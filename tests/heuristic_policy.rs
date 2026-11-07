//! Public Game/Runner checks for [`HeuristicPolicy`].
//!
//! These tests use only the public decision surface. They cover legal play,
//! seeded replay, and continuation from a cloned game with fresh policies,
//! which is equivalent to discarding every decision-local cache. They also pin
//! the public cross-decision cache contract: budget validation, the lazy status
//! lifecycle, shared groups, and identical traces and terminal results for
//! enabled, disabled, warm, and fresh contexts.

use sichuan_mahjong_sim::policy::{
    CacheConfigError, CacheState, HeuristicCache, HeuristicPolicy, Policy, DEFAULT_BUDGET_BYTES,
};
use sichuan_mahjong_sim::runner::{RunReport, Runner, RunnerError};
use sichuan_mahjong_sim::{Game, PLAYER_COUNT};

fn heuristic_runner() -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(HeuristicPolicy::new()) as Box<dyn Policy>
    }))
}

/// One runner whose four seats share `cache`.
fn runner_bound(cache: &HeuristicCache) -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(HeuristicPolicy::with_cache(cache.clone())) as Box<dyn Policy>
    }))
}

fn assert_same_run(first: &RunReport, second: &RunReport, context: &str) {
    assert_eq!(first.decisions, second.decisions, "{context}: decisions");
    assert_eq!(first.trace, second.trace, "{context}: trace");
    assert_eq!(first.scores, second.scores, "{context}: scores");
    assert_eq!(first.winners, second.winners, "{context}: winners");
    assert_eq!(first.end_reason, second.end_reason, "{context}: end reason");
    assert_eq!(score_sum(&first.scores), 0, "{context}: zero-sum");
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

/// Smallest accepted non-zero budget, observed through the public error.
fn minimum_budget() -> usize {
    match HeuristicCache::new(1) {
        Err(CacheConfigError::BudgetTooSmall { minimum_bytes, .. }) => minimum_bytes,
        other => panic!("expected a too-small budget error, got {other:?}"),
    }
}

#[test]
fn cache_budget_configuration_is_validated() {
    assert!(
        HeuristicCache::new(0).is_ok(),
        "zero is the explicit disabled budget"
    );
    let too_small = HeuristicCache::new(1).expect_err("one byte cannot hold a group");
    match too_small {
        CacheConfigError::BudgetTooSmall {
            budget_bytes,
            minimum_bytes,
        } => {
            assert_eq!(budget_bytes, 1);
            assert!(minimum_bytes > 1);
        }
        other => panic!("expected a too-small budget error, got {other:?}"),
    }
    let message = too_small.to_string();
    assert!(!message.is_empty(), "the error must be readable");
    fn through_std_error(error: &dyn std::error::Error) -> String {
        error.to_string()
    }
    assert_eq!(through_std_error(&too_small), message);

    let too_large = HeuristicCache::new(usize::MAX).expect_err("unaddressable budget");
    assert!(matches!(too_large, CacheConfigError::BudgetTooLarge { .. }));

    let default_status = HeuristicCache::default().status();
    assert_eq!(
        default_status.budget_bytes, DEFAULT_BUDGET_BYTES,
        "the default budget is the experimental candidate"
    );
    assert_ne!(
        default_status.budget_bytes, 0,
        "the default enables the cache"
    );
    assert_eq!(default_status.state, CacheState::Unallocated);
}

#[test]
fn cache_status_reports_lazy_allocation_and_real_use() {
    let cache = HeuristicCache::new(1 << 20).expect("1 MiB is a valid budget");
    let before = cache.status();
    assert_eq!(before.state, CacheState::Unallocated);
    assert_eq!(before.budget_bytes, 1 << 20);
    assert_eq!(before.allocated_bytes, 0);
    assert_eq!(before.capacity_entries, 0);
    assert_eq!(before.occupied_entries, 0);
    assert_eq!(before.hits + before.misses, 0);

    let mut runner = runner_bound(&cache);
    let report = runner.run_new(7).expect("legal self-play");
    assert!(report.decisions > 0);

    let status = cache.status();
    assert_eq!(status.state, CacheState::Ready);
    assert!(status.allocated_bytes > 0);
    assert!(
        status.allocated_bytes <= status.budget_bytes,
        "the table allocation must stay within the budget"
    );
    assert!(status.capacity_entries >= status.occupied_entries);
    assert!(status.occupied_entries > 0);
    assert!(
        status.hits > 0,
        "a full game reuses tables across decisions"
    );
    assert!(status.misses > 0);
    assert!(
        status.context_overhead_bytes > 0,
        "the shared context overhead is reported separately"
    );
    assert_eq!(
        status.context_overhead_bytes, before.context_overhead_bytes,
        "the context overhead is constant across the context lifetime"
    );

    // A one-group context is guaranteed to collide and evict, and must keep
    // the exact same game.
    let tiny = HeuristicCache::new(minimum_budget()).expect("one group");
    let mut tiny_runner = runner_bound(&tiny);
    let tiny_report = tiny_runner.run_new(7).expect("legal self-play");
    assert_same_run(&report, &tiny_report, "one-group cache");
    let tiny_status = tiny.status();
    assert_eq!(tiny_status.state, CacheState::Ready);
    assert_eq!(tiny_status.allocated_bytes, tiny_status.budget_bytes);
    assert_eq!(
        tiny_status.context_overhead_bytes, status.context_overhead_bytes,
        "the context overhead does not depend on the budget"
    );
    assert_eq!(tiny_status.capacity_entries, 4);
    assert_eq!(tiny_status.occupied_entries, 4);
    assert!(
        tiny_status.evictions > 0,
        "a full game needs more than four suit keys"
    );
}

#[test]
fn cache_context_does_not_change_game_traces() {
    let minimum = minimum_budget();
    for seed in [7u64, 13, 24] {
        // Baseline: every seat owns an independent default (enabled) context.
        let baseline = heuristic_runner().run_new(seed).expect("legal self-play");

        let disabled = {
            let cache = HeuristicCache::new(0).expect("zero disables");
            runner_bound(&cache).run_new(seed).expect("legal self-play")
        };
        let shared = {
            let group = HeuristicPolicy::shared_group();
            let policies: [Box<dyn Policy>; PLAYER_COUNT] =
                std::array::from_fn(|index| Box::new(group[index].clone()) as Box<dyn Policy>);
            Runner::new(policies)
                .run_new(seed)
                .expect("legal self-play")
        };
        let tiny = {
            let policies: [Box<dyn Policy>; PLAYER_COUNT] = std::array::from_fn(|_| {
                Box::new(HeuristicPolicy::with_cache(
                    HeuristicCache::new(minimum).expect("one group"),
                )) as Box<dyn Policy>
            });
            Runner::new(policies)
                .run_new(seed)
                .expect("legal self-play")
        };

        assert_same_run(&baseline, &disabled, &format!("seed {seed} disabled"));
        assert_same_run(&baseline, &shared, &format!("seed {seed} shared group"));
        assert_same_run(&baseline, &tiny, &format!("seed {seed} one-group"));
    }
}

#[test]
fn warm_cache_reuse_matches_cold_and_fresh_traces() {
    let cache = HeuristicCache::new(1 << 20).expect("valid budget");
    let mut runner = runner_bound(&cache);
    let cold = runner.run_new(7).expect("legal self-play");
    assert!(cache.status().occupied_entries > 0);

    let hits_before = cache.status().hits;
    let warm = runner.run_new(7).expect("legal self-play");
    assert!(
        cache.status().hits > hits_before,
        "the second game must reuse stored tables"
    );

    let fresh = heuristic_runner().run_new(7).expect("legal self-play");

    assert_same_run(&cold, &warm, "replayed game with a warm context");
    assert_same_run(&cold, &fresh, "same game with fresh default contexts");
}

#[test]
fn continuation_matches_with_warm_and_fresh_caches() {
    let cache = HeuristicCache::new(1 << 20).expect("valid budget");
    let mut game = Game::new(31);
    let mut limited = runner_bound(&cache).with_decision_limit(40);

    match limited.run_game(&mut game) {
        Err(RunnerError::DecisionLimit { limit }) => assert_eq!(limit, 40),
        other => panic!("expected a mid-game checkpoint, got {other:?}"),
    }
    assert!(game.result().is_none(), "checkpoint should be mid-game");

    let mut fork = game.clone();
    // The continuation keeps the warmed context from the first half.
    let warm = limited.run_game(&mut game).expect("legal continuation");
    // Fresh default policies start every context empty.
    let fresh = heuristic_runner()
        .run_game(&mut fork)
        .expect("legal continuation");

    assert!(warm.decisions > 0);
    assert!(
        cache.status().hits > 0,
        "the warm continuation must reuse tables"
    );
    assert_same_run(&warm, &fresh, "warm continuation");
}
