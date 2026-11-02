//! End-to-end checks for the native execution slice.
//!
//! These tests use only the public Game decision surface and the public
//! policy/runner API. They cover new-game driving, continuation from a cloned
//! mid-game with stateless and stateful policies, seeded replay, illegal policy
//! reporting, the per-call decision limit, zero-game rejection, and zero-sum
//! settlement.

use std::process::{Command, Output};

use sichuan_mahjong_sim::policy::{random_seat_seed, FirstLegalPolicy, Policy, RandomPolicy};
use sichuan_mahjong_sim::runner::{FinishReason, Runner, RunnerError, DEFAULT_DECISION_LIMIT};
use sichuan_mahjong_sim::{Action, Decision, Game, Seat};

fn submit_one(game: &mut Game, policy: &mut FirstLegalPolicy) -> bool {
    let (seat, action) = match game.next_decision() {
        Some(decision) => {
            let seat = decision.seat;
            let action = policy.choose(&decision);
            (seat, action)
        }
        None => return false,
    };
    game.submit(seat, action)
        .expect("first-legal policy must submit a legal action");
    true
}

fn score_sum(report: &sichuan_mahjong_sim::runner::RunReport) -> i32 {
    report.scores.iter().sum()
}

#[test]
fn new_games_run_to_a_zero_sum_terminal_result() {
    for seed in [0, 1, 7, 99, u64::MAX] {
        let mut runner = Runner::first_legal();
        let report = runner.run_new(seed).unwrap();

        assert!(report.decisions > 0, "seed {seed} made no decisions");
        assert!(
            report.decisions < 2_000,
            "seed {seed} made suspiciously many decisions"
        );
        assert_eq!(report.trace.len() as u64, report.decisions);
        assert_eq!(
            score_sum(&report),
            0,
            "seed {seed} produced non-zero-sum scores"
        );

        let winners = report.winners.iter().filter(|winner| **winner).count();
        match report.end_reason {
            FinishReason::ThreeWinners => {
                assert!(
                    winners >= 3,
                    "seed {seed} ended with only {winners} winners"
                );
            }
            FinishReason::WallExhausted => {}
        }
    }
}

#[test]
fn continuation_from_a_cloned_mid_game_matches() {
    let mut game = Game::new(1_234);
    let mut policy = FirstLegalPolicy::new();
    let mut advanced = 0;
    while advanced < 24 {
        if !submit_one(&mut game, &mut policy) {
            break;
        }
        advanced += 1;
    }
    assert_eq!(advanced, 24, "game ended before the intended checkpoint");
    assert!(game.result().is_none(), "checkpoint should be mid-game");

    let mut fork = game.clone();
    let mut original_runner = Runner::first_legal();
    let mut fork_runner = Runner::first_legal();

    let original = original_runner.run_game(&mut game).unwrap();
    let resumed = fork_runner.run_game(&mut fork).unwrap();

    assert_eq!(original.decisions, resumed.decisions);
    assert_eq!(original.trace, resumed.trace);
    assert_eq!(original.scores, resumed.scores);
    assert_eq!(original.winners, resumed.winners);
    assert_eq!(original.end_reason, resumed.end_reason);
    assert_eq!(score_sum(&original), 0);
}

#[test]
fn seeded_random_policy_replays_the_same_decisions() {
    let mut first = Runner::random(0x5EED_1234);
    let mut second = Runner::random(0x5EED_1234);

    let first = first.run_new(55).unwrap();
    let second = second.run_new(55).unwrap();

    assert_eq!(first.decisions, second.decisions);
    assert_eq!(first.trace, second.trace);
    assert_eq!(first.scores, second.scores);
    assert_eq!(first.winners, second.winners);
    assert_eq!(first.end_reason, second.end_reason);
    assert_eq!(score_sum(&first), 0);
}

fn random_policies(policy_seed: u64) -> [RandomPolicy; 4] {
    std::array::from_fn(|index| {
        let seat = Seat::new(index).expect("seat index is in range");
        RandomPolicy::new(random_seat_seed(policy_seed, seat))
    })
}

fn runner_from(policies: [RandomPolicy; 4]) -> Runner {
    Runner::new(policies.map(|policy| Box::new(policy) as Box<dyn Policy>))
}

#[test]
fn random_policy_continuation_needs_the_game_and_the_policy_streams() {
    let game_seed = 4_242;
    let policy_seed = 0x00C0_FFEE;

    let mut game = Game::new(game_seed);
    let mut policies = random_policies(policy_seed);

    let mut advanced = 0;
    while advanced < 40 {
        let Some(decision) = game.next_decision() else {
            break;
        };
        let seat = decision.seat;
        let action = policies[seat.index()].choose(&decision);
        game.submit(seat, action).expect("random policy is legal");
        advanced += 1;
    }
    assert_eq!(advanced, 40, "game ended before the intended checkpoint");
    assert!(game.result().is_none(), "checkpoint should be mid-game");

    // A random checkpoint is the cloned game *and* every stateful policy.
    let mut checkpoint_game = game.clone();
    let checkpoint_policies = policies.clone();

    let original = runner_from(policies).run_game(&mut game).unwrap();
    let resumed = runner_from(checkpoint_policies.clone())
        .run_game(&mut checkpoint_game)
        .unwrap();

    assert!(original.decisions > 0, "no decisions after the checkpoint");
    assert_eq!(original.decisions, resumed.decisions);
    assert_eq!(original.trace, resumed.trace);
    assert_eq!(original.scores, resumed.scores);
    assert_eq!(original.winners, resumed.winners);
    assert_eq!(original.end_reason, resumed.end_reason);
    assert_eq!(score_sum(&original), 0);

    // Rebuilding a runner from the seed restarts each stream instead of
    // restoring the checkpoint, so game-only cloning is not enough here.
    let rebuilt = random_policies(policy_seed);
    for (index, policy) in checkpoint_policies.iter().enumerate() {
        assert_ne!(
            rebuilt[index].rng_state(),
            policy.rng_state(),
            "seat {index} stream should have advanced past its seed"
        );
    }
}

#[test]
fn decision_limit_is_per_call_and_preserves_accepted_actions() {
    const LIMIT: u64 = 4;

    let mut reference_game = Game::new(808);
    let reference = Runner::first_legal().run_game(&mut reference_game).unwrap();
    assert!(reference.decisions > LIMIT);

    let mut game = Game::new(808);
    let mut runner = Runner::first_legal().with_decision_limit(LIMIT);
    match runner.run_game(&mut game).unwrap_err() {
        RunnerError::DecisionLimit { limit } => assert_eq!(limit, LIMIT),
        other => panic!("expected DecisionLimit, got {other:?}"),
    }
    let message = RunnerError::DecisionLimit { limit: LIMIT }.to_string();
    assert!(message.contains("reached"), "unexpected message: {message}");
    assert!(!message.contains("exceeded"));
    assert!(
        game.result().is_none(),
        "the limit must not finish the game"
    );

    // Accepted actions stay in the game: only the count resets per call.
    runner.set_decision_limit(DEFAULT_DECISION_LIMIT);
    let resumed = runner.run_game(&mut game).unwrap();
    assert_eq!(resumed.decisions, reference.decisions - LIMIT);
    assert_eq!(resumed.trace, reference.trace[LIMIT as usize..]);
    assert_eq!(resumed.scores, reference.scores);
    assert_eq!(score_sum(&resumed), 0);

    // A zero limit stops before the first choice without touching the game.
    let mut zero_game = Game::new(808);
    let mut zero_runner = Runner::first_legal().with_decision_limit(0);
    match zero_runner.run_game(&mut zero_game).unwrap_err() {
        RunnerError::DecisionLimit { limit } => assert_eq!(limit, 0),
        other => panic!("expected DecisionLimit, got {other:?}"),
    }
    assert!(zero_game.result().is_none());

    zero_runner.set_decision_limit(DEFAULT_DECISION_LIMIT);
    let full = zero_runner.run_game(&mut zero_game).unwrap();
    assert_eq!(full.trace, reference.trace);
    assert_eq!(full.scores, reference.scores);
}

struct AlwaysPass;

impl Policy for AlwaysPass {
    fn choose(&mut self, _decision: &Decision<'_>) -> Action {
        Action::Pass
    }
}

fn always_pass() -> Box<dyn Policy> {
    Box::new(AlwaysPass)
}

#[test]
fn illegal_policy_actions_are_reported() {
    let mut runner = Runner::new([always_pass(), always_pass(), always_pass(), always_pass()]);
    let error = runner.run_new(3).unwrap_err();

    match error {
        RunnerError::IllegalAction {
            seat,
            action,
            source,
        } => {
            // The first decision is the dealer's exchange; Pass is illegal.
            assert_eq!(seat, Seat::DEALER);
            assert_eq!(action, Action::Pass);
            assert!(!source.to_string().is_empty());
        }
        other => panic!("expected an illegal-action error, got {other:?}"),
    }
}

#[test]
fn zero_game_batches_are_rejected() {
    let mut runner = Runner::first_legal();
    match runner.run_batch(0, 0) {
        Err(RunnerError::EmptyBatch) => {}
        other => panic!("expected EmptyBatch, got {other:?}"),
    }
}

#[test]
fn batch_scores_are_zero_sum_and_reproducible() {
    let mut first_runner = Runner::first_legal();
    let first = first_runner.run_batch(17, 4).unwrap();

    assert_eq!(first.games, 4);
    assert!(first.decisions >= first.games);
    assert_eq!(first.score_sum(), 0);
    assert_eq!(first.score_totals.iter().sum::<i64>(), 0);
    assert_eq!(first.three_winners + first.wall_exhausted, first.games);

    let mut second_runner = Runner::first_legal();
    let second = second_runner.run_batch(17, 4).unwrap();

    assert_eq!(first.decisions, second.decisions);
    assert_eq!(first.score_totals, second.score_totals);
    assert_eq!(first.winners, second.winners);
    assert_eq!(first.three_winners, second.three_winners);
    assert_eq!(first.wall_exhausted, second.wall_exhausted);
}

const CLI: &str = env!("CARGO_BIN_EXE_sichuan-mahjong-sim");

fn run_cli(arguments: &[&str]) -> Output {
    Command::new(CLI)
        .args(arguments)
        .output()
        .expect("failed to execute the simulator binary")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn cli_rejects_invalid_arguments_and_prints_help() {
    let zero = run_cli(&["--games", "0"]);
    assert!(!zero.status.success());
    assert!(stderr(&zero).contains("--games"));

    let bad_policy = run_cli(&["--policy", "search"]);
    assert!(!bad_policy.status.success());
    assert!(stderr(&bad_policy).contains("--policy"));

    let missing_seed = run_cli(&["--seed"]);
    assert!(!missing_seed.status.success());

    let help = run_cli(&["--help"]);
    assert!(help.status.success());
    assert!(stdout(&help).contains("Usage:"));

    // `--help` short-circuits: invalid arguments after it are ignored.
    let help_first = run_cli(&["--help", "--games", "0"]);
    assert!(help_first.status.success());
    assert!(stdout(&help_first).contains("Usage:"));

    let short_help_first = run_cli(&["-h", "--seed", "abc", "extra"]);
    assert!(short_help_first.status.success());
    assert!(stdout(&short_help_first).contains("Usage:"));
}

fn stable_output(text: &str) -> String {
    text.lines()
        .filter(|line| !line.starts_with("time:") && !line.starts_with("throughput:"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn cli_self_play_is_reproducible_and_reports_throughput() {
    let first = run_cli(&["--games", "2", "--seed", "11", "--policy", "random"]);
    let second = run_cli(&["--games", "2", "--seed", "11", "--policy", "random"]);

    assert!(first.status.success(), "CLI failed: {}", stderr(&first));
    assert!(second.status.success(), "CLI failed: {}", stderr(&second));

    let output = stdout(&first);
    assert!(output.contains("games/sec"));
    assert!(output.contains("decisions/sec"));
    assert!(output.contains("time: total="));
    assert!(output.contains("terminal score totals:"));

    assert_eq!(stable_output(&output), stable_output(&stdout(&second)));
}
