//! Public-surface checks for [`MissingOnlyPolicy`].
//!
//! The policy is defined against two references: the deterministic baseline for
//! every decision and the heuristic rule for the missing-suit declaration. These
//! tests drive real games through the public `Game`/`Runner` surface and compare
//! every decision against the reference policy the contract names, so the
//! ablation cannot drift from either side.

use sichuan_mahjong_sim::policy::{FirstLegalPolicy, HeuristicPolicy, MissingOnlyPolicy, Policy};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::{Action, Decision, DecisionStage, Game, LegalActions};

/// True when `action` appears in the decision's legal list.
fn is_legal(decision: &Decision<'_>, action: Action) -> bool {
    match &decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => match action {
            Action::Exchange(tiles) => {
                let suit = tiles[0].suit();
                eligible_suits[suit.index()]
                    && tiles.iter().all(|tile| tile.suit() == suit)
                    && decision.observation.own_hand.contains_tiles(&tiles)
            }
            _ => false,
        },
        LegalActions::Actions(actions) => actions.contains(&action),
    }
}

fn is_claim_opportunity(decision: &Decision<'_>) -> bool {
    match &decision.legal_actions {
        LegalActions::Actions(actions) => actions.iter().any(|action| {
            matches!(
                action,
                Action::Peng | Action::OpenKong | Action::ConcealedKong(_) | Action::AddedKong(_)
            )
        }),
        LegalActions::Exchange { .. } => false,
    }
}

#[test]
fn every_decision_matches_its_reference_policy_and_stays_legal() {
    let mut missing_suit_decisions = 0usize;
    let mut exchange_decisions = 0usize;
    let mut response_decisions = 0usize;
    let mut claim_opportunities = 0usize;
    let mut baseline_divergences = 0usize;

    // 32 seeds exercise exchange, declarations, turns, and claim responses
    // without turning this focused test into an experiment run. The combined
    // response counter does not guarantee coverage of each response subtype.
    for seed in 0..32u64 {
        let mut game = Game::new(seed);
        let mut missing_only = MissingOnlyPolicy::new();
        let mut first = FirstLegalPolicy::new();
        let mut heuristic = HeuristicPolicy::new();

        while let Some(step) = game.next_decision().map(|decision| {
            let choice = missing_only.choose(&decision);
            let baseline = first.choose(&decision);
            // The heuristic reference is evaluated at every decision so the
            // divergence counter below can prove the two references really
            // disagree somewhere.
            let heuristic_choice = heuristic.choose(&decision);
            let expected = if decision.observation.stage == DecisionStage::MissingSuit {
                heuristic_choice
            } else {
                baseline
            };
            (
                decision.seat,
                decision.observation.stage,
                choice,
                expected,
                baseline,
                heuristic_choice,
                is_claim_opportunity(&decision),
                is_legal(&decision, choice),
            )
        }) {
            let (
                seat,
                stage,
                choice,
                expected,
                baseline,
                heuristic_choice,
                claim_opportunity,
                legal,
            ) = step;

            assert!(legal, "seed {seed}: {choice:?} is not legal at {stage:?}");
            assert_eq!(
                choice,
                expected,
                "seed {seed} seat {} stage {stage:?}: ablation drifted from its \
                 reference policy",
                seat.index()
            );

            match stage {
                DecisionStage::MissingSuit => missing_suit_decisions += 1,
                DecisionStage::Exchange => exchange_decisions += 1,
                DecisionStage::DiscardResponse | DecisionStage::AddedKongResponse => {
                    response_decisions += 1;
                }
                _ => {
                    if baseline != heuristic_choice {
                        // The baseline and the heuristic genuinely disagree here,
                        // so the equality above proves the ablation followed the
                        // baseline instead of the heuristic.
                        baseline_divergences += 1;
                    }
                }
            }
            if claim_opportunity {
                claim_opportunities += 1;
            }

            game.submit(seat, choice)
                .unwrap_or_else(|error| panic!("seed {seed}: rejected {choice:?}: {error}"));
        }

        let result = game
            .result()
            .unwrap_or_else(|| panic!("seed {seed} ended without a result"));
        assert!(
            result
                .scores
                .iter()
                .map(|score| i64::from(*score))
                .sum::<i64>()
                == 0,
            "seed {seed}: scores are not zero-sum: {:?}",
            result.scores
        );
    }

    assert!(
        missing_suit_decisions > 0,
        "missing-suit branch was not exercised"
    );
    assert!(exchange_decisions > 0, "exchange branch was not exercised");
    assert!(
        response_decisions > 0,
        "discard/added-kong responses were not exercised"
    );
    assert!(
        claim_opportunities > 0,
        "no peng or kong was ever legal, so claim delegation was not exercised"
    );
    assert!(
        baseline_divergences > 0,
        "the baseline and heuristic never disagreed (missing={missing_suit_decisions}, exchange={exchange_decisions}, response={response_decisions}, claims={claim_opportunities}), so the comparison is vacuous"
    );
}

#[test]
fn missing_only_games_are_deterministic_and_checkpoint_safe() {
    let runner = || {
        Runner::new(std::array::from_fn(|_| {
            Box::new(MissingOnlyPolicy::new()) as Box<dyn Policy>
        }))
    };

    for seed in [1, 7] {
        let first = runner().run_new(seed).unwrap();
        let second = runner().run_new(seed).unwrap();
        assert!(first.decisions > 0, "seed {seed} made no decisions");
        assert_eq!(first.trace, second.trace, "seed {seed} did not replay");
        assert_eq!(first.scores, second.scores);
        assert_eq!(first.end_reason, second.end_reason);
        assert_eq!(
            first
                .scores
                .iter()
                .map(|score| i64::from(*score))
                .sum::<i64>(),
            0
        );
    }

    // The policy is stateless, so cold, fresh instances reproduce the
    // continuation from a cloned game, exactly like the other deterministic
    // policies.
    let mut game = Game::new(31);
    let mut limited = runner().with_decision_limit(40);
    match limited.run_game(&mut game) {
        Err(sichuan_mahjong_sim::runner::RunnerError::DecisionLimit { limit }) => {
            assert_eq!(limit, 40);
        }
        other => panic!("expected a mid-game checkpoint, got {other:?}"),
    }
    assert!(game.result().is_none(), "checkpoint should be mid-game");

    let mut fork = game.clone();
    let original = runner().run_game(&mut game).unwrap();
    let resumed = runner().run_game(&mut fork).unwrap();
    assert!(original.decisions > 0);
    assert_eq!(original.trace, resumed.trace);
    assert_eq!(original.scores, resumed.scores);
    assert_eq!(original.end_reason, resumed.end_reason);
}
