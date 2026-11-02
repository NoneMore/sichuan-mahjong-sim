//! Black-box checks of the public decision contract, independent of Game internals.
use sichuan_mahjong_sim::{Action, Decision, Game, LegalActions, Seat, Suit};

fn legal_choice(decision: &Decision<'_>) -> Action {
    match &decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            let suit = Suit::ALL
                .into_iter()
                .find(|suit| eligible_suits[suit.index()])
                .expect("every initial hand has an exchange triple");
            let tiles: Vec<_> = decision
                .observation
                .own_hand
                .tiles()
                .filter(|tile| tile.suit() == suit)
                .take(3)
                .collect();
            Action::Exchange([tiles[0], tiles[1], tiles[2]])
        }
        LegalActions::Actions(actions) => actions
            .iter()
            .find(|action| matches!(action, Action::Hu))
            .or_else(|| actions.first())
            .cloned()
            .expect("a pending decision has at least one legal action"),
    }
}

fn decision_signature(game: &mut Game) -> Option<(Seat, [u8; 27], String)> {
    game.next_decision().map(|decision| {
        (
            decision.seat,
            *decision.observation.own_hand.counts(),
            match decision.legal_actions {
                LegalActions::Exchange { eligible_suits } => format!("{eligible_suits:?}"),
                LegalActions::Actions(actions) => format!("{actions:?}"),
            },
        )
    })
}

#[test]
fn decision_reads_are_idempotent_and_rejected_actions_are_transactional() {
    for seed in 0..12 {
        let mut game = Game::new(seed);
        let mut decisions = 0;
        while let Some((seat, action, wrong_action)) = game.next_decision().map(|decision| {
            let wrong_action = match decision.legal_actions {
                LegalActions::Exchange { .. } => Action::Pass,
                LegalActions::Actions(_) => {
                    // Opening-only exchange cannot be submitted in any later stage.
                    let tile = sichuan_mahjong_sim::Tile::ALL[0];
                    Action::Exchange([tile; 3])
                }
            };
            (decision.seat, legal_choice(&decision), wrong_action)
        }) {
            let before = format!("{game:?}");
            let first = decision_signature(&mut game);
            assert_eq!(first, decision_signature(&mut game));
            assert_eq!(before, format!("{game:?}"), "pending reads mutated Game");

            assert!(game.submit(seat.next_clockwise(), action).is_err());
            assert_eq!(
                before,
                format!("{game:?}"),
                "wrong-seat submission mutated Game"
            );
            assert!(game.submit(seat, wrong_action).is_err());
            assert_eq!(before, format!("{game:?}"), "illegal action mutated Game");

            game.submit(seat, action)
                .expect("listed action must be accepted");
            decisions += 1;
            assert!(
                decisions < 2_000,
                "game failed to terminate for seed {seed}"
            );
        }
        let result = game.result().expect("no decisions implies terminal result");
        assert_eq!(result.scores.iter().sum::<i32>(), 0);
        let terminal = format!("{game:?}");
        assert!(game.next_decision().is_none());
        assert!(game.next_decision().is_none());
        assert_eq!(
            terminal,
            format!("{game:?}"),
            "terminal read repeated settlement"
        );
    }
}

#[test]
fn clone_during_partial_exchange_replays_the_same_future() {
    let mut original = Game::new(907);
    for _ in 0..2 {
        let decision = original.next_decision().unwrap();
        let seat = decision.seat;
        let action = legal_choice(&decision);
        original.submit(seat, action).unwrap();
    }
    let mut fork = original.clone();
    for _ in 0..2_000 {
        assert_eq!(
            decision_signature(&mut original),
            decision_signature(&mut fork)
        );
        let next = original
            .next_decision()
            .map(|decision| (decision.seat, legal_choice(&decision)));
        let Some((seat, action)) = next else {
            assert!(fork.next_decision().is_none());
            assert_eq!(format!("{original:?}"), format!("{fork:?}"));
            return;
        };
        original.submit(seat, action).unwrap();
        fork.submit(seat, action).unwrap();
    }
    panic!("cloned games failed to terminate");
}

#[test]
fn seeded_initial_state_is_reproducible() {
    for seed in [0, 1, 2, u64::MAX] {
        let a = Game::new(seed);
        let b = Game::new(seed);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}
