//! End-to-end environment trace fixture for the Python binding.
//!
//! The fixture is produced from the public core and neural encoding APIs plus a
//! deterministic claim/kong script. It covers every decision stage, including
//! added-kong responses, and continues through wins until settlement. The
//! Python environment test replays the recorded canonical indices and compares
//! every state, candidate row, candidate order, identity, and terminal fact.
//!
//! Regenerate deliberately with:
//! `UPDATE_ENVIRONMENT_TRACE_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test environment_trace`

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};
use sichuan_mahjong_neural::encoding::{
    canonicalize, encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH,
};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::{Action, Decision, DecisionStage, EndReason, Game, LegalActions, Suit};

/// Aggressive claim/kong script whose fixed trajectory reaches an
/// `AddedKongResponse` at decision 45. The seed is the long-term regression
/// input, not a search result to refresh.
const SEED: u64 = 61;
const FORMAT: &str = "mahjong-eval-environment-trace-v1";
const FIXTURE_NAME: &str = "environment_trace_seed_61.json";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(FIXTURE_NAME)
}

fn hex_f32(values: &[f32]) -> String {
    let mut out = String::with_capacity(values.len() * 8);
    for value in values {
        for byte in value.to_le_bytes() {
            write!(&mut out, "{byte:02x}").expect("writing to a string");
        }
    }
    out
}

fn stage_code(stage: DecisionStage) -> u8 {
    match stage {
        DecisionStage::Exchange => 0,
        DecisionStage::MissingSuit => 1,
        DecisionStage::Turn => 2,
        DecisionStage::DiscardResponse => 3,
        DecisionStage::AddedKongResponse => 4,
    }
}

fn end_reason_name(reason: EndReason) -> &'static str {
    match reason {
        EndReason::ThreeWinners => "three_winners",
        EndReason::WallExhausted => "wall_exhausted",
    }
}

/// Deterministic scripted choice. Hu wins immediately, then kongs and claims
/// are preferred so the trajectory reaches response and added-kong stages.
fn script_choice(decision: &Decision<'_>) -> Action {
    match decision.legal_actions {
        LegalActions::Exchange { eligible_suits } => {
            let suit = Suit::ALL
                .into_iter()
                .find(|suit| {
                    eligible_suits[suit.index()]
                        && decision.observation.own_hand.suit_count(*suit) >= 3
                })
                .expect("an exchange triple exists");
            let mut tiles = decision
                .observation
                .own_hand
                .tiles()
                .filter(|tile| tile.suit() == suit);
            Action::Exchange([
                tiles.next().expect("three tiles"),
                tiles.next().expect("three tiles"),
                tiles.next().expect("three tiles"),
            ])
        }
        LegalActions::Actions(actions) => {
            if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
                return hu;
            }
            if decision.observation.stage == DecisionStage::Turn {
                if let Some(&added) = actions
                    .iter()
                    .find(|action| matches!(action, Action::AddedKong(_)))
                {
                    return added;
                }
                if let Some(&concealed) = actions
                    .iter()
                    .find(|action| matches!(action, Action::ConcealedKong(_)))
                {
                    return concealed;
                }
            } else if decision.observation.stage == DecisionStage::DiscardResponse {
                if let Some(&peng) = actions.iter().find(|action| matches!(action, Action::Peng)) {
                    return peng;
                }
                if let Some(&kong) = actions
                    .iter()
                    .find(|action| matches!(action, Action::OpenKong))
                {
                    return kong;
                }
            }
            actions.first().copied().unwrap_or(Action::Pass)
        }
    }
}

struct Trace {
    stages: [bool; 5],
    winner_seen_before_end: bool,
    value: Value,
}

fn build_trace() -> Trace {
    let mut game = Game::new(SEED);
    let mut stages = [false; 5];
    let mut decisions = Vec::new();
    let mut winner_seen_before_end = false;
    let _ = game.next_decision();
    while let Some(decision) = game.next_decision() {
        let stage = decision.observation.stage;
        stages[usize::from(stage_code(stage))] = true;
        let encoded = encode(&decision);
        let chosen = canonicalize(script_choice(&decision));
        let chosen_index = encoded
            .target_index(&chosen)
            .unwrap_or_else(|error| panic!("scripted choice is not a candidate: {error}"));
        let action = encoded.candidates[chosen_index].action;

        let mut candidate_features = Vec::with_capacity(encoded.candidates.len() * ACTION_WIDTH);
        for candidate in &encoded.candidates {
            candidate_features.extend_from_slice(&candidate.features.flatten());
        }
        decisions.push(json!({
            "decision_id": (1u64 << 32) | (decisions.len() as u64 + 1),
            "actor_seat": decision.seat.index(),
            "stage": stage_code(stage),
            "candidate_count": encoded.candidates.len(),
            "chosen_index": chosen_index,
            "state_le_f32_hex": hex_f32(&encoded.state.flatten()),
            "candidate_features_le_f32_hex": hex_f32(&candidate_features),
        }));

        let seat = decision.seat;
        let saw_winner = decision
            .observation
            .players
            .iter()
            .any(|player| player.has_won);
        let continues = game.result().is_none();
        game.submit(seat, action)
            .unwrap_or_else(|error| panic!("scripted action rejected: {error}"));
        winner_seen_before_end |= saw_winner && continues;
        assert!(decisions.len() < 2_000, "scripted game did not terminate");
    }

    let result = game.result().expect("scripted game reached settlement");
    let decision_count = decisions.len();
    Trace {
        stages,
        winner_seen_before_end,
        value: json!({
            "format": FORMAT,
            "seed": SEED,
            "decision_limit": DEFAULT_DECISION_LIMIT,
            "encoding_version": ENCODING_VERSION,
            "state_width": STATE_WIDTH,
            "candidate_width": ACTION_WIDTH,
            "decisions": decisions,
            "terminal": {
                "decisions": decision_count,
                "scores": result.scores,
                "winners": result.winners,
                "end_reason": end_reason_name(result.end_reason),
            },
        }),
    }
}

#[test]
fn environment_trace_covers_every_stage_and_plays_past_a_winner() {
    let trace = build_trace();
    assert_eq!(trace.stages, [true; 5], "trace must cover every stage");
    assert!(
        trace.winner_seen_before_end,
        "trace must continue after at least one seat has won"
    );
}

#[test]
fn environment_trace_matches_checked_in_fixture() {
    let expected = build_trace().value;
    let path = fixture_path();

    if std::env::var_os("UPDATE_ENVIRONMENT_TRACE_FIXTURE").is_some() {
        let text = format!(
            "{}\n",
            serde_json::to_string_pretty(&expected).expect("serialize fixture")
        );
        fs::create_dir_all(path.parent().expect("fixture directory"))
            .expect("create fixture directory");
        fs::write(&path, text).expect("write fixture");
    }

    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "read {}: {error}; regenerate with UPDATE_ENVIRONMENT_TRACE_FIXTURE=1",
            path.display()
        )
    });
    let actual: Value = serde_json::from_str(&text).expect("parse fixture");
    assert_eq!(
        actual, expected,
        "environment trace fixture is stale; regenerate deliberately with \
         UPDATE_ENVIRONMENT_TRACE_FIXTURE=1"
    );
}
