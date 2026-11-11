//! Teacher replay fixture shared by the Rust core/teacher and the Python binding.
//!
//! The fixture records one complete heuristic-teacher game from the public core
//! APIs: every pending decision, the teacher's canonical index and semantic
//! action, exact state/candidate content hashes, and the terminal settlement.
//! The Python test independently reproduces the trajectory through `Env`, checks
//! `teacher_index` without advancing, and compares `native_teacher_replay`.
//!
//! Regenerate deliberately with:
//! `UPDATE_TEACHER_REPLAY_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test teacher_replay`

mod support;

use serde_json::{json, Value};
use sichuan_mahjong_neural::collection::{action_kind_id, encode_action, ACTION_KIND_NAMES};
use sichuan_mahjong_neural::encoding::{
    canonicalize, encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH,
};
use sichuan_mahjong_sim::runner::DEFAULT_DECISION_LIMIT;
use sichuan_mahjong_sim::{Action, Game, HeuristicPolicy, Policy};

const SEED: u64 = 0;
const FORMAT: &str = "mahjong-eval-teacher-replay-v1";
const FIXTURE_NAME: &str = "teacher_replay_seed_0.json";

fn action_json(action: Action) -> Value {
    let kind_id = action_kind_id(&action);
    let wire: Vec<u8> = encode_action(action);
    json!({
        "kind": ACTION_KIND_NAMES[usize::from(kind_id)],
        "kind_id": kind_id,
        "wire": wire,
    })
}

fn expected_fixture() -> Value {
    let mut game = Game::new(SEED);
    let mut teacher = HeuristicPolicy::new();
    let mut decisions = Vec::new();
    let mut accepted = 0u64;

    while let Some(decision) = game.next_decision() {
        let chosen = canonicalize(teacher.choose(&decision));
        let encoded = encode(&decision);
        let teacher_index = encoded
            .target_index(&chosen)
            .unwrap_or_else(|error| panic!("teacher choice is not a candidate: {error}"));

        let mut candidate_features = Vec::with_capacity(encoded.candidates.len() * ACTION_WIDTH);
        for candidate in &encoded.candidates {
            candidate_features.extend_from_slice(&candidate.features.flatten());
        }
        let state = encoded.state.flatten();
        decisions.push(json!({
            "decision_id": (1u64 << 32) | (accepted + 1),
            "actor_seat": decision.seat.index(),
            "stage": support::stage_code(decision.observation.stage),
            "candidate_count": encoded.candidates.len(),
            "teacher_index": teacher_index,
            "action": action_json(encoded.candidates[teacher_index].action),
            "state_sha256": support::hex_sha256(&support::f32_bytes(&state)),
            "candidate_features_sha256": support::hex_sha256(&support::f32_bytes(&candidate_features)),
        }));

        let seat = decision.seat;
        game.submit(seat, encoded.candidates[teacher_index].action)
            .unwrap_or_else(|error| panic!("teacher action rejected: {error}"));
        accepted += 1;
        assert!(accepted < 2_000, "teacher game did not terminate");
    }

    let result = game.result().expect("teacher game reached settlement");
    json!({
        "format": FORMAT,
        "seed": SEED,
        "decision_limit": DEFAULT_DECISION_LIMIT,
        "encoding_version": ENCODING_VERSION,
        "state_width": STATE_WIDTH,
        "candidate_width": ACTION_WIDTH,
        "decisions": decisions,
        "terminal": {
            "decisions": accepted,
            "scores": result.scores,
            "winners": result.winners,
            "end_reason": support::end_reason_name(result.end_reason),
        },
    })
}

#[test]
fn teacher_fixture_is_a_complete_teacher_game() {
    let fixture = expected_fixture();
    let decisions = fixture["decisions"]
        .as_array()
        .expect("fixture decisions array");
    assert_eq!(decisions.len() as u64, fixture["terminal"]["decisions"]);
    assert!(
        fixture["terminal"]["scores"]
            .as_array()
            .expect("scores")
            .iter()
            .map(|value| value.as_i64().expect("score"))
            .sum::<i64>()
            == 0
    );
    assert!(
        decisions
            .iter()
            .any(|entry| entry["teacher_index"].as_u64().unwrap_or(0) > 0),
        "teacher fixture must exercise non-zero canonical indices"
    );
}

#[test]
fn teacher_replay_matches_checked_in_fixture() {
    let expected = expected_fixture();
    support::check_fixture("UPDATE_TEACHER_REPLAY_FIXTURE", FIXTURE_NAME, &expected);
}
