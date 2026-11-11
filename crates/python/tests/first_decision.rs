//! Exact first-decision fixture shared by the Rust encoder and Python binding.
//!
//! The test recomputes the opening decision from the public core and neural
//! APIs, then compares every metadata field and raw little-endian float32 byte
//! to the checked-in fixture. The Python package test reads the same file, so
//! the binding cannot drift from the Rust encoder unnoticed.
//!
//! Regenerate the fixture deliberately with:
//! `UPDATE_FIRST_DECISION_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test first_decision`

mod support;

use serde_json::{json, Value};
use sichuan_mahjong_neural::encoding::{encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};
use sichuan_mahjong_sim::Game;

const SEED: u64 = 0;
const FORMAT: &str = "mahjong-eval-first-decision-v1";
const FIXTURE_NAME: &str = "first_decision_seed_0.json";

fn expected_fixture() -> Value {
    let mut game = Game::new(SEED);
    let decision = game.next_decision().expect("opening decision");
    let encoded = encode(&decision);

    let state = encoded.state.flatten();
    let mut candidate_features = Vec::with_capacity(encoded.candidates.len() * ACTION_WIDTH);
    for candidate in &encoded.candidates {
        candidate_features.extend_from_slice(&candidate.features.flatten());
    }

    json!({
        "format": FORMAT,
        "seed": SEED,
        "decision_id": (1u64 << 32) | 1,
        "actor_seat": decision.seat.index(),
        "stage": support::stage_code(decision.observation.stage),
        "encoding_version": ENCODING_VERSION,
        "state_width": STATE_WIDTH,
        "candidate_width": ACTION_WIDTH,
        "candidate_count": encoded.candidates.len(),
        "state_le_f32_hex": support::hex_f32(&state),
        "candidate_features_le_f32_hex": support::hex_f32(&candidate_features),
    })
}

#[test]
fn first_decision_matches_checked_in_fixture() {
    let expected = expected_fixture();
    support::check_fixture("UPDATE_FIRST_DECISION_FIXTURE", FIXTURE_NAME, &expected);
}
