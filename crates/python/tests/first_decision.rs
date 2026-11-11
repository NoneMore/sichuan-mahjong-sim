//! Exact first-decision fixture shared by the Rust encoder and Python binding.
//!
//! The test recomputes the opening decision from the public core and neural
//! APIs, then compares every metadata field and raw little-endian float32 byte
//! to the checked-in fixture. The Python package test reads the same file, so
//! the binding cannot drift from the Rust encoder unnoticed.
//!
//! Regenerate the fixture deliberately with:
//! `UPDATE_FIRST_DECISION_FIXTURE=1 cargo test --manifest-path crates/python/Cargo.toml --test first_decision`

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};
use sichuan_mahjong_neural::encoding::{encode, ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};
use sichuan_mahjong_sim::{DecisionStage, Game};

const SEED: u64 = 0;
const FORMAT: &str = "mahjong-eval-first-decision-v1";
const FIXTURE_NAME: &str = "first_decision_seed_0.json";

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
        "stage": stage_code(decision.observation.stage),
        "encoding_version": ENCODING_VERSION,
        "state_width": STATE_WIDTH,
        "candidate_width": ACTION_WIDTH,
        "candidate_count": encoded.candidates.len(),
        "state_le_f32_hex": hex_f32(&state),
        "candidate_features_le_f32_hex": hex_f32(&candidate_features),
    })
}

#[test]
fn first_decision_matches_checked_in_fixture() {
    let expected = expected_fixture();
    let path = fixture_path();

    if std::env::var_os("UPDATE_FIRST_DECISION_FIXTURE").is_some() {
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
            "read {}: {error}; regenerate with UPDATE_FIRST_DECISION_FIXTURE=1",
            path.display()
        )
    });
    let actual: Value = serde_json::from_str(&text).expect("parse fixture");
    assert_eq!(
        actual, expected,
        "first-decision fixture is stale; regenerate deliberately with \
         UPDATE_FIRST_DECISION_FIXTURE=1"
    );
}
