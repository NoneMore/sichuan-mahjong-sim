//! Shared read-only fixture helpers for the Rust-side cross-language tests.
//!
//! Each integration test keeps its own domain-specific fixture builder. This
//! module only owns path resolution, deliberate regeneration, and wire-value
//! formatting so the checked-in golden files cannot drift between copies.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use sichuan_mahjong_sim::{DecisionStage, EndReason};

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Compares `expected` with a checked-in fixture.
///
/// Setting `update_var` deliberately rewrites the fixture first; the read and
/// comparison still run so a mistaken regeneration is immediately visible.
pub fn check_fixture(update_var: &str, name: &str, expected: &Value) {
    let path = fixture_path(name);
    if std::env::var_os(update_var).is_some() {
        let text = format!(
            "{}\n",
            serde_json::to_string_pretty(expected).expect("serialize fixture")
        );
        fs::create_dir_all(path.parent().expect("fixture directory"))
            .expect("create fixture directory");
        fs::write(&path, text).expect("write fixture");
    }

    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "read {}: {error}; regenerate with {update_var}=1",
            path.display()
        )
    });
    let actual: Value = serde_json::from_str(&text).expect("parse fixture");
    assert_eq!(
        &actual,
        expected,
        "fixture {} is stale; regenerate deliberately with {update_var}=1",
        path.display()
    );
}

pub fn stage_code(stage: DecisionStage) -> u8 {
    match stage {
        DecisionStage::Exchange => 0,
        DecisionStage::MissingSuit => 1,
        DecisionStage::Turn => 2,
        DecisionStage::DiscardResponse => 3,
        DecisionStage::AddedKongResponse => 4,
    }
}

pub fn end_reason_name(reason: EndReason) -> &'static str {
    match reason {
        EndReason::ThreeWinners => "three_winners",
        EndReason::WallExhausted => "wall_exhausted",
    }
}

pub fn hex_f32(values: &[f32]) -> String {
    let mut out = String::with_capacity(values.len() * 8);
    for value in values {
        for byte in value.to_le_bytes() {
            write!(&mut out, "{byte:02x}").expect("writing to a string");
        }
    }
    out
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut out, "{byte:02x}").expect("writing to a string");
    }
    out
}
