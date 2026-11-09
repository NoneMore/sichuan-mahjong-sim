#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use sichuan_mahjong_neural::collection::{
    manifest_path, read_manifest, seed_split, shard_path, CollectionConfig, CollectionOutcome,
    Corpus, DecisionSample, Episode, EpisodeId, Manifest, Seed, SourceRevision,
};
use sichuan_mahjong_neural::encoding::{encode, ENCODING_VERSION};
use sichuan_mahjong_sim::{
    Game, GameResult, HeuristicCache, HeuristicPolicy, Policy, PLAYER_COUNT,
};

pub const REV_A: &str = "0123456789abcdef0123456789abcdef01234567";
pub const REV_B: &str = "fedcba9876543210fedcba9876543210fedcba98";

pub fn revision(value: &str) -> SourceRevision {
    SourceRevision::parse(value).unwrap()
}

pub fn root() -> TempDir {
    tempfile::tempdir().unwrap()
}

pub fn collect(
    root: &Path,
    source: &SourceRevision,
    seed_start: u64,
    games: u64,
    shard_games: u64,
    decision_limit: u64,
) -> CollectionOutcome {
    let config = CollectionConfig::new(source.clone())
        .with_seed_start(Seed::from_u64(seed_start))
        .with_shard_games(shard_games)
        .with_decision_limit(decision_limit);
    let mut corpus = Corpus::create(root, &config).unwrap();
    corpus.append_with_cache(games, 0).unwrap()
}

pub fn append(root: &Path, source: &SourceRevision, games: u64) -> CollectionOutcome {
    let mut corpus = Corpus::open_append(root, source).unwrap();
    corpus.append_with_cache(games, 0).unwrap()
}

pub fn replay(seed: u64, source: &SourceRevision) -> Episode {
    let cache = HeuristicCache::new(0).unwrap();
    let mut policies: [HeuristicPolicy; PLAYER_COUNT] =
        std::array::from_fn(|_| HeuristicPolicy::with_cache(cache.clone()));
    let mut game = Game::new(seed);
    let mut decisions = Vec::new();
    let result: GameResult = loop {
        if let Some(result) = game.result() {
            break *result;
        }
        let Some(decision) = game.next_decision() else {
            panic!("game ended without a result");
        };
        let encoded = encode(&decision);
        let actor = decision.seat;
        let stage = decision.observation.stage;
        let teacher = policies[actor.index()].choose(&decision);
        let target_index = encoded.target_index(&teacher).unwrap();
        let sample = DecisionSample {
            decision_index: decisions.len() as u64,
            actor_seat: actor,
            stage,
            state: encoded.state.flatten().to_vec(),
            candidate_features: encoded
                .candidates
                .iter()
                .map(|candidate| candidate.features.flatten().to_vec())
                .collect(),
            candidates: encoded
                .candidates
                .iter()
                .map(|candidate| candidate.action)
                .collect(),
            target_index,
        };
        let action = encoded.candidates[target_index].action;
        game.submit(actor, action).unwrap();
        decisions.push(sample);
        assert!(decisions.len() < 10_000);
    };
    Episode {
        id: EpisodeId {
            source_revision: source.clone(),
            encoding_version: ENCODING_VERSION,
            seed: Seed::from_u64(seed),
        },
        split: seed_split(Seed::from_u64(seed)),
        decisions,
        result,
    }
}

pub fn manifest(root: &Path) -> Manifest {
    read_manifest(root).unwrap()
}

pub fn write_manifest(root: &Path, manifest: &Manifest) {
    let mut bytes = serde_json::to_vec_pretty(manifest).unwrap();
    bytes.push(b'\n');
    fs::write(manifest_path(root), bytes).unwrap();
}

pub fn shard_files(root: &Path, manifest: &Manifest) -> Vec<PathBuf> {
    manifest
        .shards
        .iter()
        .map(|shard| shard_path(root, shard.first_seed))
        .collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Rewrites every listed shard checksum in the manifest.
pub fn refresh_manifest_checksums(root: &Path) -> Manifest {
    let mut manifest = read_manifest(root).unwrap();
    for shard in &mut manifest.shards {
        let path = shard_path(root, shard.first_seed);
        let bytes = fs::read(path).unwrap();
        shard.sha256 = sha256_hex(&bytes);
    }
    write_manifest(root, &manifest);
    manifest
}

/// Mutates the first `decision` object of the first listed shard and updates
/// the manifest checksum.
pub fn mutate_first_decision<F>(root: &Path, mutate: F)
where
    F: FnOnce(&mut Value),
{
    let manifest = read_manifest(root).unwrap();
    let path = shard_path(root, manifest.shards[0].first_seed);
    let text = fs::read_to_string(&path).unwrap();
    let mut lines: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let decision = lines
        .iter_mut()
        .find(|line| line["type"] == Value::String("decision".to_owned()))
        .expect("first shard has a decision");
    mutate(decision);
    write_json_lines(&path, &lines);
    refresh_manifest_checksums(root);
}

/// Mutates the manifest JSON object.
pub fn mutate_manifest<F>(root: &Path, mutate: F)
where
    F: FnOnce(&mut Value),
{
    let path = manifest_path(root);
    let text = fs::read_to_string(&path).unwrap();
    let mut value: Value = serde_json::from_str(&text).unwrap();
    mutate(&mut value);
    let mut bytes = serde_json::to_vec_pretty(&value).unwrap();
    bytes.push(b'\n');
    fs::write(path, bytes).unwrap();
}

pub fn write_json_lines(path: &Path, lines: &[Value]) {
    let mut text = String::new();
    for line in lines {
        text.push_str(&serde_json::to_string(line).unwrap());
        text.push('\n');
    }
    fs::write(path, text).unwrap();
}

pub fn read_json_lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

pub fn seed_hex(value: u64) -> String {
    format!("{value:016x}")
}
