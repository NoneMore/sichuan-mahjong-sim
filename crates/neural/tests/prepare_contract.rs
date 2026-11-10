#[path = "collection_support/mod.rs"]
mod support;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use support::{append, collect, manifest, revision, root, sha256_hex, REV_A};

use sichuan_mahjong_neural::collection::{
    action_kind_id, manifest_path, read_corpus, shard_path, CorpusSnapshot, Episode, Split,
};
use sichuan_mahjong_neural::encoding::{ACTION_WIDTH, ENCODING_VERSION, STATE_WIDTH};
use sichuan_mahjong_neural::prepare::{
    compute_cache_id, prepare, read_dataset_manifest, verify_dataset, Counts, DatasetManifest,
    PrepareOptions, SplitManifest, DATASET_FILE, DATASET_FORMAT, PREPARATION_FORMAT_VERSION,
    PREPARATION_VERSION,
};

const EXE: &str = env!("CARGO_BIN_EXE_bc-prepare");

fn options() -> PrepareOptions {
    PrepareOptions {
        max_decisions_per_shard: 40,
        max_candidates_per_shard: 600,
    }
}

fn cache_root(temp: &Path, name: &str) -> PathBuf {
    temp.join(name)
}

fn read_npy(path: &Path) -> (String, Vec<u64>, Vec<u8>) {
    let bytes = fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{path:?}");
    assert_eq!((bytes[6], bytes[7]), (1, 0), "{path:?}");
    let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let dtype_start = header.find("'descr': '").unwrap() + "'descr': '".len();
    let dtype_end = header[dtype_start..].find('\'').unwrap() + dtype_start;
    let dtype = header[dtype_start..dtype_end].to_owned();
    let shape_start = header.find("'shape': (").unwrap() + "'shape': (".len();
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape = header[shape_start..shape_end]
        .split(',')
        .filter_map(|part| part.trim().parse::<u64>().ok())
        .collect::<Vec<_>>();
    (dtype, shape, bytes[10 + header_len..].to_vec())
}

fn read_f32(path: &Path) -> (Vec<u64>, Vec<f32>) {
    let (dtype, shape, bytes) = read_npy(path);
    assert_eq!(dtype, "<f4");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    (shape, values)
}

fn read_i64(path: &Path) -> (Vec<u64>, Vec<i64>) {
    let (dtype, shape, bytes) = read_npy(path);
    assert_eq!(dtype, "<i8");
    let values = bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (shape, values)
}

fn read_u64(path: &Path) -> (Vec<u64>, Vec<u64>) {
    let (dtype, shape, bytes) = read_npy(path);
    assert_eq!(dtype, "<u8");
    let values = bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (shape, values)
}

fn read_u8(path: &Path) -> (Vec<u64>, Vec<u8>) {
    let (dtype, shape, bytes) = read_npy(path);
    assert_eq!(dtype, "|u1");
    (shape, bytes)
}

fn array_path(
    root: &Path,
    shard: &sichuan_mahjong_neural::prepare::ShardManifest,
    name: &str,
) -> PathBuf {
    root.join(&shard.arrays[name].file)
}

fn split_manifest_mut(manifest: &mut DatasetManifest, split: Split) -> &mut SplitManifest {
    match split {
        Split::Train => &mut manifest.splits.train,
        Split::Validation => &mut manifest.splits.validation,
        Split::Test => &mut manifest.splits.test,
    }
}

fn merge_counts(target: &mut Counts, other: &Counts) {
    if other.decisions > 0 {
        if target.decisions == 0 {
            target.min_candidates = other.min_candidates;
            target.max_candidates = other.max_candidates;
        } else {
            target.min_candidates = target.min_candidates.min(other.min_candidates);
            target.max_candidates = target.max_candidates.max(other.max_candidates);
        }
    }
    target.episodes += other.episodes;
    target.decisions += other.decisions;
    target.candidates += other.candidates;
    target.forced_decisions += other.forced_decisions;
    target.eligible_decisions += other.eligible_decisions;
}

fn recompute_counts(manifest: &mut DatasetManifest) {
    for split in Split::ALL {
        let split_manifest = split_manifest_mut(manifest, split);
        let mut counts = Counts::default();
        for shard in &split_manifest.shards {
            merge_counts(&mut counts, &shard.counts);
        }
        split_manifest.counts = counts;
    }
    let mut counts = Counts::default();
    merge_counts(&mut counts, &manifest.splits.train.counts);
    merge_counts(&mut counts, &manifest.splits.validation.counts);
    merge_counts(&mut counts, &manifest.splits.test.counts);
    manifest.counts = counts;
}

fn write_cache_manifest(cache: &Path, manifest: &mut DatasetManifest) {
    manifest.cache_id = compute_cache_id(manifest);
    manifest.validate().unwrap();
    let mut bytes = serde_json::to_vec_pretty(manifest).unwrap();
    bytes.push(b'\n');
    fs::write(cache.join(DATASET_FILE), bytes).unwrap();
}

fn refresh_array_entry(
    cache: &Path,
    manifest: &mut DatasetManifest,
    split: Split,
    shard_index: usize,
    name: &str,
) {
    let entry = split_manifest_mut(manifest, split).shards[shard_index]
        .arrays
        .get_mut(name)
        .unwrap();
    let path = cache.join(&entry.file);
    let bytes = fs::read(&path).unwrap();
    entry.bytes = bytes.len() as u64;
    entry.sha256 = sha256_hex(&bytes);
}

fn rewrite_episode_seed(
    cache: &Path,
    manifest: &mut DatasetManifest,
    split: Split,
    shard_index: usize,
    decision_index: usize,
    seed: u64,
) {
    let file = split_manifest_mut(manifest, split).shards[shard_index].arrays["episode_seeds"]
        .file
        .clone();
    let path = cache.join(file);
    let mut bytes = fs::read(&path).unwrap();
    let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let offset = 10 + header_len + decision_index * 8;
    bytes[offset..offset + 8].copy_from_slice(&seed.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    refresh_array_entry(cache, manifest, split, shard_index, "episode_seeds");
    write_cache_manifest(cache, manifest);
}

#[derive(Clone, Debug)]
struct ExpectedDecision {
    state: Vec<f32>,
    features: Vec<Vec<f32>>,
    target: usize,
    seed: u64,
    index: u64,
    seat: u8,
    stage: u8,
    kind: u8,
}

fn expected_split(episodes: &[Episode], split: Split) -> Vec<ExpectedDecision> {
    let mut expected = Vec::new();
    for episode in episodes.iter().filter(|episode| episode.split == split) {
        for sample in &episode.decisions {
            expected.push(ExpectedDecision {
                state: sample.state.clone(),
                features: sample.candidate_features.clone(),
                target: sample.target_index,
                seed: episode.id.seed.as_u64(),
                index: sample.decision_index,
                seat: sample.actor_seat.index() as u8,
                stage: sichuan_mahjong_neural::collection::stage_id(sample.stage),
                kind: action_kind_id(&sample.candidates[sample.target_index]),
            });
        }
    }
    expected
}

fn expected_counts(episodes: &[Episode], split: Split) -> Counts {
    let mut counts = Counts::default();
    for episode in episodes.iter().filter(|episode| episode.split == split) {
        counts.episodes += 1;
        for sample in &episode.decisions {
            counts.decisions += 1;
            counts.candidates += sample.candidate_count() as u64;
            if sample.candidate_count() == 1 {
                counts.forced_decisions += 1;
            } else {
                counts.eligible_decisions += 1;
            }
            if counts.decisions == 1 {
                counts.min_candidates = sample.candidate_count();
                counts.max_candidates = sample.candidate_count();
            } else {
                counts.min_candidates = counts.min_candidates.min(sample.candidate_count());
                counts.max_candidates = counts.max_candidates.max(sample.candidate_count());
            }
        }
    }
    counts
}

#[test]
fn preparation_preserves_features_targets_order_metadata_splits_forced_and_checksums() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    let outcome = collect(&corpus_root, &source, 0, 13, 5, 10_000);
    let corpus = read_corpus(&corpus_root).unwrap();
    assert_eq!(corpus.episodes.len(), 13);
    let cache = cache_root(temp.path(), "cache");
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let manifest = prepare(&snapshot, &cache, &options()).unwrap();

    assert_eq!(manifest.format, DATASET_FORMAT);
    assert_eq!(
        manifest.preparation_format_version,
        PREPARATION_FORMAT_VERSION
    );
    assert_eq!(manifest.preparation_version, PREPARATION_VERSION);
    assert_eq!(manifest.encoding_version, ENCODING_VERSION);
    assert_eq!(manifest.source_revision, source);
    assert_eq!(manifest.corpus_manifest_sha256, snapshot.manifest_sha256());
    assert_eq!(
        manifest.corpus_manifest_sha256,
        sha256_hex(&fs::read(manifest_path(&corpus_root)).unwrap())
    );
    assert_eq!(manifest.cache_id.len(), 64);
    assert!(manifest
        .cache_id
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(manifest.layout.state_width, STATE_WIDTH as u64);
    assert_eq!(manifest.layout.candidate_width, ACTION_WIDTH as u64);
    assert_eq!(manifest.provenance.collection.source_revision, source);
    assert!(!manifest
        .provenance
        .preparation
        .build
        .package_version
        .is_empty());
    assert!(["clean", "dirty", "unavailable"]
        .contains(&manifest.provenance.preparation.build_attestation.as_str()));
    assert!(manifest
        .provenance
        .preparation
        .build_attestation
        .eq(&manifest.provenance.preparation.build_attestation));

    let total = verify_dataset(&cache).unwrap();
    assert_eq!(total.cache_id, manifest.cache_id);
    assert_eq!(total.counts.episodes, 13);
    assert_eq!(total.counts.decisions, outcome.committed_stats.decisions);

    let mut global = Counts::default();
    for split in Split::ALL {
        let expected = expected_split(&corpus.episodes, split);
        let expected_counts = expected_counts(&corpus.episodes, split);
        let split_manifest = manifest.splits.get(split);
        assert_eq!(split_manifest.split, split);
        assert_eq!(split_manifest.counts, expected_counts);
        global.episodes += expected_counts.episodes;
        global.decisions += expected_counts.decisions;
        global.candidates += expected_counts.candidates;
        global.forced_decisions += expected_counts.forced_decisions;
        global.eligible_decisions += expected_counts.eligible_decisions;
        if expected_counts.decisions > 0 {
            if global.min_candidates == 0 {
                global.min_candidates = expected_counts.min_candidates;
            } else {
                global.min_candidates = global.min_candidates.min(expected_counts.min_candidates);
            }
            global.max_candidates = global.max_candidates.max(expected_counts.max_candidates);
        }
        if expected.is_empty() {
            assert!(split_manifest.shards.is_empty());
            continue;
        }
        assert!(!split_manifest.shards.is_empty());
        let mut cursor = 0usize;
        for shard in &split_manifest.shards {
            let decisions = shard.counts.decisions as usize;
            let candidates = shard.counts.candidates as usize;
            assert!(decisions > 0);
            assert!(cursor + decisions <= expected.len());
            let (states_shape, states) = read_f32(&array_path(&cache, shard, "states"));
            assert_eq!(states_shape, vec![decisions as u64, STATE_WIDTH as u64]);
            let (features_shape, features) =
                read_f32(&array_path(&cache, shard, "candidate_features"));
            assert_eq!(features_shape, vec![candidates as u64, ACTION_WIDTH as u64]);
            let (offsets_shape, offsets) =
                read_i64(&array_path(&cache, shard, "candidate_offsets"));
            assert_eq!(offsets_shape, vec![decisions as u64 + 1]);
            let (targets_shape, targets) = read_i64(&array_path(&cache, shard, "targets"));
            assert_eq!(targets_shape, vec![decisions as u64]);
            let (seeds_shape, seeds) = read_u64(&array_path(&cache, shard, "episode_seeds"));
            assert_eq!(seeds_shape, vec![decisions as u64]);
            let (indices_shape, indices) = read_u64(&array_path(&cache, shard, "decision_indices"));
            assert_eq!(indices_shape, vec![decisions as u64]);
            let (seats_shape, seats) = read_u8(&array_path(&cache, shard, "actor_seats"));
            assert_eq!(seats_shape, vec![decisions as u64]);
            let (stages_shape, stages) = read_u8(&array_path(&cache, shard, "stages"));
            assert_eq!(stages_shape, vec![decisions as u64]);
            let (kinds_shape, kinds) = read_u8(&array_path(&cache, shard, "target_kinds"));
            assert_eq!(kinds_shape, vec![decisions as u64]);

            assert_eq!(offsets[0], 0);
            assert_eq!(*offsets.last().unwrap(), candidates as i64);
            let mut candidate_cursor = 0usize;
            for local in 0..decisions {
                let expected = &expected[cursor + local];
                let candidate_count = expected.features.len();
                assert_eq!(
                    offsets[local], candidate_cursor as i64,
                    "{split} shard {} decision {local} offset",
                    shard.index
                );
                assert_eq!(
                    offsets[local + 1],
                    (candidate_cursor + candidate_count) as i64
                );
                let state_start = local * STATE_WIDTH;
                for (slot, value) in expected.state.iter().enumerate() {
                    assert_eq!(
                        states[state_start + slot].to_bits(),
                        value.to_bits(),
                        "{split} shard {} decision {local} state {slot}",
                        shard.index
                    );
                }
                for (row, expected_row) in expected.features.iter().enumerate() {
                    for (slot, value) in expected_row.iter().enumerate() {
                        let feature = features[(candidate_cursor + row) * ACTION_WIDTH + slot];
                        assert_eq!(
                            feature.to_bits(),
                            value.to_bits(),
                            "{split} shard {} decision {local} candidate {row} feature {slot}",
                            shard.index
                        );
                    }
                }
                assert_eq!(targets[local] as usize, expected.target);
                assert_eq!(seeds[local], expected.seed);
                assert_eq!(indices[local], expected.index);
                assert_eq!(seats[local], expected.seat);
                assert_eq!(stages[local], expected.stage);
                assert_eq!(kinds[local], expected.kind);
                candidate_cursor += candidate_count;
            }
            assert_eq!(candidate_cursor, candidates);
            cursor += decisions;
        }
        assert_eq!(cursor, expected.len());
    }
    assert_eq!(manifest.counts, global);

    for split in Split::ALL {
        for shard in &manifest.splits.get(split).shards {
            for (name, entry) in &shard.arrays {
                assert_eq!(
                    entry.file,
                    format!("arrays/{}/{:05}.{name}.npy", split, shard.index)
                );
                let path = cache.join(&entry.file);
                let bytes = fs::read(&path).unwrap();
                assert_eq!(entry.bytes, bytes.len() as u64, "{path:?}");
                assert_eq!(entry.sha256, sha256_hex(&bytes), "{path:?}");
            }
        }
    }

    let dataset_path = cache.join(DATASET_FILE);
    let dataset_bytes = fs::read(&dataset_path).unwrap();
    let json: Value = serde_json::from_slice(&dataset_bytes).unwrap();
    assert_eq!(json["format"], DATASET_FORMAT);
    assert_eq!(json["splits"]["train"]["split"], "train");
    assert_eq!(json["splits"]["validation"]["split"], "validation");
    assert_eq!(json["splits"]["test"]["split"], "test");

    // A second preparation of the same snapshot has the same identity and
    // inventory, and never depends on the output path.
    let cache_again = cache_root(temp.path(), "cache-again");
    let again = prepare(&snapshot, &cache_again, &options()).unwrap();
    assert_eq!(again.cache_id, manifest.cache_id);
    assert_eq!(again.splits, manifest.splits);
    assert_eq!(again.counts, manifest.counts);

    // Existing output is never overwritten.
    let before = fs::read(&dataset_path).unwrap();
    assert!(prepare(&snapshot, &cache, &options()).is_err());
    assert_eq!(fs::read(&dataset_path).unwrap(), before);
    verify_dataset(&cache).unwrap();

    // Unsupported versions are rejected by the manifest reader.
    for (field, version) in [
        ("preparation_format_version", 2),
        ("preparation_version", 2),
        ("encoding_version", 2),
    ] {
        let mut changed: Value = serde_json::from_slice(&dataset_bytes).unwrap();
        changed[field] = Value::from(version);
        fs::write(&dataset_path, serde_json::to_vec_pretty(&changed).unwrap()).unwrap();
        let error = read_dataset_manifest(&cache).unwrap_err();
        assert!(
            error.to_string().contains(&format!("version {version}")),
            "{field}: {error}"
        );
    }
    fs::write(&dataset_path, &dataset_bytes).unwrap();
    verify_dataset(&cache).unwrap();

    // Any listed array byte change fails verification.
    let first_shard = &manifest.splits.train.shards[0];
    let state_path = array_path(&cache, first_shard, "states");
    let mut state_bytes = fs::read(&state_path).unwrap();
    let data_offset = 10 + usize::from(u16::from_le_bytes([state_bytes[8], state_bytes[9]]));
    state_bytes[data_offset] ^= 0x01;
    fs::write(&state_path, &state_bytes).unwrap();
    let error = verify_dataset(&cache).unwrap_err();
    assert!(error.to_string().contains("checksum"), "{error}");
}

#[test]
fn captured_snapshot_ignores_later_appends_and_cache_is_corpus_independent() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 3, 2, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let captured_sha = snapshot.manifest_sha256().to_owned();
    assert_eq!(snapshot.manifest().total_games(), 3);
    let appended = append(&corpus_root, &source, 2);
    assert_eq!(appended.manifest.total_games(), 5);
    assert_ne!(
        captured_sha,
        sha256_hex(&fs::read(manifest_path(&corpus_root)).unwrap())
    );

    let old_cache = cache_root(temp.path(), "old-cache");
    let old = prepare(&snapshot, &old_cache, &options()).unwrap();
    assert_eq!(old.counts.episodes, 3);
    assert_eq!(old.corpus_manifest_sha256, captured_sha);

    let fresh = CorpusSnapshot::capture(&corpus_root).unwrap();
    let new_cache = cache_root(temp.path(), "new-cache");
    let new = prepare(&fresh, &new_cache, &options()).unwrap();
    assert_eq!(new.counts.episodes, 5);
    assert!(new.counts.decisions > old.counts.decisions);

    // Appends do not invalidate an already prepared cache.
    verify_dataset(&old_cache).unwrap();
    verify_dataset(&new_cache).unwrap();

    // The cache is self-contained and usable after the corpus is gone.
    let moved = cache_root(temp.path(), "moved-cache");
    fs::rename(&old_cache, &moved).unwrap();
    fs::remove_dir_all(&corpus_root).unwrap();
    let verified = verify_dataset(&moved).unwrap();
    assert_eq!(verified.cache_id, old.cache_id);
    assert_eq!(verified.counts, old.counts);
    assert!(moved
        .join(&verified.splits.train.shards[0].arrays["states"].file)
        .is_file());
}

#[test]
fn validation_failure_publishes_nothing_and_cleans_staging() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    let outcome = collect(&corpus_root, &source, 0, 3, 2, 10_000);
    let shard = shard_path(&corpus_root, outcome.manifest.shards[0].first_seed);
    let mut bytes = fs::read(&shard).unwrap();
    bytes.extend_from_slice(b"corrupt\n");
    fs::write(&shard, bytes).unwrap();

    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "failed-cache");
    let error = prepare(&snapshot, &cache, &options()).unwrap_err();
    assert!(!error.to_string().is_empty(), "{error}");
    assert!(!cache.exists());
    let staging = fs::read_dir(temp.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".bc-prepare-")
        })
        .count();
    assert_eq!(staging, 0, "failed preparation must clean staging");
}

#[test]
fn empty_splits_are_listed_with_zero_counts_and_no_shards() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 1, 1, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "cache");
    let manifest = prepare(&snapshot, &cache, &options()).unwrap();
    assert_eq!(manifest.splits.train.counts.episodes, 1);
    assert!(!manifest.splits.train.shards.is_empty());
    for split in [Split::Validation, Split::Test] {
        let entry = manifest.splits.get(split);
        assert!(entry.shards.is_empty());
        assert_eq!(entry.counts, Counts::default());
    }
    verify_dataset(&cache).unwrap();
}

#[test]
fn cli_prepare_and_verify_are_strict() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 2, 2, 10_000);
    let cache = cache_root(temp.path(), "cli-cache");
    let prepare_output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&cache)
        .args(["--shard-decisions", "40", "--shard-candidates", "600"])
        .output()
        .unwrap();
    assert!(prepare_output.status.success(), "{prepare_output:?}");
    let json: Value = serde_json::from_slice(&prepare_output.stdout).unwrap();
    assert_eq!(json["format"], DATASET_FORMAT);
    assert_eq!(json["counts"]["episodes"], 2);
    assert!(json["cache_id"].as_str().unwrap().len() == 64);

    let verify_output = Command::new(EXE)
        .args(["verify", "--cache"])
        .arg(&cache)
        .output()
        .unwrap();
    assert!(verify_output.status.success(), "{verify_output:?}");
    let stdout = String::from_utf8(verify_output.stdout).unwrap();
    assert!(stdout.starts_with("ok: "), "{stdout}");
    assert!(stdout.contains("2 episodes"), "{stdout}");

    let repeated = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&cache)
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert!(String::from_utf8(repeated.stderr)
        .unwrap()
        .contains("already exists"));
    assert!(cache.join(DATASET_FILE).is_file());

    let help = Command::new(EXE).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("bc-prepare prepare"));
    assert!(help.contains("bc-prepare verify"));
    assert!(help.contains("20000"));
    assert!(help.contains("500000"));

    let usage = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(cache_root(temp.path(), "bad"))
        .args(["--shard-decisions", "0"])
        .output()
        .unwrap();
    assert!(!usage.status.success());
    assert!(String::from_utf8(usage.stderr)
        .unwrap()
        .contains("must be positive"));
}

#[test]
fn manifest_json_matches_the_python_handoff_shape() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 2, 2, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "cache");
    let manifest = prepare(&snapshot, &cache, &options()).unwrap();
    let value: Value =
        serde_json::from_slice(&fs::read(cache.join(DATASET_FILE)).unwrap()).unwrap();
    for key in [
        "format",
        "preparation_format_version",
        "preparation_version",
        "encoding_version",
        "cache_id",
        "split_algorithm",
        "source_revision",
        "corpus_manifest_sha256",
        "corpus_manifest",
        "layout",
        "provenance",
        "counts",
        "splits",
    ] {
        assert!(value.get(key).is_some(), "missing {key}");
    }
    for key in [
        "episodes",
        "decisions",
        "candidates",
        "forced_decisions",
        "eligible_decisions",
        "min_candidates",
        "max_candidates",
    ] {
        assert!(value["counts"].get(key).is_some(), "missing counts.{key}");
    }
    for split in ["train", "validation", "test"] {
        let entry = &value["splits"][split];
        assert_eq!(entry["split"], split);
        assert!(entry["counts"].is_object());
        assert!(entry["shards"].is_array());
        for shard in entry["shards"].as_array().unwrap() {
            for key in ["index", "counts", "arrays"] {
                assert!(shard.get(key).is_some(), "missing shard {key}");
            }
            for name in sichuan_mahjong_neural::prepare::ARRAY_NAMES {
                let array = &shard["arrays"][name];
                for key in ["file", "dtype", "shape", "bytes", "sha256"] {
                    assert!(array.get(key).is_some(), "missing arrays.{name}.{key}");
                }
            }
        }
    }
    assert_eq!(value["splits"]["validation"]["counts"]["decisions"], 0);
    assert_eq!(
        value["splits"]["validation"]["shards"],
        Value::Array(vec![])
    );
    assert_eq!(manifest.format, "mahjong-bc-dataset");
    assert_eq!(manifest.counts, manifest.splits.train.counts);
}

#[test]
fn jsonl_reader_is_not_used_twice_for_preparation() {
    // A cache prepared from a valid corpus stays valid when unlisted files are
    // added later, matching the captured-snapshot contract.
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 2, 2, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "cache");
    prepare(&snapshot, &cache, &options()).unwrap();
    let unlisted = corpus_root.join("shards").join("unlisted.jsonl");
    fs::write(&unlisted, b"not json\n").unwrap();
    assert!(manifest(&corpus_root).shards.len() == 1);
    verify_dataset(&cache).unwrap();
    // Reading the corpus strictly still ignores unlisted files.
    read_corpus(&corpus_root).unwrap();
}

#[test]
fn verify_rejects_tampered_derived_counts() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 3, 2, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "cache");
    prepare(&snapshot, &cache, &options()).unwrap();
    let pristine = read_dataset_manifest(&cache).unwrap();
    let dataset_path = cache.join(DATASET_FILE);
    let original_json = fs::read(&dataset_path).unwrap();
    assert!(!pristine.splits.train.shards.is_empty());

    let mut tampered = pristine.clone();
    let shard = &mut tampered.splits.train.shards[0];
    assert!(
        shard.counts.eligible_decisions > 0,
        "fixture needs an eligible decision"
    );
    shard.counts.forced_decisions += 1;
    shard.counts.eligible_decisions -= 1;
    recompute_counts(&mut tampered);
    write_cache_manifest(&cache, &mut tampered);
    let error = verify_dataset(&cache).unwrap_err();
    assert!(error.to_string().contains("derived counts"), "{error}");
    fs::write(&dataset_path, &original_json).unwrap();
    verify_dataset(&cache).unwrap();

    let mut tampered = pristine.clone();
    tampered.splits.train.shards[0].counts.max_candidates += 1;
    recompute_counts(&mut tampered);
    write_cache_manifest(&cache, &mut tampered);
    let error = verify_dataset(&cache).unwrap_err();
    assert!(error.to_string().contains("derived counts"), "{error}");
    fs::write(&dataset_path, &original_json).unwrap();
    verify_dataset(&cache).unwrap();
}

#[test]
fn verify_rejects_duplicate_and_out_of_range_episode_seeds() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 3, 2, 10_000);
    let snapshot = CorpusSnapshot::capture(&corpus_root).unwrap();
    let cache = cache_root(temp.path(), "cache");
    let split_options = PrepareOptions {
        max_decisions_per_shard: 20,
        max_candidates_per_shard: 200,
    };
    prepare(&snapshot, &cache, &split_options).unwrap();
    let pristine = read_dataset_manifest(&cache).unwrap();
    assert!(
        pristine.splits.train.shards.len() >= 2,
        "fixture needs at least two train output shards"
    );

    let first_path = array_path(&cache, &pristine.splits.train.shards[0], "episode_seeds");
    let second_path = array_path(&cache, &pristine.splits.train.shards[1], "episode_seeds");
    let (_, first_seeds) = read_u64(&first_path);
    let (_, second_seeds) = read_u64(&second_path);
    let first_seed = first_seeds[0];
    let original_second = fs::read(&second_path).unwrap();
    let original_first = fs::read(&first_path).unwrap();
    let dataset_path = cache.join(DATASET_FILE);
    let original_json = fs::read(&dataset_path).unwrap();
    assert_ne!(first_seed, second_seeds[0]);

    let mut tampered = pristine.clone();
    rewrite_episode_seed(&cache, &mut tampered, Split::Train, 1, 0, first_seed);
    let error = verify_dataset(&cache).unwrap_err();
    assert!(
        error.to_string().contains("more than one episode block"),
        "{error}"
    );
    fs::write(&second_path, &original_second).unwrap();
    fs::write(&dataset_path, &original_json).unwrap();
    verify_dataset(&cache).unwrap();

    let end_seed = u64::try_from(
        u128::from(pristine.corpus_manifest.seed_start.as_u64())
            + pristine.corpus_manifest.total_games(),
    )
    .unwrap();
    let mut tampered = pristine.clone();
    rewrite_episode_seed(&cache, &mut tampered, Split::Train, 0, 0, end_seed);
    let error = verify_dataset(&cache).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("outside the captured snapshot seed range"),
        "{error}"
    );
    fs::write(&first_path, &original_first).unwrap();
    fs::write(&dataset_path, &original_json).unwrap();
    verify_dataset(&cache).unwrap();
}

#[test]
fn cli_report_path_must_be_outside_cache_and_is_checked_before_prepare() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let source = revision(REV_A);
    collect(&corpus_root, &source, 0, 1, 1, 10_000);

    let out = cache_root(temp.path(), "out-inside");
    let report = out.join(DATASET_FILE);
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&report)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("outside the cache root"));

    let out = cache_root(temp.path(), "out-dotted");
    let dotted = out
        .join("..")
        .join(out.file_name().unwrap())
        .join("report.json");
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&dotted)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("outside the cache root"));

    let existing = temp.path().join("report-exists.json");
    fs::write(&existing, "{}").unwrap();
    let out = cache_root(temp.path(), "out-existing-report");
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&existing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("already exists"));

    let directory = temp.path().join("report-directory");
    fs::create_dir(&directory).unwrap();
    let out = cache_root(temp.path(), "out-directory-report");
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("is a directory"));

    let missing = temp.path().join("missing").join("report.json");
    let out = cache_root(temp.path(), "out-missing-parent");
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&missing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("does not exist"));

    let out = cache_root(temp.path(), "out-success");
    let report = temp.path().join("report-success.json");
    let output = Command::new(EXE)
        .args(["prepare", "--corpus"])
        .arg(&corpus_root)
        .args(["--out"])
        .arg(&out)
        .args(["--report"])
        .arg(&report)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(out.join(DATASET_FILE).is_file());
    assert!(report.is_file());
    let report_json: Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
    assert_eq!(report_json["format"], DATASET_FORMAT);
    assert_eq!(
        report_json["cache_id"],
        read_dataset_manifest(&out).unwrap().cache_id
    );
    assert!(!out.join("report.json").exists());
}
