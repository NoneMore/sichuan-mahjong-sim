#[path = "collection_support/mod.rs"]
mod support;

use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use support::{collect, revision, root, REV_A};

use sichuan_mahjong_neural::collection::{
    build_info, manifest_path, read_corpus, shard_path, CollectionError, Corpus, Seed,
    SourceRevision,
};

const EXE: &str = env!("CARGO_BIN_EXE_bc-collect");

#[test]
fn report_and_verify_validate_a_corpus() {
    let temp = root();
    let source = revision(REV_A);
    let outcome = collect(temp.path(), &source, 0, 3, 2, 10_000);

    let report = Command::new(EXE)
        .args(["report", "--root"])
        .arg(temp.path())
        .output()
        .unwrap();
    assert!(report.status.success(), "{:?}", report);
    let text = String::from_utf8(report.stdout).unwrap();
    assert!(text.contains("games: 3"), "{text}");
    assert!(text.contains("zero_support_all:"), "{text}");
    assert!(text.contains("kind opportunities/selections"), "{text}");
    assert!(text.contains(&source.to_string()), "{text}");

    let json = Command::new(EXE)
        .args(["report", "--root"])
        .arg(temp.path())
        .arg("--json")
        .output()
        .unwrap();
    assert!(json.status.success(), "{:?}", json);
    let value: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(value["stats"]["games"], serde_json::Value::from(3));
    assert_eq!(
        value["manifest"]["encoding_version"],
        serde_json::Value::from(1)
    );

    let verify = Command::new(EXE)
        .args(["verify", "--root"])
        .arg(temp.path())
        .output()
        .unwrap();
    assert!(verify.status.success(), "{:?}", verify);
    let text = String::from_utf8(verify.stdout).unwrap();
    assert!(text.contains("ok: 3 games"), "{text}");
    assert!(text.contains("2 shards"), "{text}");

    // Corrupt one listed shard but keep valid JSON: checksum rejection.
    let path = shard_path(temp.path(), outcome.manifest.shards[0].first_seed);
    let mut bytes = fs::read(&path).unwrap();
    let newline = bytes.iter().position(|&byte| byte == b'\n').unwrap();
    bytes.insert(newline, b' ');
    fs::write(&path, bytes).unwrap();
    let verify = Command::new(EXE)
        .args(["verify", "--root"])
        .arg(temp.path())
        .output()
        .unwrap();
    assert!(!verify.status.success());
    let stderr = String::from_utf8(verify.stderr).unwrap();
    assert!(stderr.contains("checksum"), "{stderr}");
}

#[test]
fn compiled_provenance_is_enforced_or_clean_collection_works() {
    let temp = root();
    let corpus_root = temp.path().join("corpus");
    let info = build_info();
    let output = Command::new(EXE)
        .args(["collect", "--root"])
        .arg(&corpus_root)
        .args(["--games", "1", "--cache-bytes", "0"])
        .output()
        .unwrap();
    if info.git_dirty || !info.git_available {
        assert!(!output.status.success(), "{output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("compiled source") || stderr.contains("provenance"),
            "{stderr}"
        );
        assert!(
            !corpus_root.exists(),
            "dirty provenance must fail before writing"
        );
    } else {
        assert!(output.status.success(), "{output:?}");
        let data = read_corpus(&corpus_root).unwrap();
        assert_eq!(data.episodes.len(), 1);
        assert_eq!(
            data.manifest.source_revision,
            SourceRevision::from_compiled_source().unwrap()
        );
    }
}

#[test]
fn subprocess_lock_blocks_second_committer_and_releases_on_exit() {
    let temp = root();
    let source = revision(REV_A);
    collect(temp.path(), &source, 0, 1, 1, 10_000);
    let before = fs::read(manifest_path(temp.path())).unwrap();

    let mut child = Command::new(EXE)
        .args(["lock", "--root"])
        .arg(temp.path())
        .args(["--hold-ms", "3000"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "locked", "child did not acquire the lock");

    let error = match Corpus::open_append(temp.path(), &source) {
        Ok(_) => panic!("second committer acquired a held lock"),
        Err(error) => error,
    };
    assert!(matches!(error, CollectionError::CorpusBusy { .. }));
    assert_eq!(fs::read(manifest_path(temp.path())).unwrap(), before);

    let status = child.wait().unwrap();
    assert!(status.success(), "{status:?}");

    let mut corpus = Corpus::open_append(temp.path(), &source).unwrap();
    corpus.append_with_cache(1, 0).unwrap();
    assert_eq!(corpus.manifest().total_games(), 2);
    assert_eq!(
        read_corpus(temp.path())
            .unwrap()
            .episodes
            .iter()
            .map(|episode| episode.id.seed)
            .collect::<Vec<_>>(),
        vec![Seed::from_u64(0), Seed::from_u64(1)]
    );
}

#[test]
fn command_errors_are_nonzero_and_do_not_touch_inputs() {
    let temp = root();
    let source = revision(REV_A);
    collect(temp.path(), &source, 0, 1, 1, 10_000);
    let before = fs::read(manifest_path(temp.path())).unwrap();

    let output = Command::new(EXE)
        .args(["collect", "--root"])
        .arg(temp.path())
        .args(["--games", "1", "--shard-games", "99"])
        .output()
        .unwrap();
    // In a clean build this fails on immutable shard_games; in a dirty build it
    // fails earlier on provenance. Either way it must not modify the corpus.
    assert!(!output.status.success());
    assert_eq!(fs::read(manifest_path(temp.path())).unwrap(), before);

    let output = Command::new(EXE)
        .args(["report", "--root"])
        .arg(temp.path().join("missing"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("no corpus manifest"));
}

#[test]
fn help_and_usage_are_stable() {
    let output = Command::new(EXE).arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("bc-collect collect"));
    assert!(text.contains("bc-collect report"));
    assert!(text.contains("bc-collect lock"));

    let output = Command::new(EXE).output().unwrap();
    assert!(!output.status.success());

    let output = Command::new(EXE)
        .args(["collect", "--root", "x", "--games", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("must be positive"));
}
