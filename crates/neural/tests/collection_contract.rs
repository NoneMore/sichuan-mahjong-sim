#[path = "collection_support/mod.rs"]
mod support;

use std::fs;

use serde_json::Value;
use support::{
    append, collect, manifest, mutate_first_decision, mutate_manifest, read_json_lines,
    refresh_manifest_checksums, revision, root, sha256_hex, shard_files, write_json_lines,
    write_manifest, REV_A, REV_B,
};

use sichuan_mahjong_neural::collection::{
    manifest_path, read_corpus, read_manifest, report_corpus, seed_split, shard_path, split_bucket,
    BuildInfo, CollectionConfig, CollectionError, Corpus, CorpusLock, Manifest, Seed, ShardEntry,
    SourceRevision, Split, FORMAT_VERSION, SPLIT_ALGORITHM,
};
use sichuan_mahjong_neural::encoding::ENCODING_VERSION;
use sichuan_mahjong_sim::Action;

#[test]
fn seeds_and_fixed_split_boundaries() {
    assert_eq!(Seed::from_u64(158).to_hex(), "000000000000009e");
    assert_eq!(
        Seed::parse_hex("000000000000009e").unwrap(),
        Seed::from_u64(158)
    );
    assert_eq!(
        serde_json::to_string(&Seed::from_u64(158)).unwrap(),
        "\"000000000000009e\""
    );
    assert!(Seed::parse_hex("000000000000009E").is_err());
    assert!(Seed::parse_hex("9e").is_err());
    assert!(Seed::parse_hex("000000000000000g").is_err());

    let expected = [
        (0, 41, Split::Train),
        (2, 85, Split::Validation),
        (62, 89, Split::Validation),
        (158, 79, Split::Train),
        (167, 80, Split::Validation),
        (214, 90, Split::Test),
        (u64::MAX, 60, Split::Train),
    ];
    for (seed, bucket, split) in expected {
        let seed = Seed::from_u64(seed);
        assert_eq!(split_bucket(seed), bucket, "seed {seed}");
        assert_eq!(seed_split(seed), split, "seed {seed}");
    }
    assert_eq!(Split::Train.as_str(), "train");
    assert_eq!(Split::Validation.as_str(), "validation");
    assert_eq!(Split::Test.as_str(), "test");
}

#[test]
fn source_revision_is_strict_and_dirty_builds_fail_closed() {
    assert_eq!(revision(REV_A).as_str(), REV_A);
    let sha256 = "a".repeat(64);
    assert_eq!(SourceRevision::parse(&sha256).unwrap().as_str(), sha256);
    assert!(SourceRevision::parse(&"A".repeat(40)).is_err());
    assert!(SourceRevision::parse(&"a".repeat(39)).is_err());
    assert!(SourceRevision::parse(&"g".repeat(40)).is_err());

    let mut clean = sichuan_mahjong_neural::collection::build_info();
    clean.git_available = true;
    clean.git_dirty = false;
    clean.git_commit = Some(REV_A.to_owned());
    assert_eq!(
        SourceRevision::from_build_info(&clean).unwrap(),
        revision(REV_A)
    );

    let mut dirty = clean.clone();
    dirty.git_dirty = true;
    assert!(matches!(
        SourceRevision::from_build_info(&dirty),
        Err(sichuan_mahjong_neural::collection::SourceProvenanceError::Dirty { .. })
    ));

    let mut unavailable = clean.clone();
    unavailable.git_available = false;
    assert!(matches!(
        SourceRevision::from_build_info(&unavailable),
        Err(sichuan_mahjong_neural::collection::SourceProvenanceError::Unavailable { .. })
    ));

    let mut malformed = clean;
    malformed.git_commit = Some("not-a-commit".to_owned());
    assert!(matches!(
        SourceRevision::from_build_info(&malformed),
        Err(sichuan_mahjong_neural::collection::SourceProvenanceError::InvalidCommit { .. })
    ));

    let info = sichuan_mahjong_neural::collection::build_info();
    if let Some(commit) = info.git_commit {
        assert!(commit.len() == 40 || commit.len() == 64, "{commit}");
        assert!(commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
}

#[test]
fn collection_round_trips_exactly_and_retains_canonical_exchange_forced_and_losses() {
    let temp = root();
    let source = revision(REV_A);
    let outcome = collect(temp.path(), &source, 0, 3, 2, 10_000);
    assert_eq!(outcome.requested_games, 3);
    assert_eq!(outcome.committed_games, 3);
    assert_eq!(outcome.first_seed, Seed::from_u64(0));
    assert_eq!(outcome.last_seed, Seed::from_u64(2));
    assert_eq!(outcome.committed_stats.games, 3);
    assert_eq!(outcome.manifest.shards.len(), 2);
    assert_eq!(outcome.manifest.shards[0].games, 2);
    assert_eq!(outcome.manifest.shards[1].games, 1);
    assert_eq!(outcome.cache.state, "disabled");
    assert_eq!(outcome.cache.budget_bytes, 0);
    assert!(outcome.timings.total_ns > 0);
    assert!(outcome.timings.rules_ns > 0);
    assert!(outcome.timings.teacher_ns > 0);
    assert!(outcome.timings.encoding_ns > 0);
    assert!(outcome.timings.io_ns > 0);
    assert!(outcome.timings.commit_ns > 0);
    assert!(
        outcome.timings.rules_ns
            + outcome.timings.teacher_ns
            + outcome.timings.encoding_ns
            + outcome.timings.io_ns
            + outcome.timings.commit_ns
            <= outcome.timings.total_ns
    );
    assert!(outcome.timings.games_per_sec > 0.0);
    assert!(outcome.timings.decisions_per_sec > 0.0);
    assert_eq!(outcome.timings.games, 3);
    let outcome_json = serde_json::to_string(&outcome).unwrap();
    assert!(outcome_json.contains("\"games_per_sec\""));
    assert!(outcome_json.contains("\"committed_stats\""));
    assert!(outcome_json.contains("\"total_storage_bytes\""));
    assert!(outcome_json.contains("\"cache\""));
    assert_eq!(
        outcome.total_storage_bytes,
        outcome.shard_bytes + outcome.manifest_bytes
    );

    let data = read_corpus(temp.path()).unwrap();
    assert_eq!(data.episodes.len(), 3);
    let mut exchange_decisions = 0usize;
    let mut forced = 0usize;
    let mut multi = 0usize;
    let mut selected_pass = false;
    for (index, episode) in data.episodes.iter().enumerate() {
        let seed = Seed::from_u64(index as u64);
        assert_eq!(episode.id.seed, seed);
        assert_eq!(episode.id.source_revision, source);
        assert_eq!(episode.id.encoding_version, ENCODING_VERSION);
        assert_eq!(episode.split, seed_split(seed));
        assert!(!episode.decisions.is_empty());
        assert!(episode.result.winners.iter().any(|&won| !won));
        let replayed = support::replay(index as u64, &source);
        assert_eq!(episode, &replayed);
        for sample in &episode.decisions {
            assert!(sample.target_index < sample.candidates.len());
            assert_eq!(sample.candidate_features.len(), sample.candidates.len());
            assert_eq!(sample.state.len(), 1013);
            if sample.candidates.len() == 1 {
                forced += 1;
            } else {
                multi += 1;
            }
            if sample.candidates[sample.target_index] == Action::Pass {
                selected_pass = true;
            }
            if sample.stage == sichuan_mahjong_sim::DecisionStage::Exchange {
                exchange_decisions += 1;
                for candidate in &sample.candidates {
                    match candidate {
                        Action::Exchange(tiles) => {
                            assert!(tiles[0] <= tiles[1] && tiles[1] <= tiles[2]);
                            assert_eq!(tiles[0].suit(), tiles[1].suit());
                            assert_eq!(tiles[1].suit(), tiles[2].suit());
                        }
                        other => panic!("exchange candidate {other:?}"),
                    }
                }
            }
        }
    }
    assert!(exchange_decisions >= 3);
    assert!(forced > 0, "expected at least one forced decision");
    assert!(multi > 0, "expected multi-candidate decisions");
    assert!(selected_pass, "expected retained Pass labels");
    assert_eq!(forced, outcome.committed_stats.forced_decisions as usize);
    assert_eq!(
        multi,
        outcome.committed_stats.multi_candidate_decisions as usize
    );
}

#[test]
fn append_is_source_isolated_and_never_duplicates_seeds() {
    let temp = root();
    let source = revision(REV_A);
    let first = collect(temp.path(), &source, 0, 2, 2, 10_000);
    let first_shard = shard_path(temp.path(), first.manifest.shards[0].first_seed);
    let first_shard_bytes = fs::read(&first_shard).unwrap();
    let before = fs::read(manifest_path(temp.path())).unwrap();

    let error = match Corpus::open_append(temp.path(), &revision(REV_B)) {
        Ok(_) => panic!("foreign source unexpectedly opened"),
        Err(error) => error,
    };
    assert!(matches!(error, CollectionError::SourceMismatch { .. }));
    assert_eq!(fs::read(manifest_path(temp.path())).unwrap(), before);

    let outcome = append(temp.path(), &source, 3);
    assert_eq!(outcome.committed_stats.games, 3);
    assert_eq!(outcome.manifest.total_games(), 5);
    let data = read_corpus(temp.path()).unwrap();
    let seeds: Vec<_> = data.episodes.iter().map(|ep| ep.id.seed.as_u64()).collect();
    assert_eq!(seeds, vec![0, 1, 2, 3, 4]);
    let shards = &data.manifest.shards;
    assert_eq!(shards.len(), 3);
    assert_eq!(shards[0].first_seed, Seed::from_u64(0));
    assert_eq!(shards[1].first_seed, Seed::from_u64(2));
    assert_eq!(shards[2].first_seed, Seed::from_u64(4));
    assert_eq!(fs::read(&first_shard).unwrap(), first_shard_bytes);
    let report = report_corpus(temp.path()).unwrap();
    assert_eq!(report.stats.games, 5);
    assert_eq!(
        report.stats.decisions,
        data.episodes
            .iter()
            .map(|ep| ep.decisions.len() as u64)
            .sum::<u64>()
    );
}

#[test]
fn smaller_final_shard_append_starts_at_next_seed() {
    let temp = root();
    let source = revision(REV_A);
    let first = collect(temp.path(), &source, 0, 1, 3, 10_000);
    assert_eq!(first.manifest.shards.len(), 1);
    assert_eq!(first.manifest.shards[0].games, 1);

    let second = append(temp.path(), &source, 2);
    assert_eq!(second.manifest.shards.len(), 2);
    assert_eq!(second.manifest.shards[0].first_seed, Seed::from_u64(0));
    assert_eq!(second.manifest.shards[0].games, 1);
    assert_eq!(second.manifest.shards[1].first_seed, Seed::from_u64(1));
    assert_eq!(second.manifest.shards[1].games, 2);
    assert_eq!(
        shard_path(temp.path(), second.manifest.shards[1].first_seed)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
        "0000000000000001.jsonl"
    );
    let data = read_corpus(temp.path()).unwrap();
    assert_eq!(
        data.episodes
            .iter()
            .map(|episode| episode.id.seed.as_u64())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn unlisted_orphan_shards_and_partial_temps_are_ignored_and_replayed() {
    let temp = root();
    let source = revision(REV_A);
    collect(temp.path(), &source, 0, 2, 2, 10_000);
    let first = manifest(temp.path());
    let first_path = shard_path(temp.path(), first.shards[0].first_seed);
    let first_bytes = fs::read(&first_path).unwrap();

    // Simulate a crash after publishing a shard but before the manifest
    // commit: the file for the next range is complete but unlisted.
    fs::write(shard_path(temp.path(), Seed::from_u64(2)), &first_bytes).unwrap();
    fs::write(
        temp.path()
            .join("shards")
            .join(".pending-shard-crash.jsonl.tmp"),
        b"not json\n",
    )
    .unwrap();
    fs::write(
        shard_path(temp.path(), Seed::from_u64(4)),
        b"garbage unlisted shard\n",
    )
    .unwrap();

    append(temp.path(), &source, 2);
    append(temp.path(), &source, 2);
    let data = read_corpus(temp.path()).unwrap();
    assert_eq!(
        data.episodes
            .iter()
            .map(|episode| episode.id.seed.as_u64())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
    assert_eq!(data.manifest.shards.len(), 3);
    assert_eq!(data.manifest.shards[1].first_seed, Seed::from_u64(2));
    assert_eq!(data.manifest.shards[2].first_seed, Seed::from_u64(4));
    report_corpus(temp.path()).unwrap();
}

#[test]
fn manifest_publication_failure_keeps_previous_commit_and_retry_replaces_orphan() {
    let temp = root();
    let source = revision(REV_A);
    collect(temp.path(), &source, 0, 1, 1, 10_000);
    let mut corpus = Corpus::open_append(temp.path(), &source).unwrap();
    let path = manifest_path(temp.path());
    let backup = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();

    let error = corpus.append_with_cache(1, 0).unwrap_err();
    assert!(matches!(error, CollectionError::Io { .. }), "{error}");
    assert_eq!(corpus.manifest().shards.len(), 1);
    assert_eq!(corpus.manifest().shards[0].first_seed, Seed::from_u64(0));

    fs::remove_dir(&path).unwrap();
    fs::write(&path, &backup).unwrap();
    let visible = read_corpus(temp.path()).unwrap();
    assert_eq!(visible.episodes.len(), 1);
    assert_eq!(visible.episodes[0].id.seed, Seed::from_u64(0));
    assert_eq!(visible.manifest.shards.len(), 1);
    corpus.append_with_cache(1, 0).unwrap();
    assert_eq!(corpus.manifest().shards.len(), 2);
    drop(corpus);

    let data = read_corpus(temp.path()).unwrap();
    assert_eq!(
        data.episodes
            .iter()
            .map(|episode| episode.id.seed.as_u64())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(data.manifest.shards[1].first_seed, Seed::from_u64(1));
}

#[test]
fn strict_reading_rejects_corruption_versions_and_shape_or_target_errors() {
    // Unsupported versions and split algorithm.
    for (mutate, expected) in [
        (
            (|value: &mut Value| value["format_version"] = Value::from(2)) as fn(&mut Value),
            "format",
        ),
        (
            (|value: &mut Value| value["encoding_version"] = Value::from(2)) as fn(&mut Value),
            "encoding",
        ),
        (
            (|value: &mut Value| value["split_algorithm"] = Value::from("other")) as fn(&mut Value),
            "split",
        ),
        (
            (|value: &mut Value| value["unknown"] = Value::from(1)) as fn(&mut Value),
            "unknown",
        ),
    ] {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_manifest(temp.path(), mutate);
        let error = read_manifest(temp.path()).unwrap_err();
        match expected {
            "format" => assert!(matches!(
                error,
                CollectionError::UnsupportedFormatVersion { .. }
            )),
            "encoding" => assert!(matches!(
                error,
                CollectionError::UnsupportedEncodingVersion { .. }
            )),
            "split" => assert!(matches!(
                error,
                CollectionError::UnsupportedSplitAlgorithm { .. }
            )),
            _ => assert!(matches!(error, CollectionError::ManifestJson { .. })),
        }
    }

    // Checksum mismatch with still-valid JSON.
    {
        let temp = root();
        let source = revision(REV_A);
        let outcome = collect(temp.path(), &source, 0, 1, 1, 10_000);
        let path = shard_path(temp.path(), outcome.manifest.shards[0].first_seed);
        let mut bytes = fs::read(&path).unwrap();
        let newline = bytes.iter().position(|&byte| byte == b'\n').unwrap();
        bytes.insert(newline, b' ');
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardChecksumMismatch { .. })
        ));
    }

    // Wrong state length.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_first_decision(temp.path(), |decision| {
            decision["state"].as_array_mut().unwrap().pop();
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardInvalid { .. })
        ));
    }

    // Wrong candidate feature row length.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_first_decision(temp.path(), |decision| {
            decision["candidate_features"][0]
                .as_array_mut()
                .unwrap()
                .pop();
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardInvalid { .. })
        ));
    }

    // Unknown action kind.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_first_decision(temp.path(), |decision| {
            decision["candidate_actions"][0][0] = Value::from(99);
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardActionInvalid { .. })
        ));
    }

    // Target outside the candidate list.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_first_decision(temp.path(), |decision| {
            decision["teacher_target_index"] = Value::from(1_000_000);
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardTargetOutOfRange { .. })
        ));
    }

    // Manifest game count mismatch, with a valid checksum.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 5, 10_000);
        mutate_manifest(temp.path(), |value| {
            value["shards"][0]["games"] = Value::from(2);
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardMetadataMismatch { field: "games", .. })
        ));
    }

    // Manifest decision count mismatch, with a valid checksum.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 5, 10_000);
        mutate_manifest(temp.path(), |value| {
            value["shards"][0]["decisions"] = Value::from(99_999);
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardMetadataMismatch {
                field: "decisions",
                ..
            })
        ));
    }

    // Unknown shard record field, with a valid checksum.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        mutate_first_decision(temp.path(), |decision| {
            decision["extra"] = Value::from(1);
        });
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardJson { .. })
        ));
    }

    // Missing result record.
    {
        let temp = root();
        let source = revision(REV_A);
        let outcome = collect(temp.path(), &source, 0, 1, 1, 10_000);
        let path = shard_path(temp.path(), outcome.manifest.shards[0].first_seed);
        let mut lines = read_json_lines(&path);
        lines.pop();
        write_json_lines(&path, &lines);
        refresh_manifest_checksums(temp.path());
        assert!(matches!(
            read_corpus(temp.path()),
            Err(CollectionError::ShardInvalid { .. })
        ));
    }

    // Unlisted corrupt files do not affect a strict read of listed shards.
    {
        let temp = root();
        let source = revision(REV_A);
        collect(temp.path(), &source, 0, 1, 1, 10_000);
        fs::write(
            shard_path(temp.path(), Seed::from_u64(9)),
            b"unlisted garbage\n",
        )
        .unwrap();
        read_corpus(temp.path()).unwrap();
    }
}

#[test]
fn report_counts_zero_support_and_separates_forced_from_primary() {
    let temp = root();
    let source = revision(REV_A);
    let outcome = collect(temp.path(), &source, 0, 3, 2, 10_000);
    let report = report_corpus(temp.path()).unwrap();
    assert_eq!(report.stats, outcome.committed_stats);
    assert_eq!(report.stats.kind.len(), 9);
    assert_eq!(report.stats.stage.len(), 5);
    assert_eq!(report.stats.split.len(), 3);
    assert_eq!(
        report.stats.stage.iter().map(|s| s.decisions).sum::<u64>(),
        report.stats.decisions
    );
    assert_eq!(
        report
            .stats
            .stage
            .iter()
            .map(|s| s.multi_candidate)
            .sum::<u64>(),
        report.stats.multi_candidate_decisions
    );
    assert_eq!(
        report.stats.forced_decisions + report.stats.multi_candidate_decisions,
        report.stats.decisions
    );
    assert_eq!(
        report.stats.kind.iter().map(|k| k.selections).sum::<u64>(),
        report.stats.decisions
    );
    assert_eq!(
        report
            .stats
            .kind
            .iter()
            .map(|k| k.primary_selections)
            .sum::<u64>(),
        report.stats.multi_candidate_decisions
    );
    assert_eq!(
        report.stats.split.iter().map(|s| s.games).sum::<u64>(),
        report.stats.games
    );
    let zero_support = report.stats.zero_support_kinds().count();
    let primary_zero = report.stats.primary_zero_support_kinds().count();
    assert!(
        zero_support > 0,
        "one small pilot should leave some action kinds unobserved: {:?}",
        report.stats.kind
    );
    assert!(primary_zero >= zero_support);
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(json.matches("\"kind_id\"").count(), 9);
    if zero_support > 0 {
        assert!(json.contains("\"opportunities\":0"));
    }
}

#[test]
fn cache_configuration_and_decision_limit_fail_without_partial_commit() {
    let temp = root();
    let source = revision(REV_A);
    let config = CollectionConfig::new(source.clone())
        .with_shard_games(1)
        .with_decision_limit(10_000);
    let mut corpus = Corpus::create(temp.path(), &config).unwrap();
    let error = corpus.append_with_cache(1, 1).unwrap_err();
    assert!(matches!(error, CollectionError::CacheConfig { .. }));
    assert!(corpus.manifest().shards.is_empty());
    drop(corpus);

    let temp = root();
    let config = CollectionConfig::new(source.clone())
        .with_shard_games(1)
        .with_decision_limit(1);
    let mut corpus = Corpus::create(temp.path(), &config).unwrap();
    let error = corpus.append_with_cache(1, 0).unwrap_err();
    assert!(
        matches!(error, CollectionError::DecisionLimit { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("0000000000000000"), "{error}");
    assert!(corpus.manifest().shards.is_empty());
    drop(corpus);
    let entries = fs::read_dir(temp.path().join("shards")).unwrap().count();
    assert_eq!(entries, 0, "failed game must not leave a partial shard");
    assert_eq!(read_corpus(temp.path()).unwrap().episodes.len(), 0);
}

#[test]
fn max_u64_seed_is_collectable_once_and_never_wraps() {
    let temp = root();
    let source = revision(REV_A);
    let outcome = collect(temp.path(), &source, u64::MAX, 1, 1, 10_000);
    assert_eq!(outcome.first_seed, Seed::from_u64(u64::MAX));
    assert_eq!(outcome.last_seed, Seed::from_u64(u64::MAX));
    assert_eq!(outcome.manifest.shards.len(), 1);
    let data = read_corpus(temp.path()).unwrap();
    assert_eq!(data.episodes.len(), 1);
    assert_eq!(data.episodes[0].id.seed, Seed::from_u64(u64::MAX));
    assert!(manifest_path(temp.path()).exists());

    let mut corpus = Corpus::open_append(temp.path(), &source).unwrap();
    let error = corpus.append_with_cache(1, 0).unwrap_err();
    assert!(matches!(error, CollectionError::SeedSpaceExhausted { .. }));
    assert_eq!(corpus.manifest().total_games(), 1);
}

#[test]
fn million_game_metadata_resumes_without_reading_prior_decisions() {
    let temp = root();
    let source = revision(REV_A);
    fs::create_dir_all(temp.path().join("shards")).unwrap();
    let mut manifest = Manifest {
        format_version: FORMAT_VERSION,
        source_revision: source.clone(),
        encoding_version: ENCODING_VERSION,
        split_algorithm: SPLIT_ALGORITHM.to_owned(),
        seed_start: Seed::from_u64(0),
        decision_limit: 10_000,
        shard_games: 1_000,
        shards: Vec::new(),
    };
    for index in 0..1_000u64 {
        let first_seed = Seed::from_u64(index * 1_000);
        manifest.shards.push(ShardEntry {
            first_seed,
            games: 1_000,
            decisions: 100,
            sha256: "0".repeat(64),
        });
        fs::write(
            shard_path(temp.path(), first_seed),
            b"this is not a valid shard and must not be read during resume\n",
        )
        .unwrap();
    }
    write_manifest(temp.path(), &manifest);

    let mut corpus = Corpus::open_append(temp.path(), &source).unwrap();
    let outcome = corpus.append_with_cache(1, 0).unwrap();
    assert_eq!(outcome.first_seed, Seed::from_u64(1_000_000));
    assert_eq!(outcome.manifest.shards.len(), 1_001);
    assert_eq!(outcome.manifest.total_games(), 1_000_001);
    assert_eq!(outcome.manifest.shards.last().unwrap().games, 1);
    drop(corpus);

    let listed = read_manifest(temp.path()).unwrap();
    assert_eq!(listed.shards.len(), 1_001);
}

#[test]
fn process_lock_conflict_and_release() {
    let temp = root();
    let source = revision(REV_A);
    collect(temp.path(), &source, 0, 1, 1, 10_000);
    let held = CorpusLock::try_acquire(temp.path()).unwrap();
    let error = match Corpus::open_append(temp.path(), &source) {
        Ok(_) => panic!("second writer unexpectedly acquired the lock"),
        Err(error) => error,
    };
    assert!(matches!(error, CollectionError::CorpusBusy { .. }));
    drop(held);
    let mut corpus = Corpus::open_append(temp.path(), &source).unwrap();
    corpus.append_with_cache(1, 0).unwrap();
    assert_eq!(corpus.manifest().total_games(), 2);
}

#[test]
fn file_paths_and_checksums_match_manifest() {
    let temp = root();
    let source = revision(REV_A);
    let outcome = collect(temp.path(), &source, 0, 3, 2, 10_000);
    for shard in &outcome.manifest.shards {
        let path = shard_path(temp.path(), shard.first_seed);
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            format!("{}.jsonl", shard.first_seed.to_hex())
        );
        let bytes = fs::read(&path).unwrap();
        assert_eq!(sha256_hex(&bytes), shard.sha256);
        let lines = read_json_lines(&path);
        assert_eq!(
            lines
                .iter()
                .filter(|line| line["type"] == "episode")
                .count() as u64,
            shard.games
        );
        assert_eq!(
            lines
                .iter()
                .filter(|line| line["type"] == "decision")
                .count() as u64,
            shard.decisions
        );
        assert_eq!(
            lines.iter().filter(|line| line["type"] == "result").count() as u64,
            shard.games
        );
    }
    assert_eq!(shard_files(temp.path(), &outcome.manifest).len(), 2);
    let manifest_json: Value =
        serde_json::from_slice(&fs::read(manifest_path(temp.path())).unwrap()).unwrap();
    assert_eq!(manifest_json["format_version"], Value::from(FORMAT_VERSION));
    assert_eq!(
        manifest_json["seed_start"],
        Value::String(Seed::from_u64(0).to_hex())
    );
    assert_eq!(
        manifest_json["split_algorithm"],
        Value::String(SPLIT_ALGORITHM.to_owned())
    );
    let shard_json = &manifest_json["shards"][0];
    assert_eq!(
        shard_json["first_seed"],
        Value::String("0000000000000000".to_owned())
    );
    assert_eq!(shard_json["games"], Value::from(2));
    assert!(shard_json["decisions"].as_u64().unwrap() > 0);
    assert_eq!(shard_json["sha256"].as_str().unwrap().len(), 64);
    assert!(shard_json.get("path").is_none());
    assert!(shard_json.get("episodes").is_none());
}

#[test]
fn build_info_is_serializable_and_has_profile_fields() {
    let info: BuildInfo = sichuan_mahjong_neural::collection::build_info();
    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("\"package_version\""));
    assert!(json.contains("\"profile\""));
    assert!(json.contains("\"target\""));
    let env = sichuan_mahjong_neural::collection::environment_info();
    assert!(!env.os.is_empty());
    assert!(env.available_parallelism >= 1);
    assert_eq!(env.pointer_width, (std::mem::size_of::<usize>() * 8) as u32);
}
