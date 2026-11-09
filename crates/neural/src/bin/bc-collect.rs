//! `bc-collect` command-line interface.
//!
//! `collect` rejects dirty or unavailable compiled source provenance. `report`
//! and `verify` are read-only and require no provenance; they strictly validate
//! every listed shard. `lock` is a diagnostic that acquires the exclusive
//! `LOCK` and optionally holds it, which makes process-exit release testable.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use sichuan_mahjong_neural::collection::{
    manifest_path, report_corpus, verify_corpus, CollectionConfig, CollectionError, Corpus,
    CorpusLock, CorpusReport, SourceRevision,
};
use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;

const USAGE: &str = "\
Usage:
  bc-collect collect --root DIR --games N [options]
  bc-collect report --root DIR [--json]
  bc-collect verify --root DIR
  bc-collect lock --root DIR [--hold-ms N]
  bc-collect --help

collect options:
  --seed-start S       Initial u64 seed for a new corpus (decimal or 0x hex).
  --shard-games N      Positive complete games per shard for a new corpus.
  --decision-limit N   Positive per-game decision limit for a new corpus.
  --cache-bytes B      Shared heuristic cache budget; 0 disables.
  --report PATH        Write the collection outcome JSON to PATH.
  New-corpus defaults: seed-start 0, shard-games 100, decision-limit 10000,
  and the library default shared teacher cache budget.
  The corpus source revision is always the compiled binary's clean Git commit.

report/verify strictly read and validate listed shards. lock is a diagnostic
writer lock holder.
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bc-collect: {}", error.message);
            ExitCode::from(error.code)
        }
    }
}

#[derive(Debug)]
struct CliError {
    message: String,
    code: u8,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 2,
        }
    }

    fn runtime(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 1,
        }
    }
}

impl From<CollectionError> for CliError {
    fn from(error: CollectionError) -> Self {
        Self::runtime(error.to_string())
    }
}

fn run(args: Vec<String>) -> Result<(), CliError> {
    let Some(command) = args.first() else {
        return Err(CliError::usage(USAGE));
    };
    if command == "-h" || command == "--help" {
        print!("{USAGE}");
        return Ok(());
    }
    let command = command.as_str();
    let rest = &args[1..];
    if rest.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    match command {
        "collect" => collect(rest),
        "report" => report(rest),
        "verify" => verify(rest),
        "lock" => lock(rest),
        other => Err(CliError::usage(format!(
            "unknown command {other:?}\n\n{USAGE}"
        ))),
    }
}

fn collect(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(
        &flags,
        &[
            "root",
            "games",
            "seed-start",
            "shard-games",
            "decision-limit",
            "cache-bytes",
            "report",
        ],
    )?;
    let root = PathBuf::from(required(&flags, "root")?);
    let games = positive_u64(required(&flags, "games")?, "games")?;
    let source = SourceRevision::from_compiled_source()
        .map_err(|error| CliError::runtime(error.to_string()))?;
    let cache_budget = match flags.get("cache-bytes") {
        Some(value) => parse_usize(value, "cache-bytes")?,
        None => DEFAULT_BUDGET_BYTES,
    };

    let mut corpus = if manifest_path(&root).exists() {
        let corpus = Corpus::open_append(&root, &source)?;
        check_immutable(&corpus, &flags)?;
        corpus
    } else {
        let mut config = CollectionConfig::new(source);
        if let Some(value) = flags.get("seed-start") {
            config.seed_start = parse_seed(value)?;
        }
        if let Some(value) = flags.get("shard-games") {
            config.shard_games = positive_u64(value, "shard-games")?;
        }
        if let Some(value) = flags.get("decision-limit") {
            config.decision_limit = positive_u64(value, "decision-limit")?;
        }
        Corpus::create(&root, &config)?
    };

    let outcome = corpus.append_with_cache(games, cache_budget)?;
    let json = serde_json::to_string_pretty(&outcome)
        .map_err(|error| CliError::runtime(format!("could not encode outcome JSON: {error}")))?;
    if let Some(path) = flags.get("report") {
        fs::write(path, format!("{json}\n"))
            .map_err(|error| CliError::runtime(format!("could not write {path}: {error}")))?;
    }
    println!("{json}");
    Ok(())
}

fn check_immutable(corpus: &Corpus, flags: &BTreeMap<String, String>) -> Result<(), CliError> {
    let manifest = corpus.manifest();
    if let Some(value) = flags.get("seed-start") {
        let seed = parse_seed(value)?;
        if seed != manifest.seed_start {
            return Err(CliError::runtime(format!(
                "corpus seed_start is {}, not requested {}",
                manifest.seed_start, seed
            )));
        }
    }
    if let Some(value) = flags.get("shard-games") {
        let shard_games = positive_u64(value, "shard-games")?;
        if shard_games != manifest.shard_games {
            return Err(CliError::runtime(format!(
                "corpus shard_games is {}, not requested {}",
                manifest.shard_games, shard_games
            )));
        }
    }
    if let Some(value) = flags.get("decision-limit") {
        let decision_limit = positive_u64(value, "decision-limit")?;
        if decision_limit != manifest.decision_limit {
            return Err(CliError::runtime(format!(
                "corpus decision_limit is {}, not requested {}",
                manifest.decision_limit, decision_limit
            )));
        }
    }
    Ok(())
}

fn report(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(&flags, &["root", "json"])?;
    let root = PathBuf::from(required(&flags, "root")?);
    let report = report_corpus(&root)?;
    if flags.contains_key("json") {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|error| CliError::runtime(format!("could not encode report: {error}")))?;
        println!("{json}");
    } else {
        print!("{}", human_report(&root, &report));
    }
    Ok(())
}

fn verify(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(&flags, &["root"])?;
    let root = PathBuf::from(required(&flags, "root")?);
    let report = verify_corpus(&root)?;
    println!(
        "ok: {} games, {} decisions, {} shards",
        report.stats.games,
        report.stats.decisions,
        report.manifest.shards.len()
    );
    Ok(())
}

fn lock(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(&flags, &["root", "hold-ms"])?;
    let root = PathBuf::from(required(&flags, "root")?);
    let hold_ms = match flags.get("hold-ms") {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| CliError::usage(format!("invalid --hold-ms {value:?}")))?,
        None => 0,
    };
    let _lock = CorpusLock::try_acquire(&root)?;
    println!("locked");
    std::io::stdout()
        .flush()
        .map_err(|error| CliError::runtime(format!("could not flush stdout: {error}")))?;
    if hold_ms > 0 {
        std::thread::sleep(Duration::from_millis(hold_ms));
    }
    Ok(())
}

fn human_report(root: &Path, report: &CorpusReport) -> String {
    let manifest = &report.manifest;
    let stats = &report.stats;
    let mut out = String::new();
    let _ = writeln!(out, "root: {}", root.display());
    let _ = writeln!(out, "source_revision: {}", manifest.source_revision);
    let _ = writeln!(out, "encoding_version: {}", manifest.encoding_version);
    let _ = writeln!(out, "split_algorithm: {}", manifest.split_algorithm);
    let _ = writeln!(out, "seed_start: {}", manifest.seed_start);
    let _ = writeln!(out, "shard_games: {}", manifest.shard_games);
    let _ = writeln!(out, "decision_limit: {}", manifest.decision_limit);
    let _ = writeln!(out, "shards: {}", manifest.shards.len());
    let _ = writeln!(out, "games: {}", stats.games);
    let _ = writeln!(out, "decisions: {}", stats.decisions);
    let _ = writeln!(
        out,
        "candidates: total {}, min {}, max {}, forced {}, multi {}",
        stats.candidates,
        stats.min_candidates,
        stats.max_candidates,
        stats.forced_decisions,
        stats.multi_candidate_decisions
    );
    let _ = writeln!(
        out,
        "storage: shards {} bytes, manifest {} bytes, total {} bytes",
        report.shard_bytes, report.manifest_bytes, report.total_storage_bytes
    );
    let _ = writeln!(
        out,
        "end_reason: three_winners {}, wall_exhausted {}",
        stats.three_winners_games, stats.wall_exhausted_games
    );
    out.push_str("split games/decisions:\n");
    for split in &stats.split {
        let _ = writeln!(
            out,
            "  {}: {} / {}",
            split.split, split.games, split.decisions
        );
    }
    out.push_str("stage decisions/multi_candidate:\n");
    for stage in &stats.stage {
        let _ = writeln!(
            out,
            "  {} ({}): {} / {}",
            stage.stage, stage.stage_id, stage.decisions, stage.multi_candidate
        );
    }
    out.push_str("kind opportunities/selections (all) and primary:\n");
    for kind in &stats.kind {
        let _ = writeln!(
            out,
            "  {} ({}): {}/{} all, {}/{} primary",
            kind.kind,
            kind.kind_id,
            kind.opportunities,
            kind.selections,
            kind.primary_opportunities,
            kind.primary_selections
        );
    }
    let zero: Vec<_> = stats
        .zero_support_kinds()
        .map(|kind| kind.kind.as_str())
        .collect();
    let primary_zero: Vec<_> = stats
        .primary_zero_support_kinds()
        .map(|kind| kind.kind.as_str())
        .collect();
    let _ = writeln!(
        out,
        "zero_support_all: {}",
        if zero.is_empty() {
            "none".to_owned()
        } else {
            zero.join(",")
        }
    );
    let _ = writeln!(
        out,
        "zero_support_primary: {}",
        if primary_zero.is_empty() {
            "none".to_owned()
        } else {
            primary_zero.join(",")
        }
    );
    out
}

fn parse_flags(args: &[String]) -> Result<BTreeMap<String, String>, CliError> {
    let mut flags = BTreeMap::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let Some(body) = argument.strip_prefix("--") else {
            return Err(CliError::usage(format!(
                "unexpected positional argument {argument:?}"
            )));
        };
        if body.is_empty() {
            return Err(CliError::usage("empty option name"));
        }
        let (name, inline) = match body.split_once('=') {
            Some((name, value)) => (name, Some(value.to_owned())),
            None => (body, None),
        };
        let value = match inline {
            Some(value) => {
                if value.is_empty() {
                    return Err(CliError::usage(format!("option --{name} requires a value")));
                }
                value
            }
            None => {
                if name == "json" {
                    String::new()
                } else {
                    index += 1;
                    let Some(value) = args.get(index) else {
                        return Err(CliError::usage(format!("option --{name} requires a value")));
                    };
                    value.clone()
                }
            }
        };
        if flags.insert(name.to_owned(), value).is_some() {
            return Err(CliError::usage(format!("option --{name} was repeated")));
        }
        index += 1;
    }
    Ok(flags)
}

fn ensure_keys(flags: &BTreeMap<String, String>, allowed: &[&str]) -> Result<(), CliError> {
    for key in flags.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(CliError::usage(format!("unknown option --{key}")));
        }
    }
    Ok(())
}

fn required<'a>(flags: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, CliError> {
    flags
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| CliError::usage(format!("missing required --{key}")))
}

fn positive_u64(value: &str, name: &str) -> Result<u64, CliError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| CliError::usage(format!("invalid --{name} {value:?}: expected u64")))?;
    if parsed == 0 {
        return Err(CliError::usage(format!("--{name} must be positive")));
    }
    Ok(parsed)
}

fn parse_usize(value: &str, name: &str) -> Result<usize, CliError> {
    value
        .parse::<usize>()
        .map_err(|_| CliError::usage(format!("invalid --{name} {value:?}: expected usize")))
}

fn parse_seed(value: &str) -> Result<sichuan_mahjong_neural::collection::Seed, CliError> {
    use sichuan_mahjong_neural::collection::Seed;
    if let Some(hex) = value.strip_prefix("0x") {
        let parsed = u64::from_str_radix(hex, 16)
            .map_err(|_| CliError::usage(format!("invalid seed {value:?}")))?;
        Ok(Seed::from_u64(parsed))
    } else {
        let parsed = value
            .parse::<u64>()
            .map_err(|_| CliError::usage(format!("invalid seed {value:?}")))?;
        Ok(Seed::from_u64(parsed))
    }
}
