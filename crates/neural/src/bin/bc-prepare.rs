//! `bc-prepare` command-line interface.
//!
//! `prepare` captures one committed corpus manifest, strictly streams its
//! listed shards, writes a bounded prepared cache to a new directory, and
//! publishes it atomically. It does not require clean compile-time Git
//! provenance: preparation build facts are recorded truthfully in the cache
//! instead of being presented as a collection attestation. `verify` checks the
//! prepared manifest, every array checksum, packed metadata/offset alignment,
//! derived counts, and global episode-seed coverage without reading the source
//! corpus.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use sichuan_mahjong_neural::collection::{CollectionError, CorpusSnapshot};
use sichuan_mahjong_neural::prepare::{
    prepare, verify_dataset, PrepareError, PrepareOptions, DATASET_FILE,
};

const USAGE: &str = "Usage:
  bc-prepare prepare --corpus DIR --out DIR [options]
  bc-prepare verify --cache DIR
  bc-prepare --help

prepare options:
  --shard-decisions N   Max buffered decisions per output shard (default 20000).
  --shard-candidates N  Max buffered packed candidates per output shard
                        (default 500000).
  --report PATH         Write the prepared dataset manifest JSON to a new file.
                        PATH must be outside the cache root, with relative \"..\"
                        components resolved, and must not already exist.
  --out must not exist. Preparation publishes only after all captured shards
  and the complete manifest validate. A dirty preparation build is allowed and
  recorded; collection provenance comes only from the captured corpus manifest.

verify reads dataset.json, every listed array checksum, and packed metadata
alignment without opening the source corpus.
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bc-prepare: {}", error.message);
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

impl From<PrepareError> for CliError {
    fn from(error: PrepareError) -> Self {
        Self::runtime(error.to_string())
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
        "prepare" => prepare_command(rest),
        "verify" => verify(rest),
        other => Err(CliError::usage(format!(
            "unknown command {other:?}\n\n{USAGE}"
        ))),
    }
}

fn prepare_command(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(
        &flags,
        &[
            "corpus",
            "out",
            "shard-decisions",
            "shard-candidates",
            "report",
        ],
    )?;
    let corpus = PathBuf::from(required(&flags, "corpus")?);
    let out = PathBuf::from(required(&flags, "out")?);
    let report = flags.get("report").map(PathBuf::from);
    let out_normalized = normalized_absolute(&out)?;
    if let Some(path) = &report {
        let report_normalized = normalized_absolute(path)?;
        if report_normalized.starts_with(&out_normalized) {
            return Err(CliError::usage(format!(
                "--report {} must be outside the cache root {}",
                path.display(),
                out.display()
            )));
        }
        validate_report_target(&report_normalized)?;
    }
    let options = PrepareOptions {
        max_decisions_per_shard: match flags.get("shard-decisions") {
            Some(value) => positive_usize(value, "shard-decisions")?,
            None => PrepareOptions::default().max_decisions_per_shard,
        },
        max_candidates_per_shard: match flags.get("shard-candidates") {
            Some(value) => positive_usize(value, "shard-candidates")?,
            None => PrepareOptions::default().max_candidates_per_shard,
        },
    };
    let snapshot = CorpusSnapshot::capture(&corpus)?;
    let manifest = prepare(&snapshot, &out, &options)?;
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|error| CliError::runtime(format!("could not encode dataset JSON: {error}")))?;
    if let Some(path) = &report {
        if let Err(error) = write_report(path, &json) {
            return Err(CliError::runtime(format!(
                "cache {} was published successfully (cache_id {}), but writing --report {} failed: {}",
                out.display(),
                manifest.cache_id,
                path.display(),
                error.message
            )));
        }
    }
    println!("{json}");
    Ok(())
}

fn verify(args: &[String]) -> Result<(), CliError> {
    let flags = parse_flags(args)?;
    ensure_keys(&flags, &["cache"])?;
    let cache = PathBuf::from(required(&flags, "cache")?);
    let manifest = verify_dataset(&cache)?;
    println!(
        "ok: {} ({} episodes, {} decisions, {} shards, {})",
        manifest.cache_id,
        manifest.counts.episodes,
        manifest.counts.decisions,
        manifest.total_shards(),
        cache.join(DATASET_FILE).display()
    );
    Ok(())
}

fn write_report(path: &Path, json: &str) -> Result<(), CliError> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{json}"))
        .map_err(|error| CliError::runtime(format!("could not write {}: {error}", path.display())))
}

fn normalized_absolute(path: &Path) -> Result<PathBuf, CliError> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| {
                CliError::runtime(format!("could not resolve current directory: {error}"))
            })?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

fn validate_report_target(path: &Path) -> Result<(), CliError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.is_dir() {
                return Err(CliError::usage(format!(
                    "--report {} is a directory",
                    path.display()
                )));
            }
            return Err(CliError::usage(format!(
                "--report {} already exists; refusing to overwrite",
                path.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(CliError::runtime(format!(
                "could not inspect --report {}: {error}",
                path.display()
            )))
        }
    }
    let parent = path.parent().ok_or_else(|| {
        CliError::usage(format!(
            "--report {} has no parent directory",
            path.display()
        ))
    })?;
    match fs::metadata(parent) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(CliError::usage(format!(
            "--report parent {} is not a directory",
            parent.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(CliError::usage(
            format!("--report parent {} does not exist", parent.display()),
        )),
        Err(error) => Err(CliError::runtime(format!(
            "could not inspect --report parent {}: {error}",
            parent.display()
        ))),
    }
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
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err(CliError::usage(format!("option --{name} requires a value")));
                };
                value.clone()
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

fn positive_usize(value: &str, name: &str) -> Result<usize, CliError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| CliError::usage(format!("invalid --{name} {value:?}: expected usize")))?;
    if parsed == 0 {
        return Err(CliError::usage(format!("--{name} must be positive")));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_strict_and_values_positive() {
        let flags =
            parse_flags(&["--corpus=a".to_owned(), "--out".to_owned(), "b".to_owned()]).unwrap();
        assert_eq!(required(&flags, "corpus").unwrap(), "a");
        assert_eq!(required(&flags, "out").unwrap(), "b");
        assert!(positive_usize("0", "n").is_err());
        assert!(parse_flags(&["positional".to_owned()]).is_err());
        assert!(ensure_keys(&flags, &["corpus"]).is_err());
    }

    #[test]
    fn report_never_overwrites() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("report.json");
        write_report(&path, "{}").unwrap();
        assert!(write_report(&path, "{}").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"{}\n");
    }
}
