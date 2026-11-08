//! Command line for the balanced A/B comparison.
//!
//! Parsing only validates inputs: no game is played and no artifact is written
//! here. `-h` / `--help` short-circuits, so later arguments are never inspected.

use std::collections::HashSet;
use std::path::PathBuf;

use super::{CacheBudget, Command, Options, Plan, Style};
use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;

pub(super) fn usage() -> String {
    let candidate = DEFAULT_BUDGET_BYTES;
    format!(
        "\
Usage: policy_ab --seeds LIST --policy-seed S --decision-limit N --out DIR \
                 [--a STYLE] [--b STYLE] [--cache-budget-bytes BYTES]

Runs balanced 2A2B games over an explicit seed list and writes reproducible
artifacts to a fresh directory. A defaults to FirstLegalPolicy and B to
RandomPolicy.

Every heuristic seat in the run shares one explicit cache context across all
games; non-heuristic seats never touch it. The context is created before play,
so an invalid budget fails before the output directory is created. An
allocation failure is warned about once on stderr and the run continues. The
library itself prints no cache diagnostics.

Options:
  --seeds LIST       Comma-separated, nonempty, distinct u64 game seeds.
  --policy-seed S    u64 seed for the per-seat random policy streams.
  --decision-limit N Positive per-game decision limit.
  --out DIR          Output directory; it must not already exist.
  --a STYLE          Policy in the two A seats: first, random, heuristic, or
                     missing-only. Default first.
  --b STYLE          Policy in the two remaining B seats: first, random,
                     heuristic, or missing-only. Default random.
  --cache-budget-bytes BYTES
                     Heuristic cache byte budget; 0 disables caching and the
                     default is 26.5 MiB ({candidate}), selected for batch reuse.
                     Requires at least one heuristic seat.
  -h, --help         Print this help and exit; arguments after it are ignored.
"
    )
}

/// Parses one policy style name from the command line.
///
/// Like the main CLI's `--policy`, the name is matched case-insensitively; the
/// recorded name is always the canonical lowercase form.
pub(super) fn parse_style(value: &str) -> Result<Style, String> {
    match value.to_ascii_lowercase().as_str() {
        "first" => Ok(Style::First),
        "random" => Ok(Style::Random),
        "heuristic" => Ok(Style::Heuristic),
        "missing-only" => Ok(Style::MissingOnly),
        _ => Err(format!(
            "invalid policy style '{value}': expected first, random, heuristic, or missing-only"
        )),
    }
}

/// Parses arguments after the executable name.
///
/// The four run inputs are required. The A/B styles are optional and default to
/// the historical `first`-versus-`random` comparison. `-h` / `--help`
/// short-circuits and ignores later arguments, matching the main CLI.
pub(super) fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut seeds = None;
    let mut policy_seed = None;
    let mut decision_limit = None;
    let mut out = None;
    let mut style_a = None;
    let mut style_b = None;
    let mut cache_budget = None;
    let mut arguments = args.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--seeds" => {
                if seeds.is_some() {
                    return Err("duplicate --seeds".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--seeds requires a value".to_owned())?;
                seeds = Some(parse_seeds(&value)?);
            }
            "--policy-seed" => {
                if policy_seed.is_some() {
                    return Err("duplicate --policy-seed".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--policy-seed requires a value".to_owned())?;
                policy_seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| format!("invalid --policy-seed value '{value}'"))?,
                );
            }
            "--decision-limit" => {
                if decision_limit.is_some() {
                    return Err("duplicate --decision-limit".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--decision-limit requires a value".to_owned())?;
                let parsed = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --decision-limit value '{value}'"))?;
                if parsed == 0 {
                    return Err("--decision-limit must be at least 1".to_owned());
                }
                decision_limit = Some(parsed);
            }
            "--out" => {
                if out.is_some() {
                    return Err("duplicate --out".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--out requires a value".to_owned())?;
                out = Some(PathBuf::from(value));
            }
            "--a" => {
                if style_a.is_some() {
                    return Err("duplicate --a".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--a requires a value".to_owned())?;
                style_a = Some(parse_style(&value)?);
            }
            "--b" => {
                if style_b.is_some() {
                    return Err("duplicate --b".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--b requires a value".to_owned())?;
                style_b = Some(parse_style(&value)?);
            }
            "--cache-budget-bytes" => {
                if cache_budget.is_some() {
                    return Err("duplicate --cache-budget-bytes".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--cache-budget-bytes requires a value".to_owned())?;
                cache_budget = Some(parse_usize("--cache-budget-bytes", &value)?);
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    let style_a = style_a.unwrap_or(Style::First);
    let style_b = style_b.unwrap_or(Style::Random);
    if cache_budget.is_some() && style_a != Style::Heuristic && style_b != Style::Heuristic {
        return Err("--cache-budget-bytes requires at least one heuristic seat".to_owned());
    }
    Ok(Command::Run(Options {
        plan: Plan {
            seeds: seeds.ok_or_else(|| "--seeds is required".to_owned())?,
            policy_seed: policy_seed.ok_or_else(|| "--policy-seed is required".to_owned())?,
            decision_limit: decision_limit
                .ok_or_else(|| "--decision-limit is required".to_owned())?,
            style_a,
            style_b,
            cache_budget: match cache_budget {
                Some(bytes) => CacheBudget::Explicit(bytes),
                None => CacheBudget::CandidateDefault,
            },
        },
        out: out.ok_or_else(|| "--out is required".to_owned())?,
    }))
}

/// Parses a decimal byte count that must fit this platform's `usize`.
pub(super) fn parse_usize(option: &str, value: &str) -> Result<usize, String> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "invalid {option} value '{value}': expected a decimal unsigned integer"
        ));
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("invalid {option} value '{value}': out of range"))?;
    usize::try_from(parsed).map_err(|_| {
        format!("invalid {option} value '{value}': out of range for this platform's usize")
    })
}

/// Parses and validates a comma-separated seed list.
///
/// The returned order is the input order. Only membership is hashed, so a long
/// seed list is validated in linear rather than quadratic time.
pub(super) fn parse_seeds(value: &str) -> Result<Vec<u64>, String> {
    if value.trim().is_empty() {
        return Err("--seeds requires at least one game seed".to_owned());
    }
    let mut seeds = Vec::new();
    let mut seen = HashSet::new();
    for field in value.split(',') {
        let seed = field.trim().parse::<u64>().map_err(|_| {
            format!("invalid game seed '{field}': expected an unsigned 64-bit integer")
        })?;
        if !seen.insert(seed) {
            return Err(format!("duplicate game seed {seed}"));
        }
        seeds.push(seed);
    }
    Ok(seeds)
}
