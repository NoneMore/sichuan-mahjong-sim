//! Command line and workload validation for the heuristic benchmark.
//!
//! Validation is arithmetic only: the requested games are never played here, so
//! an invalid workload fails before any cache context is created.

use super::{BudgetSource, Command, Config, Lifecycle};
use sichuan_mahjong_sim::policy::DEFAULT_BUDGET_BYTES;

/// Number of seeds on the `2^64` seed ring.
const SEED_SPACE: u128 = 1u128 << 64;

pub(super) fn usage() -> String {
    let candidate = DEFAULT_BUDGET_BYTES;
    let second_tier = candidate * 2;
    format!(
        "\
Usage: heuristic_bench --games N --seed S [OPTIONS]

Runs one optional warmup phase and one measured phase of four-seat
HeuristicPolicy self-play through the runner's batch interface. All four seats
share one heuristic cache context; the report includes its budget, actual table
allocation, state, sharing scope, and initialization timing boundary.

Measured game i uses seed S + i (mod 2^64). Warmup game j uses the W seeds
immediately before S (mod 2^64), so the measured seed sequence does not depend
on W and the two ranges must not overlap: N + W must be at most 2^64.

Options:
  --games N                Measured games (N must be at least 1).
  --seed S                 First measured game seed.
  --lifecycle MODE         cold (default) or warm. A cold warmup uses a
                           separate context that is dropped before measurement;
                           a warm measured phase inherits the warmup context.
  --warmup W               Warmup games played before the measured phase
                           (default 0). warm requires W >= 1; cold allows 0.
  --cache-budget-bytes B   Cache byte budget (0 disables caching; default
                           {candidate}, the selected 26.5 MiB default).
                           The 53 MiB tier is {second_tier}.
  -h, --help               Print this help and exit; arguments after it are
                           ignored.

Only the measured batch's timings and results are reported as measured
performance. Cold lazy allocation is included in the measured total_time; warm
initialization happens during warmup and is reported separately. A cold run with
--warmup 0 in a fresh process is a full process-and-cache cold start; startup
inclusive process wall time must be observed externally, not mixed with batch
total_time. Failed or degraded runs are reported; no performance conclusion is
made by this tool.
"
    )
}

/// Parses arguments after the executable name.
///
/// `--games` and `--seed` are required; `--lifecycle` defaults to `cold`,
/// `--warmup` to `0`, and the cache budget to the selected library default.
/// `-h` / `--help` short-circuits and ignores later arguments. Input is
/// validated before any game is played.
pub(super) fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut games = None;
    let mut seed = None;
    let mut lifecycle = None;
    let mut warmup = None;
    let mut cache_budget = None;
    let mut arguments = args.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--games" => {
                if games.is_some() {
                    return Err("duplicate --games".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--games requires a value".to_owned())?;
                games = Some(parse_decimal_u64("--games", &value)?);
            }
            "--seed" => {
                if seed.is_some() {
                    return Err("duplicate --seed".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_owned())?;
                seed = Some(parse_decimal_u64("--seed", &value)?);
            }
            "--lifecycle" => {
                if lifecycle.is_some() {
                    return Err("duplicate --lifecycle".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--lifecycle requires a value".to_owned())?;
                lifecycle = Some(Lifecycle::parse(&value)?);
            }
            "--warmup" => {
                if warmup.is_some() {
                    return Err("duplicate --warmup".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--warmup requires a value".to_owned())?;
                warmup = Some(parse_decimal_u64("--warmup", &value)?);
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

    let games = games.ok_or_else(|| "--games is required".to_owned())?;
    let seed = seed.ok_or_else(|| "--seed is required".to_owned())?;
    let (cache_budget_bytes, cache_budget_source) = match cache_budget {
        Some(bytes) => (bytes, BudgetSource::Explicit),
        None => (DEFAULT_BUDGET_BYTES, BudgetSource::CandidateDefault),
    };
    Ok(Command::Run(validate(
        games,
        seed,
        lifecycle.unwrap_or(Lifecycle::Cold),
        warmup.unwrap_or(0),
        cache_budget_bytes,
        cache_budget_source,
    )?))
}

/// Parses one strictly decimal unsigned 64-bit value.
///
/// Only ASCII digits are accepted: signs, whitespace, underscores, and radix
/// prefixes are rejected instead of being normalized. An all-digit value that
/// does not fit `u64` is reported as out of range.
pub(super) fn parse_decimal_u64(option: &str, value: &str) -> Result<u64, String> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "invalid {option} value '{value}': expected a decimal unsigned 64-bit integer"
        ));
    }
    value.parse::<u64>().map_err(|_| {
        format!("invalid {option} value '{value}': out of range for an unsigned 64-bit integer")
    })
}

/// Parses a decimal byte count that must fit this platform's `usize`.
pub(super) fn parse_usize(option: &str, value: &str) -> Result<usize, String> {
    let parsed = parse_decimal_u64(option, value)?;
    usize::try_from(parsed).map_err(|_| {
        format!("invalid {option} value '{value}': out of range for this platform's usize")
    })
}

/// Validates the workload and builds the seed plan.
///
/// The overlap check uses `u128` so it cannot be defeated by `u64` addition
/// overflow: the warmup and measured ranges must fit in the `2^64` seed ring
/// without overlapping; `N + W == 2^64` is accepted as the boundary. The check
/// is arithmetic only and never plays the requested games. A warm lifecycle
/// needs at least one warmup game to inherit a warmed context.
pub(super) fn validate(
    games: u64,
    seed: u64,
    lifecycle: Lifecycle,
    warmup: u64,
    cache_budget_bytes: usize,
    cache_budget_source: BudgetSource,
) -> Result<Config, String> {
    if games == 0 {
        return Err("--games must be at least 1".to_owned());
    }
    if lifecycle == Lifecycle::Warm && warmup == 0 {
        return Err("--lifecycle warm requires --warmup of at least 1".to_owned());
    }
    let span = u128::from(games) + u128::from(warmup);
    if span > SEED_SPACE {
        return Err(format!(
            "--games {games} plus --warmup {warmup} exceeds the 2^64 seed space; \
             the warmup and measured seed ranges would overlap"
        ));
    }
    Ok(Config {
        games,
        seed,
        warmup,
        lifecycle,
        cache_budget_bytes,
        cache_budget_source,
    })
}
