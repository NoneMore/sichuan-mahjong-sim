//! End-to-end four-seat `HeuristicPolicy` self-play benchmark with explicit
//! cache lifecycles.
//!
//! The tool drives the existing [`Runner::run_batch`] so game rules, settlement,
//! and the measured timing boundary stay owned by the production runner. It
//! only validates a small workload, plays it, reports cache status, and formats
//! one plain-text report. Nothing here changes `Game`, `Policy`, or `Runner`.
//!
//! Measurement protocol `heuristic-benchmark-v2`:
//!
//! - Measured game `i` uses seed `S + i` (mod `2^64`).
//! - Warmup game `j` uses the `W` seeds immediately before `S` (mod `2^64`), so
//!   the two ranges are adjacent on the seed ring and must not overlap:
//!   `N + W <= 2^64` as mathematical integers. Changing `W` never moves the
//!   measured seed sequence.
//! - All four seats share one `HeuristicCache` context. The context is bound
//!   once per lifecycle phase and cloned into the seat policies; the tool keeps
//!   a handle to report status.
//! - `cold` (default): the warmup phase, when requested, owns a separate
//!   context that is dropped before the measured phase starts. Warmup therefore
//!   warms process-level state (code, allocator, CPU caches) but the measured
//!   phase begins with an empty cache. A cold run with `W = 0` in a fresh
//!   process is a full process-and-cache cold start; lazy table initialization
//!   happens inside the measured batch and is included in `total_time`.
//! - `warm`: the warmup phase and the measured phase share one runner, four
//!   seat policies, and one context. The measured phase explicitly inherits the
//!   warmed context. `W` must be at least 1. Lazy initialization charged to the
//!   warmup phase is outside the measured `total_time`.
//! - Only the measured batch's timings and results are reported as measured
//!   performance. Warmup counts and timings are reported separately so the
//!   initialization boundary stays visible.
//!
//! `cache.status()` snapshots are taken immediately before and after each
//! measured batch and are not part of the timed region. The library never
//! prints; this tool warns once on stderr for each context whose allocation
//! failed and still reports the completed batch for diagnosis.

use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::process::ExitCode;

#[path = "support/cache_tool.rs"]
mod cache_tool;

use cache_tool::{snapshot_lines, CacheSnapshot, CacheWarnings, SharedCache};
use sichuan_mahjong_sim::policy::{CacheConfigError, Policy, DEFAULT_BUDGET_BYTES};
use sichuan_mahjong_sim::runner::{BatchReport, Runner, RunnerError, DEFAULT_DECISION_LIMIT};
use sichuan_mahjong_sim::PLAYER_COUNT;

/// Workload protocol identifier reported with every measurement.
const PROTOCOL: &str = "heuristic-benchmark-v2";

/// Policy used in all four seats.
const POLICY_NAME: &str = "heuristic";

/// Number of seeds on the `2^64` seed ring.
const SEED_SPACE: u128 = 1u128 << 64;

fn usage() -> String {
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

/// Cache lifecycle selected on the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Warmup, when present, uses a separate context and the measured cache is
    /// empty at the start of the measured batch.
    Cold,
    /// Warmup and measurement share one runner, policies, and context.
    Warm,
}

impl Lifecycle {
    fn name(self) -> &'static str {
        match self {
            Lifecycle::Cold => "cold",
            Lifecycle::Warm => "warm",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cold" => Ok(Lifecycle::Cold),
            "warm" => Ok(Lifecycle::Warm),
            _ => Err(format!(
                "invalid --lifecycle value '{value}': expected cold or warm"
            )),
        }
    }
}

/// Whether the cache budget came from the tool default or the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BudgetSource {
    CandidateDefault,
    Explicit,
}

impl BudgetSource {
    fn name(self) -> &'static str {
        match self {
            BudgetSource::CandidateDefault => "candidate-default",
            BudgetSource::Explicit => "explicit",
        }
    }
}

/// Validated, explicit workload configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    /// Measured games; always at least 1.
    games: u64,
    /// First measured game seed.
    seed: u64,
    /// Warmup games; `0` disables the warmup phase.
    warmup: u64,
    /// Cache lifecycle.
    lifecycle: Lifecycle,
    /// Cache byte budget; `0` disables caching.
    cache_budget_bytes: usize,
    /// Where the cache budget came from.
    cache_budget_source: BudgetSource,
}

impl Config {
    /// Whether a warmup phase is part of this workload.
    fn warmup_enabled(&self) -> bool {
        self.warmup > 0
    }

    /// First warmup seed: `W` positions before the measured start on the ring.
    fn warmup_start(&self) -> u64 {
        self.seed.wrapping_sub(self.warmup)
    }

    /// Seed of measured game `index`, valid for `index < games`.
    #[cfg(test)]
    fn measure_seed(&self, index: u64) -> u64 {
        self.seed.wrapping_add(index)
    }

    /// Seed of warmup game `index`, valid for `index < warmup`.
    #[cfg(test)]
    fn warmup_seed(&self, index: u64) -> u64 {
        self.warmup_start().wrapping_add(index)
    }
}

/// A parsed command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Help,
    Run(Config),
}

/// Which phase a run error came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Warmup,
    Measure,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Warmup => "warmup phase",
            Stage::Measure => "measured phase",
        }
    }
}

/// A failed benchmark run.
#[derive(Debug)]
enum BenchError {
    /// The requested cache context could not be configured.
    CacheConfig(CacheConfigError),
    /// A phase rejected the run; the measurement is invalid.
    Phase { stage: Stage, source: RunnerError },
    /// The report could not be written or flushed.
    Output(io::Error),
}

impl BenchError {
    fn phase(stage: Stage, source: RunnerError) -> Self {
        BenchError::Phase { stage, source }
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::CacheConfig(source) => write!(f, "invalid cache budget: {source}"),
            BenchError::Phase { stage, source } => {
                write!(f, "{} failed: {source}", stage.name())
            }
            BenchError::Output(error) => write!(f, "report output failed: {error}"),
        }
    }
}

impl Error for BenchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BenchError::CacheConfig(source) => Some(source),
            BenchError::Phase { source, .. } => Some(source),
            BenchError::Output(error) => Some(error),
        }
    }
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Command::Run(config)) => {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            match execute(&config, heuristic_runner, &mut out) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!();
            eprint!("{}", usage());
            ExitCode::from(2)
        }
    }
}

/// Parses arguments after the executable name.
///
/// `--games` and `--seed` are required; `--lifecycle` defaults to `cold`,
/// `--warmup` to `0`, and the cache budget to the selected library default.
/// `-h` / `--help` short-circuits and ignores later arguments. Input is
/// validated before any game is played.
fn parse_args<I>(args: I) -> Result<Command, String>
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
fn parse_decimal_u64(option: &str, value: &str) -> Result<u64, String> {
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
fn parse_usize(option: &str, value: &str) -> Result<usize, String> {
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
fn validate(
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

/// Builds a runner whose four seats share the supplied cache context.
fn heuristic_runner(cache: &SharedCache) -> Runner {
    Runner::new(
        cache
            .group()
            .map(|policy| Box::new(policy) as Box<dyn Policy>),
    )
}

/// One completed warmup phase and its cache lifecycle.
#[derive(Debug)]
struct WarmupPhase {
    batch: BatchReport,
    cache_before: CacheSnapshot,
    cache_after: CacheSnapshot,
    /// `separate-context` for cold, `inherited-context` for warm.
    scope: &'static str,
}

/// Cache lifecycle of the measured phase and its surrounding snapshots.
#[derive(Debug)]
struct Phases {
    warmup: Option<WarmupPhase>,
    measured: BatchReport,
    measured_cache_before: CacheSnapshot,
    measured_cache_after: CacheSnapshot,
    warnings: Vec<String>,
}

/// Runs the warmup (when enabled) and measured phases.
///
/// The runner factory receives the context that its policies must use. A cold
/// warmup gets a separate context that is dropped before the measured context
/// is created, so no table state survives into the measured batch. A warm
/// warmup and the measured batch reuse the same runner, policies, and context.
/// Phase errors carry the stage that failed and no measured aggregate is
/// returned.
fn measure<F>(config: &Config, mut runner_for: F) -> Result<Phases, BenchError>
where
    F: FnMut(&SharedCache) -> Runner,
{
    let mut warnings = Vec::new();
    let (warmup, measured_cache, mut measured_runner, measured_cache_before, mut warner) = if config
        .lifecycle
        == Lifecycle::Warm
    {
        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let mut runner = runner_for(&cache);
        let batch = runner
            .run_batch(config.warmup_start(), config.warmup)
            .map_err(|error| BenchError::phase(Stage::Warmup, error))?;
        let after = cache.snapshot();
        let mut warner = CacheWarnings::default();
        if let Some(warning) = warner.observe(after, "benchmark warmup") {
            warnings.push(warning);
        }
        let phase = WarmupPhase {
            batch,
            cache_before: before,
            cache_after: after,
            scope: "inherited-context",
        };
        (Some(phase), cache, runner, after, warner)
    } else if config.warmup_enabled() {
        // Cold warmup: a separate context is dropped before measurement.
        let warmup_cache =
            SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let warmup_before = warmup_cache.snapshot();
        let mut warmup_runner = runner_for(&warmup_cache);
        let batch = warmup_runner
            .run_batch(config.warmup_start(), config.warmup)
            .map_err(|error| BenchError::phase(Stage::Warmup, error))?;
        let warmup_after = warmup_cache.snapshot();
        let mut warmup_warner = CacheWarnings::default();
        if let Some(warning) = warmup_warner.observe(warmup_after, "benchmark warmup") {
            warnings.push(warning);
        }
        drop(warmup_runner);
        drop(warmup_cache);

        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let runner = runner_for(&cache);
        let phase = WarmupPhase {
            batch,
            cache_before: warmup_before,
            cache_after: warmup_after,
            scope: "separate-context",
        };
        (Some(phase), cache, runner, before, CacheWarnings::default())
    } else {
        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let runner = runner_for(&cache);
        (None, cache, runner, before, CacheWarnings::default())
    };

    let measured = measured_runner
        .run_batch(config.seed, config.games)
        .map_err(|error| BenchError::phase(Stage::Measure, error))?;
    let measured_cache_after = measured_cache.snapshot();
    if let Some(warning) = warner.observe(measured_cache_after, "benchmark measured") {
        warnings.push(warning);
    }
    drop(measured_runner);
    drop(measured_cache);

    Ok(Phases {
        warmup,
        measured,
        measured_cache_before,
        measured_cache_after,
        warnings,
    })
}

/// Runs the workload and writes the measured report to `out`.
///
/// Nothing is written until the run has succeeded, so a failed run emits no
/// performance summary. Allocation-failure warnings go to stderr exactly once
/// per context; the report still contains the completed batch for diagnosis.
/// A failed write or flush is an output error, not a successful measurement.
fn execute<W, F>(config: &Config, runner_factory: F, out: &mut W) -> Result<(), BenchError>
where
    W: Write,
    F: FnMut(&SharedCache) -> Runner,
{
    let phases = measure(config, runner_factory)?;
    for warning in &phases.warnings {
        eprintln!("warning: {warning}");
    }
    let report = render_report(config, &phases);
    out.write_all(report.as_bytes())
        .map_err(BenchError::Output)?;
    out.flush().map_err(BenchError::Output)
}

/// Rate per measured batch wall second, unavailable when the measured
/// duration is zero. Both games and decisions use this same denominator.
fn rate_per_second(count: u64, batch: &BatchReport) -> Option<f64> {
    let seconds = batch.total_time.as_secs_f64();
    if seconds > 0.0 {
        Some(count as f64 / seconds)
    } else {
        None
    }
}

/// Mean policy time per decision in microseconds over every decision kind, or
/// unavailable when the measured batch recorded no decisions.
fn mean_policy_time_us(batch: &BatchReport) -> Option<f64> {
    if batch.decisions == 0 {
        None
    } else {
        Some(batch.policy_time.as_secs_f64() * 1_000_000.0 / batch.decisions as f64)
    }
}

/// Where the measured context obtained its table allocation.
///
/// `warmup` means the inherited warm context was already ready (or had
/// already failed) when measurement began. `measured` means the measured batch
/// observed the transition from unallocated to ready or failed and therefore
/// includes the lazy initialization attempt. `none` means the context never
/// allocated; `disabled` means no table allocation is possible.
fn cache_init_phase(config: &Config, phases: &Phases) -> &'static str {
    let before = phases.measured_cache_before;
    let after = phases.measured_cache_after;
    if before.is_disabled() && after.is_disabled() {
        return "disabled";
    }
    if before.is_allocated() || before.is_allocation_failed() {
        return if config.lifecycle == Lifecycle::Warm {
            "warmup"
        } else {
            "pre-measured"
        };
    }
    if after.state == "ready" || after.is_allocation_failed() {
        return "measured";
    }
    "none"
}

/// Timing boundary that contains the measured context's initialization.
fn cache_init_timing_boundary(phase: &str) -> &'static str {
    match phase {
        "warmup" => "excluded-from-measured-total-time",
        "measured" => "included-in-measured-total-time",
        _ => "not-applicable",
    }
}

/// Formats one measured batch as stable, single-value "key: value" lines.
///
/// Units belong to field names, rather than natural-language value suffixes.
/// The protocol identifier distinguishes this workload and its timing and cache
/// lifecycle semantics. No prose or extra lines are needed to consume the
/// report.
fn render_report(config: &Config, phases: &Phases) -> String {
    let measured = &phases.measured;
    let mut text = format!(
        "protocol: {PROTOCOL}\n\
         policy: {POLICY_NAME}\n\
         seats: {PLAYER_COUNT}\n\
         seat_order: [0, 1, 2, 3]\n\
         cache_scope: four-seat-shared-context\n\
         cache_lifecycle: {}\n\
         cache_budget_source: {}\n\
         cache_budget_bytes: {}\n\
         cache_warmup_scope: {}\n\
         warmup_games: {}\n",
        config.lifecycle.name(),
        config.cache_budget_source.name(),
        config.cache_budget_bytes,
        phases.warmup.as_ref().map_or("none", |warmup| warmup.scope),
        config.warmup
    );

    match &phases.warmup {
        Some(warmup) => {
            text.push_str(&format!(
                "warmup_first_seed: {}\n\
                 warmup_completed_games: {}\n\
                 warmup_decisions: {}\n\
                 warmup_total_time_seconds: {:.6}\n\
                 warmup_policy_time_seconds: {:.6}\n\
                 warmup_rules_time_seconds: {:.6}\n",
                config.warmup_start(),
                warmup.batch.games,
                warmup.batch.decisions,
                warmup.batch.total_time.as_secs_f64(),
                warmup.batch.policy_time.as_secs_f64(),
                warmup.batch.rules_time.as_secs_f64()
            ));
            text.push_str(&snapshot_lines("warmup_cache_before", warmup.cache_before));
            text.push_str(&snapshot_lines("warmup_cache_after", warmup.cache_after));
        }
        None => {
            text.push_str(
                "warmup_first_seed: none\n\
                 warmup_completed_games: none\n\
                 warmup_decisions: none\n\
                 warmup_total_time_seconds: none\n\
                 warmup_policy_time_seconds: none\n\
                 warmup_rules_time_seconds: none\n",
            );
            for prefix in ["warmup_cache_before", "warmup_cache_after"] {
                text.push_str(&format!(
                    "{prefix}_state: none\n\
                     {prefix}_budget_bytes: none\n\
                     {prefix}_context_overhead_bytes: none\n\
                     {prefix}_allocated_bytes: none\n\
                     {prefix}_capacity_entries: none\n\
                     {prefix}_occupied_entries: none\n\
                     {prefix}_evictions: none\n\
                     {prefix}_hits: none\n\
                     {prefix}_misses: none\n"
                ));
            }
        }
    }

    text.push_str(&format!(
        "seed: {}\n\
         games: {}\n\
         decision_limit_per_game: {DEFAULT_DECISION_LIMIT}\n",
        config.seed, config.games
    ));
    text.push_str(&snapshot_lines(
        "cache_before_measured",
        phases.measured_cache_before,
    ));
    text.push_str(&format!(
        "completed_games: {}\n\
         decisions: {}\n\
         total_time_seconds: {:.6}\n\
         policy_time_seconds: {:.6}\n\
         rules_time_seconds: {:.6}\n",
        measured.games,
        measured.decisions,
        measured.total_time.as_secs_f64(),
        measured.policy_time.as_secs_f64(),
        measured.rules_time.as_secs_f64()
    ));
    text.push_str(&snapshot_lines(
        "cache_after_measured",
        phases.measured_cache_after,
    ));
    text.push_str(&format!(
        "cache_at_capacity_after_measured: {}\n",
        if phases.measured_cache_after.is_at_capacity() {
            "yes"
        } else {
            "no"
        }
    ));
    let init_phase = cache_init_phase(config, phases);
    text.push_str(&format!(
        "cache_init_phase: {init_phase}\n\
         cache_init_timing_boundary: {}\n\
         cache_warning_count: {}\n",
        cache_init_timing_boundary(init_phase),
        phases.warnings.len()
    ));

    match rate_per_second(measured.games, measured) {
        Some(rate) => text.push_str(&format!("games_per_sec: {rate:.3}\n")),
        None => text.push_str("games_per_sec: unavailable\n"),
    }
    match rate_per_second(measured.decisions, measured) {
        Some(rate) => text.push_str(&format!("decisions_per_sec: {rate:.3}\n")),
        None => text.push_str("decisions_per_sec: unavailable\n"),
    }
    match mean_policy_time_us(measured) {
        Some(mean) => text.push_str(&format!("mean_policy_us_per_decision: {mean:.3}\n")),
        None => text.push_str("mean_policy_us_per_decision: unavailable\n"),
    }

    text.push_str(&format!(
        "three_winners: {}\n\
         wall_exhausted: {}\n\
         score_totals: {:?}\n\
         score_sum: {}\n\
         winners: {:?}\n",
        measured.three_winners,
        measured.wall_exhausted,
        measured.score_totals,
        measured.score_sum(),
        measured.winners
    ));
    text
}

#[cfg(test)]
#[path = "heuristic_bench/tests.rs"]
mod tests;
