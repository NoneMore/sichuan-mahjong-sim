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

#[path = "heuristic_bench/cli.rs"]
mod cli;
#[path = "heuristic_bench/phases.rs"]
mod phases;
#[path = "heuristic_bench/report.rs"]
mod report;

use cache_tool::{CacheSnapshot, SharedCache};
use sichuan_mahjong_sim::policy::{CacheConfigError, Policy};
use sichuan_mahjong_sim::runner::{BatchReport, Runner, RunnerError};

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

fn main() -> ExitCode {
    match cli::parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{}", cli::usage());
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
            eprint!("{}", cli::usage());
            ExitCode::from(2)
        }
    }
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
    let phases = phases::measure(config, runner_factory)?;
    for warning in &phases.warnings {
        eprintln!("warning: {warning}");
    }
    let report = report::render_report(config, &phases);
    out.write_all(report.as_bytes())
        .map_err(BenchError::Output)?;
    out.flush().map_err(BenchError::Output)
}

/// Builds a runner whose four seats share the supplied cache context.
fn heuristic_runner(cache: &SharedCache) -> Runner {
    Runner::new(
        cache
            .group()
            .map(|policy| Box::new(policy) as Box<dyn Policy>),
    )
}

#[cfg(test)]
#[path = "heuristic_bench/tests.rs"]
mod tests;
