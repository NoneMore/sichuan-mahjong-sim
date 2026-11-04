//! End-to-end four-seat `HeuristicPolicy` self-play benchmark.
//!
//! The tool runs one optional unmeasured warmup batch and exactly one measured
//! batch, both through the existing [`Runner::run_batch`], so game rules,
//! settlement, and timing boundaries stay owned by the production runner. It
//! only parses and validates a small explicit workload, plays it, and formats
//! one plain-text report. Nothing here changes `Game`, `Policy`, or `Runner`.
//!
//! Measurement protocol `heuristic-benchmark-v1`:
//!
//! - Measured game `i` uses seed `S + i` (mod `2^64`).
//! - Warmup game `j` uses the `W` seeds immediately before `S` (mod `2^64`),
//!   so the two ranges are adjacent on the seed ring and must not overlap:
//!   `N + W <= 2^64` as mathematical integers. Changing `W` never moves the
//!   measured seed sequence.
//! - The warmup phase builds its own runner and seat policies and is dropped
//!   with its report. The measured phase builds a new runner and seat policies
//!   and reuses those instances for its whole batch. `W = 0` skips the warmup
//!   batch entirely instead of calling `run_batch(_, 0)`.
//! - Only the measured batch is reported; `total_time`, `policy_time`, and
//!   `rules_time` come straight from its [`BatchReport`].
//!
//! Warmup separates instances but does not clear process-level state such as
//! allocator or CPU cache, and it is not a cold-cache protocol.

use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::process::ExitCode;

use sichuan_mahjong_sim::policy::{HeuristicPolicy, Policy};
use sichuan_mahjong_sim::runner::{BatchReport, Runner, RunnerError, DEFAULT_DECISION_LIMIT};
use sichuan_mahjong_sim::PLAYER_COUNT;

/// Workload protocol identifier reported with every measurement.
const PROTOCOL: &str = "heuristic-benchmark-v1";

/// Policy used in all four seats.
const POLICY_NAME: &str = "heuristic";

/// Number of seeds on the `2^64` seed ring.
const SEED_SPACE: u128 = 1u128 << 64;

const USAGE: &str = "\
Usage: heuristic_bench --games N --seed S --warmup W

Runs one optional warmup phase and one measured phase of four-seat
HeuristicPolicy self-play, both driven by the runner's batch interface. All
three options are required and take decimal unsigned 64-bit integers; there is
no default workload.

Measured game i uses seed S + i (mod 2^64). Warmup game j uses the W seeds
immediately before S (mod 2^64), so the measured seed sequence does not depend
on W and the two ranges must not overlap: N + W must be at most 2^64.

Options:
  --games N     Measured games (N must be at least 1).
  --seed S      First measured game seed.
  --warmup W    Warmup games played before the measured phase (0 disables the
                warmup phase).
  -h, --help    Print this help and exit; arguments after it are ignored.

Only the measured batch's timings and results are reported. A failed warmup
or measured batch invalidates the run; no performance summary is produced.
";

/// Validated, explicit workload configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    /// Measured games; always at least 1.
    games: u64,
    /// First measured game seed.
    seed: u64,
    /// Warmup games; `0` disables the warmup phase.
    warmup: u64,
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
    fn measure_seed(&self, index: u64) -> u64 {
        self.seed.wrapping_add(index)
    }

    /// Seed of warmup game `index`, valid for `index < warmup`.
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
            BenchError::Phase { source, .. } => Some(source),
            BenchError::Output(error) => Some(error),
        }
    }
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
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
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Parses arguments after the executable name.
///
/// All three workload options are required, may appear once each, and take one
/// strictly decimal `u64` value. `-h` / `--help` short-circuits and ignores
/// later arguments; workload validation is only performed for a run command.
/// Input is validated before any game is played.
fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut games = None;
    let mut seed = None;
    let mut warmup = None;
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
            "--warmup" => {
                if warmup.is_some() {
                    return Err("duplicate --warmup".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--warmup requires a value".to_owned())?;
                warmup = Some(parse_decimal_u64("--warmup", &value)?);
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    let games = games.ok_or_else(|| "--games is required".to_owned())?;
    let seed = seed.ok_or_else(|| "--seed is required".to_owned())?;
    let warmup = warmup.ok_or_else(|| "--warmup is required".to_owned())?;
    Ok(Command::Run(validate(games, seed, warmup)?))
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

/// Validates the workload and builds the seed plan.
///
/// The overlap check uses `u128` so it cannot be defeated by `u64` addition
/// overflow: the warmup and measured ranges must fit in the `2^64` seed ring
/// without overlapping; `N + W == 2^64` is accepted as the boundary. The check
/// is arithmetic only and never plays the requested games.
fn validate(games: u64, seed: u64, warmup: u64) -> Result<Config, String> {
    if games == 0 {
        return Err("--games must be at least 1".to_owned());
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
    })
}

/// Builds a runner with one fresh, independent `HeuristicPolicy` per seat.
///
/// The same runner and policies drive every game of their phase: the batch is
/// not rebuilt per game, so policy streams and instance reuse match a normal
/// batch simulation.
fn heuristic_runner() -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(HeuristicPolicy::new()) as Box<dyn Policy>
    }))
}

/// Runs the warmup (when enabled) and measured phases and returns the measured
/// batch aggregate.
///
/// The runner factory is injected so tests can observe or fail phases with
/// local fixtures; the tool itself always passes [`heuristic_runner`]. Warmup
/// counts, times, scores, and reports are dropped before the measured phase
/// starts, and phase errors carry the stage that failed.
fn measure<F>(config: &Config, mut runner_factory: F) -> Result<BatchReport, BenchError>
where
    F: FnMut() -> Runner,
{
    if config.warmup_enabled() {
        let mut warmup_runner = runner_factory();
        let warmup = warmup_runner.run_batch(config.warmup_start(), config.warmup);
        // The warmup runner, its seat policies, and its aggregate report end
        // here: none of them may reach the measured report or instances.
        drop(warmup_runner);
        warmup.map_err(|error| BenchError::phase(Stage::Warmup, error))?;
    }

    let mut runner = runner_factory();
    runner
        .run_batch(config.seed, config.games)
        .map_err(|error| BenchError::phase(Stage::Measure, error))
}

/// Runs the workload and writes the measured report to `out`.
///
/// Nothing is written until both phases have succeeded, so a failed run emits
/// no performance summary. A failed write or flush is an output error, not a
/// successful measurement.
fn execute<W, F>(config: &Config, runner_factory: F, out: &mut W) -> Result<(), BenchError>
where
    W: Write,
    F: FnMut() -> Runner,
{
    let batch = measure(config, runner_factory)?;
    let report = render_report(config, &batch);
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

/// Formats one measured batch as stable, single-value "key: value" lines.
///
/// Units belong to field names, rather than natural-language value suffixes.
/// The protocol identifier distinguishes this workload and its timing semantics.
/// No prose or extra lines are needed to consume the report.
fn render_report(config: &Config, batch: &BatchReport) -> String {
    let mut text = format!(
        "protocol: {PROTOCOL}\n\
         policy: {POLICY_NAME}\n\
         seats: {PLAYER_COUNT}\n\
         seed: {}\n\
         games: {}\n\
         warmup_games: {}\n",
        config.seed, config.games, config.warmup
    );
    if config.warmup_enabled() {
        text.push_str(&format!("warmup_first_seed: {}\n", config.warmup_start()));
    } else {
        text.push_str("warmup_first_seed: none\n");
    }
    text.push_str(&format!(
        "decision_limit_per_game: {DEFAULT_DECISION_LIMIT}\n\
         decisions: {}\n\
         total_time_seconds: {:.6}\n\
         policy_time_seconds: {:.6}\n\
         rules_time_seconds: {:.6}\n",
        batch.decisions,
        batch.total_time.as_secs_f64(),
        batch.policy_time.as_secs_f64(),
        batch.rules_time.as_secs_f64()
    ));

    match rate_per_second(batch.games, batch) {
        Some(rate) => text.push_str(&format!("games_per_sec: {rate:.3}\n")),
        None => text.push_str("games_per_sec: unavailable\n"),
    }
    match rate_per_second(batch.decisions, batch) {
        Some(rate) => text.push_str(&format!("decisions_per_sec: {rate:.3}\n")),
        None => text.push_str("decisions_per_sec: unavailable\n"),
    }
    match mean_policy_time_us(batch) {
        Some(mean) => text.push_str(&format!("mean_policy_us_per_decision: {mean:.3}\n")),
        None => text.push_str("mean_policy_us_per_decision: unavailable\n"),
    }

    text.push_str(&format!(
        "three_winners: {}\n\
         wall_exhausted: {}\n\
         score_totals: {:?}\n\
         score_sum: {}\n\
         winners: {:?}\n",
        batch.three_winners,
        batch.wall_exhausted,
        batch.score_totals,
        batch.score_sum(),
        batch.winners
    ));
    text
}

#[cfg(test)]
#[path = "heuristic_bench/tests.rs"]
mod tests;
