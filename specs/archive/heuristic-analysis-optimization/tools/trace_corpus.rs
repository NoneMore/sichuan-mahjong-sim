//! Minimal isolated behavioral trace corpus for the
//! heuristic-analysis-optimization Change.
//!
//! The tool drives four-seat [`HeuristicPolicy`] self-play through the existing
//! [`Runner::run_new`] only. It adds no game driving, no rules, and no timing:
//! the production runner owns `next_decision` / `submit`, and this file owns
//! the frozen seed ranges and deterministic text output.
//!
//! One [`Runner`] with one fresh `HeuristicPolicy` per seat is built once and
//! reused for every game of a run, matching the benchmark's instance lifetime.
//!
//! The corpus is frozen here, not passed in: seeds `7..=38` and `10007..=10038`,
//! 32 games each. Each range is written as one `heuristic-trace-v1` section:
//!
//! - a header with the source revision, policy, seat order, and seed range;
//! - one game record per seed with the non-time result fields and a
//!   `game_trace_digest`;
//! - the full canonical action trace, one decision per line;
//! - a corpus summary with the total decision count and a `corpus_digest`.
//!
//! Sections are separated by one blank line. The output is time-free, so the
//! same library and revision reproduce it byte for byte.
//!
//! Canonical trace lines are `<decision_index> <seat_index> <action>` with
//! single spaces; [`canonical_action`] defines the action encoding.
//! `game_trace_digest` is FNV-1a 64 over that line sequence, each line followed
//! by a newline. `corpus_digest` is FNV-1a 64 over the concatenated canonical
//! game records of the section (every line of a game before its `game_trace:`
//! block, each followed by a newline), so it covers results and per-game trace
//! digests but not the revision header.
//!
//! Every game is checked before it is recorded: its trace must be non-empty and
//! its length must equal the reported decision count. A failure is an error,
//! never a silently shorter corpus.
//!
//! The same source is compiled against a baseline or candidate library. Pass
//! `--revision REV` (or set `HEURISTIC_TRACE_REVISION`) to label the source
//! version that produced a corpus; the default label is `unrecorded`.

use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::process::ExitCode;

use sichuan_mahjong_sim::policy::{HeuristicPolicy, Policy};
use sichuan_mahjong_sim::runner::{DecisionRecord, FinishReason, RunReport, Runner, RunnerError};
use sichuan_mahjong_sim::{Action, Tile, PLAYER_COUNT};

/// Output protocol identifier reported in every section header.
const PROTOCOL: &str = "heuristic-trace-v1";

/// Policy used in all four seats.
const POLICY: &str = "heuristic";

/// Frozen corpus ranges as `(first_seed, games)`.
///
/// These are the Change's comparison corpus: seeds `7..=38` and
/// `10007..=10038`, 32 games each. Changing them invalidates saved artifacts.
const CORPUS: [(u64, u64); 2] = [(7, 32), (10_007, 32)];

/// Revision label used when neither the argument nor the environment sets one.
const UNRECORDED_REVISION: &str = "unrecorded";

const USAGE: &str = "\
Usage: trace_corpus [--revision REV]

Drives four-seat HeuristicPolicy self-play through Runner::run_new for the
frozen corpus (seeds 7..=38 and 10007..=10038, 32 games each) and prints one
time-free behavioral trace section per range on stdout.

One Runner is built once and reused for a whole run. Each game reports its
decision count, trace length, terminal scores, winners, end reason, a digest of
its full action trace, and the full canonical action trace itself.

Options:
  --revision REV  Source revision label recorded in each section header, for
                  example a git commit hash. Defaults to the value of
                  HEURISTIC_TRACE_REVISION, or 'unrecorded' when unset.
  -h, --help      Print this help and exit; arguments after it are ignored.
";

/// A parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Command {
    Help,
    Run { revision: String },
}

/// A failed corpus run.
#[derive(Debug)]
enum TraceError {
    /// The runner rejected a game.
    Runner(RunnerError),
    /// A game produced an empty trace.
    EmptyTrace {
        /// Seed of the affected game.
        seed: u64,
    },
    /// A game's trace length disagreed with its decision count.
    TraceLength {
        /// Seed of the affected game.
        seed: u64,
        /// Decision count reported by the runner.
        decisions: u64,
        /// Recorded trace length.
        trace_len: usize,
    },
    /// The corpus could not be written.
    Output(io::Error),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Runner(error) => write!(f, "run failed: {error}"),
            TraceError::EmptyTrace { seed } => {
                write!(f, "seed {seed} produced an empty trace")
            }
            TraceError::TraceLength {
                seed,
                decisions,
                trace_len,
            } => write!(
                f,
                "seed {seed} reported {decisions} decisions but recorded {trace_len} trace entries"
            ),
            TraceError::Output(error) => write!(f, "trace output failed: {error}"),
        }
    }
}

impl Error for TraceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TraceError::Runner(error) => Some(error),
            TraceError::Output(error) => Some(error),
            _ => None,
        }
    }
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Run { revision }) => {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            match collect(&revision, &mut out) {
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
/// `--revision` is optional and may appear once. The label precedence is the
/// argument, then `HEURISTIC_TRACE_REVISION`, then `unrecorded`.
fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut revision = None;
    let mut arguments = args.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--revision" => {
                if revision.is_some() {
                    return Err("duplicate --revision".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--revision requires a value".to_owned())?;
                if value.is_empty() {
                    return Err("--revision requires a non-empty value".to_owned());
                }
                revision = Some(value);
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    let revision = revision
        .or_else(|| env::var("HEURISTIC_TRACE_REVISION").ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| UNRECORDED_REVISION.to_owned());
    Ok(Command::Run { revision })
}

/// Builds one runner with a fresh, independent `HeuristicPolicy` per seat.
fn heuristic_runner() -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(HeuristicPolicy::new()) as Box<dyn Policy>
    }))
}

/// Plays the frozen corpus and writes one section per seed range.
fn collect<W: Write>(revision: &str, out: &mut W) -> Result<(), TraceError> {
    let mut runner = heuristic_runner();
    for (index, &(first_seed, games)) in CORPUS.iter().enumerate() {
        if index > 0 {
            out.write_all(b"\n").map_err(TraceError::Output)?;
        }
        collect_range(&mut runner, revision, first_seed, games, out)?;
    }
    out.flush().map_err(TraceError::Output)
}

/// Plays one seed range with an existing runner and writes one section.
fn collect_range<W: Write>(
    runner: &mut Runner,
    revision: &str,
    first_seed: u64,
    games: u64,
    out: &mut W,
) -> Result<(), TraceError> {
    let header = format!(
        "protocol: {PROTOCOL}\n\
         revision: {revision}\n\
         policy: {POLICY}\n\
         seats: {PLAYER_COUNT}\n\
         seat_order: [0, 1, 2, 3]\n\
         first_seed: {first_seed}\n\
         games: {games}\n\
         digest_algorithm: fnv1a64\n"
    );
    out.write_all(header.as_bytes())
        .map_err(TraceError::Output)?;

    let mut corpus_hash = Fnv1a64::new();
    let mut total_decisions = 0u64;

    for index in 0..games {
        let seed = first_seed.wrapping_add(index);
        let report = runner.run_new(seed).map_err(TraceError::Runner)?;
        check_trace(seed, &report)?;

        let lines = canonical_trace(&report.trace);
        let mut trace_hash = Fnv1a64::new();
        for line in &lines {
            trace_hash.write(line.as_bytes());
            trace_hash.write(b"\n");
        }

        let record = game_record(seed, &report, trace_hash.finish());
        corpus_hash.write(record.as_bytes());
        out.write_all(record.as_bytes())
            .map_err(TraceError::Output)?;
        out.write_all(b"game_trace:\n")
            .map_err(TraceError::Output)?;
        for line in &lines {
            out.write_all(b"  ").map_err(TraceError::Output)?;
            out.write_all(line.as_bytes()).map_err(TraceError::Output)?;
            out.write_all(b"\n").map_err(TraceError::Output)?;
        }

        total_decisions += report.decisions;
    }

    let summary = format!(
        "corpus_games: {games}\n\
         corpus_decisions: {total_decisions}\n\
         corpus_digest: {:016x}\n",
        corpus_hash.finish()
    );
    out.write_all(summary.as_bytes())
        .map_err(TraceError::Output)
}

/// Rejects a game whose trace would not represent its decisions.
fn check_trace(seed: u64, report: &RunReport) -> Result<(), TraceError> {
    if report.trace.is_empty() {
        return Err(TraceError::EmptyTrace { seed });
    }
    if report.trace.len() as u64 != report.decisions {
        return Err(TraceError::TraceLength {
            seed,
            decisions: report.decisions,
            trace_len: report.trace.len(),
        });
    }
    Ok(())
}

/// Non-time result lines of one game, ending with the trace digest.
///
/// This block is the canonical game record hashed into `corpus_digest`; it
/// deliberately contains no timing field and no trace lines.
fn game_record(seed: u64, report: &RunReport, trace_digest: u64) -> String {
    format!(
        "game_seed: {seed}\n\
         game_decisions: {}\n\
         game_trace_len: {}\n\
         game_end_reason: {}\n\
         game_scores: {:?}\n\
         game_winners: {:?}\n\
         game_trace_digest: {trace_digest:016x}\n",
        report.decisions,
        report.trace.len(),
        end_reason_name(report.end_reason),
        report.scores,
        report.winners
    )
}

/// Stable textual end reason.
fn end_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::ThreeWinners => "ThreeWinners",
        FinishReason::WallExhausted => "WallExhausted",
    }
}

/// Canonical trace lines in decision order.
fn canonical_trace(trace: &[DecisionRecord]) -> Vec<String> {
    trace
        .iter()
        .enumerate()
        .map(|(index, record)| {
            format!(
                "{index} {} {}",
                record.seat.index(),
                canonical_action(&record.action)
            )
        })
        .collect()
}

/// Stable textual encoding of one action.
///
/// The encoding is ASCII, space-free, and independent of Rust `Debug`
/// formatting: `exchange:<tile>,<tile>,<tile>`, `missing:<suit>`,
/// `discard:<tile>`, `peng`, `open-kong`, `concealed-kong:<tile>`,
/// `added-kong:<tile>`, `hu`, `pass`.
fn canonical_action(action: &Action) -> String {
    match action {
        Action::Exchange(tiles) => format!(
            "exchange:{},{},{}",
            canonical_tile(tiles[0]),
            canonical_tile(tiles[1]),
            canonical_tile(tiles[2])
        ),
        Action::ChooseMissing(suit) => format!("missing:{}", suit.short_name()),
        Action::Discard(tile) => format!("discard:{}", canonical_tile(*tile)),
        Action::Peng => "peng".to_owned(),
        Action::OpenKong => "open-kong".to_owned(),
        Action::ConcealedKong(tile) => format!("concealed-kong:{}", canonical_tile(*tile)),
        Action::AddedKong(tile) => format!("added-kong:{}", canonical_tile(*tile)),
        Action::Hu => "hu".to_owned(),
        Action::Pass => "pass".to_owned(),
    }
}

/// Stable textual encoding of one tile: suit letter plus one-based rank.
fn canonical_tile(tile: Tile) -> String {
    format!("{}{}", tile.suit().short_name(), tile.rank())
}

/// FNV-1a 64-bit hasher, documented in the module header and standard only.
struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
