//! Minimal balanced 2A2B native policy comparison.
//!
//! For every planned game seed this example plays six fresh games with policy A
//! in each of the six two-seat pairs `[0,1] [0,2] [0,3] [1,2] [1,3] [2,3]` and
//! policy B in the remaining two seats. Fresh policies and a fresh
//! [`Runner`] are built for every placement, so no policy stream continues
//! between games or depends on loop order.
//!
//! The comparison uses only the existing `Policy`/`Runner` surface: policies see
//! `Decision` values, the runner keeps its legality checks and per-game decision
//! limit, and the game owns all rules and settlement. The tool validates the
//! comparison workflow, not policy strength.
//!
//! A run writes a reproducibility record, one raw record per completed game
//! (flushed as games finish), and a summary recomputed from the persisted raw
//! records. Any failed game, output error, or consistency failure stops the run
//! with a nonzero exit and no summary.

use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/cache_tool.rs"]
mod cache_tool;

#[path = "policy_ab/cli.rs"]
mod cli;
#[path = "policy_ab/comparison.rs"]
mod comparison;
#[path = "policy_ab/records.rs"]
mod records;
#[path = "policy_ab/report.rs"]
mod report;

use cache_tool::{CacheWarnings, SharedCache};
use sichuan_mahjong_sim::policy::{
    FirstLegalPolicy, HeuristicPolicy, MissingOnlyPolicy, Policy, RandomPolicy,
    DEFAULT_BUDGET_BYTES,
};
use sichuan_mahjong_sim::runner::FinishReason;
use sichuan_mahjong_sim::{Seat, PLAYER_COUNT};

/// The six balanced A-seat pairs, in the order games are played.
const A_PAIRS: [[Seat; 2]; 6] = [
    [seat(0), seat(1)],
    [seat(0), seat(2)],
    [seat(0), seat(3)],
    [seat(1), seat(2)],
    [seat(1), seat(3)],
    [seat(2), seat(3)],
];

/// Const seat constructor for the fixed placement table.
const fn seat(index: usize) -> Seat {
    match Seat::new(index) {
        Some(seat) => seat,
        None => panic!("seat index out of range"),
    }
}

/// Heuristic cache budget selected for the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheBudget {
    CandidateDefault,
    Explicit(usize),
}

impl CacheBudget {
    fn bytes(self) -> usize {
        match self {
            CacheBudget::CandidateDefault => DEFAULT_BUDGET_BYTES,
            CacheBudget::Explicit(bytes) => bytes,
        }
    }

    fn source(self) -> &'static str {
        match self {
            CacheBudget::CandidateDefault => "candidate-default",
            CacheBudget::Explicit(_) => "explicit",
        }
    }
}

/// One of the policies under comparison.
///
/// The command line exposes the A/B choice; tests use the same builders to
/// compare identical policies or to inject a rejecting policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    First,
    Random,
    Heuristic,
    MissingOnly,
    #[cfg(test)]
    Rejecting,
}

impl Style {
    /// Command-line name, also written to the run record.
    fn name(self) -> &'static str {
        match self {
            Style::First => "first",
            Style::Random => "random",
            Style::Heuristic => "heuristic",
            Style::MissingOnly => "missing-only",
            #[cfg(test)]
            Style::Rejecting => "rejecting",
        }
    }

    /// Builds a fresh policy. `stream_seed` is the seat's derived policy seed;
    /// deterministic policies ignore it. `HeuristicPolicy` is deterministic and
    /// has no logical checkpoint state: any internal analysis cache it may keep
    /// is discardable and never changes later choices, so a fresh instance is a
    /// valid continuation of an existing heuristic seat. `MissingOnlyPolicy` is
    /// stateless for the same reason.
    ///
    /// When `cache` is supplied, a heuristic seat binds to that shared context.
    /// Other styles ignore it. A missing cache gives a heuristic seat its own
    /// independent default context, which is useful for tests but is not how
    /// this tool runs a comparison.
    fn build(self, stream_seed: u64, cache: Option<&SharedCache>) -> Box<dyn Policy> {
        match self {
            Style::First => Box::new(FirstLegalPolicy::new()),
            Style::Random => Box::new(RandomPolicy::new(stream_seed)),
            Style::Heuristic => match cache {
                Some(cache) => Box::new(cache.policy()),
                None => Box::new(HeuristicPolicy::new()),
            },
            Style::MissingOnly => Box::new(MissingOnlyPolicy::new()),
            #[cfg(test)]
            Style::Rejecting => Box::new(RejectingPolicy),
        }
    }
}

/// Fixed comparison plan.
#[derive(Clone, Debug)]
struct Plan {
    seeds: Vec<u64>,
    policy_seed: u64,
    decision_limit: u64,
    style_a: Style,
    style_b: Style,
    cache_budget: CacheBudget,
}

impl Plan {
    /// Whether this comparison has any heuristic seat that uses a cache.
    fn uses_heuristic(&self) -> bool {
        self.style_a == Style::Heuristic || self.style_b == Style::Heuristic
    }
}

/// One completed game, also the serialized `games.tsv` row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Record {
    seed: u64,
    pair: [Seat; 2],
    scores: [i32; PLAYER_COUNT],
    end_reason: FinishReason,
    decisions: u64,
}

/// One seed block's A/B contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SeedContrast {
    seed: u64,
    contrast: f64,
}

/// Balanced comparison summary.
#[derive(Clone, Debug, PartialEq)]
struct Summary {
    seeds: usize,
    games: usize,
    a_mean: f64,
    b_mean: f64,
    delta: f64,
    se: Option<f64>,
    per_seed: Vec<SeedContrast>,
}

/// Environment facts recorded before play.
#[derive(Debug)]
struct Metadata {
    revision: String,
    dirty: Option<bool>,
    toolchain: String,
    warnings: Vec<String>,
}

#[derive(Debug)]
struct Options {
    plan: Plan,
    out: PathBuf,
}

#[derive(Debug)]
enum Command {
    Help,
    Run(Options),
}

fn main() -> ExitCode {
    match cli::parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{}", cli::usage());
            ExitCode::SUCCESS
        }
        Ok(Command::Run(options)) => {
            let invocation: Vec<String> = env::args().collect();
            match execute(&options.plan, &options.out, &invocation) {
                Ok(summary) => {
                    print!("{}", report::summary_text(&options.plan, &summary));
                    eprintln!("artifacts: {}", options.out.display());
                    ExitCode::SUCCESS
                }
                Err(message) => {
                    eprintln!("error: {message}");
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

/// Runs a complete plan, writes its artifacts, and returns the summary.
///
/// The summary is recomputed from the persisted `games.tsv` records, so the
/// stored raw records are proven sufficient for the reported numbers. A failure
/// at any point propagates before `summary.txt` is written.
fn execute(plan: &Plan, out: &Path, invocation: &[String]) -> Result<Summary, String> {
    let metadata = report::collect_metadata(Path::new(env!("CARGO_MANIFEST_DIR")));
    for warning in &metadata.warnings {
        eprintln!("warning: {warning}");
    }

    // The heuristic context is created before any artifact or game so an
    // invalid budget fails cleanly and an explicit scope is recorded up front.
    let cache = if plan.uses_heuristic() {
        Some(
            SharedCache::new(plan.cache_budget.bytes())
                .map_err(|error| format!("invalid heuristic cache budget: {error}"))?,
        )
    } else {
        None
    };
    let initial_cache = cache.as_ref().map(SharedCache::snapshot);

    fs::create_dir(out).map_err(|error| {
        if error.kind() == io::ErrorKind::AlreadyExists {
            format!(
                "output directory {} already exists; refusing to overwrite",
                out.display()
            )
        } else {
            format!("cannot create output directory {}: {error}", out.display())
        }
    })?;

    let run_record_path = out.join("run.txt");
    fs::write(
        &run_record_path,
        report::run_record_text(plan, invocation, &metadata, initial_cache),
    )
    .map_err(|error| format!("failed to write {}: {error}", run_record_path.display()))?;

    let records_path = out.join("games.tsv");
    let mut records_file = File::create(&records_path)
        .map_err(|error| format!("failed to create {}: {error}", records_path.display()))?;
    writeln!(
        records_file,
        "# seed\ta_seat0\ta_seat1\tscore0\tscore1\tscore2\tscore3\tend_reason\tdecisions"
    )
    .map_err(|error| format!("failed to write {}: {error}", records_path.display()))?;
    records_file
        .flush()
        .map_err(|error| format!("failed to write {}: {error}", records_path.display()))?;

    let run_result = comparison::run_plan_with_cache(plan, &mut records_file, cache.as_ref());
    drop(records_file);

    // Report the context once. The library never prints, so the tool owns both
    // the diagnostic and its persistence, including on a failed run.
    let mut cache_warnings = Vec::new();
    if let Some(cache) = &cache {
        let snapshot = cache.snapshot();
        let mut warner = CacheWarnings::default();
        if let Some(warning) = warner.observe(snapshot, "policy_ab heuristic cache") {
            eprintln!("warning: {warning}");
            cache_warnings.push(warning);
        }
        let cache_path = out.join("cache.txt");
        let text = report::cache_record_text(plan, initial_cache, snapshot, &cache_warnings);
        match fs::write(&cache_path, text) {
            Ok(()) => {}
            // Preserve the original run error; a degraded or failed run keeps
            // its records even if the cache observation cannot be written.
            Err(_) if run_result.is_err() => {}
            Err(error) => return Err(format!("failed to write {}: {error}", cache_path.display())),
        }
    }

    run_result.map_err(|error| {
        format!(
            "{error}; completed records retained in {}",
            records_path.display()
        )
    })?;

    let records = records::read_records(plan, &records_path)?;
    let summary = records::summarize(plan, &records)?;

    let summary_path = out.join("summary.txt");
    fs::write(&summary_path, report::summary_text(plan, &summary))
        .map_err(|error| format!("failed to write {}: {error}", summary_path.display()))?;

    Ok(summary)
}

#[cfg(test)]
struct RejectingPolicy;

#[cfg(test)]
impl Policy for RejectingPolicy {
    fn choose(
        &mut self,
        _decision: &sichuan_mahjong_sim::Decision<'_>,
    ) -> sichuan_mahjong_sim::Action {
        // Pass is not a legal exchange, missing-suit, or turn choice, so the
        // runner reports a rejected action instead of hiding it.
        sichuan_mahjong_sim::Action::Pass
    }
}

#[cfg(test)]
#[path = "policy_ab/tests.rs"]
mod tests;
