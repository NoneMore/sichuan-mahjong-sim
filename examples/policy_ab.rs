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

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};

use sichuan_mahjong_sim::policy::{
    random_seat_seed, FirstLegalPolicy, HeuristicPolicy, MissingOnlyPolicy, Policy, RandomPolicy,
};
use sichuan_mahjong_sim::runner::{FinishReason, RunReport, Runner};
use sichuan_mahjong_sim::{Seat, PLAYER_COUNT};

const USAGE: &str = "\
Usage: policy_ab --seeds LIST --policy-seed S --decision-limit N --out DIR \
                 [--a STYLE] [--b STYLE]

Runs balanced 2A2B games over an explicit seed list and writes reproducible
artifacts to a fresh directory. A defaults to FirstLegalPolicy and B to
RandomPolicy.

Options:
  --seeds LIST       Comma-separated, nonempty, distinct u64 game seeds.
  --policy-seed S    u64 seed for the per-seat random policy streams.
  --decision-limit N Positive per-game decision limit.
  --out DIR          Output directory; it must not already exist.
  --a STYLE          Policy in the two A seats: first, random, heuristic, or
                     missing-only. Default first.
  --b STYLE          Policy in the two remaining B seats: first, random,
                     heuristic, or missing-only. Default random.
  -h, --help         Print this help and exit; arguments after it are ignored.
";

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
    fn build(self, stream_seed: u64) -> Box<dyn Policy> {
        match self {
            Style::First => Box::new(FirstLegalPolicy::new()),
            Style::Random => Box::new(RandomPolicy::new(stream_seed)),
            Style::Heuristic => Box::new(HeuristicPolicy::new()),
            Style::MissingOnly => Box::new(MissingOnlyPolicy::new()),
            #[cfg(test)]
            Style::Rejecting => Box::new(RejectingPolicy),
        }
    }
}

/// Parses one policy style name from the command line.
///
/// Like the main CLI's `--policy`, the name is matched case-insensitively; the
/// recorded name is always the canonical lowercase form.
fn parse_style(value: &str) -> Result<Style, String> {
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

/// Fixed comparison plan.
#[derive(Clone, Debug)]
struct Plan {
    seeds: Vec<u64>,
    policy_seed: u64,
    decision_limit: u64,
    style_a: Style,
    style_b: Style,
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

/// Parses arguments after the executable name.
///
/// The four run inputs are required. The A/B styles are optional and default to
/// the historical `first`-versus-`random` comparison. `-h` / `--help`
/// short-circuits and ignores later arguments, matching the main CLI.
fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut seeds = None;
    let mut policy_seed = None;
    let mut decision_limit = None;
    let mut out = None;
    let mut style_a = None;
    let mut style_b = None;
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
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    Ok(Command::Run(Options {
        plan: Plan {
            seeds: seeds.ok_or_else(|| "--seeds is required".to_owned())?,
            policy_seed: policy_seed.ok_or_else(|| "--policy-seed is required".to_owned())?,
            decision_limit: decision_limit
                .ok_or_else(|| "--decision-limit is required".to_owned())?,
            style_a: style_a.unwrap_or(Style::First),
            style_b: style_b.unwrap_or(Style::Random),
        },
        out: out.ok_or_else(|| "--out is required".to_owned())?,
    }))
}

/// Parses and validates a comma-separated seed list.
///
/// The returned order is the input order. Only membership is hashed, so a long
/// seed list is validated in linear rather than quadratic time.
fn parse_seeds(value: &str) -> Result<Vec<u64>, String> {
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

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Run(options)) => {
            let invocation: Vec<String> = env::args().collect();
            match execute(&options.plan, &options.out, &invocation) {
                Ok(summary) => {
                    print!("{}", summary_text(&options.plan, &summary));
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
            eprint!("{USAGE}");
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
    let metadata = collect_metadata(Path::new(env!("CARGO_MANIFEST_DIR")));
    for warning in &metadata.warnings {
        eprintln!("warning: {warning}");
    }

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
        run_record_text(plan, invocation, &metadata),
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

    run_plan(plan, &mut records_file).map_err(|error| {
        format!(
            "{error}; completed records retained in {}",
            records_path.display()
        )
    })?;
    drop(records_file);

    let records = read_records(plan, &records_path)?;
    let summary = summarize(plan, &records)?;

    let summary_path = out.join("summary.txt");
    fs::write(&summary_path, summary_text(plan, &summary))
        .map_err(|error| format!("failed to write {}: {error}", summary_path.display()))?;

    Ok(summary)
}

/// Plays every planned placement in fixed order and writes each record as its
/// game finishes.
///
/// Only the persisted rows are produced here; the summary is later recomputed
/// from the written file, which is the sole statistics input.
fn run_plan(plan: &Plan, out: &mut dyn Write) -> Result<(), String> {
    let index = SeedIndex::new(plan);
    for &seed in &plan.seeds {
        for pair in A_PAIRS {
            let report = play_with(
                seed,
                pair,
                plan.style_a,
                plan.style_b,
                plan.policy_seed,
                plan.decision_limit,
            )
            .map_err(|error| {
                format!(
                    "game failed for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;

            let record = Record {
                seed,
                pair,
                scores: report.scores,
                end_reason: report.end_reason,
                decisions: report.decisions,
            };
            validate_record(&index, &record).map_err(|error| {
                format!(
                    "seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
            writeln!(out, "{}", format_record(&record)).map_err(|error| {
                format!(
                    "failed to write record for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
            out.flush().map_err(|error| {
                format!(
                    "failed to flush record for seed {seed} placement [{}, {}]: {error}",
                    pair[0].index(),
                    pair[1].index()
                )
            })?;
        }
    }
    Ok(())
}

/// Plays one fresh game with fresh seat policies and a fresh runner.
///
/// Every seat's policy stream starts from
/// `random_seat_seed(seed ^ policy_seed, seat)`. A deterministic policy ignores
/// the stream, and a random policy starts exactly there, so results do not
/// depend on earlier games or loop order.
fn play_with(
    seed: u64,
    pair: [Seat; 2],
    style_a: Style,
    style_b: Style,
    policy_seed: u64,
    decision_limit: u64,
) -> Result<RunReport, String> {
    let mut a_seats = [false; PLAYER_COUNT];
    for seat in pair {
        a_seats[seat.index()] = true;
    }

    let policies: [Box<dyn Policy>; PLAYER_COUNT] = std::array::from_fn(|index| {
        let seat = seat(index);
        let stream_seed = random_seat_seed(seed ^ policy_seed, seat);
        if a_seats[index] {
            style_a.build(stream_seed)
        } else {
            style_b.build(stream_seed)
        }
    });

    Runner::new(policies)
        .with_decision_limit(decision_limit)
        .run_new(seed)
        .map_err(|error| error.to_string())
}

/// Read-only seed lookup for one phase of a run.
///
/// Validation, parsing, and aggregation build this once from the plan instead
/// of rescanning `plan.seeds` for every record, which would be quadratic in the
/// number of seeds. Nothing is cached on `Plan`, so the plan stays the only
/// source of truth for the requested seed order.
struct SeedIndex<'a> {
    plan: &'a Plan,
    positions: HashMap<u64, usize>,
}

impl<'a> SeedIndex<'a> {
    fn new(plan: &'a Plan) -> Self {
        let positions = plan
            .seeds
            .iter()
            .enumerate()
            .map(|(index, seed)| (*seed, index))
            .collect();
        Self { plan, positions }
    }

    /// Position of `seed` in the plan's requested seed order.
    fn position(&self, seed: u64) -> Option<usize> {
        self.positions.get(&seed).copied()
    }
}

/// Rejects a record that does not belong to the fixed plan or is malformed.
fn validate_record(index: &SeedIndex<'_>, record: &Record) -> Result<(), String> {
    if index.position(record.seed).is_none() {
        return Err(format!("seed {} is not a planned game seed", record.seed));
    }
    if pair_index(record.pair).is_none() {
        return Err(format!(
            "A-seat pair [{}, {}] is not one of the six planned placements",
            record.pair[0].index(),
            record.pair[1].index()
        ));
    }
    let score_sum: i64 = record.scores.iter().map(|score| i64::from(*score)).sum();
    if score_sum != 0 {
        return Err(format!(
            "scores {:?} do not sum to zero (sum {score_sum})",
            record.scores
        ));
    }
    if record.decisions == 0 {
        return Err("decision count must be positive".to_owned());
    }
    if record.decisions > index.plan.decision_limit {
        return Err(format!(
            "decision count {} exceeds the configured limit {}",
            record.decisions, index.plan.decision_limit
        ));
    }
    Ok(())
}

/// Index of `pair` in the fixed placement table.
fn pair_index(pair: [Seat; 2]) -> Option<usize> {
    A_PAIRS.iter().position(|candidate| *candidate == pair)
}

/// Recomputes the balanced summary from validated raw records.
///
/// Each record is one game. `a[s,c]`/`b[s,c]` are the two A/B players' mean
/// terminal net scores, `d = a - b`, each seed's `D` is the mean of its six
/// placement contrasts, `Delta` is the mean of `D`, and `SE` is the sample
/// standard deviation of `D` over the seed blocks divided by `sqrt(N)`.
fn summarize(plan: &Plan, records: &[Record]) -> Result<Summary, String> {
    if plan.seeds.is_empty() {
        return Err("the comparison requires at least one game seed".to_owned());
    }
    let index = SeedIndex::new(plan);
    let mut seen = HashSet::new();
    let mut per_seed_sums = vec![0.0; plan.seeds.len()];
    let mut per_seed_counts = vec![0usize; plan.seeds.len()];
    let mut a_total = 0.0;
    let mut b_total = 0.0;

    for record in records {
        validate_record(&index, record)?;
        let pair_index = pair_index(record.pair).expect("validated placement");
        if !seen.insert((record.seed, pair_index)) {
            return Err(format!(
                "duplicate record for seed {} placement [{}, {}]",
                record.seed,
                record.pair[0].index(),
                record.pair[1].index()
            ));
        }
        let seed_index = index.position(record.seed).expect("validated seed");
        let (a, b) = placement_means(record);
        per_seed_sums[seed_index] += a - b;
        per_seed_counts[seed_index] += 1;
        a_total += a;
        b_total += b;
    }

    for (index, seed) in plan.seeds.iter().enumerate() {
        if per_seed_counts[index] != A_PAIRS.len() {
            return Err(format!(
                "seed {seed} has {} of {} planned placements",
                per_seed_counts[index],
                A_PAIRS.len()
            ));
        }
    }

    let per_seed: Vec<SeedContrast> = plan
        .seeds
        .iter()
        .enumerate()
        .map(|(index, seed)| SeedContrast {
            seed: *seed,
            contrast: per_seed_sums[index] / A_PAIRS.len() as f64,
        })
        .collect();
    let seeds = per_seed.len();
    let games = records.len();
    let delta = per_seed.iter().map(|entry| entry.contrast).sum::<f64>() / seeds as f64;
    let se = if seeds >= 2 {
        let variance = per_seed
            .iter()
            .map(|entry| (entry.contrast - delta).powi(2))
            .sum::<f64>()
            / (seeds - 1) as f64;
        Some(variance.sqrt() / (seeds as f64).sqrt())
    } else {
        None
    };

    Ok(Summary {
        seeds,
        games,
        a_mean: a_total / games as f64,
        b_mean: b_total / games as f64,
        delta,
        se,
        per_seed,
    })
}

/// Mean terminal net score of the two A players and of the two B players.
fn placement_means(record: &Record) -> (f64, f64) {
    let a = (i64::from(record.scores[record.pair[0].index()])
        + i64::from(record.scores[record.pair[1].index()])) as f64
        / 2.0;
    let mut b_total = 0i64;
    let mut b_count = 0usize;
    for seat in Seat::ALL {
        if record.pair.contains(&seat) {
            continue;
        }
        b_total += i64::from(record.scores[seat.index()]);
        b_count += 1;
    }
    (a, b_total as f64 / b_count as f64)
}

/// Serializes a record as one tab-separated row.
fn format_record(record: &Record) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        record.seed,
        record.pair[0].index(),
        record.pair[1].index(),
        record.scores[0],
        record.scores[1],
        record.scores[2],
        record.scores[3],
        end_reason_name(record.end_reason),
        record.decisions
    )
}

fn end_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::ThreeWinners => "three_winners",
        FinishReason::WallExhausted => "wall_exhausted",
    }
}

fn parse_end_reason(field: &str) -> Result<FinishReason, String> {
    match field {
        "three_winners" => Ok(FinishReason::ThreeWinners),
        "wall_exhausted" => Ok(FinishReason::WallExhausted),
        other => Err(format!("unsupported end reason '{other}'")),
    }
}

/// Reads and validates the persisted raw records.
fn read_records(plan: &Plan, path: &Path) -> Result<Vec<Record>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    parse_records(plan, &text)
        .map_err(|error| format!("invalid records in {}: {error}", path.display()))
}

/// Parses the `games.tsv` payload, skipping blank lines and `#` comments.
fn parse_records(plan: &Plan, text: &str) -> Result<Vec<Record>, String> {
    let index = SeedIndex::new(plan);
    let mut records = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let record = parse_record(&index, line)
            .map_err(|error| format!("records line {}: {error}", line_number + 1))?;
        records.push(record);
    }
    Ok(records)
}

/// Parses and validates one tab-separated record.
///
/// Numeric fields are parsed at their target width (`u64` seed/decision count,
/// `i32` scores), so out-of-range values are rejected instead of narrowed.
fn parse_record(index: &SeedIndex<'_>, line: &str) -> Result<Record, String> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 9 {
        return Err(format!(
            "expected 9 tab-separated fields, found {}",
            fields.len()
        ));
    }

    let seed = parse_u64(fields[0], "seed")?;
    let a_seat0 = parse_seat(fields[1])?;
    let a_seat1 = parse_seat(fields[2])?;

    let mut scores = [0i32; PLAYER_COUNT];
    for (index, score) in scores.iter_mut().enumerate() {
        *score = fields[3 + index].parse::<i32>().map_err(|_| {
            format!(
                "invalid score '{}': expected a signed 32-bit integer",
                fields[3 + index]
            )
        })?;
    }

    let end_reason = parse_end_reason(fields[7])?;
    let decisions = parse_u64(fields[8], "decision count")?;

    let record = Record {
        seed,
        pair: [a_seat0, a_seat1],
        scores,
        end_reason,
        decisions,
    };
    validate_record(index, &record)?;
    Ok(record)
}

fn parse_u64(field: &str, label: &str) -> Result<u64, String> {
    field
        .parse::<u64>()
        .map_err(|_| format!("invalid {label} '{field}': expected an unsigned 64-bit integer"))
}

fn parse_seat(field: &str) -> Result<Seat, String> {
    let index = field
        .parse::<usize>()
        .map_err(|_| format!("invalid seat '{field}': expected a seat index in 0..=3"))?;
    Seat::new(index)
        .ok_or_else(|| format!("invalid seat '{field}': expected a seat index in 0..=3"))
}

/// Writes the pre-run reproducibility record.
fn run_record_text(plan: &Plan, invocation: &[String], metadata: &Metadata) -> String {
    let mut text = String::new();
    text.push_str("# minimal native policy A/B comparison\n");
    text.push_str(&format!("simulator revision: {}\n", metadata.revision));
    text.push_str(&format!(
        "working tree dirty: {}\n",
        match metadata.dirty {
            Some(true) => "yes",
            Some(false) => "no",
            None => "unknown",
        }
    ));
    text.push_str(&format!("toolchain: {}\n", metadata.toolchain));
    text.push_str(&format!(
        "invocation argv (debug-quoted, one element per argument): {invocation:?}\n"
    ));
    text.push_str(&format!("policy A: {}\n", plan.style_a.name()));
    text.push_str(&format!("policy B: {}\n", plan.style_b.name()));
    text.push_str(&format!("policy seed: {}\n", plan.policy_seed));
    text.push_str(
        "policy stream derivation: random_seat_seed(game_seed ^ policy_seed, seat); \
         deterministic policies ignore the stream seed\n",
    );
    let seeds: Vec<String> = plan.seeds.iter().map(|seed| seed.to_string()).collect();
    text.push_str(&format!("game seeds: {}\n", seeds.join(",")));
    text.push_str(&format!(
        "decision limit per game: {}\n",
        plan.decision_limit
    ));
    let placements: Vec<String> = A_PAIRS
        .iter()
        .map(|pair| format!("[{},{}]", pair[0].index(), pair[1].index()))
        .collect();
    text.push_str(&format!(
        "placements (A seats): {}; B seats are the rest\n",
        placements.join(" ")
    ));
    text.push_str(
        "records: games.tsv fields seed, a_seat0, a_seat1, score0..score3, \
         end_reason, decisions; summary.txt is recomputed from them\n",
    );
    for warning in &metadata.warnings {
        text.push_str(&format!("warning: {warning}\n"));
    }
    text
}

/// Formats the human-readable summary also written to `summary.txt`.
fn summary_text(plan: &Plan, summary: &Summary) -> String {
    let mut text = String::new();
    text.push_str(&format!("policy A: {}\n", plan.style_a.name()));
    text.push_str(&format!("policy B: {}\n", plan.style_b.name()));
    text.push_str(&format!(
        "seeds: {}, games: {}, placements per seed: {}\n",
        summary.seeds,
        summary.games,
        A_PAIRS.len()
    ));
    text.push_str(&format!(
        "A_mean: {:.6}  B_mean: {:.6}\n",
        summary.a_mean, summary.b_mean
    ));
    text.push_str(&format!("Delta: {:.6}\n", summary.delta));
    match summary.se {
        Some(se) => text.push_str(&format!("SE: {se:.6}\n")),
        None => text.push_str("SE: unavailable (fewer than 2 seed blocks)\n"),
    }
    text.push_str("per-seed D:\n");
    for entry in &summary.per_seed {
        text.push_str(&format!("  seed {}: {:.6}\n", entry.seed, entry.contrast));
    }
    text.push_str(
        "note: SE describes variation among these fixed seed blocks; \
         it is not population-level precision.\n",
    );
    text
}

/// Collects best-effort revision, dirty, and toolchain facts.
///
/// Commands run against `source_dir` (the tree that built this example), so the
/// metadata describes that source tree rather than the caller's working
/// directory. A missing or unusable source tree degrades to `unknown` with a
/// warning instead of failing the run.
fn collect_metadata(source_dir: &Path) -> Metadata {
    let mut warnings = Vec::new();
    let revision = metadata_value(
        command_stdout("git", &["rev-parse", "HEAD"], source_dir),
        "simulator revision",
        &mut warnings,
    );
    let dirty = match command_stdout("git", &["status", "--porcelain"], source_dir) {
        Some(output) => Some(!output.trim().is_empty()),
        None => {
            warnings.push("working tree status unavailable; recorded as unknown".to_owned());
            None
        }
    };
    let toolchain = metadata_value(
        command_stdout("rustc", &["--version"], source_dir),
        "Rust toolchain",
        &mut warnings,
    );
    Metadata {
        revision,
        dirty,
        toolchain,
        warnings,
    }
}

fn metadata_value(result: Option<String>, label: &str, warnings: &mut Vec<String>) -> String {
    match result {
        Some(value) => value.trim().to_owned(),
        None => {
            warnings.push(format!("{label} unavailable; recorded as unknown"));
            "unknown".to_owned()
        }
    }
}

/// Runs a helper command in `current_dir` and returns its stdout on success.
fn command_stdout(program: &str, args: &[&str], current_dir: &Path) -> Option<String> {
    let output = ProcessCommand::new(program)
        .args(args)
        .current_dir(current_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
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
