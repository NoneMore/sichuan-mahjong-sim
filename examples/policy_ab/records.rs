//! Codec, validation, and summary statistics for the persisted game records.
//!
//! `games.tsv` is the only statistics input: every row is validated against the
//! fixed plan and the summary is recomputed from the persisted rows, so the
//! stored records are proven sufficient for the reported numbers.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use super::{Plan, Record, SeedContrast, Summary, A_PAIRS, PLAYER_COUNT};
use sichuan_mahjong_sim::runner::FinishReason;
use sichuan_mahjong_sim::Seat;

/// Read-only seed lookup for one phase of a run.
///
/// Validation, parsing, and aggregation build this once from the plan instead
/// of rescanning `plan.seeds` for every record, which would be quadratic in the
/// number of seeds. Nothing is cached on `Plan`, so the plan stays the only
/// source of truth for the requested seed order.
pub(super) struct SeedIndex<'a> {
    plan: &'a Plan,
    positions: HashMap<u64, usize>,
}

impl<'a> SeedIndex<'a> {
    pub(super) fn new(plan: &'a Plan) -> Self {
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
pub(super) fn validate_record(index: &SeedIndex<'_>, record: &Record) -> Result<(), String> {
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
pub(super) fn pair_index(pair: [Seat; 2]) -> Option<usize> {
    A_PAIRS.iter().position(|candidate| *candidate == pair)
}

/// Recomputes the balanced summary from validated raw records.
///
/// Each record is one game. `a[s,c]`/`b[s,c]` are the two A/B players' mean
/// terminal net scores, `d = a - b`, each seed's `D` is the mean of its six
/// placement contrasts, `Delta` is the mean of `D`, and `SE` is the sample
/// standard deviation of `D` over the seed blocks divided by `sqrt(N)`.
pub(super) fn summarize(plan: &Plan, records: &[Record]) -> Result<Summary, String> {
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
pub(super) fn format_record(record: &Record) -> String {
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
pub(super) fn read_records(plan: &Plan, path: &Path) -> Result<Vec<Record>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    parse_records(plan, &text)
        .map_err(|error| format!("invalid records in {}: {error}", path.display()))
}

/// Parses the `games.tsv` payload, skipping blank lines and `#` comments.
pub(super) fn parse_records(plan: &Plan, text: &str) -> Result<Vec<Record>, String> {
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
