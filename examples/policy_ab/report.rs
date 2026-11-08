//! Reproducibility and summary text for one comparison run.
//!
//! Nothing here plays a game or writes a file: the caller owns the artifact
//! paths and the failure behavior, so a failed run keeps its records and never
//! gains a summary.

use std::path::Path;
use std::process::Command as ProcessCommand;

use super::cache_tool::CacheSnapshot;
use super::{Metadata, Plan, Summary, A_PAIRS};

/// Writes the pre-run reproducibility record.
pub(super) fn run_record_text(
    plan: &Plan,
    invocation: &[String],
    metadata: &Metadata,
    cache_initial: Option<CacheSnapshot>,
) -> String {
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
    if plan.uses_heuristic() {
        text.push_str(
            "heuristic cache scope: shared by all heuristic seats and all games in this run; \
             non-heuristic seats are untouched\n",
        );
        text.push_str(&format!(
            "heuristic cache budget source: {}\n",
            plan.cache_budget.source()
        ));
        text.push_str(&format!(
            "heuristic cache budget bytes: {}\n",
            plan.cache_budget.bytes()
        ));
        match cache_initial {
            Some(snapshot) => text.push_str(&format!(
                "heuristic cache initial state: {}\nheuristic cache context overhead bytes: {}\n",
                snapshot.state, snapshot.context_overhead_bytes
            )),
            None => text.push_str("heuristic cache initial state: unknown\n"),
        }
    } else {
        text.push_str("heuristic cache: not applicable (no heuristic seat)\n");
    }
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

/// Formats the persisted heuristic-cache observation.
///
/// This record exists only when the comparison has a heuristic seat. It states
/// the explicit sharing scope, the configured and actual allocation, and any
/// single allocation-failure warning, so a run can be classified without
/// scraping stderr.
pub(super) fn cache_record_text(
    plan: &Plan,
    initial: Option<CacheSnapshot>,
    final_snapshot: CacheSnapshot,
    warnings: &[String],
) -> String {
    let mut text = String::new();
    text.push_str("# heuristic cache observation\n");
    text.push_str(
        "scope: all heuristic seats in all games of this run; non-heuristic seats are untouched\n",
    );
    text.push_str(&format!(
        "budget source: {}\nbudget bytes: {}\n",
        plan.cache_budget.source(),
        plan.cache_budget.bytes()
    ));
    if let Some(initial) = initial {
        text.push_str(&format!(
            "initial state: {}\n\
             initial context overhead bytes: {}\n\
             initial allocated bytes: {}\n\
             initial capacity entries: {}\n\
             initial occupied entries: {}\n\
             initial evictions: {}\n\
             initial hits: {}\n\
             initial misses: {}\n",
            initial.state,
            initial.context_overhead_bytes,
            initial.allocated_bytes,
            initial.capacity_entries,
            initial.occupied_entries,
            initial.evictions,
            initial.hits,
            initial.misses
        ));
    }
    text.push_str(&format!(
        "final state: {}\n\
         final context overhead bytes: {}\n\
         final allocated bytes: {}\n\
         final capacity entries: {}\n\
         final occupied entries: {}\n\
         final evictions: {}\n\
         final hits: {}\n\
         final misses: {}\n",
        final_snapshot.state,
        final_snapshot.context_overhead_bytes,
        final_snapshot.allocated_bytes,
        final_snapshot.capacity_entries,
        final_snapshot.occupied_entries,
        final_snapshot.evictions,
        final_snapshot.hits,
        final_snapshot.misses
    ));
    text.push_str(&format!("warning count: {}\n", warnings.len()));
    for warning in warnings {
        text.push_str(&format!("warning: {warning}\n"));
    }
    text
}

/// Formats the human-readable summary also written to `summary.txt`.
pub(super) fn summary_text(plan: &Plan, summary: &Summary) -> String {
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
pub(super) fn collect_metadata(source_dir: &Path) -> Metadata {
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

pub(super) fn metadata_value(
    result: Option<String>,
    label: &str,
    warnings: &mut Vec<String>,
) -> String {
    match result {
        Some(value) => value.trim().to_owned(),
        None => {
            warnings.push(format!("{label} unavailable; recorded as unknown"));
            "unknown".to_owned()
        }
    }
}

/// Runs a helper command in `current_dir` and returns its stdout on success.
pub(super) fn command_stdout(program: &str, args: &[&str], current_dir: &Path) -> Option<String> {
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
