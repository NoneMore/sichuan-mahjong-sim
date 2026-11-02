//! Experiment orchestration: seeds, six placements, per-game JSONL output,
//! and failure records.
//!
//! One run drives every distinct seed from a JSON array and the six fixed
//! A-seat placements in a fixed order. Completed games are streamed to the
//! output file one JSON object per line and flushed, so a later failure keeps
//! the diagnostic partial results. The first failure stops the run:
//!
//! - one failure record (an `error` object without `scores`) is appended to
//!   the same output file, after every completed game;
//! - the process exit status is nonzero;
//! - no baseline action or zero score is substituted.
//!
//! A run is successful only when all six placements completed for every seed.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};

use crate::driver::{
    self, Chooser, ErrorContext, FirstLegalChooser, HeuristicChooser, Mode, PlayedGame,
    SeatChoosers, SeatPlan, PLACEMENTS,
};
use crate::worker::{self, WorkerCommand, WorkerProcess};

/// Fully parsed experiment configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// JSON array of distinct game seeds.
    pub seeds: PathBuf,
    /// Per-game JSONL output. Must not exist yet.
    pub output: PathBuf,
    /// Python interpreter.
    pub python: PathBuf,
    /// Worker script.
    pub worker: PathBuf,
    /// Repository root passed to the worker as `--ninone-root`.
    pub ninone_root: PathBuf,
    /// Seat-policy mode.
    pub mode: Mode,
    /// Per-game decision safety bound.
    pub decision_limit: u64,
}

/// Successful run facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    /// Number of completed games.
    pub games: u64,
    /// Number of distinct seeds.
    pub seeds: usize,
    /// Output path.
    pub output: PathBuf,
}

/// One completed game on the wire.
///
/// `trace_hash` is the deterministic identity of the game's action trace (see
/// [`driver::trace_hash`]); timing is never part of it.
#[derive(Debug, Serialize)]
struct GameRecord {
    seed: u64,
    a_seats: [u8; 2],
    scores: [i64; 4],
    end_reason: &'static str,
    decisions: u64,
    trace_hash: String,
}

impl GameRecord {
    fn from_game(game: &PlayedGame) -> Self {
        Self {
            seed: game.seed,
            a_seats: game.a_seats,
            scores: game.scores,
            end_reason: game.end_reason,
            decisions: game.decisions,
            trace_hash: driver::trace_hash_hex(game.trace_hash),
        }
    }
}

/// One failure on the wire. `context` carries the pending decision facts and
/// never includes opponent concealed hands or the wall order.
#[derive(Debug, Serialize)]
struct FailureRecord {
    error: String,
    seed: u64,
    a_seats: [u8; 2],
    decision_index: u64,
    context: Value,
}

/// Reads and validates the distinct game seeds.
pub fn load_seeds(path: &Path) -> Result<Vec<u64>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read seeds file {}: {error}", path.display()))?;
    let seeds: Vec<u64> = serde_json::from_str(&text).map_err(|error| {
        format!(
            "seeds file {} is not a JSON array of u64: {error}",
            path.display()
        )
    })?;
    if seeds.is_empty() {
        return Err(format!("seeds file {} contains no seeds", path.display()));
    }
    let mut seen = HashSet::with_capacity(seeds.len());
    for &seed in &seeds {
        if !seen.insert(seed) {
            return Err(format!(
                "seeds file {} contains duplicate seed {seed}",
                path.display()
            ));
        }
    }
    Ok(seeds)
}

fn create_output(path: &Path) -> Result<BufWriter<File>, String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                format!(
                    "output file {} already exists (refusing to overwrite)",
                    path.display()
                )
            } else {
                format!("cannot create output file {}: {error}", path.display())
            }
        })?;
    Ok(BufWriter::new(file))
}

fn write_json_line(writer: &mut impl Write, line: &str) -> Result<(), String> {
    writeln!(writer, "{line}").map_err(|error| format!("cannot write output line: {error}"))
}

fn write_game(writer: &mut impl Write, record: &GameRecord) -> Result<(), String> {
    let line = serde_json::to_string(record)
        .map_err(|error| format!("cannot serialize game record: {error}"))?;
    write_json_line(writer, &line)
}

fn write_failure(writer: &mut impl Write, record: &FailureRecord) -> Result<(), String> {
    let line = serde_json::to_string(record)
        .map_err(|error| format!("cannot serialize failure record: {error}"))?;
    write_json_line(writer, &line)
}

/// Worker and ninone paths are made absolute because the worker process keeps
/// the caller's working directory; the interpreter is left to PATH lookup.
fn absolute(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir()
        .map_err(|error| format!("cannot resolve current directory: {error}"))?;
    Ok(cwd.join(path))
}

fn spawn_worker(config: &Config) -> Result<WorkerProcess, String> {
    let worker = absolute(&config.worker)?;
    let ninone_root = absolute(&config.ninone_root)?;
    let command = WorkerCommand::new(config.python.clone().into_os_string())
        .arg(worker.into_os_string())
        .arg("--ninone-root")
        .arg(ninone_root.into_os_string());
    WorkerProcess::spawn_with_timeout(command, worker::timeout_from_env())
}

fn play_one(
    worker: &mut Option<WorkerProcess>,
    plan: SeatPlan,
    seed: u64,
    decision_limit: u64,
) -> Result<PlayedGame, driver::PlayError> {
    let mut first = FirstLegalChooser::new();
    let mut heuristic = worker.as_mut().map(|worker| HeuristicChooser::new(worker));
    let mut choosers = SeatChoosers {
        first: &mut first,
        heuristic: heuristic
            .as_mut()
            .map(|chooser| chooser as &mut dyn Chooser),
    };
    driver::play_game(seed, plan, &mut choosers, decision_limit)
}

fn decision_context_value(context: &ErrorContext) -> Value {
    json!({
        "phase": "decision",
        "seat": context.seat,
        "stage": context.stage,
        "candidate_count": context.candidate_count,
        "request": context.request,
    })
}

/// Runs the whole experiment, stopping at the first failure.
pub fn run(config: &Config) -> Result<Summary, String> {
    if config.decision_limit == 0 {
        return Err("decision limit must be at least 1".to_owned());
    }
    let seeds = load_seeds(&config.seeds)?;
    let mut writer = create_output(&config.output)?;

    let mut worker = if config.mode.uses_heuristic() {
        match spawn_worker(config) {
            Ok(worker) => Some(worker),
            Err(message) => {
                let record = FailureRecord {
                    error: message.clone(),
                    seed: seeds[0],
                    a_seats: PLACEMENTS[0],
                    decision_index: 0,
                    context: json!({ "phase": "worker_start", "attempted": false }),
                };
                write_failure(&mut writer, &record)?;
                writer.flush().map_err(|flush_error| {
                    format!(
                        "cannot flush failure record to {}: {flush_error}",
                        config.output.display()
                    )
                })?;
                return Err(message);
            }
        }
    } else {
        None
    };

    let mut completed = 0u64;
    let mut last = (seeds[0], PLACEMENTS[0], 0u64);
    for &seed in &seeds {
        for a_seats in PLACEMENTS {
            let plan = SeatPlan::new(config.mode, a_seats)?;
            match play_one(&mut worker, plan, seed, config.decision_limit) {
                Ok(game) => {
                    write_game(&mut writer, &GameRecord::from_game(&game))?;
                    completed += 1;
                    last = (seed, a_seats, game.decisions);
                }
                Err(error) => {
                    let record = FailureRecord {
                        error: error.message(),
                        seed,
                        a_seats,
                        decision_index: error.decision_index(),
                        context: error
                            .context()
                            .map_or_else(|| json!({ "phase": "game_end" }), decision_context_value),
                    };
                    write_failure(&mut writer, &record)?;
                    writer.flush().map_err(|flush_error| {
                        format!(
                            "cannot flush partial results to {}: {flush_error}",
                            config.output.display()
                        )
                    })?;
                    if let Some(worker) = worker.as_mut() {
                        let _ = worker.shutdown();
                    }
                    return Err(format!(
                        "game failed for seed {seed} placement {a_seats:?}: {}",
                        error.message()
                    ));
                }
            }
        }
    }

    if let Some(worker) = worker.as_mut() {
        if let Err(error) = worker.shutdown() {
            let message = format!("worker shutdown failed: {error}");
            let record = FailureRecord {
                error: message.clone(),
                seed: last.0,
                a_seats: last.1,
                decision_index: last.2,
                context: json!({
                    "phase": "worker_shutdown",
                    "completed_games": completed,
                }),
            };
            write_failure(&mut writer, &record)?;
            writer.flush().map_err(|flush_error| {
                format!(
                    "cannot flush results to {}: {flush_error}",
                    config.output.display()
                )
            })?;
            return Err(message);
        }
    }

    writer
        .flush()
        .map_err(|error| format!("cannot flush {}: {error}", config.output.display()))?;
    Ok(Summary {
        games: completed,
        seeds: seeds.len(),
        output: config.output.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    fn temp_dir(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("policy_ab_test_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn write_file(path: &Path, contents: &str) {
        fs::write(path, contents).unwrap();
    }

    fn python() -> Option<String> {
        if let Ok(program) = std::env::var("POLICY_AB_TEST_PYTHON") {
            return Some(program);
        }
        ["python", "python3"].iter().find_map(|program| {
            Command::new(program)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .ok()
                .filter(|status| status.success())
                .map(|_| (*program).to_owned())
        })
    }

    fn config(directory: &Path, seeds: &str, mode: Mode) -> Config {
        let seeds_path = directory.join("seeds.json");
        write_file(&seeds_path, seeds);
        fs::create_dir_all(directory.join("ninone")).unwrap();
        Config {
            seeds: seeds_path,
            output: directory.join("results.jsonl"),
            python: PathBuf::from("python-not-needed"),
            worker: PathBuf::from("worker-not-needed.py"),
            ninone_root: directory.join("ninone"),
            mode,
            decision_limit: 10_000,
        }
    }

    fn read_lines(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn seeds_are_distinct_u64_values_in_file_order() {
        let directory = temp_dir("seeds");
        let path = directory.join("seeds.json");
        write_file(&path, "[7, 3, 99]");
        assert_eq!(load_seeds(&path).unwrap(), vec![7, 3, 99]);

        for bad in ["[]", "[1, 1]", "[1, -2]", "[1, 2.5]", "{}", "1, 2"] {
            write_file(&path, bad);
            assert!(
                load_seeds(&path).is_err(),
                "seeds {bad:?} should be rejected"
            );
        }
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn output_is_created_new() {
        let directory = temp_dir("output");
        let path = directory.join("results.jsonl");
        create_output(&path).unwrap();
        let error = create_output(&path).unwrap_err();
        assert!(error.contains("already exists"), "{error}");
        assert_eq!(read_lines(&path).len(), 0);
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn per_game_record_shape_is_pinned() {
        let record = GameRecord {
            seed: 7,
            a_seats: [0, 1],
            scores: [1, -1, 2, -2],
            end_reason: "three_winners",
            decisions: 12,
            trace_hash: "00000000000000ab".to_owned(),
        };
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            concat!(
                r#"{"seed":7,"a_seats":[0,1],"scores":[1,-1,2,-2],"#,
                r#""end_reason":"three_winners","decisions":12,"#,
                r#""trace_hash":"00000000000000ab"}"#
            )
        );
        let played = GameRecord::from_game(&PlayedGame {
            seed: 7,
            a_seats: [0, 1],
            scores: [1, -1, 2, -2],
            end_reason: "three_winners",
            decisions: 12,
            trace: Vec::new(),
            trace_hash: 0xab,
        });
        assert_eq!(played.trace_hash, "00000000000000ab");
    }

    #[test]
    fn failure_record_shape_is_pinned() {
        let record = FailureRecord {
            error: "worker returned unknown candidate id 'x' (offered 2 candidates)".to_owned(),
            seed: 4,
            a_seats: [0, 2],
            decision_index: 7,
            context: decision_context_value(&ErrorContext {
                decision_index: 7,
                seat: 1,
                stage: "turn",
                candidate_count: 2,
                request: None,
            }),
        };
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            concat!(
                r#"{"error":"worker returned unknown candidate id 'x' (offered 2 candidates)","#,
                r#""seed":4,"a_seats":[0,2],"decision_index":7,"#,
                r#""context":{"candidate_count":2,"phase":"decision","request":null,"seat":1,"stage":"turn"}}"#
            )
        );
    }

    #[test]
    fn first_only_runs_never_touch_the_worker() {
        let directory = temp_dir("first_only");
        let config = config(&directory, "[5]", Mode::FirstOnly);
        let summary = run(&config).unwrap();
        assert_eq!(summary.games, 6);
        assert_eq!(summary.seeds, 1);
        let lines = read_lines(&config.output);
        assert_eq!(lines.len(), 6);
        let mut contrasts = Vec::new();
        for (line, placement) in lines.iter().zip(PLACEMENTS) {
            let record: Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["seed"], 5);
            assert_eq!(record["a_seats"], json!([placement[0], placement[1]]));
            let scores: Vec<i64> = record["scores"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect();
            assert_eq!(scores.iter().sum::<i64>(), 0);
            let a = (scores[placement[0] as usize] + scores[placement[1] as usize]) as f64 / 2.0;
            let b_sum: i64 = (0..4)
                .filter(|seat| !placement.contains(&(*seat as u8)))
                .map(|seat| scores[seat])
                .sum();
            contrasts.push(a - b_sum as f64 / 2.0);
        }
        let mean = contrasts.iter().sum::<f64>() / contrasts.len() as f64;
        assert!(mean.abs() < 1e-9, "placement mean contrast {mean}");
        assert!(lines.iter().all(|line| !line.contains("\"error\"")));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn decision_limit_failure_preserves_context_and_exit_error() {
        let directory = temp_dir("limit");
        let mut config = config(&directory, "[6]", Mode::FirstOnly);
        config.decision_limit = 1;
        let error = run(&config).unwrap_err();
        assert!(error.contains("decision limit 1"), "{error}");
        let failures = read_lines(&config.output);
        assert_eq!(failures.len(), 1);
        let record: Value = serde_json::from_str(&failures[0]).unwrap();
        assert!(record.get("scores").is_none());
        assert_eq!(record["seed"], 6);
        assert_eq!(record["a_seats"], json!([0, 1]));
        assert_eq!(record["decision_index"], 1);
        assert_eq!(record["context"]["phase"], "decision");
        assert_eq!(record["context"]["seat"], 1);
        assert_eq!(record["context"]["stage"], "exchange");
        assert!(record["context"]["candidate_count"].as_u64().unwrap() > 0);
        assert_eq!(
            record["context"]["request"]["candidates"]
                .as_array()
                .unwrap()
                .len() as u64,
            record["context"]["candidate_count"].as_u64().unwrap()
        );
        assert!(record["context"]["request"]["missing_suit"].is_null());
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn worker_failure_keeps_completed_games_and_records_protocol_context() {
        let Some(python) = python() else {
            eprintln!("skipping experiment worker test: no interpreter found");
            return;
        };
        let directory = temp_dir("partial");
        let script = concat!(
            "import json, sys\n",
            "seen_turn = False\n",
            "for line in sys.stdin:\n",
            "    request = json.loads(line)\n",
            "    first = request['candidates'][0]['type']\n",
            "    if seen_turn and first == 'exchange_three':\n",
            "        print(json.dumps({'error': 'injected failure at game boundary'}), flush=True)\n",
            "        continue\n",
            "    if first != 'exchange_three':\n",
            "        seen_turn = True\n",
            "    print(json.dumps({'id': request['candidates'][0]['id']}), flush=True)\n",
        );
        let worker = directory.join("fake_worker.py");
        write_file(&worker, script);
        let mut config = config(&directory, "[8]", Mode::HeuristicOnly);
        config.python = PathBuf::from(python);
        config.worker = worker;

        let error = run(&config).unwrap_err();
        assert!(error.contains("injected failure"), "{error}");
        let records = read_lines(&config.output);
        assert_eq!(records.len(), 2, "one completed game, then one failure");
        let record: Value = serde_json::from_str(&records[0]).unwrap();
        assert_eq!(record["a_seats"], json!([0, 1]));
        assert_eq!(record["trace_hash"].as_str().unwrap().len(), 16);
        let failure: Value = serde_json::from_str(&records[1]).unwrap();
        assert!(failure.get("scores").is_none());
        assert_eq!(failure["a_seats"], json!([0, 2]));
        assert_eq!(failure["decision_index"], 0);
        assert_eq!(failure["context"]["stage"], "exchange");
        assert_eq!(failure["context"]["seat"], 0);
        assert!(failure["context"]["candidate_count"].as_u64().unwrap() > 0);
        assert!(failure["error"]
            .as_str()
            .unwrap()
            .contains("injected failure"));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn mixed_mode_replays_to_identical_records_across_processes() {
        let Some(python) = python() else {
            eprintln!("skipping experiment replay test: no interpreter found");
            return;
        };
        let directory = temp_dir("replay");
        let script = concat!(
            "import json, sys\n",
            "for line in sys.stdin:\n",
            "    request = json.loads(line)\n",
            "    print(json.dumps({'id': request['candidates'][0]['id']}), flush=True)\n",
        );
        let worker = directory.join("fake_worker.py");
        write_file(&worker, script);

        let mut first = config(&directory, "[9]", Mode::Mixed);
        first.python = PathBuf::from(python);
        first.worker = worker;
        first.output = directory.join("first.jsonl");
        let mut second = first.clone();
        second.output = directory.join("second.jsonl");

        let first_summary = run(&first).unwrap();
        let second_summary = run(&second).unwrap();
        assert_eq!(first_summary.games, 6);
        assert_eq!(second_summary.games, 6);
        let first_lines = read_lines(&first.output);
        let second_lines = read_lines(&second.output);
        assert_eq!(first_lines, second_lines);
        for (line, placement) in first_lines.iter().zip(PLACEMENTS) {
            let record: Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["a_seats"], json!([placement[0], placement[1]]));
            assert_eq!(record["trace_hash"].as_str().unwrap().len(), 16);
        }
        let _ = fs::remove_dir_all(&directory);
    }
}
