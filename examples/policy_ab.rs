//! One-off reproducible mixed-policy A/B experiment driver.
//!
//! This example is experiment tooling, not simulator library API. It compares
//! two Python-heuristic seats (A) with two `FirstLegalPolicy` seats (B) over
//! the six fixed A placements of preselected seeds. For every game it starts a
//! fresh `Game::new(seed)`, drives the public `next_decision` / `submit`
//! surface, and writes one JSON line with seed, placement, terminal scores,
//! end reason, decision count, and a deterministic `trace_hash`.
//!
//! The external Python worker is a one-off dependency of the experiment, not a
//! runtime dependency of the simulator. The library crate remains
//! dependency-free; the example uses dev-dependencies for JSON only.
//!
//! See `specs/active/reproducible-policy-ab/design.md` for intent and
//! `experiments/policy_ab/README.md` for the surrounding workflow.

// The example crate root cannot host nested module directories, so the
// experiment modules live under `examples/policy_ab/` via explicit paths.
#[path = "policy_ab/adapter.rs"]
mod adapter;
#[path = "policy_ab/driver.rs"]
mod driver;
#[path = "policy_ab/experiment.rs"]
mod experiment;
#[path = "policy_ab/worker.rs"]
mod worker;

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use driver::Mode;
use experiment::Config;

const USAGE: &str = "\
Usage: cargo run --example policy_ab -- --seeds PATH --output PATH \\
           --python EXE --worker PATH --ninone-root PATH \\
           --mode mixed|first-only|heuristic-only --decision-limit N

Runs the one-off mixed-policy A/B experiment. For each distinct game seed and
each of the six fixed A placements, one fresh game is played and one JSON line
is appended per completed game.

Required:
  --seeds PATH          JSON array of distinct u64 game seeds (at least one).
  --output PATH         Per-game JSONL output; the file must not already exist.
  --python EXE          Python interpreter for the heuristic worker.
  --worker PATH         Worker script speaking the JSON Lines protocol.
  --ninone-root PATH    Repository root passed to the worker as --ninone-root.
  --mode MODE           mixed | first-only | heuristic-only.
  --decision-limit N    Per-game safety bound on decisions (at least 1).

Modes:
  mixed                 A seats (the placement pair) use the Python heuristic,
                        the other two seats use FirstLegalPolicy.
  first-only            All four seats use FirstLegalPolicy.
  heuristic-only        All four seats use the Python heuristic.

Output:
  PATH holds one JSON object per completed game, in seed order and then the
  fixed placement order [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]. Fields:
  seed, a_seats, scores, end_reason, decisions, trace_hash.

Failures:
  The first failure stops the run with a nonzero exit status and appends one
  failure record (error, seed, a_seats, decision_index, context) to PATH,
  after the games that already completed. A failure record has no scores and
  an incomplete run is never reported as a normal result. Its context carries
  the deciding seat, the decision stage, the candidate count, and the exact
  minimal adapter request (hand, missing_suit, candidates) when available.

Environment:
  POLICY_AB_WORKER_TIMEOUT_SECS  Whole request/response timeout in whole seconds
                                (default 60). Never affects game results.

  -h, --help            Print this help and exit; arguments after it are ignored.
";

#[derive(Debug)]
enum Command {
    Help,
    Run(Config),
}

fn next_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    let value = arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    if value.is_empty() {
        return Err(format!("{flag} requires a non-empty value"));
    }
    Ok(value)
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("duplicate {flag}"));
    }
    *slot = Some(value);
    Ok(())
}

/// Parses arguments after the executable name.
///
/// `-h`/`--help` short-circuits: it returns [`Command::Help`] and ignores every
/// remaining argument. Earlier arguments are still validated.
fn parse_args<I>(arguments: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let mut seeds: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut python: Option<PathBuf> = None;
    let mut worker: Option<PathBuf> = None;
    let mut ninone_root: Option<PathBuf> = None;
    let mut mode: Option<Mode> = None;
    let mut decision_limit: Option<u64> = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--seeds" => {
                let value = next_value(&mut arguments, "--seeds")?;
                set_once(&mut seeds, "--seeds", PathBuf::from(value))?;
            }
            "--output" => {
                let value = next_value(&mut arguments, "--output")?;
                set_once(&mut output, "--output", PathBuf::from(value))?;
            }
            "--python" => {
                let value = next_value(&mut arguments, "--python")?;
                set_once(&mut python, "--python", PathBuf::from(value))?;
            }
            "--worker" => {
                let value = next_value(&mut arguments, "--worker")?;
                set_once(&mut worker, "--worker", PathBuf::from(value))?;
            }
            "--ninone-root" => {
                let value = next_value(&mut arguments, "--ninone-root")?;
                set_once(&mut ninone_root, "--ninone-root", PathBuf::from(value))?;
            }
            "--mode" => {
                let value = next_value(&mut arguments, "--mode")?;
                let parsed = Mode::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --mode value '{value}': expected mixed, first-only, or heuristic-only"
                    )
                })?;
                set_once(&mut mode, "--mode", parsed)?;
            }
            "--decision-limit" => {
                let value = next_value(&mut arguments, "--decision-limit")?;
                let parsed: u64 = value.parse().map_err(|_| {
                    format!("invalid --decision-limit value '{value}': expected a positive integer")
                })?;
                if parsed == 0 {
                    return Err("--decision-limit must be at least 1".to_owned());
                }
                set_once(&mut decision_limit, "--decision-limit", parsed)?;
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    for (flag, present) in [
        ("--seeds", seeds.is_some()),
        ("--output", output.is_some()),
        ("--python", python.is_some()),
        ("--worker", worker.is_some()),
        ("--ninone-root", ninone_root.is_some()),
        ("--mode", mode.is_some()),
        ("--decision-limit", decision_limit.is_some()),
    ] {
        if !present {
            return Err(format!("missing required {flag}"));
        }
    }

    Ok(Command::Run(Config {
        seeds: seeds.expect("checked above"),
        output: output.expect("checked above"),
        python: python.expect("checked above"),
        worker: worker.expect("checked above"),
        ninone_root: ninone_root.expect("checked above"),
        mode: mode.expect("checked above"),
        decision_limit: decision_limit.expect("checked above"),
    }))
}

fn main() -> ExitCode {
    let command = match parse_args(env::args().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!();
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let config = match command {
        Command::Help => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Command::Run(config) => config,
    };

    match experiment::run(&config) {
        Ok(summary) => {
            println!(
                "policy_ab: wrote {} game results for {} seed(s) to {}",
                summary.games,
                summary.seeds,
                summary.output.display()
            );
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn full_args() -> Vec<String> {
        args(&[
            "--seeds",
            "seeds.json",
            "--output",
            "out.jsonl",
            "--python",
            "python",
            "--worker",
            "worker.py",
            "--ninone-root",
            "ninone",
            "--mode",
            "mixed",
            "--decision-limit",
            "500",
        ])
    }

    #[test]
    fn full_command_line_is_parsed() {
        match parse_args(full_args()).unwrap() {
            Command::Run(config) => {
                assert_eq!(config.seeds, PathBuf::from("seeds.json"));
                assert_eq!(config.output, PathBuf::from("out.jsonl"));
                assert_eq!(config.python, PathBuf::from("python"));
                assert_eq!(config.worker, PathBuf::from("worker.py"));
                assert_eq!(config.ninone_root, PathBuf::from("ninone"));
                assert_eq!(config.mode, Mode::Mixed);
                assert_eq!(config.decision_limit, 500);
            }
            Command::Help => panic!("expected a run command"),
        }
    }

    #[test]
    fn help_takes_precedence_and_ignores_later_arguments() {
        assert!(matches!(
            parse_args(args(&["--help", "--mode", "bogus"])).unwrap(),
            Command::Help
        ));
        assert!(matches!(parse_args(args(&["-h"])).unwrap(), Command::Help));
    }

    #[test]
    fn every_flag_is_required() {
        for flag in [
            "--seeds",
            "--output",
            "--python",
            "--worker",
            "--ninone-root",
            "--mode",
            "--decision-limit",
        ] {
            let mut values = full_args();
            let position = values.iter().position(|value| value == flag).unwrap();
            values.drain(position..position + 2);
            let error = parse_args(values).unwrap_err();
            assert!(error.contains(flag), "missing {flag} reported as {error}");
        }
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(parse_args(args(&["--mode", "selfplay"])).is_err());
        assert!(parse_args(args(&["--decision-limit", "0"])).is_err());
        assert!(parse_args(args(&["--decision-limit", "x"])).is_err());
        assert!(parse_args(args(&["--seeds"])).is_err());
        assert!(parse_args(args(&["--seeds", ""])).is_err());
        assert!(parse_args(args(&["unknown"])).is_err());

        let mut duplicated = full_args();
        duplicated.extend(args(&["--mode", "mixed"]));
        let error = parse_args(duplicated).unwrap_err();
        assert!(error.contains("duplicate --mode"), "{error}");
    }
}
