//! Reproducible fixed-baseline self-play and throughput measurement.
//!
//! The command line is deliberately small: a game count, one seed, and the
//! seat policy name. The seed plays two roles by design: it seeds the game
//! sequence and, for the random policy, the per-seat policy streams. It drives
//! seeded games through the runner, which owns all rule interaction; this
//! binary only parses arguments and formats the terminal aggregate.

use std::env;
use std::process::ExitCode;

use sichuan_mahjong_sim::runner::{BatchReport, Runner};

const USAGE: &str = "\
Usage: sichuan-mahjong-sim [--games N] [--seed S] [--policy first|random]

Runs a fixed-baseline Sichuan mahjong self-play batch and reports zero-sum
terminal scores, finish reasons, and separate policy/rules timings.

Options:
  --games N        Number of games to play (default 1; must be at least 1).
  --seed S         Seed for the first game and, with --policy random, for the
                   seat policy streams (default 0). Game i uses wrapping S + i.
  --policy NAME    Seat policy for every seat: first (default) or random.
  -h, --help       Print this help and exit; arguments after it are ignored.

-h/--help takes precedence: once it is parsed, everything after it is ignored.
Arguments before it are still validated.
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolicyKind {
    First,
    Random,
}

impl PolicyKind {
    fn name(self) -> &'static str {
        match self {
            PolicyKind::First => "first",
            PolicyKind::Random => "random",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunConfig {
    games: u64,
    seed: u64,
    policy: PolicyKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Help,
    Run(RunConfig),
}

/// Parses arguments after the executable name.
///
/// `-h`/`--help` short-circuits as soon as it is parsed: it returns
/// [`Command::Help`] and ignores every remaining argument, which is the usual
/// CLI convention. Arguments before it are validated normally.
fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut games = None;
    let mut seed = None;
    let mut policy = None;
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
                let parsed: u64 = value
                    .parse()
                    .map_err(|_| format!("invalid --games value '{value}': expected an integer"))?;
                if parsed == 0 {
                    return Err("--games must be at least 1".to_owned());
                }
                games = Some(parsed);
            }
            "--seed" => {
                if seed.is_some() {
                    return Err("duplicate --seed".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_owned())?;
                let parsed: u64 = value
                    .parse()
                    .map_err(|_| format!("invalid --seed value '{value}': expected an integer"))?;
                seed = Some(parsed);
            }
            "--policy" => {
                if policy.is_some() {
                    return Err("duplicate --policy".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--policy requires a value".to_owned())?;
                let parsed = if value.eq_ignore_ascii_case("first") {
                    PolicyKind::First
                } else if value.eq_ignore_ascii_case("random") {
                    PolicyKind::Random
                } else {
                    return Err(format!(
                        "invalid --policy value '{value}': expected 'first' or 'random'"
                    ));
                };
                policy = Some(parsed);
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    Ok(Command::Run(RunConfig {
        games: games.unwrap_or(1),
        seed: seed.unwrap_or(0),
        policy: policy.unwrap_or(PolicyKind::First),
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

    let mut runner = match config.policy {
        PolicyKind::First => Runner::first_legal(),
        PolicyKind::Random => Runner::random(config.seed),
    };

    let report = match runner.run_batch(config.seed, config.games) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };

    print_report(&config, &report);
    ExitCode::SUCCESS
}

fn print_report(config: &RunConfig, report: &BatchReport) {
    println!(
        "fixed-baseline self-play: policy={} seed={} games={}",
        config.policy.name(),
        config.seed,
        report.games
    );
    println!(
        "finish reasons: three_winners={} wall_exhausted={}",
        report.three_winners, report.wall_exhausted
    );
    println!(
        "decisions: {} ({:.1} per game)",
        report.decisions,
        report.decisions as f64 / report.games as f64
    );
    println!(
        "terminal score totals: {}",
        format_scores(&report.score_totals)
    );
    println!(
        "mean score per game: {}",
        format_means(&report.mean_scores())
    );
    println!("wins by seat: {:?}", report.winners);
    println!(
        "time: total={:.3}s rules={:.3}s policy={:.3}s",
        report.total_time.as_secs_f64(),
        report.rules_time.as_secs_f64(),
        report.policy_time.as_secs_f64()
    );
    println!(
        "throughput: {:.2} games/sec, {:.2} decisions/sec",
        report.games_per_sec(),
        report.decisions_per_sec()
    );
}

fn format_scores(scores: &[i64; 4]) -> String {
    format!(
        "[{}, {}, {}, {}]",
        scores[0], scores[1], scores[2], scores[3]
    )
}

fn format_means(means: &[f64; 4]) -> String {
    format!(
        "[{:.2}, {:.2}, {:.2}, {:.2}]",
        means[0], means[1], means[2], means[3]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn defaults_are_one_seeded_first_legal_game() {
        assert_eq!(
            parse_args(args(&[])).unwrap(),
            Command::Run(RunConfig {
                games: 1,
                seed: 0,
                policy: PolicyKind::First,
            })
        );
    }

    #[test]
    fn all_flags_are_parsed() {
        assert_eq!(
            parse_args(args(&[
                "--games", "12", "--seed", "99", "--policy", "random"
            ]))
            .unwrap(),
            Command::Run(RunConfig {
                games: 12,
                seed: 99,
                policy: PolicyKind::Random,
            })
        );
        assert_eq!(
            parse_args(args(&["--policy", "FIRST"])).unwrap(),
            Command::Run(RunConfig {
                games: 1,
                seed: 0,
                policy: PolicyKind::First,
            })
        );
    }

    #[test]
    fn help_does_not_run_a_game() {
        assert_eq!(parse_args(args(&["--help"])).unwrap(), Command::Help);
        assert_eq!(parse_args(args(&["-h"])).unwrap(), Command::Help);
    }

    #[test]
    fn help_ignores_arguments_that_follow_it() {
        assert_eq!(
            parse_args(args(&["--help", "--games", "0"])).unwrap(),
            Command::Help
        );
        assert_eq!(
            parse_args(args(&["-h", "--seed", "abc", "extra"])).unwrap(),
            Command::Help
        );
    }

    #[test]
    fn zero_games_is_rejected() {
        let error = parse_args(args(&["--games", "0"])).unwrap_err();
        assert!(error.contains("--games"));
    }

    #[test]
    fn invalid_values_and_unknown_arguments_are_rejected() {
        assert!(parse_args(args(&["--seed", "abc"])).is_err());
        assert!(parse_args(args(&["--policy", "search"])).is_err());
        assert!(parse_args(args(&["--games"])).is_err());
        assert!(parse_args(args(&["--games", "2", "--games", "3"])).is_err());
        assert!(parse_args(args(&["extra"])).is_err());
    }
}
