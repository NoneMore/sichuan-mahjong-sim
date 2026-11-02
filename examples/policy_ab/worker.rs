//! Long-lived JSON Lines worker transport and strict response parsing.
//!
//! The experiment talks to one Python process for its whole run: one JSON
//! request per line on stdin, one JSON response per line on stdout. The worker
//! is imported heuristic code, not a simulator dependency; nothing here is
//! exposed by the library.
//!
//! The transport is deliberately strict. A response must be exactly one JSON
//! object containing exactly one of:
//!
//! - `{"id": "<candidate id>"}`, or
//! - `{"error": "<message>"}`.
//!
//! Unknown fields, both keys, neither key, non-string values, empty strings,
//! malformed JSON, a closed stdout, a dead process, and a response timeout are
//! all failures. There is no retry, fallback action, or resynchronisation.
//!
//! One dedicated IO thread owns both pipes and handles strictly one exchange at
//! a time: write the request line, flush, then read the matching response line.
//! The caller waits on a channel because the standard library has no portable
//! read timeout, so its timeout covers writing the request as well as reading
//! the response. On timeout the child is killed, which unblocks the IO thread;
//! a hung worker can never block the experiment forever.

use std::ffi::OsString;
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Default wait for one worker request/response exchange and for worker
/// shutdown.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Environment variable that overrides [`DEFAULT_TIMEOUT`] for the experiment.
///
/// It is an operational escape hatch only: it never affects game results or
/// the per-game `trace_hash`. Invalid or zero values fall back to the default.
pub const TIMEOUT_ENV: &str = "POLICY_AB_WORKER_TIMEOUT_SECS";

/// Transport, protocol, and worker-reported failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcError {
    /// The response line violated the wire format.
    Protocol(String),
    /// The worker itself reported an error.
    Worker(String),
    /// No response arrived within the configured timeout.
    Timeout(Duration),
    /// The process could not be written to, read from, or waited for.
    Transport(String),
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpcError::Protocol(message) => write!(f, "worker protocol error: {message}"),
            RpcError::Worker(message) => write!(f, "worker reported an error: {message}"),
            RpcError::Timeout(timeout) => {
                write!(f, "worker did not answer within {timeout:?}")
            }
            RpcError::Transport(message) => write!(f, "worker transport error: {message}"),
        }
    }
}

impl std::error::Error for RpcError {}

/// One request/response exchange with a long-lived worker.
pub trait Rpc {
    /// Sends one request line and returns the raw response line without its
    /// trailing newline.
    fn request(&mut self, line: &str) -> Result<String, RpcError>;
}

/// Parses and validates one worker response line.
///
/// Trailing `\r\n` is tolerated; anything else outside the documented shape is
/// rejected.
pub fn parse_response(line: &str) -> Result<String, RpcError> {
    let line = line.trim_end_matches(&['\r', '\n'][..]);
    if line.trim().is_empty() {
        return Err(RpcError::Protocol("empty response line".to_owned()));
    }
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|error| RpcError::Protocol(format!("response is not valid JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| RpcError::Protocol("response must be a JSON object".to_owned()))?;
    for key in object.keys() {
        if key != "id" && key != "error" {
            return Err(RpcError::Protocol(format!(
                "response contains unknown field '{key}'"
            )));
        }
    }
    match (object.get("id"), object.get("error")) {
        (Some(id), None) => {
            let id = id
                .as_str()
                .ok_or_else(|| RpcError::Protocol("response 'id' must be a string".to_owned()))?;
            if id.is_empty() {
                return Err(RpcError::Protocol(
                    "response 'id' must not be empty".to_owned(),
                ));
            }
            Ok(id.to_owned())
        }
        (None, Some(error)) => {
            let message = error.as_str().ok_or_else(|| {
                RpcError::Protocol("response 'error' must be a string".to_owned())
            })?;
            if message.is_empty() {
                return Err(RpcError::Protocol(
                    "response 'error' must not be empty".to_owned(),
                ));
            }
            Err(RpcError::Worker(message.to_owned()))
        }
        (Some(_), Some(_)) => Err(RpcError::Protocol(
            "response must contain either 'id' or 'error', not both".to_owned(),
        )),
        (None, None) => Err(RpcError::Protocol(
            "response must contain 'id' or 'error'".to_owned(),
        )),
    }
}

/// How to start the worker process.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    program: OsString,
    arguments: Vec<OsString>,
}

impl WorkerCommand {
    /// Starts building a command for `program`.
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            arguments: Vec::new(),
        }
    }

    /// Appends one argument.
    pub fn arg(mut self, argument: impl Into<OsString>) -> Self {
        self.arguments.push(argument.into());
        self
    }

    fn label(&self) -> String {
        self.program.to_string_lossy().into_owned()
    }

    fn into_command(self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.arguments);
        // stdout must be a pure response stream; diagnostics go to stderr.
        command.env("PYTHONUNBUFFERED", "1");
        command.env("PYTHONIOENCODING", "utf-8");
        command
    }
}

/// Reads [`TIMEOUT_ENV`] seconds, or returns [`DEFAULT_TIMEOUT`].
pub fn timeout_from_env() -> Duration {
    parse_timeout(std::env::var(TIMEOUT_ENV).ok().as_deref())
}

fn parse_timeout(value: Option<&str>) -> Duration {
    value
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TIMEOUT)
}

/// Serves requests on one long-lived thread that owns both pipes.
///
/// Each request is written and flushed before the matching response is read, so
/// the caller's timeout covers the whole exchange. A worker that never reads
/// stdin therefore cannot wedge the caller: the write blocks here, the caller
/// times out, and killing the child makes the blocked write fail.
fn serve(
    requests: Receiver<String>,
    mut stdin: ChildStdin,
    stdout: ChildStdout,
    responses: Sender<Result<Option<String>, RpcError>>,
) {
    let mut reader = BufReader::new(stdout);
    while let Ok(line) = requests.recv() {
        let written = stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush());
        if let Err(error) = written {
            let _ = responses.send(Err(RpcError::Transport(format!(
                "writing worker request failed: {error}"
            ))));
            break;
        }
        let mut response = String::new();
        match reader.read_line(&mut response) {
            Ok(0) => {
                let _ = responses.send(Ok(None));
                break;
            }
            Ok(_) => {
                if responses.send(Ok(Some(response))).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = responses.send(Err(RpcError::Transport(format!(
                    "reading worker output failed: {error}"
                ))));
                break;
            }
        }
    }
    // Dropping `stdin` here closes the worker's input so it can exit on EOF.
}

/// A spawned, long-lived worker with piped stdin/stdout.
pub struct WorkerProcess {
    child: Child,
    requests: Option<Sender<String>>,
    responses: Receiver<Result<Option<String>, RpcError>>,
    timeout: Duration,
}

impl WorkerProcess {
    /// Spawns the worker with an explicit response/shutdown timeout.
    pub fn spawn_with_timeout(command: WorkerCommand, timeout: Duration) -> Result<Self, String> {
        let label = command.label();
        let mut process = command.into_command();
        process
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = process
            .spawn()
            .map_err(|error| format!("failed to start worker '{label}': {error}"))?;
        let stdin = child.stdin.take().expect("worker stdin is piped");
        let stdout = child.stdout.take().expect("worker stdout is piped");

        let (request_sender, request_receiver) = mpsc::channel();
        let (response_sender, responses) = mpsc::channel();
        let io = thread::Builder::new()
            .name("policy_ab_worker_io".to_owned())
            .spawn(move || serve(request_receiver, stdin, stdout, response_sender));
        if let Err(error) = io {
            // The child already exists, so a failure to create its IO thread
            // must not leak a running process.
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("failed to start worker IO thread: {error}"));
        }

        Ok(Self {
            child,
            requests: Some(request_sender),
            responses,
            timeout,
        })
    }

    /// Sends one request line and returns the response line.
    ///
    /// The timeout applies to the whole exchange: writing the request and
    /// reading its response.
    pub fn request(&mut self, line: &str) -> Result<String, RpcError> {
        let requests = self
            .requests
            .as_ref()
            .ok_or_else(|| RpcError::Transport("worker requests are already closed".to_owned()))?;
        requests
            .send(line.to_owned())
            .map_err(|_| RpcError::Transport("worker request channel is closed".to_owned()))?;

        match self.responses.recv_timeout(self.timeout) {
            Ok(Ok(Some(line))) => Ok(line.trim_end_matches(&['\r', '\n'][..]).to_owned()),
            Ok(Ok(None)) => Err(self.exited("worker closed its output")),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => {
                // A stuck request invalidates the run: stop the process instead
                // of leaving an orphan that could still write to the pipe.
                // Killing it also unblocks an IO thread stuck writing to a
                // worker that is not reading stdin.
                self.requests = None;
                let _ = self.child.kill();
                Err(RpcError::Timeout(self.timeout))
            }
            Err(RecvTimeoutError::Disconnected) => Err(self.exited("worker IO thread stopped")),
        }
    }

    /// Closes worker stdin and requires a zero exit status.
    ///
    /// The wait is bounded by the same timeout; a worker that ignores EOF is
    /// killed and reported as a timeout.
    pub fn shutdown(&mut self) -> Result<(), RpcError> {
        // Dropping the request channel lets the IO thread close worker stdin
        // after the last exchange, so the worker sees EOF.
        self.requests = None;
        let deadline = Instant::now() + self.timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) if status.success() => return Ok(()),
                Ok(Some(status)) => {
                    return Err(RpcError::Transport(format!(
                        "worker exited with {status} after shutdown"
                    )))
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        return Err(RpcError::Timeout(self.timeout));
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    return Err(RpcError::Transport(format!(
                        "waiting for worker failed: {error}"
                    )))
                }
            }
        }
    }

    fn exited(&mut self, reason: &str) -> RpcError {
        // Reader EOF can precede observable process termination. Poll briefly
        // so the failure record carries the exit status instead of racing it.
        let deadline = Instant::now() + self.timeout.min(Duration::from_secs(2));
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    return RpcError::Transport(format!("{reason}; worker exited with {status}"))
                }
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                Ok(None) => {
                    return RpcError::Transport(format!("{reason}; worker is still running"))
                }
                Err(error) => {
                    return RpcError::Transport(format!(
                        "{reason}; inspecting worker failed: {error}"
                    ))
                }
            }
        }
    }
}

impl Rpc for WorkerProcess {
    fn request(&mut self, line: &str) -> Result<String, RpcError> {
        WorkerProcess::request(self, line)
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.requests = None;
        // Only an already-reaped child may skip the kill. A failed try_wait
        // leaves an unknown child state, so terminate before the blocking reap
        // instead of waiting forever for a live or unobservable process.
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn spawn_script(script: &str, timeout: Duration) -> WorkerProcess {
        let program = python().expect("caller checked for a python interpreter");
        WorkerProcess::spawn_with_timeout(
            WorkerCommand::new(program).arg("-c").arg(script),
            timeout,
        )
        .expect("fake worker spawns")
    }

    #[test]
    fn response_parser_accepts_only_the_documented_shapes() {
        assert_eq!(
            parse_response(r#"{"id":"discard:m1"}"#).unwrap(),
            "discard:m1"
        );
        assert_eq!(
            parse_response("{\"id\":\"win\"}\r\n").unwrap(),
            "win",
            "a trailing CRLF belongs to the line, not the JSON"
        );
        assert_eq!(
            parse_response(r#"{"error":"boom"}"#).unwrap_err(),
            RpcError::Worker("boom".to_owned())
        );

        for line in [
            "",
            "   ",
            "not json",
            "[]",
            r#"{"id":1}"#,
            r#"{"id":""}"#,
            r#"{"error":7}"#,
            r#"{"error":""}"#,
            r#"{"id":"win","error":"boom"}"#,
            r#"{"id":"win","extra":true}"#,
            r#"{}"#,
        ] {
            let error = parse_response(line).unwrap_err();
            assert!(
                matches!(error, RpcError::Protocol(_)),
                "line {line:?} should be a protocol error, got {error:?}"
            );
        }
    }

    #[test]
    fn timeout_override_uses_only_positive_whole_seconds() {
        assert_eq!(parse_timeout(Some("5")), Duration::from_secs(5));
        assert_eq!(parse_timeout(Some("0")), DEFAULT_TIMEOUT);
        assert_eq!(parse_timeout(Some("1.5")), DEFAULT_TIMEOUT);
        assert_eq!(parse_timeout(Some("abc")), DEFAULT_TIMEOUT);
        assert_eq!(parse_timeout(None), DEFAULT_TIMEOUT);
    }

    #[test]
    fn python_worker_round_trip_and_clean_shutdown() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        let script = concat!(
            "import json, sys\n",
            "for line in sys.stdin:\n",
            "    request = json.loads(line)\n",
            "    print(json.dumps({'id': request['candidates'][0]['id']}), flush=True)\n",
        );
        let mut worker = spawn_script(script, Duration::from_secs(10));
        let response = worker
            .request(
                r#"{"hand":["m1"],"missing_suit":null,"candidates":[{"id":"win","type":"win"}]}"#,
            )
            .unwrap();
        assert_eq!(parse_response(&response).unwrap(), "win");
        worker.shutdown().unwrap();
        drop(worker);
    }

    #[test]
    fn python_worker_malformed_response_is_rejected() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        let script = concat!(
            "import sys\n",
            "for line in sys.stdin:\n",
            "    print('not-json', flush=True)\n",
        );
        let mut worker = spawn_script(script, Duration::from_secs(10));
        let response = worker.request("{}").unwrap();
        let error = parse_response(&response).unwrap_err();
        assert!(
            matches!(&error, RpcError::Protocol(message) if message.contains("not valid JSON")),
            "unexpected error: {error:?}"
        );
        worker.shutdown().unwrap();
    }

    #[test]
    fn python_worker_exit_is_reported_with_its_status() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        let script = "import sys\nfor line in sys.stdin:\n    sys.exit(3)\n";
        let mut worker = spawn_script(script, Duration::from_secs(10));
        let error = worker.request("{}").unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains('3'),
            "exit status should be reported, got {message}"
        );
    }

    #[test]
    fn python_worker_timeout_kills_the_process() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        let script = "import sys, time\nfor line in sys.stdin:\n    time.sleep(30)\n";
        let timeout = Duration::from_millis(200);
        let mut worker = spawn_script(script, timeout);
        let started = Instant::now();
        let error = worker.request("{}").unwrap_err();
        assert_eq!(error, RpcError::Timeout(timeout));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout must not wait for the scripted sleep"
        );
    }

    #[test]
    fn python_worker_that_ignores_a_large_request_times_out_and_is_killed() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        // The worker never reads stdin, so the request write stays blocked as
        // soon as the pipe buffer is full; one MiB exceeds it on every
        // supported platform.
        let script = "import time\ntime.sleep(60)\n";
        let timeout = Duration::from_millis(300);
        let mut worker = spawn_script(script, timeout);
        let request = format!(r#"{{"pad":"{}"}}"#, "x".repeat(1024 * 1024));
        let started = Instant::now();
        let error = worker.request(&request).unwrap_err();
        assert_eq!(error, RpcError::Timeout(timeout));
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the blocked write must not outlive the timeout"
        );
        // The timeout kills the child, so the failure must not leave an
        // orphan behind.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match worker.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                other => panic!("worker survived the timeout kill: {other:?}"),
            }
        }
    }

    #[test]
    fn python_worker_nonzero_shutdown_status_is_reported() {
        let Some(_) = python() else {
            eprintln!("skipping python worker test: no interpreter found");
            return;
        };
        let script = concat!(
            "import json, sys\n",
            "for line in sys.stdin:\n",
            "    request = json.loads(line)\n",
            "    print(json.dumps({'id': request['candidates'][0]['id']}), flush=True)\n",
            "    sys.exit(4)\n",
        );
        let mut worker = spawn_script(script, Duration::from_secs(10));
        let response = worker.request(r#"{"candidates":[{"id":"win"}]}"#).unwrap();
        assert_eq!(parse_response(&response).unwrap(), "win");
        let error = worker.shutdown().unwrap_err();
        assert!(
            error.to_string().contains('4'),
            "shutdown should report the exit status, got {error}"
        );
    }

    #[test]
    fn missing_executable_is_a_spawn_error() {
        let error = match WorkerProcess::spawn_with_timeout(
            WorkerCommand::new("policy-ab-no-such-worker-executable"),
            Duration::from_millis(100),
        ) {
            Ok(_) => panic!("a missing executable must not spawn a worker"),
            Err(error) => error,
        };
        assert!(
            error.contains("failed to start worker"),
            "unexpected error: {error}"
        );
    }
}
