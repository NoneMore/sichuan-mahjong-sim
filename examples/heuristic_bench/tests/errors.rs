//! Phase-failure and report-output error coverage.

use super::*;

struct RejectingPolicy;

impl Policy for RejectingPolicy {
    fn choose(
        &mut self,
        _decision: &sichuan_mahjong_sim::Decision<'_>,
    ) -> sichuan_mahjong_sim::Action {
        // Pass is never a legal exchange, so the runner must reject it.
        sichuan_mahjong_sim::Action::Pass
    }
}

/// Runner whose policies always return an illegal action.
fn rejecting_runner(_cache: &SharedCache) -> Runner {
    Runner::new(std::array::from_fn(|_| {
        Box::new(RejectingPolicy) as Box<dyn Policy>
    }))
}

#[test]
fn cache_config_errors_are_reported_before_play_and_write_no_report() {
    // A one-byte non-zero budget is too small for any table; the library owns
    // the exact minimum and reports a configuration error at creation.
    let workload = config_with_budget(1, 5, 0, Lifecycle::Cold, 1);
    let error = measure(&workload, heuristic_runner).unwrap_err();
    assert!(
        matches!(error, BenchError::CacheConfig(_)),
        "expected a cache configuration error: {error}"
    );
    assert!(
        error.to_string().contains("invalid cache budget"),
        "{error}"
    );
    assert!(error.source().is_some());

    let mut buffer = Vec::new();
    let error = execute(&workload, heuristic_runner, &mut buffer).unwrap_err();
    assert!(matches!(error, BenchError::CacheConfig(_)));
    assert!(buffer.is_empty(), "no report may be written on failure");
}

#[test]
fn warmup_and_measured_failures_are_reported_by_stage() {
    let error = measure(&config(1, 5, 1), rejecting_runner).unwrap_err();
    assert_eq!(
        error.to_string().split(" failed:").next(),
        Some("warmup phase")
    );
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Warmup,
            ..
        }
    ));

    let error = measure(&config(1, 5, 0), rejecting_runner).unwrap_err();
    assert_eq!(
        error.to_string().split(" failed:").next(),
        Some("measured phase")
    );
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Measure,
            ..
        }
    ));

    // A healthy cold warmup does not hide a measured-phase failure.
    let mut factory_calls = 0;
    let error = measure(&config(1, 5, 1), |cache| {
        factory_calls += 1;
        if factory_calls == 1 {
            heuristic_runner(cache)
        } else {
            rejecting_runner(cache)
        }
    })
    .unwrap_err();
    assert!(matches!(
        error,
        BenchError::Phase {
            stage: Stage::Measure,
            ..
        }
    ));
}

#[test]
fn a_failed_phase_writes_no_report() {
    for warmup in [0, 1] {
        let mut buffer = Vec::new();
        let error = execute(&config(1, 5, warmup), rejecting_runner, &mut buffer).unwrap_err();
        assert!(buffer.is_empty(), "no summary may be written on failure");
        assert!(matches!(error, BenchError::Phase { .. }));
    }
}

#[test]
fn decision_limit_exhaustion_invalidates_each_phase_without_a_report() {
    for (warmup, expected_stage) in [(0, Stage::Measure), (1, Stage::Warmup)] {
        let mut buffer = Vec::new();
        let error = execute(
            &config(1, 5, warmup),
            |cache| heuristic_runner(cache).with_decision_limit(0),
            &mut buffer,
        )
        .unwrap_err();
        assert!(buffer.is_empty(), "no summary may be written on failure");
        assert!(matches!(
            error,
            BenchError::Phase {
                stage,
                source: RunnerError::DecisionLimit { .. },
            } if stage == expected_stage
        ));
        assert!(
            error.source().is_some(),
            "underlying error must be retained"
        );
    }
}

struct FailingWriter {
    write_error: bool,
}

impl Write for FailingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.write_error {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "sink closed"))
        } else {
            Ok(buffer.len())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "sink closed"))
    }
}

#[test]
fn report_write_and_flush_failures_are_output_errors() {
    for write_error in [true, false] {
        let error = execute(
            &config(1, 5, 0),
            heuristic_runner,
            &mut FailingWriter { write_error },
        )
        .unwrap_err();
        assert!(
            matches!(error, BenchError::Output(_)),
            "write_error {write_error}: {error}"
        );
        assert!(error.to_string().contains("sink closed"), "{error}");
    }
}
