//! Warmup and measured phases together with their cache lifecycles.
//!
//! Cold runs use separate warmup and measured runners, dropping the warmup
//! context before measurement. Warm runs reuse one runner, its policies, and
//! its cache context across both phases.

use super::cache_tool::{CacheWarnings, SharedCache};
use super::{BenchError, Config, Lifecycle, Phases, Stage, WarmupPhase};
use sichuan_mahjong_sim::runner::Runner;

/// Runs the warmup (when enabled) and measured phases.
///
/// The runner factory receives the context that its policies must use. A cold
/// warmup gets a separate context that is dropped before the measured context
/// is created, so no table state survives into the measured batch. A warm
/// warmup and the measured batch reuse the same runner, policies, and context.
/// Phase errors carry the stage that failed and no measured aggregate is
/// returned.
pub(super) fn measure<F>(config: &Config, mut runner_for: F) -> Result<Phases, BenchError>
where
    F: FnMut(&SharedCache) -> Runner,
{
    let mut warnings = Vec::new();
    let (warmup, measured_cache, mut measured_runner, measured_cache_before, mut warner) = if config
        .lifecycle
        == Lifecycle::Warm
    {
        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let mut runner = runner_for(&cache);
        let batch = runner
            .run_batch(config.warmup_start(), config.warmup)
            .map_err(|error| BenchError::phase(Stage::Warmup, error))?;
        let after = cache.snapshot();
        let mut warner = CacheWarnings::default();
        if let Some(warning) = warner.observe(after, "benchmark warmup") {
            warnings.push(warning);
        }
        let phase = WarmupPhase {
            batch,
            cache_before: before,
            cache_after: after,
            scope: "inherited-context",
        };
        (Some(phase), cache, runner, after, warner)
    } else if config.warmup_enabled() {
        // Cold warmup: a separate context is dropped before measurement.
        let warmup_cache =
            SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let warmup_before = warmup_cache.snapshot();
        let mut warmup_runner = runner_for(&warmup_cache);
        let batch = warmup_runner
            .run_batch(config.warmup_start(), config.warmup)
            .map_err(|error| BenchError::phase(Stage::Warmup, error))?;
        let warmup_after = warmup_cache.snapshot();
        let mut warmup_warner = CacheWarnings::default();
        if let Some(warning) = warmup_warner.observe(warmup_after, "benchmark warmup") {
            warnings.push(warning);
        }
        drop(warmup_runner);
        drop(warmup_cache);

        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let runner = runner_for(&cache);
        let phase = WarmupPhase {
            batch,
            cache_before: warmup_before,
            cache_after: warmup_after,
            scope: "separate-context",
        };
        (Some(phase), cache, runner, before, CacheWarnings::default())
    } else {
        let cache = SharedCache::new(config.cache_budget_bytes).map_err(BenchError::CacheConfig)?;
        let before = cache.snapshot();
        let runner = runner_for(&cache);
        (None, cache, runner, before, CacheWarnings::default())
    };

    let measured = measured_runner
        .run_batch(config.seed, config.games)
        .map_err(|error| BenchError::phase(Stage::Measure, error))?;
    let measured_cache_after = measured_cache.snapshot();
    if let Some(warning) = warner.observe(measured_cache_after, "benchmark measured") {
        warnings.push(warning);
    }
    drop(measured_runner);
    drop(measured_cache);

    Ok(Phases {
        warmup,
        measured,
        measured_cache_before,
        measured_cache_after,
        warnings,
    })
}
