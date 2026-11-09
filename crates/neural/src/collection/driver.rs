//! One teacher episode through public Game/Policy/encoding interfaces.
//!
//! Accepted records are sent to the supplied fallible sink. This module knows
//! neither filesystem paths nor shard/manifest publication, so storage errors
//! stop play without substituting actions or retaining an episode in memory.

use sichuan_mahjong_sim::{Game, HeuristicPolicy, Policy, PLAYER_COUNT};
use std::time::Instant;

use crate::encoding::{encode, ENCODING_VERSION};

use super::error::{CollectionError, Result};
use super::format::{decision_record, episode_record, outcome_record, EpisodeId, ShardRecord};
use super::report::{elapsed_ns, CollectionTimings};
use super::seed::{seed_split, Seed};
use super::source::SourceRevision;
use super::stats::StatsAccumulator;

/// Runs a complete episode, returning its accepted decision count.
///
/// Statistics and timings belong to the current invocation. The caller only
/// publishes them if all requested shard commits succeed.
pub(super) fn collect_episode(
    seed: Seed,
    source_revision: &SourceRevision,
    decision_limit: u64,
    policies: &mut [HeuristicPolicy; PLAYER_COUNT],
    stats: &mut StatsAccumulator,
    timings: &mut CollectionTimings,
    mut write_record: impl FnMut(&ShardRecord) -> Result<()>,
) -> Result<u64> {
    let split = seed_split(seed);
    let id = EpisodeId {
        source_revision: source_revision.clone(),
        encoding_version: ENCODING_VERSION,
        seed,
    };
    let timer = Instant::now();
    write_record(&episode_record(&id, split))?;
    timings.io_ns += elapsed_ns(timer);

    let timer = Instant::now();
    let mut game = Game::new(seed.as_u64());
    timings.rules_ns += elapsed_ns(timer);

    let mut decisions = 0u64;
    let result = loop {
        if let Some(result) = game.result() {
            break *result;
        }
        if decisions >= decision_limit {
            return Err(CollectionError::DecisionLimit {
                seed,
                limit: decision_limit,
            });
        }

        let timer = Instant::now();
        let decision = game.next_decision();
        timings.rules_ns += elapsed_ns(timer);
        let Some(decision) = decision else {
            return Err(CollectionError::MissingResult { seed });
        };

        let timer = Instant::now();
        let encoded = encode(&decision);
        let state = encoded.state.flatten();
        let actions: Vec<_> = encoded.candidates.iter().map(|c| c.action).collect();
        let stage = decision.observation.stage;
        let actor = decision.seat;
        timings.encoding_ns += elapsed_ns(timer);

        let timer = Instant::now();
        let teacher = policies[actor.index()].choose(&decision);
        timings.teacher_ns += elapsed_ns(timer);

        let timer = Instant::now();
        let target =
            encoded
                .target_index(&teacher)
                .map_err(|_| CollectionError::TargetNotCandidate {
                    seed,
                    decision_index: decisions,
                    action: teacher,
                })?;
        timings.encoding_ns += elapsed_ns(timer);

        let timer = Instant::now();
        game.submit(actor, encoded.candidates[target].action)
            .map_err(|source| CollectionError::Game {
                seed,
                decision_index: decisions,
                seat: actor,
                source,
            })?;
        timings.rules_ns += elapsed_ns(timer);

        let timer = Instant::now();
        let record = decision_record(decisions, actor, stage, &state, &encoded, target);
        timings.encoding_ns += elapsed_ns(timer);
        let timer = Instant::now();
        write_record(&record)?;
        timings.io_ns += elapsed_ns(timer);
        stats.observe_decision(split, stage, &actions, target);
        decisions += 1;
    };

    let timer = Instant::now();
    write_record(&outcome_record(&result, decisions))?;
    timings.io_ns += elapsed_ns(timer);
    stats.observe_episode(split, &result);
    Ok(decisions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sichuan_mahjong_sim::HeuristicCache;

    fn teachers() -> [HeuristicPolicy; PLAYER_COUNT] {
        let cache = HeuristicCache::new(0).unwrap();
        std::array::from_fn(|_| HeuristicPolicy::with_cache(cache.clone()))
    }

    fn source() -> SourceRevision {
        SourceRevision::parse("0123456789abcdef0123456789abcdef01234567").unwrap()
    }

    #[test]
    fn episode_runs_against_an_in_memory_record_sink() {
        let mut policies = teachers();
        let mut stats = StatsAccumulator::new();
        let mut timings = CollectionTimings::default();
        let mut records = Vec::new();
        let count = collect_episode(
            Seed::from_u64(0),
            &source(),
            10_000,
            &mut policies,
            &mut stats,
            &mut timings,
            |record| {
                records.push(record.clone());
                Ok(())
            },
        )
        .unwrap();
        assert!(matches!(records.first(), Some(ShardRecord::Episode { .. })));
        assert_eq!(records.len() as u64, count + 2);
        for (index, record) in records[1..records.len() - 1].iter().enumerate() {
            assert!(
                matches!(record, ShardRecord::Decision { decision_index, .. }
                if *decision_index == index as u64)
            );
        }
        assert!(
            matches!(records.last(), Some(ShardRecord::Outcome { decisions, .. })
            if *decisions == count)
        );
        let stats = stats.finish();
        assert_eq!(stats.games, 1);
        assert_eq!(stats.decisions, count);
    }

    #[test]
    fn a_sink_failure_stops_the_episode_without_retry_or_substitution() {
        let mut policies = teachers();
        let mut stats = StatsAccumulator::new();
        let mut timings = CollectionTimings::default();
        let mut calls = 0;
        let error = collect_episode(
            Seed::from_u64(0),
            &source(),
            10_000,
            &mut policies,
            &mut stats,
            &mut timings,
            |record| {
                calls += 1;
                if matches!(record, ShardRecord::Decision { .. }) {
                    return Err(CollectionError::Io {
                        path: "test-sink".into(),
                        source: std::io::Error::other("injected failure"),
                    });
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(error, CollectionError::Io { path, .. }
            if path == std::path::Path::new("test-sink")));
        assert_eq!(
            calls, 2,
            "only the episode header and first decision are attempted"
        );
        let stats = stats.finish();
        assert_eq!(stats.games, 0);
        assert_eq!(stats.decisions, 0);
    }
}
