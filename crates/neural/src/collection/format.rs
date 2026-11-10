//! Strict v1 JSONL record format for complete teacher episodes.
//!
//! One shard concatenates episode blocks. Each block contains one `episode`
//! record, one `decision` record per accepted decision, and one `result`
//! record. The parser validates source, split, tensor shapes, finite values,
//! canonical candidate order, action parameters, target bounds, episode
//! completeness, counts, and the shard checksum.

use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeTuple, Serializer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sichuan_mahjong_sim::{Action, DecisionStage, EndReason, GameResult, Seat, Suit, Tile};

use crate::encoding::{
    action_kind_code, compare_actions, stage_code, stage_from_code, ActionFeatures, ACTION_WIDTH,
    ENCODING_VERSION, STATE_WIDTH,
};

use super::error::{io, CollectionError, Result};
use super::seed::{seed_split, Seed, Split};
use super::source::SourceRevision;

/// Stable identity of one episode: `(source_revision, encoding_version, seed)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeId {
    /// Compiled source revision.
    pub source_revision: SourceRevision,
    /// Encoding version.
    pub encoding_version: u32,
    /// Game seed.
    pub seed: Seed,
}

impl Serialize for EpisodeId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tuple = serializer.serialize_tuple(3)?;
        tuple.serialize_element(&self.source_revision)?;
        tuple.serialize_element(&self.encoding_version)?;
        tuple.serialize_element(&self.seed)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for EpisodeId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EpisodeIdVisitor;

        impl<'de> Visitor<'de> for EpisodeIdVisitor {
            type Value = EpisodeId;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a three-element [revision, encoding_version, seed] array")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let source_revision = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let encoding_version = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let seed = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(4, &self));
                }
                Ok(EpisodeId {
                    source_revision,
                    encoding_version,
                    seed,
                })
            }
        }

        deserializer.deserialize_tuple(3, EpisodeIdVisitor)
    }
}

/// One accepted decision sample.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionSample {
    /// Zero-based decision index within the episode.
    pub decision_index: u64,
    /// Seat that acted.
    pub actor_seat: Seat,
    /// Decision stage.
    pub stage: DecisionStage,
    /// 1013 pre-action state features.
    pub state: Vec<f32>,
    /// `N` rows of 39 candidate features, in canonical candidate order.
    pub candidate_features: Vec<Vec<f32>>,
    /// `N` canonical semantic actions, in the same order.
    pub candidates: Vec<Action>,
    /// Index of the teacher's canonical choice.
    pub target_index: usize,
}

impl DecisionSample {
    /// Number of candidate actions.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

/// One complete episode.
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// Episode identity.
    pub id: EpisodeId,
    /// Frozen seed split.
    pub split: Split,
    /// Accepted decision samples.
    pub decisions: Vec<DecisionSample>,
    /// Terminal game result.
    pub result: GameResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ShardRecord {
    Episode {
        episode_id: EpisodeId,
        split: Split,
    },
    Decision {
        decision_index: u64,
        actor_seat: u8,
        stage: u8,
        state: Vec<f32>,
        candidate_features: Vec<Vec<f32>>,
        candidate_actions: Vec<Vec<u8>>,
        teacher_target_index: usize,
    },
    #[serde(rename = "result")]
    Outcome {
        decisions: u64,
        scores: [i32; 4],
        winners: [bool; 4],
        end_reason: WireEndReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WireEndReason {
    ThreeWinners,
    WallExhausted,
}

impl From<EndReason> for WireEndReason {
    fn from(value: EndReason) -> Self {
        match value {
            EndReason::ThreeWinners => WireEndReason::ThreeWinners,
            EndReason::WallExhausted => WireEndReason::WallExhausted,
        }
    }
}

impl From<WireEndReason> for EndReason {
    fn from(value: WireEndReason) -> Self {
        match value {
            WireEndReason::ThreeWinners => EndReason::ThreeWinners,
            WireEndReason::WallExhausted => EndReason::WallExhausted,
        }
    }
}

pub(super) fn episode_record(id: &EpisodeId, split: Split) -> ShardRecord {
    ShardRecord::Episode {
        episode_id: id.clone(),
        split,
    }
}

pub(super) fn decision_record(
    decision_index: u64,
    actor_seat: Seat,
    stage: DecisionStage,
    state: &[f32],
    encoded: &crate::encoding::EncodedDecision,
    target: usize,
) -> ShardRecord {
    ShardRecord::Decision {
        decision_index,
        actor_seat: actor_seat.index() as u8,
        stage: stage_code(stage),
        state: state.to_vec(),
        candidate_features: encoded
            .candidates
            .iter()
            .map(|candidate| candidate.features.flatten().to_vec())
            .collect(),
        candidate_actions: encoded
            .candidates
            .iter()
            .map(|candidate| encode_action(candidate.action))
            .collect(),
        teacher_target_index: target,
    }
}

pub(super) fn outcome_record(result: &GameResult, decisions: u64) -> ShardRecord {
    ShardRecord::Outcome {
        decisions,
        scores: result.scores,
        winners: result.winners,
        end_reason: result.end_reason.into(),
    }
}

/// Parsed and validated shard checksum. Episodes are delivered incrementally.
pub(super) struct ParsedShard {
    pub sha256: String,
}

pub(super) struct ShardExpectations<'a> {
    pub source_revision: &'a SourceRevision,
    pub first_seed: Seed,
    pub games: u64,
    pub decisions: u64,
    pub decision_limit: u64,
    pub sha256: Option<&'a str>,
}

/// Reads and strictly validates one shard, delivering one complete episode at
/// a time. The callback must not publish results until this function succeeds:
/// the checksum and shard totals can only be verified at EOF. A callback error
/// stops reading immediately and is returned unchanged.
///
/// `sha256` is `None` only for a just-written temporary shard that has not been
/// committed yet; the computed checksum is still returned. Memory retained by
/// this reader is bounded by one episode, not by the number of games in a shard.
pub(super) fn read_shard_file(
    path: &Path,
    expected: ShardExpectations<'_>,
    mut on_episode: impl FnMut(Episode) -> Result<()>,
) -> Result<ParsedShard> {
    let ShardExpectations {
        source_revision,
        first_seed,
        games: expected_games,
        decisions: expected_decisions,
        decision_limit,
        sha256: expected_sha256,
    } = expected;
    let file = File::open(path).map_err(|error| io(path, error))?;
    let mut reader = BufReader::new(HashingReader {
        inner: file,
        hasher: Sha256::new(),
    });

    let mut next_seed = u128::from(first_seed.as_u64());
    let mut games = 0u64;
    let mut decisions = 0u64;
    let mut line_number = 0u64;
    let mut current: Option<EpisodeBuilder> = None;

    let mut line = String::new();
    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| CollectionError::ShardInvalid {
                path: path.to_owned(),
                line: line_number + 1,
                message: format!("not valid UTF-8 or I/O failure: {error}"),
            })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        let text = line.strip_suffix('\n').ok_or_else(|| {
            shard_invalid(
                path,
                line_number,
                "JSONL record is missing its final newline",
            )
        })?;
        let text = text.strip_suffix('\r').unwrap_or(text);
        if text.is_empty() {
            return Err(shard_invalid(path, line_number, "blank JSONL line"));
        }
        let record: ShardRecord =
            serde_json::from_str(text).map_err(|source| CollectionError::ShardJson {
                path: path.to_owned(),
                line: line_number,
                source,
            })?;

        match record {
            ShardRecord::Episode { episode_id, split } => {
                if current.is_some() {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "episode record inside an unfinished episode",
                    ));
                }
                if games >= expected_games {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "extra episode beyond manifest game count",
                    ));
                }
                if episode_id.source_revision != *source_revision {
                    return Err(CollectionError::ShardSourceMismatch {
                        path: path.to_owned(),
                        expected: source_revision.clone(),
                        found: episode_id.source_revision,
                    });
                }
                if episode_id.encoding_version != ENCODING_VERSION {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!(
                            "episode encoding version {}, expected {ENCODING_VERSION}",
                            episode_id.encoding_version
                        ),
                    ));
                }
                if u128::from(episode_id.seed.as_u64()) != next_seed {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!(
                            "episode seed {} is not the next contiguous seed {:016x}",
                            episode_id.seed, next_seed
                        ),
                    ));
                }
                let expected_split = seed_split(episode_id.seed);
                if split != expected_split {
                    return Err(CollectionError::ShardSplitMismatch {
                        path: path.to_owned(),
                        seed: episode_id.seed,
                        expected: expected_split,
                        found: split,
                    });
                }
                current = Some(EpisodeBuilder {
                    id: episode_id,
                    split,
                    decisions: Vec::new(),
                });
            }
            ShardRecord::Decision {
                decision_index,
                actor_seat,
                stage,
                state,
                candidate_features,
                candidate_actions,
                teacher_target_index,
            } => {
                let Some(episode) = current.as_mut() else {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "decision record outside an episode",
                    ));
                };
                if decision_index != episode.decisions.len() as u64 {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!(
                            "decision index {decision_index} is not the next index {}",
                            episode.decisions.len()
                        ),
                    ));
                }
                if episode.decisions.len() as u64 >= decision_limit {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!("episode exceeds decision limit {decision_limit}"),
                    ));
                }
                let sample = parse_decision(
                    path,
                    line_number,
                    decision_index,
                    actor_seat,
                    stage,
                    state,
                    candidate_features,
                    candidate_actions,
                    teacher_target_index,
                )?;
                decisions += 1;
                episode.decisions.push(sample);
            }
            ShardRecord::Outcome {
                decisions: recorded_decisions,
                scores,
                winners,
                end_reason,
            } => {
                let Some(episode) = current.take() else {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "result record without an episode",
                    ));
                };
                if recorded_decisions != episode.decisions.len() as u64 {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!(
                            "result records {recorded_decisions} decisions, episode has {}",
                            episode.decisions.len()
                        ),
                    ));
                }
                if episode.decisions.is_empty() {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "episode has no accepted decisions",
                    ));
                }
                let score_sum: i64 = scores.iter().map(|&score| i64::from(score)).sum();
                if score_sum != 0 {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        format!("terminal scores sum to {score_sum}, expected zero"),
                    ));
                }
                let result = GameResult {
                    scores,
                    winners,
                    end_reason: end_reason.into(),
                };
                let winner_count = winners.iter().filter(|&&won| won).count();
                if (result.end_reason == EndReason::ThreeWinners && winner_count != 3)
                    || (result.end_reason == EndReason::WallExhausted && winner_count >= 3)
                {
                    return Err(shard_invalid(
                        path,
                        line_number,
                        "terminal winner count is inconsistent with end reason",
                    ));
                }
                games += 1;
                next_seed += 1;
                on_episode(Episode {
                    id: episode.id,
                    split: episode.split,
                    decisions: episode.decisions,
                    result,
                })?;
            }
        }
    }

    if let Some(unfinished) = current {
        return Err(shard_invalid(
            path,
            line_number,
            format!(
                "unfinished episode {} is missing its result record",
                unfinished.id.seed
            ),
        ));
    }

    let hashing_reader = reader.into_inner();
    let digest = hashing_reader.hasher.finalize();
    let sha256 = lower_hex(&digest);
    if let Some(expected) = expected_sha256 {
        if sha256 != expected {
            return Err(CollectionError::ShardChecksumMismatch {
                path: path.to_owned(),
                expected: expected.to_owned(),
                found: sha256,
            });
        }
    }
    if games != expected_games {
        return Err(CollectionError::ShardMetadataMismatch {
            path: path.to_owned(),
            field: "games",
            expected: expected_games,
            found: games,
        });
    }
    if decisions != expected_decisions {
        return Err(CollectionError::ShardMetadataMismatch {
            path: path.to_owned(),
            field: "decisions",
            expected: expected_decisions,
            found: decisions,
        });
    }

    Ok(ParsedShard { sha256 })
}

struct EpisodeBuilder {
    id: EpisodeId,
    split: Split,
    decisions: Vec<DecisionSample>,
}

#[allow(clippy::too_many_arguments)]
fn parse_decision(
    path: &Path,
    line: u64,
    decision_index: u64,
    actor_seat: u8,
    stage: u8,
    state: Vec<f32>,
    candidate_features: Vec<Vec<f32>>,
    candidate_actions: Vec<Vec<u8>>,
    target_index: usize,
) -> Result<DecisionSample> {
    let actor_seat = Seat::new(usize::from(actor_seat)).ok_or_else(|| {
        shard_invalid(path, line, format!("actor seat {actor_seat} outside 0..3"))
    })?;
    let stage = stage_from_id(stage)
        .ok_or_else(|| shard_invalid(path, line, format!("unknown stage id {stage}")))?;

    if state.len() != STATE_WIDTH {
        return Err(shard_invalid(
            path,
            line,
            format!("state has {} values, expected {STATE_WIDTH}", state.len()),
        ));
    }
    if state.iter().any(|value| !value.is_finite()) {
        return Err(CollectionError::NonFiniteFeature {
            path: path.to_owned(),
            line,
            field: "state",
        });
    }
    if candidate_actions.is_empty() {
        return Err(shard_invalid(path, line, "candidate list is empty"));
    }
    if candidate_features.len() != candidate_actions.len() {
        return Err(shard_invalid(
            path,
            line,
            format!(
                "{} candidate feature rows for {} actions",
                candidate_features.len(),
                candidate_actions.len()
            ),
        ));
    }
    if target_index >= candidate_actions.len() {
        return Err(CollectionError::ShardTargetOutOfRange {
            path: path.to_owned(),
            line,
            target: target_index,
            candidates: candidate_actions.len(),
        });
    }

    let mut candidates = Vec::with_capacity(candidate_actions.len());
    let mut previous: Option<Action> = None;
    for (index, (features, wire)) in candidate_features
        .iter()
        .zip(candidate_actions.iter())
        .enumerate()
    {
        if features.len() != ACTION_WIDTH {
            return Err(shard_invalid(
                path,
                line,
                format!(
                    "candidate {index} has {} feature values, expected {ACTION_WIDTH}",
                    features.len()
                ),
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(CollectionError::NonFiniteFeature {
                path: path.to_owned(),
                line,
                field: "candidate_features",
            });
        }
        let action =
            decode_action(wire).map_err(|message| CollectionError::ShardActionInvalid {
                path: path.to_owned(),
                line,
                message: format!("candidate {index}: {message}"),
            })?;
        validate_action_features(features, action).map_err(|message| {
            CollectionError::ShardActionInvalid {
                path: path.to_owned(),
                line,
                message: format!("candidate {index}: {message}"),
            }
        })?;
        if let Some(previous) = previous {
            if compare_actions(&previous, &action) != Ordering::Less {
                return Err(shard_invalid(
                    path,
                    line,
                    format!("candidate {index} is not in strict canonical order"),
                ));
            }
        }
        previous = Some(action);
        candidates.push(action);
    }

    Ok(DecisionSample {
        decision_index,
        actor_seat,
        stage,
        state,
        candidate_features,
        candidates,
        target_index,
    })
}

fn shard_invalid(path: &Path, line: u64, message: impl Into<String>) -> CollectionError {
    CollectionError::ShardInvalid {
        path: path.to_owned(),
        line,
        message: message.into(),
    }
}

struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        Ok(read)
    }
}

pub(super) fn lower_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).expect("hex digit"));
        out.push(char::from_digit(u32::from(byte & 0xf), 16).expect("hex digit"));
    }
    out
}

/// Stage numeric ID in encoding version 1.
pub fn stage_id(stage: DecisionStage) -> u8 {
    stage_code(stage)
}

/// Reverses [`stage_id`].
pub fn stage_from_id(id: u8) -> Option<DecisionStage> {
    stage_from_code(id)
}

/// Action kind numeric ID in encoding version 1.
pub fn action_kind_id(action: &Action) -> u8 {
    action_kind_code(action)
}

/// Stable action kind name in kind-ID order.
pub const ACTION_KIND_NAMES: [&str; 9] = [
    "exchange",
    "choose_missing",
    "discard",
    "peng",
    "open_kong",
    "concealed_kong",
    "added_kong",
    "hu",
    "pass",
];

/// Wire form `[kind_id, ...parameters]` with v1 parameter IDs.
pub fn encode_action(action: Action) -> Vec<u8> {
    let kind = action_kind_code(&action);
    match action {
        Action::Exchange(tiles) => {
            let mut tiles = tiles.map(|tile| tile.index());
            tiles.sort_unstable();
            vec![kind, tiles[0], tiles[1], tiles[2]]
        }
        Action::ChooseMissing(suit) => vec![kind, suit.index() as u8],
        Action::Discard(tile) | Action::ConcealedKong(tile) | Action::AddedKong(tile) => {
            vec![kind, tile.index()]
        }
        Action::Peng | Action::OpenKong | Action::Hu | Action::Pass => vec![kind],
    }
}

/// Decodes and canonicalizes a wire action.
pub fn decode_action(wire: &[u8]) -> std::result::Result<Action, String> {
    let (&kind, parameters) = wire
        .split_first()
        .ok_or_else(|| "empty action array".to_owned())?;
    match kind {
        0 => {
            if parameters.len() != 3 {
                return Err(format!(
                    "Exchange requires three tile parameters, got {}",
                    parameters.len()
                ));
            }
            let mut tiles = [Tile::ALL[0]; 3];
            for (slot, &value) in tiles.iter_mut().zip(parameters) {
                *slot = decode_tile(value)?;
            }
            if !(tiles[0] <= tiles[1] && tiles[1] <= tiles[2]) {
                return Err("Exchange tiles are not sorted in canonical order".to_owned());
            }
            if !(tiles[0].suit() == tiles[1].suit() && tiles[1].suit() == tiles[2].suit()) {
                return Err("Exchange tiles do not share one suit".to_owned());
            }
            Ok(Action::Exchange(tiles))
        }
        1 => {
            expect_parameters(kind, parameters, 1)?;
            let suit = Suit::from_index(usize::from(parameters[0]))
                .ok_or_else(|| format!("suit id {} outside 0..2", parameters[0]))?;
            Ok(Action::ChooseMissing(suit))
        }
        2 | 5 | 6 => {
            expect_parameters(kind, parameters, 1)?;
            let tile = decode_tile(parameters[0])?;
            Ok(match kind {
                2 => Action::Discard(tile),
                5 => Action::ConcealedKong(tile),
                _ => Action::AddedKong(tile),
            })
        }
        3 => no_parameters(kind, parameters, Action::Peng),
        4 => no_parameters(kind, parameters, Action::OpenKong),
        7 => no_parameters(kind, parameters, Action::Hu),
        8 => no_parameters(kind, parameters, Action::Pass),
        _ => Err(format!("unknown action kind id {kind}")),
    }
}

fn expect_parameters(
    kind: u8,
    parameters: &[u8],
    expected: usize,
) -> std::result::Result<(), String> {
    if parameters.len() != expected {
        return Err(format!(
            "action kind {kind} requires {expected} parameter(s), got {}",
            parameters.len()
        ));
    }
    Ok(())
}

fn no_parameters(
    kind: u8,
    parameters: &[u8],
    action: Action,
) -> std::result::Result<Action, String> {
    expect_parameters(kind, parameters, 0)?;
    Ok(action)
}

fn decode_tile(value: u8) -> std::result::Result<Tile, String> {
    Tile::from_index(value).ok_or_else(|| format!("tile id {value} outside 0..26"))
}

fn validate_action_features(features: &[f32], action: Action) -> std::result::Result<(), String> {
    let expected = ActionFeatures::new(action).flatten();
    for (index, &value) in features.iter().enumerate() {
        if value.to_bits() != expected[index].to_bits() {
            return Err(format!(
                "candidate feature {index} is {value}, expected {}",
                expected[index]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sichuan_mahjong_sim::{FirstLegalPolicy, Game, Policy};

    #[test]
    fn action_wire_round_trips_and_kind_slots_match_ids() {
        let mut policy = FirstLegalPolicy::new();
        let mut seen_kinds = [false; 9];
        for seed in 0..50u64 {
            let mut game = Game::new(seed);
            let mut decisions = 0;
            loop {
                if game.result().is_some() {
                    break;
                }
                let Some(decision) = game.next_decision() else {
                    break;
                };
                let encoded = crate::encoding::encode(&decision);
                let stage_slot = usize::from(stage_id(decision.observation.stage));
                assert_eq!(encoded.state.context.stage[stage_slot], 1.0);
                for candidate in &encoded.candidates {
                    let wire = encode_action(candidate.action);
                    assert_eq!(decode_action(&wire), Ok(candidate.action));
                    let kind = usize::from(action_kind_id(&candidate.action));
                    seen_kinds[kind] = true;
                    assert_eq!(candidate.features.kind[kind], 1.0);
                }
                let action = policy.choose(&decision);
                let target = encoded
                    .target_index(&action)
                    .expect("baseline action is a candidate");
                let seat = decision.seat;
                game.submit(seat, encoded.candidates[target].action)
                    .expect("candidate accepted");
                decisions += 1;
                assert!(decisions < 10_000);
            }
            assert!(decisions > 0);
        }
        assert_eq!(seen_kinds, [true; 9], "expected every v1 action kind");
    }

    #[test]
    fn wire_validation_rejects_malformed_actions() {
        assert_eq!(decode_action(&[]), Err("empty action array".to_owned()));
        assert!(decode_action(&[9]).is_err());
        assert!(decode_action(&[2]).is_err());
        assert!(decode_action(&[2, 27]).is_err());
        assert!(decode_action(&[1, 3]).is_err());
        assert!(decode_action(&[0, 3, 2, 1]).is_err());
        assert!(decode_action(&[0, 0, 0, 9]).is_err());
    }

    #[test]
    fn stage_ids_are_frozen_v1() {
        let stages = [
            (DecisionStage::Exchange, 0),
            (DecisionStage::MissingSuit, 1),
            (DecisionStage::Turn, 2),
            (DecisionStage::DiscardResponse, 3),
            (DecisionStage::AddedKongResponse, 4),
        ];
        for (stage, id) in stages {
            assert_eq!(stage_id(stage), id);
            assert_eq!(stage_from_id(id), Some(stage));
        }
        assert_eq!(stage_from_id(5), None);
    }

    #[test]
    fn decision_parsing_enforces_canonical_order_and_exact_features() {
        let path = Path::new("shard.jsonl");
        let state = vec![0.0f32; STATE_WIDTH];
        let actions = [
            Action::ChooseMissing(Suit::Tiao),
            Action::Discard(Tile::from_index(26).expect("tile")),
            Action::Pass,
        ];
        let features: Vec<Vec<f32>> = actions
            .iter()
            .map(|&action| ActionFeatures::new(action).flatten().to_vec())
            .collect();

        let parse = |actions: &[Action], features: &[Vec<f32>], target| {
            let wire: Vec<Vec<u8>> = actions
                .iter()
                .map(|&action| encode_action(action))
                .collect();
            parse_decision(
                path,
                1,
                0,
                0,
                stage_code(DecisionStage::Turn),
                state.clone(),
                features.to_vec(),
                wire,
                target,
            )
        };

        parse(&actions, &features, 1).expect("canonical sample");
        assert!(matches!(
            parse(
                &[actions[1], actions[0]],
                &[features[1].clone(), features[0].clone()],
                0
            ),
            Err(CollectionError::ShardInvalid { .. })
        ));
        assert!(matches!(
            parse(
                &[actions[2], actions[2]],
                &[features[2].clone(), features[2].clone()],
                0
            ),
            Err(CollectionError::ShardInvalid { .. })
        ));

        let mut corrupted = features.clone();
        corrupted[0][9] += 0.25;
        assert!(matches!(
            parse(&actions, &corrupted, 0),
            Err(CollectionError::ShardActionInvalid { .. })
        ));
    }
}
