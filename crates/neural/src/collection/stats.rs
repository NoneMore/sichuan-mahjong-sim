//! Corpus statistics, including zero-support kinds and forced-decision
//! separation.
//!
//! Every action kind appears once in the report, even with zero opportunities.
//! `primary_*` fields exclude single-candidate decisions, which carry no
//! choice-learning signal; the all-decision fields retain them for trace and
//! coverage checks.

use serde::Serialize;
use sichuan_mahjong_sim::{Action, DecisionStage, EndReason, GameResult};

use super::format::{action_kind_id, stage_id, ACTION_KIND_NAMES};
use super::seed::Split;

/// Per-stage decision counts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StageStats {
    /// Numeric v1 stage ID.
    pub stage_id: u8,
    /// Stable lowercase stage name.
    pub stage: String,
    /// All decisions in this stage.
    pub decisions: u64,
    /// Stage decisions with more than one candidate.
    pub multi_candidate: u64,
}

/// Per-kind opportunity and teacher-selection counts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KindStats {
    /// Numeric v1 action kind ID.
    pub kind_id: u8,
    /// Stable lowercase action kind name.
    pub kind: String,
    /// Decisions where the kind appeared at least once.
    pub opportunities: u64,
    /// Decisions where the teacher target had this kind.
    pub selections: u64,
    /// Multi-candidate decisions where the kind appeared at least once.
    pub primary_opportunities: u64,
    /// Multi-candidate decisions where the teacher target had this kind.
    pub primary_selections: u64,
}

/// Per-split episode and decision counts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SplitStats {
    /// `train`, `validation`, or `test`.
    pub split: String,
    /// Complete episodes in the split.
    pub games: u64,
    /// Decision records in the split.
    pub decisions: u64,
}

/// Aggregate corpus or invocation statistics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CorpusStats {
    /// Complete games (episodes).
    pub games: u64,
    /// Accepted decisions.
    pub decisions: u64,
    /// Sum of candidate counts over all decisions.
    pub candidates: u64,
    /// Decisions with exactly one candidate.
    pub forced_decisions: u64,
    /// Decisions with more than one candidate.
    pub multi_candidate_decisions: u64,
    /// Smallest candidate count observed, or zero with no decisions.
    pub min_candidates: usize,
    /// Largest candidate count observed, or zero with no decisions.
    pub max_candidates: usize,
    /// Per-stage totals, indexed by v1 stage ID.
    pub stage: Vec<StageStats>,
    /// Per-kind totals in v1 action kind order, including zero support.
    pub kind: Vec<KindStats>,
    /// Per-split totals in train, validation, test order.
    pub split: Vec<SplitStats>,
    /// Games ending with three winners.
    pub three_winners_games: u64,
    /// Games ending by wall exhaustion.
    pub wall_exhausted_games: u64,
}

impl CorpusStats {
    /// Kinds with zero all-decision opportunities.
    pub fn zero_support_kinds(&self) -> impl Iterator<Item = &KindStats> {
        self.kind.iter().filter(|kind| kind.opportunities == 0)
    }

    /// Kinds with zero opportunities among non-forced decisions.
    pub fn primary_zero_support_kinds(&self) -> impl Iterator<Item = &KindStats> {
        self.kind
            .iter()
            .filter(|kind| kind.primary_opportunities == 0)
    }
}

const STAGE_NAMES: [&str; 5] = [
    "exchange",
    "missing_suit",
    "turn",
    "discard_response",
    "added_kong_response",
];

/// Mutable accumulator shared by streaming collection and report reading.
#[derive(Clone, Debug)]
pub(super) struct StatsAccumulator {
    games: u64,
    decisions: u64,
    candidates: u64,
    forced_decisions: u64,
    multi_candidate_decisions: u64,
    min_candidates: usize,
    max_candidates: usize,
    stage_decisions: [u64; 5],
    stage_multi: [u64; 5],
    kind_opportunities: [u64; 9],
    kind_selections: [u64; 9],
    primary_kind_opportunities: [u64; 9],
    primary_kind_selections: [u64; 9],
    split_games: [u64; 3],
    split_decisions: [u64; 3],
    three_winners_games: u64,
    wall_exhausted_games: u64,
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsAccumulator {
    pub(super) fn new() -> Self {
        Self {
            games: 0,
            decisions: 0,
            candidates: 0,
            forced_decisions: 0,
            multi_candidate_decisions: 0,
            min_candidates: usize::MAX,
            max_candidates: 0,
            stage_decisions: [0; 5],
            stage_multi: [0; 5],
            kind_opportunities: [0; 9],
            kind_selections: [0; 9],
            primary_kind_opportunities: [0; 9],
            primary_kind_selections: [0; 9],
            split_games: [0; 3],
            split_decisions: [0; 3],
            three_winners_games: 0,
            wall_exhausted_games: 0,
        }
    }

    pub(super) fn observe_decision(
        &mut self,
        split: Split,
        stage: DecisionStage,
        candidates: &[Action],
        target_index: usize,
    ) {
        let count = candidates.len();
        debug_assert!(count > 0);
        debug_assert!(target_index < count);
        let stage = usize::from(stage_id(stage));
        let target_kind = usize::from(action_kind_id(&candidates[target_index]));

        self.decisions += 1;
        self.candidates += count as u64;
        self.split_decisions[split.index()] += 1;
        self.stage_decisions[stage] += 1;
        self.min_candidates = self.min_candidates.min(count);
        self.max_candidates = self.max_candidates.max(count);

        let primary = count > 1;
        if primary {
            self.multi_candidate_decisions += 1;
            self.stage_multi[stage] += 1;
        } else {
            self.forced_decisions += 1;
        }

        let mut present = [false; 9];
        for action in candidates {
            present[usize::from(action_kind_id(action))] = true;
        }
        for (kind, &seen) in present.iter().enumerate() {
            if seen {
                self.kind_opportunities[kind] += 1;
                if primary {
                    self.primary_kind_opportunities[kind] += 1;
                }
            }
        }
        self.kind_selections[target_kind] += 1;
        if primary {
            self.primary_kind_selections[target_kind] += 1;
        }
    }

    pub(super) fn observe_episode(&mut self, split: Split, result: &GameResult) {
        self.games += 1;
        self.split_games[split.index()] += 1;
        match result.end_reason {
            EndReason::ThreeWinners => self.three_winners_games += 1,
            EndReason::WallExhausted => self.wall_exhausted_games += 1,
        }
    }

    pub(super) fn finish(self) -> CorpusStats {
        let min_candidates = if self.decisions == 0 {
            0
        } else {
            self.min_candidates
        };
        CorpusStats {
            games: self.games,
            decisions: self.decisions,
            candidates: self.candidates,
            forced_decisions: self.forced_decisions,
            multi_candidate_decisions: self.multi_candidate_decisions,
            min_candidates,
            max_candidates: self.max_candidates,
            stage: (0..5)
                .map(|index| StageStats {
                    stage_id: index as u8,
                    stage: STAGE_NAMES[index].to_owned(),
                    decisions: self.stage_decisions[index],
                    multi_candidate: self.stage_multi[index],
                })
                .collect(),
            kind: (0..9)
                .map(|index| KindStats {
                    kind_id: index as u8,
                    kind: ACTION_KIND_NAMES[index].to_owned(),
                    opportunities: self.kind_opportunities[index],
                    selections: self.kind_selections[index],
                    primary_opportunities: self.primary_kind_opportunities[index],
                    primary_selections: self.primary_kind_selections[index],
                })
                .collect(),
            split: Split::ALL
                .iter()
                .map(|split| SplitStats {
                    split: split.as_str().to_owned(),
                    games: self.split_games[split.index()],
                    decisions: self.split_decisions[split.index()],
                })
                .collect(),
            three_winners_games: self.three_winners_games,
            wall_exhausted_games: self.wall_exhausted_games,
        }
    }
}
