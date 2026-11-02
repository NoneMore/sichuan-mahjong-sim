//! Missing-suit-only variant of the deterministic baselines.
//!
//! This policy is a one-decision ablation: it behaves exactly like
//! [`FirstLegalPolicy`](super::FirstLegalPolicy) at every decision, except the
//! missing-suit declaration, where it uses the same fewest-tile rule as
//! [`HeuristicPolicy`](super::HeuristicPolicy). `MissingOnlyPolicy` and
//! `HeuristicPolicy` share the declaration rule and exchange selection, so
//! comparing the ablation with
//! the baseline isolates the declaration, while comparing it with the heuristic
//! measures the remaining heuristic play logic (discard ranking and the
//! peng/kong gates).
//!
//! The policy observes only [`Decision`], like every other policy. It has no RNG
//! and no history-dependent state.

use crate::action::Action;
use crate::game::{Decision, DecisionStage, LegalActions};

use super::heuristic::choose_missing_suit;
use super::{FirstLegalPolicy, Policy};

/// Deterministic ablation of [`FirstLegalPolicy`].
///
/// The missing-suit declaration reuses the heuristic rule: fewest own tiles in
/// the post-exchange hand, lowest suit index on a tie, and the first legal
/// action if that suit is not offered. Every other decision is delegated
/// unchanged to [`FirstLegalPolicy`]: deterministic exchange tiles from one
/// eligible suit, `Hu` when legal, and otherwise the first listed legal action.
#[derive(Clone, Copy, Debug, Default)]
pub struct MissingOnlyPolicy;

impl MissingOnlyPolicy {
    /// Creates the policy. The policy is stateless.
    pub const fn new() -> Self {
        Self
    }
}

impl Policy for MissingOnlyPolicy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action {
        if decision.observation.stage == DecisionStage::MissingSuit {
            if let LegalActions::Actions(actions) = &decision.legal_actions {
                return choose_missing_suit(decision.observation.own_hand, actions);
            }
        }

        // One authoritative baseline implementation serves both policies, so
        // the ablation cannot drift from the policy it is measured against.
        FirstLegalPolicy::new().choose(decision)
    }
}

#[cfg(test)]
mod tests;
