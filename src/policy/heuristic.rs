//! Deterministic shanten-and-ukeire policy.
//!
//! The policy observes only [`Decision`], never the authoritative game or an
//! opponent's concealed tiles. It has no RNG and no history-dependent logical
//! state: all memoization lives inside one `choose` call.

use crate::action::{Action, Meld, MeldKind};
use crate::game::{Decision, DecisionStage, LegalActions, Observation};
use crate::hand::Hand;
use crate::rules;
use crate::tile::{Suit, Tile, TILE_KINDS};

use super::analysis::{self, DistanceEvaluator};
use super::Policy;

/// Deterministic policy that ranks legal post-discard hands by exact target
/// deficit (shanten), then unseen improving copies (ukeire), then a tenpai fan
/// tie-break.
///
/// Decision rules match the shanten-and-ukeire policy specification:
///
/// - exchanges use the deterministic baseline choice;
/// - the missing suit is the fewest-tile suit after the exchange, lowest suit
///   index on a tie;
/// - a legal `Hu` is taken immediately and a sole legal action is returned
///   without analysis;
/// - while missing-suit tiles remain, the lowest legal missing-suit discard is
///   chosen without analysis;
/// - a peng must strictly lower shanten;
/// - a kong must have a `(S, -U)` rank no worse than the ordinary alternative;
/// - on a response with both peng and open kong eligible, the better progress
///   rank wins and the kong wins an exact tie.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicPolicy;

impl HeuristicPolicy {
    /// Creates the policy. The policy is stateless.
    pub const fn new() -> Self {
        Self
    }
}

impl Policy for HeuristicPolicy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action {
        match &decision.legal_actions {
            LegalActions::Exchange { eligible_suits } => {
                super::choose_exchange_tiles(decision.observation.own_hand, *eligible_suits)
            }
            LegalActions::Actions(actions) => {
                // The missing suit is not public while the four declarations
                // are being collected, so this stage is handled first.
                if decision.observation.stage == DecisionStage::MissingSuit {
                    return choose_missing_suit(decision.observation.own_hand, actions);
                }

                let Some(missing) =
                    decision.observation.players[decision.seat.index()].missing_suit
                else {
                    return super::preferred_action(actions);
                };

                match decision.observation.stage {
                    DecisionStage::Turn => choose_turn(decision, actions, missing),
                    DecisionStage::DiscardResponse => {
                        choose_discard_response(decision, actions, missing)
                    }
                    DecisionStage::AddedKongResponse => super::preferred_action(actions),
                    DecisionStage::Exchange | DecisionStage::MissingSuit => {
                        super::preferred_action(actions)
                    }
                }
            }
        }
    }
}

/// Fewest-tile suit in the post-exchange hand, lowest suit index on a tie.
///
/// Shared with [`super::MissingOnlyPolicy`], which uses exactly this rule for
/// the missing-suit declaration and the deterministic baseline for every other
/// decision. Falls back to the first legal action when the computed suit is not
/// offered, which keeps the helper total for impossible legal lists.
pub(super) fn choose_missing_suit(hand: &Hand, actions: &[Action]) -> Action {
    let mut best_suit = Suit::ALL[0];
    let mut best_count = usize::MAX;
    for suit in Suit::ALL {
        let count = hand.suit_count(suit);
        if count < best_count {
            best_count = count;
            best_suit = suit;
        }
    }
    let action = Action::ChooseMissing(best_suit);
    if actions.contains(&action) {
        action
    } else {
        super::preferred_action(actions)
    }
}

/// Lexicographic progress rank: lower shanten, then higher ukeire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Progress {
    shanten: i32,
    ukeire: u32,
}

impl Progress {
    fn better_than(self, other: Self) -> bool {
        self.shanten < other.shanten
            || (self.shanten == other.shanten && self.ukeire > other.ukeire)
    }

    fn no_worse_than(self, other: Self) -> bool {
        self.shanten < other.shanten
            || (self.shanten == other.shanten && self.ukeire >= other.ukeire)
    }
}

/// Counts public physical ownership across melds and unincorporated discards.
fn base_public(observation: &Observation<'_>) -> [u8; TILE_KINDS] {
    let mut copies = [0u8; TILE_KINDS];

    // Every declared meld owns its physical tiles. This includes the deciding
    // player's melds exactly once.
    for player in observation.players {
        for meld in player.melds {
            let index = meld.tile().index() as usize;
            copies[index] = copies[index].saturating_add(meld.tile_count());
        }
    }

    // A meld-claimed discard is physically part of the claimant's meld, so it
    // is omitted here. Every other record is one physical tile regardless of
    // how many Hu winners reference it.
    for discard in observation.discards {
        if discard.meld_claim.is_none() {
            let index = discard.tile.index() as usize;
            copies[index] = copies[index].saturating_add(1);
        }
    }

    copies
}

/// One scored ordinary discard.
struct ScoredDiscard {
    tile: Tile,
    hand: Hand,
    deficit: u8,
    ukeire: u32,
    fan: u8,
}

/// Ranks candidate discards from `hand` (one tile larger than a post-discard
/// hand) and returns the best one with its progress.
///
/// `public_before_discard` is ownership before the candidate discard; the
/// discarded tile is added to public ownership for that candidate's ukeire
/// count. `with_fan` enables the tenpai-only fan tie-break.
fn rank_discards(
    hand: &Hand,
    melds: &[Meld],
    missing: Suit,
    public_before_discard: &[u8; TILE_KINDS],
    candidates: &[Tile],
    with_fan: bool,
) -> Option<(Tile, Progress)> {
    let mut evaluator = DistanceEvaluator::new(melds, missing)?;

    let mut scored = Vec::with_capacity(candidates.len());
    for &tile in candidates {
        let mut after = *hand;
        if after.remove(tile).is_err() {
            continue;
        }
        let deficit = evaluator.deficit(&after);
        scored.push(ScoredDiscard {
            tile,
            hand: after,
            deficit,
            ukeire: 0,
            fan: 0,
        });
    }
    if scored.is_empty() {
        return None;
    }

    let best_deficit = scored
        .iter()
        .map(|candidate| candidate.deficit)
        .min()
        .expect("a scored discard exists");

    // Only minimum-deficit candidates need availability analysis.
    let mut best_ukeire = 0u32;
    for candidate in scored.iter_mut().filter(|c| c.deficit == best_deficit) {
        let mut public = *public_before_discard;
        let index = candidate.tile.index() as usize;
        public[index] = public[index].saturating_add(1);
        candidate.ukeire =
            analysis::ukeire(&mut evaluator, &candidate.hand, &public, candidate.deficit);
        best_ukeire = best_ukeire.max(candidate.ukeire);
    }

    let tenpai = best_deficit == 1;
    let finalists: Vec<usize> = (0..scored.len())
        .filter(|&index| {
            scored[index].deficit == best_deficit && scored[index].ukeire == best_ukeire
        })
        .collect();

    // The fan tie-break is only needed when more than one tenpai candidate is
    // still tied on the two primary criteria.
    let best_index = if with_fan && tenpai && finalists.len() > 1 {
        let mut best = finalists[0];
        for &index in &finalists {
            scored[index].fan = analysis::max_discard_fan(&scored[index].hand, melds, missing);
            if scored[index].fan > scored[best].fan {
                best = index;
            }
        }
        best
    } else {
        finalists[0]
    };
    Some((
        scored[best_index].tile,
        Progress {
            shanten: best_deficit as i32 - 1,
            ukeire: best_ukeire,
        },
    ))
}

/// A turn: take Hu, short-circuit forced choices, then compare kongs with the
/// best ordinary discard.
fn choose_turn(decision: &Decision<'_>, actions: &[Action], missing: Suit) -> Action {
    if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
        return hu;
    }
    if actions.len() == 1 {
        return actions[0];
    }

    let observation = &decision.observation;
    let own_hand = observation.own_hand;
    let own_melds = observation.own_melds;

    // Forced missing-suit clearing needs no structural analysis: every legal
    // discard preserves the same non-missing-suit hand.
    if rules::has_missing_suit_tiles(own_hand, own_melds, missing) {
        let lowest = actions
            .iter()
            .filter_map(|action| match action {
                Action::Discard(tile) => Some(*tile),
                _ => None,
            })
            .min();
        return match lowest {
            Some(tile) => Action::Discard(tile),
            None => super::preferred_action(actions),
        };
    }

    let public = base_public(observation);
    let discard_tiles: Vec<Tile> = actions
        .iter()
        .filter_map(|action| match action {
            Action::Discard(tile) => Some(*tile),
            _ => None,
        })
        .collect();

    let Some((best_discard, alternative)) =
        rank_discards(own_hand, own_melds, missing, &public, &discard_tiles, true)
    else {
        return super::preferred_action(actions);
    };

    let active_opponents = observation
        .players
        .iter()
        .filter(|player| player.seat != decision.seat && !player.has_won)
        .count();

    let mut choices = Vec::new();
    for action in actions {
        let (kind, hand, melds, adjusted_public) =
            match kong_position(*action, own_hand, own_melds, &public) {
                Some(position) => position,
                None => continue,
            };
        let Some(mut evaluator) = DistanceEvaluator::new(&melds, missing) else {
            continue;
        };
        let deficit = evaluator.deficit(&hand);
        let progress = Progress {
            shanten: deficit as i32 - 1,
            ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted_public, deficit),
        };
        choices.push(KongChoice {
            action: *action,
            progress,
            income: rules::kong_payment(kind, active_opponents).total,
        });
    }

    match select_kong(alternative, &choices) {
        Some(action) => action,
        None => Action::Discard(best_discard),
    }
}

/// One legal kong with its structural progress and nominal income.
#[derive(Clone, Copy, Debug)]
struct KongChoice {
    action: Action,
    progress: Progress,
    income: u32,
}

/// Applies the kong gate and prefers an eligible kong to the ordinary action.
///
/// Eligibility is `(S, -U)` no worse than the alternative: a lower shanten may
/// justify lower ukeire, but equal shanten with lower ukeire fails. Among
/// eligible kongs, better progress wins, then larger nominal income, then the
/// earlier action in the fixed legal list.
fn select_kong(alternative: Progress, choices: &[KongChoice]) -> Option<Action> {
    let mut best: Option<KongChoice> = None;
    for &choice in choices {
        if !choice.progress.no_worse_than(alternative) {
            continue;
        }
        let replace = match &best {
            None => true,
            Some(best) => {
                if choice.progress.better_than(best.progress) {
                    true
                } else if choice.progress == best.progress {
                    choice.income > best.income
                } else {
                    false
                }
            }
        };
        if replace {
            best = Some(choice);
        }
    }
    best.map(|choice| choice.action)
}

/// Builds the post-declaration characteristics of a kong action.
///
/// Returns the meld kind, concealed hand, declared melds, and public ownership
/// after the declaration but before the replacement draw. The hand size is
/// already the corresponding post-discard effective size.
fn kong_position(
    action: Action,
    own_hand: &Hand,
    own_melds: &[Meld],
    public: &[u8; TILE_KINDS],
) -> Option<(MeldKind, Hand, Vec<Meld>, [u8; TILE_KINDS])> {
    match action {
        Action::ConcealedKong(tile) => {
            let mut hand = *own_hand;
            for _ in 0..4 {
                hand.remove(tile).ok()?;
            }
            let mut melds = own_melds.to_vec();
            melds.push(Meld::concealed_kong(tile));
            let mut adjusted = *public;
            let index = tile.index() as usize;
            adjusted[index] = adjusted[index].saturating_add(4);
            Some((MeldKind::ConcealedKong, hand, melds, adjusted))
        }
        Action::AddedKong(tile) => {
            let mut hand = *own_hand;
            hand.remove(tile).ok()?;
            let mut melds = own_melds.to_vec();
            let position = melds
                .iter()
                .position(|meld| meld.kind() == MeldKind::Peng && meld.tile() == tile)?;
            let source = melds[position].from()?;
            melds[position] = Meld::added_kong(tile, source);
            let mut adjusted = *public;
            let index = tile.index() as usize;
            adjusted[index] = adjusted[index].saturating_add(1);
            Some((MeldKind::AddedKong, hand, melds, adjusted))
        }
        _ => None,
    }
}

/// A discard response: compare passing with peng and open kong.
fn choose_discard_response(decision: &Decision<'_>, actions: &[Action], missing: Suit) -> Action {
    if let Some(&hu) = actions.iter().find(|action| matches!(action, Action::Hu)) {
        return hu;
    }
    if actions.len() == 1 {
        return actions[0];
    }

    let Some(response) = decision.observation.response else {
        return super::preferred_action(actions);
    };
    if response.robbing_added_kong {
        return super::preferred_action(actions);
    }

    let observation = &decision.observation;
    let own_hand = observation.own_hand;
    let own_melds = observation.own_melds;
    let public = base_public(observation);

    let Some(mut pass_evaluator) = DistanceEvaluator::new(own_melds, missing) else {
        return super::preferred_action(actions);
    };
    let pass_deficit = pass_evaluator.deficit(own_hand);
    let pass_progress = Progress {
        shanten: pass_deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut pass_evaluator, own_hand, &public, pass_deficit),
    };

    let peng_progress = if actions.contains(&Action::Peng) {
        evaluate_peng(
            own_hand,
            own_melds,
            missing,
            &public,
            response.tile,
            response.source,
        )
    } else {
        None
    };
    let kong_progress = if actions.contains(&Action::OpenKong) {
        evaluate_open_kong(
            own_hand,
            own_melds,
            missing,
            &public,
            response.tile,
            response.source,
        )
    } else {
        None
    };

    // A peng must strictly lower shanten. An availability increase at the same
    // distance is insufficient.
    let peng_eligible =
        peng_progress.is_some_and(|progress| progress.shanten < pass_progress.shanten);
    let kong_eligible = kong_progress.is_some_and(|progress| progress.no_worse_than(pass_progress));

    response_choice(peng_eligible, peng_progress, kong_eligible, kong_progress)
}

/// Chooses between the eligible response claims.
///
/// A peng must have passed the strict-shanten gate; a kong must have passed
/// the `(S, -U)` gate. When both are eligible the better progress rank wins,
/// and the kong wins an exact tie.
fn response_choice(
    peng_eligible: bool,
    peng: Option<Progress>,
    kong_eligible: bool,
    kong: Option<Progress>,
) -> Action {
    match (peng_eligible, kong_eligible) {
        (true, true) => {
            let peng = peng.expect("eligible peng has a progress rank");
            let kong = kong.expect("eligible kong has a progress rank");
            if peng.better_than(kong) {
                Action::Peng
            } else {
                // Better progress rank, and the kong wins an exact tie.
                Action::OpenKong
            }
        }
        (true, false) => Action::Peng,
        (false, true) => Action::OpenKong,
        (false, false) => Action::Pass,
    }
}

/// Position after a peng and its best legal mandatory discard.
fn evaluate_peng(
    own_hand: &Hand,
    own_melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    tile: Tile,
    source: crate::seat::Seat,
) -> Option<Progress> {
    let (_, hand, melds, public_after_claim) =
        claim_position(Action::Peng, own_hand, own_melds, public, tile, source)?;
    let candidates: Vec<Tile> = hand.distinct_tiles().collect();
    let (_, progress) = rank_discards(
        &hand,
        &melds,
        missing,
        &public_after_claim,
        &candidates,
        false,
    )?;
    Some(progress)
}

/// Position after an open kong, before its replacement draw.
fn evaluate_open_kong(
    own_hand: &Hand,
    own_melds: &[Meld],
    missing: Suit,
    public: &[u8; TILE_KINDS],
    tile: Tile,
    source: crate::seat::Seat,
) -> Option<Progress> {
    let (_, hand, melds, adjusted) =
        claim_position(Action::OpenKong, own_hand, own_melds, public, tile, source)?;
    let mut evaluator = DistanceEvaluator::new(&melds, missing)?;
    let deficit = evaluator.deficit(&hand);
    Some(Progress {
        shanten: deficit as i32 - 1,
        ukeire: analysis::ukeire(&mut evaluator, &hand, &adjusted, deficit),
    })
}

/// Builds the post-claim concealed hand, melds, and public ownership for a
/// peng or open kong.
///
/// The claimed discard was counted once in `public`; it is removed from discard
/// ownership and added to the claimant's meld. The deciding player's meld is
/// therefore counted exactly once, and any later forced discard only moves a
/// tile from concealed to public ownership.
fn claim_position(
    action: Action,
    own_hand: &Hand,
    own_melds: &[Meld],
    public: &[u8; TILE_KINDS],
    tile: Tile,
    source: crate::seat::Seat,
) -> Option<(MeldKind, Hand, Vec<Meld>, [u8; TILE_KINDS])> {
    let (remove, meld, added) = match action {
        Action::Peng => (2, Meld::peng(tile, source), 3),
        Action::OpenKong => (3, Meld::open_kong(tile, source), 4),
        _ => return None,
    };

    let mut hand = *own_hand;
    for _ in 0..remove {
        hand.remove(tile).ok()?;
    }
    let mut melds = own_melds.to_vec();
    melds.push(meld);

    let mut adjusted = *public;
    let index = tile.index() as usize;
    adjusted[index] = adjusted[index].saturating_sub(1);
    adjusted[index] = adjusted[index].saturating_add(added);
    Some((meld.kind(), hand, melds, adjusted))
}

#[cfg(test)]
mod tests;
