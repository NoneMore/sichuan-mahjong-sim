//! Authoritative single-game state, decision surface, and settlement.
//!
//! The game owns the complete mutable hand state: the shuffled wall, concealed
//! hands, declared melds, public discard history, missing-suit declarations,
//! scores, winners, the staged private choices, the passed-Hu lock, and the
//! per-payment kong ledger. It is deliberately self-contained: policies only
//! ever see a borrowed [`Decision`], never a [`Game`].
//!
//! # Decision model
//!
//! [`Game::next_decision`] performs automatic progression until a player choice
//! is required. Re-reading while a choice is pending is idempotent. The
//! legality of the pending choice is kept in a compact decision context and is
//! reused by [`Game::submit`]; an invalid submission changes no authoritative
//! state.
//!
//! Exchange, missing-suit, and discard responses are collected before they are
//! applied. A submitted exchange does not move any tile until all four seats
//! have chosen. A missing-suit choice is not public until all four are in. A
//! response choice does not change scores, hands, or other seats' legality
//! until every required response has been collected and arbitrated.
//!
//! # Tile ownership
//!
//! Physical ownership is counted exactly once:
//!
//! - wall tiles are active while `wall_head..wall_tail` is non-empty;
//! - concealed tiles are in a seat's [`Hand`];
//! - declared melds own their physical tiles (three or four);
//! - an unclaimed discard, a discard claimed by one or more Hu winners, and a
//!   robbed added-kong tile are owned by their [`DiscardInfo`] record exactly
//!   once, no matter how many winners reference that same physical tile;
//! - a discard claimed as a peng or open kong is physically part of the
//!   claimant's meld. Its [`DiscardInfo::meld_claim`] is a public reference and
//!   is not counted a second time;
//! - a self-draw winning tile remains in the winner's concealed hand and is
//!   counted there. A discard/robbing winner's scoring hand is evaluated with a
//!   virtual copy of the single physical tile recorded in the discard history.
//!
//! [`Game::debug_validate`] checks this conservation, winner metadata, the
//! mutual linkage between winner metadata and discard records, discard claims,
//! and zero-sum scores. It is a diagnostic helper, not a policy input.

use std::fmt;

use crate::action::{Action, ExchangeDirection, Meld, MeldKind};
use crate::hand::Hand;
use crate::rules::{self, WinMethod};
use crate::seat::{Seat, PLAYER_COUNT};
use crate::tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS, TOTAL_TILES};

/// Why the hand ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EndReason {
    /// Three players have won, leaving one active player.
    ThreeWinners,
    /// A required draw found the wall empty, after all preceding claims were
    /// resolved.
    WallExhausted,
}

/// Terminal facts of one hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GameResult {
    /// Net score by seat. Always sums to zero.
    pub scores: [i32; PLAYER_COUNT],
    /// Seats that won during the hand.
    pub winners: [bool; PLAYER_COUNT],
    /// Why the hand ended.
    pub end_reason: EndReason,
}

/// Public stage of a pending decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecisionStage {
    /// Opening exchange selection.
    Exchange,
    /// Missing-suit declaration.
    MissingSuit,
    /// The current player's own turn (opening, draw, replacement, or peng).
    Turn,
    /// Collecting claims on a discard.
    DiscardResponse,
    /// Collecting Hu responses to an added kong.
    AddedKongResponse,
}

/// Public information about one seat, safe to expose to every player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlayerView<'a> {
    /// Seat represented by this view.
    pub seat: Seat,
    /// Physical number of concealed tiles (not their identities).
    ///
    /// This is the count of tiles actually owned by the seat's [`Hand`], not a
    /// scoring-hand size. A discard or robbing winner therefore reports one
    /// tile fewer than a self-draw winner: the winning tile stays owned by its
    /// [`DiscardInfo`] record instead of moving into the hand, and no virtual
    /// tile is inserted here. Consumers that need the complete 14-tile scoring
    /// hand must pair this count with the discard history association.
    pub concealed_count: usize,
    /// Public declared melds.
    pub melds: &'a [Meld],
    /// Public missing-suit declaration, or `None` until all four are collected.
    pub missing_suit: Option<Suit>,
    /// Current net score.
    pub score: i32,
    /// Whether this seat has already won and left the hand.
    pub has_won: bool,
}

/// Public context of a response decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResponseView {
    /// Player whose discard or added kong is being claimed.
    pub source: Seat,
    /// Tile under claim.
    pub tile: Tile,
    /// Whether this is a robbed added kong rather than an ordinary discard.
    pub robbing_added_kong: bool,
}

/// Hidden-information-safe view for one decision.
///
/// `own_hand` is the only concealed hand present. Opponents appear as
/// [`PlayerView`] values with public melds and concealed tile counts only. The
/// wall is reduced to `wall_count`; its ordered contents are never exposed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Observation<'a> {
    /// The deciding player's concealed hand.
    pub own_hand: &'a Hand,
    /// The deciding player's declared melds.
    pub own_melds: &'a [Meld],
    /// Public information for all four seats, including the deciding seat.
    pub players: [PlayerView<'a>; PLAYER_COUNT],
    /// Global ordered public discard history.
    pub discards: &'a [DiscardInfo],
    /// Public exchange direction, fixed for the whole hand.
    pub exchange_direction: ExchangeDirection,
    /// Number of tiles still in the wall.
    pub wall_count: usize,
    /// Stage of this decision.
    pub stage: DecisionStage,
    /// Response context, present only for discard or added-kong responses.
    pub response: Option<ResponseView>,
}

/// One public discard record.
///
/// A record represents one physical tile. If `meld_claim` is set, the tile was
/// physically incorporated into that claimant's peng/open kong, and the record
/// is a public association rather than extra ownership. If one or more Hu
/// claimants are set, the tile was consumed once by the win; every winner
/// references the same record, and their scoring hands use a virtual copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DiscardInfo {
    /// Seat that discarded or whose added-kong tile was robbed.
    pub seat: Seat,
    /// The discarded or robbed tile.
    pub tile: Tile,
    /// Seat whose peng or open kong incorporated this physical tile.
    pub meld_claim: Option<Seat>,
    /// Seats that claimed Hu on this tile.
    pub winners: [bool; PLAYER_COUNT],
    /// Whether this record came from an added-kong tile rather than a discard.
    pub robbed_kong: bool,
}

impl DiscardInfo {
    fn new(seat: Seat, tile: Tile) -> Self {
        Self {
            seat,
            tile,
            meld_claim: None,
            winners: [false; PLAYER_COUNT],
            robbed_kong: false,
        }
    }

    fn robbed(seat: Seat, tile: Tile) -> Self {
        Self {
            seat,
            tile,
            meld_claim: None,
            winners: [false; PLAYER_COUNT],
            robbed_kong: true,
        }
    }
}

/// Legal actions for the pending decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LegalActions<'a> {
    /// Exchange constraint: only these suits have at least three legal tiles.
    /// The game deliberately does not enumerate every triple.
    Exchange { eligible_suits: [bool; 3] },
    /// Compact list of ordinary semantic actions.
    Actions(&'a [Action]),
}

/// A pending player decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Decision<'a> {
    /// Seat that must choose.
    pub seat: Seat,
    /// Player-visible information.
    pub observation: Observation<'a>,
    /// Legality for this exact decision.
    pub legal_actions: LegalActions<'a>,
}

/// A rejected submission.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameError {
    /// The game has already reached its terminal result.
    GameOver,
    /// No player choice is currently pending.
    NoPendingDecision,
    /// The action was submitted by a seat other than the pending one.
    WrongSeat {
        /// Pending deciding seat.
        expected: Seat,
        /// Seat that submitted.
        actual: Seat,
    },
    /// The action is not legal for the pending decision.
    IllegalAction {
        /// Seat that submitted.
        seat: Seat,
        /// Rejected action.
        action: Action,
    },
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::GameOver => f.write_str("the game has already ended"),
            GameError::NoPendingDecision => f.write_str("no decision is pending"),
            GameError::WrongSeat { expected, actual } => {
                write!(f, "pending decision belongs to {expected}, not {actual}")
            }
            GameError::IllegalAction { seat, action } => {
                write!(f, "{seat} cannot submit {action:?} in the current state")
            }
        }
    }
}

impl std::error::Error for GameError {}

#[derive(Clone, Copy, Debug)]
struct PendingDecision {
    seat: Seat,
    kind: PendingKind,
    exchange_suits: [bool; 3],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PendingKind {
    Exchange,
    Missing,
    Actions,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DrawKind {
    Normal,
    Replacement,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ActionKind {
    Opening,
    AfterDraw(DrawKind),
    AfterPeng,
}

#[derive(Clone, Debug)]
struct ResponseStage {
    source: Seat,
    tile: Tile,
    robbing: bool,
    method: WinMethod,
    discard_index: Option<usize>,
    hu_eligible: [bool; PLAYER_COUNT],
    can_peng: [bool; PLAYER_COUNT],
    can_open_kong: [bool; PLAYER_COUNT],
    hu_declined: [bool; PLAYER_COUNT],
    pending: [bool; PLAYER_COUNT],
    choices: [Option<Action>; PLAYER_COUNT],
}

impl ResponseStage {
    fn new(
        source: Seat,
        tile: Tile,
        robbing: bool,
        method: WinMethod,
        discard_index: Option<usize>,
    ) -> Self {
        Self {
            source,
            tile,
            robbing,
            method,
            discard_index,
            hu_eligible: [false; PLAYER_COUNT],
            can_peng: [false; PLAYER_COUNT],
            can_open_kong: [false; PLAYER_COUNT],
            hu_declined: [false; PLAYER_COUNT],
            pending: [false; PLAYER_COUNT],
            choices: [None; PLAYER_COUNT],
        }
    }
}

#[derive(Clone, Debug)]
enum Stage {
    Exchange,
    Missing,
    Action { seat: Seat, kind: ActionKind },
    Response(ResponseStage),
    Ended,
}

#[derive(Clone, Copy, Debug)]
struct WinInfo {
    method: WinMethod,
    fan: u8,
    winning_tile: Option<Tile>,
    payer: Option<Seat>,
}

#[derive(Clone, Copy, Debug)]
struct KongRecord {
    declarer: Seat,
    payer: Seat,
    kind: MeldKind,
    points: u32,
}

/// Deterministic SplitMix64 random source owned by the game.
#[derive(Clone, Debug)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let upper = upper as u64;
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % upper) as usize;
            }
        }
    }
}

fn shuffle_wall(wall: &mut [u8; TOTAL_TILES], rng: &mut Rng64) {
    for index in (1..wall.len()).rev() {
        let other = rng.below(index + 1);
        wall.swap(index, other);
    }
}

/// Seats clockwise after `source`, excluding `source`.
fn seats_after(source: Seat) -> [Seat; PLAYER_COUNT - 1] {
    let mut seats = [source; PLAYER_COUNT - 1];
    let mut current = source;
    for slot in &mut seats {
        current = current.next_clockwise();
        *slot = current;
    }
    seats
}

/// Full authoritative game state.
///
/// `Game` implements `Clone`; the clone includes the random state, staged
/// private choices, pending decision legality, scores, discard history, and
/// kong ledger. No separate `GameState` exists.
#[derive(Clone, Debug)]
pub struct Game {
    wall: [u8; TOTAL_TILES],
    wall_head: usize,
    wall_tail: usize,
    rng: Rng64,

    hands: [Hand; PLAYER_COUNT],
    melds: [Vec<Meld>; PLAYER_COUNT],
    discards: Vec<DiscardInfo>,
    missing: [Option<Suit>; PLAYER_COUNT],
    scores: [i32; PLAYER_COUNT],
    winners: [bool; PLAYER_COUNT],
    win_info: [Option<WinInfo>; PLAYER_COUNT],
    exchange_direction: ExchangeDirection,
    exchange_choices: [Option<[Tile; 3]>; PLAYER_COUNT],
    missing_choices: [Option<Suit>; PLAYER_COUNT],
    pass_lock: [bool; PLAYER_COUNT],
    kong_ledger: Vec<KongRecord>,
    last_draw_tile: Option<Tile>,

    stage: Stage,
    pending: Option<PendingDecision>,
    legal_actions: Vec<Action>,
    result: Option<GameResult>,
}

impl Game {
    /// Creates a seeded game: shuffles the complete 108-tile wall, chooses the
    /// exchange direction, deals 13/13/13/14, and waits for exchanges.
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng64::new(seed);
        let mut wall = [0u8; TOTAL_TILES];
        let mut cursor = 0;
        for tile in Tile::ALL {
            for _ in 0..COPIES_PER_TILE {
                wall[cursor] = tile.index();
                cursor += 1;
            }
        }
        shuffle_wall(&mut wall, &mut rng);

        let mut hands = [Hand::new(); PLAYER_COUNT];
        let mut head = 0;
        for _ in 0..rules::NON_DEALER_INITIAL_TILES {
            for seat in Seat::ALL {
                let tile = Tile::from_index(wall[head]).expect("fresh wall tile");
                head += 1;
                hands[seat.index()]
                    .add(tile)
                    .expect("fresh deal fits a hand");
            }
        }
        for _ in rules::NON_DEALER_INITIAL_TILES..rules::DEALER_INITIAL_TILES {
            let tile = Tile::from_index(wall[head]).expect("fresh wall tile");
            head += 1;
            hands[Seat::DEALER.index()]
                .add(tile)
                .expect("fresh deal fits a hand");
        }

        let direction = ExchangeDirection::ALL[rng.below(ExchangeDirection::ALL.len())];

        Self {
            wall,
            wall_head: head,
            wall_tail: TOTAL_TILES,
            rng,
            hands,
            melds: std::array::from_fn(|_| Vec::with_capacity(4)),
            discards: Vec::with_capacity(TOTAL_TILES),
            missing: [None; PLAYER_COUNT],
            scores: [0; PLAYER_COUNT],
            winners: [false; PLAYER_COUNT],
            win_info: [None; PLAYER_COUNT],
            exchange_direction: direction,
            exchange_choices: [None; PLAYER_COUNT],
            missing_choices: [None; PLAYER_COUNT],
            pass_lock: [false; PLAYER_COUNT],
            kong_ledger: Vec::new(),
            last_draw_tile: None,
            stage: Stage::Exchange,
            pending: None,
            legal_actions: Vec::with_capacity(24),
            result: None,
        }
    }

    /// Returns the next pending decision, advancing automatic steps as needed.
    ///
    /// Repeated calls while a decision is pending return the same seat and
    /// legality without mutating the game. `None` means the hand is terminal;
    /// [`Game::result`] then returns the settlement.
    pub fn next_decision(&mut self) -> Option<Decision<'_>> {
        self.advance();
        let pending = self.pending?;
        let seat = pending.seat;
        let legal_actions = match pending.kind {
            PendingKind::Exchange => LegalActions::Exchange {
                eligible_suits: pending.exchange_suits,
            },
            PendingKind::Missing | PendingKind::Actions => {
                LegalActions::Actions(self.legal_actions.as_slice())
            }
        };
        let observation = self.observation(seat);
        Some(Decision {
            seat,
            observation,
            legal_actions,
        })
    }

    /// Submits the pending player's action.
    ///
    /// Wrong-seat, wrong-stage, and illegal actions leave all authoritative
    /// state (including staged choices, hands, wall, scores, and pending
    /// decision) unchanged.
    pub fn submit(&mut self, seat: Seat, action: Action) -> Result<(), GameError> {
        if self.result.is_some() {
            return Err(GameError::GameOver);
        }
        let Some(pending) = self.pending else {
            return Err(GameError::NoPendingDecision);
        };
        if pending.seat != seat {
            return Err(GameError::WrongSeat {
                expected: pending.seat,
                actual: seat,
            });
        }

        match pending.kind {
            PendingKind::Exchange => {
                let Action::Exchange(tiles) = action else {
                    return Err(GameError::IllegalAction { seat, action });
                };
                if !pending.exchange_suits[tiles[0].suit_index()]
                    || !rules::is_valid_exchange(&self.hands[seat.index()], tiles)
                {
                    return Err(GameError::IllegalAction { seat, action });
                }
                self.exchange_choices[seat.index()] = Some(tiles);
            }
            PendingKind::Missing => {
                if !self.legal_actions.contains(&action) {
                    return Err(GameError::IllegalAction { seat, action });
                }
                let Action::ChooseMissing(suit) = action else {
                    return Err(GameError::IllegalAction { seat, action });
                };
                self.missing_choices[seat.index()] = Some(suit);
            }
            PendingKind::Actions => {
                if !self.legal_actions.contains(&action) {
                    return Err(GameError::IllegalAction { seat, action });
                }
                let stage = self.stage.clone();
                if !matches!(stage, Stage::Action { .. } | Stage::Response(_)) {
                    return Err(GameError::IllegalAction { seat, action });
                }
                // Extract the Hu offer from the validated action list before the
                // buffer is cleared: it is exactly the state that arms the
                // passed-Hu lock, so the win shape is not recomputed on apply.
                let offers_hu = self.legal_actions.contains(&Action::Hu);
                self.pending = None;
                self.legal_actions.clear();
                match stage {
                    Stage::Action {
                        seat: current,
                        kind,
                    } => self.apply_turn_action(current, kind, offers_hu, action),
                    Stage::Response(response) => self.apply_response_action(response, seat, action),
                    Stage::Exchange | Stage::Missing | Stage::Ended => unreachable!(),
                }
                self.advance();
                return Ok(());
            }
        }

        self.pending = None;
        self.legal_actions.clear();
        self.advance();
        Ok(())
    }

    /// Terminal result, or `None` while the hand is still playable.
    pub fn result(&self) -> Option<&GameResult> {
        self.result.as_ref()
    }

    fn wall_remaining(&self) -> usize {
        self.wall_tail.saturating_sub(self.wall_head)
    }

    fn active_count(&self) -> usize {
        self.winners.iter().filter(|&&won| !won).count()
    }

    fn observation(&self, seat: Seat) -> Observation<'_> {
        let players = std::array::from_fn(|index| {
            let view_seat = Seat::new(index).expect("seat index");
            PlayerView {
                seat: view_seat,
                concealed_count: self.hands[index].total(),
                melds: self.melds[index].as_slice(),
                missing_suit: self.missing[index],
                score: self.scores[index],
                has_won: self.winners[index],
            }
        });
        let (stage, response) = match &self.stage {
            Stage::Exchange => (DecisionStage::Exchange, None),
            Stage::Missing => (DecisionStage::MissingSuit, None),
            Stage::Action { .. } => (DecisionStage::Turn, None),
            Stage::Response(response_stage) => (
                if response_stage.robbing {
                    DecisionStage::AddedKongResponse
                } else {
                    DecisionStage::DiscardResponse
                },
                Some(ResponseView {
                    source: response_stage.source,
                    tile: response_stage.tile,
                    robbing_added_kong: response_stage.robbing,
                }),
            ),
            Stage::Ended => (DecisionStage::Turn, None),
        };
        Observation {
            own_hand: &self.hands[seat.index()],
            own_melds: self.melds[seat.index()].as_slice(),
            players,
            discards: self.discards.as_slice(),
            exchange_direction: self.exchange_direction,
            wall_count: self.wall_remaining(),
            stage,
            response,
        }
    }

    fn advance(&mut self) {
        if self.result.is_some() {
            self.pending = None;
            return;
        }
        if self.pending.is_some() {
            return;
        }
        loop {
            if self.result.is_some() {
                return;
            }
            match self.stage.clone() {
                Stage::Exchange => {
                    if let Some(seat) = self.first_exchange_pending() {
                        let hand = self.hands[seat.index()];
                        let exchange_suits = std::array::from_fn(|index| {
                            let suit = Suit::from_index(index).expect("suit index");
                            rules::can_exchange_suit(&hand, suit)
                        });
                        self.legal_actions.clear();
                        self.pending = Some(PendingDecision {
                            seat,
                            kind: PendingKind::Exchange,
                            exchange_suits,
                        });
                        return;
                    }
                    self.apply_exchange();
                }
                Stage::Missing => {
                    if let Some(seat) = self.first_missing_pending() {
                        self.legal_actions.clear();
                        self.legal_actions
                            .extend(Suit::ALL.into_iter().map(Action::ChooseMissing));
                        self.pending = Some(PendingDecision {
                            seat,
                            kind: PendingKind::Missing,
                            exchange_suits: [false; 3],
                        });
                        return;
                    }
                    self.apply_missing();
                }
                Stage::Action { seat, kind } => {
                    self.pending_turn_action(seat, kind);
                    return;
                }
                Stage::Response(response) => {
                    if let Some(seat) = Self::response_next_pending(&response) {
                        let pass_locked = self.pass_lock[seat.index()];
                        Self::response_options(
                            &response,
                            seat,
                            pass_locked,
                            &mut self.legal_actions,
                        );
                        if self.legal_actions.is_empty() {
                            let mut updated = response;
                            updated.pending[seat.index()] = false;
                            updated.choices[seat.index()] = Some(Action::Pass);
                            self.stage = Stage::Response(updated);
                            continue;
                        }
                        self.pending = Some(PendingDecision {
                            seat,
                            kind: PendingKind::Actions,
                            exchange_suits: [false; 3],
                        });
                        return;
                    }
                    self.resolve_response(response);
                }
                Stage::Ended => return,
            }
        }
    }

    fn first_exchange_pending(&self) -> Option<Seat> {
        Seat::ALL
            .into_iter()
            .find(|seat| self.exchange_choices[seat.index()].is_none())
    }

    fn first_missing_pending(&self) -> Option<Seat> {
        Seat::ALL
            .into_iter()
            .find(|seat| self.missing_choices[seat.index()].is_none())
    }

    fn apply_exchange(&mut self) {
        let choices = self.exchange_choices;
        for seat in Seat::ALL {
            let tiles = choices[seat.index()].expect("all exchange choices collected");
            self.hands[seat.index()]
                .remove_tiles(&tiles)
                .expect("validated exchange tiles");
        }
        for seat in Seat::ALL {
            let tiles = choices[seat.index()].expect("all exchange choices collected");
            let recipient = self.exchange_direction.recipient(seat);
            self.hands[recipient.index()]
                .add_tiles(&tiles)
                .expect("simultaneous exchange fits the recipient");
        }
        self.exchange_choices = [None; PLAYER_COUNT];
        self.stage = Stage::Missing;
    }

    fn apply_missing(&mut self) {
        for seat in Seat::ALL {
            let choice = self.missing_choices[seat.index()]
                .take()
                .expect("all missing-suit choices collected");
            self.missing[seat.index()] = Some(choice);
        }
        self.stage = Stage::Action {
            seat: Seat::DEALER,
            kind: ActionKind::Opening,
        };
    }

    fn pending_turn_action(&mut self, seat: Seat, kind: ActionKind) {
        // The list is rebuilt in the shared decision buffer, keeping the
        // pipeline order Hu, concealed kongs, added kongs, discards, Pass.
        let can_hu = self.can_self_hu(seat, kind) && !self.pass_lock[seat.index()];
        let wall_open = self.wall_remaining() > 0;
        let Self {
            legal_actions,
            hands,
            melds,
            missing,
            ..
        } = self;
        let hand = hands[seat.index()];
        let melds = &melds[seat.index()];
        let missing = missing[seat.index()].expect("missing suit declared before play");

        legal_actions.clear();

        if can_hu {
            legal_actions.push(Action::Hu);
        }

        if kind != ActionKind::AfterPeng && wall_open {
            for tile in Tile::ALL {
                if rules::can_concealed_kong(&hand, melds, missing, tile) {
                    legal_actions.push(Action::ConcealedKong(tile));
                }
            }
            for tile in Tile::ALL {
                if rules::can_added_kong(&hand, melds, missing, tile) {
                    legal_actions.push(Action::AddedKong(tile));
                }
            }
        }

        for tile in Tile::ALL {
            if rules::can_discard(&hand, missing, tile) {
                legal_actions.push(Action::Discard(tile));
            }
        }

        if can_hu {
            legal_actions.push(Action::Pass);
        }

        if legal_actions.is_empty() {
            unreachable!("an active turn must have at least one legal action");
        }

        self.pending = Some(PendingDecision {
            seat,
            kind: PendingKind::Actions,
            exchange_suits: [false; 3],
        });
    }

    fn can_self_hu(&self, seat: Seat, kind: ActionKind) -> bool {
        if kind == ActionKind::AfterPeng {
            return false;
        }
        let Some(missing) = self.missing[seat.index()] else {
            return false;
        };
        rules::can_hu(
            &self.hands[seat.index()],
            &self.melds[seat.index()],
            missing,
        )
    }

    fn can_hu_on_tile(&self, seat: Seat, tile: Tile) -> bool {
        if self.winners[seat.index()] {
            return false;
        }
        let Some(missing) = self.missing[seat.index()] else {
            return false;
        };
        let mut hand = self.hands[seat.index()];
        if hand.add(tile).is_err() {
            return false;
        }
        rules::can_hu(&hand, &self.melds[seat.index()], missing)
    }

    fn apply_turn_action(&mut self, seat: Seat, kind: ActionKind, offers_hu: bool, action: Action) {
        let index = seat.index();
        // `offers_hu` is the Hu slot taken from the pending action list, so the
        // passed-Hu lock is armed without recomputing the winning shape.
        if offers_hu && action != Action::Hu {
            self.pass_lock[index] = true;
        }

        match action {
            Action::Discard(tile) => {
                self.hands[index]
                    .remove(tile)
                    .expect("discard action lists a concealed tile");
                let method = match kind {
                    ActionKind::AfterDraw(DrawKind::Replacement) => WinMethod::KongDiscard,
                    _ => WinMethod::Discard,
                };
                let discard_index = self.discards.len();
                self.discards.push(DiscardInfo::new(seat, tile));
                self.begin_discard_response(seat, tile, method, discard_index);
            }
            Action::Hu => {
                self.settle_self_draw_win(seat, kind);
                if self.active_count() <= 1 {
                    self.end_hand(EndReason::ThreeWinners);
                } else {
                    self.start_next_turn_after(seat);
                }
            }
            Action::Pass => {
                // The passed-Hu lock was set above; the same action stage is
                // rebuilt without Hu on the next decision read.
            }
            Action::ConcealedKong(tile) => {
                self.commit_concealed_kong(seat, tile);
            }
            Action::AddedKong(tile) => {
                self.begin_added_kong_response(seat, tile);
            }
            Action::Exchange(_) | Action::ChooseMissing(_) | Action::Peng | Action::OpenKong => {
                unreachable!("turn action list only contains turn actions")
            }
        }
    }

    fn begin_discard_response(
        &mut self,
        source: Seat,
        tile: Tile,
        method: WinMethod,
        discard_index: usize,
    ) {
        let mut response = ResponseStage::new(source, tile, false, method, Some(discard_index));
        for seat in seats_after(source) {
            let index = seat.index();
            if self.winners[index] {
                response.choices[index] = Some(Action::Pass);
                continue;
            }
            if let Some(missing) = self.missing[index] {
                response.hu_eligible[index] = self.can_hu_on_tile(seat, tile);
                response.can_peng[index] =
                    rules::can_peng(&self.hands[index], &self.melds[index], missing, tile);
                response.can_open_kong[index] = self.wall_remaining() > 0
                    && rules::can_open_kong(&self.hands[index], &self.melds[index], missing, tile);
            }
            let has_choice = (response.hu_eligible[index] && !self.pass_lock[index])
                || response.can_peng[index]
                || response.can_open_kong[index];
            if has_choice {
                response.pending[index] = true;
            } else {
                response.choices[index] = Some(Action::Pass);
            }
        }
        response.choices[source.index()] = Some(Action::Pass);
        self.stage = Stage::Response(response);
    }

    fn begin_added_kong_response(&mut self, declarer: Seat, tile: Tile) {
        let mut response = ResponseStage::new(declarer, tile, true, WinMethod::RobbingKong, None);
        for seat in seats_after(declarer) {
            let index = seat.index();
            if self.winners[index] {
                response.choices[index] = Some(Action::Pass);
                continue;
            }
            response.hu_eligible[index] = self.can_hu_on_tile(seat, tile);
            if response.hu_eligible[index] && !self.pass_lock[index] {
                response.pending[index] = true;
            } else {
                response.choices[index] = Some(Action::Pass);
            }
        }
        response.choices[declarer.index()] = Some(Action::Pass);
        self.stage = Stage::Response(response);
    }

    fn response_next_pending(response: &ResponseStage) -> Option<Seat> {
        seats_after(response.source)
            .into_iter()
            .find(|seat| response.pending[seat.index()])
    }

    /// Fills `out` with the response options of `seat`, highest claim first.
    ///
    /// The buffer and the passed-Hu flag are passed explicitly (instead of
    /// borrowing the game) so the pending decision can reuse `legal_actions`.
    fn response_options(
        response: &ResponseStage,
        seat: Seat,
        pass_locked: bool,
        out: &mut Vec<Action>,
    ) {
        let index = seat.index();
        out.clear();
        if response.hu_eligible[index] && !response.hu_declined[index] && !pass_locked {
            out.push(Action::Hu);
        }
        if !response.robbing {
            if response.can_peng[index] {
                out.push(Action::Peng);
            }
            if response.can_open_kong[index] {
                out.push(Action::OpenKong);
            }
        }
        if !out.is_empty() {
            out.push(Action::Pass);
        }
    }

    fn apply_response_action(&mut self, mut response: ResponseStage, seat: Seat, action: Action) {
        let index = seat.index();
        let was_locked = self.pass_lock[index];
        let declining_hu = response.hu_eligible[index] && !was_locked;

        if action != Action::Hu && declining_hu {
            // Choosing a lower claim or passing both decline the win. A later
            // draw clears this lock.
            self.pass_lock[index] = true;
        }

        match action {
            Action::Hu | Action::Peng | Action::OpenKong => {
                response.pending[index] = false;
                response.choices[index] = Some(action);
                self.stage = Stage::Response(response);
            }
            Action::Pass => {
                let has_lower_claim = !response.robbing
                    && (response.can_peng[index] || response.can_open_kong[index]);
                let follow_up = declining_hu && !response.hu_declined[index] && has_lower_claim;
                if follow_up {
                    // Keep the seat pending for a second, same-information
                    // choice between its still-legal lower claims and Pass.
                    response.hu_declined[index] = true;
                } else {
                    response.pending[index] = false;
                    response.choices[index] = Some(Action::Pass);
                }
                self.stage = Stage::Response(response);
            }
            Action::Exchange(_)
            | Action::ChooseMissing(_)
            | Action::Discard(_)
            | Action::ConcealedKong(_)
            | Action::AddedKong(_) => {
                unreachable!("response action list only contains response actions")
            }
        }
    }

    fn resolve_response(&mut self, response: ResponseStage) {
        let source = response.source;
        let tile = response.tile;
        let hu_winners: Vec<Seat> = Seat::ALL
            .into_iter()
            .filter(|seat| response.choices[seat.index()] == Some(Action::Hu))
            .collect();

        if !hu_winners.is_empty() {
            if response.robbing {
                self.hands[source.index()]
                    .remove(tile)
                    .expect("robbed added-kong tile is still in the declarer hand");
                let mut discard = DiscardInfo::robbed(source, tile);
                for &winner in &hu_winners {
                    discard.winners[winner.index()] = true;
                }
                self.discards.push(discard);
            } else if let Some(discard_index) = response.discard_index {
                for &winner in &hu_winners {
                    self.discards[discard_index].winners[winner.index()] = true;
                }
            }

            for &winner in &hu_winners {
                self.settle_discard_win(winner, tile, response.method, source);
            }
            if self.active_count() <= 1 {
                self.end_hand(EndReason::ThreeWinners);
            } else {
                self.start_next_turn_after(source);
            }
            return;
        }

        if response.robbing {
            self.commit_added_kong(source, tile);
            return;
        }

        if let Some(claimant) = Self::nearest_claimant(&response, Action::OpenKong) {
            self.apply_open_kong_claim(claimant, source, tile, response.discard_index);
            return;
        }
        if let Some(claimant) = Self::nearest_claimant(&response, Action::Peng) {
            self.apply_peng_claim(claimant, source, tile, response.discard_index);
            return;
        }

        self.start_next_turn_after(source);
    }

    fn nearest_claimant(response: &ResponseStage, action: Action) -> Option<Seat> {
        seats_after(response.source)
            .into_iter()
            .find(|seat| response.choices[seat.index()] == Some(action))
    }

    fn apply_peng_claim(
        &mut self,
        claimant: Seat,
        source: Seat,
        tile: Tile,
        discard_index: Option<usize>,
    ) {
        let index = claimant.index();
        self.hands[index]
            .remove_tiles(&[tile, tile])
            .expect("peng claimant holds two matching tiles");
        self.melds[index].push(Meld::peng(tile, source));
        if let Some(discard_index) = discard_index {
            self.discards[discard_index].meld_claim = Some(claimant);
        }
        self.stage = Stage::Action {
            seat: claimant,
            kind: ActionKind::AfterPeng,
        };
    }

    fn apply_open_kong_claim(
        &mut self,
        claimant: Seat,
        source: Seat,
        tile: Tile,
        discard_index: Option<usize>,
    ) {
        let index = claimant.index();
        self.hands[index]
            .remove_tiles(&[tile, tile, tile])
            .expect("open kong claimant holds three matching tiles");
        self.melds[index].push(Meld::open_kong(tile, source));
        if let Some(discard_index) = discard_index {
            self.discards[discard_index].meld_claim = Some(claimant);
        }
        let payment = rules::kong_payment(MeldKind::OpenKong, self.active_count() - 1);
        self.transfer(source, claimant, payment.per_payer);
        self.kong_ledger.push(KongRecord {
            declarer: claimant,
            payer: source,
            kind: MeldKind::OpenKong,
            points: payment.per_payer,
        });
        if !self.draw_replacement(claimant) {
            self.end_hand(EndReason::WallExhausted);
        }
    }

    fn commit_concealed_kong(&mut self, declarer: Seat, tile: Tile) {
        if self.wall_remaining() == 0 {
            self.end_hand(EndReason::WallExhausted);
            return;
        }
        let index = declarer.index();
        self.hands[index]
            .remove_tiles(&[tile, tile, tile, tile])
            .expect("concealed kong declaration holds four matching tiles");
        self.melds[index].push(Meld::concealed_kong(tile));
        self.pay_kong_opponents(declarer, MeldKind::ConcealedKong);
        if !self.draw_replacement(declarer) {
            self.end_hand(EndReason::WallExhausted);
        }
    }

    fn commit_added_kong(&mut self, declarer: Seat, tile: Tile) {
        if self.wall_remaining() == 0 {
            self.end_hand(EndReason::WallExhausted);
            return;
        }
        let index = declarer.index();
        let peng_index = self.melds[index]
            .iter()
            .position(|meld| meld.kind() == MeldKind::Peng && meld.tile() == tile)
            .expect("added kong has an existing peng");
        let source = self.melds[index][peng_index]
            .from()
            .expect("peng records its source");
        self.hands[index]
            .remove(tile)
            .expect("added kong tile is in the declarer hand");
        self.melds[index][peng_index] = Meld::added_kong(tile, source);
        self.pay_kong_opponents(declarer, MeldKind::AddedKong);
        if !self.draw_replacement(declarer) {
            self.end_hand(EndReason::WallExhausted);
        }
    }

    fn start_next_turn_after(&mut self, source: Seat) {
        if self.active_count() <= 1 {
            self.end_hand(EndReason::ThreeWinners);
            return;
        }
        let Some(next) = self.next_active_after(source) else {
            self.end_hand(EndReason::ThreeWinners);
            return;
        };
        self.draw_normal(next);
    }

    fn next_active_after(&self, source: Seat) -> Option<Seat> {
        seats_after(source)
            .into_iter()
            .find(|seat| !self.winners[seat.index()])
    }

    fn draw_normal(&mut self, seat: Seat) {
        if self.wall_remaining() == 0 {
            self.end_hand(EndReason::WallExhausted);
            return;
        }
        let tile = Tile::from_index(self.wall[self.wall_head]).expect("wall tile");
        self.wall_head += 1;
        self.pass_lock[seat.index()] = false;
        self.hands[seat.index()]
            .add(tile)
            .expect("wall tile fits the drawing hand");
        self.last_draw_tile = Some(tile);
        self.stage = Stage::Action {
            seat,
            kind: ActionKind::AfterDraw(DrawKind::Normal),
        };
    }

    fn draw_replacement(&mut self, seat: Seat) -> bool {
        if self.wall_remaining() == 0 {
            return false;
        }
        self.wall_tail -= 1;
        let tile = Tile::from_index(self.wall[self.wall_tail]).expect("wall tile");
        self.pass_lock[seat.index()] = false;
        self.hands[seat.index()]
            .add(tile)
            .expect("replacement tile fits the drawing hand");
        self.last_draw_tile = Some(tile);
        self.stage = Stage::Action {
            seat,
            kind: ActionKind::AfterDraw(DrawKind::Replacement),
        };
        true
    }

    /// Transitions to the terminal result, settling wall exhaustion first.
    ///
    /// The first ending wins: a later call is ignored so an already reported
    /// result is never settled or recorded twice.
    fn end_hand(&mut self, reason: EndReason) {
        if self.result.is_some() {
            return;
        }
        if reason == EndReason::WallExhausted {
            self.settle_wall_exhaustion();
        }
        self.stage = Stage::Ended;
        self.pending = None;
        self.legal_actions.clear();
        self.result = Some(GameResult {
            scores: self.scores,
            winners: self.winners,
            end_reason: reason,
        });
    }
}

mod diagnostics;
mod settlement;

#[cfg(test)]
mod tests;
