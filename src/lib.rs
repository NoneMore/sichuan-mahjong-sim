//! Core types, fixed rules, and authoritative game for a Sichuan mahjong
//! ("blood battle") simulator.
//!
//! The crate contains the stable foundation (tiles, seats, actions, melds,
//! pure winning-shape evaluation, and fixed baseline scoring facts), the
//! authoritative single-game state machine, and a native policy/runner
//! execution layer.
//!
//! The baseline rules are documented in `docs/baseline-rules.md`.
//!
//! Neural-policy feature encoding lives in the `sichuan-mahjong-neural`
//! workspace crate, which depends on this core rather than the reverse.

#![forbid(unsafe_code)]

pub mod action;
pub mod eval;
pub mod game;
pub mod hand;
pub mod policy;
pub mod rules;
pub mod runner;
pub mod seat;
pub mod tile;

pub use game::{
    Decision, DecisionStage, DiscardInfo, EndReason, Game, GameError, GameResult, LegalActions,
    Observation, PlayerView, ResponseView,
};
pub use policy::{
    CacheConfigError, CacheState, CacheStatus, FirstLegalPolicy, HeuristicCache, HeuristicPolicy,
    MissingOnlyPolicy, Policy, RandomPolicy,
};
pub use runner::{BatchReport, DecisionRecord, FinishReason, RunReport, Runner, RunnerError};

pub use action::{Action, ExchangeDirection, Meld, MeldError, MeldKind};
pub use eval::{
    expected_concealed_len, is_seven_pairs, is_standard_winning, is_tenpai, is_winning,
    waiting_tiles, win_shape, WinShape,
};
pub use hand::{Hand, HandError};
pub use rules::{
    can_added_kong, can_concealed_kong, can_discard, can_exchange_suit, can_hu, can_open_kong,
    can_peng, has_missing_suit_tiles, is_all_pungs, is_pure_suit, is_valid_exchange, kong_payment,
    max_discard_win_fan, must_refund_kong_income, root_count, settle_wall_exhaustion, win_fan,
    win_payment, FanBreakdown, KongPayment, Transfer, WallExhaustionEntry, WinMethod, WinPayment,
    ADDED_KONG_POINTS, BASE_WIN_FAN, CONCEALED_KONG_POINTS, DEALER_INITIAL_TILES, FAN_CAP,
    FLOWER_PIG_PENALTY, NON_DEALER_INITIAL_TILES, OPEN_KONG_POINTS,
};
pub use seat::{Seat, PLAYER_COUNT};
pub use tile::{Suit, Tile, COPIES_PER_TILE, TILE_KINDS, TOTAL_TILES};

/// Compile-time checks for the fixed deck size and encoded tile domain.
const _: () = {
    assert!(TILE_KINDS == 27);
    assert!(TOTAL_TILES == 108);
    assert!(PLAYER_COUNT == 4);
};
