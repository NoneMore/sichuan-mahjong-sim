//! Unit and scenario tests for the authoritative game state machine.
//!
//! The module is a child of `game`, so it can construct small synthetic states
//! with exact hands, melds, walls, and ledgers. Every scenario ends with
//! `debug_validate`, which checks physical tile conservation and zero-sum
//! scores across the documented ownership representation.
//!
//! Scenarios are grouped by theme, with shared builders in `fixtures`.

mod contracts;
mod fixtures;
mod kongs;
mod responses;
mod settlement;
mod setup;
