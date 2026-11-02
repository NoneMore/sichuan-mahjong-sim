//! Counted hands.
//!
//! The implementation lives in [`crate::tile`] so tile encoding and hand
//! counting stay together; this module provides the stable `hand` path.

pub use crate::tile::{Hand, HandError};
