//! Versioned neural-policy decision encoding for the Sichuan mahjong simulator.
//!
//! The crate is an optional layer over the simulator core: it observes only
//! the public [`sichuan_mahjong_sim::Decision`] boundary and produces numeric
//! state and candidate features for learning and inference.
//! [`encoding`] provides the public API; the core has no dependency on this
//! crate.
//!
//! The encoding is versioned by [`encoding::ENCODING_VERSION`], and the
//! layout, scales, and category order are documented in the [`encoding`]
//! module.
//!
//! [`collection`] adds the neural-side behavior-cloning teacher collector:
//! strict versioned JSONL corpora, seeded splits, source provenance, exclusive
//! writer locking, atomic publication/recovery, and zero-support reporting. It
//! uses only public core and encoding interfaces.

#![forbid(unsafe_code)]

pub mod collection;
pub mod encoding;
