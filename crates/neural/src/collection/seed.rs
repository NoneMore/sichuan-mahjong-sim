//! Episode seeds, their fixed storage form, and the frozen train/validation/test
//! split.
//!
//! Seeds are unsigned 64-bit game seeds. In every persisted form they use
//! exactly sixteen lowercase hexadecimal digits. The split algorithm hashes
//! only the seed, so the same physical deal stays in the same split across
//! teacher revisions and corpus growth.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Fixed split algorithm identifier stored in the manifest.
pub const SPLIT_ALGORITHM: &str = "seed-sha256-80-10-10-v1";

/// Dataset split assigned to one seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// Training bucket, 0 through 79.
    Train,
    /// Validation bucket, 80 through 89.
    Validation,
    /// Test bucket, 90 through 99.
    Test,
}

impl Split {
    /// All splits in report order.
    pub const ALL: [Split; 3] = [Split::Train, Split::Validation, Split::Test];

    /// Stable lowercase report name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }

    /// Index in `0..3`, matching [`Split::ALL`].
    pub const fn index(self) -> usize {
        match self {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A persisted episode seed.
///
/// Construction never normalizes the value. Serialization always emits the
/// fixed sixteen-digit lowercase hexadecimal form, and deserialization accepts
/// only that exact form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Seed(u64);

impl Seed {
    /// Wraps a raw game seed.
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Raw game seed used by [`sichuan_mahjong_sim::Game::new`].
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Fixed sixteen-digit lowercase hexadecimal form.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parses exactly sixteen lowercase hexadecimal digits.
    pub fn parse_hex(value: &str) -> Result<Self, SeedError> {
        let bytes = value.as_bytes();
        if bytes.len() != 16 {
            return Err(SeedError::Length {
                length: bytes.len(),
            });
        }
        let mut parsed = 0u64;
        for &byte in bytes {
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => {
                    return Err(SeedError::Character {
                        character: byte as char,
                    })
                }
            };
            parsed = (parsed << 4) | u64::from(digit);
        }
        Ok(Self(parsed))
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl FromStr for Seed {
    type Err = SeedError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_hex(value)
    }
}

impl Serialize for Seed {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Seed {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SeedVisitor;

        impl Visitor<'_> for SeedVisitor {
            type Value = Seed;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("sixteen lowercase hexadecimal seed digits")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Seed::parse_hex(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(SeedVisitor)
    }
}

/// A rejected [`Seed`] text form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedError {
    /// The string was not exactly sixteen bytes.
    Length {
        /// Observed byte length.
        length: usize,
    },
    /// A byte outside `0`-`9` or `a`-`f` was present.
    Character {
        /// Rejected character.
        character: char,
    },
}

impl fmt::Display for SeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeedError::Length { length } => {
                write!(
                    f,
                    "seed must be 16 lowercase hex digits, got {length} bytes"
                )
            }
            SeedError::Character { character } => {
                write!(f, "seed contains non-lowercase-hex character {character:?}")
            }
        }
    }
}

impl std::error::Error for SeedError {}

/// Computes the frozen split bucket in `0..100`.
///
/// ```text
/// digest = SHA256(ASCII("bc-split-v1") || byte(0) || u64_big_endian(seed))
/// bucket = unsigned_big_endian(digest[0:8]) % 100
/// ```
pub fn split_bucket(seed: Seed) -> u8 {
    let mut hasher = Sha256::new();
    hasher.update(b"bc-split-v1");
    hasher.update([0u8]);
    hasher.update(seed.as_u64().to_be_bytes());
    let digest = hasher.finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(head) % 100) as u8
}

/// Assigns the frozen split: `0..=79` train, `80..=89` validation, 90..=99
/// test.
pub fn seed_split(seed: Seed) -> Split {
    match split_bucket(seed) {
        0..=79 => Split::Train,
        80..=89 => Split::Validation,
        _ => Split::Test,
    }
}
