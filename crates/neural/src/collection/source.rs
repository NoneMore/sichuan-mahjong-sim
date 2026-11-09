//! Compiled-source provenance and build/environment facts.
//!
//! A reusable corpus is identified by the full Git commit that produced the
//! simulator, teacher, and collector code. The commit and dirty flag are fixed
//! when this crate is compiled by [`crate`'s build script]; collection does not
//! ask the caller's current checkout. A dirty or unavailable build therefore
//! fails closed instead of borrowing a clean commit ID.

use std::fmt;
use std::mem;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A validated full lowercase Git commit ID (SHA-1 or SHA-256 object format).
///
/// The type validates text form only. A caller-supplied value is
/// caller-attested provenance for controlled tests or an external pipeline;
/// reusable collection of this binary's output should obtain the identity from
/// [`SourceRevision::from_compiled_source`], which also rejects dirty or
/// unavailable build-time Git facts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceRevision(String);

impl SourceRevision {
    /// Validates a full lowercase Git commit ID of 40 or 64 hexadecimal digits.
    pub fn parse(value: &str) -> Result<Self, SourceRevisionError> {
        let bytes = value.as_bytes();
        if !matches!(bytes.len(), 40 | 64) {
            return Err(SourceRevisionError::Length {
                length: bytes.len(),
            });
        }
        for &byte in bytes {
            if !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte) {
                return Err(SourceRevisionError::Character {
                    character: byte as char,
                });
            }
        }
        Ok(Self(value.to_owned()))
    }

    /// The revision that produced this binary.
    ///
    /// Fails when the build script could not observe Git, when the compiled
    /// working tree was dirty, or when the recorded commit is malformed.
    pub fn from_compiled_source() -> Result<Self, SourceProvenanceError> {
        Self::from_build_info(&build_info())
    }

    /// Applies the compiled-source rule to a supplied build snapshot.
    ///
    /// This is the testable form of [`SourceRevision::from_compiled_source`].
    pub fn from_build_info(info: &BuildInfo) -> Result<Self, SourceProvenanceError> {
        if !info.git_available {
            return Err(SourceProvenanceError::Unavailable {
                reason: "Git provenance was unavailable when the crate was compiled".to_owned(),
            });
        }
        let Some(commit) = info.git_commit.as_deref() else {
            return Err(SourceProvenanceError::Unavailable {
                reason: "the compiled source has no recorded Git commit".to_owned(),
            });
        };
        if info.git_dirty {
            return Err(SourceProvenanceError::Dirty {
                commit: commit.to_owned(),
            });
        }
        Self::parse(commit).map_err(|source| SourceProvenanceError::InvalidCommit {
            value: commit.to_owned(),
            source,
        })
    }

    /// Borrowed lowercase commit ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SourceRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for SourceRevision {
    type Err = SourceRevisionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl Serialize for SourceRevision {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SourceRevision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RevisionVisitor;

        impl Visitor<'_> for RevisionVisitor {
            type Value = SourceRevision;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a full lowercase 40- or 64-digit Git commit ID")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                SourceRevision::parse(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(RevisionVisitor)
    }
}

/// A rejected commit ID text form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceRevisionError {
    /// The string was not 40 or 64 bytes.
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

impl fmt::Display for SourceRevisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceRevisionError::Length { length } => write!(
                f,
                "source revision must be 40 or 64 lowercase hex digits, got {length} bytes"
            ),
            SourceRevisionError::Character { character } => write!(
                f,
                "source revision contains non-lowercase-hex character {character:?}"
            ),
        }
    }
}

impl std::error::Error for SourceRevisionError {}

/// Why compiled-source provenance cannot identify a reusable corpus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceProvenanceError {
    /// Git or the commit was unavailable at compile time.
    Unavailable {
        /// Human-readable cause.
        reason: String,
    },
    /// The compiled working tree contained uncommitted source changes.
    Dirty {
        /// Commit the dirty tree was based on.
        commit: String,
    },
    /// The build script recorded a malformed commit ID.
    InvalidCommit {
        /// Recorded value.
        value: String,
        /// Validation failure.
        source: SourceRevisionError,
    },
}

impl fmt::Display for SourceProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceProvenanceError::Unavailable { reason } => {
                write!(f, "compiled source provenance is unavailable: {reason}")
            }
            SourceProvenanceError::Dirty { commit } => write!(
                f,
                "compiled source is dirty at commit {commit}; reusable collection \
                 requires a clean committed build"
            ),
            SourceProvenanceError::InvalidCommit { value, source } => {
                write!(
                    f,
                    "compiled source recorded invalid commit {value:?}: {source}"
                )
            }
        }
    }
}

impl std::error::Error for SourceProvenanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SourceProvenanceError::InvalidCommit { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Build facts captured when the neural crate was compiled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BuildInfo {
    /// Neural crate package version.
    pub package_version: String,
    /// Full Git commit, when available.
    pub git_commit: Option<String>,
    /// Whether Git could be queried at build time.
    pub git_available: bool,
    /// Whether the source tree had uncommitted changes at build time.
    pub git_dirty: bool,
    /// First line of `rustc --version`, when available.
    pub rustc_version: Option<String>,
    /// Cargo profile (`debug`, `release`, or a custom name).
    pub profile: String,
    /// Compiler optimization level recorded by Cargo.
    pub opt_level: String,
    /// Cargo target triple.
    pub target: String,
    /// Whether debug assertions were enabled.
    pub debug_assertions: bool,
}

/// Returns the current binary's compile-time build facts.
pub fn build_info() -> BuildInfo {
    let git_commit = option_env!("BC_GIT_COMMIT")
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let rustc_version = option_env!("BC_RUSTC_VERSION")
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    BuildInfo {
        package_version: env!("CARGO_PKG_VERSION").to_owned(),
        git_commit,
        git_available: option_env!("BC_GIT_AVAILABLE") == Some("1"),
        // Unknown provenance is treated as dirty so it cannot be published as a
        // clean commit by accident.
        git_dirty: option_env!("BC_GIT_DIRTY") != Some("0"),
        rustc_version,
        profile: option_env!("BC_BUILD_PROFILE")
            .unwrap_or("unknown")
            .to_owned(),
        opt_level: option_env!("BC_BUILD_OPT_LEVEL")
            .unwrap_or("unknown")
            .to_owned(),
        target: option_env!("BC_BUILD_TARGET")
            .unwrap_or("unknown")
            .to_owned(),
        debug_assertions: cfg!(debug_assertions),
    }
}

/// Runtime platform facts relevant to pilot measurements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EnvironmentInfo {
    /// Operating system name.
    pub os: String,
    /// Target architecture.
    pub arch: String,
    /// Operating-system family.
    pub family: String,
    /// Useful parallelism reported by the standard library.
    pub available_parallelism: usize,
    /// Pointer width in bits.
    pub pointer_width: u32,
}

/// Returns the current process environment facts.
pub fn environment_info() -> EnvironmentInfo {
    EnvironmentInfo {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        family: std::env::consts::FAMILY.to_owned(),
        available_parallelism: std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
        pointer_width: (mem::size_of::<usize>() * 8) as u32,
    }
}
