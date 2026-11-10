//! Errors from dataset preparation, publication, and cache verification.

use std::fmt;
use std::io;
use std::path::PathBuf;

use crate::collection::CollectionError;

/// Result type for dataset preparation and cache reading.
pub type Result<T> = std::result::Result<T, PrepareError>;

/// A preparation, publication, or prepared-cache verification failure.
#[derive(Debug)]
pub enum PrepareError {
    /// Strict corpus snapshot reading failed.
    Collection(CollectionError),
    /// A filesystem operation failed.
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying failure.
        source: io::Error,
    },
    /// A preparation option violates a positive or memory-bound requirement.
    InvalidOptions {
        /// Human-readable cause.
        message: String,
    },
    /// The output cache directory already exists.
    OutputExists {
        /// Requested output path.
        path: PathBuf,
    },
    /// The captured corpus contains no committed games.
    EmptyCorpus,
    /// `dataset.json` could not be decoded.
    DatasetJson {
        /// Manifest path.
        path: PathBuf,
        /// Decoder failure.
        source: serde_json::Error,
    },
    /// The cache format version is not supported.
    UnsupportedFormatVersion {
        /// Path of the manifest that declared the version.
        path: PathBuf,
        /// Version found in the file.
        found: u32,
        /// Supported version.
        supported: u32,
    },
    /// The preparation algorithm version is not supported.
    UnsupportedPreparationVersion {
        /// Path of the manifest that declared the version.
        path: PathBuf,
        /// Version found in the file.
        found: u32,
        /// Supported version.
        supported: u32,
    },
    /// The cache declares an unsupported decision encoding.
    UnsupportedEncodingVersion {
        /// Path of the manifest that declared the version.
        path: PathBuf,
        /// Version found in the file.
        found: u32,
        /// Supported version.
        supported: u32,
    },
    /// Cache manifest metadata violates the format invariants.
    InvalidManifest {
        /// Manifest path.
        path: PathBuf,
        /// Human-readable cause.
        message: String,
    },
    /// A listed array file violates its declared dtype, shape, or header.
    InvalidArray {
        /// Array path.
        path: PathBuf,
        /// Human-readable cause.
        message: String,
    },
    /// An array file checksum differs from the manifest.
    ChecksumMismatch {
        /// Array path.
        path: PathBuf,
        /// Manifest checksum.
        expected: String,
        /// Recomputed checksum.
        found: String,
    },
    /// The stored cache identity does not match the manifest inventory.
    CacheIdMismatch {
        /// Manifest value.
        expected: String,
        /// Recomputed value.
        found: String,
    },
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrepareError::Collection(error) => write!(f, "{error}"),
            PrepareError::Io { path, source } => {
                write!(f, "I/O error at {}: {source}", path.display())
            }
            PrepareError::InvalidOptions { message } => {
                write!(f, "invalid preparation options: {message}")
            }
            PrepareError::OutputExists { path } => write!(
                f,
                "output cache {} already exists; preparation never overwrites it",
                path.display()
            ),
            PrepareError::EmptyCorpus => {
                f.write_str("the captured corpus contains no committed games")
            }
            PrepareError::DatasetJson { path, source } => {
                write!(f, "could not decode {}: {source}", path.display())
            }
            PrepareError::UnsupportedFormatVersion {
                path,
                found,
                supported,
            } => write!(
                f,
                "{} uses unsupported preparation format version {found} (supported: {supported})",
                path.display()
            ),
            PrepareError::UnsupportedPreparationVersion {
                path,
                found,
                supported,
            } => write!(
                f,
                "{} uses unsupported preparation version {found} (supported: {supported})",
                path.display()
            ),
            PrepareError::UnsupportedEncodingVersion {
                path,
                found,
                supported,
            } => write!(
                f,
                "{} uses unsupported encoding version {found} (supported: {supported})",
                path.display()
            ),
            PrepareError::InvalidManifest { path, message } => {
                write!(f, "invalid dataset manifest {}: {message}", path.display())
            }
            PrepareError::InvalidArray { path, message } => {
                write!(f, "invalid cache array {}: {message}", path.display())
            }
            PrepareError::ChecksumMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "cache array {} checksum mismatch: manifest {expected}, actual {found}",
                path.display()
            ),
            PrepareError::CacheIdMismatch { expected, found } => write!(
                f,
                "cache_id mismatch: manifest {expected}, computed {found}"
            ),
        }
    }
}

impl std::error::Error for PrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PrepareError::Collection(error) => Some(error),
            PrepareError::Io { source, .. } => Some(source),
            PrepareError::DatasetJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<CollectionError> for PrepareError {
    fn from(error: CollectionError) -> Self {
        PrepareError::Collection(error)
    }
}

pub(super) fn io(path: impl Into<PathBuf>, source: io::Error) -> PrepareError {
    PrepareError::Io {
        path: path.into(),
        source,
    }
}

pub(super) fn invalid_manifest(path: &std::path::Path, message: impl Into<String>) -> PrepareError {
    PrepareError::InvalidManifest {
        path: path.to_owned(),
        message: message.into(),
    }
}

pub(super) fn invalid_array(path: &std::path::Path, message: impl Into<String>) -> PrepareError {
    PrepareError::InvalidArray {
        path: path.to_owned(),
        message: message.into(),
    }
}

/// Maps a preparation failure into the collection error type used by the
/// fallible snapshot visitor. I/O context is preserved; other failures become a
/// descriptive configuration error so the visitor stops immediately.
pub(super) fn into_collection(error: PrepareError) -> CollectionError {
    match error {
        PrepareError::Collection(error) => error,
        PrepareError::Io { path, source } => CollectionError::Io { path, source },
        other => CollectionError::InvalidConfig {
            message: other.to_string(),
        },
    }
}
