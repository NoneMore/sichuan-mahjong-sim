//! Filesystem commit boundary: exclusive writer lock, pending shards, and
//! atomic manifest publication. No game or teacher execution belongs here.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use fs4::FileExt;
use sha2::{Digest, Sha256};
use tempfile::{Builder, NamedTempFile};

use super::error::{io, CollectionError, Result};
use super::format::{lower_hex, read_shard_file, ShardExpectations, ShardRecord};
use super::manifest::{lock_path, manifest_path, shard_path, Manifest, ShardEntry};
use super::seed::Seed;

/// Exclusive corpus writer lock.
///
/// Acquisition is nonblocking and dropping the guard explicitly unlocks it.
/// The lock file stays in place so all writers lock the same inode.
#[derive(Debug)]
pub struct CorpusLock {
    file: File,
}

impl CorpusLock {
    /// Creates `root` if needed and tries to acquire the exclusive `LOCK`.
    ///
    /// Returns [`CollectionError::CorpusBusy`] if another handle or process
    /// currently holds it.
    pub fn try_acquire(root: &Path) -> Result<Self> {
        fs::create_dir_all(root).map_err(|error| io(root, error))?;
        let path = lock_path(root);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|source| CollectionError::Lock {
                path: path.clone(),
                source,
            })?;
        match FileExt::try_lock(&file) {
            Ok(()) => Ok(Self { file }),
            Err(fs4::TryLockError::WouldBlock) => Err(CollectionError::CorpusBusy {
                root: root.to_owned(),
            }),
            Err(fs4::TryLockError::Error(source)) => Err(CollectionError::Lock { path, source }),
        }
    }
}

impl Drop for CorpusLock {
    fn drop(&mut self) {
        // A forked child can keep the open file description alive until exec.
        // Unlock before close so that inherited descriptors cannot retain it.
        let _ = FileExt::unlock(&self.file);
    }
}

/// A shard being streamed to a temporary file.
pub(super) struct ActiveShard {
    writer: BufWriter<NamedTempFile>,
    hasher: Sha256,
    first_seed: Seed,
    games: u64,
    decisions: u64,
}

impl ActiveShard {
    pub(super) fn games(&self) -> u64 {
        self.games
    }

    pub(super) fn complete_episode(&mut self, decisions: u64) {
        self.games += 1;
        self.decisions += decisions;
    }

    pub(super) fn new(shards: &Path, first_seed: Seed) -> Result<Self> {
        let temp = Builder::new()
            .prefix(".pending-shard-")
            .suffix(".jsonl.tmp")
            .tempfile_in(shards)
            .map_err(|error| io(shards, error))?;
        Ok(Self {
            writer: BufWriter::new(temp),
            hasher: Sha256::new(),
            first_seed,
            games: 0,
            decisions: 0,
        })
    }

    pub(super) fn write_record(&mut self, record: &ShardRecord) -> Result<()> {
        let path = self.writer.get_ref().path().to_owned();
        let mut writer = HashingWriter {
            inner: &mut self.writer,
            hasher: &mut self.hasher,
        };
        serde_json::to_writer(&mut writer, record)
            .map_err(|error| io(&path, io::Error::new(io::ErrorKind::InvalidData, error)))?;
        writer.write_all(b"\n").map_err(|error| io(&path, error))
    }

    pub(super) fn finish(self, root: &Path, manifest: &Manifest) -> Result<ShardEntry> {
        let Self {
            mut writer,
            hasher,
            first_seed,
            games,
            decisions,
        } = self;
        let temp_path = writer.get_ref().path().to_owned();
        writer.flush().map_err(|error| io(&temp_path, error))?;
        writer
            .get_ref()
            .as_file()
            .sync_all()
            .map_err(|error| io(&temp_path, error))?;
        let temp = writer
            .into_inner()
            .map_err(|error| io(&temp_path, error.into_error()))?;
        let parsed = read_shard_file(
            temp.path(),
            ShardExpectations {
                source_revision: &manifest.source_revision,
                first_seed,
                games,
                decisions,
                decision_limit: manifest.decision_limit,
                sha256: None,
            },
            |_| Ok(()),
        )?;
        let written_sha256 = lower_hex(&hasher.finalize());
        if parsed.sha256 != written_sha256 {
            return Err(CollectionError::ShardChecksumMismatch {
                path: temp.path().to_owned(),
                expected: written_sha256,
                found: parsed.sha256,
            });
        }
        let final_path = shard_path(root, first_seed);
        temp.persist(&final_path)
            .map_err(|error| io(&final_path, error.error))?;
        Ok(ShardEntry {
            first_seed,
            games,
            decisions,
            sha256: parsed.sha256,
        })
    }
}

struct HashingWriter<'a, W> {
    inner: &'a mut W,
    hasher: &'a mut Sha256,
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub(super) fn write_manifest_atomic(root: &Path, manifest: &Manifest) -> Result<()> {
    let path = manifest_path(root);
    manifest.validate_at(&path)?;
    let mut bytes =
        serde_json::to_vec_pretty(manifest).map_err(|error| CollectionError::InvalidManifest {
            path: path.clone(),
            message: format!("could not serialize manifest: {error}"),
        })?;
    bytes.push(b'\n');
    let mut temp = Builder::new()
        .prefix(".manifest-")
        .suffix(".tmp")
        .tempfile_in(root)
        .map_err(|error| io(root, error))?;
    temp.write_all(&bytes)
        .map_err(|error| io(temp.path(), error))?;
    temp.as_file()
        .sync_all()
        .map_err(|error| io(temp.path(), error))?;
    temp.persist(&path)
        .map_err(|error| io(&path, error.error))?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::CorpusLock;

    #[test]
    fn dropping_the_guard_unlocks_with_a_duplicated_descriptor_still_open() {
        let root = tempfile::tempdir().unwrap();
        let guard = CorpusLock::try_acquire(root.path()).unwrap();
        // A duplicated descriptor shares the open file description, as a fork does.
        let inherited = guard.file.try_clone().unwrap();
        drop(guard);

        let next = CorpusLock::try_acquire(root.path()).unwrap();
        drop(inherited);
        drop(next);
    }
}
