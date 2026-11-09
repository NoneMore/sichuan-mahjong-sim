//! Captures best-effort build provenance for reusable data collection.
//!
//! Provenance is fixed when the neural crate is compiled: the full Git commit,
//! whether the working tree was dirty, the Rust compiler version, and the
//! build profile/target. Collection compares these values instead of
//! re-reading the caller's current directory, so a binary built from dirty
//! source cannot claim the commit ID of a clean checkout.
//!
//! The build script never fails the build when Git or the compiler cannot be
//! queried. It records the unavailable state, and the collection API rejects
//! that artifact as a provenance source.

use std::path::Path;
use std::process::Command;

fn main() {
    // Cargo reruns this script only for watched inputs. Watch the workspace
    // source that contributes to the simulator, teacher, and collector so a
    // clean/dirty change in a sibling crate is reflected in the next build.
    println!("cargo:rerun-if-changed=../../src");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=RUSTC");

    watch_git_metadata();

    let commit =
        git_stdout(&["rev-parse", "--verify", "HEAD"]).map(|value| value.trim().to_owned());
    let dirty = git_status_dirty();
    let available = commit.is_some() && dirty.is_some();
    let commit = commit.filter(|value| !value.is_empty());

    emit("BC_GIT_AVAILABLE", if available { "1" } else { "0" });
    emit(
        "BC_GIT_DIRTY",
        if dirty.unwrap_or(true) { "1" } else { "0" },
    );
    emit("BC_GIT_COMMIT", commit.as_deref().unwrap_or(""));

    let rustc = std::env::var_os("RUSTC")
        .and_then(|path| {
            Command::new(path)
                .arg("--version")
                .output()
                .ok()
                .and_then(|output| {
                    if output.status.success() {
                        String::from_utf8(output.stdout).ok()
                    } else {
                        None
                    }
                })
        })
        .map(|value| value.lines().next().unwrap_or("").trim().to_owned())
        .filter(|value| !value.is_empty());
    emit("BC_RUSTC_VERSION", rustc.as_deref().unwrap_or(""));

    emit(
        "BC_BUILD_PROFILE",
        &std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "BC_BUILD_OPT_LEVEL",
        &std::env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "BC_BUILD_TARGET",
        &std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "BC_BUILD_DEBUG",
        &std::env::var("DEBUG").unwrap_or_else(|_| "unknown".to_owned()),
    );
}

fn emit(key: &str, value: &str) {
    println!("cargo:rustc-env={key}={value}");
}

fn git_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn git_status_dirty() -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(!output.stdout.is_empty())
}

/// Watch Git reference/index files so a branch or commit change can refresh the
/// recorded commit even when the working source files are unchanged.
fn watch_git_metadata() {
    let Some(git_dir) = git_stdout(&["rev-parse", "--absolute-git-dir"]) else {
        return;
    };
    let git_dir = Path::new(git_dir.trim());
    for relative in ["HEAD", "index", "packed-refs", "refs/heads"] {
        let path = git_dir.join(relative);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
