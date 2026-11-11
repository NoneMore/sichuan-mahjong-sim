//! Captures extension-scoped source identity at compile time.
//!
//! The online extension combines its own binding crate, the simulator core, and
//! the neural encoder. The neural crate's collection provenance covers only the
//! build of that crate, so this build script records a second, extension-scoped
//! identity: the full Git commit and dirty flag when Git is available, plus a
//! SHA-256 content hash over the sorted relative paths and bytes of every
//! source, manifest, and lockfile that contributes to the extension.
//!
//! The hash is computed from the build-time source tree, never from the
//! caller's working directory. Missing source files mark the identity
//! incomplete instead of silently hashing a subset. Git failures do not fail
//! the build; they are recorded as unavailable.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

/// Relative source scopes included in the content identity.
const SOURCE_SCOPE: [&str; 3] = ["crates/python", "src", "crates/neural"];

fn main() {
    watch_build_inputs();
    watch_git_metadata();

    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("extension crate lives two levels below the repository root")
        .to_path_buf();

    let mut files = Vec::new();
    let mut complete = true;
    collect_extension_files(&root, &mut files, &mut complete);
    collect_core_files(&root, &mut files, &mut complete);
    collect_neural_files(&root, &mut files, &mut complete);
    files.sort();

    let mut hasher = Sha256::new();
    let mut listed = String::new();
    for relative in &files {
        let path = root.join(relative);
        match fs::read(&path) {
            Ok(bytes) => {
                hasher.update(relative.as_bytes());
                hasher.update([0u8]);
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
                listed.push_str(relative);
                listed.push(';');
            }
            Err(_) => complete = false,
        }
    }
    emit("EXT_SOURCE_SHA256", &hex(&hasher.finalize()));
    emit("EXT_SOURCE_FILES", &listed);
    emit("EXT_SOURCE_COMPLETE", if complete { "1" } else { "0" });
    emit(
        "EXT_SOURCE_IDENTITY",
        &format!("sha256-paths-bytes-v1:{}", SOURCE_SCOPE.join("+")),
    );
    emit("EXT_SOURCE_SCOPE", &SOURCE_SCOPE.join("+"));

    let commit =
        git_stdout(&root, &["rev-parse", "--verify", "HEAD"]).map(|value| value.trim().to_owned());
    let dirty = git_status_dirty(&root);
    let available = commit.as_ref().is_some_and(|value| !value.is_empty()) && dirty.is_some();
    emit("EXT_GIT_AVAILABLE", if available { "1" } else { "0" });
    emit(
        "EXT_GIT_DIRTY",
        if dirty.unwrap_or(true) { "1" } else { "0" },
    );
    emit(
        "EXT_GIT_COMMIT",
        commit
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(""),
    );

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
    emit("EXT_RUSTC_VERSION", rustc.as_deref().unwrap_or(""));
    emit(
        "EXT_BUILD_PROFILE",
        &std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "EXT_BUILD_OPT_LEVEL",
        &std::env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "EXT_BUILD_TARGET",
        &std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned()),
    );
    emit(
        "EXT_BUILD_DEBUG",
        &std::env::var("DEBUG").unwrap_or_else(|_| "unknown".to_owned()),
    );
}

fn watch_build_inputs() {
    for relative in [
        "src",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "../../src",
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../neural/src",
        "../neural/Cargo.toml",
        "../neural/build.rs",
    ] {
        println!("cargo:rerun-if-changed={relative}");
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
}

fn collect_extension_files(root: &Path, out: &mut Vec<String>, complete: &mut bool) {
    for relative in [
        "crates/python/Cargo.toml",
        "crates/python/Cargo.lock",
        "crates/python/build.rs",
    ] {
        push_required(root, relative, out, complete);
    }
    collect_tree(root, "crates/python/src", out, complete);
}

fn collect_core_files(root: &Path, out: &mut Vec<String>, complete: &mut bool) {
    for relative in ["Cargo.toml", "Cargo.lock"] {
        push_required(root, relative, out, complete);
    }
    collect_tree(root, "src", out, complete);
}

fn collect_neural_files(root: &Path, out: &mut Vec<String>, complete: &mut bool) {
    for relative in ["crates/neural/Cargo.toml", "crates/neural/build.rs"] {
        push_required(root, relative, out, complete);
    }
    collect_tree(root, "crates/neural/src", out, complete);
}

fn push_required(root: &Path, relative: &str, out: &mut Vec<String>, complete: &mut bool) {
    if root.join(relative).is_file() {
        out.push(relative.to_owned());
    } else {
        *complete = false;
    }
}

/// Adds every Rust file below `relative_dir` in deterministic path order.
fn collect_tree(root: &Path, relative_dir: &str, out: &mut Vec<String>, complete: &mut bool) {
    let directory = root.join(relative_dir);
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(_) => {
            *complete = false;
            return;
        }
    };
    for entry in entries {
        let Ok(entry) = entry else {
            *complete = false;
            continue;
        };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            *complete = false;
            continue;
        };
        if path.is_dir() {
            let child = format!("{relative_dir}/{name}");
            collect_tree(root, &child, out, complete);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            let Ok(relative) = path.strip_prefix(root) else {
                *complete = false;
                continue;
            };
            let text = relative.to_string_lossy().replace('\\', "/");
            out.push(text);
        }
    }
}

fn emit(key: &str, value: &str) {
    println!("cargo:rustc-env={key}={value}");
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn git_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn git_status_dirty(root: &Path) -> Option<bool> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(!output.stdout.is_empty())
}

/// Watches Git reference/index files so a commit or branch change refreshes
/// recorded provenance even when working source files are unchanged.
fn watch_git_metadata() {
    let Some(git_dir) = git_stdout(Path::new("."), &["rev-parse", "--absolute-git-dir"]) else {
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
