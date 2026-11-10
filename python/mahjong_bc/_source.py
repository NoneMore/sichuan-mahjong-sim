"""Training-source identity for the imported ``mahjong_bc`` package.

The hash covers the Python source files actually imported by this installation,
not the process working directory. Relative file names and file contents are
length-delimited so the canonical byte stream is unambiguous. Git metadata is
attached only when this module lives at ``<git root>/python/mahjong_bc``;
a wheel installed inside some unrelated outer repository therefore reports
``revision``/``dirty`` as null instead of borrowing that repository's state.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .constants import PACKAGE_VERSION

_PACKAGE = "mahjong-bc"
_DOMAIN = b"mahjong-bc-python-source-v1\0"
_REPOSITORY_PACKAGE = Path("python") / "mahjong_bc"


def _source_files(module_directory: Path) -> list[Path]:
    """Return sorted Python sources below one package directory."""

    return sorted(
        (
            path
            for path in module_directory.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(module_directory).as_posix(),
    )


def hash_source_directory(module_directory: Path | str) -> str:
    """Return the stable SHA-256 of one package's sorted Python sources."""

    directory = Path(module_directory).resolve()
    files = _source_files(directory)
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(len(files).to_bytes(8, "big"))
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _git_output(
    directory: Path, arguments: list[str], *, git_executable: str
) -> str | None:
    try:
        completed = subprocess.run(
            [git_executable, "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_source_identity(
    module_directory: Path, *, git_executable: str
) -> tuple[str | None, bool | None]:
    root_text = _git_output(
        module_directory,
        ["rev-parse", "--show-toplevel"],
        git_executable=git_executable,
    )
    if root_text is None:
        return None, None
    try:
        root = Path(root_text.strip()).resolve()
    except OSError:
        return None, None
    if (root / _REPOSITORY_PACKAGE).resolve() != module_directory:
        return None, None
    revision_text = _git_output(
        root, ["rev-parse", "HEAD"], git_executable=git_executable
    )
    status_text = _git_output(
        root, ["status", "--porcelain"], git_executable=git_executable
    )
    revision = None
    if revision_text is not None:
        candidate = revision_text.strip()
        if candidate:
            revision = candidate
    dirty = None if status_text is None else bool(status_text.strip())
    return revision, dirty


def collect_source_record(
    module_directory: Path | str,
    *,
    git_executable: str = "git",
) -> dict[str, object]:
    """Build the source identity for one package directory.

    ``module_directory`` is the directory containing this imported package.
    Supplying it explicitly keeps tests and callers anchored to the module path
    rather than whatever repository contains the process working directory.
    """

    directory = Path(module_directory).resolve()
    revision, dirty = _git_source_identity(
        directory, git_executable=git_executable
    )
    return {
        "package": _PACKAGE,
        "package_version": PACKAGE_VERSION,
        "hash": hash_source_directory(directory),
        "revision": revision,
        "dirty": dirty,
    }


def collect_training_source() -> dict[str, object]:
    """Capture the imported package directory once for a training run."""

    return collect_source_record(Path(__file__).resolve().parent)
