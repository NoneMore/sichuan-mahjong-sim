from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import mahjong_bc
from mahjong_bc._source import (
    collect_source_record,
    collect_training_source,
    hash_source_directory,
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _write_package(root: Path, *, contents: str = "value = 1\n") -> Path:
    package = root / "mahjong_bc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(contents, encoding="utf-8")
    (package / "module.py").write_text("value = 2\n", encoding="utf-8")
    return package


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Source Test",
        "-c",
        "user.email=source-test@example.invalid",
        "commit",
        "-m",
        "initial",
    )


def test_source_hash_is_stable_filename_sensitive_and_excludes_pycache(
    tmp_path, monkeypatch
):
    package = _write_package(tmp_path / "first")
    ignored = package / "__pycache__"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    first = hash_source_directory(package)
    assert _HEX_SHA256.fullmatch(first)

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    assert hash_source_directory(package) == first
    assert hash_source_directory(tmp_path / "first" / "mahjong_bc") == first

    (ignored / "ignored.py").write_text("ignored = False\n", encoding="utf-8")
    assert hash_source_directory(package) == first

    copy = _write_package(tmp_path / "copy")
    assert hash_source_directory(copy) == first

    (package / "__init__.py").write_text("value = 99\n", encoding="utf-8")
    assert hash_source_directory(package) != first


def test_training_source_record_has_package_hash_and_git_fields(
    tmp_path, monkeypatch
):
    module_directory = Path(mahjong_bc.__file__).resolve().parent
    monkeypatch.chdir(tmp_path)
    first = collect_training_source()
    second = collect_training_source()
    assert set(first) == {
        "package",
        "package_version",
        "hash",
        "revision",
        "dirty",
    }
    assert first["package"] == "mahjong-bc"
    assert first["package_version"] == mahjong_bc.__version__
    assert _HEX_SHA256.fullmatch(first["hash"])
    assert first["hash"] == hash_source_directory(module_directory)
    assert first["revision"] == second["revision"]
    assert first["dirty"] == second["dirty"]
    assert first["revision"] is None or re.fullmatch(
        r"[0-9a-f]{40}", first["revision"]
    )
    assert first["dirty"] is None or isinstance(first["dirty"], bool)


def test_git_metadata_is_anchored_to_the_module_path(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")

    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "README.md").write_text("outer\n", encoding="utf-8")
    _init_repo(outer)
    wheel_package = _write_package(
        outer / "venv" / "lib" / "site-packages"
    )
    wheel_record = collect_source_record(
        wheel_package, git_executable=git
    )
    assert wheel_record["revision"] is None
    assert wheel_record["dirty"] is None
    assert _HEX_SHA256.fullmatch(wheel_record["hash"])

    repo = tmp_path / "repo"
    package = _write_package(repo / "python")
    _init_repo(repo)
    expected_revision = _git(repo, "rev-parse", "HEAD").stdout.strip()
    record = collect_source_record(package, git_executable=git)
    assert record["revision"] == expected_revision
    assert record["dirty"] is False

    (package / "module.py").write_text("value = 3\n", encoding="utf-8")
    dirty_record = collect_source_record(package, git_executable=git)
    assert dirty_record["revision"] == expected_revision
    assert dirty_record["dirty"] is True
    assert dirty_record["hash"] != record["hash"]


def test_git_failure_and_non_repo_paths_still_report_a_source_hash(tmp_path):
    module_directory = Path(mahjong_bc.__file__).resolve().parent
    missing_git = collect_source_record(
        module_directory, git_executable="definitely-not-a-real-git-command"
    )
    assert _HEX_SHA256.fullmatch(missing_git["hash"])
    assert missing_git["hash"] == hash_source_directory(module_directory)
    assert missing_git["revision"] is None
    assert missing_git["dirty"] is None

    package = _write_package(tmp_path / "plain")
    plain = collect_source_record(package)
    assert _HEX_SHA256.fullmatch(plain["hash"])
    assert plain["revision"] is None
    assert plain["dirty"] is None
