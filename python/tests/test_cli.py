from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

import mahjong_bc
from mahjong_bc.cli import main
from mahjong_bc.constants import PACKAGE_VERSION

from .cache_fixture import synthetic_cache


def _parse(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "dataset.json").read_text(encoding="utf-8"))


def test_cache_info_reports_manifest_inventory(tmp_path, capsys):
    cache = synthetic_cache(tmp_path / "cache")
    manifest = _read_manifest(cache)

    assert main(["cache-info", "--dataset", str(cache)]) == 0
    info = _parse(capsys)

    assert info["command"] == "cache-info"
    assert info["package_version"] == PACKAGE_VERSION
    for key in (
        "format",
        "cache_id",
        "encoding_version",
        "preparation_format_version",
        "preparation_version",
        "source_revision",
        "corpus_manifest_sha256",
        "split_algorithm",
        "counts",
    ):
        assert info[key] == manifest[key]
    assert set(info["splits"]) == {"train", "validation", "test"}
    for split in info["splits"]:
        assert info["splits"][split]["counts"] == manifest["splits"][split]["counts"]
        assert info["splits"][split]["shards"] == len(
            manifest["splits"][split]["shards"]
        )
    assert info["counts"]["decisions"] == sum(
        info["splits"][split]["counts"]["decisions"] for split in info["splits"]
    )
    assert info["counts"]["eligible_decisions"] > 0


def test_cache_info_rejects_missing_cache(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert main(["cache-info", "--dataset", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "dataset.json" in captured.err


def test_version_reports_package_and_dependency_versions(capsys):
    assert main(["version"]) == 0
    version = _parse(capsys)
    assert version["package"] == "mahjong-bc"
    assert version["package_version"] == PACKAGE_VERSION == mahjong_bc.__version__
    dependencies = version["dependencies"]
    assert set(dependencies) == {"python", "torch", "numpy", "cuda_build"}
    assert dependencies["python"]
    assert dependencies["torch"]
    assert dependencies["numpy"]
    assert dependencies["cuda_build"] is None or isinstance(
        dependencies["cuda_build"], str
    )


def test_help_lists_available_commands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "cache-info" in output
    assert "version" in output

    with pytest.raises(SystemExit) as exc_info:
        main(["cache-info", "--help"])
    assert exc_info.value.code == 0
    assert "--dataset" in capsys.readouterr().out


def test_module_entry_point_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "mahjong_bc", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["package"] == "mahjong-bc"


def test_installed_metadata_and_console_script():
    try:
        distribution = importlib_metadata.distribution("mahjong-bc")
    except importlib_metadata.PackageNotFoundError:
        pytest.skip("package is not installed")
    assert distribution.version == mahjong_bc.__version__
    entry_points = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert entry_points.get("mahjong-bc") == "mahjong_bc.cli:main"

    executable = Path(sys.executable).parent / (
        "mahjong-bc.exe" if os.name == "nt" else "mahjong-bc"
    )
    if not executable.is_file():
        pytest.skip("console script is not installed next to the interpreter")
    result = subprocess.run(
        [str(executable), "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["package"] == "mahjong-bc"
