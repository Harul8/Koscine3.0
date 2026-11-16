"""Rules every file in src/koscine4 must follow (requirements N-1, N-2, N-3, N-8, F-2)."""
from __future__ import annotations

import ast
import importlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import koscine4
from koscine4.legacy import LEGACY_PACKAGES, REVIEWED_LEGACY_MODULES
from koscine4.settings import project_root

ROOT = project_root()
PKG = ROOT / "src" / "koscine4"
SOURCES = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_has_sources():
    assert len(SOURCES) >= len(koscine4.COMPONENTS) + 1


@pytest.mark.parametrize("name", koscine4.COMPONENTS)
def test_every_component_imports_and_states_its_job(name):
    mod = importlib.import_module(f"koscine4.{name}")
    assert mod.__doc__ and "Requirements:" in mod.__doc__


def test_component_folders_match_the_declared_list():
    folders = {p.name for p in PKG.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert folders == set(koscine4.COMPONENTS)


# N-2: no machine-specific paths. Windows drive letters, and Unix home folders.
_MACHINE_PATH = re.compile(r"""["'](?:[A-Za-z]:[\\/]|/home/|/Users/)""")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_machine_specific_paths(path):
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _MACHINE_PATH.search(line)
    ]
    assert not hits, f"hard-coded machine path in {path.name}: {hits}"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


# N-3: old code is reachable only through the reviewed list.
@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_unreviewed_imports_of_old_code(path):
    bad = sorted(
        m
        for m in _imported_modules(path)
        if m.split(".")[0] in LEGACY_PACKAGES and m not in REVIEWED_LEGACY_MODULES
    )
    assert not bad, f"{path.name} imports old code that is not on the reviewed list: {bad}"


def test_the_import_scan_catches_a_planted_import(tmp_path):
    f = tmp_path / "planted.py"
    f.write_text("from koscine3.largemove.mover_v3 import load_panel\nimport pipeline.train\n")
    mods = _imported_modules(f)
    assert {"koscine3.largemove.mover_v3", "pipeline.train"} <= mods
    assert all(m.split(".")[0] in LEGACY_PACKAGES for m in mods)


# F-2: features are never chosen by excluding column names.
_BLOCKLIST = re.compile(r"\b(LEAK|BLOCKED_PREFIXES|EXCLUDE_PREFIXES|BLOCKED_EXACT)\s*=")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_name_based_feature_blocklist(path):
    assert not _BLOCKLIST.search(path.read_text(encoding="utf-8"))


# N-8: nothing that looks like a key or password.
_SECRET = re.compile(r"""(?i)\b(api[_-]?key|secret|password|access[_-]?token)\b\s*=\s*["'][^"']{8,}["']""")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_keys_or_passwords(path):
    assert not _SECRET.search(path.read_text(encoding="utf-8"))


# N-1: source folders must be visible to git, so a fresh clone has them.
# Every koscine4 source file is checked, because a top-level ignore rule can match a
# package folder of the same name (data/, models/ and Universe/ all did).
@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_every_package_file_is_visible_to_git(path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    rel = path.relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    assert r.returncode == 1, f"{rel} is ignored by .gitignore, so a fresh clone would lack it"


@pytest.mark.parametrize(
    "rel",
    [
        "src/koscine3/data/sources.py",
        "src/koscine3/models/train.py",
        "configs/settings.toml",
        "records/README.md",
    ],
)
def test_source_files_are_not_git_ignored(rel):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert (ROOT / rel).exists(), f"{rel} is missing"
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    assert r.returncode == 1, f"{rel} is ignored by .gitignore, so a fresh clone would lack it"


@pytest.mark.parametrize("rel", ["data/silver/eod_stock.parquet", "models/x.cbm", "runs/r1/report.json"])
def test_data_and_outputs_stay_ignored(rel):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    assert r.returncode == 0, f"{rel} should stay out of git"
