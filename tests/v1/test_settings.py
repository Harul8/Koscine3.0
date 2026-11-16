"""Settings loader: paths come from one file, and risk settings block a live list until set."""
from __future__ import annotations

from pathlib import Path

import pytest

from koscine4.settings import (
    REQUIRED_RISK_KEYS,
    SettingsError,
    load_settings,
    project_root,
)

BASE = """
[paths]
silver = "data/silver"
pit = "data/pit"
[universe]
top_n = 30
[harness]
gap_trading_days = 10
[selection]
max_new_trades_per_week = 5
[costs]
brokerage_per_order = 20.0
"""


def _write(tmp_path: Path, text: str) -> Path:
    f = tmp_path / "settings.toml"
    f.write_text(text, encoding="utf-8")
    return f


def test_repo_settings_load_and_match_requirements():
    s = load_settings()
    assert s.root == project_root()
    assert (s.root / "pyproject.toml").exists()
    assert s.selection["max_new_trades_per_week"] == 5      # S-7
    assert s.universe["top_n"] == 30                        # U-1
    assert s.harness["gap_trading_days"] >= 10              # V-3
    assert s.harness["sealed_months"] == 12                 # V-12
    for name in ("silver", "pit", "outcomes", "runs", "records", "strategies", "model_store"):
        assert s.path(name).is_relative_to(s.root)


def test_inputs_and_outcomes_live_in_separate_folders():
    s = load_settings()
    pit, outcomes = s.path("pit"), s.path("outcomes")
    assert pit != outcomes
    assert not outcomes.is_relative_to(pit) and not pit.is_relative_to(outcomes)   # D-2


def test_repo_risk_settings_are_unset_and_block_a_live_list():
    s = load_settings()
    assert s.missing_risk_keys() == list(REQUIRED_RISK_KEYS)
    with pytest.raises(SettingsError, match="capital"):
        s.require_risk()                                     # R-7


def test_complete_risk_settings_pass(tmp_path):
    risk = "\n[risk]\n" + "\n".join(f"{k} = 1" for k in REQUIRED_RISK_KEYS)
    s = load_settings(root=tmp_path, file=_write(tmp_path, BASE + risk))
    assert s.missing_risk_keys() == []
    assert s.require_risk()["capital"] == 1


def test_one_missing_risk_setting_is_named(tmp_path):
    keys = [k for k in REQUIRED_RISK_KEYS if k != "drawdown_pause_pct"]
    risk = "\n[risk]\n" + "\n".join(f"{k} = 1" for k in keys)
    s = load_settings(root=tmp_path, file=_write(tmp_path, BASE + risk))
    with pytest.raises(SettingsError, match="drawdown_pause_pct"):
        s.require_risk()


def test_absolute_paths_are_refused(tmp_path):
    bad = BASE.replace('silver = "data/silver"', f'silver = "{(tmp_path / "x").as_posix()}"')
    with pytest.raises(SettingsError, match="relative"):
        load_settings(root=tmp_path, file=_write(tmp_path, bad))               # N-2


def test_missing_section_and_missing_file_are_reported(tmp_path):
    with pytest.raises(SettingsError, match="costs"):
        load_settings(root=tmp_path, file=_write(tmp_path, BASE.split("[costs]")[0]))
    with pytest.raises(SettingsError, match="not found"):
        load_settings(root=tmp_path, file=tmp_path / "nope.toml")


def test_unknown_path_name_is_reported():
    with pytest.raises(SettingsError, match="no_such_path"):
        load_settings().path("no_such_path")


def test_root_can_be_overridden_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KOSCINE_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
