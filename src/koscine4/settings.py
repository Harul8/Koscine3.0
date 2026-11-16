"""Settings loader (requirements N-2, N-6, R-7).

Every path, cost and limit comes from `configs/settings.toml`. Nothing in the package
hard-codes a machine path or a threshold.

The project root is the folder that holds `pyproject.toml`. Set the `KOSCINE_ROOT`
environment variable to point somewhere else (used by tests and by a second checkout).
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SETTINGS_FILE = Path("configs") / "settings.toml"

# Risk settings the owner must fill in before any live list is produced (R-7).
REQUIRED_RISK_KEYS = (
    "capital",
    "risk_per_trade_pct",
    "max_open_risk_pct",
    "drawdown_halve_pct",
    "drawdown_pause_pct",
    "expiry_safety_days",
    "index_book_trades_per_week",
    "stock_book_trades_per_week",
    "max_correlated_trades",
)


class SettingsError(RuntimeError):
    """Raised when the settings file is missing, malformed or incomplete."""


def project_root() -> Path:
    env = os.environ.get("KOSCINE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SettingsError("Could not find the project root (no pyproject.toml above this file).")


@dataclass(frozen=True)
class Settings:
    root: Path
    paths: Mapping[str, Path]
    universe: Mapping[str, Any]
    harness: Mapping[str, Any]
    selection: Mapping[str, Any]
    costs: Mapping[str, Any]
    risk: Mapping[str, Any] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        try:
            return self.paths[name]
        except KeyError:
            raise SettingsError(f"No path named '{name}' in [paths] of {SETTINGS_FILE}.") from None

    def missing_risk_keys(self) -> list[str]:
        return [k for k in REQUIRED_RISK_KEYS if self.risk.get(k) is None]

    def require_risk(self) -> Mapping[str, Any]:
        """Return the risk settings, or stop if the owner has not set them all (R-7)."""
        missing = self.missing_risk_keys()
        if missing:
            raise SettingsError(
                "Risk settings are not complete, so no live trade list can be produced. "
                f"Set these in [risk] of {SETTINGS_FILE}: {', '.join(missing)}"
            )
        return self.risk


def load_settings(root: Path | None = None, file: Path | None = None) -> Settings:
    root = Path(root).resolve() if root is not None else project_root()
    file = Path(file) if file is not None else root / SETTINGS_FILE
    if not file.exists():
        raise SettingsError(f"Settings file not found: {file}")
    with file.open("rb") as fh:
        raw = tomllib.load(fh)

    for section in ("paths", "universe", "harness", "selection", "costs"):
        if section not in raw:
            raise SettingsError(f"Section [{section}] is missing from {file}")

    paths: dict[str, Path] = {}
    for name, value in raw["paths"].items():
        p = Path(value)
        if p.is_absolute():
            raise SettingsError(
                f"[paths].{name} must be relative to the project root, got an absolute path: {value}"
            )
        paths[name] = root / p

    return Settings(
        root=root,
        paths=paths,
        universe=dict(raw["universe"]),
        harness=dict(raw["harness"]),
        selection=dict(raw["selection"]),
        costs=dict(raw["costs"]),
        risk=dict(raw.get("risk", {})),
    )
