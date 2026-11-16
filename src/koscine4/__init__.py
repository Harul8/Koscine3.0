"""Koscine version 1 rebuild.

One pipeline shared by the backtest and the daily run, so what is tested is what trades.
A strategy reaches the trade list only through the test harness.

Components (one sub-package each): store, universe, features, outcomes, models, strategies,
harness, selection, risk, daily, ledger, api.

This package does not import the older `koscine`, `koscine3`, `pipeline` or `analysis` code,
except modules listed in `koscine4.legacy.REVIEWED_LEGACY_MODULES`.
"""

__version__ = "0.0.1"

COMPONENTS = (
    "store",
    "universe",
    "features",
    "outcomes",
    "models",
    "strategies",
    "harness",
    "selection",
    "risk",
    "daily",
    "ledger",
    "api",
)
