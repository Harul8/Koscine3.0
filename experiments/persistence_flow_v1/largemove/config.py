"""LOCKED production configuration for the Large-Move engine. Do not edit in experiments —
clone the package into experiments/<name>/ and edit the copy (see experiments/EXPERIMENT_POLICY.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VERSION = "exp_persistence_flow_v1"

# 20 lean, level-dominated features (ablation-validated; more features add ~nothing).
LEAN_FEATURES: tuple[str, ...] = (
    "atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
    "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
    "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
    "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month",
)

# Per-group move thresholds (A mega-caps move less -> lower bar).
GROUP_THRESHOLDS: dict[str, float] = {"A_mcap30": 0.03, "B_turn35": 0.04}

XGB_CLF_PARAMS: dict = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                            colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)
XGB_REG_PARAMS: dict = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                            colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)


@dataclass(frozen=True)
class LargeMoveConfig:
    version: str = VERSION
    window_days: int = 5
    cooldown_trading_days: int = 3          # picked t -> blocked t+1..t+2, repeat t+3
    min_underlying: float = 100.0           # non-penny eligibility (close >= 100)
    requires_optionable: bool = True        # eligible only when atm_iv present that day
    features: tuple[str, ...] = LEAN_FEATURES
    group_thresholds: tuple[tuple[str, float], ...] = tuple(GROUP_THRESHOLDS.items())
    # walk-forward: base-fit < T-1, isotonic-calibrate on T-1, predict T
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    eval_end: str = "2026-05-31"
    universe_groups_file: str = "universe_groups.json"   # locked alongside this config

    def thr(self, group: str) -> float:
        return dict(self.group_thresholds)[group]


PROD = LargeMoveConfig()

# Canonical locations (PROD artifacts live under locks/prod_largemove_v1/)
LOCK_DIR = Path(__file__).resolve().parents[1]   # sandbox root (experiment-local artifacts)
MODELS_DIR = LOCK_DIR / "models"
PREDICTIONS_DIR = LOCK_DIR / "predictions"
