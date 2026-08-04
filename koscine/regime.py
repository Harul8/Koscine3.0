from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import SILVER_DATA_ROOT


@dataclass(frozen=True)
class RegimeConfig:
    long_score_normal: float = 0.65
    long_score_caution: float = 0.80
    short_score_normal: float = 0.70
    short_score_caution: float = 0.85
    vol_quantile_window: int = 252
    sma_window: int = 50


def compute_nifty_regime(
    silver_root: Path = SILVER_DATA_ROOT,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    config = config or RegimeConfig()
    indices_path = silver_root / "indices.parquet"
    if not indices_path.exists():
        return pd.DataFrame(columns=["date"])
    idx = pd.read_parquet(indices_path)
    idx["index_key"] = idx["index_name"].astype(str).str.upper().str.strip()
    nifty = idx[idx["index_key"].isin({"NIFTY 50", "CNX NIFTY", "S&P CNX NIFTY"})].copy()
    if nifty.empty:
        return pd.DataFrame(columns=["date"])
    nifty["date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("date").drop_duplicates("date", keep="last")
    nifty["nifty_close"] = pd.to_numeric(nifty["close"], errors="coerce")
    nifty["nifty_ret_1d"] = nifty["nifty_close"].pct_change()
    nifty["nifty_sma50"] = nifty["nifty_close"].rolling(config.sma_window, min_periods=20).mean()
    nifty["nifty_sma200"] = nifty["nifty_close"].rolling(200, min_periods=60).mean()
    nifty["nifty_above_sma50"] = (nifty["nifty_close"] > nifty["nifty_sma50"]).astype(float)
    nifty["nifty_above_sma200"] = (nifty["nifty_close"] > nifty["nifty_sma200"]).astype(float)
    nifty["nifty_realized_vol_20"] = nifty["nifty_ret_1d"].rolling(20, min_periods=10).std()
    nifty["nifty_vol_pctile_252"] = nifty["nifty_realized_vol_20"].rolling(
        config.vol_quantile_window, min_periods=60
    ).rank(pct=True)
    nifty["nifty_ret_20d"] = nifty["nifty_close"].pct_change(20)
    nifty["nifty_drawdown_60d"] = (
        nifty["nifty_close"] / nifty["nifty_close"].rolling(60, min_periods=20).max() - 1.0
    )

    bullish = (nifty["nifty_above_sma50"].eq(1)) & (nifty["nifty_vol_pctile_252"].le(0.75))
    bearish = (nifty["nifty_above_sma50"].eq(0)) & (nifty["nifty_drawdown_60d"].le(-0.05))
    panic = nifty["nifty_vol_pctile_252"].ge(0.90)

    nifty["regime"] = np.select(
        [panic, bearish, bullish],
        ["panic", "bearish", "bullish"],
        default="neutral",
    )
    nifty["long_score_floor"] = np.select(
        [nifty["regime"].eq("bullish"), nifty["regime"].eq("neutral"),
         nifty["regime"].eq("bearish"), nifty["regime"].eq("panic")],
        [config.long_score_normal, config.long_score_normal + 0.05,
         config.long_score_caution, 1.01],
        default=config.long_score_normal,
    )
    nifty["short_score_floor"] = np.select(
        [nifty["regime"].eq("bearish"), nifty["regime"].eq("panic"),
         nifty["regime"].eq("neutral"), nifty["regime"].eq("bullish")],
        [config.short_score_normal, config.short_score_normal,
         config.short_score_normal + 0.05, config.short_score_caution],
        default=config.short_score_normal,
    )
    return nifty[[
        "date", "nifty_close", "nifty_above_sma50", "nifty_above_sma200",
        "nifty_realized_vol_20", "nifty_vol_pctile_252", "nifty_ret_20d",
        "nifty_drawdown_60d", "regime", "long_score_floor", "short_score_floor",
    ]]


def apply_regime_gate(
    predictions: pd.DataFrame,
    silver_root: Path = SILVER_DATA_ROOT,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    regime = compute_nifty_regime(silver_root=silver_root, config=config)
    if regime.empty:
        out = predictions.copy()
        out["regime"] = "unknown"
        out["regime_score_floor"] = np.nan
        out["passes_regime_gate"] = True
        return out
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    out = out.merge(
        regime[["date", "regime", "long_score_floor", "short_score_floor",
                "nifty_vol_pctile_252", "nifty_drawdown_60d"]],
        on="date",
        how="left",
    )
    floor = np.where(out["side"].eq("up"), out["long_score_floor"], out["short_score_floor"])
    out["regime_score_floor"] = floor
    score_col = "score" if "score" in out.columns else (
        "meta_final_score" if "meta_final_score" in out.columns else None
    )
    if score_col is None:
        out["passes_regime_gate"] = True
        return out
    out["passes_regime_gate"] = out[score_col].ge(out["regime_score_floor"]).fillna(False)
    return out
