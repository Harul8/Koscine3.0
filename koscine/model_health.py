from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import REPORTS_DIR


@dataclass(frozen=True)
class HealthConfig:
    rolling_trades: int = 20
    healthy_hit_rate: float = 0.55
    warning_hit_rate: float = 0.45
    critical_hit_rate: float = 0.35
    min_trades_for_signal: int = 10
    derisk_factor: float = 0.5


def rolling_model_hit_rate(
    trades: pd.DataFrame,
    config: HealthConfig | None = None,
    model_col: str = "model_id",
) -> pd.DataFrame:
    config = config or HealthConfig()
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    if "hit" not in out:
        if "actual_hit" in out:
            out["hit"] = out["actual_hit"].astype(bool)
        elif "net_return" in out:
            out["hit"] = out["net_return"].gt(0)
        else:
            out["hit"] = False
    out["hit"] = out["hit"].astype(bool).astype(int)
    if model_col not in out:
        out[model_col] = "all"
    out["date"] = pd.to_datetime(out.get("exit_date", out.get("date")))
    out = out.sort_values(["date"]).reset_index(drop=True)
    out["rolling_hit_rate"] = (
        out.groupby(model_col)["hit"]
        .transform(lambda s: s.rolling(config.rolling_trades, min_periods=config.min_trades_for_signal).mean())
    )
    out["rolling_trade_count"] = (
        out.groupby(model_col)["hit"].cumcount() + 1
    )
    out["allocation_multiplier"] = np.where(
        out["rolling_hit_rate"].ge(config.warning_hit_rate),
        1.0,
        np.where(
            out["rolling_hit_rate"].ge(config.critical_hit_rate),
            config.derisk_factor,
            0.0,
        ),
    )
    out["health_status"] = np.select(
        [
            out["rolling_hit_rate"].ge(config.healthy_hit_rate),
            out["rolling_hit_rate"].ge(config.warning_hit_rate),
            out["rolling_hit_rate"].ge(config.critical_hit_rate),
        ],
        ["healthy", "ok", "warning"],
        default="critical",
    )
    return out


def latest_model_health(
    trades: pd.DataFrame,
    config: HealthConfig | None = None,
    model_col: str = "model_id",
) -> pd.DataFrame:
    enriched = rolling_model_hit_rate(trades, config=config, model_col=model_col)
    if enriched.empty:
        return pd.DataFrame()
    latest = enriched.sort_values("date").groupby(model_col, as_index=False).tail(1)
    return latest[[
        model_col, "date", "rolling_hit_rate", "rolling_trade_count",
        "allocation_multiplier", "health_status",
    ]].sort_values(model_col).reset_index(drop=True)


def write_health_report(
    trades: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
    config: HealthConfig | None = None,
    model_col: str = "model_id",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = rolling_model_hit_rate(trades, config=config, model_col=model_col)
    latest = latest_model_health(trades, config=config, model_col=model_col)
    enriched_path = output_dir / "model_health_rolling.csv"
    latest_path = output_dir / "model_health_latest.csv"
    enriched.to_csv(enriched_path, index=False)
    latest.to_csv(latest_path, index=False)
    return output_dir
