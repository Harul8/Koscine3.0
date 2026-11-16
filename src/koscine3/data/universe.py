from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class UniverseConfig:
    cutoff_date: str = "2025-12-31"
    lookback_days: int = 252
    top_n: int = 65
    liquid_n: int = 30
    min_coverage: float = 0.80
    min_total_observed_days: int = 252
    max_ohlc_missing: float = 0.02
    min_median_turnover_lacs: float = 1.0
    min_median_volume: float = 1.0
    liquid_threshold: float = 0.04
    wide_threshold: float = 0.07


def build_universe(market_df: pd.DataFrame, config: UniverseConfig | None = None) -> pd.DataFrame:
    config = config or UniverseConfig()
    required = {"date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume"}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for universe builder: {missing}")

    df = market_df[list(required | {"fut_close"} & set(market_df.columns))].copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(config.cutoff_date)
    eligible_dates = sorted(df.loc[df["date"].le(cutoff), "date"].dropna().unique())
    if not eligible_dates:
        raise ValueError(f"No data available on or before universe cutoff {config.cutoff_date}")
    lookback_dates = eligible_dates[-config.lookback_days :]
    recent = df[df["date"].isin(lookback_dates)].copy()
    expected_days = len(lookback_dates)
    if expected_days == 0:
        raise ValueError("Universe lookback window has no trading days")

    grouped = recent.groupby("symbol", sort=False)
    full_history = df[df["date"].le(cutoff)].groupby("symbol", sort=False)
    metrics = grouped.agg(
        observed_days=("date", "nunique"),
        median_turnover_lacs=("turnover_lacs", "median"),
        median_volume=("volume", "median"),
    )
    metrics["total_observed_days"] = full_history["date"].nunique()
    metrics["first_observed_date"] = full_history["date"].min()
    metrics["coverage"] = metrics["observed_days"] / expected_days
    metrics["ohlc_missing_rate"] = recent.groupby("symbol", sort=False)[
        ["open", "high", "low", "close"]
    ].apply(lambda frame: float(frame.isna().any(axis=1).mean()))
    if "fut_close" in recent.columns:
        metrics["futures_coverage"] = grouped["fut_close"].apply(lambda s: float(s.notna().mean()))
    else:
        metrics["futures_coverage"] = 0.0

    metrics = metrics.reset_index()
    metrics["eligible"] = (
        metrics["coverage"].ge(config.min_coverage)
        & metrics["total_observed_days"].fillna(0).ge(config.min_total_observed_days)
        & metrics["ohlc_missing_rate"].le(config.max_ohlc_missing)
        & metrics["median_turnover_lacs"].fillna(0).ge(config.min_median_turnover_lacs)
        & metrics["median_volume"].fillna(0).ge(config.min_median_volume)
    )
    selected = (
        metrics[metrics["eligible"]]
        .sort_values(["median_turnover_lacs", "median_volume"], ascending=False)
        .head(config.top_n)
        .reset_index(drop=True)
    )
    if len(selected) < config.top_n:
        raise ValueError(f"Only {len(selected)} symbols passed universe filters; need {config.top_n}")

    selected["rank"] = selected.index + 1
    selected["band"] = selected["rank"].le(config.liquid_n).map({True: "liquid", False: "wide"})
    selected["threshold"] = selected["band"].map(
        {"liquid": config.liquid_threshold, "wide": config.wide_threshold}
    )
    selected["universe_cutoff"] = config.cutoff_date
    return selected[
        [
            "symbol",
            "rank",
            "band",
            "threshold",
            "universe_cutoff",
            "observed_days",
            "total_observed_days",
            "first_observed_date",
            "coverage",
            "ohlc_missing_rate",
            "median_turnover_lacs",
            "median_volume",
            "futures_coverage",
            "eligible",
        ]
    ]


def write_universe_manifest(universe: pd.DataFrame, output_path: Path, config: UniverseConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.__dict__,
        "symbols": universe.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
