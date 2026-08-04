from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from koscine3.data.feature_registry import FeatureRegistry
from koscine3.outcomes.swing_contract import compute_swing_outcomes


ENGINEERED_FEATURE_COLUMNS = [
    "universe_rank",
    "target_threshold",
    "is_liquid_band",
    "universe_coverage",
    "universe_total_observed_days",
    "universe_ohlc_missing_rate",
    "universe_median_turnover_lacs",
    "universe_median_volume",
    "universe_futures_coverage",
    "hist_hit_near_rate_63",
    "hist_hit_near_rate_252",
    "hist_opposite_rate_63",
    "hist_opposite_rate_252",
    "hist_favorable_move_mean_63",
    "hist_favorable_move_mean_252",
    "hist_signed_close_mean_63",
    "hist_signed_close_mean_252",
    "symbol_return_1d",
    "symbol_return_5d",
    "symbol_return_20d",
    "symbol_volatility_20d",
    "symbol_turnover_mean_20d",
    "symbol_return_5d_rank_pct",
    "symbol_return_20d_rank_pct",
    "relative_return_5d",
    "relative_return_20d",
    "xsec_mean_return_1d",
    "xsec_mean_return_5d",
    "xsec_mean_return_20d",
    "xsec_positive_share_1d",
    "xsec_positive_share_5d",
    "xsec_positive_share_20d",
    "xsec_mean_volatility_20d",
]


def model_feature_columns(registry: FeatureRegistry, dataset: pd.DataFrame | None = None) -> list[str]:
    engineered = ENGINEERED_FEATURE_COLUMNS
    if dataset is not None:
        engineered = [c for c in engineered if c in dataset.columns]
    return [*registry.feature_columns, *engineered]


def _universe_feature_frame(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    rename = {
        "rank": "universe_rank",
        "threshold": "target_threshold",
        "coverage": "universe_coverage",
        "total_observed_days": "universe_total_observed_days",
        "ohlc_missing_rate": "universe_ohlc_missing_rate",
        "median_turnover_lacs": "universe_median_turnover_lacs",
        "median_volume": "universe_median_volume",
        "futures_coverage": "universe_futures_coverage",
    }
    frame = frame.rename(columns=rename)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["is_liquid_band"] = frame["band"].eq("liquid").astype(int)
    cols = ["symbol", *[c for c in rename.values() if c in frame.columns], "is_liquid_band"]
    return frame[cols]


def _add_historical_outcome_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.sort_values(["symbol", "side", "date"]).copy()
    grouped = out.groupby(["symbol", "side"], sort=False)
    lag = 6
    for window in (63, 252):
        out[f"hist_hit_near_rate_{window}"] = grouped["hit_or_near"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=20).mean()
        )
        out[f"hist_opposite_rate_{window}"] = grouped["opposite"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=20).mean()
        )
        out[f"hist_favorable_move_mean_{window}"] = grouped["favorable_move"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=20).mean()
        )
        out[f"hist_signed_close_mean_{window}"] = grouped["signed_close_return"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=20).mean()
        )
    return out


def _add_market_regime_features(dataset: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    price = base[["date", "symbol", "close", "turnover_lacs"]].copy()
    price["date"] = pd.to_datetime(price["date"])
    price["symbol"] = price["symbol"].astype(str)
    price = price.sort_values(["symbol", "date"])
    grouped = price.groupby("symbol", sort=False)
    price["symbol_return_1d"] = grouped["close"].pct_change(1)
    price["symbol_return_5d"] = grouped["close"].pct_change(5)
    price["symbol_return_20d"] = grouped["close"].pct_change(20)
    price["symbol_volatility_20d"] = grouped["symbol_return_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    price["symbol_turnover_mean_20d"] = grouped["turnover_lacs"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    price["symbol_return_5d_rank_pct"] = price.groupby("date")["symbol_return_5d"].rank(pct=True)
    price["symbol_return_20d_rank_pct"] = price.groupby("date")["symbol_return_20d"].rank(pct=True)

    daily = price.groupby("date", sort=False).agg(
        xsec_mean_return_1d=("symbol_return_1d", "mean"),
        xsec_mean_return_5d=("symbol_return_5d", "mean"),
        xsec_mean_return_20d=("symbol_return_20d", "mean"),
        xsec_positive_share_1d=("symbol_return_1d", lambda s: float((s > 0).mean())),
        xsec_positive_share_5d=("symbol_return_5d", lambda s: float((s > 0).mean())),
        xsec_positive_share_20d=("symbol_return_20d", lambda s: float((s > 0).mean())),
        xsec_mean_volatility_20d=("symbol_volatility_20d", "mean"),
    )
    price = price.merge(daily, on="date", how="left")
    price["relative_return_5d"] = price["symbol_return_5d"] - price["xsec_mean_return_5d"]
    price["relative_return_20d"] = price["symbol_return_20d"] - price["xsec_mean_return_20d"]
    feature_cols = [
        "date",
        "symbol",
        "symbol_return_1d",
        "symbol_return_5d",
        "symbol_return_20d",
        "symbol_volatility_20d",
        "symbol_turnover_mean_20d",
        "symbol_return_5d_rank_pct",
        "symbol_return_20d_rank_pct",
        "relative_return_5d",
        "relative_return_20d",
        "xsec_mean_return_1d",
        "xsec_mean_return_5d",
        "xsec_mean_return_20d",
        "xsec_positive_share_1d",
        "xsec_positive_share_5d",
        "xsec_positive_share_20d",
        "xsec_mean_volatility_20d",
    ]
    return dataset.merge(price[feature_cols], on=["date", "symbol"], how="left")


def build_supervised_dataset(
    market_df: pd.DataFrame,
    universe: pd.DataFrame,
    registry: FeatureRegistry,
    output_manifest_path: Path | None = None,
) -> pd.DataFrame:
    symbols = set(universe["symbol"].astype(str))
    base = market_df[market_df["symbol"].astype(str).isin(symbols)].copy()
    base["date"] = pd.to_datetime(base["date"])
    outcomes = compute_swing_outcomes(base, universe=universe)

    feature_frame = base[["date", "symbol", *registry.feature_columns]].copy()
    feature_frame["symbol"] = feature_frame["symbol"].astype(str)
    dataset = outcomes.merge(feature_frame, on=["date", "symbol"], how="left")
    dataset = dataset.merge(_universe_feature_frame(universe), on="symbol", how="left")
    dataset["hit_or_near"] = dataset["hit"] | dataset["near"]
    dataset["clean_success"] = dataset["hit_or_near"] & ~dataset["opposite"]
    dataset["side_code"] = dataset["side"].map({"long": 1, "short": -1}).astype(int)
    dataset = _add_historical_outcome_features(dataset)
    dataset = _add_market_regime_features(dataset, base)

    if output_manifest_path is not None:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "rows": int(len(dataset)),
            "symbols": int(dataset["symbol"].nunique()),
            "date_min": str(dataset["date"].min().date()),
            "date_max": str(dataset["date"].max().date()),
            "feature_count": len(model_feature_columns(registry, dataset)),
            "feature_columns": model_feature_columns(registry, dataset),
            "engineered_feature_columns": [
                c for c in ENGINEERED_FEATURE_COLUMNS if c in dataset.columns
            ],
        }
        output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset
