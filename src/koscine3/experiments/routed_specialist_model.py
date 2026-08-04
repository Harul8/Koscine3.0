from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from koscine3.data.feature_registry import build_feature_registry
from koscine3.data.sources import load_market_data, read_data_source
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.datasets.splits import DEFAULT_SPLITS, WalkForwardSplit, between_dates
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns
from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.evaluation.reports import write_manifest, write_report_tables
from koscine3.experiments.large_move_ranker import add_setup_features
from koscine3.experiments.strict_hit_model import (
    add_strict_hit_labels,
    add_strict_research_features,
    build_strict_capture_report,
    build_strict_hit_report,
    strict_feature_columns,
)
from koscine3.paths import RUNS_DIR


try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False


ARCHETYPE_ORDER = [
    "long_trend_participation",
    "short_fragile_reversion",
    "mega_liquid_low_vol",
    "mid_liquidity_mixed",
]

ROUTED_SOURCE_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover_lacs",
    "trades",
    "delivery_qty",
    "delivery_pct",
    "fut_close",
    "fut_oi",
    "fut_chg_oi",
    "fut_vol",
    "opt_call_oi",
    "opt_put_oi",
    "opt_call_vol",
    "opt_put_vol",
    "pcr_oi",
    "pcr_vol",
    "atm_ce_iv",
    "atm_pe_iv",
    "atm_iv",
    "put_call_iv_skew",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "atr_pct_14",
    "range_pct",
    "vol_sma20_ratio",
    "bb_width_20",
    "realized_vol_20",
    "delivery_pct_chg_5",
    "turnover_ratio_20",
    "fut_oi_chg_5",
    "fut_oi_ratio_20",
    "fut_chg_oi_chg_5",
    "fut_chg_oi_ratio_20",
    "fut_vol_chg_5",
    "fut_vol_ratio_20",
    "atm_iv_chg_5",
    "atm_iv_ratio_20",
    "pcr_oi_chg_5",
    "pcr_oi_ratio_20",
    "pcr_vol_chg_5",
    "pcr_vol_ratio_20",
    "nifty_ret_1d",
    "nifty_ret_5d",
    "nifty_realized_vol_20",
    "rel_ret_5d_vs_nifty",
    "ret_5d_cs_rank",
    "ret_20d_cs_rank",
    "vol_sma20_ratio_cs_rank",
    "atr_pct_14_cs_rank",
    "bb_width_20_cs_rank",
    "oi_buildup_ratio",
    "oi_long_buildup",
    "oi_short_buildup",
    "oi_long_unwind",
    "oi_short_unwind",
    "oi_long_buildup_5d",
    "oi_short_buildup_5d",
    "oi_long_unwind_5d",
    "oi_short_unwind_5d",
    "iv_skew_chg_5d",
    "gap_pct",
    "intraday_body_pct",
    "sector_ret_5d",
    "stock_rel_sector_ret_5d",
    "consec_up_days",
    "consec_down_days",
    "pos_day_share_20d",
    "range_contraction_5v20",
    "compression_composite",
    "ema_20_dist",
    "ema_50_dist",
    "ema_20_slope_5d",
    "ema_50_slope_5d",
    "adx_14",
    "di_diff",
    "oi_acceleration",
    "fut_oi_z_60d",
    "price_oi_divergence",
    "vol_5v20_ratio",
    "volume_dryup_score",
    "mkt_pct_above_sma20",
    "mkt_pct_above_sma50",
    "mkt_advance_ratio",
]


@dataclass(frozen=True)
class RoutedSpecialistConfig:
    run_id: str = "koscine3_routed_specialist_v1"
    train_start: str = "2018-01-01"
    universe_cutoff: str = "2025-12-31"
    prediction_top_n: int = 100
    liquid_n: int = 30
    training_top_n: int | None = None
    n_estimators: int = 120
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 80
    min_route_train_rows: int = 900
    strict_close_fraction: float = 0.80
    opposite_close_floor: float = 0.01
    weekly_target: int = 6
    max_signals_per_day: int = 2
    max_signals_per_week_side: int = 4
    max_symbol_per_week: int = 1
    max_symbol_per_month: int = 3
    daily_pool_rank: int = 36
    primary_side_only: bool = True
    min_pair_hit_probability: float = 0.58
    min_full_hit_probability: float = 0.18
    max_opposite_probability: float = 0.70
    max_range_probability: float = 0.85
    min_route_utility: float = 0.35
    pair_weight: float = 1.35
    full_hit_weight: float = 1.10
    opposite_penalty: float = 1.10
    range_penalty: float = 0.25
    random_state: int = 71


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class RouteBundle:
    archetype: str
    side: str
    pair_hit_model: Any
    full_hit_model: Any
    opposite_model: Any
    range_model: Any
    train_rows: int
    pair_train_rows: int
    fallback_route: str | None = None


@dataclass
class RoutedSplitModel:
    model_id: str
    split: WalkForwardSplit
    feature_columns: list[str]
    config: RoutedSpecialistConfig
    route_bundles: dict[tuple[str, str], RouteBundle]
    fallback_bundles: dict[str, RouteBundle]


def _safe_div(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    out = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def _load_focused_market_data() -> pd.DataFrame:
    source = read_data_source()
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(source.path).names)
    except Exception:
        available = set(ROUTED_SOURCE_COLUMNS)
    columns = [c for c in ROUTED_SOURCE_COLUMNS if c in available]
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "turnover_lacs"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Focused routed dataset is missing required columns: {missing}")
    return load_market_data(columns=columns, source=source, sort=True)


def build_liquidity_universe(
    market: pd.DataFrame,
    config: RoutedSpecialistConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"Missing required columns for routed universe: {missing}")

    df = market.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    cutoff = pd.Timestamp(config.universe_cutoff)
    history = df[df["date"].le(cutoff)].copy()
    recent_dates = sorted(history["date"].dropna().unique())[-252:]
    recent = history[history["date"].isin(recent_dates)].copy()
    expected_days = max(1, len(recent_dates))

    grouped = recent.groupby("symbol", sort=False)
    metrics = grouped.agg(
        observed_days=("date", "nunique"),
        median_turnover_lacs=("turnover_lacs", "median"),
        median_volume=("volume", "median"),
    )
    metrics["total_observed_days"] = history.groupby("symbol", sort=False)["date"].nunique()
    metrics["first_observed_date"] = history.groupby("symbol", sort=False)["date"].min()
    metrics["coverage"] = metrics["observed_days"] / expected_days
    metrics["ohlc_missing_rate"] = recent.groupby("symbol", sort=False)[
        ["open", "high", "low", "close"]
    ].apply(lambda frame: float(frame.isna().any(axis=1).mean()))
    if "fut_close" in recent.columns:
        metrics["futures_coverage"] = grouped["fut_close"].apply(lambda s: float(s.notna().mean()))
    else:
        metrics["futures_coverage"] = 0.0
    metrics = metrics.reset_index()
    metrics["eligible_training_symbol"] = (
        metrics["total_observed_days"].fillna(0).ge(252)
        & metrics["coverage"].fillna(0).ge(0.50)
        & metrics["ohlc_missing_rate"].fillna(1).le(0.05)
        & metrics["median_turnover_lacs"].fillna(0).gt(0)
        & metrics["median_volume"].fillna(0).gt(0)
    )
    metrics = metrics.sort_values(["median_turnover_lacs", "median_volume"], ascending=False)
    metrics = metrics.reset_index(drop=True)
    metrics["rank"] = metrics.index + 1
    metrics["band"] = metrics["rank"].le(config.liquid_n).map({True: "liquid", False: "wide"})
    metrics["threshold"] = metrics["band"].map({"liquid": 0.04, "wide": 0.07})
    metrics["universe_cutoff"] = config.universe_cutoff
    metrics["eligible_prediction_symbol"] = metrics["rank"].le(config.prediction_top_n)

    if config.training_top_n is None:
        train_universe = metrics[metrics["eligible_training_symbol"]].copy()
    else:
        train_universe = metrics[
            metrics["eligible_training_symbol"] & metrics["rank"].le(config.training_top_n)
        ].copy()
    prediction_universe = metrics[metrics["rank"].le(config.prediction_top_n)].copy()
    if train_universe.empty:
        raise ValueError("No training symbols passed routed-specialist filters")
    if prediction_universe.empty:
        raise ValueError("No prediction symbols selected")
    return train_universe, prediction_universe


def _model_universe_frame(universe: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
        "eligible_training_symbol",
    ]
    out = universe[[c for c in cols if c in universe.columns]].copy()
    out["symbol"] = out["symbol"].astype(str)
    return out


def build_routed_dataset(config: RoutedSpecialistConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    market = _load_focused_market_data()
    market["symbol"] = market["symbol"].astype(str)
    registry = build_feature_registry(market)
    registry.assert_safe()

    train_universe, prediction_universe = build_liquidity_universe(market, config)
    dataset = build_supervised_dataset(
        market,
        _model_universe_frame(train_universe),
        registry,
    )
    strict_config = type(
        "StrictConfig",
        (),
        {
            "strict_close_fraction": config.strict_close_fraction,
            "opposite_close_floor": config.opposite_close_floor,
        },
    )()
    dataset = add_strict_hit_labels(dataset, strict_config)
    dataset = add_setup_features(dataset)
    dataset = add_strict_research_features(dataset, include_symbol_identity=False)
    dataset = add_compact_strict_history_features(dataset)
    feature_columns = strict_feature_columns(model_feature_columns(registry, dataset), dataset)
    return dataset, train_universe, prediction_universe, feature_columns


def add_compact_strict_history_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy().sort_values(["symbol", "side", "date"]).reset_index(drop=True)
    lag = 6
    grouped = df.groupby(["symbol", "side"], sort=False)
    for window in (63, 252):
        min_periods = 20 if window == 63 else 60
        df[f"strict_hist_hit_rate_{window}"] = grouped["strict_hit"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
        )
        df[f"strict_hist_opposite_rate_{window}"] = grouped["strict_opposite"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
        )
        df[f"strict_hist_range_rate_{window}"] = grouped["strict_range_bound"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
        )
        df[f"strict_hist_signed_close_mean_{window}"] = grouped["signed_close_return"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
        )
        df[f"strict_hist_favorable_move_mean_{window}"] = grouped["favorable_move"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
        )
        df[f"strict_hist_edge_{window}"] = (
            df[f"strict_hist_hit_rate_{window}"] - df[f"strict_hist_opposite_rate_{window}"]
        )

    daily_side = (
        df[df["status"].eq("evaluated")]
        .groupby(["date", "side"], sort=False)
        .agg(
            strict_side_hit=("strict_hit", "mean"),
            strict_side_opposite=("strict_opposite", "mean"),
            strict_side_signed_close=("signed_close_return", "mean"),
        )
        .reset_index()
        .sort_values(["side", "date"])
    )
    side_grouped = daily_side.groupby("side", sort=False)
    for window in (21, 63):
        daily_side[f"strict_side_hit_rate_{window}"] = side_grouped["strict_side_hit"].transform(
            lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 3)).mean()
        )
        daily_side[f"strict_side_opposite_rate_{window}"] = side_grouped[
            "strict_side_opposite"
        ].transform(lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 3)).mean())
        daily_side[f"strict_side_signed_close_mean_{window}"] = side_grouped[
            "strict_side_signed_close"
        ].transform(lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 3)).mean())
    keep_cols = [
        "date",
        "side",
        *[c for c in daily_side.columns if c.startswith("strict_side_") and c not in {"strict_side_hit", "strict_side_opposite", "strict_side_signed_close"}],
    ]
    df = df.merge(daily_side[keep_cols], on=["date", "side"], how="left")
    return df.replace([np.inf, -np.inf], np.nan)


def assign_split_archetypes(
    dataset: pd.DataFrame,
    train_universe: pd.DataFrame,
    split: WalkForwardSplit,
    config: RoutedSpecialistConfig,
) -> pd.DataFrame:
    train = dataset[
        dataset["status"].eq("evaluated")
        & between_dates(dataset, start=config.train_start, end=split.base_train_end)
    ].copy()
    rows = []
    liquidity = train_universe.set_index("symbol")
    if train.empty:
        raise ValueError(f"No training rows available for archetype assignment: {split.name}")

    atr_cut = pd.to_numeric(train.get("atr_pct_14"), errors="coerce").median()
    turnover_rank = liquidity["rank"].to_dict() if "rank" in liquidity.columns else {}

    for symbol, group in train.groupby("symbol", sort=False):
        row: dict[str, object] = {"symbol": symbol}
        row["liquidity_rank"] = float(turnover_rank.get(symbol, np.nan))
        row["median_atr_pct_14"] = float(pd.to_numeric(group.get("atr_pct_14"), errors="coerce").median())
        row["median_turnover_lacs"] = float(liquidity.loc[symbol, "median_turnover_lacs"]) if symbol in liquidity.index else np.nan
        for side in ["long", "short"]:
            part = group[group["side"].eq(side)]
            row[f"{side}_rows"] = int(len(part))
            row[f"{side}_strict_hit_rate"] = float(part["strict_hit"].mean()) if len(part) else 0.0
            row[f"{side}_strict_opposite_rate"] = float(part["strict_opposite"].mean()) if len(part) else 0.0
        long_hit = float(row["long_strict_hit_rate"])
        short_hit = float(row["short_strict_hit_rate"])
        median_atr = float(row["median_atr_pct_14"]) if np.isfinite(row["median_atr_pct_14"]) else 0.0
        rank = float(row["liquidity_rank"]) if np.isfinite(row["liquidity_rank"]) else 9999.0
        if long_hit - short_hit >= 0.035 and long_hit >= 0.14:
            archetype = "long_trend_participation"
            primary_side = "long"
        elif short_hit - long_hit >= 0.025 and short_hit >= 0.14:
            archetype = "short_fragile_reversion"
            primary_side = "short"
        elif rank <= 75 and median_atr <= atr_cut:
            archetype = "mega_liquid_low_vol"
            primary_side = "both"
        else:
            archetype = "mid_liquidity_mixed"
            primary_side = "both"
        row["archetype"] = archetype
        row["primary_side"] = primary_side
        rows.append(row)
    archetypes = pd.DataFrame(rows)
    archetypes["split"] = split.name
    return archetypes


def _classifier(config: RoutedSpecialistConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("run-routed-specialist-model requires lightgbm")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    num_leaves=config.num_leaves,
                    min_child_samples=config.min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=random_state,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _clean_x(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].replace([np.inf, -np.inf], np.nan)


def _fit_probability_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: RoutedSpecialistConfig,
    random_state: int,
) -> Any:
    y = train[target].astype(int)
    if y.nunique() < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = _classifier(config, random_state)
    model.fit(_clean_x(train, feature_columns), y)
    return model


def _fit_route_bundle(
    route_train: pd.DataFrame,
    archetype: str,
    side: str,
    feature_columns: list[str],
    config: RoutedSpecialistConfig,
    random_state: int,
    fallback_route: str | None = None,
) -> RouteBundle:
    pair_train = route_train[route_train["strict_pair_target"].notna()].copy()
    if pair_train.empty:
        pair_train = route_train.copy()
        pair_train["strict_pair_is_hit"] = pair_train["strict_hit"].astype(int)
    else:
        pair_train["strict_pair_is_hit"] = pair_train["strict_pair_target"].astype(int)
    return RouteBundle(
        archetype=archetype,
        side=side,
        pair_hit_model=_fit_probability_model(
            pair_train,
            feature_columns,
            "strict_pair_is_hit",
            config,
            random_state,
        ),
        full_hit_model=_fit_probability_model(
            route_train,
            feature_columns,
            "strict_hit",
            config,
            random_state + 11,
        ),
        opposite_model=_fit_probability_model(
            route_train,
            feature_columns,
            "strict_opposite",
            config,
            random_state + 23,
        ),
        range_model=_fit_probability_model(
            route_train,
            feature_columns,
            "strict_range_bound",
            config,
            random_state + 37,
        ),
        train_rows=int(len(route_train)),
        pair_train_rows=int(len(pair_train)),
        fallback_route=fallback_route,
    )


def train_routed_split_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    archetypes: pd.DataFrame,
    split: WalkForwardSplit,
    config: RoutedSpecialistConfig,
) -> RoutedSplitModel:
    train = dataset[
        dataset["status"].eq("evaluated")
        & between_dates(dataset, start=config.train_start, end=split.base_train_end)
    ].copy()
    train = train.merge(archetypes[["symbol", "archetype", "primary_side"]], on="symbol", how="left")
    train["archetype"] = train["archetype"].fillna("mid_liquidity_mixed")
    route_bundles: dict[tuple[str, str], RouteBundle] = {}
    fallback_bundles: dict[str, RouteBundle] = {}

    for side in ["long", "short"]:
        side_train = train[train["side"].eq(side)].copy()
        fallback_bundles[side] = _fit_route_bundle(
            side_train,
            "all_symbols_fallback",
            side,
            feature_columns,
            config,
            config.random_state + (0 if side == "long" else 1000),
        )

    seed = config.random_state
    for archetype in ARCHETYPE_ORDER:
        for side in ["long", "short"]:
            route_train = train[train["archetype"].eq(archetype) & train["side"].eq(side)].copy()
            fallback_route = None
            if len(route_train) < config.min_route_train_rows:
                route_train = train[train["side"].eq(side)].copy()
                fallback_route = "all_symbols_fallback"
            route_bundles[(archetype, side)] = _fit_route_bundle(
                route_train,
                archetype,
                side,
                feature_columns,
                config,
                seed,
                fallback_route=fallback_route,
            )
            seed += 53
    return RoutedSplitModel(
        model_id=f"{config.run_id}_{split.name}_routed_specialists",
        split=split,
        feature_columns=feature_columns,
        config=config,
        route_bundles=route_bundles,
        fallback_bundles=fallback_bundles,
    )


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _route_allowed(archetype: str, side: str, primary_side: str, config: RoutedSpecialistConfig) -> bool:
    if not config.primary_side_only:
        return True
    if primary_side == "both":
        return True
    return side == primary_side


def predict_routed_split_model(
    model: RoutedSplitModel,
    dataset: pd.DataFrame,
    prediction_universe: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> pd.DataFrame:
    allowed_symbols = set(prediction_universe["symbol"].astype(str))
    period = dataset[dataset["symbol"].astype(str).isin(allowed_symbols)].copy()
    period = period.merge(archetypes[["symbol", "archetype", "primary_side"]], on="symbol", how="left")
    period["archetype"] = period["archetype"].fillna("mid_liquidity_mixed")
    period["primary_side"] = period["primary_side"].fillna("both")
    frames: list[pd.DataFrame] = []
    passthrough = [
        "date",
        "symbol",
        "side",
        "band",
        "threshold",
        "target_threshold",
        "entry_date",
        "entry_open",
        "window_end_date",
        "status",
        "verdict",
        "hit",
        "near",
        "hit_or_near",
        "opposite",
        "small",
        "strict_hit",
        "strict_opposite",
        "strict_range_bound",
        "strict_category",
        "favorable_move",
        "signed_close_return",
        "setup_id",
        "archetype",
        "primary_side",
    ]
    for (archetype, side), bundle in model.route_bundles.items():
        part = period[period["archetype"].eq(archetype) & period["side"].eq(side)].copy()
        if part.empty:
            continue
        x = _clean_x(part, model.feature_columns)
        out = part[[c for c in passthrough if c in part.columns]].copy()
        out["route_allowed"] = [
            _route_allowed(archetype, side, primary, model.config)
            for primary in out["primary_side"].astype(str)
        ]
        out["p_route_pair_hit"] = np.clip(_positive_probability(bundle.pair_hit_model, x), 0, 1)
        out["p_route_strict_hit"] = np.clip(_positive_probability(bundle.full_hit_model, x), 0, 1)
        out["p_route_opposite"] = np.clip(_positive_probability(bundle.opposite_model, x), 0, 1)
        out["p_route_range"] = np.clip(_positive_probability(bundle.range_model, x), 0, 1)
        out["route_edge"] = out["p_route_pair_hit"] - out["p_route_opposite"]
        out["route_utility"] = (
            model.config.pair_weight * out["p_route_pair_hit"]
            + model.config.full_hit_weight * out["p_route_strict_hit"]
            - model.config.opposite_penalty * out["p_route_opposite"]
            - model.config.range_penalty * out["p_route_range"]
        )
        out["model_id"] = model.model_id
        out["route_model_id"] = f"{model.model_id}_{archetype}_{side}"
        out["route_fallback"] = bundle.fallback_route or ""
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "route_utility"], ascending=[True, False])


def select_routed_weekly_signals(
    predictions: pd.DataFrame,
    config: RoutedSpecialistConfig,
) -> pd.DataFrame:
    df = predictions.copy().sort_values(["date", "route_utility"], ascending=[True, False])
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selection_reason"] = "candidate_not_selected"
    df["selector_config_id"] = f"{config.run_id}_raw_routed_weekly_selector"
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.strftime("%G-W%V")

    keep = (
        df.sort_values(["date", "symbol", "route_utility"], ascending=[True, True, False])
        .drop_duplicates(["date", "symbol"], keep="first")
        .index
    )
    eligible = df.index.isin(keep)
    df.loc[~eligible, "selection_reason"] = "symbol_side_conflict"

    gates = [
        ("route_side_disabled", ~df["route_allowed"].astype(bool)),
        ("below_pair_hit_probability", df["p_route_pair_hit"].lt(config.min_pair_hit_probability)),
        ("below_full_hit_probability", df["p_route_strict_hit"].lt(config.min_full_hit_probability)),
        ("above_opposite_probability", df["p_route_opposite"].gt(config.max_opposite_probability)),
        ("above_range_probability", df["p_route_range"].gt(config.max_range_probability)),
        ("below_route_utility", df["route_utility"].lt(config.min_route_utility)),
    ]
    for reason, mask in gates:
        reject = eligible & mask
        df.loc[reject, "selection_reason"] = reason
        eligible &= ~reject

    df["date_pool_rank"] = (
        df[eligible].groupby("date")["route_utility"].rank(method="first", ascending=False)
    )
    below_pool = eligible & df["date_pool_rank"].gt(config.daily_pool_rank)
    df.loc[below_pool, "selection_reason"] = "below_daily_rank_pool"
    eligible &= ~below_pool

    month_symbol_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, week_rows in df[eligible].groupby("week", sort=True):
        selected_this_week = 0
        week_symbols: dict[str, int] = defaultdict(int)
        week_side_counts: dict[str, int] = defaultdict(int)
        date_counts: dict[pd.Timestamp, int] = defaultdict(int)
        for idx, row in week_rows.sort_values("route_utility", ascending=False).iterrows():
            if selected_this_week >= config.weekly_target:
                df.at[idx, "selection_reason"] = "weekly_target_met"
                continue
            date = pd.Timestamp(row["date"])
            symbol = str(row["symbol"])
            side = str(row["side"])
            month_key = (str(row["month"]), symbol)
            if date_counts[date] >= config.max_signals_per_day:
                df.at[idx, "selection_reason"] = "daily_cap"
                continue
            if week_side_counts[side] >= config.max_signals_per_week_side:
                df.at[idx, "selection_reason"] = "weekly_side_cap"
                continue
            if week_symbols[symbol] >= config.max_symbol_per_week:
                df.at[idx, "selection_reason"] = "weekly_symbol_cap"
                continue
            if month_symbol_counts[month_key] >= config.max_symbol_per_month:
                df.at[idx, "selection_reason"] = "monthly_symbol_cap"
                continue
            df.at[idx, "selected"] = True
            df.at[idx, "selection_reason"] = "selected"
            selected_this_week += 1
            week_symbols[symbol] += 1
            week_side_counts[side] += 1
            date_counts[date] += 1
            month_symbol_counts[month_key] += 1

    return df.sort_values(["date", "selected", "route_utility"], ascending=[True, False, False])


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def _save_model(model: RoutedSplitModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    bundle_rows = [
        {
            "archetype": bundle.archetype,
            "side": bundle.side,
            "train_rows": bundle.train_rows,
            "pair_train_rows": bundle.pair_train_rows,
            "fallback_route": bundle.fallback_route or "",
        }
        for bundle in model.route_bundles.values()
    ]
    pd.DataFrame(bundle_rows).to_csv(output_dir / "route_bundles.csv", index=False)
    write_manifest(
        {
            "model_id": model.model_id,
            "split": asdict(model.split),
            "config": asdict(model.config),
            "algorithm": "raw_lightgbm_archetype_side_specialists",
            "probability_output": "raw_lightgbm_predict_proba",
            "calibration": "none",
            "feature_count": len(model.feature_columns),
            "feature_columns": model.feature_columns,
        },
        output_dir / "manifest.json",
    )


def run_routed_specialist_model(config: RoutedSpecialistConfig) -> Path:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is required for run-routed-specialist-model")

    run_dir = RUNS_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, train_universe, prediction_universe, feature_columns = build_routed_dataset(config)

    train_universe.to_csv(run_dir / "training_universe.csv", index=False)
    prediction_universe.to_csv(run_dir / "prediction_universe.csv", index=False)
    (run_dir / "feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")

    selected_frames: list[pd.DataFrame] = []
    archetype_frames: list[pd.DataFrame] = []
    for split in DEFAULT_SPLITS:
        archetypes = assign_split_archetypes(dataset, train_universe, split, config)
        archetype_frames.append(archetypes)
        model = train_routed_split_model(dataset, feature_columns, archetypes, split, config)
        _save_model(model, run_dir / "models" / split.name)

        period = dataset[between_dates(dataset, split.prediction_start, split.prediction_end)].copy()
        predictions = predict_routed_split_model(model, period, prediction_universe, archetypes)
        selected = select_routed_weekly_signals(predictions, config)
        selected["split"] = split.name

        split_dir = run_dir / "model_predictions" / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        archetypes.to_csv(split_dir / "archetypes.csv", index=False)
        predictions.to_parquet(split_dir / "routed_predictions.parquet", index=False)
        selected.to_parquet(split_dir / "signals.parquet", index=False)
        selected[selected["selected"]].to_csv(split_dir / "selected_signals.csv", index=False)
        write_report_tables(build_strict_hit_report(selected), split_dir / "strict_report")
        write_report_tables(build_gold_report(selected), split_dir / "gold_report")
        _write_tables(build_strict_capture_report(selected, period), split_dir / "capture_report")
        selected_frames.append(selected)

    combined = pd.concat(selected_frames, ignore_index=True)
    all_archetypes = pd.concat(archetype_frames, ignore_index=True)
    combined.to_parquet(run_dir / "all_signals.parquet", index=False)
    combined[combined["selected"]].to_csv(run_dir / "selected_signals.csv", index=False)
    all_archetypes.to_csv(run_dir / "all_split_archetypes.csv", index=False)
    write_report_tables(build_strict_hit_report(combined), run_dir / "combined_strict_report")
    write_report_tables(build_gold_report(combined), run_dir / "combined_gold_report")
    combined_period = dataset[
        between_dates(dataset, DEFAULT_SPLITS[0].prediction_start, DEFAULT_SPLITS[-1].prediction_end)
    ].copy()
    _write_tables(build_strict_capture_report(combined, combined_period), run_dir / "combined_capture_report")

    write_manifest(
        {
            "run_id": config.run_id,
            "source": read_data_source().__dict__,
            "config": asdict(config),
            "feature_count": len(feature_columns),
            "training_symbols": int(train_universe["symbol"].nunique()),
            "prediction_symbols": int(prediction_universe["symbol"].nunique()),
            "splits": [asdict(split) for split in DEFAULT_SPLITS],
            "notes": [
                "Routed archetype/side specialist models.",
                "Training can use all eligible raw-data symbols; prediction is restricted by prediction_top_n.",
                "All probabilities are raw LightGBM predict_proba outputs.",
                "No calibration selector, no overlays, no old Koscine 2.0 predictions.",
            ],
        },
        run_dir / "manifest.json",
    )
    return run_dir
