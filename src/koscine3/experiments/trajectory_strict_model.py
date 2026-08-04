from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from koscine3.data.feature_registry import build_feature_registry
from koscine3.data.sources import read_data_source
from koscine3.datasets.splits import DEFAULT_SPLITS, WalkForwardSplit, between_dates
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns
from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.evaluation.reports import write_manifest, write_report_tables
from koscine3.experiments.large_move_ranker import add_setup_features
from koscine3.experiments.routed_specialist_model import (
    ARCHETYPE_ORDER,
    ROUTED_SOURCE_COLUMNS,
    _model_universe_frame,
    _route_allowed,
    build_liquidity_universe,
    assign_split_archetypes,
)
from koscine3.experiments.strict_hit_model import (
    BLOCKED_PREFIXES,
    OUTCOME_COLUMNS,
    add_strict_hit_labels,
    add_strict_research_features,
    build_strict_capture_report,
    build_strict_hit_report,
    strict_feature_columns,
)
from koscine3.experiments.routed_specialist_model import add_compact_strict_history_features
from koscine3.paths import RUNS_DIR


try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False


TRAJECTORY_SOURCE_COLUMNS = sorted(
    {
        *ROUTED_SOURCE_COLUMNS,
        "fut_settle",
        "max_pain",
        "call_wall_1",
        "put_wall_1",
        "close_sma5_dist",
        "close_sma10_dist",
        "close_sma20_dist",
        "close_sma50_dist",
        "vol_sma5_ratio",
        "vol_sma10_ratio",
        "vol_sma50_ratio",
        "donchian_width_20",
        "max_pain_dist",
        "call_wall_1_dist",
        "put_wall_1_dist",
        "gap_up_flag",
        "gap_down_flag",
        "gap_up_count_10d",
        "gap_down_count_10d",
        "gap_up_count_20d",
        "gap_down_count_20d",
        "sector_ret_1d",
        "sector_ret_20d",
        "stock_rel_sector_ret_20d",
        "pos_day_share_10d",
        "range_pct_today",
        "nr7_flag",
        "inside_bar_flag",
        "inside_bar_count_5d",
        "atr_pct_14_rank_60d",
        "bb_width_20_rank_60d",
        "ema_5_dist",
        "ema_10_dist",
        "new_high_10d",
        "new_low_10d",
        "new_high_count_20d",
        "new_low_count_20d",
    }
)


@dataclass(frozen=True)
class TrajectoryStrictConfig:
    run_id: str = "koscine3_trajectory_strict_v1"
    train_start: str = "2018-01-01"
    universe_cutoff: str = "2025-12-31"
    prediction_top_n: int = 100
    liquid_n: int = 30
    training_top_n: int | None = None
    n_estimators: int = 140
    learning_rate: float = 0.032
    num_leaves: int = 31
    min_child_samples: int = 90
    min_route_train_rows: int = 900
    strict_close_fraction: float = 0.80
    opposite_close_floor: float = 0.01
    matched_opposites_per_hit: int = 3
    min_matched_pair_rows: int = 250
    weekly_target: int = 6
    max_signals_per_day: int = 2
    max_signals_per_week_side: int = 4
    max_symbol_per_week: int = 1
    max_symbol_per_month: int = 3
    daily_pool_rank: int = 40
    primary_side_only: bool = True
    min_pair_hit_probability: float = 0.58
    min_full_hit_probability: float = 0.16
    max_opposite_probability: float = 0.68
    max_range_probability: float = 0.86
    min_trajectory_edge: float = -0.02
    min_trajectory_utility: float = 0.32
    pair_weight: float = 1.55
    full_hit_weight: float = 1.00
    opposite_penalty: float = 1.30
    range_penalty: float = 0.20
    setup_score_weight: float = 0.22
    resume_completed_splits: bool = True
    random_state: int = 97


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class TrajectoryRouteBundle:
    archetype: str
    side: str
    pair_hit_model: Any
    full_hit_model: Any
    opposite_model: Any
    range_model: Any
    train_rows: int
    pair_train_rows: int
    matched_hits: int
    matched_opposites: int
    fallback_route: str | None = None


@dataclass
class TrajectorySplitModel:
    model_id: str
    split: WalkForwardSplit
    feature_columns: list[str]
    config: TrajectoryStrictConfig
    route_bundles: dict[tuple[str, str], TrajectoryRouteBundle]


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_div(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    out = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def _group_pct_change(base: pd.DataFrame, group: Any, column: str, window: int) -> pd.Series:
    if column not in base.columns:
        return pd.Series(np.nan, index=base.index, dtype=float)
    return group[column].pct_change(window).replace([np.inf, -np.inf], np.nan)


def _group_diff(base: pd.DataFrame, group: Any, column: str, window: int) -> pd.Series:
    if column not in base.columns:
        return pd.Series(np.nan, index=base.index, dtype=float)
    return group[column].diff(window).replace([np.inf, -np.inf], np.nan)


def _rolling_z(base: pd.DataFrame, group: Any, column: str, window: int, min_periods: int) -> pd.Series:
    if column not in base.columns:
        return pd.Series(np.nan, index=base.index, dtype=float)
    values = pd.to_numeric(base[column], errors="coerce")
    mean = group[column].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    std = group[column].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
    return _safe_div(values - mean, std)


def _daily_rank(df: pd.DataFrame, values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.groupby(df["date"]).rank(pct=True, ascending=ascending).fillna(0.5)


def _load_trajectory_market_data() -> pd.DataFrame:
    source = read_data_source()
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(source.path).names)
    except Exception:
        available = set(TRAJECTORY_SOURCE_COLUMNS)
    columns = [c for c in TRAJECTORY_SOURCE_COLUMNS if c in available]
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "turnover_lacs"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Trajectory dataset is missing required columns: {missing}")
    market = pd.read_parquet(source.path, columns=columns)
    market["date"] = pd.to_datetime(market["date"])
    market["symbol"] = market["symbol"].astype(str)
    return market.sort_values(["symbol", "date"]).reset_index(drop=True)


def _first_existing(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def add_trajectory_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    base_cols = [
        "date",
        "symbol",
        *[c for c in TRAJECTORY_SOURCE_COLUMNS if c in df.columns and c not in {"date", "symbol"}],
    ]
    base = (
        df[base_cols]
        .drop_duplicates(["date", "symbol"], keep="first")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    group = base.groupby("symbol", sort=False)

    high = _num(base, "high")
    low = _num(base, "low")
    open_ = _num(base, "open")
    close = _num(base, "close")
    prev_close = _num(base, "prev_close")
    span = (high - low).abs().replace(0, np.nan)
    base["traj_close_location"] = _safe_div(close - low, span, default=0.5)
    base["traj_body_pct"] = _safe_div(close - open_, open_)
    base["traj_abs_body_to_range"] = _safe_div((close - open_).abs(), span)
    base["traj_upper_wick_pct"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), span)
    base["traj_lower_wick_pct"] = _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, span)
    base["traj_wick_imbalance"] = base["traj_lower_wick_pct"] - base["traj_upper_wick_pct"]
    base["traj_gap_pct"] = _safe_div(open_ - prev_close, prev_close)
    base["traj_gap_follow_through"] = np.sign(base["traj_gap_pct"]) * base["traj_body_pct"]

    for window in (1, 2, 3, 5, 10):
        base[f"traj_close_ret_{window}"] = _group_pct_change(base, group, "close", window)
        base[f"traj_volume_chg_{window}"] = _group_pct_change(base, group, "volume", window)
        base[f"traj_turnover_chg_{window}"] = _group_pct_change(base, group, "turnover_lacs", window)
        base[f"traj_delivery_pct_diff_{window}"] = _group_diff(base, group, "delivery_pct", window)
        base[f"traj_fut_oi_chg_{window}"] = _group_pct_change(base, group, "fut_oi", window)
        base[f"traj_fut_vol_chg_{window}"] = _group_pct_change(base, group, "fut_vol", window)
        base[f"traj_pcr_oi_diff_{window}"] = _group_diff(base, group, "pcr_oi", window)
        base[f"traj_pcr_vol_diff_{window}"] = _group_diff(base, group, "pcr_vol", window)
        base[f"traj_atm_iv_diff_{window}"] = _group_diff(base, group, "atm_iv", window)
        base[f"traj_iv_skew_diff_{window}"] = _group_diff(base, group, "put_call_iv_skew", window)
        base[f"traj_atr_pct_diff_{window}"] = _group_diff(base, group, "atr_pct_14", window)
        base[f"traj_range_pct_diff_{window}"] = _group_diff(base, group, "range_pct", window)
        base[f"traj_bb_width_diff_{window}"] = _group_diff(base, group, "bb_width_20", window)
        base[f"traj_realized_vol_diff_{window}"] = _group_diff(base, group, "realized_vol_20", window)
        base[f"traj_compression_diff_{window}"] = _group_diff(base, group, "compression_composite", window)
        base[f"traj_ema20_dist_diff_{window}"] = _group_diff(base, group, "ema_20_dist", window)
        base[f"traj_di_diff_change_{window}"] = _group_diff(base, group, "di_diff", window)
        base[f"traj_body_mean_{window}"] = group["traj_body_pct"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        base[f"traj_close_location_mean_{window}"] = group["traj_close_location"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        base[f"traj_wick_imbalance_mean_{window}"] = group["traj_wick_imbalance"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    for column in [
        "volume",
        "turnover_lacs",
        "delivery_pct",
        "fut_oi",
        "fut_vol",
        "pcr_oi",
        "atm_iv",
        "range_pct",
        "bb_width_20",
        "realized_vol_20",
        "compression_composite",
    ]:
        base[f"traj_{column}_z_63"] = _rolling_z(base, group, column, 63, 20)

    rolling_high = group["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    rolling_low = group["low"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).min())
    rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
    base["traj_close_position_20"] = _safe_div(close - rolling_low, rolling_range, default=0.5)
    base["traj_breakout_20"] = _safe_div(close, rolling_high, default=1.0) - 1.0
    base["traj_breakdown_20"] = _safe_div(close, rolling_low, default=1.0) - 1.0
    base["traj_range_expansion_1v20"] = _safe_div(
        _num(base, "range_pct"),
        group["range_pct"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()),
        default=1.0,
    )
    base["traj_volume_expansion_1v20"] = _safe_div(
        _num(base, "volume"),
        group["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()),
        default=1.0,
    )
    base["traj_turnover_expansion_1v20"] = _safe_div(
        _num(base, "turnover_lacs"),
        group["turnover_lacs"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()),
        default=1.0,
    )
    base["traj_ret_accel_3v10"] = base["traj_close_ret_3"] - base["traj_close_ret_10"] * 0.30
    base["traj_volume_accel_3v10"] = base["traj_volume_chg_3"] - base["traj_volume_chg_10"] * 0.30
    base["traj_oi_accel_3v10"] = base["traj_fut_oi_chg_3"] - base["traj_fut_oi_chg_10"] * 0.30
    base["traj_iv_accel_3v10"] = base["traj_atm_iv_diff_3"] - base["traj_atm_iv_diff_10"] * 0.30
    base["traj_compression_release"] = (
        -base["traj_bb_width_diff_5"].fillna(0.0)
        * base["traj_range_expansion_1v20"].clip(0.0, 5.0).fillna(1.0)
    )
    base["traj_liquidity_impulse"] = (
        base["traj_turnover_expansion_1v20"].clip(0.0, 6.0).fillna(1.0)
        + base["traj_volume_expansion_1v20"].clip(0.0, 6.0).fillna(1.0)
    ) / 2.0
    base["traj_iv_oi_impulse"] = base["traj_iv_accel_3v10"].fillna(0.0) * base[
        "traj_oi_accel_3v10"
    ].fillna(0.0)

    base["traj_rel_ret_5d_vs_nifty"] = _first_existing(base, ["rel_ret_5d_vs_nifty", "relative_return_5d"])
    base["traj_rel_sector_ret_5d"] = _first_existing(base, ["stock_rel_sector_ret_5d"])
    base["traj_rel_sector_ret_20d"] = _first_existing(base, ["stock_rel_sector_ret_20d"])
    base["traj_nifty_ret_5d"] = _first_existing(base, ["nifty_ret_5d"])
    base["traj_sector_ret_5d"] = _first_existing(base, ["sector_ret_5d"])
    base["traj_market_breadth"] = _first_existing(base, ["mkt_pct_above_sma20", "mkt_advance_ratio"], 0.5)
    base["traj_nifty_realized_vol_20"] = _first_existing(base, ["nifty_realized_vol_20"])
    daily_market = (
        base[["date", "traj_market_breadth", "traj_nifty_realized_vol_20"]]
        .drop_duplicates("date")
        .sort_values("date")
        .copy()
    )
    daily_market["traj_market_breadth_change_5"] = daily_market["traj_market_breadth"].diff(5)
    vol = pd.to_numeric(daily_market["traj_nifty_realized_vol_20"], errors="coerce")
    daily_market["traj_market_regime_vol_z"] = (
        vol - vol.shift(1).rolling(252, min_periods=60).mean()
    ) / vol.shift(1).rolling(252, min_periods=60).std()
    base = base.merge(
        daily_market[["date", "traj_market_breadth_change_5", "traj_market_regime_vol_z"]],
        on="date",
        how="left",
    )

    for column in [
        "traj_close_ret_5",
        "traj_ret_accel_3v10",
        "traj_volume_accel_3v10",
        "traj_oi_accel_3v10",
        "traj_iv_accel_3v10",
        "traj_compression_release",
        "traj_liquidity_impulse",
        "traj_close_position_20",
        "traj_rel_ret_5d_vs_nifty",
    ]:
        base[f"{column}_xrank"] = _daily_rank(base, pd.to_numeric(base[column], errors="coerce"))

    bins_breadth = [-np.inf, 0.35, 0.55, 0.75, np.inf]
    bins_vol = [-np.inf, -0.5, 0.5, np.inf]
    base["traj_regime_breadth_bin"] = pd.cut(
        pd.to_numeric(base["traj_market_breadth"], errors="coerce"),
        bins=bins_breadth,
        labels=False,
    ).fillna(1)
    base["traj_regime_vol_bin"] = pd.cut(
        pd.to_numeric(base["traj_market_regime_vol_z"], errors="coerce"),
        bins=bins_vol,
        labels=False,
    ).fillna(1)

    base = base.copy()
    traj_cols = ["date", "symbol", *[c for c in base.columns if c.startswith("traj_")]]
    df = df.merge(base[traj_cols], on=["date", "symbol"], how="left")

    long_side = df["side"].astype(str).eq("long")
    sign = np.where(long_side, 1.0, -1.0)
    for window in (1, 2, 3, 5, 10):
        df[f"traj_side_ret_{window}"] = sign * pd.to_numeric(df[f"traj_close_ret_{window}"], errors="coerce")
        df[f"traj_side_body_mean_{window}"] = sign * pd.to_numeric(
            df[f"traj_body_mean_{window}"], errors="coerce"
        )
        df[f"traj_side_di_change_{window}"] = sign * pd.to_numeric(
            df[f"traj_di_diff_change_{window}"], errors="coerce"
        )
        df[f"traj_side_pcr_oi_pressure_{window}"] = np.where(
            long_side,
            -pd.to_numeric(df[f"traj_pcr_oi_diff_{window}"], errors="coerce"),
            pd.to_numeric(df[f"traj_pcr_oi_diff_{window}"], errors="coerce"),
        )

    df["traj_side_gap_pct"] = sign * pd.to_numeric(df["traj_gap_pct"], errors="coerce")
    df["traj_side_wick_pressure"] = np.where(
        long_side,
        pd.to_numeric(df["traj_wick_imbalance"], errors="coerce"),
        -pd.to_numeric(df["traj_wick_imbalance"], errors="coerce"),
    )
    df["traj_side_close_position_20"] = np.where(
        long_side,
        pd.to_numeric(df["traj_close_position_20"], errors="coerce"),
        1.0 - pd.to_numeric(df["traj_close_position_20"], errors="coerce"),
    )
    df["traj_side_break_pressure_20"] = np.where(
        long_side,
        pd.to_numeric(df["traj_breakout_20"], errors="coerce"),
        -pd.to_numeric(df["traj_breakdown_20"], errors="coerce"),
    )
    df["traj_side_rel_nifty_5"] = sign * pd.to_numeric(df["traj_rel_ret_5d_vs_nifty"], errors="coerce")
    df["traj_side_rel_sector_5"] = sign * pd.to_numeric(df["traj_rel_sector_ret_5d"], errors="coerce")
    df["traj_side_ret_accel_3v10"] = sign * pd.to_numeric(df["traj_ret_accel_3v10"], errors="coerce")
    df["traj_side_ema20_dist_diff_5"] = sign * pd.to_numeric(df["traj_ema20_dist_diff_5"], errors="coerce")
    df["traj_side_oi_confirmation_5"] = (
        df["traj_side_ret_5"].fillna(0.0) * pd.to_numeric(df["traj_fut_oi_chg_5"], errors="coerce").fillna(0.0)
    )
    df["traj_side_volume_confirmation_5"] = (
        df["traj_side_ret_5"].fillna(0.0) * pd.to_numeric(df["traj_volume_accel_3v10"], errors="coerce").fillna(0.0)
    )
    df["traj_side_iv_confirmation_5"] = (
        df["traj_side_ret_5"].fillna(0.0) * pd.to_numeric(df["traj_iv_accel_3v10"], errors="coerce").fillna(0.0)
    )
    df["traj_reversal_setup_score"] = -df["traj_side_ret_10"].fillna(0.0) + df[
        "traj_side_ret_2"
    ].fillna(0.0)
    df["traj_continuation_setup_score"] = (
        df["traj_side_ret_5"].fillna(0.0)
        + df["traj_side_rel_nifty_5"].fillna(0.0)
        + df["traj_side_oi_confirmation_5"].fillna(0.0)
    )
    df["traj_exhaustion_risk"] = (
        df["traj_side_ret_10"].clip(lower=0.0).fillna(0.0)
        - df["traj_side_body_mean_2"].fillna(0.0)
        - df["traj_side_wick_pressure"].fillna(0.0)
    )
    df["traj_setup_quality_score"] = (
        0.35 * df["traj_side_break_pressure_20"].fillna(0.0)
        + 0.25 * df["traj_side_ret_accel_3v10"].fillna(0.0)
        + 0.20 * df["traj_side_volume_confirmation_5"].fillna(0.0)
        + 0.20 * df["traj_compression_release"].fillna(0.0)
        - 0.15 * df["traj_exhaustion_risk"].fillna(0.0)
    )
    return df.replace([np.inf, -np.inf], np.nan)


def _is_safe_trajectory_feature(name: str) -> bool:
    if name in OUTCOME_COLUMNS or name in {"date", "symbol", "side", "band", "setup_id"}:
        return False
    return not any(name.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def trajectory_feature_columns(base_features: list[str], dataset: pd.DataFrame) -> list[str]:
    columns = strict_feature_columns(base_features, dataset)
    for col in dataset.columns:
        if (
            col.startswith("traj_")
            and _is_safe_trajectory_feature(col)
            and pd.api.types.is_numeric_dtype(dataset[col])
            and col not in columns
        ):
            columns.append(col)
    leaked = [c for c in columns if not _is_safe_trajectory_feature(c) and c.startswith("traj_")]
    if leaked:
        raise ValueError(f"Trajectory feature leak detected: {leaked[:20]}")
    return columns


def build_trajectory_dataset(
    config: TrajectoryStrictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    market = _load_trajectory_market_data()
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
    dataset = add_trajectory_features(dataset)
    features = trajectory_feature_columns(model_feature_columns(registry, dataset), dataset)
    return dataset, train_universe, prediction_universe, features


def _classifier(config: TrajectoryStrictConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("run-trajectory-strict-model requires lightgbm")
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
                    colsample_bytree=0.80,
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
    config: TrajectoryStrictConfig,
    random_state: int,
) -> Any:
    y = train[target].astype(int)
    if y.nunique() < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = _classifier(config, random_state)
    model.fit(_clean_x(train, feature_columns), y)
    return model


def _nearest_opposite_indices(hit_dates: pd.Series, candidates: pd.DataFrame, quota: int) -> list[int]:
    if quota <= 0 or candidates.empty or hit_dates.empty:
        return []
    hit_ord = np.sort(pd.to_datetime(hit_dates).astype("int64").to_numpy())
    cand_ord = pd.to_datetime(candidates["date"]).astype("int64").to_numpy()
    pos = np.searchsorted(hit_ord, cand_ord)
    right = np.where(pos < len(hit_ord), np.abs(hit_ord[np.minimum(pos, len(hit_ord) - 1)] - cand_ord), np.inf)
    left = np.where(pos > 0, np.abs(hit_ord[np.maximum(pos - 1, 0)] - cand_ord), np.inf)
    distance = np.minimum(left, right)
    order = np.lexsort((candidates.index.to_numpy(), distance))
    return list(candidates.index.to_numpy()[order[:quota]])


def build_matched_pair_training_frame(
    route_train: pd.DataFrame,
    config: TrajectoryStrictConfig,
) -> pd.DataFrame:
    pair = route_train[route_train["strict_hit"] | route_train["strict_opposite"]].copy()
    hits = pair[pair["strict_hit"]].copy()
    opposites = pair[pair["strict_opposite"]].copy()
    if hits.empty or opposites.empty:
        out = pair.copy()
        out["strict_pair_is_hit"] = out["strict_hit"].astype(int)
        return out

    desired_opposites = min(len(opposites), len(hits) * config.matched_opposites_per_hit)
    selected_opposites: set[int] = set()
    group_cols = [
        c
        for c in [
            "symbol",
            "band",
            "setup_id",
            "traj_regime_breadth_bin",
            "traj_regime_vol_bin",
        ]
        if c in pair.columns
    ]
    if group_cols:
        for key, hit_group in hits.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            mask = pd.Series(True, index=opposites.index)
            for col, value in zip(group_cols, key, strict=True):
                mask &= opposites[col].eq(value)
            candidates = opposites[mask & ~opposites.index.isin(selected_opposites)]
            quota = len(hit_group) * config.matched_opposites_per_hit
            selected_opposites.update(_nearest_opposite_indices(hit_group["date"], candidates, quota))

    if len(selected_opposites) < desired_opposites:
        remaining = opposites[~opposites.index.isin(selected_opposites)]
        selected_opposites.update(
            _nearest_opposite_indices(hits["date"], remaining, desired_opposites - len(selected_opposites))
        )

    matched = pd.concat([hits, opposites.loc[sorted(selected_opposites)]], ignore_index=False)
    if len(matched) < config.min_matched_pair_rows:
        matched = pair.copy()
    matched = matched.copy()
    matched["strict_pair_is_hit"] = matched["strict_hit"].astype(int)
    return matched


def _fit_route_bundle(
    route_train: pd.DataFrame,
    archetype: str,
    side: str,
    feature_columns: list[str],
    config: TrajectoryStrictConfig,
    random_state: int,
    fallback_route: str | None = None,
) -> TrajectoryRouteBundle:
    pair_train = build_matched_pair_training_frame(route_train, config)
    return TrajectoryRouteBundle(
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
        matched_hits=int(pair_train["strict_hit"].sum()) if not pair_train.empty else 0,
        matched_opposites=int(pair_train["strict_opposite"].sum()) if not pair_train.empty else 0,
        fallback_route=fallback_route,
    )


def train_trajectory_split_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    archetypes: pd.DataFrame,
    split: WalkForwardSplit,
    config: TrajectoryStrictConfig,
) -> TrajectorySplitModel:
    train = dataset[
        dataset["status"].eq("evaluated")
        & between_dates(dataset, start=config.train_start, end=split.base_train_end)
    ].copy()
    train = train.merge(archetypes[["symbol", "archetype", "primary_side"]], on="symbol", how="left")
    train["archetype"] = train["archetype"].fillna("mid_liquidity_mixed")

    route_bundles: dict[tuple[str, str], TrajectoryRouteBundle] = {}
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
            seed += 67

    return TrajectorySplitModel(
        model_id=f"{config.run_id}_{split.name}_trajectory_strict",
        split=split,
        feature_columns=feature_columns,
        config=config,
        route_bundles=route_bundles,
    )


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def predict_trajectory_split_model(
    model: TrajectorySplitModel,
    dataset: pd.DataFrame,
    prediction_universe: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> pd.DataFrame:
    allowed_symbols = set(prediction_universe["symbol"].astype(str))
    period = dataset[dataset["symbol"].astype(str).isin(allowed_symbols)].copy()
    period = period.merge(archetypes[["symbol", "archetype", "primary_side"]], on="symbol", how="left")
    period["archetype"] = period["archetype"].fillna("mid_liquidity_mixed")
    period["primary_side"] = period["primary_side"].fillna("both")

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
        "traj_setup_quality_score",
        "traj_reversal_setup_score",
        "traj_continuation_setup_score",
        "traj_exhaustion_risk",
        "traj_side_ret_5",
        "traj_side_rel_nifty_5",
        "traj_side_oi_confirmation_5",
        "traj_side_volume_confirmation_5",
    ]
    frames: list[pd.DataFrame] = []
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
        out["p_traj_pair_hit"] = np.clip(_positive_probability(bundle.pair_hit_model, x), 0, 1)
        out["p_traj_strict_hit"] = np.clip(_positive_probability(bundle.full_hit_model, x), 0, 1)
        out["p_traj_opposite"] = np.clip(_positive_probability(bundle.opposite_model, x), 0, 1)
        out["p_traj_range"] = np.clip(_positive_probability(bundle.range_model, x), 0, 1)
        out["traj_edge"] = out["p_traj_pair_hit"] - out["p_traj_opposite"]
        out["trajectory_utility"] = (
            model.config.pair_weight * out["p_traj_pair_hit"]
            + model.config.full_hit_weight * out["p_traj_strict_hit"]
            - model.config.opposite_penalty * out["p_traj_opposite"]
            - model.config.range_penalty * out["p_traj_range"]
            + model.config.setup_score_weight * out.get("traj_setup_quality_score", 0.0)
        )
        out["model_id"] = model.model_id
        out["route_model_id"] = f"{model.model_id}_{archetype}_{side}"
        out["route_fallback"] = bundle.fallback_route or ""
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "trajectory_utility"], ascending=[True, False])


def select_trajectory_weekly_signals(
    predictions: pd.DataFrame,
    config: TrajectoryStrictConfig,
) -> pd.DataFrame:
    df = predictions.copy().sort_values(["date", "trajectory_utility"], ascending=[True, False])
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selection_reason"] = "candidate_not_selected"
    df["selector_config_id"] = f"{config.run_id}_raw_trajectory_weekly_selector"
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.strftime("%G-W%V")

    keep = (
        df.sort_values(["date", "symbol", "trajectory_utility"], ascending=[True, True, False])
        .drop_duplicates(["date", "symbol"], keep="first")
        .index
    )
    eligible = df.index.isin(keep)
    df.loc[~eligible, "selection_reason"] = "symbol_side_conflict"

    gates = [
        ("route_side_disabled", ~df["route_allowed"].astype(bool)),
        ("below_pair_hit_probability", df["p_traj_pair_hit"].lt(config.min_pair_hit_probability)),
        ("below_full_hit_probability", df["p_traj_strict_hit"].lt(config.min_full_hit_probability)),
        ("above_opposite_probability", df["p_traj_opposite"].gt(config.max_opposite_probability)),
        ("above_range_probability", df["p_traj_range"].gt(config.max_range_probability)),
        ("below_trajectory_edge", df["traj_edge"].lt(config.min_trajectory_edge)),
        ("below_trajectory_utility", df["trajectory_utility"].lt(config.min_trajectory_utility)),
    ]
    for reason, mask in gates:
        reject = eligible & mask
        df.loc[reject, "selection_reason"] = reason
        eligible &= ~reject

    df["date_pool_rank"] = (
        df[eligible].groupby("date")["trajectory_utility"].rank(method="first", ascending=False)
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
        for idx, row in week_rows.sort_values("trajectory_utility", ascending=False).iterrows():
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

    return df.sort_values(["date", "selected", "trajectory_utility"], ascending=[True, False, False])


def build_raw_bucket_report(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions[predictions["status"].eq("evaluated")].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(str)
    rows: list[dict[str, object]] = []
    bucket_sizes = [25, 50, 100, 200, 300, 500, 1000]

    def add_rows(scope: str, group: pd.DataFrame) -> None:
        ordered = group.sort_values("trajectory_utility", ascending=False)
        for n in bucket_sizes:
            top = ordered.head(n)
            if top.empty:
                continue
            rows.append(
                {
                    "scope": scope,
                    "top_n": int(min(n, len(top))),
                    "evaluated": int(len(top)),
                    "strict_hit_rate": float(top["strict_hit"].mean()),
                    "strict_opposite_rate": float(top["strict_opposite"].mean()),
                    "strict_range_bound_rate": float(top["strict_range_bound"].mean()),
                    "avg_trajectory_utility": float(top["trajectory_utility"].mean()),
                    "unique_symbols": int(top["symbol"].nunique()),
                }
            )

    add_rows("ALL", df)
    for year, group in df.groupby("year", sort=True):
        add_rows(str(year), group)
    for (year, archetype, side), group in df.groupby(["year", "archetype", "side"], sort=True):
        add_rows(f"{year}|{archetype}|{side}", group)
    return pd.DataFrame(rows)


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def _save_model(model: TrajectorySplitModel, output_dir: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    bundle_rows = [
        {
            "archetype": bundle.archetype,
            "side": bundle.side,
            "train_rows": bundle.train_rows,
            "pair_train_rows": bundle.pair_train_rows,
            "matched_hits": bundle.matched_hits,
            "matched_opposites": bundle.matched_opposites,
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
            "algorithm": "raw_lightgbm_trajectory_matched_archetype_side_specialists",
            "probability_output": "raw_lightgbm_predict_proba",
            "calibration": "none",
            "matching": "strict_hit_vs_nearest_strict_opposite_by_symbol_setup_regime",
            "feature_count": len(model.feature_columns),
            "feature_columns": model.feature_columns,
        },
        output_dir / "manifest.json",
    )


def run_trajectory_strict_model(config: TrajectoryStrictConfig) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is required for run-trajectory-strict-model")

    run_dir = RUNS_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, train_universe, prediction_universe, feature_columns = build_trajectory_dataset(config)

    train_universe.to_csv(run_dir / "training_universe.csv", index=False)
    prediction_universe.to_csv(run_dir / "prediction_universe.csv", index=False)
    (run_dir / "feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")

    selected_frames: list[pd.DataFrame] = []
    raw_bucket_frames: list[pd.DataFrame] = []
    archetype_frames: list[pd.DataFrame] = []
    for split in DEFAULT_SPLITS:
        split_dir = run_dir / "model_predictions" / split.name
        completed_signals = split_dir / "signals.parquet"
        completed_archetypes = split_dir / "archetypes.csv"
        completed_raw_bucket = split_dir / "raw_bucket_report.csv"
        completed_model = run_dir / "models" / split.name / "model.joblib"
        if (
            config.resume_completed_splits
            and completed_signals.exists()
            and completed_archetypes.exists()
            and completed_raw_bucket.exists()
            and completed_model.exists()
        ):
            selected = pd.read_parquet(completed_signals)
            selected["split"] = split.name
            archetypes = pd.read_csv(completed_archetypes)
            raw_bucket = pd.read_csv(completed_raw_bucket)
            raw_bucket["split"] = split.name
            selected_frames.append(selected)
            archetype_frames.append(archetypes)
            raw_bucket_frames.append(raw_bucket)
            print(f"Reused completed trajectory split: {split.name}", flush=True)
            continue

        print(f"Training trajectory split: {split.name}", flush=True)
        archetypes = assign_split_archetypes(dataset, train_universe, split, config)
        archetype_frames.append(archetypes)
        model = train_trajectory_split_model(dataset, feature_columns, archetypes, split, config)
        _save_model(model, run_dir / "models" / split.name)

        period = dataset[between_dates(dataset, split.prediction_start, split.prediction_end)].copy()
        predictions = predict_trajectory_split_model(model, period, prediction_universe, archetypes)
        selected = select_trajectory_weekly_signals(predictions, config)
        selected["split"] = split.name
        raw_bucket = build_raw_bucket_report(predictions)
        raw_bucket["split"] = split.name

        split_dir.mkdir(parents=True, exist_ok=True)
        archetypes.to_csv(split_dir / "archetypes.csv", index=False)
        predictions.to_parquet(split_dir / "trajectory_predictions.parquet", index=False)
        selected.to_parquet(split_dir / "signals.parquet", index=False)
        selected[selected["selected"]].to_csv(split_dir / "selected_signals.csv", index=False)
        raw_bucket.to_csv(split_dir / "raw_bucket_report.csv", index=False)
        write_report_tables(build_strict_hit_report(selected), split_dir / "strict_report")
        write_report_tables(build_gold_report(selected), split_dir / "gold_report")
        _write_tables(build_strict_capture_report(selected, period), split_dir / "capture_report")
        selected_frames.append(selected)
        raw_bucket_frames.append(raw_bucket)

    combined = pd.concat(selected_frames, ignore_index=True)
    all_archetypes = pd.concat(archetype_frames, ignore_index=True)
    raw_buckets = pd.concat(raw_bucket_frames, ignore_index=True)
    combined.to_parquet(run_dir / "all_signals.parquet", index=False)
    combined[combined["selected"]].to_csv(run_dir / "selected_signals.csv", index=False)
    all_archetypes.to_csv(run_dir / "all_split_archetypes.csv", index=False)
    raw_buckets.to_csv(run_dir / "combined_raw_bucket_report.csv", index=False)
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
            "trajectory_feature_count": int(sum(c.startswith("traj_") for c in feature_columns)),
            "training_symbols": int(train_universe["symbol"].nunique()),
            "prediction_symbols": int(prediction_universe["symbol"].nunique()),
            "splits": [asdict(split) for split in DEFAULT_SPLITS],
            "notes": [
                "Trajectory-feature strict-hit model with matched hit-vs-opposite pair training.",
                "Training can use all eligible raw-data symbols; prediction is restricted by prediction_top_n.",
                "All probabilities are raw LightGBM predict_proba outputs.",
                "No calibration selector, no overlays, no old Koscine 2.0 predictions.",
            ],
        },
        run_dir / "manifest.json",
    )
    return run_dir
