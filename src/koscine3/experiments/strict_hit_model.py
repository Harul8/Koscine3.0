from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from koscine3.datasets.splits import DEFAULT_SPLITS, WalkForwardSplit, between_dates
from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.evaluation.reports import write_manifest, write_report_tables
from koscine3.experiments.large_move_ranker import add_setup_features
from koscine3.paths import RUNS_DIR


try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


LOCKED_SOURCE_RUN_ID = "koscine3_crossregime_v19_full_v8_n80"

OUTCOME_COLUMNS = {
    "entry_date",
    "entry_open",
    "window_end_date",
    "window_close",
    "window_high",
    "window_low",
    "window_observations",
    "threshold",
    "favorable_move",
    "signed_close_return",
    "hit",
    "near",
    "hit_or_near",
    "clean_success",
    "opposite_close",
    "opposite",
    "small",
    "status",
    "verdict",
    "strict_hit",
    "strict_opposite",
    "strict_range_bound",
    "strict_category",
    "strict_pair_target",
}
BLOCKED_PREFIXES = (
    "future_",
    "entry_",
    "up_move_",
    "down_move_",
    "fwd_return_",
    "long_adverse_",
    "short_adverse_",
    "label_",
)
ENGINEERED_PREFIXES = (
    "setup_",
    "strictatlas_",
    "strict_hist_",
    "strict_side_",
    "micro_",
    "shock_",
    "iv_",
    "liquidity_",
    "side_",
    "xrank_",
    "positioning_",
    "regime_",
    "id_",
)


@dataclass(frozen=True)
class StrictHitModelConfig:
    run_id: str = "koscine3_strict_hit_v1"
    source_run_id: str = LOCKED_SOURCE_RUN_ID
    train_start: str = "2018-01-01"
    n_estimators: int = 180
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 100
    strict_close_fraction: float = 0.80
    opposite_close_floor: float = 0.01
    weekly_target: int = 5
    max_signals_per_day: int = 2
    max_signals_per_week_side: int = 4
    max_symbol_per_week: int = 1
    max_symbol_per_month: int = 3
    daily_pool_rank: int = 24
    min_pair_hit_probability: float = 0.56
    min_full_hit_probability: float = 0.06
    max_opposite_probability: float = 0.62
    max_range_probability: float = 0.90
    min_strict_edge: float = -0.05
    pair_weight: float = 1.40
    full_hit_weight: float = 1.00
    opposite_penalty: float = 1.20
    range_penalty: float = 0.20
    favorable_move_weight: float = 0.25
    signed_close_weight: float = 0.35
    random_state: int = 53


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


@dataclass
class StrictSideBundle:
    side: str
    pair_hit_model: Any
    full_hit_model: Any
    opposite_model: Any
    range_model: Any
    favorable_model: Any
    signed_close_model: Any
    train_rows: int
    pair_train_rows: int


@dataclass
class StrictHitSplitModel:
    model_id: str
    split: WalkForwardSplit
    feature_columns: list[str]
    config: StrictHitModelConfig
    algorithm: str
    long_bundle: StrictSideBundle
    short_bundle: StrictSideBundle


def load_strict_source_dataset(config: StrictHitModelConfig) -> tuple[pd.DataFrame, list[str]]:
    run_dir = RUNS_DIR / config.source_run_id
    dataset_path = run_dir / "dataset.parquet"
    manifest_path = run_dir / "dataset_manifest.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {dataset_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    dataset = pd.read_parquet(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_columns = [c for c in manifest["feature_columns"] if c in dataset.columns]
    return dataset, feature_columns


def add_strict_hit_labels(dataset: pd.DataFrame, config: StrictHitModelConfig) -> pd.DataFrame:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_numeric(df["target_threshold"], errors="coerce")
    signed_close = pd.to_numeric(df["signed_close_return"], errors="coerce")
    hit = df["hit"].fillna(False).astype(bool)

    df["strict_hit"] = hit & signed_close.gt(config.strict_close_fraction * target)
    df["strict_opposite"] = ~df["strict_hit"] & signed_close.lt(config.opposite_close_floor)
    df["strict_range_bound"] = ~(df["strict_hit"] | df["strict_opposite"])
    df["strict_category"] = np.select(
        [df["strict_hit"], df["strict_opposite"], df["strict_range_bound"]],
        ["hit", "opposite", "range_bound"],
        default="pending",
    )
    pair_target = pd.Series(np.nan, index=df.index, dtype=float)
    pair_target.loc[df["strict_hit"]] = 1.0
    pair_target.loc[df["strict_opposite"]] = 0.0
    df["strict_pair_target"] = pair_target
    return df


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_div(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    out = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_z(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce")
    grouped = values.groupby(df[group_col], sort=False)
    mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    std = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
    return _safe_div(values - mean, std)


def _rolling_median_ratio(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce")
    median = values.groupby(df[group_col], sort=False).transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).median()
    )
    return _safe_div(values, median, default=1.0)


def _daily_rank(df: pd.DataFrame, value: pd.Series, ascending: bool = True) -> pd.Series:
    return value.groupby(df["date"]).rank(pct=True, ascending=ascending).fillna(0.5)


def add_strict_research_features(
    dataset: pd.DataFrame,
    include_symbol_identity: bool = True,
) -> pd.DataFrame:
    df = dataset.copy().sort_values(["symbol", "side", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    high = _num(df, "high")
    low = _num(df, "low")
    open_ = _num(df, "open")
    close = _num(df, "close")
    prev_close = _num(df, "prev_close")
    price_range = (high - low).abs()
    positive_range = price_range.replace(0, np.nan)

    df["micro_close_location_value"] = _safe_div(close - low, positive_range, default=0.5)
    df["micro_body_to_range"] = _safe_div((close - open_).abs(), positive_range)
    df["micro_upper_wick_pct"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), positive_range)
    df["micro_lower_wick_pct"] = _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, positive_range)
    df["micro_wick_imbalance"] = df["micro_lower_wick_pct"] - df["micro_upper_wick_pct"]
    df["micro_gap_pct"] = _safe_div(open_ - prev_close, prev_close)
    df["micro_intraday_return"] = _safe_div(close - open_, open_)
    df["micro_gap_follow_through"] = np.sign(df["micro_gap_pct"]) * df["micro_intraday_return"]
    df["micro_parkinson_vol_proxy"] = (np.log(_safe_div(high, low, default=1.0)).pow(2)) / (
        4.0 * np.log(2.0)
    )
    log_hl = np.log(_safe_div(high, low, default=1.0))
    log_co = np.log(_safe_div(close, open_, default=1.0))
    df["micro_garman_klass_proxy"] = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    df["micro_amihud_illiq"] = _safe_div(_num(df, "ret_1d").abs(), _num(df, "turnover_lacs") + 1.0)

    df["iv_atm_z_252"] = _rolling_z(df, "symbol", "atm_iv", 252, 60)
    df["iv_atm_ratio_to_symbol_median_252"] = _rolling_median_ratio(df, "symbol", "atm_iv", 252, 60)
    df["iv_atm_minus_realized_vol_20"] = _num(df, "atm_iv") - _num(df, "realized_vol_20")
    df["iv_change_5d_x_range_contraction"] = _num(df, "atm_iv_chg_5") * _num(
        df, "range_contraction_5v20"
    )
    df["iv_skew_abs"] = _num(df, "put_call_iv_skew").abs()
    df["iv_rank_x_liquid_band"] = _daily_rank(df, _num(df, "atm_iv")) * _num(
        df, "is_liquid_band", 0.0
    )

    df["liquidity_turnover_z_63"] = _rolling_z(df, "symbol", "turnover_lacs", 63, 20)
    df["liquidity_volume_z_63"] = _rolling_z(df, "symbol", "volume", 63, 20)
    df["liquidity_delivery_pct_z_252"] = _rolling_z(df, "symbol", "delivery_pct", 252, 60)
    df["liquidity_delivery_pct_drop_5d"] = -_num(df, "delivery_pct_chg_5")
    df["liquidity_turnover_accel_5v20"] = _num(df, "turnover_ratio_20")
    df["liquidity_fut_vol_accel_5v20"] = _num(df, "fut_vol_ratio_20")
    df["liquidity_amihud_rank"] = _daily_rank(df, df["micro_amihud_illiq"], ascending=False)

    daily = (
        df.drop_duplicates("date")
        .sort_values("date")
        .loc[:, ["date", "nifty_realized_vol_20", "xsec_positive_share_5d", "xsec_positive_share_20d"]]
        .copy()
    )
    daily["regime_nifty_vol_z_252"] = (
        daily["nifty_realized_vol_20"]
        - daily["nifty_realized_vol_20"].shift(1).rolling(252, min_periods=60).mean()
    ) / daily["nifty_realized_vol_20"].shift(1).rolling(252, min_periods=60).std()
    daily["regime_market_breadth_drop_5_vs_20"] = (
        daily["xsec_positive_share_20d"] - daily["xsec_positive_share_5d"]
    )
    df = df.merge(
        daily[["date", "regime_nifty_vol_z_252", "regime_market_breadth_drop_5_vs_20"]],
        on="date",
        how="left",
    )
    df["regime_stock_vol_z_minus_market_vol_z"] = _rolling_z(
        df, "symbol", "symbol_volatility_20d", 252, 60
    ) - df["regime_nifty_vol_z_252"].fillna(0.0)

    side = df["side"].astype(str)
    long_side = side.eq("long")
    df["side_aligned_ret_5d"] = np.where(long_side, _num(df, "ret_5d"), -_num(df, "ret_5d"))
    df["side_aligned_ret_20d"] = np.where(long_side, _num(df, "ret_20d"), -_num(df, "ret_20d"))
    df["side_aligned_relative_return_20d"] = np.where(
        long_side, _num(df, "relative_return_20d"), -_num(df, "relative_return_20d")
    )
    df["side_aligned_gap_pct"] = np.where(long_side, df["micro_gap_pct"], -df["micro_gap_pct"])
    df["side_aligned_intraday_return"] = np.where(
        long_side, df["micro_intraday_return"], -df["micro_intraday_return"]
    )
    df["side_aligned_close_location"] = np.where(
        long_side, df["micro_close_location_value"], 1.0 - df["micro_close_location_value"]
    )
    df["side_aligned_wick_pressure"] = np.where(
        long_side, df["micro_wick_imbalance"], -df["micro_wick_imbalance"]
    )
    df["side_aligned_di_diff"] = np.where(long_side, _num(df, "di_diff"), -_num(df, "di_diff"))
    long_oi = _num(df, "oi_long_buildup_5d") + _num(df, "oi_short_unwind_5d")
    short_oi = _num(df, "oi_short_buildup_5d") + _num(df, "oi_long_unwind_5d")
    df["side_aligned_oi_pressure"] = np.where(long_side, long_oi, short_oi)
    df["positioning_side_aligned_oi_change"] = np.where(
        long_side, _num(df, "fut_oi_chg_5"), -_num(df, "fut_oi_chg_5")
    )
    df["positioning_oi_iv_impulse"] = df["positioning_side_aligned_oi_change"] * _num(
        df, "atm_iv_chg_5"
    )
    df["positioning_pcr_oi_side_pressure"] = np.where(long_side, -_num(df, "pcr_oi_chg_5"), _num(df, "pcr_oi_chg_5"))

    rank_sources = {
        "xrank_atm_iv": _num(df, "atm_iv"),
        "xrank_atr_pct_14": _num(df, "atr_pct_14"),
        "xrank_turnover_z": df["liquidity_turnover_z_63"],
        "xrank_volume_z": df["liquidity_volume_z_63"],
        "xrank_side_aligned_ret_20d": pd.Series(df["side_aligned_ret_20d"], index=df.index),
        "xrank_side_aligned_close_location": pd.Series(
            df["side_aligned_close_location"], index=df.index
        ),
        "xrank_iv_realized_dislocation": df["iv_atm_minus_realized_vol_20"],
    }
    for name, values in rank_sources.items():
        df[name] = _daily_rank(df, pd.to_numeric(values, errors="coerce"))

    target = _num(df, "target_threshold", 0.04)
    for col in [
        "iv_atm_z_252",
        "liquidity_turnover_z_63",
        "side_aligned_ret_20d",
        "side_aligned_close_location",
        "regime_market_breadth_drop_5_vs_20",
    ]:
        df[f"{col}_x_target"] = df[col].fillna(0.0) * target

    identity_cols = ["band", "setup_id"]
    identity_prefixes = ["id_band", "id_setup"]
    if include_symbol_identity:
        identity_cols.insert(0, "symbol")
        identity_prefixes.insert(0, "id_symbol")
    identity_source = df[identity_cols].fillna("unknown").astype(str)
    identity = pd.get_dummies(
        identity_source,
        prefix=identity_prefixes,
        dtype=np.int8,
    )
    df = pd.concat([df, identity], axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def add_strict_atlas_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy().sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    lag = 6

    def add_daily_priors(group_cols: list[str], prefix: str) -> None:
        nonlocal df
        daily = (
            df[df["status"].eq("evaluated")]
            .groupby(["date", *group_cols], dropna=False)
            .agg(
                strict_hit=("strict_hit", "mean"),
                strict_opposite=("strict_opposite", "mean"),
                strict_range_bound=("strict_range_bound", "mean"),
                signed_close_return=("signed_close_return", "mean"),
                favorable_move=("favorable_move", "mean"),
            )
            .reset_index()
            .sort_values([*group_cols, "date"])
        )
        grouped = daily.groupby(group_cols, dropna=False, sort=False)
        for window in (21, 63, 252):
            min_periods = max(5, window // 4)
            for source in [
                "strict_hit",
                "strict_opposite",
                "strict_range_bound",
                "signed_close_return",
                "favorable_move",
            ]:
                daily[f"{prefix}_{source}_{window}"] = grouped[source].transform(
                    lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
                )
            daily[f"{prefix}_edge_{window}"] = (
                daily[f"{prefix}_strict_hit_{window}"]
                - daily[f"{prefix}_strict_opposite_{window}"]
            )
        daily[f"{prefix}_success_drop_21_vs_252"] = (
            daily[f"{prefix}_strict_hit_252"] - daily[f"{prefix}_strict_hit_21"]
        )
        daily[f"{prefix}_opposite_spike_21_vs_252"] = (
            daily[f"{prefix}_strict_opposite_21"] - daily[f"{prefix}_strict_opposite_252"]
        )
        keep_cols = ["date", *group_cols, *[c for c in daily.columns if c.startswith(prefix)]]
        df = df.merge(daily[keep_cols], on=["date", *group_cols], how="left")

    add_daily_priors(["side"], "strictatlas_side")
    add_daily_priors(["side", "band"], "strictatlas_band_side")
    add_daily_priors(["side", "setup_id"], "strictatlas_setup_side")
    add_daily_priors(["symbol", "side"], "strictatlas_symbol_side")
    return df.replace([np.inf, -np.inf], np.nan)


def _is_safe_feature(name: str) -> bool:
    if name in OUTCOME_COLUMNS or name in {"date", "symbol", "side", "band"}:
        return False
    return not any(name.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def strict_feature_columns(base_features: list[str], dataset: pd.DataFrame) -> list[str]:
    columns = [c for c in base_features if c in dataset.columns and _is_safe_feature(c)]
    engineered = [
        c
        for c in dataset.columns
        if c != "setup_id"
        and _is_safe_feature(c)
        and any(c.startswith(prefix) for prefix in ENGINEERED_PREFIXES)
        and pd.api.types.is_numeric_dtype(dataset[c])
    ]
    for col in engineered:
        if col not in columns:
            columns.append(col)
    leaked = [c for c in columns if not _is_safe_feature(c)]
    if leaked:
        raise ValueError(f"Strict-hit feature leak detected: {leaked[:20]}")
    return columns


def prepare_strict_dataset(
    config: StrictHitModelConfig,
) -> tuple[pd.DataFrame, list[str]]:
    dataset, base_features = load_strict_source_dataset(config)
    dataset = add_strict_hit_labels(dataset, config)
    dataset = add_setup_features(dataset)
    dataset = add_strict_research_features(dataset)
    dataset = add_strict_atlas_features(dataset)
    features = strict_feature_columns(base_features, dataset)
    return dataset, features


def _classifier(config: StrictHitModelConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("run-strict-hit-model requires lightgbm")
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


def _regressor(config: StrictHitModelConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("run-strict-hit-model requires lightgbm")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    num_leaves=config.num_leaves,
                    min_child_samples=config.min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.85,
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
    config: StrictHitModelConfig,
    random_state: int,
) -> Any:
    y_train = train[target].astype(int)
    if y_train.nunique() < 2:
        return ConstantProbabilityModel(float(y_train.mean()))
    base = _classifier(config, random_state)
    base.fit(_clean_x(train, feature_columns), y_train)
    return base


def _fit_regression_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: StrictHitModelConfig,
    random_state: int,
) -> Any:
    y = pd.to_numeric(train[target], errors="coerce").astype(float)
    if y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()))
    model = _regressor(config, random_state)
    model.fit(_clean_x(train, feature_columns), y)
    return model


def _side_training_frames(
    dataset: pd.DataFrame,
    side: str,
    split: WalkForwardSplit,
    config: StrictHitModelConfig,
) -> pd.DataFrame:
    side_df = dataset[
        dataset["side"].eq(side)
        & dataset["status"].eq("evaluated")
        & between_dates(dataset, start=config.train_start, end=split.base_train_end)
    ].copy()
    train = side_df[between_dates(side_df, end=split.base_train_end)].copy()
    if train.empty:
        raise ValueError(f"No strict-hit training rows for side={side} split={split.name}")
    return train


def train_strict_side_bundle(
    dataset: pd.DataFrame,
    side: str,
    split: WalkForwardSplit,
    feature_columns: list[str],
    config: StrictHitModelConfig,
    random_state: int,
) -> StrictSideBundle:
    train = _side_training_frames(dataset, side, split, config)
    pair_train = train[train["strict_pair_target"].notna()].copy()
    if pair_train.empty:
        raise ValueError(f"No strict-hit/opposite pair rows for side={side} split={split.name}")

    pair_train["strict_pair_is_hit"] = pair_train["strict_pair_target"].astype(int)

    pair_hit_model = _fit_probability_model(
        pair_train,
        feature_columns,
        "strict_pair_is_hit",
        config,
        random_state,
    )
    full_hit_model = _fit_probability_model(
        train,
        feature_columns,
        "strict_hit",
        config,
        random_state + 11,
    )
    opposite_model = _fit_probability_model(
        train,
        feature_columns,
        "strict_opposite",
        config,
        random_state + 23,
    )
    range_model = _fit_probability_model(
        train,
        feature_columns,
        "strict_range_bound",
        config,
        random_state + 37,
    )
    favorable_model = _fit_regression_model(
        train,
        feature_columns,
        "favorable_move",
        config,
        random_state + 47,
    )
    signed_close_model = _fit_regression_model(
        train,
        feature_columns,
        "signed_close_return",
        config,
        random_state + 59,
    )
    return StrictSideBundle(
        side=side,
        pair_hit_model=pair_hit_model,
        full_hit_model=full_hit_model,
        opposite_model=opposite_model,
        range_model=range_model,
        favorable_model=favorable_model,
        signed_close_model=signed_close_model,
        train_rows=int(len(train)),
        pair_train_rows=int(len(pair_train)),
    )


def train_strict_hit_split_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split: WalkForwardSplit,
    config: StrictHitModelConfig,
) -> StrictHitSplitModel:
    model_id = f"{config.run_id}_{split.name}_strict_hit"
    long_bundle = train_strict_side_bundle(
        dataset,
        "long",
        split,
        feature_columns,
        config,
        config.random_state,
    )
    short_bundle = train_strict_side_bundle(
        dataset,
        "short",
        split,
        feature_columns,
        config,
        config.random_state + 101,
    )
    return StrictHitSplitModel(
        model_id=model_id,
        split=split,
        feature_columns=feature_columns,
        config=config,
        algorithm="lightgbm_strict_hit_pair_plus_abstain",
        long_bundle=long_bundle,
        short_bundle=short_bundle,
    )


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        proba = model.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def _predict_regression(model: Any, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        return np.asarray(model.predict(x), dtype=float)


def predict_strict_hit_model(model: StrictHitSplitModel, dataset: pd.DataFrame) -> pd.DataFrame:
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
        "hist_hit_near_rate_252",
        "hist_opposite_rate_252",
        "iv_atm_z_252",
        "liquidity_turnover_z_63",
        "side_aligned_close_location",
        "regime_market_breadth_drop_5_vs_20",
    ]
    for side, bundle in [("long", model.long_bundle), ("short", model.short_bundle)]:
        part = dataset[dataset["side"].eq(side)].copy()
        if part.empty:
            continue
        x = _clean_x(part, model.feature_columns)
        out = part[[c for c in passthrough if c in part.columns]].copy()
        out["p_strict_hit_pair"] = np.clip(_positive_probability(bundle.pair_hit_model, x), 0, 1)
        out["p_strict_hit_full"] = np.clip(_positive_probability(bundle.full_hit_model, x), 0, 1)
        out["p_strict_opposite"] = np.clip(_positive_probability(bundle.opposite_model, x), 0, 1)
        out["p_range_bound"] = np.clip(_positive_probability(bundle.range_model, x), 0, 1)
        out["expected_favorable_move"] = np.maximum(
            _predict_regression(bundle.favorable_model, x), 0.0
        )
        out["expected_signed_close_return"] = _predict_regression(bundle.signed_close_model, x)
        out["strict_edge"] = out["p_strict_hit_pair"] - out["p_strict_opposite"]
        expected_over_target = out["expected_favorable_move"] / pd.to_numeric(
            out["target_threshold"], errors="coerce"
        ).replace(0, np.nan)
        out["strict_hit_utility"] = (
            model.config.pair_weight * out["p_strict_hit_pair"]
            + model.config.full_hit_weight * out["p_strict_hit_full"]
            - model.config.opposite_penalty * out["p_strict_opposite"]
            - model.config.range_penalty * out["p_range_bound"]
            + model.config.favorable_move_weight * expected_over_target.clip(0.0, 3.0).fillna(0.0)
            + model.config.signed_close_weight * out["expected_signed_close_return"]
        )
        out["model_id"] = model.model_id
        frames.append(out)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "strict_hit_utility"], ascending=[True, False])


def select_strict_weekly_signals(
    predictions: pd.DataFrame,
    config: StrictHitModelConfig,
) -> pd.DataFrame:
    df = predictions.copy().sort_values(["date", "strict_hit_utility"], ascending=[True, False])
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selection_reason"] = "candidate_not_selected"
    df["selector_config_id"] = f"{config.run_id}_strict_weekly_selector"
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.strftime("%G-W%V")

    keep = (
        df.sort_values(["date", "symbol", "strict_hit_utility"], ascending=[True, True, False])
        .drop_duplicates(["date", "symbol"], keep="first")
        .index
    )
    conflict_losers = ~df.index.isin(keep)
    df.loc[conflict_losers, "selection_reason"] = "symbol_side_conflict"

    eligible = ~conflict_losers
    gates = [
        ("below_pair_hit_probability", df["p_strict_hit_pair"].lt(config.min_pair_hit_probability)),
        ("below_full_hit_probability", df["p_strict_hit_full"].lt(config.min_full_hit_probability)),
        ("above_opposite_probability", df["p_strict_opposite"].gt(config.max_opposite_probability)),
        ("above_range_probability", df["p_range_bound"].gt(config.max_range_probability)),
        ("below_strict_edge", df["strict_edge"].lt(config.min_strict_edge)),
    ]
    for reason, mask in gates:
        reject = eligible & mask
        df.loc[reject, "selection_reason"] = reason
        eligible &= ~reject

    df["date_pool_rank"] = (
        df[eligible].groupby("date")["strict_hit_utility"].rank(method="first", ascending=False)
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
        candidates = week_rows.sort_values("strict_hit_utility", ascending=False)

        for idx, row in candidates.iterrows():
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

    return df.sort_values(["date", "selected", "strict_hit_utility"], ascending=[True, False, False])


def _add_time_slices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(str)
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out


def summarize_strict_signals(signals: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    df = _add_time_slices(signals)
    if "selected" in df.columns:
        df = df[df["selected"]].copy()
    group_cols = group_cols or []
    rows: list[dict[str, object]] = []
    grouped = [(("ALL",), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        evaluated = group[group["status"].eq("evaluated")]
        calls = len(group)
        evaluated_count = len(evaluated)

        def rate(column: str) -> float:
            if evaluated_count == 0:
                return 0.0
            return float(evaluated[column].fillna(False).mean())

        row: dict[str, object] = {}
        if group_cols:
            row.update(dict(zip(group_cols, key, strict=True)))
        else:
            row["slice"] = "ALL"
        row.update(
            {
                "calls": int(calls),
                "evaluated": int(evaluated_count),
                "pending": int(calls - evaluated_count),
                "strict_hit_rate": rate("strict_hit"),
                "strict_opposite_rate": rate("strict_opposite"),
                "strict_range_bound_rate": rate("strict_range_bound"),
                "avg_favorable_move": float(evaluated["favorable_move"].mean())
                if evaluated_count
                else 0.0,
                "median_favorable_move": float(evaluated["favorable_move"].median())
                if evaluated_count
                else 0.0,
                "avg_signed_close_return": float(evaluated["signed_close_return"].mean())
                if evaluated_count
                else 0.0,
                "unique_symbols": int(group["symbol"].nunique()) if calls else 0,
                "max_symbol_share": float(group["symbol"].value_counts(normalize=True).max())
                if calls
                else 0.0,
                "metric_contract_version": "strict_hit_close80_v1",
            }
        )
        row["passes_strict_bar"] = bool(
            evaluated_count > 0
            and row["strict_hit_rate"] >= 0.60
            and row["strict_opposite_rate"] < 0.25
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_strict_hit_report(signals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = _add_time_slices(signals)
    report: dict[str, pd.DataFrame] = {"aggregate": summarize_strict_signals(df)}
    available = set(df.columns)
    for name, cols in {
        "year": ["year"],
        "quarter": ["quarter"],
        "month": ["month"],
        "side": ["side"],
        "band": ["band"],
        "symbol": ["symbol"],
        "setup": ["setup_id"],
        "model_id": ["model_id"],
        "selector_config_id": ["selector_config_id"],
    }.items():
        if set(cols).issubset(available):
            report[name] = summarize_strict_signals(df, cols)
    daily_source = df[df["selected"]] if "selected" in df.columns else df
    report["daily_counts"] = daily_source.groupby("date").size().reset_index(name="signals")
    return report


def build_strict_capture_report(selected: pd.DataFrame, period: pd.DataFrame) -> dict[str, pd.DataFrame]:
    chosen = _add_time_slices(selected[selected["selected"]])
    evaluated = chosen[chosen["status"].eq("evaluated")].copy()
    period_eval = _add_time_slices(period[period["status"].eq("evaluated")])

    def capture_by(cols: list[str]) -> pd.DataFrame:
        rows = []
        for key, group in period_eval.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            mask = pd.Series(True, index=evaluated.index)
            for col_name, value in zip(cols, key, strict=True):
                mask &= evaluated[col_name].eq(value)
            sub = evaluated[mask]
            available_hits = int(group["strict_hit"].sum())
            rows.append(
                {
                    **dict(zip(cols, key, strict=True)),
                    "available_rows": int(len(group)),
                    "strict_hits_available": available_hits,
                    "selected": int(len(sub)),
                    "selected_strict_hits": int(sub["strict_hit"].sum()) if not sub.empty else 0,
                    "selected_strict_opposite": int(sub["strict_opposite"].sum()) if not sub.empty else 0,
                    "selected_range_bound": int(sub["strict_range_bound"].sum()) if not sub.empty else 0,
                    "strict_hit_capture_rate": float(sub["strict_hit"].sum() / available_hits)
                    if available_hits
                    else 0.0,
                    "strict_hit_precision": float(sub["strict_hit"].mean()) if not sub.empty else 0.0,
                    "strict_opposite_rate": float(sub["strict_opposite"].mean()) if not sub.empty else 0.0,
                    "unique_symbols": int(sub["symbol"].nunique()) if not sub.empty else 0,
                }
            )
        return pd.DataFrame(rows)

    aggregate = pd.DataFrame(
        [
            {
                "slice": "ALL",
                "available_rows": int(len(period_eval)),
                "strict_hits_available": int(period_eval["strict_hit"].sum()),
                "selected": int(len(evaluated)),
                "selected_strict_hits": int(evaluated["strict_hit"].sum()) if not evaluated.empty else 0,
                "selected_strict_opposite": int(evaluated["strict_opposite"].sum())
                if not evaluated.empty
                else 0,
                "selected_range_bound": int(evaluated["strict_range_bound"].sum())
                if not evaluated.empty
                else 0,
                "strict_hit_capture_rate": float(
                    evaluated["strict_hit"].sum() / period_eval["strict_hit"].sum()
                )
                if int(period_eval["strict_hit"].sum())
                else 0.0,
                "strict_hit_precision": float(evaluated["strict_hit"].mean())
                if not evaluated.empty
                else 0.0,
                "strict_opposite_rate": float(evaluated["strict_opposite"].mean())
                if not evaluated.empty
                else 0.0,
                "unique_symbols": int(evaluated["symbol"].nunique()) if not evaluated.empty else 0,
            }
        ]
    )
    return {
        "aggregate_capture": aggregate,
        "year_capture": capture_by(["year"]),
        "quarter_capture": capture_by(["quarter"]),
        "side_capture": capture_by(["side"]),
        "band_capture": capture_by(["band"]),
        "setup_capture": capture_by(["setup_id"]),
        "symbol_capture": capture_by(["symbol"]),
    }


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def write_strict_feature_separation(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    evaluated = dataset[dataset["status"].eq("evaluated")].copy()
    extremes = evaluated[evaluated["strict_hit"] | evaluated["strict_opposite"]].copy()
    rows = []
    if extremes.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    target = extremes["strict_hit"].astype(int)
    for feature in feature_columns:
        if feature not in extremes.columns or not pd.api.types.is_numeric_dtype(extremes[feature]):
            continue
        values = pd.to_numeric(extremes[feature], errors="coerce")
        hit_values = values[target.eq(1)]
        opp_values = values[target.eq(0)]
        if hit_values.notna().sum() < 50 or opp_values.notna().sum() < 50:
            continue
        pooled = float(values.std(skipna=True) or 0.0)
        if not np.isfinite(pooled) or pooled == 0:
            continue
        try:
            from sklearn.metrics import roc_auc_score

            filled = values.fillna(values.median())
            auc = float(roc_auc_score(target, filled))
            directional_auc = max(auc, 1.0 - auc)
        except Exception:
            directional_auc = np.nan
        rows.append(
            {
                "feature": feature,
                "hit_mean": float(hit_values.mean()),
                "opposite_mean": float(opp_values.mean()),
                "hit_median": float(hit_values.median()),
                "opposite_median": float(opp_values.median()),
                "std_mean_diff": float((hit_values.mean() - opp_values.mean()) / pooled),
                "directional_auc": directional_auc,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("directional_auc", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def save_strict_model(model: StrictHitSplitModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    write_manifest(
        {
            "model_id": model.model_id,
            "split": asdict(model.split),
            "config": asdict(model.config),
            "algorithm": model.algorithm,
            "feature_count": len(model.feature_columns),
            "feature_columns": model.feature_columns,
            "long_train_rows": model.long_bundle.train_rows,
            "long_pair_train_rows": model.long_bundle.pair_train_rows,
            "short_train_rows": model.short_bundle.train_rows,
            "short_pair_train_rows": model.short_bundle.pair_train_rows,
            "probability_output": "raw_lightgbm_predict_proba",
            "calibration": "none",
        },
        output_dir / "manifest.json",
    )


def run_strict_hit_model(config: StrictHitModelConfig) -> Path:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is required for run-strict-hit-model")

    run_dir = RUNS_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, feature_columns = prepare_strict_dataset(config)
    (run_dir / "feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")
    write_strict_feature_separation(
        dataset,
        feature_columns,
        run_dir / "feature_research" / "strict_hit_vs_opposite.csv",
    )

    selected_frames: list[pd.DataFrame] = []
    for split in DEFAULT_SPLITS:
        model = train_strict_hit_split_model(dataset, feature_columns, split, config)
        save_strict_model(model, run_dir / "models" / split.name)

        period = dataset[between_dates(dataset, split.prediction_start, split.prediction_end)].copy()
        predictions = predict_strict_hit_model(model, period)
        selected = select_strict_weekly_signals(predictions, config)
        selected["split"] = split.name

        split_dir = run_dir / "model_predictions" / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(split_dir / "strict_predictions.parquet", index=False)
        selected.to_parquet(split_dir / "signals.parquet", index=False)
        selected[selected["selected"]].to_csv(split_dir / "selected_signals.csv", index=False)
        write_report_tables(build_strict_hit_report(selected), split_dir / "strict_report")
        write_report_tables(build_gold_report(selected), split_dir / "gold_report")
        _write_tables(build_strict_capture_report(selected, period), split_dir / "capture_report")
        selected_frames.append(selected)

    combined = pd.concat(selected_frames, ignore_index=True)
    combined.to_parquet(run_dir / "all_signals.parquet", index=False)
    combined[combined["selected"]].to_csv(run_dir / "selected_signals.csv", index=False)
    write_report_tables(build_strict_hit_report(combined), run_dir / "combined_strict_report")
    write_report_tables(build_gold_report(combined), run_dir / "combined_gold_report")

    combined_period = dataset[
        between_dates(dataset, DEFAULT_SPLITS[0].prediction_start, DEFAULT_SPLITS[-1].prediction_end)
    ].copy()
    _write_tables(build_strict_capture_report(combined, combined_period), run_dir / "combined_capture_report")
    write_manifest(
        {
            "run_id": config.run_id,
            "source_run_id": config.source_run_id,
            "source_run_dir": str(RUNS_DIR / config.source_run_id),
            "config": asdict(config),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "splits": [asdict(split) for split in DEFAULT_SPLITS],
            "notes": [
                "Strict-hit model branch: target touched and 5-day signed close > 80% target.",
                "Opposite is not strict hit and signed close < +1%.",
                "Range-bound is every evaluated row outside strict hit/opposite.",
                "Locked v19 artifacts are read only as the stable data contract.",
            ],
        },
        run_dir / "manifest.json",
    )
    return run_dir


def clone_config_with_selector(
    config: StrictHitModelConfig,
    **kwargs: Any,
) -> StrictHitModelConfig:
    return replace(config, **kwargs)
