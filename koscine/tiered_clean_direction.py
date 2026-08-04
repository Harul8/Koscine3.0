from __future__ import annotations

import json
import gc
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from koscine.clean_direction import (
    add_label_validity_flags,
    add_liquid30_features,
    add_sector_features,
    liquid30_symbols,
    normalize_symbol,
)
from koscine.config import (
    HORIZON_DAYS,
    MODEL_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    TARGET_UNIVERSE,
)
from koscine.training import feature_columns


PROD_TIERED_ROOT = MODEL_DIR / "prod"
ARBITER_MODEL_FILE = "trade_arbiter_v1_lgbm.txt"
ARBITER_META_FILE = "trade_arbiter_v1.json"
ARBITER_CLASS_TO_LABEL = {0: "none", 1: "up", 2: "down"}
ARBITER_LABEL_TO_CLASS = {v: k for k, v in ARBITER_CLASS_TO_LABEL.items()}
ARBITER_FEATURES = [
    "up_score",
    "down_score",
    "score_gap",
    "abs_score_gap",
    "max_side_score",
    "min_side_score",
    "up_model_score",
    "down_model_score",
    "up_lgbm_score",
    "down_lgbm_score",
    "up_catboost_score",
    "down_catboost_score",
    "up_rule_gate_strength",
    "down_rule_gate_strength",
    "up_rule_gate_pass",
    "down_rule_gate_pass",
    "threshold",
    "tier_liquid30",
    "tier_rest35",
]


def _trim_process_memory() -> None:
    gc.collect()
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessWorkingSetSize.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            kernel32.SetProcessWorkingSetSize(
                kernel32.GetCurrentProcess(),
                ctypes.c_size_t(-1).value,
                ctypes.c_size_t(-1).value,
            )
        except Exception:
            pass


@dataclass(frozen=True)
class TierSpec:
    name: str
    side: str
    threshold: float
    predict_symbols: tuple[str, ...]
    train_liquid_weight: float
    model_id: str


@dataclass(frozen=True)
class TieredCleanConfig:
    train_start_year: int = 2012
    start_test_year: int = 2018
    end_test_year: int = 2025
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    adverse_limit: float = 0.0091
    max_weekly_abs_move: float = 0.50
    min_validation_calls: int = 20
    bad_rate_cap: float = 0.15
    topn_step: int = 5
    topn_max: int = 300
    lgbm_weight: float = 0.60
    use_catboost: bool = True
    train_all_symbols: bool = True
    use_vol_adjusted_labels: bool = False
    temporal_decay_per_year: float = 0.0
    n_seeds: int = 1
    use_calibration: bool = False


@dataclass(frozen=True)
class TieredProdConfig:
    train_start_year: int = 2012
    train_cutoff_day: int = 20
    adverse_limit: float = 0.0091
    max_weekly_abs_move: float = 0.50
    lgbm_weight: float = 0.60
    use_catboost: bool = True
    min_score: float = 0.0
    train_all_symbols: bool = True
    use_vol_adjusted_labels: bool = False
    temporal_decay_per_year: float = 0.0
    n_seeds: int = 1
    use_calibration: bool = False


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_seed(value: str, base: int = 0) -> int:
    return base + sum((idx + 1) * ord(char) for idx, char in enumerate(value)) % 10000


def rest35_symbols() -> list[str]:
    liquid = set(liquid30_symbols())
    out: list[str] = []
    for symbol in TARGET_UNIVERSE:
        canonical = normalize_symbol(symbol)
        if canonical in liquid or canonical in out:
            continue
        out.append(canonical)
        if len(out) == 35:
            break
    return out


def tier_specs() -> list[TierSpec]:
    liquid = tuple(liquid30_symbols())
    rest = tuple(rest35_symbols())
    return [
        TierSpec("liquid30", "up", 0.04, liquid, 2.5, "liquid30_up_4pct_5d"),
        TierSpec("liquid30", "down", 0.04, liquid, 2.5, "liquid30_down_4pct_5d"),
        TierSpec("rest35", "up", 0.07, rest, 1.0, "rest35_up_7pct_5d"),
        TierSpec("rest35", "down", 0.07, rest, 1.0, "rest35_down_7pct_5d"),
    ]


def label_col(spec: TierSpec) -> str:
    pct = int(round(spec.threshold * 100))
    return f"label_{spec.name}_{spec.side}_{pct}pct_clean_{HORIZON_DAYS}d"


def opposite_label_col(spec: TierSpec) -> str:
    side = "down" if spec.side == "up" else "up"
    pct = int(round(spec.threshold * 100))
    return f"label_{spec.name}_{side}_{pct}pct_clean_{HORIZON_DAYS}d"


def _future_known_mask(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.Series:
    future_col = f"future_{HORIZON_DAYS}d_date"
    if future_col not in df:
        return pd.Series(True, index=df.index)
    return pd.to_datetime(df[future_col]).le(train_end)


def _rolling_pctile(grouped: pd.core.groupby.generic.SeriesGroupBy, window: int, min_periods: int):
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).rank(pct=True))


def add_tiered_research_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    close = out["close"]

    for window in (100, 200):
        ma = grouped["close"].transform(lambda s, w=window: s.rolling(w, min_periods=w // 2).mean())
        out[f"close_sma{window}_dist"] = close / ma - 1.0

    if "realized_vol_20" in out:
        out["hv_20"] = out["realized_vol_20"] * np.sqrt(252.0)
    if "atm_iv" in out and "hv_20" in out:
        out["iv_vs_hv"] = out["atm_iv"] - out["hv_20"]

    if "high" in out and "low" in out:
        for window in (5, 10, 20):
            high = grouped["high"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).max())
            low = grouped["low"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).min())
            out[f"path_range_{window}d"] = high / low - 1.0
        out["w_range_pct"] = out["path_range_5d"]
        out["tight_range_10d"] = out["path_range_10d"]

    if "fut_oi" in out:
        fut_mean = grouped["fut_oi"].transform(lambda s: s.rolling(60, min_periods=25).mean())
        fut_std = grouped["fut_oi"].transform(lambda s: s.rolling(60, min_periods=25).std())
        out["fut_oi_z_60d"] = (out["fut_oi"] - fut_mean) / fut_std.replace(0, np.nan)

    if "put_wall_1_dist" in out:
        out["dist_put_wall"] = out["put_wall_1_dist"]
    if "call_wall_1_dist" in out:
        out["dist_call_wall"] = out["call_wall_1_dist"]

    prev_close = grouped["close"].shift(1)
    gap = out["open"] / prev_close - 1.0
    out["gap_up"] = gap.gt(0.005).astype(float)
    out["gap_down"] = gap.lt(-0.005).astype(float)
    out["gap_up_count_20d"] = grouped["gap_up"].transform(lambda s: s.rolling(20, min_periods=5).sum())
    out["gap_down_count_20d"] = grouped["gap_down"].transform(lambda s: s.rolling(20, min_periods=5).sum())
    out["gap_both_count_20d"] = out["gap_up_count_20d"] + out["gap_down_count_20d"]

    for col in (
        "atm_iv",
        "atm_ce_iv",
        "atm_pe_iv",
        "pcr_vol",
        "pcr_oi",
        "fut_oi",
        "fut_oi_chg_5",
        "fut_chg_oi_chg_5",
        "iv_vs_hv",
        "bb_width_20",
        "tight_range_10d",
        "delivery_pct",
        "ret_20d",
        "hv_20",
    ):
        if col in out:
            out[f"{col}_rank_252d"] = _rolling_pctile(grouped[col], 252, 80)

    for col in (
        "atm_iv",
        "pcr_vol",
        "fut_oi_z_60d",
        "fut_oi_chg_5",
        "iv_vs_hv",
        "bb_width_20",
        "delivery_pct",
        "ret_20d",
        "close_sma50_dist",
        "close_sma200_dist",
        "dist_call_wall",
    ):
        if col in out:
            out[f"{col}_cs_rank"] = out.groupby("date")[col].rank(pct=True)

    out["above_50ma"] = out.get("close_sma50_dist", pd.Series(np.nan, index=out.index)).gt(0).astype(float)
    out["above_200ma"] = out.get("close_sma200_dist", pd.Series(np.nan, index=out.index)).gt(0).astype(float)
    return out


def add_tiered_clean_labels(
    df: pd.DataFrame,
    specs: list[TierSpec],
    adverse_limit: float,
    max_weekly_abs_move: float,
    use_vol_adjusted: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    up = out[f"up_move_{HORIZON_DAYS}d"]
    down = out[f"down_move_{HORIZON_DAYS}d"]
    fwd_close = out[f"fwd_return_{HORIZON_DAYS}d"]
    invalid_price = out.get("future_invalid_price_event_5d", pd.Series(False, index=out.index))
    valid = (
        up.notna()
        & down.notna()
        & out["entry_1d_open"].notna()
        & up.le(max_weekly_abs_move)
        & down.le(max_weekly_abs_move)
        & fwd_close.abs().le(max_weekly_abs_move)
        & ~invalid_price.astype(bool)
    )
    if use_vol_adjusted and "vol_adj_adverse_limit" in out:
        bull_limit = out["vol_adj_adverse_limit"]
        bear_limit = out["vol_adj_adverse_limit"]
    else:
        bull_limit = pd.Series(adverse_limit, index=out.index)
        bear_limit = pd.Series(adverse_limit, index=out.index)
    thresholds = sorted({spec.threshold for spec in specs})
    for threshold in thresholds:
        pct = int(round(threshold * 100))
        bull = np.where(valid, (up > threshold) & (down < bull_limit), np.nan)
        bear = np.where(valid, (down > threshold) & (up < bear_limit), np.nan)
        for tier in sorted({spec.name for spec in specs}):
            out[f"label_{tier}_up_{pct}pct_clean_{HORIZON_DAYS}d"] = bull
            out[f"label_{tier}_down_{pct}pct_clean_{HORIZON_DAYS}d"] = bear
    return out


def prepare_tiered_frame(
    dataset_path: Path,
    config: TieredCleanConfig | TieredProdConfig,
    specs: list[TierSpec] | None = None,
) -> tuple[pd.DataFrame, list[str], list[TierSpec]]:
    specs = specs or tier_specs()
    liquid = liquid30_symbols()
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if f"future_{HORIZON_DAYS}d_date" in df:
        df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    df = add_liquid30_features(df, liquid)
    df = add_sector_features(df, liquid)
    df = add_tiered_research_features(df)
    df = add_label_validity_flags(df, config.max_weekly_abs_move)
    df = add_tiered_clean_labels(
        df,
        specs,
        config.adverse_limit,
        config.max_weekly_abs_move,
        use_vol_adjusted=getattr(config, "use_vol_adjusted_labels", False),
    )
    df["is_liquid30"] = df["symbol"].isin(liquid)
    df["is_rest35"] = df["symbol"].isin(rest35_symbols())
    df["tier_name"] = np.where(df["is_liquid30"], "liquid30", np.where(df["is_rest35"], "rest35", "other"))

    features = feature_columns(df)
    blocked = {
        "price_discontinuity",
        "corp_price_action",
        "invalid_price_event",
        "future_invalid_price_event_5d",
    }
    features = [col for col in features if col not in blocked]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    # Downcast float64 feature columns to float32 — halves memory of the
    # ~1.1M-row x ~320-col frame (2.7 GiB -> 1.35 GiB) so the per-spec
    # _train_frame .copy() does not blow up in the long-running API process.
    # LightGBM/CatBoost accept float32 without accuracy loss.
    float64_cols = [c for c in features if df[c].dtype == "float64"]
    if float64_cols:
        df[float64_cols] = df[float64_cols].astype("float32")
    return df, features, specs


def _model_params(y: pd.Series, seed: int, spec: TierSpec) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31 if spec.name == "liquid30" else 47,
        "min_data_in_leaf": 350 if spec.name == "liquid30" else 250,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 18.0 if spec.name == "liquid30" else 12.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }


def _fit_lgbm(
    inner: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    spec: TierSpec,
    weight_col: str,
    seed: int,
) -> lgb.Booster:
    col = label_col(spec)
    train_set = lgb.Dataset(inner[features], label=inner[col].astype(int), weight=inner[weight_col])
    valid_set = lgb.Dataset(valid[features], label=valid[col].astype(int), weight=valid[weight_col], reference=train_set)
    return lgb.train(
        _model_params(inner[col].astype(int), seed, spec),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def _fit_catboost(
    inner: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    spec: TierSpec,
    weight_col: str,
    seed: int,
) -> CatBoostClassifier:
    col = label_col(spec)
    y = inner[col].astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=450,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=16 if spec.name == "liquid30" else 12,
        random_strength=1.5,
        bagging_temperature=1.0,
        class_weights=[1.0, negatives / max(positives, 1)],
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        random_seed=seed,
        verbose=False,
    )
    model.fit(
        Pool(inner[features], label=inner[col].astype(int), weight=inner[weight_col]),
        eval_set=Pool(valid[features], label=valid[col].astype(int), weight=valid[weight_col]),
        use_best_model=True,
    )
    return model


def _score_lgbm(model: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(frame[features], num_iteration=model.best_iteration)


def _score_catboost(model: CatBoostClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def _safe_quantile(frame: pd.DataFrame, col: str, q: float, fallback: float) -> float:
    if col not in frame:
        return fallback
    value = frame[col].replace([np.inf, -np.inf], np.nan).quantile(q)
    if pd.isna(value):
        return fallback
    return float(value)


def rule_gate(frame: pd.DataFrame, train: pd.DataFrame, spec: TierSpec) -> pd.DataFrame:
    out = frame.copy()
    q = {
        "iv70": _safe_quantile(train, "atm_iv", 0.70, np.inf),
        "iv80": _safe_quantile(train, "atm_iv", 0.80, np.inf),
        "pcr70": _safe_quantile(train, "pcr_vol", 0.70, np.inf),
        "ret30": _safe_quantile(train, "ret_20d", 0.30, -np.inf),
        "ret40": _safe_quantile(train, "ret_20d", 0.40, -np.inf),
        "ret50": _safe_quantile(train, "ret_20d", 0.50, 0.0),
        "fut70": _safe_quantile(train, "fut_oi_z_60d", 0.70, np.inf),
        "bb70": _safe_quantile(train, "bb_width_20", 0.70, np.inf),
        "del50": _safe_quantile(train, "delivery_pct", 0.50, np.inf),
        "sma50med": _safe_quantile(train, "close_sma50_dist", 0.50, 0.0),
    }
    iv_rank = out.get("atm_iv_rank_252d", pd.Series(np.nan, index=out.index))
    iv_high = out.get("atm_iv", pd.Series(np.nan, index=out.index)).ge(q["iv70"]) | iv_rank.ge(0.70)
    iv_very_high = out.get("atm_iv", pd.Series(np.nan, index=out.index)).ge(q["iv80"]) | iv_rank.ge(0.80)
    compression = (
        out.get("bb_width_20", pd.Series(np.nan, index=out.index)).ge(q["bb70"])
        | out.get("bb_width_20_rank_252d", pd.Series(np.nan, index=out.index)).ge(0.70)
        | out.get("tight_range_10d_rank_252d", pd.Series(np.nan, index=out.index)).ge(0.70)
    )
    low_delivery = out.get("delivery_pct", pd.Series(np.nan, index=out.index)).le(q["del50"])
    pcr_high = (
        out.get("pcr_vol", pd.Series(np.nan, index=out.index)).ge(q["pcr70"])
        | out.get("pcr_vol_rank_252d", pd.Series(np.nan, index=out.index)).ge(0.70)
    )
    above50 = out.get("close_sma50_dist", pd.Series(np.nan, index=out.index)).gt(q["sma50med"])
    weak20 = out.get("ret_20d", pd.Series(np.nan, index=out.index)).le(q["ret30"])
    below200 = out.get("close_sma200_dist", pd.Series(np.nan, index=out.index)).lt(0)
    fut_trap = (
        out.get("fut_oi_z_60d", pd.Series(np.nan, index=out.index)).ge(q["fut70"])
        | out.get("fut_oi_z_60d_cs_rank", pd.Series(np.nan, index=out.index)).ge(0.70)
    )
    iv_vs_hv_high = out.get("iv_vs_hv_rank_252d", pd.Series(np.nan, index=out.index)).ge(0.70)
    call_room = out.get("dist_call_wall", pd.Series(np.nan, index=out.index)).gt(0)

    if spec.name == "liquid30" and spec.side == "up":
        components = [iv_high, compression, low_delivery, pcr_high | above50 | call_room]
        rule_name = "liquid_iv_compression_lowdelivery_bull_bias"
        required = 3
    elif spec.name == "liquid30" and spec.side == "down":
        components = [iv_high, compression | low_delivery, weak20 | below200, iv_very_high | fut_trap]
        rule_name = "liquid_iv_weaktrend_breakdown"
        required = 3
    elif spec.name == "rest35" and spec.side == "up":
        components = [iv_high, pcr_high, above50 | call_room, compression]
        rule_name = "rest_iv_pcr_above50_bull"
        required = 3
    else:
        components = [iv_high, fut_trap, weak20 | below200 | iv_vs_hv_high, compression | low_delivery]
        rule_name = "rest_iv_longtrap_bear"
        required = 3

    score = sum(component.fillna(False).astype(float) for component in components)
    out["rule_name"] = rule_name
    out["rule_component_count"] = score
    out["rule_gate_pass"] = score.ge(required)
    out["rule_gate_strength"] = score / len(components)
    return out


def _sample_weights(
    train: pd.DataFrame,
    spec: TierSpec,
    temporal_decay_per_year: float = 0.0,
) -> pd.Series:
    weights = pd.Series(1.0, index=train.index)
    is_target_tier = train["symbol"].isin(spec.predict_symbols)
    weights.loc[is_target_tier] *= spec.train_liquid_weight if spec.name == "liquid30" else 2.0
    if spec.name == "liquid30":
        weights.loc[~is_target_tier] *= 0.60
    else:
        weights.loc[train["is_liquid30"]] *= 0.45
    actual = train[label_col(spec)].astype(bool)
    opposite = train[opposite_label_col(spec)].astype(bool)
    weights.loc[actual] *= 3.0
    weights.loc[opposite] *= 1.8
    if temporal_decay_per_year and temporal_decay_per_year > 0:
        dates = pd.to_datetime(train["date"])
        ref_date = dates.max()
        years_ago = (ref_date - dates).dt.days / 365.25
        decay_arr = np.power(1.0 - float(temporal_decay_per_year), years_ago.clip(lower=0).values)
        decay = pd.Series(decay_arr, index=train.index).clip(lower=0.20)
        weights = weights * decay
    return weights


def _train_frame(
    df: pd.DataFrame,
    spec: TierSpec,
    train_start_year: int,
    train_end: pd.Timestamp,
    train_all_symbols: bool,
    temporal_decay_per_year: float = 0.0,
) -> pd.DataFrame:
    frame = df[
        df["date"].dt.year.ge(train_start_year)
        & df["date"].le(train_end)
        & _future_known_mask(df, train_end)
    ].dropna(subset=[label_col(spec), opposite_label_col(spec)]).copy()
    if not train_all_symbols:
        frame = frame[frame["symbol"].isin(spec.predict_symbols)].copy()
    frame["sample_weight"] = _sample_weights(
        frame, spec, temporal_decay_per_year=temporal_decay_per_year
    )
    return frame


def _prediction_frame(df: pd.DataFrame, spec: TierSpec, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[
        df["date"].between(start, end)
        & df["symbol"].isin(spec.predict_symbols)
    ].dropna(subset=[label_col(spec), opposite_label_col(spec)]).copy()


def _fit_spec_models(
    train: pd.DataFrame,
    features: list[str],
    spec: TierSpec,
    validation_days: int,
    use_catboost: bool,
    seed: int,
    n_seeds: int = 1,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    if inner.empty or valid.empty:
        raise ValueError(f"Empty inner/valid split for {spec.model_id}")
    n_seeds = max(1, int(n_seeds))
    if n_seeds == 1:
        models: dict[str, object] = {"lgbm": _fit_lgbm(inner, valid, features, spec, "sample_weight", seed)}
        if use_catboost:
            models["catboost"] = _fit_catboost(inner, valid, features, spec, "sample_weight", seed + 1000)
        return models, inner, valid
    lgbm_models = [
        _fit_lgbm(inner, valid, features, spec, "sample_weight", seed + 7 * i)
        for i in range(n_seeds)
    ]
    models = {"lgbm_ensemble": lgbm_models}
    if use_catboost:
        cb_models = [
            _fit_catboost(inner, valid, features, spec, "sample_weight", seed + 1000 + 11 * i)
            for i in range(n_seeds)
        ]
        models["catboost_ensemble"] = cb_models
    return models, inner, valid


def _score_spec(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    spec: TierSpec,
    models: dict[str, object],
    lgbm_weight: float,
    calibrator: "CalibratorBundle | None" = None,
) -> pd.DataFrame:
    out = rule_gate(frame, train[train["symbol"].isin(spec.predict_symbols)], spec)
    if "lgbm_ensemble" in models:
        lgbm_preds = np.mean(
            [_score_lgbm(m, out, features) for m in models["lgbm_ensemble"]],  # type: ignore[arg-type]
            axis=0,
        )
        out["lgbm_score"] = lgbm_preds
    else:
        out["lgbm_score"] = _score_lgbm(models["lgbm"], out, features)  # type: ignore[arg-type]
    if "catboost_ensemble" in models:
        cb_preds = np.mean(
            [_score_catboost(m, out, features) for m in models["catboost_ensemble"]],  # type: ignore[arg-type]
            axis=0,
        )
        out["catboost_score"] = cb_preds
        out["model_score"] = lgbm_weight * out["lgbm_score"] + (1.0 - lgbm_weight) * out["catboost_score"]
    elif "catboost" in models:
        out["catboost_score"] = _score_catboost(models["catboost"], out, features)  # type: ignore[arg-type]
        out["model_score"] = lgbm_weight * out["lgbm_score"] + (1.0 - lgbm_weight) * out["catboost_score"]
    else:
        out["catboost_score"] = np.nan
        out["model_score"] = out["lgbm_score"]
    out["raw_score"] = out["model_score"] * (0.85 + 0.30 * out["rule_gate_strength"])
    if calibrator is not None:
        try:
            out["calibrated_score"] = calibrator.predict(out["raw_score"].values)
            out["score"] = out["calibrated_score"]
        except Exception:
            out["calibrated_score"] = np.nan
            out["score"] = out["raw_score"]
    else:
        out["calibrated_score"] = np.nan
        out["score"] = out["raw_score"]
    out["score"] = out["score"].clip(lower=0.0, upper=1.0)
    out["side"] = spec.side
    out["tier"] = spec.name
    out["threshold"] = spec.threshold
    out["model_id"] = spec.model_id
    out["actual_hit"] = out[label_col(spec)].astype(bool)
    out["actual_opposite"] = out[opposite_label_col(spec)].astype(bool)
    side_move = out[f"{'up' if spec.side == 'up' else 'down'}_move_{HORIZON_DAYS}d"]
    out["actual_move"] = side_move
    sign = 1.0 if spec.side == "up" else -1.0
    out["signed_close_return_5d"] = sign * out[f"fwd_return_{HORIZON_DAYS}d"]
    return out


def _scored_keep_cols() -> list[str]:
    return [
        "date",
        "symbol",
        "tier",
        "side",
        "threshold",
        "score",
        "raw_score",
        "calibrated_score",
        "model_score",
        "lgbm_score",
        "catboost_score",
        "model_id",
        "rule_name",
        "rule_gate_pass",
        "rule_gate_strength",
        "rule_component_count",
        "close",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_high",
        f"future_{HORIZON_DAYS}d_low",
        f"future_{HORIZON_DAYS}d_close",
        f"future_{HORIZON_DAYS}d_date",
        f"up_move_{HORIZON_DAYS}d",
        f"down_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        "atr_pct_14",
        "atm_iv",
        "actual_move",
        "actual_hit",
        "actual_opposite",
        "signed_close_return_5d",
    ]


def _pair_for_arbiter(scored: pd.DataFrame, include_labels: bool) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    pair_keys = ["date", "symbol", "tier", "threshold"]
    pair_keys.extend([col for col in ("prediction_year", "train_end", "split") if col in scored.columns])
    base_cols = [
        "date",
        "symbol",
        "tier",
        "threshold",
        "score",
        "model_score",
        "lgbm_score",
        "catboost_score",
        "rule_gate_pass",
        "rule_gate_strength",
    ]
    label_cols = [
        f"up_move_{HORIZON_DAYS}d",
        f"down_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
    ]
    use_cols = list(dict.fromkeys(col for col in base_cols + label_cols + ["side"] + pair_keys if col in scored.columns))
    rows = scored[use_cols].copy()
    sides = []
    for side in ("up", "down"):
        part = rows[rows["side"].eq(side)].drop_duplicates(pair_keys, keep="last").copy()
        rename = {
            "score": f"{side}_score",
            "model_score": f"{side}_model_score",
            "lgbm_score": f"{side}_lgbm_score",
            "catboost_score": f"{side}_catboost_score",
            "rule_gate_pass": f"{side}_rule_gate_pass",
            "rule_gate_strength": f"{side}_rule_gate_strength",
        }
        part = part.rename(columns=rename)
        keep = pair_keys + list(rename.values())
        if include_labels:
            keep.extend([col for col in label_cols if col in part.columns])
        sides.append(part[[col for col in keep if col in part.columns]])
    if len(sides) != 2:
        return pd.DataFrame()
    paired = sides[0].merge(sides[1], on=pair_keys, how="inner", suffixes=("", "_downrow"))
    for col in label_cols:
        down_col = f"{col}_downrow"
        if down_col in paired.columns:
            if col not in paired:
                paired[col] = paired[down_col]
            paired = paired.drop(columns=down_col)
    if paired.empty:
        return paired
    paired["score_gap"] = paired["up_score"] - paired["down_score"]
    paired["abs_score_gap"] = paired["score_gap"].abs()
    paired["max_side_score"] = paired[["up_score", "down_score"]].max(axis=1)
    paired["min_side_score"] = paired[["up_score", "down_score"]].min(axis=1)
    paired["up_rule_gate_pass"] = paired["up_rule_gate_pass"].fillna(False).astype(float)
    paired["down_rule_gate_pass"] = paired["down_rule_gate_pass"].fillna(False).astype(float)
    paired["tier_liquid30"] = paired["tier"].eq("liquid30").astype(float)
    paired["tier_rest35"] = paired["tier"].eq("rest35").astype(float)
    for feature in ARBITER_FEATURES:
        if feature not in paired:
            paired[feature] = np.nan
    return paired


def _label_arbiter_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    out = paired.copy()
    up_move = out[f"up_move_{HORIZON_DAYS}d"].astype(float)
    down_move = out[f"down_move_{HORIZON_DAYS}d"].astype(float)
    close_ret = out[f"fwd_return_{HORIZON_DAYS}d"].astype(float)
    threshold = out["threshold"].astype(float)
    dominance_buffer = 0.80 * threshold
    long_clean = up_move.ge(threshold) & down_move.lt(dominance_buffer) & close_ret.ge(0)
    short_clean = down_move.ge(threshold) & up_move.lt(dominance_buffer) & close_ret.le(0)
    out["arbiter_label"] = np.select(
        [long_clean, short_clean],
        ["up", "down"],
        default="none",
    )
    out["arbiter_y"] = out["arbiter_label"].map(ARBITER_LABEL_TO_CLASS).astype(int)
    return out


def _train_trade_arbiter(
    validation_scored: pd.DataFrame,
    run_dir: Path,
) -> dict | None:
    paired = _label_arbiter_pairs(_pair_for_arbiter(validation_scored, include_labels=True))
    paired = paired.dropna(subset=["arbiter_y"]).copy()
    if paired.empty or paired["arbiter_y"].nunique() < 2:
        return None

    paired = paired.sort_values("date").reset_index(drop=True)
    unique_dates = pd.Series(pd.to_datetime(paired["date"]).drop_duplicates().sort_values().to_numpy())
    if len(unique_dates) >= 20:
        valid_start = unique_dates.iloc[max(1, int(len(unique_dates) * 0.70))]
        train_part = paired[pd.to_datetime(paired["date"]) < valid_start].copy()
        valid_part = paired[pd.to_datetime(paired["date"]) >= valid_start].copy()
    else:
        train_part = paired.sample(frac=0.75, random_state=stable_seed("trade_arbiter", 77))
        valid_part = paired.drop(train_part.index).copy()
    if train_part.empty or valid_part.empty:
        train_part = paired.copy()
        valid_part = paired.copy()

    y_train = train_part["arbiter_y"].astype(int)
    class_counts = y_train.value_counts().to_dict()
    weights = y_train.map(lambda cls: len(y_train) / (3.0 * max(class_counts.get(cls, 1), 1))).astype(float)
    train_set = lgb.Dataset(train_part[ARBITER_FEATURES], label=y_train, weight=weights)
    valid_set = lgb.Dataset(valid_part[ARBITER_FEATURES], label=valid_part["arbiter_y"].astype(int), reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 8.0,
        "verbosity": -1,
        "seed": stable_seed("trade_arbiter", 900),
    }
    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=800,
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    probs = model.predict(valid_part[ARBITER_FEATURES], num_iteration=model.best_iteration)
    pred = probs.argmax(axis=1)
    metrics = {
        "train_rows": int(len(train_part)),
        "valid_rows": int(len(valid_part)),
        "total_rows": int(len(paired)),
        "accuracy": float(accuracy_score(valid_part["arbiter_y"].astype(int), pred)),
        "label_counts": {str(ARBITER_CLASS_TO_LABEL[int(k)]): int(v) for k, v in paired["arbiter_y"].value_counts().items()},
    }
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / ARBITER_MODEL_FILE
    model.save_model(str(model_path))
    meta = {
        "model_stack": "trade_arbiter_v1_lgbm_multiclass",
        "model_file": str(Path("models") / ARBITER_MODEL_FILE),
        "features": ARBITER_FEATURES,
        "class_to_label": ARBITER_CLASS_TO_LABEL,
        "metrics": metrics,
    }
    (run_dir / ARBITER_META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    paired.to_parquet(run_dir / "trade_arbiter_training_pairs.parquet", index=False)
    return meta


def model_metrics(frame: pd.DataFrame, spec: TierSpec) -> dict:
    if frame.empty:
        return {"rows": 0, "positives": 0, "positive_rate": np.nan}
    y = frame["actual_hit"].astype(int)
    p = frame["score"].astype(float)
    out = {
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan
    except ValueError:
        out["roc_auc"] = np.nan
    ranked = frame.sort_values("score", ascending=False)
    for k in (25, 50, 100, 150, 200):
        top = ranked.head(k)
        if len(top):
            out[f"precision_at_{k}"] = float(top["actual_hit"].mean())
            out[f"opposite_at_{k}"] = float(top["actual_opposite"].mean())
            out[f"avg_move_at_{k}"] = float(top["actual_move"].mean())
    return out


def selected_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "calls": 0,
            "hit_rate": np.nan,
            "opposite_rate": np.nan,
            "avg_move": np.nan,
            "avg_signed_close_return": np.nan,
            "gate_pass_rate": np.nan,
        }
    return {
        "calls": int(len(frame)),
        "hit_rate": float(frame["actual_hit"].mean()),
        "opposite_rate": float(frame["actual_opposite"].mean()),
        "avg_move": float(frame["actual_move"].mean()),
        "avg_signed_close_return": float(frame["signed_close_return_5d"].mean()),
        "gate_pass_rate": float(frame["rule_gate_pass"].mean()),
        "up_calls": int(frame["side"].eq("up").sum()),
        "down_calls": int(frame["side"].eq("down").sum()),
    }


def choose_rule(validation: pd.DataFrame, config: TieredCleanConfig) -> tuple[dict, pd.DataFrame]:
    rows = []
    if validation.empty:
        rule = {"min_score": 1.0, "require_gate": True, "daily_top": 1}
        return rule, pd.DataFrame()
    scores = validation["score"].dropna()
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        min_score = float(scores.quantile(q))
        for require_gate in (True, False):
            for daily_top in (1, 2, 3, 5, 10, 20):
                selected = validation[validation["score"].ge(min_score)].copy()
                if require_gate:
                    selected = selected[selected["rule_gate_pass"]]
                selected = (
                    selected.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
                    .groupby("date")
                    .head(daily_top)
                )
                rows.append(
                    {
                        "min_score": min_score,
                        "score_quantile": q,
                        "require_gate": require_gate,
                        "daily_top": daily_top,
                        **selected_metrics(selected),
                    }
                )
    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["calls"] >= config.min_validation_calls)
        & (grid["opposite_rate"].fillna(1.0) <= config.bad_rate_cap)
    ].copy()
    if eligible.empty:
        eligible = grid[grid["calls"] >= max(5, config.min_validation_calls // 2)].copy()
    if eligible.empty:
        eligible = grid.copy()
    chosen = eligible.sort_values(
        ["hit_rate", "avg_move", "calls", "opposite_rate"],
        ascending=[False, False, False, True],
    ).iloc[0]
    rule = {
        "min_score": float(chosen["min_score"]),
        "require_gate": bool(chosen["require_gate"]),
        "daily_top": int(chosen["daily_top"]),
    }
    return rule, grid


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    selected = frame[frame["score"].ge(rule["min_score"])].copy()
    if rule.get("require_gate", False):
        selected = selected[selected["rule_gate_pass"]]
    if selected.empty:
        return selected
    selected = (
        selected.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
        .groupby("date")
        .head(int(rule["daily_top"]))
        .sort_values(["date", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    selected["selection_rule"] = (
        f"score>={rule['min_score']:.4f};"
        f"gate={bool(rule.get('require_gate', False))};"
        f"daily_top={int(rule['daily_top'])}"
    )
    return selected


def _save_models(
    run_dir: Path,
    model_prefix: str,
    models: dict[str, object],
    features: list[str],
    spec: TierSpec,
    train: pd.DataFrame,
    train_end: pd.Timestamp,
) -> list[dict]:
    saved = []
    for name, model in models.items():
        if name.endswith("_ensemble") and isinstance(model, list):
            files = []
            for i, m in enumerate(model):
                model_file = f"{model_prefix}_{name}_seed{i}"
                if isinstance(m, lgb.Booster):
                    path = run_dir / "models" / f"{model_file}.txt"
                    m.save_model(path)
                    best_iteration = m.best_iteration
                else:
                    path = run_dir / "models" / f"{model_file}.cbm"
                    m.save_model(str(path))
                    best_iteration = m.get_best_iteration()
                files.append({"model_file": path.name, "best_iteration": best_iteration})
            saved.append({"family": name, "files": files, "n_seeds": len(model)})
            continue
        model_file = f"{model_prefix}_{name}"
        if isinstance(model, lgb.Booster):
            path = run_dir / "models" / f"{model_file}.txt"
            model.save_model(path)
            best_iteration = model.best_iteration
        else:
            path = run_dir / "models" / f"{model_file}.cbm"
            model.save_model(str(path))
            best_iteration = model.get_best_iteration()
        saved.append(
            {
                "family": name,
                "model_file": path.name,
                "best_iteration": best_iteration,
            }
        )
    metadata = {
        "model_id": spec.model_id,
        "tier": spec.name,
        "side": spec.side,
        "threshold": spec.threshold,
        "label_col": label_col(spec),
        "opposite_label_col": opposite_label_col(spec),
        "train_end": str(train_end.date()),
        "train_rows": len(train),
        "features": features,
        "models": saved,
    }
    (run_dir / "models" / f"{model_prefix}.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return [metadata]


def train_score_one_spec(
    df: pd.DataFrame,
    features: list[str],
    spec: TierSpec,
    year: int,
    config: TieredCleanConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    train_end = pd.Timestamp(year=year - 1, month=config.train_cutoff_month, day=config.train_cutoff_day)
    train = _train_frame(
        df, spec, config.train_start_year, train_end, config.train_all_symbols,
        temporal_decay_per_year=getattr(config, "temporal_decay_per_year", 0.0),
    )
    models, inner, valid = _fit_spec_models(
        train,
        features,
        spec,
        config.validation_days,
        config.use_catboost,
        seed=stable_seed(spec.model_id, 42 + year),
        n_seeds=getattr(config, "n_seeds", 1),
    )
    valid_scored = _score_spec(
        _prediction_frame(df, spec, valid["date"].min(), train_end),
        train,
        features,
        spec,
        models,
        config.lgbm_weight,
    )
    calibrator = None
    if getattr(config, "use_calibration", False):
        from koscine.calibration import fit_isotonic, save_calibrator

        valid_labels = valid_scored[label_col(spec)].astype(float).values
        calibrator = fit_isotonic(valid_scored["raw_score"].values, valid_labels)
        # Re-score valid with calibrator for downstream metric reporting
        valid_scored["calibrated_score"] = calibrator.predict(valid_scored["raw_score"].values)
        valid_scored["score"] = valid_scored["calibrated_score"].clip(0.0, 1.0)

    test_scored = _score_spec(
        _prediction_frame(
            df,
            spec,
            pd.Timestamp(year=year, month=1, day=1),
            pd.Timestamp(year=year, month=12, day=31),
        ),
        train,
        features,
        spec,
        models,
        config.lgbm_weight,
        calibrator=calibrator,
    )
    model_prefix = f"{run_dir.name}_{year}_{spec.model_id}"
    model_meta = _save_models(run_dir, model_prefix, models, features, spec, train, train_end)
    if calibrator is not None:
        save_calibrator(calibrator, run_dir / "models" / f"{model_prefix}_calibrator.json")
    for frame, split in ((valid_scored, "valid"), (test_scored, "test")):
        frame["prediction_year"] = year
        frame["split"] = split
        frame["train_end"] = train_end

    metrics_rows = [
        {
            "year": year,
            "split": "valid",
            "tier": spec.name,
            "side": spec.side,
            "threshold": spec.threshold,
            **model_metrics(valid_scored, spec),
        },
        {
            "year": year,
            "split": "test",
            "tier": spec.name,
            "side": spec.side,
            "threshold": spec.threshold,
            **model_metrics(test_scored, spec),
        },
    ]
    train_summary = pd.DataFrame(
        [
            {
                "year": year,
                "tier": spec.name,
                "side": spec.side,
                "threshold": spec.threshold,
                "train_end": train_end,
                "train_rows": len(train),
                "inner_rows": len(inner),
                "valid_rows": len(valid),
                "train_positives": int(train[label_col(spec)].sum()),
                "train_positive_rate": float(train[label_col(spec)].mean()),
            }
        ]
    )
    return valid_scored, test_scored, pd.DataFrame(metrics_rows), model_meta, train_summary


def run_tiered_clean_direction(
    dataset_path: Path,
    config: TieredCleanConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"tiered_clean_direction_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df, features, specs = prepare_tiered_frame(dataset_path, config)
    manifest = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.__dict__,
        "feature_count": len(features),
        "features": features,
        "tiers": [
            {
                "model_id": spec.model_id,
                "tier": spec.name,
                "side": spec.side,
                "threshold": spec.threshold,
                "predict_symbols": list(spec.predict_symbols),
            }
            for spec in specs
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    all_valid = []
    all_test = []
    all_metrics = []
    all_model_meta = []
    all_train_summary = []
    for year in range(config.start_test_year, config.end_test_year + 1):
        for spec in specs:
            valid_scored, test_scored, metrics, meta, train_summary = train_score_one_spec(
                df,
                features,
                spec,
                year,
                config,
                run_dir,
            )
            keep_cols = _scored_keep_cols() + ["prediction_year", "split", "train_end"]
            valid_keep = [c for c in keep_cols if c in valid_scored.columns]
            test_keep = [c for c in keep_cols if c in test_scored.columns]
            all_valid.append(valid_scored[valid_keep])
            all_test.append(test_scored[test_keep])
            all_metrics.append(metrics)
            all_model_meta.extend(meta)
            all_train_summary.append(train_summary)

    valid = pd.concat(all_valid, ignore_index=True)
    test = pd.concat(all_test, ignore_index=True)
    scored = pd.concat([valid, test], ignore_index=True)
    model_metrics_df = pd.concat(all_metrics, ignore_index=True)
    train_summary_df = pd.concat(all_train_summary, ignore_index=True)

    selected_frames = []
    rule_rows = []
    grids = []
    for year in range(config.start_test_year, config.end_test_year + 1):
        for spec in specs:
            key = (scored["prediction_year"].eq(year) & scored["model_id"].eq(spec.model_id))
            valid_part = scored[key & scored["split"].eq("valid")].copy()
            test_part = scored[key & scored["split"].eq("test")].copy()
            rule, grid = choose_rule(valid_part, config)
            if not grid.empty:
                grid["year"] = year
                grid["model_id"] = spec.model_id
                grids.append(grid)
            selected = apply_rule(test_part, rule)
            selected["prediction_year"] = year
            selected["split"] = "test_selected"
            selected_frames.append(selected)
            rule_rows.append(
                {
                    "year": year,
                    "model_id": spec.model_id,
                    "tier": spec.name,
                    "side": spec.side,
                    "threshold": spec.threshold,
                    **rule,
                    **{f"valid_{k}": v for k, v in selected_metrics(apply_rule(valid_part, rule)).items()},
                    **{f"test_{k}": v for k, v in selected_metrics(selected).items()},
                }
            )

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    rules = pd.DataFrame(rule_rows)
    grid_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    stability = (
        rules.groupby(["model_id", "tier", "side", "threshold"], dropna=False)
        .agg(
            years=("year", "nunique"),
            total_calls=("test_calls", "sum"),
            avg_calls=("test_calls", "mean"),
            avg_hit_rate=("test_hit_rate", "mean"),
            avg_opposite_rate=("test_opposite_rate", "mean"),
            avg_move=("test_avg_move", "mean"),
            avg_signed_close_return=("test_avg_signed_close_return", "mean"),
            max_opposite_rate=("test_opposite_rate", "max"),
        )
        .reset_index()
    )

    run_stamp = run_dir.name
    scored.to_parquet(run_dir / "predictions" / f"{run_stamp}_all_scored.parquet", index=False)
    selected.to_csv(run_dir / "predictions" / f"{run_stamp}_selected_calls.csv", index=False)
    model_metrics_df.to_csv(run_dir / "reports" / f"{run_stamp}_model_metrics.csv", index=False)
    train_summary_df.to_csv(run_dir / "reports" / f"{run_stamp}_train_summary.csv", index=False)
    rules.to_csv(run_dir / "reports" / f"{run_stamp}_yearly_rules.csv", index=False)
    grid_df.to_csv(run_dir / "reports" / f"{run_stamp}_validation_rule_grid.csv", index=False)
    stability.to_csv(run_dir / "reports" / f"{run_stamp}_stability.csv", index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(PREDICTIONS_DIR / "tiered_clean_direction_scored_latest.parquet", index=False)
    selected.to_csv(PREDICTIONS_DIR / "tiered_clean_direction_selected_latest.csv", index=False)
    model_metrics_df.to_csv(REPORTS_DIR / "tiered_clean_direction_model_metrics_latest.csv", index=False)
    rules.to_csv(REPORTS_DIR / "tiered_clean_direction_yearly_rules_latest.csv", index=False)
    stability.to_csv(REPORTS_DIR / "tiered_clean_direction_stability_latest.csv", index=False)
    return run_dir


def cutoff_for_prediction_month(prediction_month: str, cutoff_day: int) -> pd.Timestamp:
    month_start = pd.Timestamp(prediction_month + "-01")
    previous_month_start = month_start - pd.offsets.MonthBegin(1)
    return pd.Timestamp(
        year=previous_month_start.year,
        month=previous_month_start.month,
        day=cutoff_day,
    )


def train_tiered_prod_models(
    dataset_path: Path,
    prediction_month: str,
    config: TieredProdConfig | None = None,
    run_name: str | None = None,
    update_current: bool = True,
    output_root: Path | None = None,
) -> Path:
    config = config or TieredProdConfig()
    run_id = run_name or f"prod_tiered_{prediction_month.replace('-', '')}_{timestamp()}"
    root = output_root or PROD_TIERED_ROOT
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)

    specs = tier_specs()
    df, features, _ = prepare_tiered_frame(dataset_path, config, specs)
    train_end = cutoff_for_prediction_month(prediction_month, config.train_cutoff_day)
    models_meta = []
    arbiter_validation_frames = []
    for spec in specs:
        train = _train_frame(
            df, spec, config.train_start_year, train_end, config.train_all_symbols,
            temporal_decay_per_year=getattr(config, "temporal_decay_per_year", 0.0),
        )
        models, inner, valid = _fit_spec_models(
            train,
            features,
            spec,
            365,
            config.use_catboost,
            seed=stable_seed(spec.model_id, 9000),
            n_seeds=getattr(config, "n_seeds", 1),
        )
        valid_scored = _score_spec(
            _prediction_frame(df, spec, valid["date"].min(), train_end),
            train,
            features,
            spec,
            models,
            config.lgbm_weight,
        )
        arbiter_validation_frames.append(valid_scored[_scored_keep_cols()].copy())
        prefix = spec.model_id
        saved = _save_models(run_dir, prefix, models, features, spec, train, train_end)[0]
        saved.update(
            {
                "prediction_month": prediction_month,
                "inner_rows": len(inner),
                "valid_rows": len(valid),
                "feature_profile": "tiered_clean_direction",
                "lgbm_weight": config.lgbm_weight,
                "predict_symbols": list(spec.predict_symbols),
            }
        )
        models_meta.append(saved)
        del train, models, inner, valid, valid_scored, saved
        _trim_process_memory()
    arbiter_meta = None
    if arbiter_validation_frames:
        arbiter_meta = _train_trade_arbiter(pd.concat(arbiter_validation_frames, ignore_index=True), run_dir)
        del arbiter_validation_frames
        _trim_process_memory()

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "prediction_month": prediction_month,
        "train_end": str(train_end.date()),
        "model_stack": "tiered_clean_direction_lgbm_catboost",
        "config": config.__dict__,
        "models": models_meta,
        "arbiter": arbiter_meta,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    del df, features, specs
    _trim_process_memory()
    if update_current:
        current_dir = PROD_TIERED_ROOT / "current"
        if current_dir.exists():
            shutil.rmtree(current_dir)
        shutil.copytree(run_dir, current_dir)
    return run_dir


def _load_prod_model(meta: dict, prod_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for family in meta["models"]:
        if family["family"].endswith("_ensemble") and "files" in family:
            loaded = []
            for f in family["files"]:
                path = prod_dir / f["model_file"]
                if not path.exists():
                    path = prod_dir / "models" / f["model_file"]
                if family["family"].startswith("lgbm"):
                    loaded.append(lgb.Booster(model_file=str(path)))
                else:
                    model = CatBoostClassifier()
                    model.load_model(str(path))
                    loaded.append(model)
            out[family["family"]] = loaded
            continue
        path = prod_dir / family["model_file"]
        if not path.exists():
            path = prod_dir / "models" / family["model_file"]
        if family["family"] == "lgbm":
            out["lgbm"] = lgb.Booster(model_file=str(path))
        elif family["family"] == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(path))
            out["catboost"] = model
    return out


def _load_trade_arbiter(prod_dir: Path | None, manifest: dict | None) -> tuple[lgb.Booster, dict] | None:
    prod_dir = prod_dir or (PROD_TIERED_ROOT / "current")
    if manifest is None:
        manifest_path = prod_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    meta = manifest.get("arbiter")
    if not meta:
        meta_path = prod_dir / ARBITER_META_FILE
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_file = meta.get("model_file", str(Path("models") / ARBITER_MODEL_FILE))
    model_path = prod_dir / model_file
    if not model_path.exists():
        model_path = prod_dir / "models" / Path(model_file).name
    if not model_path.exists():
        return None
    return lgb.Booster(model_file=str(model_path)), meta


def _apply_trade_arbiter(out: pd.DataFrame, prod_dir: Path | None, manifest: dict | None) -> pd.DataFrame:
    loaded = _load_trade_arbiter(prod_dir, manifest)
    if loaded is None:
        out["arbiter_available"] = False
        return out
    model, meta = loaded
    features = list(meta.get("features", ARBITER_FEATURES))
    pairs = _pair_for_arbiter(out, include_labels=False)
    if pairs.empty:
        out["arbiter_available"] = False
        return out
    for feature in features:
        if feature not in pairs:
            pairs[feature] = np.nan
    probs = model.predict(pairs[features], num_iteration=model.best_iteration)
    pairs["arbiter_no_trade_prob"] = probs[:, ARBITER_LABEL_TO_CLASS["none"]]
    pairs["arbiter_long_prob"] = probs[:, ARBITER_LABEL_TO_CLASS["up"]]
    pairs["arbiter_short_prob"] = probs[:, ARBITER_LABEL_TO_CLASS["down"]]
    pairs["arbiter_score"] = pairs[["arbiter_long_prob", "arbiter_short_prob"]].max(axis=1)
    high_long = pairs["arbiter_long_prob"].ge(0.50) & pairs["up_score"].ge(0.80) & pairs["down_score"].le(0.60)
    high_short = pairs["arbiter_short_prob"].ge(0.45) & pairs["down_score"].ge(0.75) & pairs["up_score"].le(0.60)
    low_long = ~high_long & pairs["arbiter_long_prob"].ge(0.45) & pairs["up_score"].ge(0.75) & pairs["down_score"].le(0.65)
    low_short = ~high_short & pairs["arbiter_short_prob"].ge(0.40) & pairs["down_score"].ge(0.70) & pairs["up_score"].le(0.65)
    conflict = (
        (pairs["arbiter_long_prob"].ge(0.30) & pairs["arbiter_short_prob"].ge(0.30))
        | (pairs["up_score"].ge(0.70) & pairs["down_score"].ge(0.70))
    )
    pairs["final_side"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        ["up", "down", "up", "down", "conflict"],
        default="none",
    )
    pairs["trade_priority"] = np.select(
        [high_long | high_short, low_long | low_short, conflict],
        ["high", "low", "conflict"],
        default="watch",
    )
    pairs["trade_bucket"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        ["high_long", "high_short", "low_long", "low_short", "conflict"],
        default="watch",
    )
    pairs["trade_arbitration_reason"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        [
            "arbiter_long>=0.50_up>=0.80_down<=0.60",
            "arbiter_short>=0.45_down>=0.75_up<=0.60",
            "arbiter_long>=0.45_up>=0.75_down<=0.65",
            "arbiter_short>=0.40_down>=0.70_up<=0.65",
            "arbiter_long_short_conflict",
        ],
        default="arbiter_no_trade",
    )
    pairs["arbiter_available"] = True
    pair_keys = ["date", "symbol", "tier", "threshold"]
    pair_keys.extend([col for col in ("prediction_year", "train_end", "split") if col in out.columns and col in pairs.columns])
    arb_cols = [
        "arbiter_available",
        "arbiter_no_trade_prob",
        "arbiter_long_prob",
        "arbiter_short_prob",
        "arbiter_score",
        "final_side",
        "trade_priority",
        "trade_bucket",
        "trade_arbitration_reason",
    ]
    return out.drop(columns=[col for col in arb_cols if col in out.columns]).merge(
        pairs[pair_keys + arb_cols],
        on=pair_keys,
        how="left",
    )


def apply_downside_production_lock(
    predictions: pd.DataFrame,
    prod_dir: Path | None = None,
    manifest: dict | None = None,
    use_arbiter: bool = False,
) -> pd.DataFrame:
    out = predictions.copy()
    derived_cols = [
        "production_signal",
        "lock_profile",
        "lock_reason",
        "daily_downside_lock_count",
        "paired_down_score",
        "paired_down_rule_gate_pass",
        "paired_down_production_signal",
        "paired_up_score",
        "paired_up_rule_gate_pass",
        "paired_up_production_signal",
        "up_score",
        "down_score",
        "opposite_score",
        "score_gap",
        "final_side",
        "trade_priority",
        "trade_bucket",
        "trade_arbitration_reason",
        "arbiter_available",
        "arbiter_no_trade_prob",
        "arbiter_long_prob",
        "arbiter_short_prob",
        "arbiter_score",
        "upside_research_signal",
        "upside_research_profile",
        "upside_research_reason",
    ]
    out = out.drop(columns=[col for col in derived_cols if col in out.columns])
    liquid_down = out["model_id"].eq("liquid30_down_4pct_5d")
    rest_down = out["model_id"].eq("rest35_down_7pct_5d")
    liquid_up = out["model_id"].eq("liquid30_up_4pct_5d")
    rest_up = out["model_id"].eq("rest35_up_7pct_5d")
    gate = out["rule_gate_pass"].fillna(False).astype(bool)
    # Two tiers derived from 989-day backtest:
    #   * production_signal (PROD) — score >= 0.70 + gate. Quality tier
    #     (plus/normal/minus) is computed downstream and organizes the
    #     volume into 3-5 per category. No flat-mover filter here — the
    #     plus tier (expansion_calibrated>=0.50) does that job better.
    #   * actionable_signal (GO)  — tight per-side/tier thresholds. Also
    #     organized by quality tier downstream.
    prod_score_floor = out["score"].ge(0.70)
    out["production_signal"] = gate & prod_score_floor

    # Strict (GO) thresholds — relaxed from earlier overly-tight gate to give
    # ~14/day instead of ~8/day. Dropped rule_gate_strength>=0.75 (didn't add
    # precision in backtest) and lowered rest35 from 0.85->0.80.
    liquid_score_floor = out["score"].ge(0.80)
    rest_score_floor = out["score"].ge(0.80)
    out["strict_signal"] = (
        (liquid_down & gate & liquid_score_floor)
        | (rest_down & gate & rest_score_floor)
        | (liquid_up & gate & liquid_score_floor)
        | (rest_up & gate & rest_score_floor)
    )

    daily_downside_locks = (
        out[out["production_signal"]]
        .groupby("date")["production_signal"]
        .sum()
        .rename("daily_downside_lock_count")
    )
    out["daily_downside_lock_count"] = out["date"].map(daily_downside_locks).fillna(0).astype(int)

    pair_keys = ["date", "symbol", "tier"]
    pair_keys.extend([col for col in ("prediction_year", "train_end") if col in out.columns])
    up_pairs = out[out["side"].eq("up")][
        pair_keys + ["score", "rule_gate_pass", "production_signal"]
    ].drop_duplicates(pair_keys, keep="last").rename(
        columns={
            "score": "paired_up_score",
            "rule_gate_pass": "paired_up_rule_gate_pass",
            "production_signal": "paired_up_production_signal",
        }
    )
    down_pairs = out[out["side"].eq("down")][
        pair_keys + ["score", "rule_gate_pass", "production_signal"]
    ].drop_duplicates(pair_keys, keep="last").rename(
        columns={
            "score": "paired_down_score",
            "rule_gate_pass": "paired_down_rule_gate_pass",
            "production_signal": "paired_down_production_signal",
        }
    )
    out = (
        out.reset_index(names="_row_order")
        .merge(up_pairs, on=pair_keys, how="left")
        .merge(down_pairs, on=pair_keys, how="left")
        .sort_values("_row_order")
        .drop(columns="_row_order")
    )
    out["paired_up_rule_gate_pass"] = out["paired_up_rule_gate_pass"].fillna(False).astype(bool)
    out["paired_up_production_signal"] = out["paired_up_production_signal"].fillna(False).astype(bool)
    out["paired_down_rule_gate_pass"] = out["paired_down_rule_gate_pass"].fillna(False).astype(bool)
    out["paired_down_production_signal"] = out["paired_down_production_signal"].fillna(False).astype(bool)
    out["up_score"] = np.where(out["side"].eq("up"), out["score"], out["paired_up_score"])
    out["down_score"] = np.where(out["side"].eq("down"), out["score"], out["paired_down_score"])
    out["opposite_score"] = np.where(out["side"].eq("up"), out["down_score"], out["up_score"])
    out["score_gap"] = out["up_score"] - out["down_score"]

    high_long = out["up_score"].ge(0.75) & out["down_score"].le(0.55)
    high_short = out["down_score"].ge(0.75) & out["up_score"].le(0.60)
    conflict = out["up_score"].ge(0.70) & out["down_score"].ge(0.70)
    low_long = ~high_long & ~conflict & out["up_score"].ge(0.70) & out["score_gap"].ge(0.10)
    low_short = ~high_short & ~conflict & out["down_score"].ge(0.70) & out["score_gap"].le(-0.10)
    out["final_side"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        ["up", "down", "up", "down", "conflict"],
        default="none",
    )
    out["trade_priority"] = np.select(
        [high_long | high_short, low_long | low_short, conflict],
        ["high", "low", "conflict"],
        default="watch",
    )
    out["trade_bucket"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        ["high_long", "high_short", "low_long", "low_short", "conflict"],
        default="watch",
    )
    out["trade_arbitration_reason"] = np.select(
        [high_long, high_short, low_long, low_short, conflict],
        [
            "up_score>=0.75_and_down_score<=0.55",
            "down_score>=0.75_and_up_score<=0.60",
            "up_score>=0.70_and_score_gap>=0.10",
            "down_score>=0.70_and_score_gap<=-0.10",
            "both_scores>=0.70_direction_conflict",
        ],
        default="no_clear_score_arbitration",
    )
    if use_arbiter:
        out = _apply_trade_arbiter(out, prod_dir, manifest)
    else:
        out["arbiter_available"] = False

    liquid_up = out["model_id"].eq("liquid30_up_4pct_5d")
    rest_up = out["model_id"].eq("rest35_up_7pct_5d")
    paired_down_score = out["paired_down_score"].fillna(1.0)
    no_paired_down_lock = ~out["paired_down_production_signal"]
    quiet_downside_day = out["daily_downside_lock_count"].le(2)
    liquid_up_research = (
        liquid_up
        & no_paired_down_lock
        & quiet_downside_day
        & paired_down_score.le(0.55)
        & out["score"].ge(0.50)
    )
    rest_up_research = (
        rest_up
        & no_paired_down_lock
        & quiet_downside_day
        & paired_down_score.le(0.60)
        & out["score"].ge(0.55)
    )
    out["upside_research_signal"] = liquid_up_research | rest_up_research
    out["upside_research_profile"] = np.select(
        [liquid_up_research, rest_up_research, out["side"].eq("up")],
        ["upside_v1_liquid_quiet_downside", "upside_v1_rest_quiet_downside", "upside_v1_filtered_out"],
        default="not_upside",
    )
    out["upside_research_reason"] = np.select(
        [
            liquid_up_research,
            rest_up_research,
            out["side"].ne("up"),
            out["paired_down_production_signal"],
            out["daily_downside_lock_count"].gt(2),
            liquid_up & paired_down_score.gt(0.55),
            rest_up & paired_down_score.gt(0.60),
            liquid_up & out["score"].lt(0.50),
            rest_up & out["score"].lt(0.55),
        ],
        [
            "liquid_up_low_downside_pressure",
            "rest_up_low_downside_pressure",
            "not_upside_model",
            "paired_downside_prod_veto",
            "daily_downside_pressure_veto",
            "paired_down_score_too_high",
            "paired_down_score_too_high",
            "up_score_too_low",
            "up_score_too_low",
        ],
        default="upside_filter_not_met",
    )

    out["lock_profile"] = "two_tier_v3"
    gate = out["rule_gate_pass"].fillna(False).astype(bool)
    score_75 = out["score"].ge(0.75)
    strict = out["strict_signal"].fillna(False).astype(bool)
    out["lock_reason"] = np.select(
        [
            strict,
            gate & score_75 & ~strict,
            gate & ~score_75,
            ~gate,
        ],
        [
            "strict_GO_passes_tight_gate",
            "prod_score_0.75_gate_pass",
            "below_prod_floor_score<0.75",
            "rule_gate_failed",
        ],
        default="filter_not_met",
    )
    prod = out["production_signal"].fillna(False).astype(bool)
    long_prod = prod & out["side"].eq("up")
    short_prod = prod & out["side"].eq("down")
    out["final_side"] = np.select([long_prod, short_prod], ["up", "down"], default="none")
    out["trade_priority"] = np.where(prod, "high", "watch")
    out["trade_bucket"] = np.select(
        [long_prod, short_prod],
        ["high_long", "high_short"],
        default="watch",
    )
    out["trade_arbitration_reason"] = np.where(
        prod,
        out["lock_reason"],
        "production_gate_not_met",
    )
    out["arbiter_available"] = False
    out["arbiter_no_trade_prob"] = np.nan
    out["arbiter_long_prob"] = np.nan
    out["arbiter_short_prob"] = np.nan
    out["arbiter_score"] = np.nan

    # Quality tier (matches API enrichment so promote script can cap by tier).
    # Plus = expansion_calibrated>=0.50 AND tier_reliable (liquid30 OR rest35-up)
    # Normal = exactly one of the two
    # Minus  = neither
    exp_cal_col = out.get("expansion_calibrated")
    if exp_cal_col is None:
        exp_cal_num = pd.Series(0.0, index=out.index)
    else:
        exp_cal_num = pd.to_numeric(exp_cal_col, errors="coerce").fillna(0.0)
    high_exp = exp_cal_num.ge(0.50)
    tier_col = out.get("tier", pd.Series("", index=out.index))
    tier_reliable = tier_col.eq("liquid30") | (tier_col.eq("rest35") & out["side"].eq("up"))
    out["tier_reliable"] = tier_reliable
    quality_score = high_exp.astype(int) + tier_reliable.astype(int)
    out["signal_quality"] = np.select(
        [quality_score == 2, quality_score == 1],
        ["plus", "normal"],
        default="minus",
    )
    return out


def predict_tiered_prod(
    dataset_path: Path,
    as_of_date: str,
    prod_dir: Path = PROD_TIERED_ROOT / "current",
    output_dir: Path = PREDICTIONS_DIR / "prod",
) -> pd.DataFrame:
    manifest_path = prod_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No production manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TieredProdConfig(**{k: v for k, v in manifest.get("config", {}).items() if k in TieredProdConfig.__dataclass_fields__})
    df, features, specs = prepare_tiered_frame(dataset_path, config)
    as_of = pd.Timestamp(as_of_date)
    frames = []
    train_end = pd.Timestamp(manifest["train_end"])
    for meta in manifest["models"]:
        spec = next(s for s in specs if s.model_id == meta["model_id"])
        train = _train_frame(df, spec, config.train_start_year, train_end, config.train_all_symbols)
        rows = df[df["date"].eq(as_of) & df["symbol"].isin(spec.predict_symbols)].copy()
        if rows.empty:
            continue
        models = _load_prod_model(meta, prod_dir)
        scored = _score_spec(rows, train, features, spec, models, float(meta.get("lgbm_weight", config.lgbm_weight)))
        scored["prediction_for_entry"] = "next_trading_day_open"
        scored["prod_min_score"] = manifest["config"].get("min_score", 0.0)
        keep = [
            "date",
            "symbol",
            "close",
            "side",
            "threshold",
            "score",
            "model_id",
            "prediction_for_entry",
            "tier",
            "rule_name",
            "rule_gate_pass",
            "rule_gate_strength",
            "model_score",
            "lgbm_score",
            "catboost_score",
            "prod_min_score",
            "entry_1d_date",
            "entry_1d_open",
            f"future_{HORIZON_DAYS}d_date",
            "atr_pct_14",
            "atm_iv",
        ]
        keep = [c for c in keep if c in scored.columns]
        frames.append(scored[keep])
    if not frames:
        raise ValueError(f"No tiered prediction rows found for {as_of_date}")
    predictions = apply_downside_production_lock(pd.concat(frames, ignore_index=True), prod_dir=prod_dir, manifest=manifest)
    predictions["passes_min_score"] = predictions["production_signal"]
    predictions = predictions.sort_values(
        ["score", "threshold"], ascending=[False, False]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_date = as_of.strftime("%Y%m%d")
    predictions.to_csv(output_dir / f"prod_predictions_{safe_date}.csv", index=False)
    predictions.to_parquet(output_dir / f"prod_predictions_{safe_date}.parquet", index=False)
    return predictions


def predict_tiered_prod_many(
    dataset_path: Path,
    as_of_dates: list[str],
    prod_dir: Path = PROD_TIERED_ROOT / "current",
    output_dir: Path = PREDICTIONS_DIR / "prod",
    progress=None,
) -> pd.DataFrame:
    manifest_path = prod_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No production manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TieredProdConfig(**{k: v for k, v in manifest.get("config", {}).items() if k in TieredProdConfig.__dataclass_fields__})
    df, features, specs = prepare_tiered_frame(dataset_path, config)
    train_end = pd.Timestamp(manifest["train_end"])
    requested = [pd.Timestamp(date).normalize() for date in as_of_dates]
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_contexts = []
    for meta in manifest["models"]:
        spec = next(s for s in specs if s.model_id == meta["model_id"])
        if progress:
            progress(f"loading {spec.model_id}")
        train = _train_frame(df, spec, config.train_start_year, train_end, config.train_all_symbols)
        models = _load_prod_model(meta, prod_dir)
        spec_contexts.append((meta, spec, train, models, float(meta.get("lgbm_weight", config.lgbm_weight))))

    summary_rows = []
    total_dates = len(requested)
    keep = [
        "date",
        "symbol",
        "close",
        "side",
        "threshold",
        "score",
        "model_id",
        "prediction_for_entry",
        "tier",
        "rule_name",
        "rule_gate_pass",
        "rule_gate_strength",
        "model_score",
        "lgbm_score",
        "catboost_score",
        "prod_min_score",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_date",
        "atr_pct_14",
        "atm_iv",
    ]
    for idx, date in enumerate(requested, start=1):
        date_label = date.strftime("%Y-%m-%d")
        if progress:
            progress(f"predicting {date_label} ({idx}/{total_dates})")
        frames = []
        for _meta, spec, train, models, lgbm_weight in spec_contexts:
            rows = df[df["date"].eq(date) & df["symbol"].isin(spec.predict_symbols)].copy()
            if rows.empty:
                continue
            scored = _score_spec(rows, train, features, spec, models, lgbm_weight)
            scored["prediction_for_entry"] = "next_trading_day_open"
            scored["prod_min_score"] = manifest["config"].get("min_score", 0.0)
            keep_present = [c for c in keep if c in scored.columns]
            frames.append(scored[keep_present])
        if not frames:
            summary_rows.append({"date": date_label, "rows": 0, "max_score": np.nan})
            continue
        predictions = apply_downside_production_lock(pd.concat(frames, ignore_index=True), prod_dir=prod_dir, manifest=manifest)
        predictions["passes_min_score"] = predictions["production_signal"]
        predictions = predictions.sort_values(
            ["score", "threshold"], ascending=[False, False]
        )
        safe_date = date.strftime("%Y%m%d")
        predictions.to_csv(output_dir / f"prod_predictions_{safe_date}.csv", index=False)
        predictions.to_parquet(output_dir / f"prod_predictions_{safe_date}.parquet", index=False)
        if progress:
            progress(f"wrote predictions for {date_label} ({len(predictions)} rows)")
        summary_rows.append(
            {
                "date": date_label,
                "rows": len(predictions),
                "max_score": float(predictions["score"].max()),
                "gate_pass": int(predictions["rule_gate_pass"].fillna(False).sum()),
                "liquid_rows": int(predictions["tier"].eq("liquid30").sum()),
                "rest_rows": int(predictions["tier"].eq("rest35").sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def tiered_prod_auto(
    dataset_path: Path,
    as_of_date: str,
    config: TieredProdConfig | None = None,
) -> pd.DataFrame:
    config = config or TieredProdConfig()
    current_manifest = PROD_TIERED_ROOT / "current" / "manifest.json"
    if not current_manifest.exists():
        raise FileNotFoundError("No current tiered production model found. Run tiered-prod-train manually first.")
    manifest = json.loads(current_manifest.read_text(encoding="utf-8"))
    if not str(manifest.get("model_stack", "")).startswith("tiered_clean_direction"):
        raise RuntimeError("Current production model is not a tiered clean-direction model. Run tiered-prod-train manually first.")
    return predict_tiered_prod(dataset_path, as_of_date)
