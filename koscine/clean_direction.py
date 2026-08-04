from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import PREDICTIONS_DIR, REPORTS_DIR, RUNS_DIR, SILVER_DATA_ROOT, SYMBOL_ALIASES
from koscine.training import feature_columns


HORIZON_DAYS = 5

LIQUID30_RAW = [
    "HDFCBANK",
    "RELIANCE",
    "ICICIBANK",
    "INFY",
    "TCS",
    "KOTAKBANK",
    "AXISBANK",
    "SBIN",
    "BAJFINANCE",
    "HINDUNILVR",
    "ITC",
    "LT",
    "WIPRO",
    "HCLTECH",
    "MARUTI",
    "ASIANPAINT",
    "TITAN",
    "SUNPHARMA",
    "ULTRACEMCO",
    "NESTLEIND",
    "TATAMOTORS",
    "BAJAJFINSV",
    "M&M",
    "ADANIENT",
    "JIOFIN",
    "ETERNAL",
    "CHOLAFIN",
    "NTPC",
    "POWERGRID",
    "TMPV",
]

LIQUID30_SECTOR_INDEX = {
    "HDFCBANK": "NIFTY BANK",
    "ICICIBANK": "NIFTY BANK",
    "KOTAKBANK": "NIFTY BANK",
    "AXISBANK": "NIFTY BANK",
    "SBIN": "NIFTY BANK",
    "BAJFINANCE": "NIFTY FINANCIAL SERVICES",
    "BAJAJFINSV": "NIFTY FINANCIAL SERVICES",
    "CHOLAFIN": "NIFTY FINANCIAL SERVICES",
    "JIOFIN": "NIFTY FINANCIAL SERVICES",
    "RELIANCE": "NIFTY OIL & GAS",
    "INFY": "NIFTY IT",
    "TCS": "NIFTY IT",
    "WIPRO": "NIFTY IT",
    "HCLTECH": "NIFTY IT",
    "HINDUNILVR": "NIFTY FMCG",
    "ITC": "NIFTY FMCG",
    "NESTLEIND": "NIFTY FMCG",
    "MARUTI": "NIFTY AUTO",
    "M&M": "NIFTY AUTO",
    "TMPV": "NIFTY AUTO",
    "TATAMOTORS": "NIFTY AUTO",
    "ASIANPAINT": "NIFTY CONSUMER DURABLES",
    "TITAN": "NIFTY CONSUMER DURABLES",
    "SUNPHARMA": "NIFTY PHARMA",
    "ULTRACEMCO": "NIFTY REALTY",
    "ADANIENT": "NIFTY METAL",
    "NTPC": "NIFTY ENERGY",
    "POWERGRID": "NIFTY ENERGY",
    "LT": "NIFTY INFRASTRUCTURE",
    "ETERNAL": "NIFTY CONSUMER SERVICES",
}


def normalize_symbol(symbol: object) -> str:
    value = str(symbol).strip().upper()
    return SYMBOL_ALIASES.get(value, value)


def liquid30_symbols() -> list[str]:
    out: list[str] = []
    for symbol in LIQUID30_RAW:
        canonical = normalize_symbol(symbol)
        if canonical not in out:
            out.append(canonical)
    return out


def load_sector_index_features(symbols: list[str]) -> pd.DataFrame:
    path = SILVER_DATA_ROOT / "indices.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "symbol"])
    idx = pd.read_parquet(path)
    idx = idx.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx["index_key"] = idx["index_name"].astype(str).str.upper().str.strip()
    needed = set(LIQUID30_SECTOR_INDEX.values()) | {"NIFTY 50", "CNX NIFTY", "S&P CNX NIFTY"}
    idx = idx[idx["index_key"].isin(needed)].sort_values(["index_key", "date"])
    if idx.empty:
        return pd.DataFrame(columns=["date", "symbol"])

    frames = []
    for _, group in idx.groupby("index_key"):
        group = group.sort_values("date").copy()
        close = pd.to_numeric(group["close"], errors="coerce")
        group["sector_ret_1d"] = close.pct_change()
        group["sector_ret_5d"] = close.pct_change(5)
        group["sector_ret_20d"] = close.pct_change(20)
        group["sector_vol_20"] = group["sector_ret_1d"].rolling(20, min_periods=10).std()
        group["sector_close_sma20_dist"] = close / close.rolling(20, min_periods=10).mean() - 1.0
        group["sector_close_sma50_dist"] = close / close.rolling(50, min_periods=25).mean() - 1.0
        frames.append(group)
    sector_idx = pd.concat(frames, ignore_index=True)

    nifty = sector_idx[sector_idx["index_key"].isin({"NIFTY 50", "CNX NIFTY", "S&P CNX NIFTY"})]
    nifty = nifty.sort_values("date").drop_duplicates("date", keep="last")[
        ["date", "sector_ret_5d", "sector_ret_20d", "sector_vol_20"]
    ].rename(
        columns={
            "sector_ret_5d": "nifty_idx_ret_5d",
            "sector_ret_20d": "nifty_idx_ret_20d",
            "sector_vol_20": "nifty_idx_vol_20",
        }
    )

    symbol_frames = []
    for raw_symbol, index_key in LIQUID30_SECTOR_INDEX.items():
        symbol = normalize_symbol(raw_symbol)
        if symbol not in symbols:
            continue
        part = sector_idx[sector_idx["index_key"].eq(index_key)].copy()
        if part.empty:
            continue
        part["symbol"] = symbol
        keep = [
            "date",
            "symbol",
            "sector_ret_1d",
            "sector_ret_5d",
            "sector_ret_20d",
            "sector_vol_20",
            "sector_close_sma20_dist",
            "sector_close_sma50_dist",
        ]
        symbol_frames.append(part[keep])
    if not symbol_frames:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.concat(symbol_frames, ignore_index=True)
    out = out.merge(nifty, on="date", how="left")
    out["sector_rel_ret_5d"] = out["sector_ret_5d"] - out["nifty_idx_ret_5d"]
    out["sector_rel_ret_20d"] = out["sector_ret_20d"] - out["nifty_idx_ret_20d"]
    out["sector_risk_on"] = (
        out["sector_ret_20d"].gt(0) & out["sector_rel_ret_20d"].gt(0)
    ).astype(float)
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def load_price_action_dates() -> pd.DataFrame:
    path = SILVER_DATA_ROOT / "corp_actions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "action_date", "price_action_flag"])
    ca = pd.read_parquet(path)
    if ca.empty:
        return pd.DataFrame(columns=["symbol", "action_date", "price_action_flag"])
    text = ca["action_type"].astype(str)
    mask = text.str.contains(
        "split|bonus|face|sub.?division|demerger|scheme|arrangement|rights|merger",
        case=False,
        regex=True,
        na=False,
    )
    out = ca.loc[mask, ["symbol", "ex_date", "date", "action_type"]].copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["action_date"] = pd.to_datetime(out["ex_date"]).fillna(pd.to_datetime(out["date"]))
    out = out.dropna(subset=["symbol", "action_date"])
    out["price_action_flag"] = 1
    return out[["symbol", "action_date", "price_action_flag", "action_type"]].drop_duplicates()


@dataclass(frozen=True)
class CleanDirectionConfig:
    train_start_year: int = 2012
    start_test_year: int = 2018
    end_test_year: int = 2025
    target_pct: float = 0.04
    adverse_limit: float = 0.0091
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    precision_floor: float = 0.90
    min_validation_calls: int = 20
    train_universe: str = "all"
    liquid_weight: float = 2.0
    max_weekly_abs_move: float = 0.50
    episode_decay_weight: float = 0.35
    ensemble_lgbm_weight: float = 0.55
    use_catboost: bool = True
    direct_side_weight: float = 0.50


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def add_liquid30_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    liquid_mask = out["symbol"].isin(symbols)
    liquid = out[liquid_mask].copy()

    breadth = (
        liquid.groupby("date")
        .agg(
            l30_breadth_ret1_pos=("ret_1d", lambda s: (s > 0).mean()),
            l30_breadth_ret3_pos=("ret_3d", lambda s: (s > 0).mean()),
            l30_breadth_ret5_pos=("ret_5d", lambda s: (s > 0).mean()),
            l30_breadth_ret20_pos=("ret_20d", lambda s: (s > 0).mean()),
            l30_median_ret1=("ret_1d", "median"),
            l30_median_ret5=("ret_5d", "median"),
            l30_median_ret20=("ret_20d", "median"),
            l30_avg_atr_rank=("atr_pct_14_cs_rank", "mean"),
            l30_avg_vol_rank=("vol_sma20_ratio_cs_rank", "mean"),
            l30_avg_bb_rank=("bb_width_20_cs_rank", "mean"),
            l30_avg_relret_rank=("rel_ret_5d_vs_nifty", "mean"),
        )
        .reset_index()
    )
    out = out.merge(breadth, on="date", how="left")

    rank_cols = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_sma20_ratio",
        "atr_pct_14",
        "bb_width_20",
        "realized_vol_20",
        "turnover_ratio_20",
        "delivery_pct_chg_5",
        "close_sma20_dist",
        "close_sma50_dist",
        "dist_high_20",
        "dist_low_20",
        "pcr_oi",
        "pcr_vol",
        "atm_iv",
        "fut_oi_chg_5",
        "fut_chg_oi_chg_5",
    ]
    for col in rank_cols:
        if col in out:
            out[f"{col}_l30_rank"] = np.nan
            out.loc[liquid_mask, f"{col}_l30_rank"] = out.loc[liquid_mask].groupby("date")[
                col
            ].rank(pct=True)

    grouped = out.groupby("symbol", group_keys=False)
    prev_close = grouped["close"].shift(1)
    green = out["close"].gt(prev_close).astype(float)
    red = out["close"].lt(prev_close).astype(float)
    out["green_count_5"] = green.groupby(out["symbol"]).transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    out["red_count_5"] = red.groupby(out["symbol"]).transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    out["gap_pct"] = out["open"] / prev_close - 1.0
    out["intraday_body_pct"] = out["close"] / out["open"] - 1.0
    out["upper_wick_pct"] = out["high"] / out[["open", "close"]].max(axis=1) - 1.0
    out["lower_wick_pct"] = 1.0 - out["low"] / out[["open", "close"]].min(axis=1)

    for window in (10, 20, 50):
        rolling_high = grouped["high"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).max()
        )
        rolling_low = grouped["low"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).min()
        )
        out[f"dist_high_{window}"] = out["close"] / rolling_high - 1.0
        out[f"dist_low_{window}"] = out["close"] / rolling_low - 1.0
        out[f"close_position_{window}"] = (out["close"] - rolling_low) / (
            rolling_high - rolling_low
        ).replace(0, np.nan)

    for window in (3, 5, 10, 20):
        rolling_high = grouped["high"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).max()
        )
        rolling_low = grouped["low"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).min()
        )
        rolling_open = grouped["open"].shift(window - 1)
        prev_high = grouped["high"].shift(1)
        prev_low = grouped["low"].shift(1)
        prev_high_roll = prev_high.groupby(out["symbol"]).transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).max()
        )
        prev_low_roll = prev_low.groupby(out["symbol"]).transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).min()
        )
        out[f"past_up_move_{window}d"] = rolling_high / rolling_open - 1.0
        out[f"past_down_move_{window}d"] = 1.0 - rolling_low / rolling_open
        out[f"past_range_{window}d"] = rolling_high / rolling_low - 1.0
        out[f"close_vs_open_{window}d"] = out["close"] / rolling_open - 1.0
        out[f"high_breakout_{window}d"] = out["close"] / prev_high_roll - 1.0
        out[f"low_breakdown_{window}d"] = out["close"] / prev_low_roll - 1.0

    percentile_cols = [
        "atr_pct_14",
        "bb_width_20",
        "realized_vol_20",
        "range_pct",
        "vol_sma20_ratio",
        "turnover_ratio_20",
        "delivery_qty_ratio_20",
    ]
    for col in percentile_cols:
        if col in out:
            out[f"{col}_pctile_126"] = grouped[col].transform(
                lambda s: s.rolling(126, min_periods=40).rank(pct=True)
            )
            out[f"{col}_pctile_252"] = grouped[col].transform(
                lambda s: s.rolling(252, min_periods=80).rank(pct=True)
            )

    symbol_codes = {symbol: i for i, symbol in enumerate(symbols)}
    out["liquid30_symbol_code"] = out["symbol"].map(symbol_codes).astype(float)
    return out


def add_sector_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    sector = load_sector_index_features(symbols)
    if sector.empty:
        return df
    out = df.merge(sector, on=["date", "symbol"], how="left")
    if "ret_5d" in out and "sector_ret_5d" in out:
        out["stock_rel_sector_ret_5d"] = out["ret_5d"] - out["sector_ret_5d"]
    if "ret_20d" in out and "sector_ret_20d" in out:
        out["stock_rel_sector_ret_20d"] = out["ret_20d"] - out["sector_ret_20d"]
    return out


def add_label_validity_flags(df: pd.DataFrame, max_weekly_abs_move: float) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    prev_close = grouped["close"].shift(1)
    out["overnight_gap_abs"] = (out["open"] / prev_close - 1.0).abs()
    out["price_discontinuity"] = out["overnight_gap_abs"].gt(max_weekly_abs_move)
    actions = load_price_action_dates()
    out["corp_price_action"] = False
    if not actions.empty:
        action_keys = set(zip(actions["symbol"], pd.to_datetime(actions["action_date"]).dt.normalize()))
        keys = list(zip(out["symbol"], pd.to_datetime(out["date"]).dt.normalize()))
        out["corp_price_action"] = [key in action_keys for key in keys]
    out["invalid_price_event"] = out["price_discontinuity"] | out["corp_price_action"]
    out["future_invalid_price_event_5d"] = grouped["invalid_price_event"].transform(
        lambda s: s.iloc[::-1].shift(1).rolling(HORIZON_DAYS, min_periods=1).max().iloc[::-1]
    ).fillna(False).astype(bool)
    return out


def add_clean_direction_labels(
    df: pd.DataFrame,
    target_pct: float,
    adverse_limit: float,
    max_weekly_abs_move: float,
) -> pd.DataFrame:
    out = df.copy()
    up = out[f"up_move_{HORIZON_DAYS}d"]
    down = out[f"down_move_{HORIZON_DAYS}d"]
    fwd_close = out[f"fwd_return_{HORIZON_DAYS}d"]
    invalid_price = (
        out["future_invalid_price_event_5d"]
        if "future_invalid_price_event_5d" in out
        else pd.Series(False, index=out.index)
    )
    valid = (
        up.notna()
        & down.notna()
        & out["entry_1d_open"].notna()
        & up.le(max_weekly_abs_move)
        & down.le(max_weekly_abs_move)
        & fwd_close.abs().le(max_weekly_abs_move)
        & ~invalid_price.astype(bool)
    )
    out["label_clean_bull_4pct_5d"] = np.where(
        valid,
        (up > target_pct) & (down < adverse_limit),
        np.nan,
    )
    out["label_clean_bear_4pct_5d"] = np.where(
        valid,
        (down > target_pct) & (up < adverse_limit),
        np.nan,
    )
    out["clean_direction"] = np.select(
        [
            out["label_clean_bull_4pct_5d"].eq(1),
            out["label_clean_bear_4pct_5d"].eq(1),
        ],
        ["bull", "bear"],
        default="other",
    )
    out.loc[~valid, "clean_direction"] = np.nan
    out["label_clean_any_4pct_5d"] = np.where(
        valid,
        out["label_clean_bull_4pct_5d"].eq(1) | out["label_clean_bear_4pct_5d"].eq(1),
        np.nan,
    )
    out["label_clean_direction_bull_4pct_5d"] = np.where(
        out["label_clean_any_4pct_5d"].eq(1),
        out["label_clean_bull_4pct_5d"].astype(float),
        np.nan,
    )
    out = add_event_episode_weights(out)
    return out


def add_event_episode_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    out["clean_episode_start"] = False
    out["clean_episode_age"] = np.nan
    for _, idxs in out.groupby("symbol").groups.items():
        positions = list(idxs)
        last_direction = None
        age = 0
        for idx in positions:
            direction = out.at[idx, "clean_direction"]
            if direction in {"bull", "bear"}:
                if direction != last_direction:
                    age = 0
                    out.at[idx, "clean_episode_start"] = True
                else:
                    age += 1
                out.at[idx, "clean_episode_age"] = age
                last_direction = direction
            else:
                last_direction = None
                age = 0
    return out


def train_end_for_year(year: int, config: CleanDirectionConfig) -> pd.Timestamp:
    return pd.Timestamp(year=year - 1, month=config.train_cutoff_month, day=config.train_cutoff_day)


def lgbm_params(y: pd.Series, side: str, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31 if side in {"bull", "event"} else 47,
        "min_data_in_leaf": 260 if side == "event" else 180,
        "feature_fraction": 0.76,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 18.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }


def fit_side_model(
    inner: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    label_col: str,
    side: str,
    seed: int,
    weight_col: str | None = None,
) -> lgb.Booster:
    train_weight = inner[weight_col] if weight_col else None
    valid_weight = valid[weight_col] if weight_col else None
    train_set = lgb.Dataset(
        inner[features],
        label=inner[label_col].astype(int),
        weight=train_weight,
    )
    valid_set = lgb.Dataset(
        valid[features],
        label=valid[label_col].astype(int),
        weight=valid_weight,
        reference=train_set,
    )
    return lgb.train(
        lgbm_params(inner[label_col].astype(int), side=side, seed=seed),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(180), lgb.log_evaluation(0)],
    )


def catboost_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 1200,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 14,
        "random_strength": 1.5,
        "bagging_temperature": 1.0,
        "class_weights": [1.0, negatives / max(positives, 1)],
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "random_seed": seed,
        "verbose": False,
    }


def fit_catboost_model(
    inner: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    label_col: str,
    seed: int,
    weight_col: str | None = None,
) -> CatBoostClassifier:
    train_weight = inner[weight_col] if weight_col else None
    valid_weight = valid[weight_col] if weight_col else None
    model = CatBoostClassifier(**catboost_params(inner[label_col].astype(int), seed=seed))
    model.fit(
        Pool(inner[features], label=inner[label_col].astype(int), weight=train_weight),
        eval_set=Pool(valid[features], label=valid[label_col].astype(int), weight=valid_weight),
        use_best_model=True,
    )
    return model


def predict_lgbm(model: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(frame[features], num_iteration=model.best_iteration)


def predict_catboost(model: CatBoostClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def score_side_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> dict:
    y = df[label_col].astype(int)
    p = df[pred_col].astype(float)
    out = {
        "rows": len(df),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan
    except ValueError:
        out["roc_auc"] = np.nan
    ranked = df.sort_values(pred_col, ascending=False)
    for k in (25, 50, 100, 150, 200, 300):
        top = ranked.head(k)
        if len(top):
            out[f"precision_at_{k}"] = float(top[label_col].mean())
    return out


def make_side_candidates(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "date",
        "symbol",
        "entry_1d_date",
        "entry_1d_open",
        "up_move_5d",
        "down_move_5d",
        "fwd_return_5d",
        "clean_direction",
        "label_clean_bull_4pct_5d",
        "label_clean_bear_4pct_5d",
        "sector_risk_on",
        "sector_rel_ret_5d",
        "sector_rel_ret_20d",
        "stock_rel_sector_ret_5d",
        "stock_rel_sector_ret_20d",
        "nifty_idx_vol_20",
    ]
    base_cols = [col for col in base_cols if col in df.columns]
    bull = df[base_cols + ["bull_score", "bear_score"]].copy()
    bull["predicted_side"] = "bull"
    bull["side_score"] = bull["bull_score"]
    bull["opposite_score"] = bull["bear_score"]
    bull["actual_positive"] = bull["label_clean_bull_4pct_5d"].astype(bool)
    bull["actual_opposite"] = bull["label_clean_bear_4pct_5d"].astype(bool)

    bear = df[base_cols + ["bull_score", "bear_score"]].copy()
    bear["predicted_side"] = "bear"
    bear["side_score"] = bear["bear_score"]
    bear["opposite_score"] = bear["bull_score"]
    bear["actual_positive"] = bear["label_clean_bear_4pct_5d"].astype(bool)
    bear["actual_opposite"] = bear["label_clean_bull_4pct_5d"].astype(bool)

    out = pd.concat([bull, bear], ignore_index=True)
    out = out[out["side_score"].gt(out["opposite_score"])].copy()
    out["score_margin"] = out["side_score"] - out["opposite_score"]
    out["direction_confidence"] = out["side_score"] / (
        out["side_score"] + out["opposite_score"] + 1e-12
    )
    if "sector_risk_on" in out:
        out["regime_bucket"] = np.where(out["sector_risk_on"].fillna(0).gt(0), "risk_on", "risk_off")
    else:
        out["regime_bucket"] = "unknown"
    out["selection_score"] = out["side_score"] * (1.0 - out["opposite_score"]).clip(lower=0) * (
        1.0 + out["score_margin"]
    )
    out["clean_event_any"] = out["clean_direction"].isin(["bull", "bear"])
    out["direction_correct"] = out["actual_positive"]
    out["wrong_clean_direction"] = out["actual_opposite"]
    return out


def apply_symbol_regime_calibration(
    valid_candidates: pd.DataFrame,
    target_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid_candidates.empty or target_candidates.empty:
        return valid_candidates, target_candidates
    valid = valid_candidates.copy()
    target = target_candidates.copy()
    global_precision = float(valid["direction_correct"].mean()) if len(valid) else 0.0

    symbol_stats = (
        valid.groupby(["symbol", "predicted_side"])["direction_correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    symbol_stats["symbol_side_precision"] = (symbol_stats["sum"] + 2 * global_precision) / (
        symbol_stats["count"] + 2
    )
    symbol_stats = symbol_stats[["symbol", "predicted_side", "symbol_side_precision"]]

    regime_stats = (
        valid.groupby(["regime_bucket", "predicted_side"])["direction_correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    regime_stats["regime_side_precision"] = (regime_stats["sum"] + 5 * global_precision) / (
        regime_stats["count"] + 5
    )
    regime_stats = regime_stats[["regime_bucket", "predicted_side", "regime_side_precision"]]

    def calibrate(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.merge(symbol_stats, on=["symbol", "predicted_side"], how="left")
        out = out.merge(regime_stats, on=["regime_bucket", "predicted_side"], how="left")
        out["symbol_side_precision"] = out["symbol_side_precision"].fillna(global_precision)
        out["regime_side_precision"] = out["regime_side_precision"].fillna(global_precision)
        out["raw_selection_score"] = out["selection_score"]
        out["calibration_multiplier"] = (
            0.5
            + out["symbol_side_precision"].clip(lower=0, upper=1)
            + 0.5 * out["regime_side_precision"].clip(lower=0, upper=1)
        )
        out["selection_score"] = out["raw_selection_score"] * out["calibration_multiplier"]
        return out

    return calibrate(valid), calibrate(target)


def apply_selection_rule(candidates: pd.DataFrame, rule: dict) -> pd.DataFrame:
    frame = candidates[
        candidates["side_score"].ge(rule["min_side_score"])
        & candidates["selection_score"].ge(rule["min_selection_score"])
        & candidates["score_margin"].ge(rule["min_score_margin"])
        & candidates["direction_confidence"].ge(rule["min_direction_confidence"])
    ].copy()
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            ["date", "selection_score", "side_score", "score_margin", "symbol"],
            ascending=[True, False, False, False, True],
        )
        .groupby("date")
        .head(int(rule["daily_top"]))
        .sort_values(["date", "selection_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def selection_metrics(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "calls": 0,
            "directional_precision": np.nan,
            "clean_event_any_rate": np.nan,
            "wrong_clean_direction_rate": np.nan,
            "avg_up_move_5d": np.nan,
            "avg_down_move_5d": np.nan,
            "avg_signed_move_5d": np.nan,
        }
    side_sign = np.where(selected["predicted_side"].eq("bull"), 1.0, -1.0)
    signed_move = np.where(
        selected["predicted_side"].eq("bull"),
        selected["up_move_5d"],
        selected["down_move_5d"],
    )
    close_move = side_sign * selected["fwd_return_5d"].fillna(0).to_numpy()
    return {
        "calls": int(len(selected)),
        "directional_precision": float(selected["direction_correct"].mean()),
        "clean_event_any_rate": float(selected["clean_event_any"].mean()),
        "wrong_clean_direction_rate": float(selected["wrong_clean_direction"].mean()),
        "avg_up_move_5d": float(selected["up_move_5d"].mean()),
        "avg_down_move_5d": float(selected["down_move_5d"].mean()),
        "avg_signed_move_5d": float(np.nanmean(signed_move)),
        "avg_signed_close_return_5d": float(np.nanmean(close_move)),
        "bull_calls": int(selected["predicted_side"].eq("bull").sum()),
        "bear_calls": int(selected["predicted_side"].eq("bear").sum()),
    }


def build_rules_from_validation(
    valid_candidates: pd.DataFrame,
    config: CleanDirectionConfig,
) -> tuple[pd.DataFrame, dict]:
    if valid_candidates.empty:
        fallback = {
            "daily_top": 1,
            "min_side_score": 1.0,
            "min_selection_score": 1.0,
            "min_score_margin": 1.0,
            "min_direction_confidence": 1.0,
        }
        return pd.DataFrame(), fallback

    side_scores = valid_candidates["side_score"].dropna()
    sel_scores = valid_candidates["selection_score"].dropna()
    margins = valid_candidates["score_margin"].dropna()
    confs = valid_candidates["direction_confidence"].dropna()
    rows = []
    for daily_top in (1, 2, 3, 5, 10, 30):
        for side_q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
            for sel_q in (0.50, 0.70, 0.80, 0.90, 0.95):
                for margin_q in (0.20, 0.50, 0.70, 0.85):
                    for conf_q in (0.20, 0.50, 0.70):
                        rule = {
                            "daily_top": daily_top,
                            "min_side_score": float(side_scores.quantile(side_q)),
                            "min_selection_score": float(sel_scores.quantile(sel_q)),
                            "min_score_margin": float(margins.quantile(margin_q)),
                            "min_direction_confidence": float(confs.quantile(conf_q)),
                        }
                        selected = apply_selection_rule(valid_candidates, rule)
                        metrics = selection_metrics(selected)
                        rows.append(
                            {
                                **rule,
                                **metrics,
                                "side_score_quantile": side_q,
                                "selection_score_quantile": sel_q,
                                "margin_quantile": margin_q,
                                "confidence_quantile": conf_q,
                            }
                        )
    grid = pd.DataFrame(rows)
    viable = grid[
        (grid["calls"] >= config.min_validation_calls)
        & (grid["directional_precision"] >= config.precision_floor)
    ].copy()
    if viable.empty:
        viable = grid[grid["calls"] >= max(5, config.min_validation_calls // 2)].copy()
    if viable.empty:
        viable = grid.copy()
    chosen = viable.sort_values(
        [
            "directional_precision",
            "calls",
            "wrong_clean_direction_rate",
            "avg_signed_move_5d",
        ],
        ascending=[False, False, True, False],
    ).iloc[0]
    rule = {
        "daily_top": int(chosen["daily_top"]),
        "min_side_score": float(chosen["min_side_score"]),
        "min_selection_score": float(chosen["min_selection_score"]),
        "min_score_margin": float(chosen["min_score_margin"]),
        "min_direction_confidence": float(chosen["min_direction_confidence"]),
    }
    return grid, rule


def topk_diagnostics(candidates: pd.DataFrame, year: int, split: str) -> pd.DataFrame:
    rows = []
    ranked = candidates.sort_values("selection_score", ascending=False)
    for side in ("all", "bull", "bear"):
        subset = ranked if side == "all" else ranked[ranked["predicted_side"].eq(side)]
        for k in (25, 50, 100, 150, 200, 300, 500):
            selected = subset.head(k)
            metrics = selection_metrics(selected)
            rows.append({"year": year, "split": split, "side": side, "top_k": k, **metrics})
    return pd.DataFrame(rows)


def run_clean_direction(
    dataset_path: Path,
    config: CleanDirectionConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"clean_direction_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    symbols = liquid30_symbols()
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = add_liquid30_features(df, symbols)
    df = add_sector_features(df, symbols)
    df = add_label_validity_flags(df, config.max_weekly_abs_move)
    df = add_clean_direction_labels(
        df,
        config.target_pct,
        config.adverse_limit,
        config.max_weekly_abs_move,
    )
    df["is_liquid30"] = df["symbol"].isin(symbols)
    model_df = df.copy() if config.train_universe == "all" else df[df["is_liquid30"]].copy()
    eval_df = df[df["is_liquid30"]].copy()

    features = feature_columns(df)
    features = [
        col
        for col in features
        if col
        not in {
            "liquid30_symbol_code",
            "clean_episode_start",
            "clean_episode_age",
            "price_discontinuity",
            "corp_price_action",
            "invalid_price_event",
        }
    ] + ["liquid30_symbol_code"]
    model_df[features] = model_df[features].replace([np.inf, -np.inf], np.nan)
    eval_df[features] = eval_df[features].replace([np.inf, -np.inf], np.nan)
    model_df["sample_weight"] = np.where(model_df["is_liquid30"], config.liquid_weight, 1.0)
    continuation = model_df["label_clean_any_4pct_5d"].eq(1) & ~model_df["clean_episode_start"]
    model_df.loc[continuation, "sample_weight"] *= config.episode_decay_weight
    eval_df["sample_weight"] = np.where(eval_df["is_liquid30"], config.liquid_weight, 1.0)

    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "raw_universe": LIQUID30_RAW,
        "canonical_universe": symbols,
        "feature_count": len(features),
        "features": features,
        "labels": {
            "bull": f"up_move_{HORIZON_DAYS}d > {config.target_pct} and down_move_{HORIZON_DAYS}d < {config.adverse_limit}",
            "bear": f"down_move_{HORIZON_DAYS}d > {config.target_pct} and up_move_{HORIZON_DAYS}d < {config.adverse_limit}",
            "event": "bull or bear clean event",
        },
        "model_stack": "LGBM + CatBoost event model, LGBM + CatBoost event-direction model, validation-only symbol/regime calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    all_scored = []
    all_candidates = []
    all_selected = []
    rules = []
    side_metrics = []
    topk_rows = []
    validation_grids = []
    label_rows = []

    for year in range(config.start_test_year, config.end_test_year + 1):
        train_end = train_end_for_year(year, config)
        train = model_df[
            model_df["date"].dt.year.ge(config.train_start_year)
            & model_df["date"].le(train_end)
            & pd.to_datetime(model_df[f"future_{HORIZON_DAYS}d_date"]).le(train_end)
        ].dropna(
            subset=[
                "label_clean_bull_4pct_5d",
                "label_clean_bear_4pct_5d",
                "label_clean_any_4pct_5d",
            ]
        ).copy()
        valid_cutoff = train["date"].max() - pd.Timedelta(days=config.validation_days)
        inner = train[train["date"] < valid_cutoff].copy()
        valid = train[train["date"] >= valid_cutoff].copy()
        valid_eval = eval_df[
            eval_df["date"].ge(valid_cutoff) & eval_df["date"].le(train_end)
        ].dropna(
            subset=[
                "label_clean_bull_4pct_5d",
                "label_clean_bear_4pct_5d",
                "label_clean_any_4pct_5d",
            ]
        ).copy()
        test = eval_df[eval_df["date"].dt.year.eq(year)].dropna(
            subset=[
                "label_clean_bull_4pct_5d",
                "label_clean_bear_4pct_5d",
                "label_clean_any_4pct_5d",
            ]
        ).copy()

        for split_name, frame in (("train", train), ("valid_eval", valid_eval), ("test", test)):
            label_rows.append(
                {
                    "year": year,
                    "split": split_name,
                    "rows": len(frame),
                    "symbols": frame["symbol"].nunique(),
                    "bull": int(frame["label_clean_bull_4pct_5d"].sum()),
                    "bear": int(frame["label_clean_bear_4pct_5d"].sum()),
                    "bull_rate": float(frame["label_clean_bull_4pct_5d"].mean()),
                    "bear_rate": float(frame["label_clean_bear_4pct_5d"].mean()),
                }
            )

        models = {}
        lgb_event = fit_side_model(
            inner,
            valid,
            features,
            "label_clean_any_4pct_5d",
            side="event",
            seed=42 + year,
            weight_col="sample_weight",
        )
        direction_inner = inner[inner["label_clean_any_4pct_5d"].eq(1)].dropna(
            subset=["label_clean_direction_bull_4pct_5d"]
        )
        direction_valid = valid[valid["label_clean_any_4pct_5d"].eq(1)].dropna(
            subset=["label_clean_direction_bull_4pct_5d"]
        )
        lgb_direction = fit_side_model(
            direction_inner,
            direction_valid,
            features,
            "label_clean_direction_bull_4pct_5d",
            side="direction",
            seed=1042 + year,
            weight_col="sample_weight",
        )
        models["lgb_event"] = lgb_event
        models["lgb_direction"] = lgb_direction
        models["lgb_bull_direct"] = fit_side_model(
            inner,
            valid,
            features,
            "label_clean_bull_4pct_5d",
            side="bull",
            seed=4042 + year,
            weight_col="sample_weight",
        )
        models["lgb_bear_direct"] = fit_side_model(
            inner,
            valid,
            features,
            "label_clean_bear_4pct_5d",
            side="bear",
            seed=5042 + year,
            weight_col="sample_weight",
        )

        if config.use_catboost:
            models["cat_event"] = fit_catboost_model(
                inner,
                valid,
                features,
                "label_clean_any_4pct_5d",
                seed=2042 + year,
                weight_col="sample_weight",
            )
            models["cat_direction"] = fit_catboost_model(
                direction_inner,
                direction_valid,
                features,
                "label_clean_direction_bull_4pct_5d",
                seed=3042 + year,
                weight_col="sample_weight",
            )
            models["cat_bull_direct"] = fit_catboost_model(
                inner,
                valid,
                features,
                "label_clean_bull_4pct_5d",
                seed=6042 + year,
                weight_col="sample_weight",
            )
            models["cat_bear_direct"] = fit_catboost_model(
                inner,
                valid,
                features,
                "label_clean_bear_4pct_5d",
                seed=7042 + year,
                weight_col="sample_weight",
            )

        for name, model in models.items():
            model_id = f"{run_id}_{year}_{name}"
            model_path = run_dir / "models" / f"{model_id}.txt"
            if isinstance(model, lgb.Booster):
                model.save_model(model_path)
                best_iteration = model.best_iteration
                model_file = model_path.name
            else:
                model_path = run_dir / "models" / f"{model_id}.cbm"
                model.save_model(str(model_path))
                best_iteration = model.get_best_iteration()
                model_file = model_path.name
            meta = {
                "model_id": model_id,
                "year": year,
                "model": name,
                "train_end": str(train_end.date()),
                "label_col": "label_clean_any_4pct_5d"
                if name == "event"
                else "label_clean_direction_bull_4pct_5d",
                "train_rows": len(train),
                "inner_rows": len(inner),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "best_iteration": best_iteration,
                "features": features,
                "model_file": model_file,
            }
            (run_dir / "models" / f"{model_id}.json").write_text(
                json.dumps(meta, indent=2, default=str),
                encoding="utf-8",
            )

        for frame in (valid_eval, test):
            frame["event_lgb_score"] = predict_lgbm(models["lgb_event"], frame, features)
            frame["direction_lgb_bull_score"] = predict_lgbm(models["lgb_direction"], frame, features)
            frame["bull_direct_lgb_score"] = predict_lgbm(models["lgb_bull_direct"], frame, features)
            frame["bear_direct_lgb_score"] = predict_lgbm(models["lgb_bear_direct"], frame, features)
            if config.use_catboost:
                frame["event_cat_score"] = predict_catboost(models["cat_event"], frame, features)
                frame["direction_cat_bull_score"] = predict_catboost(
                    models["cat_direction"], frame, features
                )
                frame["bull_direct_cat_score"] = predict_catboost(
                    models["cat_bull_direct"], frame, features
                )
                frame["bear_direct_cat_score"] = predict_catboost(
                    models["cat_bear_direct"], frame, features
                )
                w = config.ensemble_lgbm_weight
                frame["event_score"] = (
                    w * frame["event_lgb_score"] + (1.0 - w) * frame["event_cat_score"]
                )
                frame["direction_bull_score"] = (
                    w * frame["direction_lgb_bull_score"]
                    + (1.0 - w) * frame["direction_cat_bull_score"]
                )
                frame["bull_direct_score"] = (
                    w * frame["bull_direct_lgb_score"] + (1.0 - w) * frame["bull_direct_cat_score"]
                )
                frame["bear_direct_score"] = (
                    w * frame["bear_direct_lgb_score"] + (1.0 - w) * frame["bear_direct_cat_score"]
                )
            else:
                frame["event_score"] = frame["event_lgb_score"]
                frame["direction_bull_score"] = frame["direction_lgb_bull_score"]
                frame["bull_direct_score"] = frame["bull_direct_lgb_score"]
                frame["bear_direct_score"] = frame["bear_direct_lgb_score"]
            event_bull = frame["event_score"] * frame["direction_bull_score"]
            event_bear = frame["event_score"] * (1.0 - frame["direction_bull_score"])
            direct_weight = config.direct_side_weight
            frame["bull_score"] = (
                (1.0 - direct_weight) * event_bull + direct_weight * frame["bull_direct_score"]
            )
            frame["bear_score"] = (
                (1.0 - direct_weight) * event_bear + direct_weight * frame["bear_direct_score"]
            )
            frame["model_year"] = year

        for split_name, frame in (("valid", valid_eval), ("test", test)):
            event_metrics = score_side_metrics(frame, "label_clean_any_4pct_5d", "event_score")
            side_metrics.append({"year": year, "split": split_name, "side": "event", **event_metrics})
            for side, label_col in (
                ("bull", "label_clean_bull_4pct_5d"),
                ("bear", "label_clean_bear_4pct_5d"),
            ):
                metrics = score_side_metrics(frame, label_col, f"{side}_score")
                side_metrics.append({"year": year, "split": split_name, "side": side, **metrics})

        valid_candidates = make_side_candidates(valid_eval)
        test_candidates = make_side_candidates(test)
        valid_candidates, test_candidates = apply_symbol_regime_calibration(
            valid_candidates,
            test_candidates,
        )
        grid, rule = build_rules_from_validation(valid_candidates, config)
        if not grid.empty:
            grid["year"] = year
            validation_grids.append(grid)

        valid_selected = apply_selection_rule(valid_candidates, rule)
        test_selected = apply_selection_rule(test_candidates, rule)
        valid_rule_metrics = selection_metrics(valid_selected)
        test_rule_metrics = selection_metrics(test_selected)
        rules.append(
            {
                "year": year,
                **rule,
                **{f"valid_{k}": v for k, v in valid_rule_metrics.items()},
                **{f"test_{k}": v for k, v in test_rule_metrics.items()},
            }
        )

        valid_candidates["year"] = year
        valid_candidates["split"] = "valid"
        test_candidates["year"] = year
        test_candidates["split"] = "test"
        valid_selected["year"] = year
        valid_selected["split"] = "valid_selected"
        test_selected["year"] = year
        test_selected["split"] = "test_selected"
        test["year"] = year

        all_scored.append(test)
        all_candidates.extend([valid_candidates, test_candidates])
        all_selected.extend([valid_selected, test_selected])
        topk_rows.append(topk_diagnostics(valid_candidates, year, "valid"))
        topk_rows.append(topk_diagnostics(test_candidates, year, "test"))

    scored = pd.concat(all_scored, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    rules_df = pd.DataFrame(rules)
    side_metrics_df = pd.DataFrame(side_metrics)
    topk_df = pd.concat(topk_rows, ignore_index=True)
    label_summary = pd.DataFrame(label_rows)
    validation_grid = pd.concat(validation_grids, ignore_index=True) if validation_grids else pd.DataFrame()

    run_stamp = run_dir.name
    scored.to_parquet(run_dir / "predictions" / f"{run_stamp}_scored_rows.parquet", index=False)
    candidates.to_parquet(run_dir / "predictions" / f"{run_stamp}_side_candidates.parquet", index=False)
    selected.to_csv(run_dir / "predictions" / f"{run_stamp}_selected_calls.csv", index=False)
    rules_df.to_csv(run_dir / "reports" / f"{run_stamp}_yearly_rules.csv", index=False)
    side_metrics_df.to_csv(run_dir / "reports" / f"{run_stamp}_side_model_metrics.csv", index=False)
    topk_df.to_csv(run_dir / "reports" / f"{run_stamp}_topk_diagnostics.csv", index=False)
    label_summary.to_csv(run_dir / "reports" / f"{run_stamp}_label_summary.csv", index=False)
    validation_grid.to_csv(run_dir / "reports" / f"{run_stamp}_validation_rule_grid.csv", index=False)

    test_selected = selected[selected["split"].eq("test_selected")].copy()
    yearly = (
        test_selected.groupby("year")
        .apply(lambda g: pd.Series(selection_metrics(g)), include_groups=False)
        .reset_index()
    )
    overall = pd.DataFrame([{"split": "test_selected", **selection_metrics(test_selected)}])
    yearly.to_csv(run_dir / "reports" / f"{run_stamp}_test_selected_yearly.csv", index=False)
    overall.to_csv(run_dir / "reports" / f"{run_stamp}_test_selected_overall.csv", index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(PREDICTIONS_DIR / "clean_direction_scored_latest.parquet", index=False)
    candidates.to_parquet(PREDICTIONS_DIR / "clean_direction_candidates_latest.parquet", index=False)
    selected.to_csv(PREDICTIONS_DIR / "clean_direction_selected_latest.csv", index=False)
    rules_df.to_csv(REPORTS_DIR / "clean_direction_yearly_rules_latest.csv", index=False)
    side_metrics_df.to_csv(REPORTS_DIR / "clean_direction_side_model_metrics_latest.csv", index=False)
    topk_df.to_csv(REPORTS_DIR / "clean_direction_topk_diagnostics_latest.csv", index=False)
    label_summary.to_csv(REPORTS_DIR / "clean_direction_label_summary_latest.csv", index=False)
    yearly.to_csv(REPORTS_DIR / "clean_direction_test_selected_yearly_latest.csv", index=False)
    overall.to_csv(REPORTS_DIR / "clean_direction_test_selected_overall_latest.csv", index=False)
    return run_dir
