from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import PREDICTIONS_DIR, REPORTS_DIR, RUNS_DIR, TOP30_LIQUID_UNIVERSE
from koscine.training import feature_columns


HORIZON_DAYS = 5
COST = 0.002


@dataclass(frozen=True)
class Top30TradeQualityConfig:
    train_start_year: int = 2012
    start_test_year: int = 2018
    end_test_year: int = 2025
    target_pct: float = 0.04
    stop_pct: float = 0.03
    max_hold_days: int = HORIZON_DAYS
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    min_trades: int = 100


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def add_top30_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    top_mask = out["symbol"].isin(TOP30_LIQUID_UNIVERSE)
    top = out[top_mask].copy()

    breadth = (
        top.groupby("date")
        .agg(
            top30_breadth_ret1_pos=("ret_1d", lambda s: (s > 0).mean()),
            top30_breadth_ret3_pos=("ret_3d", lambda s: (s > 0).mean()),
            top30_breadth_ret5_pos=("ret_5d", lambda s: (s > 0).mean()),
            top30_breadth_ret20_pos=("ret_20d", lambda s: (s > 0).mean()),
            top30_median_ret1=("ret_1d", "median"),
            top30_median_ret5=("ret_5d", "median"),
            top30_median_ret20=("ret_20d", "median"),
            top30_avg_atr_rank=("atr_pct_14_cs_rank", "mean"),
            top30_avg_vol_rank=("vol_sma20_ratio_cs_rank", "mean"),
            top30_avg_bb_rank=("bb_width_20_cs_rank", "mean"),
        )
        .reset_index()
    )
    out = out.merge(breadth, on="date", how="left")

    top_rank_cols = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_sma20_ratio",
        "atr_pct_14",
        "bb_width_20",
        "turnover_ratio_20",
        "close_sma20_dist",
        "close_sma50_dist",
    ]
    for col in top_rank_cols:
        if col in out:
            out[f"{col}_top30_rank"] = np.nan
            ranks = out.loc[top_mask].groupby("date")[col].rank(pct=True)
            out.loc[top_mask, f"{col}_top30_rank"] = ranks

    grouped = out.groupby("symbol", group_keys=False)
    green = out["close"].gt(out.groupby("symbol")["close"].shift(1)).astype(float)
    red = out["close"].lt(out.groupby("symbol")["close"].shift(1)).astype(float)
    out["green_count_5"] = green.groupby(out["symbol"]).transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    out["red_count_5"] = red.groupby(out["symbol"]).transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    for window in (10, 20, 50):
        out[f"dist_high_{window}"] = out["close"] / grouped["high"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).max()
        ) - 1.0
        out[f"dist_low_{window}"] = out["close"] / grouped["low"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).min()
        ) - 1.0

    symbol_codes = {symbol: i for i, symbol in enumerate(sorted(TOP30_LIQUID_UNIVERSE))}
    out["top30_symbol_code"] = out["symbol"].map(symbol_codes).astype(float)
    return out


def add_trade_path_labels(
    df: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    label_cols = [
        "tq_entry_date",
        "tq_entry_open",
        "tq_exit_date",
        "tq_exit_price",
        "tq_exit_reason",
        "tq_bars_held",
        "tq_gross_return",
        "tq_net_return",
        "tq_clean_target",
        "tq_bad",
        "tq_positive_timeout",
        "tq_relevance",
    ]
    for col in label_cols:
        out[col] = np.nan
    for col in ("tq_entry_date", "tq_exit_date", "tq_exit_reason"):
        out[col] = None

    for _, idxs in out.groupby("symbol").groups.items():
        positions = list(idxs)
        g = out.loc[positions].reset_index()
        for pos in range(len(g)):
            if pos + 1 >= len(g):
                continue
            entry = float(g.loc[pos + 1, "open"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            target_px = entry * (1.0 + target_pct)
            stop_px = entry * (1.0 - stop_pct)
            last_pos = min(pos + max_hold_days, len(g) - 1)
            exit_reason = None
            exit_price = np.nan
            exit_date = pd.NaT
            bars_held = 0
            for future_pos in range(pos + 1, last_pos + 1):
                day = g.loc[future_pos]
                bars_held = future_pos - pos
                if float(day["low"]) <= stop_px:
                    exit_reason = "stop"
                    exit_price = stop_px
                    exit_date = day["date"]
                    break
                if float(day["high"]) >= target_px:
                    exit_reason = "target"
                    exit_price = target_px
                    exit_date = day["date"]
                    break
            if exit_reason is None:
                day = g.loc[last_pos]
                bars_held = last_pos - pos
                exit_price = float(day["close"])
                exit_date = day["date"]
                exit_reason = "timeout"

            gross = exit_price / entry - 1.0
            net = gross - COST
            clean_target = exit_reason == "target"
            positive_timeout = exit_reason == "timeout" and gross >= 0
            bad = exit_reason == "stop" or (exit_reason == "timeout" and gross < 0)
            relevance = 2 if clean_target else (1 if positive_timeout else 0)
            orig_index = int(g.loc[pos, "index"])
            out.loc[orig_index, "tq_entry_date"] = g.loc[pos + 1, "date"]
            out.loc[orig_index, "tq_entry_open"] = entry
            out.loc[orig_index, "tq_exit_date"] = exit_date
            out.loc[orig_index, "tq_exit_price"] = exit_price
            out.loc[orig_index, "tq_exit_reason"] = exit_reason
            out.loc[orig_index, "tq_bars_held"] = bars_held
            out.loc[orig_index, "tq_gross_return"] = gross
            out.loc[orig_index, "tq_net_return"] = net
            out.loc[orig_index, "tq_clean_target"] = clean_target
            out.loc[orig_index, "tq_bad"] = bad
            out.loc[orig_index, "tq_positive_timeout"] = positive_timeout
            out.loc[orig_index, "tq_relevance"] = relevance
    out["tq_entry_date"] = pd.to_datetime(out["tq_entry_date"])
    out["tq_exit_date"] = pd.to_datetime(out["tq_exit_date"])
    for col in ("tq_clean_target", "tq_bad", "tq_positive_timeout"):
        out[col] = out[col].astype("boolean")
    out["tq_relevance"] = pd.to_numeric(out["tq_relevance"], errors="coerce")
    return out


def lgbm_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 350,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 20.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }


def fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    validation_days: int,
    positive_weight: float = 1.0,
) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    inner[label_col] = inner[label_col].astype(int)
    valid[label_col] = valid[label_col].astype(int)
    inner_weight = np.where(inner[label_col].astype(bool), positive_weight, 1.0)
    valid_weight = np.where(valid[label_col].astype(bool), positive_weight, 1.0)
    train_set = lgb.Dataset(inner[features], label=inner[label_col], weight=inner_weight)
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], weight=valid_weight, reference=train_set)
    return lgb.train(
        lgbm_params(inner[label_col]),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def fit_return_model(
    train: pd.DataFrame,
    features: list[str],
    validation_days: int,
) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 350,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 20.0,
        "verbosity": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(inner[features], label=inner["tq_net_return"])
    valid_set = lgb.Dataset(valid[features], label=valid["tq_net_return"], reference=train_set)
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def train_end_for_year(year: int, config: Top30TradeQualityConfig) -> pd.Timestamp:
    return pd.Timestamp(year=year - 1, month=config.train_cutoff_month, day=config.train_cutoff_day)


def metric_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "sum_return": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "target_rate": np.nan,
            "stop_rate": np.nan,
            "timeout_rate": np.nan,
            "avg_bars": np.nan,
        }
    gross_profit = trades.loc[trades["tq_net_return"] > 0, "tq_net_return"].sum()
    gross_loss = -trades.loc[trades["tq_net_return"] < 0, "tq_net_return"].sum()
    equity = (1.0 + trades["tq_net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "trades": len(trades),
        "win_rate": float(trades["tq_net_return"].gt(0).mean()),
        "avg_return": float(trades["tq_net_return"].mean()),
        "sum_return": float(trades["tq_net_return"].sum()),
        "compounded_return": float(equity.iloc[-1] - 1.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "max_drawdown": float(drawdown.min()),
        "target_rate": float(trades["tq_exit_reason"].eq("target").mean()),
        "stop_rate": float(trades["tq_exit_reason"].eq("stop").mean()),
        "timeout_rate": float(trades["tq_exit_reason"].eq("timeout").mean()),
        "avg_bars": float(trades["tq_bars_held"].mean()),
    }


def select_trades(
    predictions: pd.DataFrame,
    score_col: str,
    daily_top: int,
    min_score: float | None,
) -> pd.DataFrame:
    frame = predictions[predictions[score_col].notna()].copy()
    if min_score is not None:
        frame = frame[frame[score_col].ge(min_score)]
    selected = (
        frame.sort_values(["date", score_col, "clean_score", "symbol"], ascending=[True, False, False, True])
        .groupby("date")
        .head(daily_top)
        .copy()
    )
    selected["selection_score_col"] = score_col
    selected["selection_score"] = selected[score_col]
    selected["selection_daily_top"] = daily_top
    selected["selection_min_score"] = min_score
    return selected.sort_values("tq_entry_date")


def run_top30_trade_quality(
    dataset_path: Path,
    config: Top30TradeQualityConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"top30_trade_quality_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = add_top30_features(df)
    df = add_trade_path_labels(df, config.target_pct, config.stop_pct, config.max_hold_days)
    df = df[df["symbol"].isin(TOP30_LIQUID_UNIVERSE)].copy()
    features = feature_columns(df)
    features = [
        col
        for col in features
        if not col.startswith("tq_") and col not in {"top30_symbol_code"}
    ] + ["top30_symbol_code"]

    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": "entry=t+1 open, target/stop/timeout with stop-first same-candle rule",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    scored_years = []
    run_stamp = run_dir.name
    for year in range(config.start_test_year, config.end_test_year + 1):
        train_end = train_end_for_year(year, config)
        train = df[
            df["date"].dt.year.ge(config.train_start_year)
            & df["date"].le(train_end)
            & pd.to_datetime(df["tq_exit_date"]).le(train_end)
        ].dropna(subset=["tq_clean_target", "tq_bad", "tq_net_return"]).copy()
        test = df[df["date"].dt.year.eq(year)].dropna(subset=["tq_clean_target", "tq_bad", "tq_net_return"]).copy()

        models = {
            "clean": fit_classifier(train, features, "tq_clean_target", config.validation_days, positive_weight=2.0),
            "bad": fit_classifier(train, features, "tq_bad", config.validation_days, positive_weight=2.5),
            "return": fit_return_model(train, features, config.validation_days),
        }
        for name, model in models.items():
            model_id = f"{run_stamp}_{year}_{name}"
            model_path = run_dir / "models" / f"{model_id}.txt"
            model.save_model(model_path)
            meta = {
                "model_id": model_id,
                "year": year,
                "train_end": str(train_end.date()),
                "model": name,
                "features": features,
                "best_iteration": model.best_iteration,
                "train_rows": len(train),
                "model_file": model_path.name,
            }
            (run_dir / "models" / f"{model_id}.json").write_text(
                json.dumps(meta, indent=2, default=str),
                encoding="utf-8",
            )

        test["clean_score"] = models["clean"].predict(test[features], num_iteration=models["clean"].best_iteration)
        test["bad_score"] = models["bad"].predict(test[features], num_iteration=models["bad"].best_iteration)
        test["return_score"] = models["return"].predict(test[features], num_iteration=models["return"].best_iteration)
        test["safe_score"] = 1.0 - test["bad_score"]
        test["quality_score"] = test["clean_score"] * (test["safe_score"] ** 2)
        test["return_quality_score"] = test["quality_score"] * test["return_score"].rank(pct=True)
        test["prediction_year"] = year
        scored_years.append(test)

    predictions = pd.concat(scored_years, ignore_index=True)
    pred_path = run_dir / "predictions" / f"{run_stamp}_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)

    rows = []
    selected_candidates = {}
    score_cols = ["clean_score", "quality_score", "return_quality_score", "return_score"]
    for score_col in score_cols:
        values = predictions[score_col].dropna()
        thresholds = [None] + [
            float(values.quantile(q))
            for q in (0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
        ]
        for daily_top in (1, 2, 3, 5, 10):
            for threshold in thresholds:
                selected = select_trades(predictions, score_col, daily_top, threshold)
                if len(selected) < config.min_trades:
                    continue
                summary = metric_summary(selected)
                yearly = selected.groupby(pd.to_datetime(selected["tq_entry_date"]).dt.year).apply(
                    lambda g: pd.Series(metric_summary(g)),
                    include_groups=False,
                )
                summary.update(
                    {
                        "score_col": score_col,
                        "daily_top": daily_top,
                        "min_score": threshold,
                        "years": int(yearly.index.nunique()),
                        "positive_years": int(yearly["sum_return"].gt(0).sum()),
                        "worst_year_sum_return": float(yearly["sum_return"].min()),
                    }
                )
                summary["objective"] = (
                    min(summary["profit_factor"], 5.0) * 2.0
                    + summary["sum_return"]
                    + summary["win_rate"]
                    + summary["positive_years"] / max(summary["years"], 1)
                    - abs(summary["max_drawdown"])
                )
                rows.append(summary)
                selected_candidates[(score_col, daily_top, threshold)] = selected

    grid = pd.DataFrame(rows)
    grid_path = run_dir / "reports" / f"{run_stamp}_strategy_grid.csv"
    grid.to_csv(grid_path, index=False)
    robust = grid[
        (grid["profit_factor"] >= 1.25)
        & (grid["positive_years"] >= np.maximum(4, grid["years"] - 2))
        & (grid["max_drawdown"] >= -0.5)
    ].copy()
    selected_grid = robust if not robust.empty else grid
    chosen = selected_grid.sort_values(
        ["objective", "trades"], ascending=[False, False]
    ).iloc[0]
    key = (
        chosen["score_col"],
        int(chosen["daily_top"]),
        None if pd.isna(chosen["min_score"]) else float(chosen["min_score"]),
    )
    chosen_trades = selected_candidates[key].copy()
    chosen_trades_path = run_dir / "predictions" / f"{run_stamp}_chosen_trades.csv"
    chosen_trades.to_csv(chosen_trades_path, index=False)

    yearly = chosen_trades.groupby(pd.to_datetime(chosen_trades["tq_entry_date"]).dt.year).apply(
        lambda g: pd.Series(metric_summary(g)),
        include_groups=False,
    ).reset_index(names="year")
    monthly = chosen_trades.assign(
        month=pd.to_datetime(chosen_trades["tq_entry_date"]).dt.to_period("M").astype(str)
    ).groupby("month").apply(lambda g: pd.Series(metric_summary(g)), include_groups=False).reset_index()
    summary = pd.DataFrame([chosen.to_dict()])
    summary.to_csv(run_dir / "reports" / f"{run_stamp}_chosen_summary.csv", index=False)
    yearly.to_csv(run_dir / "reports" / f"{run_stamp}_chosen_yearly.csv", index=False)
    monthly.to_csv(run_dir / "reports" / f"{run_stamp}_chosen_monthly.csv", index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_DIR / "top30_trade_quality_predictions_latest.parquet", index=False)
    chosen_trades.to_csv(PREDICTIONS_DIR / "top30_trade_quality_chosen_trades_latest.csv", index=False)
    grid.to_csv(REPORTS_DIR / "top30_trade_quality_strategy_grid_latest.csv", index=False)
    summary.to_csv(REPORTS_DIR / "top30_trade_quality_chosen_summary_latest.csv", index=False)
    yearly.to_csv(REPORTS_DIR / "top30_trade_quality_chosen_yearly_latest.csv", index=False)
    monthly.to_csv(REPORTS_DIR / "top30_trade_quality_chosen_monthly_latest.csv", index=False)
    config_out = {
        "name": "top30_trade_quality",
        "created_at": timestamp(),
        "run_dir": str(run_dir),
        "config": config.__dict__,
        "chosen": chosen.to_dict(),
        "prediction_file": str(PREDICTIONS_DIR / "top30_trade_quality_predictions_latest.parquet"),
        "chosen_trades_file": str(PREDICTIONS_DIR / "top30_trade_quality_chosen_trades_latest.csv"),
    }
    Path("configs/top30_trade_quality_strategy.json").write_text(
        json.dumps(config_out, indent=2, default=str),
        encoding="utf-8",
    )
    return run_dir
