from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import BACKTEST_DIR, DEFAULT_COST_BPS, DEFAULT_TOP_N, HORIZON_DAYS


def _prediction_col(frame: pd.DataFrame) -> str:
    values = frame["pred_col"].dropna().unique()
    if len(values) != 1:
        raise ValueError("Each bucket frame must contain exactly one prediction column")
    return str(values[0])


def select_daily_top_n(predictions: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    selections = []
    for (_, _, _), bucket in predictions.groupby(["date", "side", "threshold"], sort=True):
        pred_col = _prediction_col(bucket)
        top = bucket.sort_values(pred_col, ascending=False).head(top_n).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top["score"] = top[pred_col]
        selections.append(top)
    if not selections:
        return pd.DataFrame()
    return pd.concat(selections, ignore_index=True)


def score_trades(
    selections: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    trades = selections.copy()
    direction = np.where(trades["side"].eq("down"), -1.0, 1.0)
    trades["gross_return"] = direction * trades[f"fwd_return_{HORIZON_DAYS}d"]
    trades["cost_return"] = cost_bps / 10000.0
    trades["net_return"] = trades["gross_return"] - trades["cost_return"]
    trades["hit"] = trades.apply(lambda row: row[row["label_col"]] == 1, axis=1)
    return trades


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (side, threshold), group in trades.groupby(["side", "threshold"], sort=True):
        daily = group.groupby("date")["net_return"].mean().sort_index()
        equity = (1.0 + daily.fillna(0.0)).cumprod()
        cumulative = equity - 1.0
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "side": side,
                "threshold": threshold,
                "trades": len(group),
                "days": group["date"].nunique(),
                "hit_rate": group["hit"].mean(),
                "avg_score": group["score"].mean(),
                "avg_gross_return": group["gross_return"].mean(),
                "avg_net_return": group["net_return"].mean(),
                "median_net_return": group["net_return"].median(),
                "win_rate_net": group["net_return"].gt(0).mean(),
                "total_compounded_return": cumulative.iloc[-1] if len(cumulative) else np.nan,
                "max_drawdown": drawdown.min() if len(drawdown) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    predictions: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    cost_bps: float = DEFAULT_COST_BPS,
    output_dir: Path = BACKTEST_DIR,
    name: str = "backtest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = select_daily_top_n(predictions, top_n=top_n)
    trades = score_trades(selections, cost_bps=cost_bps)
    summary = summarize_trades(trades)

    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output_dir / f"{name}_trades.parquet", index=False)
    summary.to_csv(output_dir / f"{name}_summary.csv", index=False)
    return trades, summary
