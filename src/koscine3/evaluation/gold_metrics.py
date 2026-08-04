from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GoldMetricConfig:
    metric_contract_version: str = "gold_swing_v1"
    round_trip_cost: float = 0.003
    hit_near_bar: float = 0.60
    opposite_preferred_max: float = 0.20
    opposite_hard_max: float = 0.25


def add_time_slices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(str)
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out


def summarize_signals(
    signals: pd.DataFrame,
    group_cols: list[str] | None = None,
    config: GoldMetricConfig | None = None,
) -> pd.DataFrame:
    config = config or GoldMetricConfig()
    if signals.empty:
        return pd.DataFrame()
    df = add_time_slices(signals)
    if "selected" in df.columns:
        df = df[df["selected"]].copy()
    group_cols = group_cols or []

    rows: list[dict[str, object]] = []
    grouped = [(("ALL",), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        evaluated = group[group["status"].eq("evaluated")]
        evaluated_count = len(evaluated)
        calls = len(group)

        def rate(column: str) -> float:
            if evaluated_count == 0:
                return 0.0
            return float(evaluated[column].fillna(False).mean())

        row: dict[str, object] = {}
        if group_cols:
            row.update({col: value for col, value in zip(group_cols, key, strict=True)})
        else:
            row["slice"] = "ALL"
        row.update(
            {
                "calls": int(calls),
                "evaluated": int(evaluated_count),
                "pending": int(calls - evaluated_count),
                "hit_rate": rate("hit"),
                "near_rate": rate("near"),
                "hit_near_rate": rate("hit_or_near"),
                "opposite_rate": rate("opposite"),
                "small_rate": rate("small"),
                "avg_favorable_move": float(evaluated["favorable_move"].mean())
                if evaluated_count
                else 0.0,
                "median_favorable_move": float(evaluated["favorable_move"].median())
                if evaluated_count
                else 0.0,
                "avg_signed_close_return": float(evaluated["signed_close_return"].mean())
                if evaluated_count
                else 0.0,
                "avg_net_close_return": float(
                    evaluated["signed_close_return"].sub(config.round_trip_cost).mean()
                )
                if evaluated_count
                else 0.0,
                "unique_symbols": int(group["symbol"].nunique()),
                "max_symbol_share": float(group["symbol"].value_counts(normalize=True).max())
                if calls
                else 0.0,
                "metric_contract_version": config.metric_contract_version,
            }
        )
        row["passes_gold"] = bool(
            evaluated_count > 0
            and row["hit_near_rate"] >= config.hit_near_bar
            and row["opposite_rate"] <= config.opposite_hard_max
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_gold_report(signals: pd.DataFrame, config: GoldMetricConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or GoldMetricConfig()
    df = add_time_slices(signals)
    available = set(df.columns)
    report: dict[str, pd.DataFrame] = {"aggregate": summarize_signals(df, config=config)}
    for name, cols in {
        "year": ["year"],
        "quarter": ["quarter"],
        "month": ["month"],
        "side": ["side"],
        "band": ["band"],
        "symbol": ["symbol"],
        "model_id": ["model_id"],
        "selector_config_id": ["selector_config_id"],
    }.items():
        if set(cols).issubset(available):
            report[name] = summarize_signals(df, cols, config=config)
    daily_source = df[df["selected"]] if "selected" in df.columns else df
    daily = daily_source.groupby("date").size().reset_index(name="signals")
    report["daily_counts"] = daily
    return report
