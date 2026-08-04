from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGNAL_LABELS = {"GO+", "GO", "RGO+", "XGO+"}


@dataclass(frozen=True)
class FreshnessSelectionConfig:
    weekly_symbol_cap: int = 2
    same_signal_cooling_days: int = 3
    max_weekly_signals: int | None = 10
    rank_score_col: str = "fresh_rank_score"


def signal_label_mask(series: pd.Series) -> pd.Series:
    labels = series.fillna("").astype(str)
    return labels.isin(SIGNAL_LABELS) | labels.str.contains("RGO+", regex=False) | labels.str.contains("XGO+", regex=False)


def signal_family(label: object, side: object) -> str:
    text = str(label or "")
    side_text = str(side or "")
    if "RGO+" in text:
        return f"{side_text}:RGO+"
    if "GO+" in text:
        return f"{side_text}:GO+"
    if text == "GO":
        return f"{side_text}:GO"
    return f"{side_text}:WATCH"


def add_repeat_context(
    frame: pd.DataFrame,
    *,
    signal_col: str = "go_label",
    rank_col: str = "trade_quality_score",
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if signal_col not in out.columns:
        out[signal_col] = "WATCH"
    out["is_visible_signal"] = signal_label_mask(out[signal_col])
    out["signal_family"] = [
        signal_family(label, side)
        for label, side in zip(out[signal_col], out.get("side", pd.Series("", index=out.index)))
    ]
    sort_cols = ["symbol", "side", "signal_family", "date"]
    out = out.sort_values(sort_cols).copy()
    signal_dates = out["date"].where(out["is_visible_signal"])
    prev_signal_date = signal_dates.groupby([out["symbol"], out["side"], out["signal_family"]]).transform(
        lambda s: s.ffill().shift(1)
    )
    out["days_since_same_signal"] = (out["date"] - prev_signal_date).dt.days

    symbol_signal_dates = out["date"].where(out["is_visible_signal"])
    prev_symbol_signal_date = symbol_signal_dates.groupby(out["symbol"]).transform(lambda s: s.ffill().shift(1))
    out["days_since_symbol_signal"] = (out["date"] - prev_symbol_signal_date).dt.days

    by_symbol_side = out.groupby(["symbol", "side"], group_keys=False)
    recent_signal = out["is_visible_signal"].astype(int)
    out["same_side_signal_count_5d"] = by_symbol_side["is_visible_signal"].transform(
        lambda s: s.astype(int).shift(1).rolling(5, min_periods=1).sum()
    )
    out["same_side_signal_count_10d"] = by_symbol_side["is_visible_signal"].transform(
        lambda s: s.astype(int).shift(1).rolling(10, min_periods=1).sum()
    )
    out["symbol_signal_count_5d"] = recent_signal.groupby(out["symbol"]).transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    out["symbol_signal_count_10d"] = recent_signal.groupby(out["symbol"]).transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )

    rank = pd.to_numeric(out.get(rank_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["freshness_penalty"] = 0.0
    out["freshness_penalty"] += out["days_since_same_signal"].le(3).fillna(False).astype(float) * 0.18
    out["freshness_penalty"] += out["same_side_signal_count_5d"].fillna(0).clip(0, 3) * 0.05
    out["freshness_penalty"] += out["symbol_signal_count_10d"].fillna(0).clip(0, 5) * 0.025
    out["freshness_bonus"] = 0.0
    out["freshness_bonus"] += out["days_since_same_signal"].gt(10).fillna(True).astype(float) * 0.08
    out["freshness_bonus"] += out["days_since_symbol_signal"].gt(10).fillna(True).astype(float) * 0.05
    out["fresh_rank_score"] = rank.fillna(0.0) + out["freshness_bonus"] - out["freshness_penalty"]
    return out.sort_values(["date", "symbol", "side", "signal_family"]).reset_index(drop=True)


def apply_final_shortlist_policy(
    frame: pd.DataFrame,
    config: FreshnessSelectionConfig = FreshnessSelectionConfig(),
) -> pd.DataFrame:
    out = add_repeat_context(frame, rank_col=config.rank_score_col if config.rank_score_col in frame.columns else "trade_quality_score")
    if out.empty:
        return out
    rank_col = config.rank_score_col if config.rank_score_col in out.columns else "fresh_rank_score"
    out["final_shortlist_signal"] = False
    out["final_shortlist_rank"] = np.nan
    out["freshness_filter_reason"] = ""
    selected_indices: list[int] = []

    for week, week_frame in out[out["is_visible_signal"]].groupby(out["date"].dt.to_period("W-FRI"), sort=True):
        symbol_counts: dict[str, int] = {}
        last_signal_date: dict[tuple[str, str, str], pd.Timestamp] = {}
        week_selected = 0
        ordered = week_frame.sort_values([rank_col, "trade_quality_score", "score"], ascending=[False, False, False])
        for idx, row in ordered.iterrows():
            if config.max_weekly_signals is not None and week_selected >= config.max_weekly_signals:
                out.loc[idx, "freshness_filter_reason"] = "weekly_signal_limit"
                continue
            symbol = str(row["symbol"])
            family = str(row["signal_family"])
            side = str(row.get("side", ""))
            key = (symbol, side, family)
            date = pd.Timestamp(row["date"]).normalize()
            if symbol_counts.get(symbol, 0) >= config.weekly_symbol_cap:
                out.loc[idx, "freshness_filter_reason"] = "weekly_symbol_cap"
                continue
            prev_date = last_signal_date.get(key)
            if prev_date is not None and (date - prev_date).days <= config.same_signal_cooling_days:
                out.loc[idx, "freshness_filter_reason"] = "same_signal_cooling"
                continue
            selected_indices.append(idx)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            last_signal_date[key] = date
            week_selected += 1

    if selected_indices:
        out.loc[selected_indices, "final_shortlist_signal"] = True
        selected = out.loc[selected_indices].sort_values(["date", rank_col], ascending=[True, False])
        out.loc[selected.index, "final_shortlist_rank"] = selected.groupby("date").cumcount() + 1
    out.loc[out["is_visible_signal"] & ~out["final_shortlist_signal"] & out["freshness_filter_reason"].eq(""), "freshness_filter_reason"] = "lower_ranked_repeat"
    return out.sort_values(["date", rank_col], ascending=[True, False]).reset_index(drop=True)
