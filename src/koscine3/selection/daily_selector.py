from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SelectorConfig:
    selector_id: str = "selector_v8_conflict_hist_rank_guard_max5"
    max_signals_per_day: int = 5
    target_min_signals_per_day: int = 2
    max_signals_per_side: int = 3
    cooldown_trading_days: int = 25
    min_p_hit_near: float = 0.62
    max_p_opposite: float = 0.61
    long_min_p_hit_near: float | None = 0.66
    long_max_p_opposite: float | None = 0.55
    short_min_p_hit_near: float | None = 0.62
    short_max_p_opposite: float | None = 0.52
    min_p_clean_success: float = 0.0
    min_probability_margin: float = 0.0
    min_utility_score: float = -0.70
    max_symbol_return_20d_rank_pct: float | None = 0.90
    long_max_hist_opposite_rate_63: float | None = 0.45
    resolve_symbol_side_conflicts: bool = True
    opposite_penalty: float = 1.5
    signed_close_bonus: float = 0.25


def add_utility_score(predictions: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    df = predictions.copy()
    success_probability = (
        df["p_clean_success"] if "p_clean_success" in df.columns else df["p_hit_near"]
    )
    df["probability_margin"] = df["p_hit_near"] - df["p_opposite"]
    df["utility_score"] = (
        success_probability * df["expected_favorable_move"]
        - config.opposite_penalty * df["p_opposite"]
        + config.signed_close_bonus * df["expected_signed_close_return"]
    )
    return df


def _min_p_hit_near_for_side(config: SelectorConfig, side: str) -> float:
    if side == "long" and config.long_min_p_hit_near is not None:
        return config.long_min_p_hit_near
    if side == "short" and config.short_min_p_hit_near is not None:
        return config.short_min_p_hit_near
    return config.min_p_hit_near


def _max_p_opposite_for_side(config: SelectorConfig, side: str) -> float:
    if side == "long" and config.long_max_p_opposite is not None:
        return config.long_max_p_opposite
    if side == "short" and config.short_max_p_opposite is not None:
        return config.short_max_p_opposite
    return config.max_p_opposite


def select_daily_signals(
    predictions: pd.DataFrame,
    config: SelectorConfig | None = None,
) -> pd.DataFrame:
    config = config or SelectorConfig()
    df = add_utility_score(predictions, config)
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selection_reason"] = "not_ranked"
    df["selector_config_id"] = config.selector_id

    side = df["side"].astype(str)
    min_p_hit_near = pd.Series(config.min_p_hit_near, index=df.index, dtype=float)
    max_p_opposite = pd.Series(config.max_p_opposite, index=df.index, dtype=float)
    if config.long_min_p_hit_near is not None:
        min_p_hit_near.loc[side.eq("long")] = config.long_min_p_hit_near
    if config.short_min_p_hit_near is not None:
        min_p_hit_near.loc[side.eq("short")] = config.short_min_p_hit_near
    if config.long_max_p_opposite is not None:
        max_p_opposite.loc[side.eq("long")] = config.long_max_p_opposite
    if config.short_max_p_opposite is not None:
        max_p_opposite.loc[side.eq("short")] = config.short_max_p_opposite

    candidate_mask = pd.Series(True, index=df.index)
    long_low_hit = side.eq("long") & df["p_hit_near"].lt(min_p_hit_near)
    short_low_hit = side.eq("short") & df["p_hit_near"].lt(min_p_hit_near)
    df.loc[long_low_hit, "selection_reason"] = "below_min_p_hit_near_long"
    df.loc[short_low_hit, "selection_reason"] = "below_min_p_hit_near_short"
    candidate_mask &= ~(long_low_hit | short_low_hit)

    if "p_clean_success" in df.columns:
        low_clean = candidate_mask & df["p_clean_success"].lt(config.min_p_clean_success)
        df.loc[low_clean, "selection_reason"] = "below_min_p_clean_success"
        candidate_mask &= ~low_clean

    long_high_opposite = candidate_mask & side.eq("long") & df["p_opposite"].gt(max_p_opposite)
    short_high_opposite = candidate_mask & side.eq("short") & df["p_opposite"].gt(max_p_opposite)
    df.loc[long_high_opposite, "selection_reason"] = "above_max_p_opposite_long"
    df.loc[short_high_opposite, "selection_reason"] = "above_max_p_opposite_short"
    candidate_mask &= ~(long_high_opposite | short_high_opposite)

    low_margin = candidate_mask & df["probability_margin"].lt(config.min_probability_margin)
    df.loc[low_margin, "selection_reason"] = "below_probability_margin"
    candidate_mask &= ~low_margin

    low_utility = candidate_mask & df["utility_score"].lt(config.min_utility_score)
    df.loc[low_utility, "selection_reason"] = "below_min_utility"
    candidate_mask &= ~low_utility

    if (
        config.max_symbol_return_20d_rank_pct is not None
        and "symbol_return_20d_rank_pct" in df.columns
    ):
        overextended = candidate_mask & df["symbol_return_20d_rank_pct"].gt(
            config.max_symbol_return_20d_rank_pct
        )
        df.loc[overextended, "selection_reason"] = "above_max_symbol_return_20d_rank"
        candidate_mask &= ~overextended

    if (
        config.long_max_hist_opposite_rate_63 is not None
        and "hist_opposite_rate_63" in df.columns
    ):
        high_long_history_risk = (
            candidate_mask
            & side.eq("long")
            & df["hist_opposite_rate_63"].gt(config.long_max_hist_opposite_rate_63)
        )
        df.loc[high_long_history_risk, "selection_reason"] = "above_long_hist_opposite_rate_63"
        candidate_mask &= ~high_long_history_risk

    if config.resolve_symbol_side_conflicts and candidate_mask.any():
        conflict_candidates = df[candidate_mask].sort_values(
            ["date", "symbol", "p_hit_near", "p_opposite"],
            ascending=[True, True, False, True],
        )
        keep_indices = conflict_candidates.drop_duplicates(["date", "symbol"], keep="first").index
        conflict_losers = candidate_mask & ~df.index.isin(keep_indices)
        df.loc[conflict_losers, "selection_reason"] = "symbol_side_conflict"
        candidate_mask &= ~conflict_losers

    df.loc[candidate_mask, "selection_reason"] = "candidate_not_selected"

    trading_dates = sorted(df["date"].dropna().unique())
    date_index = {d: i for i, d in enumerate(trading_dates)}
    last_selected_index: dict[str, int] = {}

    candidates_df = df[candidate_mask]
    for date, day in candidates_df.groupby("date", sort=True):
        day_idx = date_index[date]
        side_counts = {"long": 0, "short": 0}
        selected_indices: list[int] = []
        candidates = day.sort_values("utility_score", ascending=False)
        for idx, row in candidates.iterrows():
            side = str(row["side"])
            if len(selected_indices) >= config.max_signals_per_day:
                df.at[idx, "selection_reason"] = "daily_cap"
                continue
            if side_counts.get(side, 0) >= config.max_signals_per_side:
                df.at[idx, "selection_reason"] = "side_cap"
                continue
            symbol = str(row["symbol"])
            previous_idx = last_selected_index.get(symbol)
            if previous_idx is not None and day_idx - previous_idx <= config.cooldown_trading_days:
                df.at[idx, "selection_reason"] = "symbol_cooldown"
                continue
            df.at[idx, "selected"] = True
            df.at[idx, "selection_reason"] = "selected"
            selected_indices.append(idx)
            side_counts[side] = side_counts.get(side, 0) + 1
            last_selected_index[symbol] = day_idx

    return df
