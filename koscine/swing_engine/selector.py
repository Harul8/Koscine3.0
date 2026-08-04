from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


SIGNAL_LABELS = {"ISWING", "ISWING+"}


@dataclass(frozen=True)
class SelectorConfig:
    name: str = "balanced"
    max_daily_signals: int = 5
    max_daily_per_side: int = 4
    min_validation_signals: int = 20
    min_daily_avg: float = 0.35
    min_hit_near: float = 0.60
    max_opposite: float = 0.25
    preferred_opposite: float = 0.20
    min_slice_calls: int = 5
    min_slice_pass_fraction: float = 0.67
    plus_min_hit_near: float = 0.68
    plus_max_opposite: float = 0.18
    allow_best_effort: bool = True


PROFILE_CONFIGS = {
    "balanced": SelectorConfig(name="balanced"),
    "strict": SelectorConfig(
        name="strict",
        max_daily_signals=4,
        min_validation_signals=15,
        min_hit_near=0.63,
        max_opposite=0.22,
        preferred_opposite=0.18,
        plus_min_hit_near=0.72,
        plus_max_opposite=0.15,
    ),
    "low_opp": SelectorConfig(
        name="low_opp",
        max_daily_signals=4,
        min_validation_signals=12,
        min_hit_near=0.58,
        max_opposite=0.18,
        preferred_opposite=0.15,
        plus_min_hit_near=0.66,
        plus_max_opposite=0.12,
    ),
    "recall": SelectorConfig(
        name="recall",
        max_daily_signals=6,
        max_daily_per_side=5,
        min_validation_signals=25,
        min_daily_avg=0.50,
        min_hit_near=0.58,
        max_opposite=0.25,
        preferred_opposite=0.22,
        plus_min_hit_near=0.66,
        plus_max_opposite=0.20,
    ),
}


def _dedupe_date_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(["date", "symbol", "edge_score"], ascending=[True, True, False]).drop_duplicates(["date", "symbol"])


def apply_daily_constraints(
    scored: pd.DataFrame,
    threshold: float,
    plus_threshold: float,
    config: SelectorConfig,
) -> pd.DataFrame:
    out = scored.copy()
    out["signal_label"] = "WATCH"
    out["signal_bucket"] = "WATCH"
    out["selection_profile"] = config.name
    if not np.isfinite(threshold):
        return out
    candidate = out[out["edge_score"].ge(threshold)].copy()
    candidate = _dedupe_date_symbol(candidate)
    selected_parts = []
    for date, group in candidate.groupby("date", sort=True):
        side_kept = []
        for _side, side_group in group.sort_values("edge_score", ascending=False).groupby("side", sort=True):
            side_kept.append(side_group.head(config.max_daily_per_side))
        constrained = pd.concat(side_kept, ignore_index=False) if side_kept else group.iloc[0:0]
        selected_parts.append(constrained.sort_values("edge_score", ascending=False).head(config.max_daily_signals))
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else candidate.iloc[0:0]
    if selected.empty:
        return out
    out.loc[selected.index, "signal_label"] = "ISWING"
    out.loc[selected.index, "signal_bucket"] = "independent_swing"
    if np.isfinite(plus_threshold):
        plus_idx = selected[selected["edge_score"].ge(plus_threshold)].index
        out.loc[plus_idx, "signal_label"] = "ISWING+"
        out.loc[plus_idx, "signal_bucket"] = "independent_swing_plus"
    return out


def selected_signals(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["signal_label"].isin(SIGNAL_LABELS)].copy()


def metric_row(frame: pd.DataFrame, label: str, extra: dict | None = None) -> dict:
    extra = extra or {}
    evaluated = frame[frame["window_status"].eq("evaluated")].copy()
    n = len(evaluated)
    hit = int(evaluated["hit"].sum()) if n else 0
    near = int(evaluated["near"].sum()) if n else 0
    opposite = int(evaluated["opposite"].sum()) if n else 0
    small = int(evaluated["small"].sum()) if n else 0
    return {
        **extra,
        "label": label,
        "signals": int(len(frame)),
        "evaluated": int(n),
        "dates": int(frame["date"].nunique()) if len(frame) else 0,
        "symbols": int(frame["symbol"].nunique()) if len(frame) else 0,
        "hit_n": hit,
        "near_n": near,
        "hit_near_n": hit + near,
        "opposite_n": opposite,
        "small_n": small,
        "hit_pct": hit / n if n else np.nan,
        "near_pct": near / n if n else np.nan,
        "hit_near_pct": (hit + near) / n if n else np.nan,
        "opposite_pct": opposite / n if n else np.nan,
        "small_pct": small / n if n else np.nan,
        "avg_favorable_move": float(evaluated["favorable_move"].mean()) if n else np.nan,
        "avg_signed_close": float(evaluated["signed_close_return"].mean()) if n else np.nan,
        "top5_hits": int(evaluated["top5_mover"].sum()) if n else 0,
        "daily_signal_avg": len(frame) / max(int(frame["date"].nunique()), 1) if len(frame) else 0.0,
    }


def _slice_rows(selected: pd.DataFrame, config: SelectorConfig) -> list[dict]:
    if selected.empty:
        return []
    frame = selected.copy()
    frame["slice_key"] = pd.to_datetime(frame["date"]).dt.to_period("Q").astype(str)
    rows = []
    for key, group in frame.groupby("slice_key", sort=True):
        row = metric_row(group, str(key), {"slice_key": str(key)})
        row["passes_slice"] = bool(
            row["evaluated"] >= config.min_slice_calls
            and row["hit_near_pct"] >= config.min_hit_near
            and row["opposite_pct"] <= config.max_opposite
        )
        rows.append(row)
    return rows


def _passes(row: dict, slice_rows: list[dict], config: SelectorConfig, *, plus: bool = False) -> bool:
    min_hit_near = config.plus_min_hit_near if plus else config.min_hit_near
    max_opposite = config.plus_max_opposite if plus else config.max_opposite
    if row["evaluated"] < config.min_validation_signals:
        return False
    if row["daily_signal_avg"] < config.min_daily_avg:
        return False
    if row["hit_near_pct"] < min_hit_near or row["opposite_pct"] > max_opposite:
        return False
    callable_slices = [r for r in slice_rows if r["evaluated"] >= config.min_slice_calls]
    if not callable_slices:
        return True
    pass_fraction = sum(
        bool(r["hit_near_pct"] >= min_hit_near and r["opposite_pct"] <= max_opposite) for r in callable_slices
    ) / len(callable_slices)
    return pass_fraction >= config.min_slice_pass_fraction


def _utility(row: dict) -> float:
    evaluated = max(float(row.get("evaluated", 0.0)), 1.0)
    top5_rate = float(row.get("top5_hits", 0.0)) / evaluated
    return (
        2.00 * float(row.get("hit_near_pct", 0.0))
        + 0.50 * float(row.get("avg_favorable_move", 0.0))
        + 0.40 * top5_rate
        + 0.03 * min(float(row.get("daily_signal_avg", 0.0)), 5.0)
        - 2.60 * float(row.get("opposite_pct", 0.0))
        - 0.25 * float(row.get("small_pct", 0.0))
    )


def choose_thresholds(valid_scored: pd.DataFrame, config: SelectorConfig) -> dict:
    scores = pd.to_numeric(valid_scored["edge_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return {
            "threshold": np.inf,
            "quantile": np.nan,
            "plus_threshold": np.inf,
            "plus_quantile": np.nan,
            "contract_met": False,
            "reason": "empty_scores",
            "grid": [],
            "config": asdict(config),
        }
    quantiles = np.unique(np.r_[np.linspace(0.50, 0.98, 33), np.linspace(0.985, 0.998, 6)])
    grid = []
    for q in quantiles:
        threshold = float(scores.quantile(q))
        labeled = apply_daily_constraints(valid_scored, threshold, np.inf, config)
        signals = selected_signals(labeled)
        row = metric_row(signals, f"q{q:.3f}", {"quantile": float(q), "threshold": threshold})
        slices = _slice_rows(signals, config)
        row["slice_count"] = len(slices)
        row["slice_pass_count"] = sum(bool(s.get("passes_slice")) for s in slices)
        row["passes_go"] = _passes(row, slices, config, plus=False)
        row["passes_plus"] = _passes(row, slices, config, plus=True)
        row["utility"] = _utility(row)
        grid.append(row)
    grid_df = pd.DataFrame(grid).sort_values("quantile")
    pass_pool = grid_df[grid_df["passes_go"]].copy()
    if pass_pool.empty and config.allow_best_effort:
        viable = grid_df[grid_df["evaluated"].ge(max(5, config.min_validation_signals // 2))].copy()
        pass_pool = viable if not viable.empty else grid_df.copy()
    if pass_pool.empty:
        threshold = np.inf
        quantile = np.nan
        contract_met = False
        reason = "no_validation_threshold_met_contract"
    else:
        best = pass_pool.sort_values(["passes_go", "utility", "hit_near_pct", "opposite_pct"], ascending=[False, False, False, True]).iloc[0]
        threshold = float(best["threshold"])
        quantile = float(best["quantile"])
        contract_met = bool(best["passes_go"])
        reason = "validation_contract_met" if contract_met else "best_effort_threshold"

    plus_pool = grid_df[grid_df["passes_plus"] & grid_df["threshold"].ge(threshold)].copy()
    if plus_pool.empty:
        plus_threshold = np.inf
        plus_quantile = np.nan
    else:
        plus_row = plus_pool.sort_values(["utility", "hit_near_pct"], ascending=False).iloc[0]
        plus_threshold = float(plus_row["threshold"])
        plus_quantile = float(plus_row["quantile"])
    return {
        "threshold": threshold,
        "quantile": quantile,
        "plus_threshold": plus_threshold,
        "plus_quantile": plus_quantile,
        "contract_met": contract_met,
        "reason": reason,
        "grid": grid_df.to_dict("records"),
        "config": asdict(config),
    }
