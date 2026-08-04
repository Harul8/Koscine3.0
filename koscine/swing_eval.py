from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, PREDICTIONS_DIR, PROCESSED_DIR, REPORTS_DIR


SWING_CONTRACT_VERSION = "swing_peak_5d_v1"
SIGNAL_LABELS = {"GO+", "GO", "RGO+", "XGO+", "CSGO+", "CCGO+", "CCGO"}
PROMOTED_BASELINE_BUCKETS = {
    "PROD_LONG_GO_PLUS",
    "PROD_SHORT_GO_PLUS",
    "PROD_SHORT_GO",
}
PREFERRED_OPPOSITE_CAP = 0.20
HARD_OPPOSITE_CAP = 0.25
MIN_HIT_NEAR = 0.60
DEFAULT_COST_BPS = 20.0


@dataclass(frozen=True)
class SwingEvalConfig:
    dataset_path: Path = PROCESSED_DIR / "daily_features.parquet"
    near_fraction: float = 0.80
    cost_bps: float = DEFAULT_COST_BPS
    min_hit_near: float = MIN_HIT_NEAR
    preferred_opposite_cap: float = PREFERRED_OPPOSITE_CAP
    hard_opposite_cap: float = HARD_OPPOSITE_CAP


def _normalize_dates(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _read_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_prediction_inputs(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("prod_predictions_*.parquet"))
        if not files:
            files = sorted(path.glob("prod_predictions_*.csv"))
        if not files:
            raise FileNotFoundError(f"No prod_predictions_* files found in {path}")
        frames = [_read_prediction_file(file) for file in files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _read_prediction_file(path)


def _ohlc_with_swing_paths(dataset_path: Path) -> pd.DataFrame:
    cols = ["date", "symbol", "open", "high", "low", "close"]
    df = pd.read_parquet(dataset_path, columns=cols)
    df = _normalize_dates(df, ["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = df.groupby("symbol", group_keys=False)
    df["swing_entry_date_calc"] = grouped["date"].shift(-1)
    df["swing_entry_open_calc"] = grouped["open"].shift(-1)
    df["swing_exit_date_calc"] = grouped["date"].shift(-HORIZON_DAYS)
    df["swing_exit_close_calc"] = grouped["close"].shift(-HORIZON_DAYS)
    df["swing_window_high_calc"] = grouped["high"].transform(
        lambda s: s.shift(-1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).max().shift(
            -(HORIZON_DAYS - 1)
        )
    )
    df["swing_window_low_calc"] = grouped["low"].transform(
        lambda s: s.shift(-1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).min().shift(
            -(HORIZON_DAYS - 1)
        )
    )
    return df[
        [
            "date",
            "symbol",
            "swing_entry_date_calc",
            "swing_entry_open_calc",
            "swing_exit_date_calc",
            "swing_exit_close_calc",
            "swing_window_high_calc",
            "swing_window_low_calc",
        ]
    ]


def add_branch_metadata(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    label = out.get("go_label", pd.Series("", index=out.index)).fillna("").astype(str)
    prod_bucket = out.get("production_bucket", pd.Series("", index=out.index)).fillna("").astype(str)
    final_bucket = out.get("final_signal_bucket", prod_bucket).fillna("").astype(str)
    xgo_label = out.get("xgo_label", pd.Series("", index=out.index)).fillna("").astype(str)
    crash_label = out.get("crash_short_label", pd.Series("", index=out.index)).fillna("").astype(str)

    overlay = pd.Series("core", index=out.index, dtype="object")
    overlay.loc[label.str.contains("RGO", regex=False) | final_bucket.str.contains("RESCUE", regex=False)] = "RGO"
    overlay.loc[xgo_label.ne("") | label.str.contains("XGO", regex=False) | final_bucket.str.contains("XGO", regex=False)] = "XGO"
    overlay.loc[crash_label.ne("") | label.str.contains("CSGO", regex=False) | final_bucket.str.contains("CRASH", regex=False)] = "CSGO"
    out["overlay_source"] = overlay

    bucket = final_bucket.where(final_bucket.ne(""), prod_bucket)
    out["branch_id"] = bucket.where(bucket.ne(""), label.where(label.ne(""), "WATCH"))
    out["promotion_status"] = np.where(
        bucket.isin(PROMOTED_BASELINE_BUCKETS),
        "baseline_promoted",
        np.where(label.isin(SIGNAL_LABELS) | bucket.ne("WATCH"), "candidate", "watch"),
    )
    out["swing_contract_version"] = SWING_CONTRACT_VERSION
    return out


def attach_swing_outcomes(
    predictions: pd.DataFrame,
    config: SwingEvalConfig | None = None,
) -> pd.DataFrame:
    config = config or SwingEvalConfig()
    if predictions.empty:
        return add_branch_metadata(predictions)
    if not config.dataset_path.exists():
        return add_branch_metadata(predictions)

    out = _normalize_dates(predictions, ["date", "entry_1d_date", f"future_{HORIZON_DAYS}d_date"])
    out["symbol"] = out["symbol"].astype(str)
    calc = _ohlc_with_swing_paths(config.dataset_path)
    out = out.merge(calc, on=["date", "symbol"], how="left")

    rename_pairs = {
        "swing_entry_date_calc": "swing_entry_date",
        "swing_entry_open_calc": "swing_entry_open",
        "swing_exit_date_calc": "swing_exit_date",
        "swing_exit_close_calc": "swing_exit_close",
        "swing_window_high_calc": "swing_window_high",
        "swing_window_low_calc": "swing_window_low",
    }
    for source, target in rename_pairs.items():
        if target not in out.columns:
            out[target] = out[source]
        else:
            out[target] = out[target].where(out[target].notna(), out[source])
        out = out.drop(columns=[source])

    entry = pd.to_numeric(out["swing_entry_open"], errors="coerce")
    high = pd.to_numeric(out["swing_window_high"], errors="coerce")
    low = pd.to_numeric(out["swing_window_low"], errors="coerce")
    close = pd.to_numeric(out["swing_exit_close"], errors="coerce")
    side = out.get("side", pd.Series("", index=out.index)).fillna("").astype(str)

    favorable = np.where(side.eq("down"), entry / low - 1.0, high / entry - 1.0)
    signed_close = np.where(side.eq("down"), entry / close - 1.0, close / entry - 1.0)
    opposite_move = np.where(side.eq("down"), high / entry - 1.0, entry / low - 1.0)
    out["swing_favorable_move"] = favorable
    out["swing_signed_close_return"] = signed_close
    out["swing_opposite_move"] = opposite_move
    out["swing_net_close_return"] = out["swing_signed_close_return"] - float(config.cost_bps) / 10000.0

    threshold = pd.to_numeric(out.get("threshold"), errors="coerce")
    evaluated = entry.notna() & high.notna() & low.notna() & close.notna() & threshold.notna()
    has_entry = entry.notna()
    out["swing_window_status"] = np.select(
        [evaluated, has_entry],
        ["evaluated", "pending_full_5d_window"],
        default="pending_entry_bar",
    )

    favorable_s = pd.Series(favorable, index=out.index)
    signed_close_s = pd.Series(signed_close, index=out.index)
    hit = evaluated & favorable_s.ge(threshold)
    near = evaluated & ~hit & favorable_s.ge(float(config.near_fraction) * threshold)
    opposite = evaluated & ~hit & signed_close_s.lt(0)
    small = evaluated & ~hit & ~near & ~opposite

    out["swing_hit"] = hit
    out["swing_near"] = near
    out["swing_hit_near"] = hit | near
    out["swing_opposite"] = opposite
    out["swing_small"] = small
    out["swing_verdict"] = ""
    out.loc[hit, "swing_verdict"] = "hit"
    out.loc[near, "swing_verdict"] = "near"
    out.loc[opposite, "swing_verdict"] = "opposite"
    out.loc[small, "swing_verdict"] = "small"
    out.loc[out["swing_window_status"].ne("evaluated"), "swing_verdict"] = "pending"
    out["swing_side_rank"] = np.nan
    evaluated_rank = out["swing_window_status"].eq("evaluated")
    if evaluated_rank.any():
        out.loc[evaluated_rank, "swing_side_rank"] = (
            out.loc[evaluated_rank]
            .groupby(["date", "side"])["swing_favorable_move"]
            .rank(method="first", ascending=False)
        )
    out["swing_top5_mover"] = out["swing_side_rank"].le(5)

    # Backward-compatible UI aliases, now sourced from the canonical contract.
    out["actual_move"] = out["swing_favorable_move"]
    out["actual_signed_move"] = out["swing_signed_close_return"]
    out["actual_opposite_move"] = out["swing_opposite_move"]
    out["actual_window_status"] = out["swing_window_status"]
    out["actual_hit"] = out["swing_hit"]
    out["actual_verdict"] = out["swing_verdict"].replace({"near": "partial hit"})
    entry_date = out.get("entry_1d_date", pd.Series(pd.NaT, index=out.index))
    entry_open = out.get("entry_1d_open", pd.Series(np.nan, index=out.index))
    future_date = out.get("future_5d_date", pd.Series(pd.NaT, index=out.index))
    future_close = out.get("future_5d_close", pd.Series(np.nan, index=out.index))
    out["entry_1d_date"] = entry_date.where(entry_date.notna(), out["swing_entry_date"])
    out["entry_1d_open"] = entry_open.where(entry_open.notna(), out["swing_entry_open"])
    out["future_5d_date"] = future_date.where(future_date.notna(), out["swing_exit_date"])
    out["future_5d_close"] = future_close.where(future_close.notna(), out["swing_exit_close"])
    return add_branch_metadata(out)


def actionable_mask(frame: pd.DataFrame, signal_set: str = "visible") -> pd.Series:
    label = frame.get("go_label", pd.Series("", index=frame.index)).fillna("").astype(str)
    bucket = frame.get("final_signal_bucket", pd.Series("", index=frame.index)).fillna("").astype(str)
    prod_bucket = frame.get("production_bucket", pd.Series("", index=frame.index)).fillna("").astype(str)
    production = frame.get("production_signal", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    strict = frame.get("strict_signal", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    visible_bucket = bucket.ne("") & bucket.ne("WATCH")
    visible_prod_bucket = prod_bucket.ne("") & prod_bucket.ne("WATCH")
    visible = label.isin(SIGNAL_LABELS) | label.str.contains("GO", regex=False) | visible_bucket | visible_prod_bucket
    if signal_set == "baseline":
        return bucket.isin(PROMOTED_BASELINE_BUCKETS) | prod_bucket.isin(PROMOTED_BASELINE_BUCKETS)
    if signal_set == "strict":
        return strict
    if signal_set == "production":
        return production
    if signal_set == "all":
        return pd.Series(True, index=frame.index)
    return visible


def summarize_swing(
    frame: pd.DataFrame,
    signal_set: str = "visible",
    config: SwingEvalConfig | None = None,
) -> pd.DataFrame:
    config = config or SwingEvalConfig()
    selected = frame[actionable_mask(frame, signal_set)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["year"] = pd.to_datetime(selected["date"]).dt.year.astype("Int64").astype(str)
    selected["quarter"] = pd.to_datetime(selected["date"]).dt.to_period("Q").astype(str)
    selected["month"] = pd.to_datetime(selected["date"]).dt.to_period("M").astype(str)
    selected["ALL"] = "ALL"
    groups: list[tuple[str, list[str]]] = [
        ("aggregate", ["ALL"]),
        ("year", ["year"]),
        ("quarter", ["quarter"]),
        ("month", ["month"]),
        ("side", ["side"]),
        ("tier", ["tier"]),
        ("model", ["model_id"]),
        ("bucket", ["final_signal_bucket"]),
        ("overlay", ["overlay_source"]),
        ("sector", ["sector"]),
        ("promotion", ["promotion_status"]),
        ("year_side", ["year", "side"]),
        ("quarter_side", ["quarter", "side"]),
        ("bucket_side", ["final_signal_bucket", "side"]),
    ]
    rows = []
    for slice_name, cols in groups:
        present = [col for col in cols if col in selected.columns]
        if len(present) != len(cols):
            continue
        for key, group in selected.groupby(present, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            evaluated = group[group["swing_window_status"].eq("evaluated")]
            calls = len(group)
            eval_n = len(evaluated)
            hit = int(evaluated["swing_hit"].sum()) if eval_n else 0
            near = int(evaluated["swing_near"].sum()) if eval_n else 0
            opposite = int(evaluated["swing_opposite"].sum()) if eval_n else 0
            small = int(evaluated["swing_small"].sum()) if eval_n else 0
            top5_pool = selected[selected["date"].isin(group["date"])]
            if "side" in present:
                top5_pool = top5_pool[top5_pool["side"].isin(group["side"].dropna().unique())]
            top5_available = int(
                top5_pool[top5_pool["swing_top5_mover"].fillna(False)]
                .drop_duplicates(["date", "side", "symbol"])
                .shape[0]
            )
            top5_hits = int(evaluated["swing_top5_mover"].fillna(False).sum()) if eval_n else 0
            hit_near_rate = (hit + near) / eval_n if eval_n else np.nan
            opposite_rate = opposite / eval_n if eval_n else np.nan
            rows.append(
                {
                    "slice": slice_name,
                    "key": "|".join(str(part) for part in key_tuple),
                    "signals": calls,
                    "evaluated": eval_n,
                    "pending": calls - eval_n,
                    "dates": int(group["date"].nunique()),
                    "symbols": int(group["symbol"].nunique()) if "symbol" in group else 0,
                    "long": int(group.get("side", pd.Series("", index=group.index)).eq("up").sum()),
                    "short": int(group.get("side", pd.Series("", index=group.index)).eq("down").sum()),
                    "hit_n": hit,
                    "hit_pct": hit / eval_n if eval_n else np.nan,
                    "near_n": near,
                    "near_pct": near / eval_n if eval_n else np.nan,
                    "hit_near_n": hit + near,
                    "hit_near_pct": hit_near_rate,
                    "opposite_n": opposite,
                    "opposite_pct": opposite_rate,
                    "small_n": small,
                    "small_pct": small / eval_n if eval_n else np.nan,
                    "avg_favorable_move": float(evaluated["swing_favorable_move"].mean()) if eval_n else np.nan,
                    "median_favorable_move": float(evaluated["swing_favorable_move"].median()) if eval_n else np.nan,
                    "avg_signed_close_return": float(evaluated["swing_signed_close_return"].mean()) if eval_n else np.nan,
                    "avg_net_close_return": float(evaluated["swing_net_close_return"].mean()) if eval_n else np.nan,
                    "top5_hits": top5_hits,
                    "top5_available": top5_available,
                    "top5_capture_pct": top5_hits / top5_available if top5_available else np.nan,
                    "daily_signal_avg": calls / max(int(group["date"].nunique()), 1),
                    "max_symbol_concentration_pct": (
                        float(group["symbol"].value_counts(normalize=True).max()) if "symbol" in group and calls else np.nan
                    ),
                    "max_sector_concentration_pct": (
                        float(group["sector"].value_counts(normalize=True).max()) if "sector" in group and calls else np.nan
                    ),
                    "passes_gold": bool(
                        eval_n > 0
                        and hit_near_rate >= float(config.min_hit_near)
                        and opposite_rate <= float(config.hard_opposite_cap)
                    ),
                    "passes_preferred": bool(
                        eval_n > 0
                        and hit_near_rate >= float(config.min_hit_near)
                        and opposite_rate <= float(config.preferred_opposite_cap)
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_swing_evaluation(
    predictions: pd.DataFrame,
    output_dir: Path,
    source: str,
    config: SwingEvalConfig | None = None,
    signal_set: str = "visible",
    extra_manifest: dict | None = None,
) -> dict[str, Path]:
    config = config or SwingEvalConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = attach_swing_outcomes(predictions, config)
    summary = summarize_swing(enriched, signal_set=signal_set, config=config)

    enriched_path = output_dir / "swing_predictions.parquet"
    summary_path = output_dir / "swing_summary.csv"
    manifest_path = output_dir / "manifest.json"
    enriched.to_parquet(enriched_path, index=False)
    enriched.to_csv(enriched_path.with_suffix(".csv"), index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "contract_version": SWING_CONTRACT_VERSION,
        "source": source,
        "dataset_path": str(config.dataset_path),
        "signal_set": signal_set,
        "near_fraction": config.near_fraction,
        "cost_bps": config.cost_bps,
        "promotion_bar": {
            "min_hit_near": config.min_hit_near,
            "preferred_opposite_cap": config.preferred_opposite_cap,
            "hard_opposite_cap": config.hard_opposite_cap,
        },
        "rows": int(len(enriched)),
        "summary_rows": int(len(summary)),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "predictions": enriched_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def default_eval_output_dir(name: str | None = None) -> Path:
    tag = name or pd.Timestamp.now().strftime("swing_eval_%Y%m%d_%H%M%S")
    return REPORTS_DIR / "swing_eval" / tag
