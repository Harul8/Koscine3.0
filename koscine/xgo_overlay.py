from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import PREDICTIONS_DIR
from koscine.production_veto import balanced_veto_pass


XGO_LABEL = "XGO+"
XGO_PROFILE = "balanced_sector_confirm3_v1"
SIGNAL_LABELS = {"GO+", "GO", "RGO+", XGO_LABEL}


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def has_xgo_label(label: object) -> bool:
    return XGO_LABEL in str(label or "")


def is_signal_label(label: object) -> bool:
    text = str(label or "")
    return text in SIGNAL_LABELS or "RGO+" in text or XGO_LABEL in text


def combine_with_xgo_label(label: object) -> str:
    text = str(label or "").strip()
    if not text or text == "WATCH":
        return XGO_LABEL
    parts = [part.strip() for part in text.split("/") if part.strip() and part.strip() != "WATCH"]
    if XGO_LABEL not in parts:
        parts.append(XGO_LABEL)
    return "/".join(parts) if parts else XGO_LABEL


def add_xgo_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    side = out.get("side", pd.Series("", index=out.index)).astype(str)
    sign = pd.Series(np.where(side.eq("up"), 1.0, -1.0), index=out.index)
    out["side_ret_5d"] = sign * _num(out, "ret_5d", 0.0)
    out["side_ret_10d"] = sign * _num(out, "ret_10d", 0.0)
    out["side_sector_rel_ret_5d"] = sign * _num(out, "sector_rel_ret_5d", 0.0)
    out["side_sector_rel_ret_20d"] = sign * _num(out, "sector_rel_ret_20d", 0.0)
    out["side_ema_20_slope_5d"] = sign * _num(out, "ema_20_slope_5d", 0.0)
    out["side_confirm_votes"] = (
        out["side_sector_rel_ret_5d"].gt(0).astype(int)
        + out["side_sector_rel_ret_20d"].gt(0).astype(int)
        + out["side_ret_5d"].gt(0).astype(int)
        + out["side_ret_10d"].gt(0).astype(int)
        + out["side_ema_20_slope_5d"].gt(0).astype(int)
    )
    out["xgo_balanced_veto_pass"] = balanced_veto_pass(out)
    return out


def _candidate_source_label(frame: pd.DataFrame) -> pd.Series:
    if "go_label_raw" in frame.columns:
        return frame["go_label_raw"].fillna("").astype(str)
    if "go_label" in frame.columns:
        return frame["go_label"].fillna("").astype(str)
    return pd.Series("", index=frame.index)


def xgo_signal_mask(frame: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.Series:
    out = add_xgo_features(frame)
    source_label = _candidate_source_label(out)
    source_pool = source_label.isin(["GO+", "GO"])
    if not bool(source_pool.any()):
        return pd.Series(False, index=out.index)

    thresholds = thresholds or {}
    score = _num(out, "score")
    side_ret5 = _num(out, "side_ret_5d")
    score_q50 = float(thresholds.get("score_q50", score.loc[source_pool].quantile(0.50)))
    side_ret5_q40 = float(thresholds.get("side_ret5_q40", side_ret5.loc[source_pool].quantile(0.40)))
    score_pool = source_pool & score.ge(score_q50)
    side_ret_pool = score_pool & side_ret5.ge(side_ret5_q40)
    confirm_pool = score_pool & _num(out, "side_confirm_votes", 0).ge(2)
    final_rule = (
        out["xgo_balanced_veto_pass"].fillna(False).astype(bool)
        & _num(out, "side_confirm_votes", 0).ge(3)
        & _num(out, "side_sector_rel_ret_5d", -999).gt(0)
    )
    return (side_ret_pool | confirm_pool) & final_rule


def apply_xgo_overlay_to_frame(frame: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    if frame.empty or "model_id" not in frame.columns:
        return frame
    out = add_xgo_features(frame)
    if "go_label" not in out.columns:
        out["go_label"] = "WATCH"
    if "go_label_raw" not in out.columns:
        out["go_label_raw"] = out["go_label"]
    if "production_bucket" not in out.columns:
        out["production_bucket"] = "WATCH"
    if "final_signal_bucket" not in out.columns:
        out["final_signal_bucket"] = out["production_bucket"]

    mask = xgo_signal_mask(out, thresholds=thresholds)
    out["xgo_profile"] = XGO_PROFILE
    out["xgo_signal"] = mask
    out["xgo_label"] = np.where(mask, XGO_LABEL, "")
    out["xgo_source_label"] = _candidate_source_label(out)
    out["xgo_reason"] = ""
    out.loc[mask, "xgo_reason"] = "score_q50_plus_balanced_sector_confirm3"
    out["xgo_side_confirm_votes"] = out["side_confirm_votes"]

    for idx in out.index[mask]:
        current_label = out.at[idx, "go_label"]
        combined_label = combine_with_xgo_label(current_label)
        side = str(out.at[idx, "side"])
        xgo_bucket = "PROD_LONG_XGO_PLUS" if side == "up" else "PROD_SHORT_XGO_PLUS"
        out.at[idx, "go_label"] = combined_label
        out.at[idx, "production_signal"] = True
        out.at[idx, "passes_min_score"] = True
        out.at[idx, "production_locked"] = True
        out.at[idx, "xgo_bucket"] = xgo_bucket
        if str(out.at[idx, "production_bucket"]) == "WATCH":
            out.at[idx, "production_bucket"] = xgo_bucket
        if str(out.at[idx, "final_signal_bucket"]) == "WATCH":
            out.at[idx, "final_signal_bucket"] = xgo_bucket
        out.at[idx, "trade_priority"] = "high"
        out.at[idx, "trade_bucket"] = "xgo_long" if side == "up" else "xgo_short"
        out.at[idx, "final_side"] = side
        note = str(out.at[idx, "production_lock_note"]) if "production_lock_note" in out.columns else ""
        suffix = "xgo_balanced_sector_confirm3"
        out.at[idx, "production_lock_note"] = suffix if not note or note == "nan" else f"{note};{suffix}"
    return out


def apply_xgo_overlay_to_prod_dir(
    prod_dir: Path = PREDICTIONS_DIR / "prod",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    files = sorted(prod_dir.glob("prod_predictions_*.parquet"))
    if start:
        start_ts = pd.Timestamp(start).normalize()
        files = [path for path in files if pd.Timestamp(path.stem.replace("prod_predictions_", "")).normalize() >= start_ts]
    if end:
        end_ts = pd.Timestamp(end).normalize()
        files = [path for path in files if pd.Timestamp(path.stem.replace("prod_predictions_", "")).normalize() <= end_ts]

    threshold_frames = []
    for parquet_path in files:
        frame = add_xgo_features(pd.read_parquet(parquet_path))
        source_pool = _candidate_source_label(frame).isin(["GO+", "GO"])
        if bool(source_pool.any()):
            threshold_frames.append(frame.loc[source_pool, ["score", "side_ret_5d"]])
    thresholds = None
    if threshold_frames:
        threshold_source = pd.concat(threshold_frames, ignore_index=True)
        thresholds = {
            "score_q50": float(_num(threshold_source, "score").quantile(0.50)),
            "side_ret5_q40": float(_num(threshold_source, "side_ret_5d").quantile(0.40)),
        }

    written = 0
    xgo_rows = 0
    for parquet_path in files:
        before = pd.read_parquet(parquet_path)
        after = apply_xgo_overlay_to_frame(before, thresholds=thresholds)
        xgo_rows += int(after.get("xgo_signal", pd.Series(False, index=after.index)).fillna(False).sum())
        after.to_parquet(parquet_path, index=False)
        after.to_csv(parquet_path.with_suffix(".csv"), index=False)
        written += 1
    return {
        "profile": XGO_PROFILE,
        "files": written,
        "xgo_rows": xgo_rows,
        "thresholds": thresholds,
    }
