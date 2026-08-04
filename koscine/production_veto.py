from __future__ import annotations

import numpy as np
import pandas as pd


BALANCED_VETO_PROFILE = "balanced_v1"


def _num_col(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def balanced_veto_pass(frame: pd.DataFrame) -> pd.Series:
    """Return the balanced production-veto pass mask for current model IDs."""
    model_id = frame["model_id"].astype(str)
    keep = pd.Series(True, index=frame.index)

    liquid_down = model_id.eq("liquid30_down_4pct_5d")
    liquid_up = model_id.eq("liquid30_up_4pct_5d")
    rest_down = model_id.eq("rest35_down_7pct_5d")
    rest_up = model_id.eq("rest35_up_7pct_5d")

    pcr_oi = _num_col(frame, "pcr_oi")
    delivery_pct_chg_5 = _num_col(frame, "delivery_pct_chg_5")
    atr_pct_14 = _num_col(frame, "atr_pct_14")
    ret_20d = _num_col(frame, "ret_20d")
    score_gap = _num_col(frame, "score_gap")
    mkt_pct_above_sma20 = _num_col(frame, "mkt_pct_above_sma20")

    keep.loc[liquid_down] &= pcr_oi.loc[liquid_down].le(0.75) & delivery_pct_chg_5.loc[liquid_down].le(2.0)
    keep.loc[liquid_up] &= atr_pct_14.loc[liquid_up].ge(0.032)
    keep.loc[rest_down] &= pcr_oi.loc[rest_down].le(0.65) & ret_20d.loc[rest_down].le(0.10)
    keep.loc[rest_up] &= score_gap.loc[rest_up].ge(0.08) & mkt_pct_above_sma20.loc[rest_up].ge(0.55)
    return keep.fillna(False)


def apply_balanced_veto_to_production_frame(
    frame: pd.DataFrame,
    *,
    demote_nonproduction_go: bool = False,
) -> pd.DataFrame:
    """Apply balanced veto to final production signals.

    When demote_nonproduction_go is true, the visible go_label is forced to
    match final production status. That keeps the UI Signal column aligned with
    the production bucket rather than the broader research GO labels.
    """
    out = frame.copy()
    if out.empty or "model_id" not in out.columns:
        return out

    if "go_label_raw" not in out.columns and "go_label" in out.columns:
        out["go_label_raw"] = out["go_label"]
    out["production_veto_profile"] = BALANCED_VETO_PROFILE
    out["production_veto_pass"] = balanced_veto_pass(out)

    go_label = out.get("go_label", pd.Series("", index=out.index)).astype(str)
    production = out.get("production_signal", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if "production_bucket" in out.columns:
        production |= out["production_bucket"].astype(str).ne("WATCH")
    if "final_signal_bucket" in out.columns:
        production |= out["final_signal_bucket"].astype(str).ne("WATCH")

    rescue_production = go_label.str.contains("RGO+", regex=False) | out.get("production_bucket", pd.Series("", index=out.index)).astype(str).isin(["PROD_LONG_RESCUE_GO_PLUS", "PROD_LONG_GO_PLUS_AND_RESCUE"])
    demote = production & ~rescue_production & ~out["production_veto_pass"]
    out["production_veto_reason"] = ""
    out.loc[demote, "production_veto_reason"] = "balanced_veto_failed"

    for col in ("production_signal", "strict_signal", "production_locked", "passes_min_score"):
        if col in out.columns:
            out.loc[demote, col] = False
    for col in ("production_bucket", "final_signal_bucket", "trade_bucket", "trade_priority"):
        if col in out.columns:
            out.loc[demote, col] = "WATCH"
    if "lock_reason" in out.columns:
        out.loc[demote, "lock_reason"] = "balanced_veto_failed"
    if demote_nonproduction_go and "go_label" in out.columns:
        out.loc[demote, "go_label"] = "WATCH"

    final_production = pd.Series(False, index=out.index)
    if "production_signal" in out.columns:
        final_production |= out["production_signal"].fillna(False).astype(bool)
    if "go_label" in out.columns:
        visible_label = out["go_label"].fillna("").astype(str)
        final_production |= visible_label.isin(["GO+", "GO", "RGO+"]) | visible_label.str.contains("RGO+", regex=False)
    if "production_bucket" in out.columns:
        final_production |= out["production_bucket"].astype(str).ne("WATCH")
    if "final_signal_bucket" in out.columns:
        final_production |= out["final_signal_bucket"].astype(str).ne("WATCH")

    if demote_nonproduction_go and "go_label" in out.columns:
        out.loc[~final_production, "go_label"] = "WATCH"
    return out
