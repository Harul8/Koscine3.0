"""Target-matched, decision-time ensemble checks for the locked signal books.

This is research only: it neither changes selectors nor writes lock files.
The available v3 files contain only their top-three per group/day, so results
measure a *post-selector ordering overlay*, not a replacement for a full-
universe walk-forward selection experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "locks"
REPORTS = ROOT / "reports"


def _daily_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["date", "group"])[column].rank(pct=True)


def _metrics(frame: pd.DataFrame, score: str, actual: str, threshold: float, *, pct_forecast: bool = True) -> dict[str, float | int]:
    completed = frame.dropna(subset=[score, actual]).copy()
    completed["rank"] = completed.groupby(["date", "group"])[score].rank(ascending=False, method="first")
    top = completed[completed["rank"].eq(1)]
    daily_ic = completed.groupby(["date", "group"]).apply(
        lambda x: x[score].corr(x[actual], method="spearman"), include_groups=False).dropna()
    result: dict[str, float | int] = {
        "rows": int(len(completed)),
        "mean_daily_rank_ic": round(float(daily_ic.mean()), 3), "top1_rows": int(len(top)),
        "top1_mean_move_pct": round(float(top[actual].mean()), 3),
        f"top1_hit_ge_{threshold:g}_pct": round(float((top[actual] >= threshold).mean() * 100), 1),
    }
    if pct_forecast:
        result["mae_pct"] = round(float((completed[score] - completed[actual]).abs().mean()), 3)
    return result


def _load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(LOCKS / name)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run() -> dict[str, object]:
    v3_5 = _load("prod_largemove_v3/mover_v3_book_5d.csv")
    v3_1 = _load("prod_largemove_v3/mover_v3_book_1d.csv")
    v2 = _load("prod_largemove_v2/book_2024_26.csv")
    next_day = _load("prod_largemove_v2/next_day_book.csv")

    # 5d: raw v3 move forecast + IV can be compared only as a reordering
    # overlay inside the already-selected v3 candidates.  V2 agreement is a
    # stricter selector-overlap test, evaluated with the v3 target.
    five = v3_5.dropna(subset=["move_mag", "pred_move_pct"]).copy()
    five["actual_peak_pct"] = five.move_mag * 100
    five["v3_score"] = _daily_rank(five, "pred_move_pct")
    five["iv_score"] = _daily_rank(five, "atm_iv")
    five["equal_rank_blend"] = (five.v3_score + five.iv_score) / 2
    v2_keys = v2[["date", "group", "symbol"]].drop_duplicates().assign(v2_agrees=True)
    five = five.merge(v2_keys, on=["date", "group", "symbol"], how="left")
    five["v2_agrees"] = five["v2_agrees"].notna()
    agreement = five[five.v2_agrees]
    disagreement = five[~five.v2_agrees]

    # 1d: both inputs forecast the same next-session intraday peak.  Use
    # equal within-day ranks, which is decision-time and requires no fitting.
    one = v3_1.merge(next_day[["date", "group", "symbol", "pred_move_pct", "next_move_pct"]].rename(
        columns={"pred_move_pct": "nextday_peak_pred_pct", "next_move_pct": "actual_peak_pct"}),
        on=["date", "group", "symbol"], how="inner")
    one = one.dropna(subset=["move_mag", "pred_move_pct", "nextday_peak_pred_pct", "actual_peak_pct"]).copy()
    one["v3_score"] = _daily_rank(one, "pred_move_pct")
    one["nextday_score"] = _daily_rank(one, "nextday_peak_pred_pct")
    one["equal_rank_blend"] = (one.v3_score + one.nextday_score) / 2
    # Scale blend is useful for MAE only; rank blend is used for top-pick IC.
    one["equal_value_blend_pct"] = (one.pred_move_pct + one.nextday_peak_pred_pct) / 2

    result = {
        "scope": "Historical target-matched overlay research; no production selector was changed.",
        "limitations": [
            "v3 source files retain only top-three/group/day, so the test cannot establish a full-universe replacement selector.",
            "5d v2 and v3 outcomes use different entry definitions; only their overlap is evaluated against the v3 peak target.",
            "The close-to-close expected-move model is excluded because it forecasts a different target from intraday peak excursion.",
        ],
        "five_day": {
            "candidate_reordering": {
                "v3_regression_rank": _metrics(five, "v3_score", "actual_peak_pct", 6, pct_forecast=False),
                "atm_iv_rank": _metrics(five, "iv_score", "actual_peak_pct", 6, pct_forecast=False),
                "equal_rank_blend": _metrics(five, "equal_rank_blend", "actual_peak_pct", 6, pct_forecast=False),
            },
            "v2_v3_agreement": {
                "agreement_rows": int(len(agreement)), "agreement_mean_peak_pct": round(float(agreement.actual_peak_pct.mean()), 3),
                "agreement_hit_ge_6_pct": round(float((agreement.actual_peak_pct >= 6).mean() * 100), 1),
                "v3_only_rows": int(len(disagreement)), "v3_only_mean_peak_pct": round(float(disagreement.actual_peak_pct.mean()), 3),
                "v3_only_hit_ge_6_pct": round(float((disagreement.actual_peak_pct >= 6).mean() * 100), 1),
            },
        },
        "one_day": {
            "v3_regression": _metrics(one, "pred_move_pct", "actual_peak_pct", 4),
            "nextday_peak_model": _metrics(one, "nextday_peak_pred_pct", "actual_peak_pct", 4),
            "equal_value_blend": _metrics(one, "equal_value_blend_pct", "actual_peak_pct", 4),
            "equal_rank_blend": _metrics(one, "equal_rank_blend", "actual_peak_pct", 4, pct_forecast=False),
        },
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ensemble_value_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
