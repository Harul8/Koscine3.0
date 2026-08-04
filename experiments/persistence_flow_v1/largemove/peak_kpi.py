"""Exit-at-peak KPIs for the locked shortlist — the official capture metric.

The book exits options at the favorable PEAK inside the 5-day window (not at the close — the
close-persistence experiment showed close-direction is unpredictable). These KPIs quantify the
peak that is actually capturable: hit rate, magnitude, and *timing* (which window day the peak
lands on, and the first day the move crosses the threshold = earliest exit opportunity).

    python -m largemove.peak_kpi

Writes predictions/peak_capture_report.csv (per group×year) and predictions/peak_kpi_summary.json
(consumed by lock.py). Deterministic given the locked predictions + market data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data
from largemove.config import PROD, PREDICTIONS_DIR

THR = dict(PROD.group_thresholds)
WIN = PROD.window_days


def _forward_window() -> pd.DataFrame:
    m = load_market_data(columns=["date", "symbol", "open", "high", "low"])
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"])
    g = m.groupby("symbol", sort=False)
    base = pd.DataFrame({"date": m["date"].values, "symbol": m["symbol"].values,
                         "entry_open": g["open"].shift(-1).values})
    for i in range(1, WIN + 1):
        base[f"h{i}"] = g["high"].shift(-i).values
        base[f"l{i}"] = g["low"].shift(-i).values
    return base


def compute() -> pd.DataFrame:
    picks = pd.read_csv(PREDICTIONS_DIR / "combined_shortlist.csv", parse_dates=["date"])
    picks["symbol"] = picks["symbol"].astype(str)
    picks = picks.merge(_forward_window(), on=["date", "symbol"], how="left")
    picks = picks[picks["entry_open"].notna()].copy()

    H = picks[[f"h{i}" for i in range(1, WIN + 1)]].to_numpy(float)
    L = picks[[f"l{i}" for i in range(1, WIN + 1)]].to_numpy(float)
    e = picks["entry_open"].to_numpy(float)[:, None]
    is_long = picks["side"].eq("long").to_numpy()[:, None]
    gains = np.where(is_long, (H - e) / e, (e - L) / e)   # favorable gain per window day

    valid = ~np.isnan(gains).all(axis=1)
    picks, gains = picks[valid].copy(), gains[valid]
    thr = picks["threshold"].to_numpy(float)[:, None]

    picks["peak_ceiling"] = np.nanmax(gains, axis=1)
    picks["days_to_peak"] = np.nanargmax(np.where(np.isnan(gains), -np.inf, gains), axis=1) + 1
    reach = gains >= thr
    picks["reach_thr_day"] = np.where(reach.any(axis=1), reach.argmax(axis=1) + 1, np.nan)
    picks["peak_hit"] = reach.any(axis=1).astype(int)
    picks["year"] = picks["date"].dt.year
    return picks


def _agg(d: pd.DataFrame) -> dict:
    hits = d[d.peak_hit == 1]
    return {
        "trades": int(len(d)),
        "peak_hit_pct": round(d.peak_hit.mean() * 100, 1),
        "avg_peak_among_hits_pct": round(hits.peak_ceiling.mean() * 100, 1) if len(hits) else 0.0,
        "median_days_to_peak": float(hits.days_to_peak.median()) if len(hits) else 0.0,
        "pct_peak_by_day2": round((hits.days_to_peak <= 2).mean() * 100, 1) if len(hits) else 0.0,
        "median_first_cross_day": float(hits.reach_thr_day.median()) if len(hits) else 0.0,
    }


def build_report():
    picks = compute()
    rows = []
    for b in THR:
        d = picks[picks.group == b]
        rows.append({"scope": b, **_agg(d)})
        for yr, dy in d.groupby("year"):
            rows.append({"scope": f"  {b} {yr}", **_agg(dy)})
    rows.append({"scope": "COMBINED", **_agg(picks)})
    report = pd.DataFrame(rows)
    report.to_csv(PREDICTIONS_DIR / "peak_capture_report.csv", index=False)

    summary = {
        "strategy": "exit-at-peak (capture favorable peak via long options; downside = premium)",
        "selected": "top-2/group/day, t+3 cooldown (locked shortlist)",
        "kpi_note": "close-to-t+5 rejected as objective (experiment close_persistence_v1: close direction AUC~0.52)",
        "by_group": {b: _agg(picks[picks.group == b]) for b in THR},
        "combined": _agg(picks),
    }
    (PREDICTIONS_DIR / "peak_kpi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report, summary


def main():
    report, summary = build_report()
    pd.set_option("display.width", 200)
    print("===== EXIT-AT-PEAK capture KPIs (locked shortlist) =====")
    print(report.to_string(index=False))
    print("\npeak_hit = favorable peak crossed threshold inside the 5-day window (option-exit metric)")
    print("days_to_peak = window day of the favorable peak | first_cross_day = earliest exit day")
    print(f"\nsaved peak_capture_report.csv + peak_kpi_summary.json -> {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
