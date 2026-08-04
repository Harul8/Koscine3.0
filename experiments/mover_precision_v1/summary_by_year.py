"""Year-by-year summary of the mover signals + the real ATM+2% option peak gain (t+1 open -> peak).

Signals: mover_book_final.csv (top-3/day, and the conviction ~1/day tier). Target hit = stock move_mag >= 6%.
Option gain: join each signal (date t) to the ATM+2% option entered at the NEXT trading day's OPEN (t+1) from the
real-bhavcopy tape (option_gain_study_v1/results/option_gain_trades.csv); peak gain = high_ratio-1. Direction-
agnostic -> report the FAVORABLE side (max of CALL/PUT peak), i.e. assuming you pick the correct side offline.

    python experiments/mover_precision_v1/summary_by_year.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
mb = pd.read_csv(HERE / "results" / "mover_book_final.csv", parse_dates=["date"])
og = pd.read_csv(ROOT / "experiments" / "option_gain_study_v1" / "results" / "option_gain_trades.csv", parse_dates=["entry_date"])

atm2 = og[og.strike_label.eq("ATM+2%")].copy()
best = atm2.groupby(["entry_date", "symbol"]).agg(opt_peak=("high_ratio", "max"),       # favorable side peak multiple
                                                  opt_held=("close_ratio", "max"),
                                                  entry_prem=("entry_open", "mean")).reset_index()
both = atm2.groupby(["entry_date", "symbol"]).high_ratio.mean().reset_index().rename(columns={"high_ratio": "opt_peak_bothavg"})
best = best.merge(both, on=["entry_date", "symbol"])

cal = np.sort(pd.Index(mb.date.unique()).union(og.entry_date.unique()))
nxt = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}


def summarize(sig, label):
    s = sig.copy()
    s["opt_date"] = s.date.map(nxt)
    s = s.merge(best, left_on=["opt_date", "symbol"], right_on=["entry_date", "symbol"], how="left")
    s["yr"] = s.date.dt.year
    print(f"\n===== {label} =====")
    print(f"{'year':5s} {'stocks':>6s} {'trades':>7s} {'hit>=6%':>8s} {'in_top5':>8s} {'opt_priced':>10s} "
          f"{'ATM+2% peak gain (favorable side)':>34s} {'opt>=2x':>8s} {'opt>=3x':>8s}")
    for yr, d in s.groupby("yr"):
        op = d.dropna(subset=["opt_peak"])
        gain = (op.opt_peak - 1) * 100
        print(f"{yr:5d} {d.symbol.nunique():>6d} {len(d):>7d} {d.hit6.mean():>8.3f} {d.in_top5.mean():>8.3f} "
              f"{len(op):>10d} {'mean %+.0f%% / med %+.0f%%' % (gain.mean(), gain.median()):>34s} "
              f"{(op.opt_peak>=2).mean():>8.3f} {(op.opt_peak>=3).mean():>8.3f}")
    op = s.dropna(subset=["opt_peak"]); gain = (op.opt_peak - 1) * 100
    print(f"{'ALL':5s} {s.symbol.nunique():>6d} {len(s):>7d} {s.hit6.mean():>8.3f} {s.in_top5.mean():>8.3f} "
          f"{len(op):>10d} {'mean %+.0f%% / med %+.0f%%' % (gain.mean(), gain.median()):>34s} "
          f"{(op.opt_peak>=2).mean():>8.3f} {(op.opt_peak>=3).mean():>8.3f}")
    print(f"   (blind-side-avg ATM+2% peak gain, for reference: mean {((op.opt_peak_bothavg-1)*100).mean():+.0f}%)")


summarize(mb[mb.signal_top3], "TOP-3/day (daily core, ~2.8/day)")
hi = mb[mb.signal_top3 & (mb.conv_pctile >= mb.loc[mb.signal_top3, "conv_pctile"].quantile(0.65))]
summarize(hi, "CONVICTION tier (~1/day, top-3 & high conv)")
