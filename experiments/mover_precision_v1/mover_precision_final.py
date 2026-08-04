"""Final mover-precision model + forward book (2024->2026, quarterly walk-forward).

Best config from the sweep: ENSEMBLE = rank-avg(clf>=8%, reg move_mag, atm_iv) for the daily top-3 signals,
with a DUAL-GATE flag (model AND atm_iv both top-5 = agreement) and a conviction percentile for volume/quality
tiering. Direction-agnostic large-move signals for option buying; user picks side + exit offline.

    set PYTHONPATH=src && python experiments/mover_precision_v1/mover_precision_final.py
Writes results/mover_book_final.csv (the forward book) + prints the recommended operating points. PROD untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from mover_precision import load, score_walkforward  # noqa: E402


def main():
    m, feats = load()
    reg = score_walkforward(m, feats, lambda tr: tr.move_mag.clip(0, 0.4))
    clf8 = score_walkforward(m, feats, lambda tr: (tr.move_mag >= 0.08).astype(int), clf=True)

    key = ["date", "symbol"]
    b = clf8[key + ["group", "move_mag", "atm_iv"]].rename(columns={}).copy()
    b = b.merge(clf8[key + ["score"]].rename(columns={"score": "s_clf"}), on=key)
    b = b.merge(reg[key + ["score"]].rename(columns={"score": "s_reg"}), on=key)
    b["s_iv"] = b["atm_iv"]
    for c in ("s_clf", "s_reg", "s_iv"):
        b[c + "r"] = b.groupby("date")[c].rank(pct=True)
    b["ens"] = b[["s_clfr", "s_regr", "s_ivr"]].mean(axis=1)
    b["srank"] = b.groupby("date")["ens"].rank(ascending=False, method="first")
    b["arank"] = b.groupby("date")["move_mag"].rank(ascending=False, method="first")
    b["iv_rank"] = b.groupby("date")["atm_iv"].rank(ascending=False, method="first")
    b["clf_rank"] = b.groupby("date")["s_clf"].rank(ascending=False, method="first")
    b["dual_gate"] = (b.clf_rank <= 5) & (b.iv_rank <= 5)
    b["conv_pctile"] = b.groupby("date")["ens"].rank(pct=True)
    b["in_top3"] = b.arank <= 3
    b["in_top5"] = b.arank <= 5
    b["hit6"] = b.move_mag >= 0.06
    b["hit8"] = b.move_mag >= 0.08
    b["signal_top3"] = b.srank <= 3

    out = HERE / "results"; out.mkdir(exist_ok=True)
    cols = ["date", "group", "symbol", "ens", "srank", "dual_gate", "conv_pctile", "atm_iv",
            "move_mag", "arank", "in_top3", "in_top5", "hit6", "hit8", "signal_top3"]
    b.sort_values(["date", "srank"])[cols].to_csv(out / "mover_book_final.csv", index=False)

    def stats(sig, lab):
        if sig.empty:
            return
        pday = sig.groupby("date").apply(lambda x: (x.arank <= 5).sum(), include_groups=False)
        print(f"  {lab:34s} {round(len(sig)/2.45):>5d}/yr | in3 {(sig.in_top3).mean():.3f} in5 {(sig.in_top5).mean():.3f} "
              f">=1t5/d {(pday>=1).mean():.3f} | hit6 {(sig.hit6).mean():.3f} hit8 {(sig.hit8).mean():.3f} move% {sig.move_mag.mean()*100:.2f}")

    print("\n=== FINAL operating points (forward 2024-2026, walk-forward) ===")
    print("  baseline universe move_mag mean %.2f%% ; random top3 in5 ~0.10\n" % (m[m.eligible & m.in_univ].move_mag.mean()*100))
    stats(b[b.srank <= 2], "A) top-2/day (daily core)")
    stats(b[b.srank <= 3], "B) top-3/day (daily core)")
    stats(b[(b.srank <= 3) & (b.conv_pctile >= b[b.srank <= 3].conv_pctile.quantile(0.5))], "C) top-3 + conviction>=median (~1/day)")
    stats(b[(b.srank <= 5) & b.dual_gate], "D) dual-gate agree (model & atm_iv top5)")
    stats(b[(b.srank <= 3) & b.dual_gate], "E) dual-gate AND top-3 (highest conviction)")
    print(f"\nsaved forward book -> results/mover_book_final.csv ({len(b):,} rows, {b.date.min().date()}..{b.date.max().date()})")
    print("Each day: take signal_top3 rows (or a tighter tier). Direction-agnostic — buy CALL or PUT per offline view.")


if __name__ == "__main__":
    main()
