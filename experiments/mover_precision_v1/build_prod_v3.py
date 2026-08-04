"""Assemble the PROD v3 book — ONE ranked list, top-3/day across all IV, liquidity-gated, cost-tagged.

Spec (user): rank all eligible-65 names by the ensemble move-precision score; keep only names whose ATM+2%
option traded >=1000 contracts (both legs); take top-3/day. Tag each signal with iv_group (low/high), an
'expensive' flag (high-IV = move must be large to justify the premium), and the ATM+2% premiums. The user
takes direction + the expensive/skip call per signal offline.

Inputs: mover_book_final.csv (ensemble-scored) + atm2_liquidity.csv. Output: results/prod_v3_book.csv + summary.

    python experiments/mover_precision_v1/build_prod_v3.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_CONTRACTS = 1000
K = 3


def main():
    b = pd.read_csv(HERE / "results" / "mover_book_final.csv", parse_dates=["date"])
    liq = pd.read_csv(HERE / "results" / "atm2_liquidity.csv", parse_dates=["date"])
    b = b.merge(liq[["date", "symbol", "c_vol", "p_vol", "c_prem", "p_prem"]], on=["date", "symbol"], how="left")
    b["atm2_contracts"] = b[["c_vol", "p_vol"]].min(axis=1)
    b["liquid"] = b.atm2_contracts >= MIN_CONTRACTS
    b["iv_med"] = b.groupby("date").atm_iv.transform("median")
    b["iv_group"] = np.where(b.atm_iv > b.iv_med, "HIGH", "LOW")
    b["expensive"] = b.iv_group == "HIGH"

    elig = b[b.liquid].copy()
    elig["rank"] = elig.groupby("date")["ens"].rank(ascending=False, method="first")
    sig = elig[elig["rank"] <= K].copy()
    sig["needs_big_move"] = sig.expensive          # high-IV: move must be large to justify premium
    cols = ["date", "group", "symbol", "rank", "iv_group", "expensive", "needs_big_move", "atm_iv",
            "atm2_contracts", "c_prem", "p_prem", "ens", "conv_pctile", "move_mag", "arank",
            "in_top3", "in_top5", "hit6", "hit8"]
    sig.sort_values(["date", "rank"])[cols].to_csv(HERE / "results" / "prod_v3_book.csv", index=False)

    # liquidity impact: of the would-be all-IV top-3 (no liquidity gate), how many survive?
    raw3 = b.assign(r=b.groupby("date")["ens"].rank(ascending=False, method="first"))
    raw3 = raw3[raw3.r <= K]
    print(f"liquidity gate: all-IV top-3 candidates with ATM+2% >=1000 contracts (both legs): {raw3.liquid.mean():.1%}")
    print(f"signal days with full 3 liquid signals: {(sig.groupby('date').size() == K).mean():.1%}; avg signals/day {len(sig)/sig.date.nunique():.2f}")

    sig["y"] = sig.date.dt.year
    print(f"\n=== PROD v3 book (top-3/day, all-IV, liquid, tagged) ===")
    print(f"{'year':5s} {'stocks':>6s} {'signals':>7s} {'sig/day':>7s} {'hit>=6%':>8s} {'hit>=8%':>8s} {'in_top5':>8s} {'%expensive':>10s} {'>=1t5/day':>9s}")
    for y, d in sig.groupby("y"):
        nd = d.date.nunique()
        p1t5 = d.groupby("date").apply(lambda x: (x.arank <= 5).sum() >= 1, include_groups=False).mean()
        print(f"{y:5d} {d.symbol.nunique():>6d} {len(d):>7d} {len(d)/nd:>7.2f} {d.hit6.mean():>8.3f} {d.hit8.mean():>8.3f} {d.in_top5.mean():>8.3f} {d.expensive.mean():>10.3f} {p1t5:>9.3f}")
    nd = sig.date.nunique()
    p1t5 = sig.groupby("date").apply(lambda x: (x.arank <= 5).sum() >= 1, include_groups=False).mean()
    print(f"{'ALL':5s} {sig.symbol.nunique():>6d} {len(sig):>7d} {len(sig)/nd:>7.2f} {sig.hit6.mean():>8.3f} {sig.hit8.mean():>8.3f} {sig.in_top5.mean():>8.3f} {sig.expensive.mean():>10.3f} {p1t5:>9.3f}")
    print(f"\nby IV tag: LOW hit6 {sig[sig.iv_group=='LOW'].hit6.mean():.3f} (n {sum(sig.iv_group=='LOW')}) | "
          f"HIGH hit6 {sig[sig.iv_group=='HIGH'].hit6.mean():.3f} (n {sum(sig.iv_group=='HIGH')})")
    print(f"saved -> results/prod_v3_book.csv ({len(sig):,} signals)")


if __name__ == "__main__":
    main()
