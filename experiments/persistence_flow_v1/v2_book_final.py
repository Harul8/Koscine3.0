"""Final v2 book: atm_iv rank + t+3 cooldown + per-stock share cap, top-3/group/day, both groups.
Confirms the cap fixes Group A concentration; reports the final per-group summary."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

from v2_summary import build  # reuse dataset builder (move_mag, closed_opp, atm_iv, group, eligible)

pd.set_option("display.width", 240)
GROUP_SIZE = {"A_mcap30": 30, "B_turn35": 35}
DEPTH = 3


def select(eg: pd.DataFrame, cooldown: int = 3, cap_frac=None) -> pd.DataFrame:
    """atm_iv rank, top-DEPTH/day, t+cooldown, optional per-stock cap on share of TRADES."""
    days = sorted(eg.date.unique())
    last, cum, keep, total = {}, {}, [], 0
    for di, day in enumerate(days):
        g = eg[eg.date == day].sort_values("atm_iv", ascending=False)
        picked = 0
        for idx, s in zip(g.index, g.symbol):
            if di - last.get(s, -10**9) < cooldown:
                continue
            if cap_frac is not None and total > 0 and cum.get(s, 0) >= cap_frac * total:
                continue
            keep.append(idx); last[s] = di; cum[s] = cum.get(s, 0) + 1; total += 1; picked += 1
            if picked >= DEPTH:
                break
    return eg.loc[keep]


def summary(d: pd.DataFrame, grp: str, yrs: float) -> dict:
    vc = d.symbol.value_counts()
    return {"group": grp, "trades": len(d), "per_yr": round(len(d) / yrs),
            "move>6%": round((d.move_mag >= 0.06).mean() * 100, 1),
            "move>8%": round((d.move_mag >= 0.08).mean() * 100, 1),
            "closed_opp%": round(d.closed_opp.mean() * 100, 1),
            "coverage": f"{d.symbol.nunique()}/{GROUP_SIZE[grp]}",
            "max_stock%": round(vc.iloc[0] / len(d) * 100, 1),
            "top5_share%": round(vc.head(5).sum() / len(d) * 100, 1),
            "top5": ", ".join(f"{s}({c})" for s, c in vc.head(5).items())}


def main():
    base = build()
    ev = base[base.date.dt.year.isin([2024, 2025, 2026]) & base.eligible & base.group.notna()].copy()
    yrs = (ev.date.max() - ev.date.min()).days / 365.25

    print(f"v2 book tradeoff: atm_iv rank, top-{DEPTH}/group/day | {ev.date.min().date()}..{ev.date.max().date()}")
    print("per-stock cap = max share of a group's TRADES one name can take.\n")
    cols = ["group", "per_yr", "move>6%", "move>8%", "closed_opp%", "coverage", "max_stock%", "top5_share%"]
    variants = [("naive top-3 (no cooldown/cap)", dict(cooldown=0, cap_frac=None)),
                ("t+3 cooldown only", dict(cooldown=3, cap_frac=None)),
                ("t+3 + 20% cap", dict(cooldown=3, cap_frac=0.20)),
                ("t+3 + 15% cap", dict(cooldown=3, cap_frac=0.15)),
                ("t+3 + 10% cap", dict(cooldown=3, cap_frac=0.10))]
    for label, kw in variants:
        rows = [summary(select(ev[ev.group == grp], **kw), grp, yrs) for grp in ("A_mcap30", "B_turn35")]
        print(f"### {label}")
        print(pd.DataFrame(rows)[cols].to_string(index=False))
        print()
    # names under the recommended t+3 + 15% cap
    print("top-5 most-picked under t+3 + 15% cap:")
    for grp in ("A_mcap30", "B_turn35"):
        r = summary(select(ev[ev.group == grp], cooldown=3, cap_frac=0.15), grp, yrs)
        print(f"  {grp}: {r['top5']}")


if __name__ == "__main__":
    main()
