"""Top-2/day per bucket (t+3 cooldown). Per bucket per year: trades, how many CLOSED >= thr
(A 3% / B 4%) favorable + avg close move among them, how many closed opposite. Peak-hit for ref.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
PRED = ROOT / "reports" / "predictions"
COOLDOWN = 3
THR = {"A_mcap30": 0.03, "B_turn35": 0.04}


def cooldown_topn(p, n, cal):
    p = p.sort_values(["date", "confidence"], ascending=[True, False]); last, keep = {}, []
    for day, g in p.groupby("date", sort=True):
        i = cal[pd.Timestamp(day)]; c = 0
        for idx, s in zip(g.index, g["symbol"]):
            if i - last.get(s, -10**9) < COOLDOWN: continue
            keep.append(idx); last[s] = i; c += 1
            if c >= n: break
    return p.loc[keep]


def main():
    # 5-day close move per (signal date, symbol)
    m = load_market_data(columns=["date", "symbol", "open", "close"])
    m["symbol"] = m["symbol"].astype(str); m = m.sort_values(["symbol", "date"])
    g = m.groupby("symbol", sort=False)
    cm = pd.DataFrame({"date": m["date"], "symbol": m["symbol"],
                       "entry_open": g["open"].shift(-1).values, "win_close": g["close"].shift(-5).values})
    cm = cm.dropna()
    cal = {pd.Timestamp(d): i for i, d in enumerate(sorted(m["date"].unique()))}

    out = []
    for b in ["A_mcap30", "B_turn35"]:
        thr = THR[b]
        p = pd.read_csv(PRED / f"group_{b}_predictions.csv", parse_dates=["date"])
        sel = cooldown_topn(p, 2, cal).merge(cm, on=["date", "symbol"], how="left").dropna(subset=["win_close"])
        sel["close_move"] = np.where(sel.side.eq("long"), (sel.win_close - sel.entry_open) / sel.entry_open,
                                     (sel.entry_open - sel.win_close) / sel.entry_open)
        sel["year"] = sel.date.dt.year
        for yr in [2024, 2025, 2026]:
            d = sel[sel.year == yr]
            if d.empty: continue
            win = d[d.close_move >= thr]; small = d[(d.close_move >= 0) & (d.close_move < thr)]; opp = d[d.close_move < 0]
            out.append({"bucket": b, "thr": f"{int(thr*100)}%", "year": yr, "trades": len(d),
                        "closed>=thr": f"{len(win)} ({len(win)/len(d)*100:.0f}%)",
                        "avg_close_move(wins)%": round(win["close_move"].mean()*100, 1) if len(win) else 0,
                        "closed_small(0-thr)": f"{len(small)} ({len(small)/len(d)*100:.0f}%)",
                        "closed_OPPOSITE": f"{len(opp)} ({len(opp)/len(d)*100:.0f}%)",
                        "avg_opp_move%": round(opp["close_move"].mean()*100, 1) if len(opp) else 0,
                        "ref:peak>=thr": f"{int(d['hit'].sum())} ({d['hit'].mean()*100:.0f}%)"})
    res = pd.DataFrame(out)
    pd.set_option("display.width", 240)
    print("===== TOP-2/day per bucket (t+3 cooldown): CLOSE-based outcomes =====")
    print(res.to_string(index=False))
    print("\nclosed>=thr = 5-day CLOSE reached the bucket threshold in the trade's direction.")
    print("closed_OPPOSITE = 5-day close went AGAINST the trade. ref:peak>=thr = the PEAK hit (what the model targets / option-exit metric).")


if __name__ == "__main__":
    main()
