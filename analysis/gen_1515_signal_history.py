"""Emit the 14:30-signal / 15:15-entry Broken-Wing system in the SAME schema as
locks/prod_sell_strategies/broken_wing_signal_history.csv, so it is directly comparable to the
EOD history table (max_profit, pnl, ror_pct, max_dd_pct, outcome, per-tier EV, ...).

STRATEGY SCOPE: Broken-Wing only (tiers t1_2x6 / t2_3x8 / t3_2x10). Not Skew, not Condor.

Differences from the EOD generator, all deliberate:
  * signal_date == entry_date. The EOD system signals off day t's close and enters at t+1's
    open; this one signals at 14:30 and enters at 15:15 the SAME day, so there is no overnight
    gap between deciding and trading.
  * prices are the real quoted prices at 15:15 rather than the 09:15 opening print, which the
    2026-08 pilot showed is unfillable on ~55% of trades.
  * a fillability gate the EOD system has no equivalent of: every leg must have actually traded
    in the 15:00-15:15 window.
  * exits are marked at 15:15 on subsequent days (same clock as entry), not at the close.
  * ROR gate defaults to 75%, not 150%/155% -- the inherited bar is unreachable at transactable
    prices (median achievable entry ROR at 15:15 is 32%). Override with --ror-gate.

Usage:
    python analysis/gen_1515_signal_history.py
    python analysis/gen_1515_signal_history.py --ror-gate 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
LOCK = ROOT / "locks" / "prod_sell_strategies"
LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
OUT = LOCK / "bw_1515_signal_history.csv"

ENTRY_MARK, EXIT_MARK, PRE_MARK = "15:15", "15:15", "15:00"
SHORT_OTM = 0.02
TIERS = [("t1_2x6", 0.02, 0.06), ("t2_3x8", 0.03, 0.08), ("t3_2x10", 0.02, 0.10)]
DTE_MIN, FWD, SAFE_DTE = 7, 5, 4
MIN_PROFIT_PER_LOT = 10_000.0
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 1.0

COLS = ["tier", "symbol", "group", "signal_date", "entry_date", "expiry", "dte", "underlying",
        "iv_ratio", "short_ce", "long_ce", "short_pe", "long_pe",
        "oi_short_ce", "oi_long_ce", "oi_short_pe", "oi_long_pe",
        "call_width_pct", "put_width_pct", "sell_premium", "buy_premium", "credit",
        "max_risk", "max_profit", "entry_ror_pct", "lot_size", "max_risk_per_lot",
        "max_profit_per_lot", "exit_value", "pnl", "pnl_per_lot", "ror_pct", "max_dd_pct",
        "outcome"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ror-gate", type=float, default=75.0)
    args = ap.parse_args()

    print("loading ...", flush=True)
    m = pd.read_parquet(ROOT / "data/intraday/n50_candidate_marks.parquet")
    m = m[m["mark"].isin([ENTRY_MARK, PRE_MARK])]
    w = m.pivot_table(index=["symbol", "expiry", "strike", "opt_type", "date"],
                      columns="mark", values=["price", "cum_vol", "oi"], aggfunc="last")
    w.columns = [f"{a}|{b}" for a, b in w.columns]
    w = w[w[f"price|{ENTRY_MARK}"].notna()]
    px = w[f"price|{ENTRY_MARK}"].to_dict()
    oi = w[f"oi|{ENTRY_MARK}"].to_dict()
    traded = ((w[f"cum_vol|{ENTRY_MARK}"] - w[f"cum_vol|{PRE_MARK}"]).fillna(0) > 0).to_dict()

    eq = pd.read_parquet(ROOT / "data/intraday/n50_equity_5m.parquet")
    eq["date"] = eq["timestamp"].dt.strftime("%Y-%m-%d")
    eq["hhmm"] = eq["timestamp"].dt.strftime("%H:%M")
    e_at = eq[eq["hhmm"] == "14:30"].set_index(["symbol", "date"])["close"]
    e_cl = eq[eq["hhmm"] == "15:25"].set_index(["symbol", "date"])["close"]

    pan = pd.read_parquet(ROOT / "bw_panel.parquet", columns=["date", "symbol", "underlying"])
    pan["date"] = pd.to_datetime(pan["date"]).dt.strftime("%Y-%m-%d")
    u_eod = pan.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["underlying"]
    # strikes are unadjusted while Upstox equity history is split-adjusted, so anchor the level
    # to the EOD panel and take only the scale-invariant intraday ratio from the equity feed
    u1430 = (u_eod * (e_at / e_cl).dropna()).dropna().to_dict()

    from koscine.config import SILVER_DATA_ROOT
    ls = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet",
                         columns=["symbol", "expiry_month", "lot"])
    lots = ls.set_index(["symbol", pd.to_datetime(ls["expiry_month"]).dt.to_period("M")])["lot"].to_dict()

    from koscine3.data.sources import load_market_data
    mkt = load_market_data(columns=["date", "symbol", "atm_iv"])
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values(["symbol", "date"])
    mkt["iv_ratio"] = mkt.groupby("symbol")["atm_iv"].transform(
        lambda s: s / s.rolling(252, min_periods=60).median())
    mkt["ds"] = mkt["date"].dt.strftime("%Y-%m-%d")
    IV = mkt.set_index(["symbol", "ds"])["iv_ratio"].to_dict()

    groups = json.loads((LOCK_V2 / "universe_groups.json").read_text())
    g2 = {s: g for g, syms in groups.items() for s in syms}

    avail = (w.reset_index().groupby(["symbol", "date", "expiry", "opt_type"])["strike"]
             .apply(lambda s: np.sort(s.unique())).to_dict())
    exp_by: dict = {}
    for (s, d, e, _) in avail:
        exp_by.setdefault((s, d), set()).add(e)
    print(f"  {len(w):,} contract-days at {ENTRY_MARK}\n", flush=True)

    def snap(sym, d, exp, ot, target, u, strict):
        st = avail.get((sym, d, exp, ot))
        if st is None or not len(st):
            return None
        k = st[np.abs(st - target).argmin()]
        if strict and abs(k - target) / u * 100 > MAX_SNAP_ERR_PCT:
            return None
        return float(k)

    def oi_lots(key, lot):
        v = oi.get(key)
        if v is None or pd.isna(v) or not lot or v <= 0:
            return None
        return round(float(v) / float(lot))

    days = sorted({d for (_, d) in u1430})
    dpos = {d: i for i, d in enumerate(days)}
    prev_day = {d: days[i - 1] for i, d in enumerate(days) if i}

    cands = []
    for (sym, d), u in u1430.items():
        exps = sorted(exp_by.get((sym, d), ()))
        exp = next((e for e in exps if (pd.Timestamp(e) - pd.Timestamp(d)).days >= DTE_MIN), None)
        if exp is None:
            continue
        lot = lots.get((sym, pd.Timestamp(exp).to_period("M")))
        if lot is None or pd.isna(lot):
            continue
        lot = float(lot)
        for tier, w_ce, w_pe in TIERS:
            ks = [snap(sym, d, exp, "CE", u * (1 + SHORT_OTM), u, True),
                  snap(sym, d, exp, "CE", u * (1 + SHORT_OTM + w_ce), u, False),
                  snap(sym, d, exp, "PE", u * (1 - SHORT_OTM), u, True),
                  snap(sym, d, exp, "PE", u * (1 - SHORT_OTM - w_pe), u, False)]
            if any(k is None for k in ks) or ks[0] == ks[1] or ks[2] == ks[3]:
                continue
            sides = ["CE", "CE", "PE", "PE"]
            keys = [(sym, exp, ks[i], sides[i], d) for i in range(4)]
            if any(k not in px for k in keys) or not all(traded.get(k, False) for k in keys):
                continue
            p = [px[k] for k in keys]
            sell_prem, buy_prem = p[0] + p[2], p[1] + p[3]
            credit = sell_prem - buy_prem
            call_w, put_w = ks[1] - ks[0], ks[2] - ks[3]
            width = max(call_w, put_w)
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
                continue
            if credit * lot < MIN_PROFIT_PER_LOT:
                continue
            pv = prev_day.get(d)
            cands.append({
                "tier": tier, "symbol": sym, "group": g2.get(sym, "C_top50oi"),
                "signal_date": d, "entry_date": d, "expiry": exp,
                "dte": int((pd.Timestamp(exp) - pd.Timestamp(d)).days),
                "underlying": round(u, 1),
                "iv_ratio": (round(float(IV[(sym, pv)]), 2)
                             if pv and (sym, pv) in IV and pd.notna(IV[(sym, pv)]) else None),
                "short_ce": ks[0], "long_ce": ks[1], "short_pe": ks[2], "long_pe": ks[3],
                "oi_short_ce": oi_lots(keys[0], lot), "oi_long_ce": oi_lots(keys[1], lot),
                "oi_short_pe": oi_lots(keys[2], lot), "oi_long_pe": oi_lots(keys[3], lot),
                "call_width_pct": round(call_w / u * 100, 1),
                "put_width_pct": round(put_w / u * 100, 1),
                "sell_premium": round(sell_prem, 2), "buy_premium": round(buy_prem, 2),
                "credit": round(credit, 2), "max_risk": round(risk, 2),
                "max_profit": round(credit, 2),
                "entry_ror_pct": round(credit / risk * 100, 1),
                "lot_size": int(lot), "max_risk_per_lot": round(risk * lot, 1),
                "max_profit_per_lot": round(credit * lot, 1),
                "_width": width, "_credit": credit, "_risk": risk, "_lot": lot,
            })

    c = pd.DataFrame(cands)
    print(f"fillable candidates clearing structure + profit/lot: {len(c):,}")
    c = c[c["entry_ror_pct"] > args.ror_gate]
    print(f"clearing ROR gate {args.ror_gate:.0f}%: {len(c):,}")
    sel = c.loc[c.groupby(["tier", "entry_date"])["entry_ror_pct"].idxmax()].reset_index(drop=True)
    print(f"selected (1 per tier per day): {len(sel):,}\n")

    rows = []
    for _, r in sel.iterrows():
        i0 = dpos[r["entry_date"]]
        legs = [(r["short_ce"], "CE", 1), (r["long_ce"], "CE", -1),
                (r["short_pe"], "PE", 1), (r["long_pe"], "PE", -1)]
        vals, dd, exited = [], 0.0, False
        for k in range(1, FWD + 1):
            if i0 + k >= len(days):
                break
            d2 = days[i0 + k]
            if (pd.Timestamp(r["expiry"]) - pd.Timestamp(d2)).days <= SAFE_DTE:
                exited = True
            v, ok = 0.0, True
            for st, ot, sg in legs:
                key = (r["symbol"], r["expiry"], st, ot, d2)
                if key not in px:
                    ok = False
                    break
                v += sg * px[key]
            if ok:
                val = max(0.0, min(r["_width"], v))
                vals.append(val)
                dd = min(dd, r["_credit"] - val)
            if exited:
                break
        if not vals:
            continue
        complete = exited or len(vals) == FWD
        exit_value = vals[-1]
        pnl = r["_credit"] - exit_value
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec.update({
            "exit_value": round(exit_value, 2), "pnl": round(pnl, 2),
            "pnl_per_lot": round(pnl * r["_lot"], 1),
            "ror_pct": round(pnl / r["_risk"] * 100, 1),
            "max_dd_pct": round(dd / r["_risk"] * 100, 1),
            "outcome": ("win" if pnl > 0 else "loss") if complete else "pending",
        })
        rows.append(rec)

    t = pd.DataFrame(rows)[COLS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    done = t[t["outcome"] != "pending"]
    summ = done.groupby("tier").apply(lambda g: pd.Series({
        "n": len(g),
        "win": round((g["outcome"] == "win").mean(), 2),
        "ev_ror": round(g["ror_pct"].mean(), 1),
        "median_ror": round(g["ror_pct"].median(), 1),
        "worst_ror": round(g["ror_pct"].min(), 1),
        "worst_dd": round(g["max_dd_pct"].min(), 1),
    }), include_groups=False)
    print("=" * 78)
    print(f"BROKEN-WING, 14:30 signal / 15:15 entry (ROR gate {args.ror_gate:.0f}%)")
    print("=" * 78)
    print(summ.to_string())
    print(f"\n  total pnl_per_lot   Rs.{done['pnl_per_lot'].sum():,.0f}")
    print(f"  mean  pnl_per_lot   Rs.{done['pnl_per_lot'].mean():,.0f}")
    print(f"  mean  max_profit/lot Rs.{done['max_profit_per_lot'].mean():,.0f}  "
          f"(captured {done['pnl_per_lot'].mean()/done['max_profit_per_lot'].mean()*100:.1f}% of it)")
    print(f"  mean  max_risk/lot  Rs.{done['max_risk_per_lot'].mean():,.0f}")
    print(f"  return on risk      "
          f"{done['pnl_per_lot'].sum()/done['max_risk_per_lot'].sum()*100:.1f}%")
    print(f"\nwrote {len(t):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
