"""Does the credit-spread book work on INDEX options, where the single-stock version died?

Every failure of the single-stock Broken-Wing book traced to option illiquidity: four thin legs,
a break-even slippage of 0.43% per leg, and a thinnest leg that often did not trade for minutes.
Index options are the natural test of whether that was the strategy or the instrument -- they are
far more liquid, so if the economics work anywhere they work here.

Runs the same structure (2% OTM shorts, tiered wings, one pick per tier per day, 5-day hold with
a SAFE_DTE backstop, entry and exits at the same clock mark) on the 5-minute index option chain,
and prices the result through the same cost model.

Reports the LIQUIDITY comparison explicitly -- traded volume in the entry bar is what decides
whether the break-even slippage is reachable, and it is the whole reason to look at indices.

Usage:
    python analysis/index_credit_spread_1515.py                     # NIFTY
    python analysis/index_credit_spread_1515.py --index banknifty
    python analysis/index_credit_spread_1515.py --entry 15:15
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")

SHORT_OTM = 0.02
# index vol is far lower than single-stock, so the single-stock 6/8/10% put wings would sit at
# strikes that barely trade. Tiers here are scaled to index moves while keeping the same shape
# (narrow call wing, wider put wing -- the persistent put skew the stock book exploits).
TIERS = [("t1_1x2", 0.01, 0.02), ("t2_15x3", 0.015, 0.03), ("t3_1x4", 0.01, 0.04)]
DTE_MIN, FWD, SAFE_DTE = 7, 5, 4
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 0.35     # index strike spacing is tight (50pt on ~24000 = 0.2%)

BROKERAGE_PER_ORDER = 20.0
STT_SELL_PCT = 0.0625 / 100
EXCH_TXN_PCT = 0.03503 / 100
SEBI_PCT = 10.0 / 1e7
STAMP_PCT = 0.003 / 100
GST_PCT = 0.18


def load_chain(index: str, entry: str) -> pd.DataFrame:
    d = ROOT / "data" / "intraday" / f"{index}_option_chain_5m"
    files = sorted(glob.glob(str(d / "*.parquet")))
    if not files:
        raise SystemExit(f"no chain data in {d} -- run the downloader for {index} first")
    keep = ["timestamp", "expiry", "strike", "underlying_close",
            "ce_close", "ce_volume", "ce_lot_size", "pe_close", "pe_volume", "pe_lot_size"]
    out = []
    for f in files:
        x = pd.read_parquet(f, columns=keep)
        x = x[x["timestamp"].dt.strftime("%H:%M") == entry]
        if not x.empty:
            out.append(x)
    c = pd.concat(out, ignore_index=True)
    c["date"] = c["timestamp"].dt.strftime("%Y-%m-%d")
    c["expiry"] = pd.to_datetime(c["expiry"]).dt.strftime("%Y-%m-%d")
    return c


def costs_per_lot(sell_prem, buy_prem, exit_val, credit, lot, n_legs=4):
    gross_entry = (sell_prem + buy_prem) * lot
    scale = np.clip(exit_val / np.maximum(credit, 1e-9), 0.0, 3.0)
    turnover = gross_entry + gross_entry * scale
    brokerage = BROKERAGE_PER_ORDER * n_legs * 2
    stt = STT_SELL_PCT * turnover * 0.5
    exch = EXCH_TXN_PCT * turnover
    sebi = SEBI_PCT * turnover
    stamp = STAMP_PCT * turnover * 0.5
    gst = GST_PCT * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="nifty")
    ap.add_argument("--entry", default="15:15")
    ap.add_argument("--ror-gate", type=float, default=75.0)
    args = ap.parse_args()

    print(f"loading {args.index} chain at {args.entry} ...", flush=True)
    c = load_chain(args.index, args.entry)
    print(f"  {len(c):,} chain rows, {c['date'].nunique():,} days, "
          f"{c['expiry'].nunique()} expiries\n", flush=True)

    ce = {(r.date, r.expiry, r.strike): (r.ce_close, r.ce_volume)
          for r in c[c["ce_close"].notna()].itertuples()}
    pe = {(r.date, r.expiry, r.strike): (r.pe_close, r.pe_volume)
          for r in c[c["pe_close"].notna()].itertuples()}
    lot = float(c["ce_lot_size"].dropna().iloc[0])
    spot = c.drop_duplicates(["date"]).set_index("date")["underlying_close"].to_dict()
    strikes: dict = {}
    for r in c.itertuples():
        strikes.setdefault((r.date, r.expiry), []).append(r.strike)
    strikes = {k: np.sort(np.unique(v)) for k, v in strikes.items()}
    exp_by: dict = {}
    for (d, e) in strikes:
        exp_by.setdefault(d, set()).add(e)
    days = sorted(exp_by)
    dpos = {d: i for i, d in enumerate(days)}
    print(f"  lot size {lot:.0f}", flush=True)

    def snap(d, e, tgt, u, strict):
        st = strikes.get((d, e))
        if st is None or not len(st):
            return None
        k = st[np.abs(st - tgt).argmin()]
        if strict and abs(k - tgt) / u * 100 > MAX_SNAP_ERR_PCT:
            return None
        return float(k)

    cands = []
    for d in days:
        u = spot.get(d)
        if u is None or not np.isfinite(u):
            continue
        exps = sorted(exp_by[d])
        e = next((x for x in exps if (pd.Timestamp(x) - pd.Timestamp(d)).days >= DTE_MIN), None)
        if e is None:
            continue
        for tier, wce, wpe in TIERS:
            sce = snap(d, e, u * (1 + SHORT_OTM), u, True)
            lce = snap(d, e, u * (1 + SHORT_OTM + wce), u, False)
            spe = snap(d, e, u * (1 - SHORT_OTM), u, True)
            lpe = snap(d, e, u * (1 - SHORT_OTM - wpe), u, False)
            if None in (sce, lce, spe, lpe) or sce == lce or spe == lpe:
                continue
            legs = [ce.get((d, e, sce)), ce.get((d, e, lce)),
                    pe.get((d, e, spe)), pe.get((d, e, lpe))]
            if any(x is None for x in legs):
                continue
            vols = [x[1] for x in legs]
            if not all(v and v > 0 for v in vols):        # every leg must trade in the entry bar
                continue
            p = [x[0] for x in legs]
            sell_prem, buy_prem = p[0] + p[2], p[1] + p[3]
            credit = sell_prem - buy_prem
            width = max(lce - sce, spe - lpe)
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
                continue
            cands.append({"tier": tier, "date": d, "expiry": e,
                          "dte": int((pd.Timestamp(e) - pd.Timestamp(d)).days),
                          "underlying": u, "sce": sce, "lce": lce, "spe": spe, "lpe": lpe,
                          "sell_premium": sell_prem, "buy_premium": buy_prem,
                          "credit": credit, "width": width, "max_risk": risk,
                          "entry_ror_pct": credit / risk * 100, "lot_size": lot,
                          "min_leg_vol": min(vols)})

    a = pd.DataFrame(cands)
    if a.empty:
        raise SystemExit("no candidates")
    print("=" * 92)
    print(f"{args.index.upper()} INDEX -- fillable spreads at {args.entry}: {len(a):,}")
    print("=" * 92)
    print("  achievable entry ROR: " + "  ".join(
        f"p{int(k*100)}={v:.0f}%" for k, v in a["entry_ror_pct"].quantile([.5, .75, .9, .99]).items()))
    print(f"  median volume on the THINNEST leg in the entry bar: {a['min_leg_vol'].median():,.0f}")
    print(f"  (single-stock equivalent was frequently 0 -- that is the whole comparison)")

    sel_pool = a[a["entry_ror_pct"] > args.ror_gate]
    print(f"\n  clearing ROR gate {args.ror_gate:.0f}%: {len(sel_pool):,}")
    if sel_pool.empty:
        print("  -> nothing clears the gate; falling back to the top decile by ROR")
        sel_pool = a[a["entry_ror_pct"] >= a["entry_ror_pct"].quantile(0.90)]
    sel = sel_pool.loc[sel_pool.groupby(["tier", "date"])["entry_ror_pct"].idxmax()]
    print(f"  selected (1 per tier per day): {len(sel):,}\n")

    rows = []
    for _, r in sel.iterrows():
        i0 = dpos[r["date"]]
        vals, dd, exited = [], 0.0, False
        for k in range(1, FWD + 1):
            if i0 + k >= len(days):
                break
            d2 = days[i0 + k]
            if (pd.Timestamp(r["expiry"]) - pd.Timestamp(d2)).days <= SAFE_DTE:
                exited = True
            lg = [ce.get((d2, r["expiry"], r["sce"])), ce.get((d2, r["expiry"], r["lce"])),
                  pe.get((d2, r["expiry"], r["spe"])), pe.get((d2, r["expiry"], r["lpe"]))]
            if all(x is not None for x in lg):
                v = (lg[0][0] - lg[1][0]) + (lg[2][0] - lg[3][0])
                val = max(0.0, min(r["width"], v))
                vals.append(val)
                dd = min(dd, r["credit"] - val)
            if exited:
                break
        if not vals:
            continue
        complete = exited or len(vals) == FWD
        ev = vals[-1]
        pnl = r["credit"] - ev
        rows.append({**r.to_dict(), "exit_value": ev, "pnl": pnl,
                     "pnl_per_lot": pnl * r["lot_size"],
                     "max_risk_per_lot": r["max_risk"] * r["lot_size"],
                     "max_profit_per_lot": r["credit"] * r["lot_size"],
                     "ror_pct": pnl / r["max_risk"] * 100,
                     "max_dd_pct": dd / r["max_risk"] * 100,
                     "outcome": ("win" if pnl > 0 else "loss") if complete else "pending"})

    t = pd.DataFrame(rows)
    t = t[t["outcome"] != "pending"]
    out = ROOT / "data" / "intraday" / f"{args.index}_index_1515_trades.csv"
    t.to_csv(out, index=False)

    gross = t["pnl_per_lot"].sum()
    risk = t["max_risk_per_lot"].sum()
    t["stat_cost"] = costs_per_lot(t["sell_premium"], t["buy_premium"], t["exit_value"],
                                   t["credit"], t["lot_size"])
    print("=" * 92)
    print(f"{args.index.upper()} RESULTS ({len(t)} trades, entry {args.entry})")
    print("=" * 92)
    print(t.groupby("tier").apply(lambda g: pd.Series({
        "n": len(g), "win": round((g["outcome"] == "win").mean(), 2),
        "ev_ror": round(g["ror_pct"].mean(), 1),
        "median_ror": round(g["ror_pct"].median(), 1),
        "worst_ror": round(g["ror_pct"].min(), 1),
        "worst_dd": round(g["max_dd_pct"].min(), 1)}), include_groups=False).to_string())
    print(f"\n  win rate            {(t['outcome'] == 'win').mean()*100:.1f}%")
    print(f"  gross total PnL/lot Rs.{gross:,.0f}   ({gross/risk*100:.2f}% of risk)")
    print(f"  mean gross PnL/lot  Rs.{t['pnl_per_lot'].mean():,.0f}")
    print(f"  mean statutory cost Rs.{t['stat_cost'].mean():,.0f}")
    net_stat = gross - t["stat_cost"].sum()
    print(f"  net of statutory    Rs.{net_stat:,.0f}   ({net_stat/risk*100:.2f}% of risk)")

    print(f"\n  {'slippage/leg':>13} {'NET PnL/lot':>14} {'net/risk':>10} {'win rate':>9}")
    print("  " + "-" * 50)
    gt = (t["sell_premium"] + t["buy_premium"]) * t["lot_size"]
    sc = np.clip(t["exit_value"] / np.maximum(t["credit"], 1e-9), 0, 3)
    for slip in (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02):
        net = t["pnl_per_lot"] - t["stat_cost"] - slip * (gt + gt * sc)
        print(f"  {slip*100:>12.2f}% {net.sum():>14,.0f} {net.sum()/risk*100:>9.2f}% "
              f"{(net > 0).mean()*100:>8.1f}%")
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if (t["pnl_per_lot"] - t["stat_cost"] - mid * (gt + gt * sc)).sum() > 0:
            lo = mid
        else:
            hi = mid
    print(f"\n  BREAK-EVEN slippage: {lo*100:.3f}% per leg per side")
    print(f"  (single-stock Broken-Wing broke even at 0.43%)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
