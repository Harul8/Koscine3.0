"""Skew (single-side credit vertical) at 14:30 signal / 15:15 entry, in the SAME schema as
locks/prod_sell_strategies/skew_signal_history.csv.

Companion to gen_1515_signal_history.py, which does the same for Broken-Wing. Skew is the more
interesting candidate to survive the fill-artifact finding: it needs only TWO legs rather than
four, so the "every leg must have traded" gate is far easier to clear, and its EOD history
showed better ROR on less margin.

Side selection: the EOD generator backs out Black-Scholes implied vol for the ~2%-OTM call and
put and sells whichever is richer (skew = ce_iv - pe_iv). Reproduced here with the same solver
on the 15:15 price, so the side chosen is decided the same way -- not by a premium-size proxy,
which would drift from the EOD rule whenever the two sides sit at different distances from spot.

Everything else matches the BW 15:15 script: real quoted 15:15 prices, a fillability gate
requiring both legs to trade in the 15:00-15:15 window, exits marked at 15:15 on later days,
signal_date == entry_date. ROR gate defaults to 75% rather than the inherited 115%, which the
2026-08 work showed is not reachable at transactable prices.

Usage:
    python analysis/gen_1515_skew_history.py
    python analysis/gen_1515_skew_history.py --ror-gate 100
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
LOCK = ROOT / "locks" / "prod_sell_strategies"
LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
OUT = LOCK / "skew_1515_signal_history.csv"

ENTRY_MARK, PRE_MARK = "15:15", "15:00"
SHORT_OTM, FWD, SAFE_DTE = 0.02, 5, 4
DTE_MIN = 7
MIN_PROFIT_PER_LOT = 10_000.0
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 1.0
R = 0.065
TIERS = [("t1_skew35", 0.035), ("t2_skew6", 0.06), ("t3_skew8", 0.08)]

COLS = ["tier", "symbol", "group", "signal_date", "entry_date", "expiry", "dte", "underlying",
        "iv_ratio", "side", "ce_iv", "pe_iv", "skew", "short_strike", "long_strike",
        "oi_short", "oi_long", "sell_premium", "buy_premium", "credit", "max_risk",
        "max_profit", "entry_ror_pct", "lot_size", "max_risk_per_lot", "max_profit_per_lot",
        "exit_value", "pnl", "pnl_per_lot", "ror_pct", "max_dd_pct", "outcome"]


def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, r, is_call):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    return K * math.exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


def implied_vol(price, S, K, T, r, is_call, lo=1e-4, hi=5.0):
    if price <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price < intrinsic:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, r, is_call) > price:
            hi = mid
        else:
            lo = mid
    v = 0.5 * (lo + hi)
    return None if v < 1e-3 or v > 4.9 else v


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
    e14 = eq[eq["hhmm"] == "14:30"].set_index(["symbol", "date"])["close"]
    ecl = eq[eq["hhmm"] == "15:25"].set_index(["symbol", "date"])["close"]
    pan = pd.read_parquet(ROOT / "bw_panel.parquet", columns=["date", "symbol", "underlying"])
    pan["date"] = pd.to_datetime(pan["date"]).dt.strftime("%Y-%m-%d")
    u_eod = pan.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["underlying"]
    u1430 = (u_eod * (e14 / ecl).dropna()).dropna().to_dict()

    from koscine.config import SILVER_DATA_ROOT
    ls = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet",
                         columns=["symbol", "expiry_month", "lot"])
    lots = ls.set_index(["symbol", pd.to_datetime(ls["expiry_month"]).dt.to_period("M")])["lot"].to_dict()

    from koscine3.data.sources import load_market_data
    mkt = load_market_data(columns=["date", "symbol", "atm_iv"]).sort_values(["symbol", "date"])
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["iv_ratio"] = mkt.groupby("symbol")["atm_iv"].transform(
        lambda s: s / s.rolling(252, min_periods=60).median())
    mkt["ds"] = mkt["date"].dt.strftime("%Y-%m-%d")
    IV = mkt.set_index(["symbol", "ds"])["iv_ratio"].to_dict()

    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items()
          for s in syms}
    avail = (w.reset_index().groupby(["symbol", "date", "expiry", "opt_type"])["strike"]
             .apply(lambda s: np.sort(s.unique())).to_dict())
    exp_by: dict = {}
    for (s, d, e, _) in avail:
        exp_by.setdefault((s, d), set()).add(e)
    print(f"  {len(w):,} contract-days at {ENTRY_MARK}\n", flush=True)

    def snap(sym, d, exp, ot, tgt, u, strict):
        st = avail.get((sym, d, exp, ot))
        if st is None or not len(st):
            return None
        k = st[np.abs(st - tgt).argmin()]
        if strict and abs(k - tgt) / u * 100 > MAX_SNAP_ERR_PCT:
            return None
        return float(k)

    days = sorted({d for (_, d) in u1430})
    dpos = {d: i for i, d in enumerate(days)}
    prev_day = {d: days[i - 1] for i, d in enumerate(days) if i}

    cands = []
    for (sym, d), u in u1430.items():
        exps = sorted(exp_by.get((sym, d), ()))
        exp = next((e for e in exps if (pd.Timestamp(e) - pd.Timestamp(d)).days >= DTE_MIN), None)
        if exp is None:
            continue
        dte = int((pd.Timestamp(exp) - pd.Timestamp(d)).days)
        lot = lots.get((sym, pd.Timestamp(exp).to_period("M")))
        if lot is None or pd.isna(lot):
            continue
        lot = float(lot)
        T = dte / 365.0
        sce = snap(sym, d, exp, "CE", u * (1 + SHORT_OTM), u, True)
        spe = snap(sym, d, exp, "PE", u * (1 - SHORT_OTM), u, True)
        if sce is None or spe is None:
            continue
        pce = px.get((sym, exp, sce, "CE", d))
        ppe = px.get((sym, exp, spe, "PE", d))
        if pce is None or ppe is None:
            continue
        ce_iv = implied_vol(pce, u, sce, T, R, True)
        pe_iv = implied_vol(ppe, u, spe, T, R, False)
        if ce_iv is None or pe_iv is None:
            continue
        skew = ce_iv - pe_iv
        side = "CE" if skew > 0 else "PE"
        for tier, wing in TIERS:
            if side == "CE":
                short_k, long_k = sce, snap(sym, d, exp, "CE", u * (1 + SHORT_OTM + wing), u, False)
            else:
                short_k, long_k = spe, snap(sym, d, exp, "PE", u * (1 - SHORT_OTM - wing), u, False)
            if long_k is None or long_k == short_k:
                continue
            ks = (short_k, long_k)
            keys = [(sym, exp, k, side, d) for k in ks]
            if any(k not in px for k in keys) or not all(traded.get(k, False) for k in keys):
                continue
            sell_prem, buy_prem = px[keys[0]], px[keys[1]]
            credit = sell_prem - buy_prem
            width = abs(long_k - short_k)
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
                continue
            if credit * lot < MIN_PROFIT_PER_LOT:
                continue
            pv = prev_day.get(d)

            def oil(key):
                v = oi.get(key)
                return None if v is None or pd.isna(v) or v <= 0 else round(float(v) / lot)

            cands.append({
                "tier": tier, "symbol": sym, "group": g2.get(sym, "C_top50oi"),
                "signal_date": d, "entry_date": d, "expiry": exp, "dte": dte,
                "underlying": round(u, 1),
                "iv_ratio": (round(float(IV[(sym, pv)]), 2)
                             if pv and (sym, pv) in IV and pd.notna(IV[(sym, pv)]) else None),
                "side": side, "ce_iv": round(ce_iv, 3), "pe_iv": round(pe_iv, 3),
                "skew": round(skew, 3), "short_strike": short_k, "long_strike": long_k,
                "oi_short": oil(keys[0]), "oi_long": oil(keys[1]),
                "sell_premium": round(sell_prem, 2), "buy_premium": round(buy_prem, 2),
                "credit": round(credit, 2), "max_risk": round(risk, 2),
                "max_profit": round(credit, 2),
                "entry_ror_pct": round(credit / risk * 100, 1), "lot_size": int(lot),
                "max_risk_per_lot": round(risk * lot, 1),
                "max_profit_per_lot": round(credit * lot, 1),
                "_w": width, "_c": credit, "_r": risk, "_lot": lot, "_ks": ks, "_side": side,
            })

    c = pd.DataFrame(cands)
    print(f"fillable candidates clearing structure + profit/lot: {len(c):,}")
    if c.empty:
        raise SystemExit("none")
    print("  achievable entry ROR: " + "  ".join(
        f"p{int(k*100)}={v:.0f}%" for k, v in c["entry_ror_pct"].quantile([.5, .75, .9, .99]).items()))
    c = c[c["entry_ror_pct"] > args.ror_gate]
    print(f"clearing ROR gate {args.ror_gate:.0f}%: {len(c):,}")
    sel = c.loc[c.groupby(["tier", "entry_date"])["entry_ror_pct"].idxmax()].reset_index(drop=True)
    print(f"selected (1 per tier per day): {len(sel):,}\n")

    rows = []
    for _, r in sel.iterrows():
        i0 = dpos[r["entry_date"]]
        ks, side = r["_ks"], r["_side"]
        vals, dd, exited = [], 0.0, False
        for k in range(1, FWD + 1):
            if i0 + k >= len(days):
                break
            d2 = days[i0 + k]
            if (pd.Timestamp(r["expiry"]) - pd.Timestamp(d2)).days <= SAFE_DTE:
                exited = True
            a_ = px.get((r["symbol"], r["expiry"], ks[0], side, d2))
            b_ = px.get((r["symbol"], r["expiry"], ks[1], side, d2))
            if a_ is not None and b_ is not None:
                val = max(0.0, min(r["_w"], a_ - b_))
                vals.append(val)
                dd = min(dd, r["_c"] - val)
            if exited:
                break
        if not vals:
            continue
        complete = exited or len(vals) == FWD
        exit_value = vals[-1]
        pnl = r["_c"] - exit_value
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec.update({"exit_value": round(exit_value, 2), "pnl": round(pnl, 2),
                    "pnl_per_lot": round(pnl * r["_lot"], 1),
                    "ror_pct": round(pnl / r["_r"] * 100, 1),
                    "max_dd_pct": round(dd / r["_r"] * 100, 1),
                    "outcome": ("win" if pnl > 0 else "loss") if complete else "pending"})
        rows.append(rec)

    t = pd.DataFrame(rows)[COLS]
    t.to_csv(OUT, index=False)
    done = t[t["outcome"] != "pending"]
    summ = done.groupby("tier").apply(lambda g: pd.Series({
        "n": len(g), "win": round((g["outcome"] == "win").mean(), 2),
        "ev_ror": round(g["ror_pct"].mean(), 1),
        "median_ror": round(g["ror_pct"].median(), 1),
        "worst_ror": round(g["ror_pct"].min(), 1),
        "worst_dd": round(g["max_dd_pct"].min(), 1)}), include_groups=False)
    print("=" * 78)
    print(f"SKEW, 14:30 signal / 15:15 entry (ROR gate {args.ror_gate:.0f}%)")
    print("=" * 78)
    print(summ.to_string())
    print(f"\n  total pnl_per_lot    Rs.{done['pnl_per_lot'].sum():,.0f}")
    print(f"  mean  pnl_per_lot    Rs.{done['pnl_per_lot'].mean():,.0f}")
    print(f"  mean  max_profit/lot Rs.{done['max_profit_per_lot'].mean():,.0f}  "
          f"(captured {done['pnl_per_lot'].mean()/done['max_profit_per_lot'].mean()*100:.1f}%)")
    print(f"  mean  max_risk/lot   Rs.{done['max_risk_per_lot'].mean():,.0f}")
    print(f"  return on risk       "
          f"{done['pnl_per_lot'].sum()/done['max_risk_per_lot'].sum()*100:.1f}%")
    print(f"  side split: {dict(done['side'].value_counts())}")
    print(f"\nwrote {len(t):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
