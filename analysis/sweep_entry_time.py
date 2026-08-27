"""Which time of day is the best moment to ENTER a broken-wing credit spread?

Re-runs the whole system independently at each of 12 session marks. At every mark the strikes
are chosen off THAT moment's underlying, the credit is THAT moment's quoted credit, the
fillability gate uses volume traded in the window ending at that mark, and the exit walk prices
at the SAME mark on later days -- so each row is a self-consistent strategy, not one set of
trades re-priced.

Motivation: 09:15 is not executable in practice (the opening minutes are wide and volatile), and
the 2026-08 pilot showed the 09:15 print is not fillable at all on ~55% of trades. So the real
question is which PRACTICAL moment retains the most value.

Fairness note: the marks are unevenly spaced, so a "traded in the preceding window" gate is more
permissive where the lead-in is 60 minutes than at the 5-minute one. Volume per MINUTE in that
window is reported alongside, so the liquidity comparison stays honest.

Usage:
    python analysis/sweep_entry_time.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))

MARKS = ["09:15", "09:20", "09:30", "10:00", "11:00", "12:00",
         "13:00", "14:00", "14:30", "15:00", "15:15", "15:25"]
# minutes of trading in the window ending at each mark (09:15 is the opening bar itself)
WIN_MIN = {"09:15": 1, "09:20": 5, "09:30": 10, "10:00": 30, "11:00": 60, "12:00": 60,
           "13:00": 60, "14:00": 60, "14:30": 30, "15:00": 30, "15:15": 15, "15:25": 10}

SHORT_OTM = 0.02
TIERS = [("t1_2x6", 0.02, 0.06), ("t2_3x8", 0.03, 0.08), ("t3_2x10", 0.02, 0.10)]
DTE_MIN, FWD, SAFE_DTE = 7, 5, 4
MIN_PROFIT_PER_LOT = 10_000.0
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 1.0
ROR_GATE = 75.0     # the achievable band established by validate_1515_entry_system.py; the
                     # inherited 150% bar is unreachable at genuinely transactable prices


def main() -> None:
    print("loading ...", flush=True)
    m = pd.read_parquet(ROOT / "data/intraday/n50_candidate_marks.parquet")
    w = m.pivot_table(index=["symbol", "expiry", "strike", "opt_type", "date"],
                      columns="mark", values=["price", "cum_vol"], aggfunc="last")
    w.columns = [f"{a}|{b}" for a, b in w.columns]

    eq = pd.read_parquet(ROOT / "data/intraday/n50_equity_5m.parquet")
    eq["date"] = eq["timestamp"].dt.strftime("%Y-%m-%d")
    eq["hhmm"] = eq["timestamp"].dt.strftime("%H:%M")
    eq_at = {mk: eq[eq["hhmm"] == mk].set_index(["symbol", "date"])["close"] for mk in MARKS}
    eq_close = eq[eq["hhmm"] == "15:25"].set_index(["symbol", "date"])["close"]

    pan = pd.read_parquet(ROOT / "bw_panel.parquet", columns=["date", "symbol", "underlying"])
    pan["date"] = pd.to_datetime(pan["date"]).dt.strftime("%Y-%m-%d")
    u_eod = pan.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["underlying"]

    from koscine.config import SILVER_DATA_ROOT
    ls = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet",
                         columns=["symbol", "expiry_month", "lot"])
    lots = ls.set_index(["symbol", pd.to_datetime(ls["expiry_month"]).dt.to_period("M")])["lot"].to_dict()
    print(f"  {len(w):,} contract-days\n", flush=True)

    rows, all_trades = [], []
    for mi, mk in enumerate(MARKS):
        pcol, vcol = f"price|{mk}", f"cum_vol|{mk}"
        if pcol not in w.columns:
            continue
        sub = w[w[pcol].notna()]
        px = sub[pcol].to_dict()
        prev = MARKS[mi - 1] if mi else None
        if prev and f"cum_vol|{prev}" in sub.columns:
            vol_win = (sub[vcol] - sub[f"cum_vol|{prev}"]).fillna(0)
        else:
            vol_win = sub[vcol].fillna(0)
        liq = (vol_win > 0).to_dict()
        volrate = (vol_win / WIN_MIN[mk]).to_dict()

        # underlying at this mark, anchored to the unadjusted EOD level -- Upstox equity history
        # is split/bonus adjusted but option strikes are not, so only the intraday RATIO from the
        # equity feed is scale-safe (see validate_1515_entry_system.py)
        ratio = (eq_at[mk] / eq_close).dropna()
        u_mk = (u_eod * ratio).dropna().to_dict()

        avail = (sub.reset_index().groupby(["symbol", "date", "expiry", "opt_type"])["strike"]
                 .apply(lambda s: np.sort(s.unique())).to_dict())
        exp_by: dict = {}
        for (s, d, e, _) in avail:
            exp_by.setdefault((s, d), set()).add(e)

        def snap(sym, d, exp, ot, target, u, strict):
            st = avail.get((sym, d, exp, ot))
            if st is None or not len(st):
                return None
            k = st[np.abs(st - target).argmin()]
            if strict and abs(k - target) / u * 100 > MAX_SNAP_ERR_PCT:
                return None
            return float(k)

        cands = []
        for (sym, d), u in u_mk.items():
            exps = sorted(exp_by.get((sym, d), ()))
            exp = next((e for e in exps
                        if (pd.Timestamp(e) - pd.Timestamp(d)).days >= DTE_MIN), None)
            if exp is None:
                continue
            lot = lots.get((sym, pd.Timestamp(exp).to_period("M")))
            if lot is None or pd.isna(lot):
                continue
            for tier, w_ce, w_pe in TIERS:
                ks = [snap(sym, d, exp, "CE", u * (1 + SHORT_OTM), u, True),
                      snap(sym, d, exp, "CE", u * (1 + SHORT_OTM + w_ce), u, False),
                      snap(sym, d, exp, "PE", u * (1 - SHORT_OTM), u, True),
                      snap(sym, d, exp, "PE", u * (1 - SHORT_OTM - w_pe), u, False)]
                if any(k is None for k in ks) or ks[0] == ks[1] or ks[2] == ks[3]:
                    continue
                sides = ["CE", "CE", "PE", "PE"]
                keys = [(sym, exp, ks[i], sides[i], d) for i in range(4)]
                if any(k not in px for k in keys):
                    continue
                if not all(liq.get(k, False) for k in keys):
                    continue
                p = [px[k] for k in keys]
                credit = (p[0] + p[2]) - (p[1] + p[3])
                width = max(ks[1] - ks[0], ks[2] - ks[3])
                risk = width - credit
                if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
                    continue
                if credit * lot < MIN_PROFIT_PER_LOT:
                    continue
                cands.append({"tier": tier, "symbol": sym, "date": d, "expiry": exp,
                              "sce": ks[0], "lce": ks[1], "spe": ks[2], "lpe": ks[3],
                              "credit": credit, "width": width, "max_risk": risk,
                              "entry_ror_pct": credit / risk * 100, "lot": lot,
                              "volrate": min(volrate.get(k, 0) for k in keys)})
        c = pd.DataFrame(cands)
        if c.empty:
            print(f"  {mk}: no candidates", flush=True)
            continue
        med_ror = c["entry_ror_pct"].median()
        med_vr = c["volrate"].median()
        n_cand = len(c)
        c = c[c["entry_ror_pct"] > ROR_GATE]
        if c.empty:
            print(f"  {mk}: none clear {ROR_GATE:.0f}% ROR", flush=True)
            continue
        sel = c.loc[c.groupby(["tier", "date"])["entry_ror_pct"].idxmax()]

        days = sorted({d for (_, d) in u_mk})
        dpos = {d: i for i, d in enumerate(days)}
        out = []
        for _, r in sel.iterrows():
            i0 = dpos.get(r["date"])
            if i0 is None:
                continue
            legs = [(r["sce"], "CE", 1), (r["lce"], "CE", -1),
                    (r["spe"], "PE", 1), (r["lpe"], "PE", -1)]
            vals, exited = [], False
            for k in range(1, FWD + 1):
                if i0 + k >= len(days):
                    break
                dd = days[i0 + k]
                if (pd.Timestamp(r["expiry"]) - pd.Timestamp(dd)).days <= SAFE_DTE:
                    exited = True
                v, ok = 0.0, True
                for st, ot, sg in legs:
                    key = (r["symbol"], r["expiry"], st, ot, dd)
                    if key not in px:
                        ok = False
                        break
                    v += sg * px[key]
                if ok:
                    vals.append(max(0.0, min(r["width"], v)))
                if exited:
                    break
            if not vals:
                continue
            pnl = r["credit"] - vals[-1]
            out.append({"pnl": pnl, "pnl_lot": pnl * r["lot"],
                        "risk_lot": r["max_risk"] * r["lot"], "win": pnl > 0,
                        "credit": r["credit"], "mark": mk, "date": r["date"],
                        "symbol": r["symbol"], "tier": r["tier"]})
        t = pd.DataFrame(out)
        if t.empty:
            continue
        all_trades.append(t)
        rows.append({
            "mark": mk, "n_cand": n_cand, "med_ror_all": med_ror, "med_volrate": med_vr,
            "trades": len(t), "win_pct": t["win"].mean() * 100,
            "mean_credit": t["credit"].mean(), "mean_pnl_lot": t["pnl_lot"].mean(),
            "total_pnl_lot": t["pnl_lot"].sum(),
            "ret_on_risk": t["pnl_lot"].sum() / t["risk_lot"].sum() * 100,
        })
        print(f"  {mk} done -- {len(t)} trades", flush=True)

    r = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(f"ENTRY-TIME SWEEP  (strikes, credit, fillability and exits all at the stated mark; "
          f"ROR gate {ROR_GATE:.0f}%)")
    print("=" * 110)
    print(f"  {'entry':>7} {'fillable':>9} {'medROR':>7} {'vol/min':>9} {'trades':>7} "
          f"{'win%':>6} {'credit':>8} {'mean/lot':>10} {'total/lot':>12} {'ret/risk':>9}")
    print("  " + "-" * 98)
    for _, x in r.iterrows():
        print(f"  {x['mark']:>7} {x['n_cand']:>9,.0f} {x['med_ror_all']:>6.0f}% "
              f"{x['med_volrate']:>9,.0f} {x['trades']:>7,.0f} {x['win_pct']:>5.1f}% "
              f"{x['mean_credit']:>8.2f} {x['mean_pnl_lot']:>10,.0f} "
              f"{x['total_pnl_lot']:>12,.0f} {x['ret_on_risk']:>8.1f}%")
    r.to_csv(ROOT / "data/intraday/entry_time_sweep.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(
        ROOT / "data/intraday/entry_time_trades.csv", index=False)
    print("\n  fillable = spreads passing structure + liquidity + profit/lot at that mark")
    print("  vol/min  = median per-minute volume on the thinnest leg in the window ending there")
    print("\n-> data/intraday/entry_time_sweep.csv")


if __name__ == "__main__":
    main()
