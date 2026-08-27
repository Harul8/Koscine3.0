"""Validate a 14:30-signal / 15:15-entry Broken-Wing system against the EOD backtest.

This is the test the pilot could not do. The pilot RE-PRICED the trades the EOD backtest had
already chosen; a genuine 15:15 system SELECTS DIFFERENT TRADES, because it picks strikes off
the 14:30 underlying and gates on the credit actually available at 15:15. Both differences
change which symbol wins each day, so only a full re-selection answers the question.

What it replays, per trading day:
  1. underlying at 14:30 (see the split-adjustment note below)
  2. earliest expiry clearing DTE_MIN
  3. per tier, strikes nearest the 2%-OTM short / tier-wing long targets
  4. the credit ACTUALLY quoted at 15:15
  5. the same gates the EOD system uses (stratified ROR, profit/lot), PLUS a fillability gate
     the EOD system never had: every leg must have traded in the 15:00->15:15 window
  6. one pick per tier per day, highest entry ROR -- matching the EOD rule
  7. exit walk on subsequent days at 15:15, with the same 5-day / SAFE_DTE backstops

SPLIT ADJUSTMENT: Upstox equity history is split/bonus-adjusted but option strikes are not
(BAJFINANCE shows an underlying of 868 against 9400 strikes). Comparing them directly would
mis-select strikes for every symbol with a corporate action (~7% of symbol-days: KOTAKBANK 5x,
NESTLEIND 2x, TRENT 1.5x, HDFCBANK/RELIANCE/BAJFINANCE at various dates). So the 14:30 level is
anchored to bw_panel's EOD underlying -- which IS on the strike scale -- and the equity feed is
used only for the scale-invariant intraday ratio:  u_1430 = panel_close * (eq_1430 / eq_1525).

Usage:
    python analysis/validate_1515_entry_system.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))
MARKS = ROOT / "data" / "intraday" / "n50_candidate_marks.parquet"
EQ = ROOT / "data" / "intraday" / "n50_equity_5m.parquet"
PANEL = ROOT / "bw_panel.parquet"
LOCK = ROOT / "locks" / "prod_sell_strategies"

# identical to analysis/gen_broken_wing_signal_history.py
SHORT_OTM = 0.02
TIERS = [("t1_2x6", 0.02, 0.06), ("t2_3x8", 0.03, 0.08), ("t3_2x10", 0.02, 0.10)]
DTE_MIN, FWD, SAFE_DTE = 7, 5, 4
MCAP_MIN_ROR, OTHER_MIN_ROR = 150.0, 155.0
MIN_PROFIT_PER_LOT = 10_000.0
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 1.0    # reject a spread whose nearest available strike is >1% of underlying
                           # off target -- beyond that we would be pricing a different spread


def load() -> tuple:
    print("loading ...", flush=True)
    m = pd.read_parquet(MARKS)
    m = m[m["mark"].isin(["15:00", "15:15"])]
    w = m.pivot_table(index=["symbol", "expiry", "strike", "opt_type", "date"],
                      columns="mark", values=["price", "cum_vol"], aggfunc="last")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna(subset=["price_15:15"])
    w["traded_late"] = (w["cum_vol_15:15"] - w["cum_vol_15:00"]).fillna(0) > 0
    px = w["price_15:15"].to_dict()
    liq = w["traded_late"].to_dict()

    eq = pd.read_parquet(EQ)
    eq["date"] = eq["timestamp"].dt.strftime("%Y-%m-%d")
    eq["hhmm"] = eq["timestamp"].dt.strftime("%H:%M")
    a = eq[eq["hhmm"] == "14:30"].set_index(["symbol", "date"])["close"].rename("e1430")
    b = eq[eq["hhmm"] == "15:25"].set_index(["symbol", "date"])["close"].rename("e1525")
    ratio = pd.concat([a, b], axis=1).dropna()
    ratio = (ratio["e1430"] / ratio["e1525"]).rename("r")

    pan = pd.read_parquet(PANEL, columns=["date", "symbol", "underlying", "expiry"])
    pan["date"] = pd.to_datetime(pan["date"]).dt.strftime("%Y-%m-%d")
    pan["expiry"] = pd.to_datetime(pan["expiry"]).dt.strftime("%Y-%m-%d")
    u_eod = pan.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["underlying"]
    u1430 = (u_eod * ratio).dropna().to_dict()

    from koscine.config import SILVER_DATA_ROOT
    ls = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet",
                         columns=["symbol", "expiry_month", "lot"])
    lots = ls.set_index(["symbol", pd.to_datetime(ls["expiry_month"]).dt.to_period("M")])["lot"].to_dict()

    grp = (w.reset_index().groupby(["symbol", "date", "expiry", "opt_type"])["strike"]
           .apply(lambda s: np.sort(s.unique())).to_dict())

    g2 = pd.read_parquet(LOCK / "broken_wing_candidates_raw.parquet",
                         columns=["symbol", "group"]).drop_duplicates("symbol")
    grp_of = dict(zip(g2["symbol"], g2["group"]))
    print(f"  {len(px):,} contract-days priceable at 15:15", flush=True)
    return px, liq, u1430, lots, grp, grp_of


def main() -> None:
    px, liq, u1430, lots, avail, grp_of = load()

    days = sorted({d for (_, d) in u1430})
    dpos = {d: i for i, d in enumerate(days)}
    sym_days = sorted(u1430.keys(), key=lambda k: (k[1], k[0]))

    def snap(sym, d, exp, ot, target, u, strict):
        """Nearest available strike. `strict` applies only to the SHORT legs: those define the
        spread's character (2% OTM), so pricing one several percent away would be a different
        trade. The long wings are protection -- the EOD system also just takes the nearest
        strike there, and the width used for risk is computed from whatever strike is actually
        picked, so a loose snap stays internally consistent."""
        st = avail.get((sym, d, exp, ot))
        if st is None or not len(st):
            return None
        k = st[np.abs(st - target).argmin()]
        if strict and abs(k - target) / u * 100 > MAX_SNAP_ERR_PCT:
            return None
        return float(k)

    cands, n_no_exp, n_snap_fail, n_illiquid = [], 0, 0, 0
    for (sym, d) in sym_days:
        u = u1430[(sym, d)]
        exps = sorted({e for (s, dd, e, _) in avail if s == sym and dd == d})
        exp = next((e for e in exps
                    if (pd.Timestamp(e) - pd.Timestamp(d)).days >= DTE_MIN), None)
        if exp is None:
            n_no_exp += 1
            continue
        dte = (pd.Timestamp(exp) - pd.Timestamp(d)).days
        lot = lots.get((sym, pd.Timestamp(exp).to_period("M")))
        if lot is None or pd.isna(lot):
            continue
        for tier, w_ce, w_pe in TIERS:
            ks = [snap(sym, d, exp, "CE", u * (1 + SHORT_OTM), u, True),
                  snap(sym, d, exp, "CE", u * (1 + SHORT_OTM + w_ce), u, False),
                  snap(sym, d, exp, "PE", u * (1 - SHORT_OTM), u, True),
                  snap(sym, d, exp, "PE", u * (1 - SHORT_OTM - w_pe), u, False)]
            if any(k is None for k in ks) or ks[0] == ks[1] or ks[2] == ks[3]:
                n_snap_fail += 1
                continue
            sides = ["CE", "CE", "PE", "PE"]
            keys = [(sym, exp, ks[i], sides[i], d) for i in range(4)]
            if any(k not in px for k in keys):
                n_snap_fail += 1
                continue
            if not all(liq.get(k, False) for k in keys):
                n_illiquid += 1
                continue
            p = [px[k] for k in keys]
            credit = (p[0] + p[2]) - (p[1] + p[3])
            width = max(ks[1] - ks[0], ks[2] - ks[3])
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
                continue
            ror = credit / risk * 100
            mppl = credit * lot
            bar = MCAP_MIN_ROR if grp_of.get(sym) == "A_mcap30" else OTHER_MIN_ROR
            # NOTE: the ROR/profit gates are applied later, per threshold, so the distribution of
            # what is actually ACHIEVABLE at 15:15 can be measured rather than assumed. The EOD
            # system's 150%/155% bars were calibrated on 09:15 open prints that the pilot showed
            # are unfillable, so treating them as given would beg the question.
            cands.append({"tier": tier, "symbol": sym, "date": d, "expiry": exp, "dte": dte,
                          "ror_bar": bar,
                          "underlying": u, "sce": ks[0], "lce": ks[1], "spe": ks[2], "lpe": ks[3],
                          "credit": credit, "width": width, "max_risk": risk,
                          "entry_ror_pct": ror, "lot": lot, "max_profit_per_lot": mppl})

    c = pd.DataFrame(cands)
    print(f"\nstructurally-valid, fillable spreads at 15:15: {len(c):,}")
    print(f"  rejected -- no expiry >= DTE {DTE_MIN}: {n_no_exp:,} | "
          f"strike unavailable/off-target: {n_snap_fail:,} | "
          f"leg didn't trade 15:00-15:15: {n_illiquid:,}")
    if c.empty:
        raise SystemExit("no candidates -- nothing to validate")

    print("\n" + "=" * 92)
    print("WHAT ENTRY ROR IS ACTUALLY ACHIEVABLE AT 15:15?")
    print("=" * 92)
    q = c["entry_ror_pct"].quantile([.5, .75, .9, .95, .99, 1.0])
    print("  percentiles: " + "  ".join(f"p{int(k*100)}={v:.0f}%" for k, v in q.items()))
    for bar in (50, 75, 100, 125, 150):
        n = (c["entry_ror_pct"] > bar).sum()
        print(f"  spreads clearing {bar:>3}% ROR: {n:>7,} ({n/len(c)*100:>5.2f}%)  "
              f"on {c.loc[c['entry_ror_pct'] > bar, 'date'].nunique():>4} distinct days")
    print(f"\n  (the EOD system's live bar is {MCAP_MIN_ROR:.0f}%/{OTHER_MIN_ROR:.0f}%, "
          f"calibrated on 09:15 open prints)")

    c = c[c["max_profit_per_lot"] >= MIN_PROFIT_PER_LOT]
    sel = c.loc[c.groupby(["tier", "date"])["entry_ror_pct"].idxmax()].reset_index(drop=True)
    print(f"\nafter profit/lot gate and 1-pick-per-tier-per-day: {len(sel):,} "
          f"(ROR gate applied in the sweep below)")

    # ---- exit walk: same legs, priced at 15:15 on subsequent days -----------
    out = []
    for _, r in sel.iterrows():
        i0 = dpos[r["date"]]
        legs = [(r["sce"], "CE", +1), (r["lce"], "CE", -1),
                (r["spe"], "PE", +1), (r["lpe"], "PE", -1)]
        vals, exited = [], False
        for k in range(1, FWD + 1):
            if i0 + k >= len(days):
                break
            dd = days[i0 + k]
            if (pd.Timestamp(r["expiry"]) - pd.Timestamp(dd)).days <= SAFE_DTE:
                exited = True
            v, ok = 0.0, True
            for strike, ot, sign in legs:
                key = (r["symbol"], r["expiry"], strike, ot, dd)
                if key not in px:
                    ok = False
                    break
                v += sign * px[key]
            if ok:
                vals.append(max(0.0, min(r["width"], v)))
            if exited:
                break
        if not vals:
            continue
        exit_value = vals[-1]
        pnl = r["credit"] - exit_value
        out.append({**r.to_dict(), "exit_value": exit_value, "pnl": pnl,
                    "pnl_per_lot": pnl * r["lot"], "ror_pct": pnl / r["max_risk"] * 100,
                    "outcome": "win" if pnl > 0 else "loss", "n_days_held": len(vals)})

    t = pd.DataFrame(out)
    print(f"resolved trades: {len(t):,}\n")

    print("=" * 92)
    print("15:15-ENTRY SYSTEM (selected at 14:30, entered at 15:15, real quoted prices)")
    print("=" * 92)
    print(f"  {'ROR gate':>9} {'trades':>7} {'win%':>6} {'mean PnL/lot':>13} "
          f"{'total PnL/lot':>14} {'ret on risk':>12}")
    print("  " + "-" * 68)
    for bar in (0, 50, 75, 100, 125, 150):
        s = t[t["entry_ror_pct"] > bar]
        if s.empty:
            print(f"  {bar:>8}% {0:>7}")
            continue
        ror = s["pnl_per_lot"].sum() / (s["max_risk"] * s["lot"]).sum() * 100
        print(f"  {bar:>8}% {len(s):>7,} {(s['outcome'] == 'win').mean()*100:>5.1f}% "
              f"{s['pnl_per_lot'].mean():>13,.0f} {s['pnl_per_lot'].sum():>14,.0f} {ror:>11.1f}%")
    print(f"\n  date range {t['date'].min()} -> {t['date'].max()}")
    print("\n  by tier (no ROR gate):")
    print(t.groupby("tier").agg(n=("pnl", "size"),
                                win=("outcome", lambda s: round((s == "win").mean()*100, 1)),
                                mean_pnl_lot=("pnl_per_lot", lambda s: round(s.mean())),
                                total_pnl_lot=("pnl_per_lot", lambda s: round(s.sum()))).to_string())

    print()
    print("=" * 92)
    print("EOD BACKTEST (09:15 open-print entry) -- same period, for comparison")
    print("=" * 92)
    bw = pd.read_parquet(LOCK / "broken_wing_candidates_raw.parquet")
    bar = bw["group"].eq("A_mcap30").map({True: MCAP_MIN_ROR, False: OTHER_MIN_ROR})
    ok = bw[(bw["entry_ror_pct"] > bar) & (bw["max_profit_per_lot"] >= MIN_PROFIT_PER_LOT)]
    e = ok.loc[ok.groupby(["tier", "entry_date"])["entry_ror_pct"].idxmax()]
    e = e[(e["outcome"] != "pending") & (e["entry_date"] >= t["date"].min())
          & (e["entry_date"] <= t["date"].max())]
    print(f"  trades              {len(e):,}")
    print(f"  win rate            {(e['outcome'] == 'win').mean()*100:.1f}%")
    print(f"  mean PnL            {e['pnl'].mean():.2f}")
    print(f"  mean PnL/lot        Rs.{e['pnl_per_lot'].mean():,.0f}")
    print(f"  total PnL/lot       Rs.{e['pnl_per_lot'].sum():,.0f}")
    print(f"  mean entry ROR      {e['entry_ror_pct'].mean():.1f}%")
    print("\n  (the pilot showed this entry price is unfillable on ~55% of trades)")

    t.to_csv(ROOT / "data" / "intraday" / "validate_1515_trades.csv", index=False)
    print(f"\nper-trade detail -> data/intraday/validate_1515_trades.csv")


if __name__ == "__main__":
    main()
