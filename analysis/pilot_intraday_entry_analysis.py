"""Analyse the pilot intraday download (analysis/pilot_intraday_entry_test.py).

Answers, per fired Broken-Wing trade, using REAL 1-minute prices on the entry day:

  A. Is the 09:15 open print tradeable? -- compare the credit implied by the 09:15 print
     against the credit available a few minutes later (09:20/09:30), and against how much
     volume actually traded in that window. A print you cannot follow up on within minutes is
     an artifact; one that persists is a real fill.

  B. What does entry time cost? -- the credit curve across the session (09:15 ... 15:15),
     which directly prices the "signal at 14:30, enter by 15:15" proposal against the
     next-morning-open entry the backtest assumes.

Credit for a broken-wing = (short_ce + short_pe) - (long_ce + long_pe), all at the same minute.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
LOCK = ROOT / "locks" / "prod_sell_strategies"
PILOT = ROOT / "data" / "intraday" / "pilot_bw_legs_1m.parquet"

MCAP_MIN_ROR, OTHER_MIN_ROR = 150.0, 155.0
MIN_PROFIT_PER_LOT = 10_000.0
MARKS = ["09:15", "09:16", "09:20", "09:30", "10:00", "11:00", "12:00",
         "13:00", "14:00", "14:30", "15:00", "15:15", "15:25"]


def fired_trades() -> pd.DataFrame:
    bw = pd.read_parquet(LOCK / "broken_wing_candidates_raw.parquet")
    bar = bw["group"].eq("A_mcap30").map({True: MCAP_MIN_ROR, False: OTHER_MIN_ROR})
    ok = bw[(bw["entry_ror_pct"] > bar) & (bw["max_profit_per_lot"] >= MIN_PROFIT_PER_LOT)]
    sel = ok.loc[ok.groupby(["tier", "entry_date"])["entry_ror_pct"].idxmax()]
    return sel[sel["outcome"] != "pending"].reset_index(drop=True)


def main() -> None:
    if not PILOT.exists():
        raise SystemExit(f"{PILOT} not found -- run analysis/pilot_intraday_entry_test.py first")
    px = pd.read_parquet(PILOT)
    px["hhmm"] = px["timestamp"].dt.strftime("%H:%M")
    px["strike"] = px["strike"].astype(float)
    # one row per contract-minute (the downloader can legitimately see a minute twice only if
    # re-run mid-write; guard anyway so a dup never double-counts a leg)
    px = px.drop_duplicates(["symbol", "expiry", "strike", "opt_type", "entry_date", "timestamp"])

    # last print at or before each mark -> what you could actually transact at that moment
    px = px.sort_values("timestamp")
    key = ["symbol", "expiry", "strike", "opt_type", "entry_date"]
    lut: dict[tuple, pd.DataFrame] = {k: g for k, g in px.groupby(key, sort=False)}
    vol_lut = {k: g for k, g in px.groupby(key, sort=False)}

    sel = fired_trades()
    rows = []
    for _, t in sel.iterrows():
        d, sym, exp = str(t["entry_date"]), t["symbol"], str(t["expiry"])
        legs = [(+1, "CE", t["short_ce"]), (-1, "CE", t["long_ce"]),
                (+1, "PE", t["short_pe"]), (-1, "PE", t["long_pe"])]
        series = {}
        vol915 = []
        ok = True
        for sign, ot, k in legs:
            g = lut.get((sym, exp, float(k), ot, d))
            if g is None or g.empty:
                ok = False
                break
            s = g.set_index("hhmm")["close"]
            series[(sign, ot, k)] = (g, s)
            v = g[g["hhmm"] <= "09:20"]["volume"].sum()
            vol915.append(v)
        if not ok:
            continue

        rec = {"symbol": sym, "entry_date": d, "tier": t["tier"],
               "width": t["credit"] + t["max_risk"], "max_risk": t["max_risk"],
               "lot": t["lot_size"], "pnl_recorded": t["pnl"],
               "vol_first5min_min_leg": min(vol915)}
        for m in MARKS:
            tot = 0.0
            good = True
            for (sign, ot, k), (g, s) in series.items():
                upto = g[g["hhmm"] <= m]
                if upto.empty:
                    good = False
                    break
                tot += sign * float(upto["close"].iloc[-1])
            rec[f"credit_{m}"] = tot if good else None
        # the backtest's own assumption: the OPEN of the 09:15 bar
        tot = 0.0
        for (sign, ot, k), (g, s) in series.items():
            first = g.iloc[0]
            tot += sign * float(first["open"])
        rec["credit_open_print"] = tot
        rows.append(rec)

    a = pd.DataFrame(rows)
    if a.empty:
        raise SystemExit("no trades matched the pilot data")
    print("=" * 104)
    print(f"PILOT: real intraday credit curve on {len(a)} fired Broken-Wing trades")
    print("=" * 104)

    base = a["credit_open_print"]
    print(f"\n  backtest's assumed entry credit (09:15 bar OPEN): {base.mean():8.2f}")
    print(f"\n  {'mark':<8} {'mean credit':>12} {'vs open':>9} {'mean ROR':>10} {'clears 150%':>12}")
    print("  " + "-" * 56)
    for m in MARKS:
        c = a[f"credit_{m}"].dropna()
        if c.empty:
            continue
        idx = a[f"credit_{m}"].notna()
        ror = a.loc[idx, f"credit_{m}"] / (a.loc[idx, "width"] - a.loc[idx, f"credit_{m}"]) * 100
        print(f"  {m:<8} {c.mean():>12.2f} {c.mean()/base[idx].mean()*100-100:>8.1f}% "
              f"{ror.mean():>9.1f}% {(ror > 150).mean()*100:>11.0f}%")

    print()
    print("=" * 104)
    print("A. IS THE OPEN PRINT TRADEABLE?")
    print("=" * 104)
    slip = (a["credit_09:20"] - a["credit_open_print"]) / a["credit_open_print"] * 100
    print(f"  credit at 09:20 vs the 09:15 open print : {slip.mean():+.1f}% mean, "
          f"{slip.median():+.1f}% median")
    print(f"  trades where 09:20 credit is >=10% WORSE : {(slip <= -10).mean()*100:.0f}%")
    print(f"  median traded volume in first 5 min, thinnest leg: "
          f"{a['vol_first5min_min_leg'].median():,.0f}")
    print(f"  trades where thinnest leg traded ZERO in first 5 min: "
          f"{(a['vol_first5min_min_leg'] == 0).mean()*100:.0f}%")

    print()
    print("=" * 104)
    print("B. WHAT WOULD A 15:15 SAME-DAY ENTRY HAVE COLLECTED?")
    print("=" * 104)
    d1 = (a["credit_15:15"] - a["credit_open_print"]) / a["credit_open_print"] * 100
    print(f"  credit at 15:15 vs next-morning open print: {d1.mean():+.1f}% mean")
    ror15 = a["credit_15:15"] / (a["width"] - a["credit_15:15"]) * 100
    roro = base / (a["width"] - base) * 100
    print(f"  mean ROR  open={roro.mean():.1f}%   15:15={ror15.mean():.1f}%")
    print(f"  clears the 150% gate:  open={(roro > 150).mean()*100:.0f}%   "
          f"15:15={(ror15 > 150).mean()*100:.0f}%")

    a.to_csv(ROOT / "data" / "intraday" / "pilot_entry_curve.csv", index=False)
    print(f"\nper-trade detail -> data/intraday/pilot_entry_curve.csv")


if __name__ == "__main__":
    main()
