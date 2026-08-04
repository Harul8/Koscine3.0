"""Realized option leverage on >=5% moves, TOP-30, strike ~2% away from spot, 2020 + 2024-2026.
Event = (stock, side) with 5-day favourable peak (ceiling) >= 5% (deduped 5 trading days).
Buy the strike nearest to entry_open*1.02 (calls) / entry_open*0.98 (puts).
metric = option %move(at peak) / underlying %move(at peak). Per year: mean/median/max/min.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from analysis.options_bhavcopy import load_bhavcopy

YEARS = [2020, 2024, 2025, 2026]
THRESH, COOLDOWN, MIN_ENTRY_PREM, AWAY = 0.05, 5, 1.0, 0.02


def _prem_open(r):
    for c in ("open", "close", "settle"):
        if pd.notna(r[c]) and r[c] > 0: return float(r[c])
    return np.nan


def main():
    market = load_market_data()
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))
    syms = set(uni["symbol"].astype(str))
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")].copy(); oc["symbol"] = oc["symbol"].astype(str)
    oc["year"] = pd.to_datetime(oc["date"]).dt.year
    ev = oc[(oc["ceiling"] >= THRESH) & (oc["year"].isin(YEARS))].copy()
    ev["opt_type"] = np.where(ev["side"].eq("long"), "CE", "PE")

    cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    ev["idx"] = ev["date"].map(lambda d: pos[pd.Timestamp(d)])
    ev = ev.sort_values(["symbol", "side", "idx"]); keep, last = [], {}
    for r in ev.itertuples(index=False):
        k = (r.symbol, r.side)
        if r.idx - last.get(k, -10**9) > COOLDOWN: keep.append(r); last[k] = r.idx
    ev = pd.DataFrame(keep)
    print(f"deduped >=5% events (top30): {len(ev)} by year {ev['year'].value_counts().sort_index().to_dict()}", flush=True)

    win = lambda ed: [pd.Timestamp(cal[j]) for j in range(pos[pd.Timestamp(ed)], min(pos[pd.Timestamp(ed)] + 5, len(cal)))] if pd.Timestamp(ed) in pos else []
    close_map = market.set_index(["date", "symbol"])["close"]
    per_event = []
    for yr in YEARS:
        evy = ev[ev["year"].eq(yr)]
        needed = sorted({d for _, r in evy.iterrows() for d in win(r["entry_date"])})
        opt = {}
        for d in needed:
            bc = load_bhavcopy(d)
            if bc.empty: continue
            bc = bc[bc["symbol"].isin(syms) & bc["opt_type"].isin(["CE", "PE"])].copy()
            ms = bc["underlying"].isna()
            if ms.any(): bc.loc[ms, "underlying"] = bc.loc[ms].apply(lambda r: close_map.get((pd.Timestamp(d), r["symbol"]), np.nan), axis=1)
            opt[pd.Timestamp(d)] = bc
        n_ok = 0
        for r in evy.itertuples(index=False):
            wd = win(r.entry_date)
            if len(wd) < 5: continue
            b = opt.get(wd[0])
            if b is None: continue
            ch = b[b["symbol"].eq(r.symbol) & b["opt_type"].eq(r.opt_type)]
            if ch.empty: continue
            wend = pd.Timestamp(r.window_end_date)
            exps = sorted(e for e in ch["expiry"].dropna().unique() if pd.Timestamp(e) >= wend)
            if not exps and ch["expiry"].dropna().empty: continue
            expiry = pd.Timestamp(exps[0]) if exps else pd.Timestamp(sorted(ch["expiry"].dropna().unique())[-1])
            ce = ch[ch["expiry"].eq(expiry)]
            ks = sorted(ce["strike"].dropna().unique())
            if not ks: continue
            target = r.entry_open * (1 + AWAY if r.opt_type == "CE" else 1 - AWAY)
            strike = min(ks, key=lambda k: abs(k - target))
            krow = ce[ce["strike"].eq(strike)].iloc[0]
            entry_opt = _prem_open(krow)
            if not (entry_opt and entry_opt >= MIN_ENTRY_PREM): continue
            highs = []
            for d in wd:
                bb = opt.get(pd.Timestamp(d))
                if bb is None: continue
                row = bb[bb.symbol.eq(r.symbol) & bb.opt_type.eq(r.opt_type) & bb.expiry.eq(expiry) & bb.strike.eq(strike)]
                if row.empty: continue
                row = row.iloc[0]; hi = row["high"] if pd.notna(row["high"]) and row["high"] > 0 else row["close"]
                if pd.notna(hi): highs.append(float(hi))
            if not highs: continue
            opt_move = (max(highs) - entry_opt) / entry_opt
            per_event.append({"year": yr, "side": r.side, "symbol": r.symbol, "strike": strike,
                              "entry_open": round(float(r.entry_open), 1), "strike_away_%": round((strike / r.entry_open - 1) * 100, 1),
                              "entry_opt": round(entry_opt, 2), "und_move_%": round(float(r.ceiling) * 100, 2),
                              "opt_move_%": round(opt_move * 100, 1), "ratio": round(opt_move / float(r.ceiling), 2)})
            n_ok += 1
        print(f"  {yr}: {n_ok} events with option data", flush=True)
        opt.clear()

    res = pd.DataFrame(per_event)
    res.to_csv(ROOT / "reports" / "option_leverage_top30_2pct_events.csv", index=False)
    print(f"\nactual mean strike distance from spot: {res['strike_away_%'].mean():.1f}%  (target {AWAY*100:.0f}%)")
    print("\n===== OPTION-MOVE / UNDERLYING-MOVE  (TOP-30, ~2% strike, at peak) =====")
    rows = []
    for yr in YEARS:
        for lbl, d in [("ALL", res[res.year.eq(yr)]), ("long(call)", res[res.year.eq(yr) & res.side.eq("long")]),
                       ("short(put)", res[res.year.eq(yr) & res.side.eq("short")])]:
            if d.empty: continue
            rt = d["ratio"]
            rows.append({"year": yr, "side": lbl, "n": len(d), "stocks": d["symbol"].nunique(),
                         "mean_ratio": round(rt.mean(), 2), "median_ratio": round(rt.median(), 2),
                         "mean_winsor99": round(rt.clip(upper=rt.quantile(.99)).mean(), 2),
                         "max_ratio": round(rt.max(), 1), "min_ratio": round(rt.min(), 2),
                         "mean_und_%": round(d["und_move_%"].mean(), 1), "mean_opt_%": round(d["opt_move_%"].mean(), 0)})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nratio = option %move / underlying %move at peak. Detail: reports/option_leverage_top30_2pct_events.csv")


if __name__ == "__main__":
    main()
