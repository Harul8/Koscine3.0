"""FINAL clean option EV: top-20 @4%, point-in-time tradeable, lean classifier, real payoff.
Eligible = atm_iv present (optionable) AND close>=100 (non-penny). Pick daily top-1 call + top-1 put.
Buy ~2% strike. Real bhavcopy payoff (peak-close realistic / best / t5). EV by year. The money number.
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

TRAIN_END = pd.Timestamp("2023-12-31")
OTM, MIN_PREM, MIN_UND, WEEKLY = 0.02, 1.0, 100.0, 5
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)
def _prem(r):
    for c in ("open", "close", "settle"):
        if pd.notna(r[c]) and r[c] > 0: return float(r[c])
    return np.nan


TIER_THRESH = 0.04   # top-20 @4%

def main():
    market = load_market_data()
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=20))
    syms = set(uni["symbol"].astype(str))
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date","symbol","side","ceiling","days_to_peak","entry_date","entry_open","window_end_date"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
    df = oc.merge(mk[["date","symbol","close",*LEAN]], on=["date","symbol"], how="left")
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(MIN_UND)
    df["opt_type"] = np.where(df["side"].eq("long"), "CE", "PE")
    train, evl = df[df.date <= TRAIN_END], df[df.date > TRAIN_END].copy()

    print("train P(>=5%) per side ...", flush=True)
    for side in ("long","short"):
        t = train[train.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_clean(t, LEAN))
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(imp.transform(_clean(t, LEAN)), (t["ceiling"]>=TIER_THRESH).astype(int))
        m = evl.side.eq(side)
        evl.loc[m, "score"] = clf.predict_proba(imp.transform(_clean(evl[m], LEAN)))[:,1]

    # daily top-1 per side among eligible
    elig = evl[evl["eligible"]].copy()
    picks = elig.sort_values("score", ascending=False).groupby(["date","side"]).head(1)
    print(f"picks: {len(picks)} ({picks.side.value_counts().to_dict()})", flush=True)

    cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i,d in enumerate(cal)}
    win = lambda ed: [pd.Timestamp(cal[j]) for j in range(pos[pd.Timestamp(ed)], min(pos[pd.Timestamp(ed)]+5, len(cal)))] if pd.Timestamp(ed) in pos else []
    close_map = market.set_index(["date","symbol"])["close"]
    needed = sorted({d for ed in picks["entry_date"].dropna().unique() for d in win(ed)})
    print(f"loading options for {len(needed)} days ...", flush=True)
    opt = {}
    for k,d in enumerate(needed):
        bc = load_bhavcopy(d)
        if bc.empty: continue
        bc = bc[bc["symbol"].isin(syms) & bc["opt_type"].isin(["CE","PE"])].copy()
        ms = bc["underlying"].isna()
        if ms.any(): bc.loc[ms,"underlying"] = bc.loc[ms].apply(lambda r: close_map.get((pd.Timestamp(d), r["symbol"]), np.nan), axis=1)
        opt[pd.Timestamp(d)] = bc[(bc["strike"]>=bc["underlying"]*0.85)&(bc["strike"]<=bc["underlying"]*1.15)]
        if (k+1)%120==0: print(f"  {k+1}/{len(needed)}", flush=True)

    trades = []
    for r in picks.itertuples(index=False):
        wd = win(r.entry_date)
        if len(wd) < 5: continue
        b = opt.get(wd[0])
        if b is None: continue
        ch = b[b["symbol"].eq(r.symbol) & b["opt_type"].eq(r.opt_type)]
        if ch.empty: continue
        wend = pd.Timestamp(r.window_end_date)
        exps = sorted(e for e in ch["expiry"].dropna().unique() if pd.Timestamp(e) >= wend)
        if ch["expiry"].dropna().empty: continue
        expiry = pd.Timestamp(exps[0]) if exps else pd.Timestamp(sorted(ch["expiry"].dropna().unique())[-1])
        ce = ch[ch["expiry"].eq(expiry)]
        ks = sorted(ce["strike"].dropna().unique())
        if not ks: continue
        target = r.entry_open*(1+OTM if r.opt_type=="CE" else 1-OTM)
        strike = min(ks, key=lambda k: abs(k-target))
        kr = ce[ce["strike"].eq(strike)].iloc[0]; pr, und = _prem(kr), kr["underlying"]
        if not (pr and pr>=MIN_PREM) or not (pd.notna(und) and und>=MIN_UND) or not (kr["oi"]>0): continue
        highs, t5c, pkc = [], np.nan, np.nan; dtp = int(r.days_to_peak) if pd.notna(r.days_to_peak) else 5
        for j,d in enumerate(wd,1):
            bb = opt.get(pd.Timestamp(d))
            if bb is None: continue
            row = bb[bb.symbol.eq(r.symbol) & bb.opt_type.eq(r.opt_type) & bb.expiry.eq(expiry) & bb.strike.eq(strike)]
            if row.empty: continue
            row = row.iloc[0]; hi = row["high"] if pd.notna(row["high"]) and row["high"]>0 else row["close"]
            if pd.notna(hi): highs.append(float(hi))
            if j==5 and pd.notna(row["close"]): t5c=float(row["close"])
            if j==dtp and pd.notna(row["close"]): pkc=float(row["close"])
        if not highs: continue
        best = max(highs)
        trades.append({"date":pd.Timestamp(r.date),"symbol":r.symbol,"side":r.side,"hit5":int(r.ceiling>=TIER_THRESH),
            "move_%":round(r.ceiling*100,2),"entry_prem":round(pr,2),"mult_best":round(best/pr,2),
            "mult_peak":round(pkc/pr,2) if pd.notna(pkc) else np.nan,"mult_t5":round(t5c/pr,2) if pd.notna(t5c) else np.nan,
            "year":pd.Timestamp(r.date).year})
    res = pd.DataFrame(trades); res.to_csv(ROOT/"reports"/"final_option_ev_trades.csv", index=False)
    print(f"\nmatched trades: {len(res)} | distinct stocks: {res['symbol'].nunique()}")
    def blk(d,l):
        pk=d["mult_peak"].dropna(); t5=d["mult_t5"].dropna()
        return {"book":l,"n":len(d),"stocks":d["symbol"].nunique(),"hit>=thr":round(d["hit5"].mean()*100,1),
                "EV_peak_%":round((pk.clip(0)-1).mean()*100,0),"EV_t5_%":round((t5.clip(0)-1).mean()*100,0),
                "EV_best_%":round((d["mult_best"].clip(0)-1).mean()*100,0),"P>=2x":round((pk>=2).mean()*100,1),"P>=3x":round((pk>=3).mean()*100,1)}
    rows=[blk(res,"ALL"),blk(res[res.side.eq("long")],"long/call"),blk(res[res.side.eq("short")],"short/put")]
    for y in sorted(res.year.unique()): rows.append(blk(res[res.year.eq(y)], f"year {y}"))
    pd.set_option("display.width", 200)
    print("\n===== FINAL CLEAN OPTION EV (top-20 @4%, point-in-time, ~2% strike) =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
