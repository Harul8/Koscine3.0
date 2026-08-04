"""Strategy v3: dual-gate flags (P(move) high AND P(top5) high), no daily cap, ~150/yr.
top-30 uses P(>=5%); next-35 uses P(>=10%); both also require high P(top5-mover).
Gate at PCTILE per (bucket,side). Weekly (stock,side) cooldown. Tradeable-options gate.
Real bhavcopy option payoff. Reports hit-rate + EV by bucket and year.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.feature_registry import build_feature_registry
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from analysis.options_bhavcopy import load_bhavcopy
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

TRAIN_END = pd.Timestamp("2023-12-31")
OTM, MIN_PREM, MIN_UND = 0.03, 2.0, 100.0   # underlying>=100 is the real penny filter; prem floor light
WEEKLY = 5
PCTILE = 0.93


def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)
def _prem(r):
    for c in ("open", "close", "settle"):
        if pd.notna(r[c]) and r[c] > 0: return float(r[c])
    return np.nan


def main():
    print("equity + features ...", flush=True)
    market = load_market_data(); reg = build_feature_registry(market)
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    rk = uni.set_index(uni["symbol"].astype(str))["rank"]
    bof = {s: ("A" if r <= 30 else "B") for s, r in rk.items()}; syms = set(rk.index)
    ds = build_supervised_dataset(market, uni, reg); feats = model_feature_columns(reg, ds)
    ds["symbol"] = ds["symbol"].astype(str)
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date","symbol","side","ceiling","days_to_peak","entry_date","entry_open","window_end_date"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    oc["rank"] = oc.groupby(["date","side"])["ceiling"].rank(method="min", ascending=False)
    df = ds.merge(oc, on=["date","symbol","side"], how="inner", suffixes=("","_oc"))
    for c in ("entry_date","entry_open","window_end_date"):
        if f"{c}_oc" in df.columns: df[c] = df[f"{c}_oc"]
    df["bucket"] = df["symbol"].map(bof)
    tr, evl = df[df["date"] <= TRAIN_END].copy(), df[df["date"] > TRAIN_END].copy()

    print("fit P(>=5%), P(>=10%), P(top5) ...", flush=True)
    for side in ("long","short"):
        t = tr[tr["side"].eq(side)]; imp = SimpleImputer(strategy="median").fit(_clean(t, feats)); Xt = imp.transform(_clean(t, feats))
        for name, lab in [("c5", t["ceiling"]>=0.05), ("c10", t["ceiling"]>=0.10), ("ct5", t["rank"]<=5)]:
            clf = LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(Xt, lab.astype(int))
            evl.loc[evl["side"].eq(side), name] = clf.predict_proba(imp.transform(_clean(evl[evl["side"].eq(side)], feats)))[:,1]
            tr.loc[tr["side"].eq(side), name] = clf.predict_proba(Xt)[:,1]
    evl["conf_move"] = np.where(evl["bucket"].eq("A"), evl["c5"], evl["c10"])
    tr["conf_move"] = np.where(tr["bucket"].eq("A"), tr["c5"], tr["c10"])
    th = {(b, s): (np.quantile(tr[(tr.bucket==b)&(tr.side==s)]["conf_move"], PCTILE),
                   np.quantile(tr[(tr.bucket==b)&(tr.side==s)]["ct5"], PCTILE))
          for b in ("A","B") for s in ("long","short")}

    flag = evl.apply(lambda r: (r["conf_move"]>=th[(r["bucket"],r["side"])][0]) and (r["ct5"]>=th[(r["bucket"],r["side"])][1]), axis=1)
    cand = evl[flag].copy()
    cand["opt_type"] = np.where(cand["side"].eq("long"), "CE", "PE")
    print(f"flagged candidates (pre cooldown/options): {len(cand)}", flush=True)

    cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i,d in enumerate(cal)}
    win = lambda ed: [pd.Timestamp(cal[j]) for j in range(pos[pd.Timestamp(ed)], min(pos[pd.Timestamp(ed)]+5, len(cal)))] if pd.Timestamp(ed) in pos else []
    close_map = market.set_index(["date","symbol"])["close"]
    needed = sorted({d for ed in cand["entry_date"].dropna().unique() for d in win(ed)})
    print(f"loading options for {len(needed)} days ...", flush=True)
    opt = {}
    for k,d in enumerate(needed):
        bc = load_bhavcopy(d)
        if bc.empty: continue
        bc = bc[bc["symbol"].isin(syms) & bc["opt_type"].isin(["CE","PE"])].copy()
        ms = bc["underlying"].isna()
        if ms.any(): bc.loc[ms,"underlying"] = bc.loc[ms].apply(lambda r: close_map.get((pd.Timestamp(d), r["symbol"]), np.nan), axis=1)
        opt[pd.Timestamp(d)] = bc[(bc["strike"]>=bc["underlying"]*0.85)&(bc["strike"]<=bc["underlying"]*1.15)]
        if (k+1)%150==0: print(f"  {k+1}/{len(needed)}", flush=True)

    def econ(p):
        b = opt.get(pd.Timestamp(p["entry_date"]))
        if b is None: return None, "no_bhavcopy"
        ch = b[b["symbol"].eq(p["symbol"]) & b["opt_type"].eq(p["opt_type"])]
        if ch.empty: return None, "no_chain(symbol?)"
        wend = pd.Timestamp(p["window_end_date"]); exps = sorted(e for e in ch["expiry"].dropna().unique() if pd.Timestamp(e)>=wend)
        if ch["expiry"].dropna().empty: return None, "no_expiry"
        expiry = pd.Timestamp(exps[0]) if exps else pd.Timestamp(sorted(ch["expiry"].dropna().unique())[-1])
        ce = ch[ch["expiry"].eq(expiry)]
        if ce.empty: return None, "no_expiry"
        tgt = p["entry_open"]*(1+OTM if p["opt_type"]=="CE" else 1-OTM)
        kr = ce.iloc[(ce["strike"]-tgt).abs().argsort().iloc[0]]; pr, und = _prem(kr), kr["underlying"]
        if not (pd.notna(und) and und>=MIN_UND): return None, "low_underlying"
        if not (pr and pr>=MIN_PREM): return None, "low_premium"
        if not (kr["oi"]>0): return None, "no_oi"
        highs, t5c, pkc = [], np.nan, np.nan; dtp = int(p["days_to_peak"]) if pd.notna(p["days_to_peak"]) else 5
        for j,d in enumerate(win(p["entry_date"]),1):
            bb = opt.get(pd.Timestamp(d))
            if bb is None: continue
            r = bb[bb.symbol.eq(p["symbol"]) & bb.opt_type.eq(p["opt_type"]) & bb.expiry.eq(expiry) & bb.strike.eq(kr["strike"])]
            if r.empty: continue
            r = r.iloc[0]; hi = r["high"] if pd.notna(r["high"]) and r["high"]>0 else r["close"]
            if pd.notna(hi): highs.append(float(hi))
            if j==5 and pd.notna(r["close"]): t5c=float(r["close"])
            if j==dtp and pd.notna(r["close"]): pkc=float(r["close"])
        if not highs: return None, "no_window_data"
        return (pr, max(highs), pkc, t5c), "ok"

    from collections import Counter
    last, trades, reasons = {}, [], Counter()
    n_cool = 0
    for row in cand.sort_values(["symbol","side","date"]).itertuples(index=False):
        i = pos[pd.Timestamp(row.date)]
        if i - last.get((row.symbol,row.side), -10**9) <= WEEKLY: continue
        last[(row.symbol,row.side)] = i   # dedup the FLAG (move) regardless of option outcome
        if len(win(row.entry_date)) < 5: continue
        n_cool += 1
        p = {"symbol":row.symbol,"side":row.side,"opt_type":row.opt_type,"entry_date":row.entry_date,"entry_open":row.entry_open,"window_end_date":row.window_end_date,"days_to_peak":row.days_to_peak}
        e, reason = econ(p); reasons[reason] += 1
        if e is None: continue
        pr,best,pkc,t5c = e
        thr = 0.05 if row.bucket=="A" else 0.10
        trades.append({"date":pd.Timestamp(row.date),"bucket":row.bucket,"symbol":row.symbol,"side":row.side,
            "hit":int(row.ceiling>=thr),"move_%":round(row.ceiling*100,2),"rank":int(row.rank),
            "entry_prem":round(pr,2),"mult_best":round(best/pr,2),
            "mult_peakclose":round(pkc/pr,2) if pd.notna(pkc) else np.nan,
            "mult_t5close":round(t5c/pr,2) if pd.notna(t5c) else np.nan,"year":pd.Timestamp(row.date).year})
    res = pd.DataFrame(trades); res.to_csv(ROOT/"reports"/"strategy_v3_trades.csv", index=False)
    print("\ndistinct flags after weekly cooldown:", n_cool, "| gate reasons:", dict(reasons))

    def blk(d,l):
        pk=d["mult_peakclose"].dropna(); t5=d["mult_t5close"].dropna()
        return {"book":l,"n":len(d),"per_yr":round(len(d[d.year.isin([2024,2025])])/2,0),"stocks":d["symbol"].nunique(),
                "hit_%":round(d["hit"].mean()*100,1),"top3_%":round((d["rank"]<=3).mean()*100,1),"top5_%":round((d["rank"]<=5).mean()*100,1),
                "EV_peak_%":round((pk.clip(0)-1).mean()*100,0),"EV_t5_%":round((t5.clip(0)-1).mean()*100,0),"P>=2x":round((pk>=2).mean()*100,1)}
    rows=[blk(res,"ALL"),blk(res[res.bucket.eq("A")],"A_top30 @5%"),blk(res[res.bucket.eq("B")],"B_next35 @10%")]
    for b in ("A","B"):
        for y in sorted(res["year"].unique()): rows.append(blk(res[res.bucket.eq(b)&res.year.eq(y)], f"{b} {y}"))
    print(f"\nPCTILE={PCTILE} | flagged {len(cand)} -> after cooldown {n_cool} -> option-gate rejected {n_none} -> traded {len(res)}")
    print(f"total trades {len(res)} (~{round(len(res[res.year.isin([2024,2025])])/2)}/yr)")
    print("\n===== STRATEGY v3 (dual-gate) — hit + real option EV =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
