"""Strategy v2: top-30 daily @5%, next-35 OPPORTUNISTIC @10% (high-conviction only).

- Top-30 bucket: rank by P(ceiling>=5%); take top-1/day (daily slot).
- Next-35 bucket: rank by P(ceiling>=10%); fire the top pick ONLY if its P(>=10%) clears a
  high-conviction threshold (calibrated on train to a high fired-hit-rate). Skip the day otherwise.
- Both: tradeable-options gate (prem>=5, OI>0, underlying>=100) + diversity (weekly per (stock,side),
  <=6/quarter per stock). Real bhavcopy option payoff. Reports hit-rate + EV by bucket and year.
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
OTM, MIN_PREM, MIN_UND = 0.03, 5.0, 100.0
WEEKLY, QTR_CAP = 5, 6
CONV_TARGET = 0.45   # next-35 fires only where train P(>=10%) hit-rate among fired >= this


def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)
def _prem(r):
    for c in ("open", "close", "settle"):
        if pd.notna(r[c]) and r[c] > 0: return float(r[c])
    return np.nan
def calib(p, y, t):
    o = np.argsort(-p); ps, ys = p[o], y[o]
    cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(cum >= t)[0]
    return float(ps[ok[-1]]) if len(ok) else float(np.quantile(p, 0.95))


def main():
    print("equity + features ...", flush=True)
    market = load_market_data()
    reg = build_feature_registry(market)
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    rk = uni.set_index(uni["symbol"].astype(str))["rank"]
    bucket_of = {s: ("A_top30" if r <= 30 else "B_next35") for s, r in rk.items()}
    syms = set(rk.index)
    ds = build_supervised_dataset(market, uni, reg); feats = model_feature_columns(reg, ds)
    ds["symbol"] = ds["symbol"].astype(str)
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date","symbol","side","ceiling","days_to_peak","entry_date","entry_open","window_end_date"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    df = ds.merge(oc, on=["date","symbol","side"], how="inner", suffixes=("","_oc"))
    for c in ("entry_date","entry_open","window_end_date"):
        if f"{c}_oc" in df.columns: df[c] = df[f"{c}_oc"]
    df["bucket"] = df["symbol"].map(bucket_of)
    tr, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END].copy()

    print("fit P(>=5%) and P(>=10%) ...", flush=True)
    conv = {}
    for side in ("long","short"):
        t = tr[tr["side"].eq(side)]
        imp = SimpleImputer(strategy="median").fit(_clean(t, feats)); Xt = imp.transform(_clean(t, feats))
        c5 = LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(Xt, (t["ceiling"]>=0.05).astype(int))
        c10 = LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(Xt, (t["ceiling"]>=0.10).astype(int))
        m = evl["side"].eq(side); Xe = imp.transform(_clean(evl[m], feats))
        evl.loc[m, "conf5"] = c5.predict_proba(Xe)[:,1]
        evl.loc[m, "conf10"] = c10.predict_proba(Xe)[:,1]
        # conviction threshold for next-35: among B-bucket train rows, P10 score
        tb = t[t["symbol"].map(bucket_of).eq("B_next35")]
        p10_tr = c10.predict_proba(imp.transform(_clean(tb, feats)))[:,1]
        conv[side] = calib(p10_tr, (tb["ceiling"]>=0.10).astype(int).to_numpy(), CONV_TARGET)
    print("conviction thresholds (next-35 P>=10%):", {k: round(v,3) for k,v in conv.items()}, flush=True)

    cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i,d in enumerate(cal)}
    win = lambda ed: [pd.Timestamp(cal[j]) for j in range(pos[pd.Timestamp(ed)], min(pos[pd.Timestamp(ed)]+5, len(cal)))] if pd.Timestamp(ed) in pos else []
    close_map = market.set_index(["date","symbol"])["close"]
    needed = sorted({d for ed in evl["entry_date"].dropna().unique() for d in win(ed)})
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

    def entry_con(p):
        b = opt.get(pd.Timestamp(p["entry_date"]))
        if b is None: return None
        ch = b[b["symbol"].eq(p["symbol"]) & b["opt_type"].eq(p["opt_type"])]
        if ch.empty: return None
        wend = pd.Timestamp(p["window_end_date"])
        exps = sorted(e for e in ch["expiry"].dropna().unique() if pd.Timestamp(e) >= wend)
        expiry = pd.Timestamp(exps[0]) if exps else pd.Timestamp(sorted(ch["expiry"].dropna().unique())[-1])
        ce = ch[ch["expiry"].eq(expiry)]
        if ce.empty: return None
        tgt = p["entry_open"] * (1+OTM if p["opt_type"]=="CE" else 1-OTM)
        kr = ce.iloc[(ce["strike"]-tgt).abs().argsort().iloc[0]]
        pr, und = _prem(kr), kr["underlying"]
        if not (pr and pr>=MIN_PREM) or not (pd.notna(und) and und>=MIN_UND) or not (kr["oi"]>0): return None
        return {"expiry":expiry, "strike":kr["strike"], "prem":pr, "vol":int(kr["vol"]) if pd.notna(kr["vol"]) else 0}

    def payoff(p, con):
        highs, t5c, pkc = [], np.nan, np.nan
        dtp = int(p["days_to_peak"]) if pd.notna(p["days_to_peak"]) else 5
        for j,d in enumerate(win(p["entry_date"]), 1):
            b = opt.get(pd.Timestamp(d))
            if b is None: continue
            r = b[b["symbol"].eq(p["symbol"]) & b["opt_type"].eq(p["opt_type"]) & b["expiry"].eq(con["expiry"]) & b["strike"].eq(con["strike"])]
            if r.empty: continue
            r = r.iloc[0]; hi = r["high"] if pd.notna(r["high"]) and r["high"]>0 else r["close"]
            if pd.notna(hi): highs.append(float(hi))
            if j==5 and pd.notna(r["close"]): t5c = float(r["close"])
            if j==dtp and pd.notna(r["close"]): pkc = float(r["close"])
        return (max(highs), pkc, t5c) if highs else None

    evl["opt_type"] = np.where(evl["side"].eq("long"), "CE", "PE")
    last, qcnt, trades = {}, {}, []
    for day in sorted(evl["date"].unique()):
        i, yq = pos[pd.Timestamp(day)], (day.year, day.quarter)
        for bucket, score, thr_move, conv_gate in [("A_top30","conf5",0.05,None), ("B_next35","conf10",0.10,conv)]:
            cands = evl[(evl["date"].eq(day)) & (evl["bucket"].eq(bucket))].sort_values(score, ascending=False)
            for _, p in cands.iterrows():
                if conv_gate is not None and p[score] < conv_gate[p["side"]]: break  # not high-conviction -> no trade
                s = p["symbol"]
                if i - last.get((s,p["side"]), -10**9) <= WEEKLY: continue
                if qcnt.get((s,yq),0) >= QTR_CAP: continue
                if len(win(p["entry_date"])) < 5: continue
                con = entry_con(p)
                if con is None: continue
                pay = payoff(p, con)
                if pay is None: continue
                best,pkc,t5c = pay; ep = con["prem"]
                trades.append({"date":pd.Timestamp(day),"bucket":bucket,"symbol":s,"side":p["side"],
                    "conf":round(float(p[score]),3),"hit":int(p["ceiling"]>=thr_move),"move_%":round(float(p["ceiling"])*100,2),
                    "entry_prem":round(ep,2),"mult_best":round(best/ep,2),
                    "mult_peakclose":round(pkc/ep,2) if pd.notna(pkc) else np.nan,
                    "mult_t5close":round(t5c/ep,2) if pd.notna(t5c) else np.nan,"year":pd.Timestamp(day).year})
                last[(s,p["side"])] = i; qcnt[(s,yq)] = qcnt.get((s,yq),0)+1
                break
    res = pd.DataFrame(trades)
    res.to_csv(ROOT/"reports"/"strategy_v2_trades.csv", index=False)

    def blk(d, lbl):
        pk = d["mult_peakclose"].dropna(); t5 = d["mult_t5close"].dropna()
        return {"book":lbl,"n":len(d),"stocks":d["symbol"].nunique(),"hit_rate_%":round(d["hit"].mean()*100,1),
                "EV_peakclose_%":round((pk.clip(0)-1).mean()*100,0),"EV_t5_%":round((t5.clip(0)-1).mean()*100,0),
                "EV_best_%":round((d["mult_best"].clip(0)-1).mean()*100,0),"P>=2x":round((pk>=2).mean()*100,1)}
    rows = [blk(res,"ALL"), blk(res[res.bucket.eq("A_top30")],"A_top30 (daily 5%)"), blk(res[res.bucket.eq("B_next35")],"B_next35 (opp 10%)")]
    days_total = evl["date"].nunique()
    for b in ("A_top30","B_next35"):
        for y in sorted(res["year"].unique()):
            rows.append(blk(res[res.bucket.eq(b) & res.year.eq(y)], f"{b} {y}"))
    print(f"\nnext-35 fired on {res[res.bucket.eq('B_next35')]['date'].nunique()}/{days_total} days "
          f"({res[res.bucket.eq('B_next35')]['date'].nunique()/days_total*100:.0f}%)", flush=True)
    print("\n===== STRATEGY v2 — hit-rate + real option EV =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nby year top30 coverage:", res[res.bucket.eq('A_top30')].groupby('year')['symbol'].nunique().to_dict())
    print("next35 coverage:", res[res.bucket.eq('B_next35')].groupby('year')['symbol'].nunique().to_dict())


if __name__ == "__main__":
    main()
