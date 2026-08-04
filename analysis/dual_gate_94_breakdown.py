"""94th-percentile dual-gate flags (~138/yr): underlying-outcome breakdown.
 - closed opposite ('hit stop loss' proxy: 5d close against the trade)
 - closed in same direction
 - mean favourable peak among >5% moves vs 0-5% moves
 - adverse excursion (how far against) for context
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

TRAIN_END = pd.Timestamp("2023-12-31"); WEEKLY = 5; PCTILE = 0.94
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)

market = load_market_data(); reg = build_feature_registry(market)
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
rk = uni.set_index(uni["symbol"].astype(str))["rank"]
bof = {s: ("A_top30" if r <= 30 else "B_next35") for s, r in rk.items()}
ds = build_supervised_dataset(market, uni, reg); feats = model_feature_columns(reg, ds)
ds["symbol"] = ds["symbol"].astype(str)
oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
oc = oc[oc["status"].eq("evaluated")][["date","symbol","side","ceiling","floor_depth","entry_open","window_end_date"]].copy()
oc["symbol"] = oc["symbol"].astype(str)
oc["rank"] = oc.groupby(["date","side"])["ceiling"].rank(method="min", ascending=False)
df = ds.merge(oc, on=["date","symbol","side"], how="inner", suffixes=("","_oc"))
for c in ("entry_open","window_end_date"):
    if f"{c}_oc" in df.columns: df[c] = df[f"{c}_oc"]
df["bucket"] = df["symbol"].map(bof)
tr, evl = df[df["date"] <= TRAIN_END].copy(), df[df["date"] > TRAIN_END].copy()

for side in ("long","short"):
    t = tr[tr["side"].eq(side)]; imp = SimpleImputer(strategy="median").fit(_clean(t, feats)); Xt = imp.transform(_clean(t, feats))
    for name, lab in [("c5", t["ceiling"]>=0.05), ("c10", t["ceiling"]>=0.10), ("ct5", t["rank"]<=5)]:
        clf = LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(Xt, lab.astype(int))
        evl.loc[evl["side"].eq(side), name] = clf.predict_proba(imp.transform(_clean(evl[evl["side"].eq(side)], feats)))[:,1]
        tr.loc[tr["side"].eq(side), name] = clf.predict_proba(Xt)[:,1]
evl["conf_move"] = np.where(evl["bucket"].eq("A_top30"), evl["c5"], evl["c10"])
tr["conf_move"] = np.where(tr["bucket"].eq("A_top30"), tr["c5"], tr["c10"])
th = {(b,s): (np.quantile(tr[(tr.bucket==b)&(tr.side==s)]["conf_move"], PCTILE), np.quantile(tr[(tr.bucket==b)&(tr.side==s)]["ct5"], PCTILE))
      for b in ("A_top30","B_next35") for s in ("long","short")}
flag = evl.apply(lambda r: (r["conf_move"]>=th[(r["bucket"],r["side"])][0]) and (r["ct5"]>=th[(r["bucket"],r["side"])][1]), axis=1)

cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i,d in enumerate(cal)}
sel = evl[flag].copy(); sel["idx"] = sel["date"].map(lambda d: pos[pd.Timestamp(d)])
sel = sel.sort_values(["symbol","side","idx"]); keep, last = [], {}
for row in sel.itertuples(index=False):
    k=(row.symbol,row.side)
    if row.idx - last.get(k,-10**9) > WEEKLY: keep.append(row); last[k]=row.idx
sel = pd.DataFrame(keep)

# signed 5d close direction
cl = market.set_index(["date","symbol"])["close"]
sel["win_close"] = sel.apply(lambda r: cl.get((pd.Timestamp(r["window_end_date"]), r["symbol"]), np.nan), axis=1)
sel["closed_fav"] = np.where(sel["side"].eq("long"), sel["win_close"] > sel["entry_open"], sel["win_close"] < sel["entry_open"])

def block(d, lbl):
    big = d[d["ceiling"] >= 0.05]; mid = d[(d["ceiling"] > 0) & (d["ceiling"] < 0.05)]
    return {"book": lbl, "n": len(d),
            "closed_SAME_dir": f"{int(d['closed_fav'].sum())} ({d['closed_fav'].mean()*100:.0f}%)",
            "closed_OPPOSITE": f"{int((~d['closed_fav']).sum())} ({(~d['closed_fav']).mean()*100:.0f}%)",
            "n_>5%": len(big), "mean_move_>5%": f"{big['ceiling'].mean()*100:.1f}%" if len(big) else "-",
            "n_0-5%": len(mid), "mean_move_0-5%": f"{mid['ceiling'].mean()*100:.1f}%" if len(mid) else "-",
            "mean_adverse": f"{d['floor_depth'].mean()*100:.1f}%"}
rows = [block(sel, "ALL"), block(sel[sel.bucket.eq("A_top30")], "A_top30 @5%"), block(sel[sel.bucket.eq("B_next35")], "B_next35 @10%")]
pd.set_option("display.width", 220)
print(f"\n94th-pctile dual-gate: {len(sel)} trades ({round(len(sel[sel.date.dt.year.isin([2024,2025])])/2)}/yr)")
print("\n===== UNDERLYING OUTCOME BREAKDOWN =====")
print(pd.DataFrame(rows).to_string(index=False))
print("\n'closed OPPOSITE' = 5-day close went against the trade (the option-loss case).")
print("'mean_move' = favourable peak (ceiling). 'mean_adverse' = avg worst move against entry.")

# CLOSE-based (signed 5-day close return), distinct from the peak/ceiling numbers above
sel["scr"] = np.where(sel["side"].eq("long"),
                      (sel["win_close"] - sel["entry_open"]) / sel["entry_open"],
                      (sel["entry_open"] - sel["win_close"]) / sel["entry_open"])
print("\n===== CLOSE-BASED (signed 5-day CLOSE return, not peak) =====")
for lbl, d in [("ALL", sel), ("A_top30", sel[sel.bucket.eq("A_top30")]), ("B_next35", sel[sel.bucket.eq("B_next35")])]:
    c5 = (d["scr"] >= 0.05); c10 = (d["scr"] >= 0.10)
    favmean = d.loc[d["scr"] > 0, "scr"].mean() * 100
    print(f"  {lbl:9s} n={len(d):3d} | closed>=5%: {int(c5.sum()):3d} ({c5.mean()*100:.0f}%) | "
          f"closed>=10%: {int(c10.sum()):3d} ({c10.mean()*100:.0f}%) | mean close (fav only): {favmean:.1f}%")

print("\n===== STOCK DIVERSITY (of the 295 trades) =====")
vc = sel["symbol"].value_counts()
print(f"  distinct stocks: {sel['symbol'].nunique()}/65  "
      f"(A_top30: {sel[sel.bucket.eq('A_top30')]['symbol'].nunique()}/30, "
      f"B_next35: {sel[sel.bucket.eq('B_next35')]['symbol'].nunique()}/35)")
print(f"  most-traded stock: {vc.index[0]} = {vc.iloc[0]} trades ({vc.iloc[0]/len(sel)*100:.0f}%) | "
      f"top-5 stocks = {vc.head(5).sum()/len(sel)*100:.0f}% of trades")
freq = vc.value_counts().sort_index()
print(f"  selection frequency (stocks selected k times): {dict(freq)}")
print("\n  top 15 stocks by trade count:")
print(vc.head(15).to_string())
