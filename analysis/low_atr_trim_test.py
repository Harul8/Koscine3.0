"""Does removing low-ATR (rarely-move) stocks from TRAINING help predict big moves on top-30?
Long side. Targets: ceiling>=5% and >=10%. Train BROAD (all ~450) vs trims that drop the
lowest-median-ATR symbols. Evaluate on the top-30 trade set: AUC + daily-pick hit-rate.
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
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

TRAIN_END = pd.Timestamp("2023-12-31")
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)

market = load_market_data(); reg = build_feature_registry(market); feats = reg.feature_columns
oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
oc = oc[(oc.status == "evaluated") & (oc.side == "long")][["date", "symbol", "ceiling", "atr_pct"]].copy()
oc["symbol"] = oc["symbol"].astype(str)
mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
df = oc.merge(mk[["date", "symbol", *feats]], on=["date", "symbol"], how="left")

top30 = set(build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))["symbol"].astype(str))
train = df[df.date <= TRAIN_END]
# per-symbol median ATR% over the training period
med_atr = train.groupby("symbol")["atr_pct"].median().sort_values()
print(f"median ATR% range across {len(med_atr)} stocks: "
      f"min {med_atr.min()*100:.2f}%  p25 {med_atr.quantile(.25)*100:.2f}%  median {med_atr.median()*100:.2f}%  max {med_atr.max()*100:.2f}%")
print(f"top-30 trade stocks: median ATR% range {med_atr[med_atr.index.isin(top30)].min()*100:.2f}%–{med_atr[med_atr.index.isin(top30)].max()*100:.2f}% "
      f"({(med_atr[med_atr.index.isin(top30)] < med_atr.quantile(.25)).sum()} of 30 are in the low-ATR bottom-quartile)")

def fit_eval(keep_syms, thr, label, drop_desc):
    tr = train[train.symbol.isin(keep_syms)].copy(); tr["y"] = (tr["ceiling"] >= thr).astype(int)
    imp = SimpleImputer(strategy="median").fit(_clean(tr, feats))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                         colsample_bytree=0.85, class_weight="balanced", random_state=17,
                         verbosity=-1).fit(imp.transform(_clean(tr, feats)), tr["y"])
    ev = df[(df.date > TRAIN_END) & (df.symbol.isin(top30))].copy(); ev["y"] = (ev["ceiling"] >= thr).astype(int)
    ev["p"] = clf.predict_proba(imp.transform(_clean(ev, feats)))[:, 1]
    top = ev.sort_values("p", ascending=False).groupby("date").head(1)
    return {"target": f">={int(thr*100)}%", "train_set": label, "dropped": drop_desc,
            "train_stocks": tr.symbol.nunique(), "AUC": round(roc_auc_score(ev["y"], ev["p"]), 4),
            "daily_hit": round(top["y"].mean()*100, 1)}

all_syms = set(med_atr.index)
keep75 = set(med_atr[med_atr >= med_atr.quantile(.25)].index)   # drop bottom 25% ATR
keep50 = set(med_atr[med_atr >= med_atr.quantile(.50)].index)   # drop bottom 50% ATR
rows = []
for thr in (0.05, 0.10):
    rows.append(fit_eval(all_syms, thr, "BROAD (all)", "none"))
    rows.append(fit_eval(keep75, thr, "drop low-ATR 25%", f"{len(all_syms)-len(keep75)} stocks"))
    rows.append(fit_eval(keep50, thr, "drop low-ATR 50%", f"{len(all_syms)-len(keep50)} stocks"))
pd.set_option("display.width", 200)
print("\n=== removing low-ATR stocks from training, evaluated on top-30 ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\nIf 'drop' rows >= BROAD on AUC/daily_hit, removing low-ATR stocks helps. Else keep them.")
