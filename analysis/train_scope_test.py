"""Does training on ALL ~450 stocks add noise vs training only on the target top-30/40?
Long side, target = ceiling>=5%. Raw (universe-independent) features only.
Model BROAD: train on all eligible stocks <=2023. Model FOCUSED: train on target stocks only.
Both evaluated on the TARGET set (2024-2026): AUC + daily top-pick hit-rate.
Also reports training-row counts (is there enough data for top-30/40?).
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

market = load_market_data()
reg = build_feature_registry(market)
feats = reg.feature_columns  # raw, universe-independent
print(f"raw features: {len(feats)}")

oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
oc = oc[(oc.status == "evaluated") & (oc.side == "long")][["date", "symbol", "ceiling"]].copy()
oc["symbol"] = oc["symbol"].astype(str)
mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
df = oc.merge(mk[["date", "symbol", *feats]], on=["date", "symbol"], how="left")
df["y"] = (df["ceiling"] >= 0.05).astype(int)

top30 = set(build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))["symbol"].astype(str))
top40 = set(build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=40))["symbol"].astype(str))
all_elig = set(df["symbol"].unique())

train = df[df.date <= TRAIN_END]
print("\n=== Q1: training rows (long, <=2023) ===")
for name, s in [("top-30", top30), ("top-40", top40), ("ALL eligible", all_elig)]:
    t = train[train.symbol.isin(s)]
    print(f"  {name:12s}: {len(t):>8,} rows | {t.symbol.nunique()} stocks | {int(t['y'].sum()):>6,} positives ({t['y'].mean()*100:.0f}%)")

def fit_eval(train_syms, target_syms, label):
    tr = train[train.symbol.isin(train_syms)]
    imp = SimpleImputer(strategy="median").fit(_clean(tr, feats))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                         colsample_bytree=0.85, class_weight="balanced", random_state=17,
                         verbosity=-1).fit(imp.transform(_clean(tr, feats)), tr["y"])
    ev = df[(df.date > TRAIN_END) & (df.symbol.isin(target_syms))].copy()
    ev["p"] = clf.predict_proba(imp.transform(_clean(ev, feats)))[:, 1]
    auc = roc_auc_score(ev["y"], ev["p"])
    top = ev.sort_values("p", ascending=False).groupby("date").head(1)   # daily best pick
    return {"model": label, "train_rows": len(tr), "eval_AUC": round(auc, 4),
            "daily_pick_hit>=5%": round(top["y"].mean()*100, 1), "base_rate": round(ev["y"].mean()*100, 1)}

print("\n=== Q2: BROAD (all) vs FOCUSED (target only) training, evaluated on the TARGET set ===")
rows = []
for tgt_name, tgt in [("top-30", top30), ("top-40", top40)]:
    rows.append({"target": tgt_name, **fit_eval(all_elig, tgt, "BROAD (all ~450)")})
    rows.append({"target": tgt_name, **fit_eval(tgt, tgt, f"FOCUSED ({tgt_name})")})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
print("\nHigher AUC / hit-rate = better. If FOCUSED >= BROAD, training on all 450 adds noise.")
