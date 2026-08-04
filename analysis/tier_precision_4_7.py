"""Precision@1 for top-20 @4% and 21-50 @7%, point-in-time tradeable, train broad, classifier."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

TRAIN_END = pd.Timestamp("2023-12-31")
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)

market = load_market_data()
oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
oc = oc[(oc.status == "evaluated") & (oc.side == "long")][["date", "symbol", "ceiling"]].copy()
oc["symbol"] = oc["symbol"].astype(str)
mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
df = oc.merge(mk[["date", "symbol", "close", *LEAN]], on=["date", "symbol"], how="left")
rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50)).set_index(
    build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50))["symbol"].astype(str))["rank"]
df["rank"] = df["symbol"].map(rk)
df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
df["year"] = df["date"].dt.year
train, evl = df[df.date <= TRAIN_END], df[df.date > TRAIN_END]

rows, byyear = [], []
for tier, mask, thr in [("A_top20 >=4%", df["rank"] <= 20, 0.04), ("B_21-50 >=7%", (df["rank"] > 20) & (df["rank"] <= 50), 0.07)]:
    tr = train[mask.loc[train.index]].copy(); tr["y"] = (tr["ceiling"] >= thr).astype(int)
    imp = SimpleImputer(strategy="median").fit(_clean(tr, LEAN))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                         colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(imp.transform(_clean(tr, LEAN)), tr["y"])
    ev = evl[mask.loc[evl.index] & evl["eligible"]].copy()
    ev["y"] = (ev["ceiling"] >= thr).astype(int)
    ev["p"] = clf.predict_proba(imp.transform(_clean(ev, LEAN)))[:, 1]
    top = ev.sort_values("p", ascending=False).groupby("date").head(1)
    rows.append({"tier": tier, "AUC": round(roc_auc_score(ev["y"], ev["p"]), 4), "base_rate": round(ev["y"].mean()*100, 1),
                 "n_days": len(top), "precision@1": round(top["y"].mean()*100, 1)})
    yb = top.groupby("year")["y"].mean().mul(100).round(1)
    byyear.append((tier, yb.to_dict()))
    # tier-B conviction gate
    if "B_" in tier:
        gate = [{"fire_top_%": int(f*100), "days": int(len(top)*f), "per_yr": round(len(top)*f/2.4),
                 "precision@1": round(top.sort_values("p", ascending=False).head(int(len(top)*f))["y"].mean()*100, 1)}
                for f in [1.0, 0.7, 0.5, 0.35]]

pd.set_option("display.width", 200)
print("===== top-20 @4% & 21-50 @7%, point-in-time tradeable, train broad =====")
print(pd.DataFrame(rows).to_string(index=False))
print("\nprecision@1 by year:")
for tier, yb in byyear: print(f"  {tier}: {yb}")
print("\ntier-B (21-50 @7%) conviction gate:")
print(pd.DataFrame(gate).to_string(index=False))
