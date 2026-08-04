"""Ground the feature-engineering plan: inventory the columns, flag gaps (earnings/events),
and rank feature importance on the exact targets (top-20 >=5%, 21-50 >=10%)."""
from __future__ import annotations
import sys, re
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

TRAIN_END = pd.Timestamp("2023-12-31")
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)

market = load_market_data()
reg = build_feature_registry(market)
feats = reg.feature_columns
cats = {
    "options/OI/IV": r"opt|oi|pcr|iv|call_wall|put_wall|max_pain|skew|gamma",
    "futures": r"fut",
    "volatility/compression": r"vol|atr|realized|range|dryup|compress|bb|band|nr7|nr4|squeeze",
    "momentum/trend": r"sma|ema|ret_|return|rsi|adx|macd|momentum|slope|dist|breakout|high_|low_",
    "delivery/liquidity": r"deliv|turnover|volume|trades",
    "market/sector/breadth": r"nifty|sector|mkt_|advance|breadth|index",
    "EVENT/earnings/calendar": r"earn|result|event|expiry|days_to|ex_date|dividend|announce|calendar|dow|month|quarter",
}
print(f"total feature columns: {len(feats)}")
seen = set()
for name, pat in cats.items():
    hits = [c for c in feats if re.search(pat, c, re.I)]
    seen |= set(hits)
    print(f"\n[{name}] ({len(hits)}): {hits}")
print(f"\n[uncategorized] ({len(set(feats)-seen)}): {sorted(set(feats)-seen)}")
print("\n*** EARNINGS/EVENT check:", [c for c in market.columns if re.search(r'earn|result|event|announce|expiry|days_to', c, re.I)] or "NONE FOUND <-- likely critical gap")

oc = compute_clean_move_outcomes(market, universe=build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50)), contract=CleanMoveContract())
oc = oc[(oc.status == "evaluated") & (oc.side == "long")][["date", "symbol", "ceiling"]].copy()
oc["symbol"] = oc["symbol"].astype(str)
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50))
rk = uni.set_index(uni["symbol"].astype(str))["rank"]
mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
df = oc.merge(mk[["date", "symbol", *feats]], on=["date", "symbol"], how="left")
df["rank"] = df["symbol"].map(rk)
train = df[df.date <= TRAIN_END]

for lbl, mask, thr in [("top20 >=5%", train["rank"] <= 20, 0.05), ("21-50 >=10%", train["rank"] > 20, 0.10)]:
    t = train[mask].copy(); t["y"] = (t["ceiling"] >= thr).astype(int)
    imp = SimpleImputer(strategy="median").fit(_clean(t, feats))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, importance_type="gain",
                         class_weight="balanced", random_state=17, verbosity=-1).fit(imp.transform(_clean(t, feats)), t["y"])
    fi = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\n===== TOP-25 features by gain — {lbl} (base rate {t['y'].mean()*100:.0f}%) =====")
    print(fi.head(25).round(0).to_string())
