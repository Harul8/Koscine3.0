"""Dual-gate flag: fire when BOTH P(move>=thresh) AND P(top-5 mover) are high.
top-30 uses P(>=5%); next-35 uses P(>=10%); both also require high P(top5-mover).
No daily cap; weekly (stock,side) cooldown. Sweep gate tightness -> find ~150 trades/yr.
Fast: volume + hit-rate only (no option payoff).
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

TRAIN_END = pd.Timestamp("2023-12-31")
WEEKLY = 5


def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)


def main():
    print("equity + features ...", flush=True)
    market = load_market_data()
    reg = build_feature_registry(market)
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    rk = uni.set_index(uni["symbol"].astype(str))["rank"]
    bof = {s: ("A" if r <= 30 else "B") for s, r in rk.items()}
    ds = build_supervised_dataset(market, uni, reg); feats = model_feature_columns(reg, ds)
    ds["symbol"] = ds["symbol"].astype(str)
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    oc["rank"] = oc.groupby(["date", "side"])["ceiling"].rank(method="min", ascending=False)
    df = ds.merge(oc, on=["date", "symbol", "side"], how="inner")
    df["bucket"] = df["symbol"].map(bof)
    tr, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END].copy()

    print("fit P(>=5%), P(>=10%), P(top5) ...", flush=True)
    for side in ("long", "short"):
        t = tr[tr["side"].eq(side)]
        imp = SimpleImputer(strategy="median").fit(_clean(t, feats)); Xt = imp.transform(_clean(t, feats))
        for name, lab in [("c5", t["ceiling"] >= 0.05), ("c10", t["ceiling"] >= 0.10), ("ct5", t["rank"] <= 5)]:
            clf = LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(Xt, lab.astype(int))
            m = evl["side"].eq(side)
            evl.loc[m, name] = clf.predict_proba(imp.transform(_clean(evl[m], feats)))[:, 1]
            tr.loc[tr["side"].eq(side), name] = clf.predict_proba(Xt)[:, 1]

    cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    evl["conf_move"] = np.where(evl["bucket"].eq("A"), evl["c5"], evl["c10"])
    tr["conf_move"] = np.where(tr["bucket"].eq("A"), tr["c5"], tr["c10"])

    def thresholds(p):
        th = {}
        for b in ("A", "B"):
            for side in ("long", "short"):
                g = tr[tr["bucket"].eq(b) & tr["side"].eq(side)]
                th[(b, side)] = (np.quantile(g["conf_move"], p), np.quantile(g["ct5"], p))
        return th

    def run(p):
        th = thresholds(p)
        flagged = evl[evl.apply(lambda r: (r["conf_move"] >= th[(r["bucket"], r["side"])][0])
                                and (r["ct5"] >= th[(r["bucket"], r["side"])][1]), axis=1)].copy()
        flagged["idx"] = flagged["date"].map(lambda d: pos[pd.Timestamp(d)])
        flagged = flagged.sort_values(["symbol", "side", "idx"])
        keep, last = [], {}
        for row in flagged.itertuples(index=False):
            k = (row.symbol, row.side)
            if row.idx - last.get(k, -10**9) > WEEKLY:
                keep.append(row); last[k] = row.idx
        sel = pd.DataFrame(keep)
        if sel.empty:
            return None
        sel["year"] = pd.to_datetime(sel["date"]).dt.year
        full = sel[sel["year"].isin([2024, 2025])]
        per_year = len(full) / 2
        def hit(b, thr):
            d = sel[sel["bucket"].eq(b)]
            return round((d["ceiling"] >= thr).mean() * 100, 1) if len(d) else np.nan
        return {"pctile": int(p*100), "total": len(sel), "per_year": round(per_year, 0),
                "A_n": int((sel["bucket"]=="A").sum()), "B_n": int((sel["bucket"]=="B").sum()),
                "A_hit5%": hit("A", 0.05), "B_hit10%": hit("B", 0.10),
                "top5_rate": round((sel["rank"] <= 5).mean()*100, 1),
                "top3_rate": round((sel["rank"] <= 3).mean()*100, 1),
                "stocks": sel["symbol"].nunique()}

    rows = [r for p in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94] if (r := run(p))]
    pd.set_option("display.width", 200)
    print("\n===== DUAL-GATE (P(move) high AND P(top5) high), weekly cooldown, no daily cap =====")
    print("GOAL: >=100/yr (pref ~150).  hit = % of picks reaching its bucket threshold.")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
