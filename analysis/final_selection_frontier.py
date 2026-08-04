"""Lock the selection at the achievable precision frontier WITH broad coverage.

Constraint: weekly rule (a (stock,side) at most once per 5 trading days) + a per-symbol
quarterly cap tuned so >= 50% of the 65 universe stocks are selected each FULL year.
Ranks by P(ceiling>=10%) (best precision + best mean move). Reports precision (top3/5/10),
mean move, and per-year stock coverage for several caps -> pick the one meeting >=50%/yr.
Read-only research (no option payoff).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from koscine3.data.feature_registry import build_feature_registry  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.data.universe import UniverseConfig, build_universe  # noqa: E402
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns  # noqa: E402
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-31")
WEEKLY = 5
N_UNIV = 65


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def main() -> None:
    print("equity + features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=N_UNIV))
    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    oc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    oc["rank"] = oc.groupby(["date", "side"])["ceiling"].rank(method="min", ascending=False)
    df = dataset.merge(oc, on=["date", "symbol", "side"], how="inner")
    df["big10"] = (df["ceiling"] >= 0.10).astype(int)
    train, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END].copy()

    print("fit P(>=10%) model ...")
    for side in ("long", "short"):
        tr = train[train["side"].eq(side)]
        imp = SimpleImputer(strategy="median").fit(_clean(tr, feats))
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, class_weight="balanced", random_state=17,
                             verbosity=-1).fit(imp.transform(_clean(tr, feats)), tr["big10"])
        m = evl["side"].eq(side)
        evl.loc[m, "score"] = clf.predict_proba(imp.transform(_clean(evl[m], feats)))[:, 1]

    cal = np.array(sorted(market["date"].unique()))
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}

    def select(per_day, weekly, qtr_cap):
        last, qcnt, out = {}, {}, []
        for day, g in evl.groupby("date"):
            i, yq = pos[pd.Timestamp(day)], (day.year, day.quarter)
            n = 0
            for _, r in g.sort_values("score", ascending=False).iterrows():
                ks, sym = (r["symbol"], r["side"]), r["symbol"]
                if weekly and i - last.get(ks, -10**9) <= WEEKLY:
                    continue
                if qtr_cap and qcnt.get((sym, yq), 0) >= qtr_cap:
                    continue
                out.append(r); last[ks] = i; qcnt[(sym, yq)] = qcnt.get((sym, yq), 0) + 1
                n += 1
                if n >= per_day:
                    break
        return pd.DataFrame(out)

    def report(picks):
        r = picks["rank"]
        by_year = picks.assign(y=picks["date"].dt.year).groupby("y")["symbol"].nunique()
        full = by_year.loc[[y for y in by_year.index if y in (2024, 2025)]]
        cov = {f"cov_{y}": f"{int(by_year[y])} ({int(by_year[y])/N_UNIV*100:.0f}%)" for y in by_year.index}
        return {"n": len(picks), "stocks_total": picks["symbol"].nunique(),
                "min_yr_cov_%": round(full.min()/N_UNIV*100, 0) if len(full) else np.nan,
                "top3": round((r <= 3).mean()*100, 1), "top5": round((r <= 5).mean()*100, 1),
                "top10": round((r <= 10).mean()*100, 1), "mean_move_%": round(picks["ceiling"].mean()*100, 2),
                **cov}

    rows = []
    for cap in [0, 6, 4, 3, 2]:
        label = "weekly only" if cap == 0 else f"weekly + <={cap}/qtr/stock"
        rows.append({"config": label, **report(select(2, True, cap))})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n===== FRONTIER + COVERAGE (pooled 2/day, rank=P(>=10%), per side rank within 65) =====")
    print("GOAL: min full-year coverage >= 50% (>=33 of 65 stocks)")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
