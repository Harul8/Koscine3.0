"""Re-target the model to predict TOP-MOVER status and test precision targets.

Old target = P(ceiling >= 10%). New target = P(stock is a top-5 mover within its
(date, side) cross-section of the top-65 universe). Rank picks by that probability.

Measures top-1/3/5/10 precision (per side, within the 65 universe) under selection
policies of increasing diversity, vs the user's bar: top3>=20%, top5>=35%, top10>=50%.
Diversity: weekly = a (stock,side) at most once per 5 trading days; +fairness = <=2/month.
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
WEEKLY_COOLDOWN = 5     # trading days; a (stock,side) at most once per week
MONTH_CAP = 2           # fair-representation: <=2 selections per (stock,side) per month
TARGETS = {"top3": 20.0, "top5": 35.0, "top10": 50.0}


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def main() -> None:
    print("equity + features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    oc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    oc["rank"] = oc.groupby(["date", "side"])["ceiling"].rank(method="min", ascending=False)
    oc["is_top5"] = (oc["rank"] <= 5).astype(int)
    df = dataset.merge(oc, on=["date", "symbol", "side"], how="inner")
    df["big10"] = (df["ceiling"] >= 0.10).astype(int)
    train, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END].copy()
    print(f"train {len(train):,} | eval {len(evl):,}")

    for tgt, col in [("topmover", "is_top5"), ("big10", "big10")]:
        for side in ("long", "short"):
            tr = train[train["side"].eq(side)]
            imp = SimpleImputer(strategy="median").fit(_clean(tr, feats))
            clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(imp.transform(_clean(tr, feats)), tr[col])
            m = evl["side"].eq(side)
            evl.loc[m, f"score_{tgt}"] = clf.predict_proba(imp.transform(_clean(evl[m], feats)))[:, 1]

    cal = np.array(sorted(market["date"].unique()))
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}

    def select(score_col, per_day, weekly, month_cap):
        last, mcnt, out = {}, {}, []
        for day, g in evl.groupby("date"):
            i = pos[pd.Timestamp(day)]
            ym = (day.year, day.month)
            n = 0
            for _, r in g.sort_values(score_col, ascending=False).iterrows():
                key = (r["symbol"], r["side"])
                if weekly and i - last.get(key, -10**9) <= WEEKLY_COOLDOWN:
                    continue
                if month_cap and mcnt.get((key, ym), 0) >= month_cap:
                    continue
                out.append(r)
                last[key] = i
                mcnt[(key, ym)] = mcnt.get((key, ym), 0) + 1
                n += 1
                if n >= per_day:
                    break
        return pd.DataFrame(out)

    def precision(picks):
        r = picks["rank"]
        return {"n": len(picks), "stocks": picks["symbol"].nunique(),
                "top1": round((r <= 1).mean()*100, 1), "top3": round((r <= 3).mean()*100, 1),
                "top5": round((r <= 5).mean()*100, 1), "top10": round((r <= 10).mean()*100, 1),
                "mean_move_%": round(picks["ceiling"].mean()*100, 2)}

    configs = [
        ("OLD target P(>=10%), 2/day, no diversity", "score_big10", 2, False, 0),
        ("TOPMOVER target, 2/day, no diversity", "score_topmover", 2, False, 0),
        ("TOPMOVER, 2/day, weekly cooldown", "score_topmover", 2, True, 0),
        ("TOPMOVER, 2/day, weekly + <=2/month", "score_topmover", 2, True, MONTH_CAP),
        ("TOPMOVER, 1/day (most selective), weekly+month", "score_topmover", 1, True, MONTH_CAP),
    ]
    rows = []
    for label, col, per_day, weekly, mcap in configs:
        rows.append({"config": label, **precision(select(col, per_day, weekly, mcap))})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n===== TOP-MOVER PRECISION (rank within 65-universe, per side) =====")
    print(f"TARGET: top3>={TARGETS['top3']:.0f}%  top5>={TARGETS['top5']:.0f}%  top10>={TARGETS['top10']:.0f}%")
    print(res.to_string(index=False))
    print("\nrandom baseline (1 of 65): top3=4.6%  top5=7.7%  top10=15.4%")


if __name__ == "__main__":
    main()
