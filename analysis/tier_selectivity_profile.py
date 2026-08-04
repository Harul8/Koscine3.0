"""Per-tier risk/reward when trading only the best 1-3 ideas/day TOTAL (either side),
ranked by confidence x quantum  =  P(clean) * E[ceiling].

Tiers (by median turnover_lacs rank): TOP30 (rank 1-30) and MID 31-100 (rank 31-100).
Sweeps stop width to find where stop-out < 30% with a tolerable stop size per tier.
Read-only research.
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

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-31")
WIDTHS = [0.6, 0.8, 1.0, 1.2]
NS = [1, 2, 3]  # trades/day TOTAL (either side)


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def _profile(picks: pd.DataFrame, w: float) -> dict:
    fd = picks["floor_depth"]
    clean = fd <= w * picks["atr_pct"]
    return {
        "trades": len(picks),
        "n_long": int((picks["side"] == "long").sum()),
        "n_short": int((picks["side"] == "short").sum()),
        "atr_stop_%": round(float((w * picks["atr_pct"]).median()) * 100, 2),
        "stopout_%": round(float((~clean).mean()) * 100, 1),
        "breach_-2%_%": round(float((fd >= 0.02).mean()) * 100, 1),
        "meanfav_%": round(float(picks["ceiling"].mean()) * 100, 2),
        "meanfav_clean_%": round(float(picks.loc[clean, "ceiling"].mean()) * 100, 2),
    }


def main() -> None:
    print("loading + building features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=100))
    rank = universe.set_index(universe["symbol"].astype(str))["rank"]
    tier_of = rank.apply(lambda r: "TOP30" if r <= 30 else "MID_31_100")

    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    outc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    outc = outc[outc["status"].eq("evaluated")][
        ["date", "symbol", "side", "floor_depth", "ceiling", "atr_pct"]
    ].copy()
    outc["symbol"] = outc["symbol"].astype(str)
    df = dataset.merge(outc, on=["date", "symbol", "side"], how="inner")
    df["tier"] = df["symbol"].map(tier_of)
    train = df[df["date"] <= TRAIN_END]
    evl = df[df["date"] > TRAIN_END]

    print("median ATR% by tier (unconditional eval):")
    print(evl.groupby("tier")["atr_pct"].median().mul(100).round(2).to_string())
    print(f"\ntrain {len(train):,} | eval {len(evl):,} | feats {len(feats)}\n")

    # Per-side: fit ceiling regressor once; p_clean classifier per width.
    side_cache = {}
    for side in ("long", "short"):
        tr_s = train[train["side"].eq(side)]
        ev_s = evl[evl["side"].eq(side)].copy()
        imp = SimpleImputer(strategy="median").fit(_clean(tr_s, feats))
        Xtr, Xev = imp.transform(_clean(tr_s, feats)), imp.transform(_clean(ev_s, feats))
        reg = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.85,
                            colsample_bytree=0.85, random_state=17, verbosity=-1).fit(Xtr, tr_s["ceiling"])
        ev_s["e_ceiling"] = np.maximum(reg.predict(Xev), 0)
        side_cache[side] = (tr_s, ev_s, Xtr, Xev)

    results = {"TOP30": [], "MID_31_100": []}
    for w in WIDTHS:
        ev_parts = []
        for side in ("long", "short"):
            tr_s, ev_s, Xtr, Xev = side_cache[side]
            ytr = (tr_s["floor_depth"] <= w * tr_s["atr_pct"]).astype(int)
            clf = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(Xtr, ytr)
            e = ev_s.copy()
            e["p_clean"] = clf.predict_proba(Xev)[:, 1]
            e["score"] = e["p_clean"] * e["e_ceiling"]
            ev_parts.append(e)
        ev_all = pd.concat(ev_parts, ignore_index=True)
        for tier in ("TOP30", "MID_31_100"):
            pool = ev_all[ev_all["tier"].eq(tier)]
            for n in NS:
                picks = pool.sort_values("score", ascending=False).groupby("date").head(n)
                results[tier].append({"width": f"{w}xATR", "N/day": n, **_profile(picks, w)})

    pd.set_option("display.width", 220)
    for tier in ("TOP30", "MID_31_100"):
        print(f"================= TIER {tier} — best N/day TOTAL, rank=P(clean)*E[ceiling] (2024-2026) =================")
        print(pd.DataFrame(results[tier]).to_string(index=False))
        print()
    print("stopout=hit ATR stop | breach_-2%=a fixed 2% stop is hit | meanfav=favourable peak.")


if __name__ == "__main__":
    main()
