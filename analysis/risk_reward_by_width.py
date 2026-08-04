"""Risk/reward profile of the SELECTED basket (top-3/day per side, ranked by P(clean))
across stop widths 1.0-1.3 x ATR.

For each width, on the selected trades, report under three stop placements + the upside:
  - stopout @ ATR stop  : floor_depth > width*ATR%        (the strategy's own clean stop)
  - breach @ -2%        : floor_depth >= 0.02             (a fixed 2% stop is hit)
  - touched <= entry    : floor_depth > 0                 (price revisited entry; breakeven-stop risk*)
  - mean favorable move : ceiling (window favourable peak vs entry_open)
* breakeven proxy: counts any dip below entry in the window; with EOD data we can't confirm it
  happened only after the trade was first in profit.

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

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-31")
WIDTHS = [1.0, 1.1, 1.2, 1.3]
K = 3  # signals/day per side, ranked by P(clean)


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def _profile(picks: pd.DataFrame, w: float) -> dict:
    fd = picks["floor_depth"]
    clean = fd <= w * picks["atr_pct"]
    return {
        "n": len(picks),
        "atr_stop_%": round(float((w * picks["atr_pct"]).median()) * 100, 2),
        "stopout_atr_%": round(float((~clean).mean()) * 100, 1),
        "breach_-2%_%": round(float((fd >= 0.02).mean()) * 100, 1),
        "touch<=entry_%": round(float((fd > 0).mean()) * 100, 1),
        "meanfav_%": round(float(picks["ceiling"].mean()) * 100, 2),
        "meanfav_clean_%": round(float(picks.loc[clean, "ceiling"].mean()) * 100, 2),
        "medfav_%": round(float(picks["ceiling"].median()) * 100, 2),
    }


def main() -> None:
    print("loading + building features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=100))
    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    outc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    outc = outc[outc["status"].eq("evaluated")][
        ["date", "symbol", "side", "floor_depth", "ceiling", "atr_pct"]
    ].copy()
    outc["symbol"] = outc["symbol"].astype(str)
    df = dataset.merge(outc, on=["date", "symbol", "side"], how="inner")
    train = df[df["date"] <= TRAIN_END]
    evl = df[df["date"] > TRAIN_END]
    print(f"train {len(train):,} | eval {len(evl):,} | feats {len(feats)} | select top-{K}/day/side\n")

    out_rows = {"long": [], "short": []}
    for side in ("long", "short"):
        tr_s = train[train["side"].eq(side)]
        ev_s = evl[evl["side"].eq(side)].copy()
        imp = SimpleImputer(strategy="median").fit(_clean(tr_s, feats))
        Xtr, Xev = imp.transform(_clean(tr_s, feats)), imp.transform(_clean(ev_s, feats))
        for w in WIDTHS:
            ytr = (tr_s["floor_depth"] <= w * tr_s["atr_pct"]).astype(int)
            clf = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(Xtr, ytr)
            ev_s["p_clean"] = clf.predict_proba(Xev)[:, 1]
            picks = ev_s.sort_values("p_clean", ascending=False).groupby("date").head(K)
            row = {"width": f"{w}xATR", **_profile(picks, w)}
            out_rows[side].append(row)

    pd.set_option("display.width", 220)
    for side in ("long", "short"):
        print(f"===== {side.upper()} — selected top-{K}/day, ranked by P(clean) (eval 2024-2026) =====")
        print(pd.DataFrame(out_rows[side]).to_string(index=False))
        print()
    print("stopout_atr = hit the ATR stop | breach_-2% = a fixed 2% stop is hit | "
          "touch<=entry = revisited entry (breakeven risk) | meanfav = favourable peak.")


if __name__ == "__main__":
    main()
