"""Frontier: realized STOP-OUT rate vs stop width x selectivity.

Goal: find the operating point where stop-out (1 - clean) < 30%.
Levers: stop width (ATR multiple) and selectivity (rank by P(clean), take few/day).

Reuses real leakage-safe features. Ceiling/floor_depth/atr_pct are width-independent and
computed once; only the `clean` label and its classifier are refit per stop width.
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
WIDTHS = [1.0, 1.1]                    # ATR multiples for the stop
KS = [1, 2, 3, 5]                       # signals/day per side (ranked by P(clean))
EVAL_YEARS = [2024, 2025, 2026]


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


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
    df["year"] = df["date"].dt.year
    train = df[df["date"] <= TRAIN_END]
    evl = df[df["date"] > TRAIN_END]
    print(f"train {len(train):,} | eval {len(evl):,} | feats {len(feats)}")

    rows = []
    for side in ("long", "short"):
        tr_s = train[train["side"].eq(side)]
        ev_s = evl[evl["side"].eq(side)].copy()
        imp = SimpleImputer(strategy="median").fit(_clean(tr_s, feats))
        Xtr, Xev = imp.transform(_clean(tr_s, feats)), imp.transform(_clean(ev_s, feats))
        # ceiling regressor (width-independent) for reporting size of selected moves
        reg = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.85,
                            colsample_bytree=0.85, random_state=17, verbosity=-1)
        reg.fit(Xtr, tr_s["ceiling"].astype(float))
        ev_s["e_ceiling"] = np.maximum(reg.predict(Xev), 0)

        for w in WIDTHS:
            ytr = (tr_s["floor_depth"] <= w * tr_s["atr_pct"]).astype(int)
            clf = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(Xtr, ytr)
            ev_s["p_clean"] = clf.predict_proba(Xev)[:, 1]
            ev_s["clean_w"] = (ev_s["floor_depth"] <= w * ev_s["atr_pct"]).astype(int)
            ev_s["stop_loss_pct"] = w * ev_s["atr_pct"]
            base_clean = ev_s["clean_w"].mean()
            for k in KS:
                picks = ev_s.sort_values("p_clean", ascending=False).groupby("date").head(k)
                rows.append({
                    "side": side, "stop_width": f"{w}xATR",
                    "stop_loss_%": round(float(ev_s["stop_loss_pct"].median()) * 100, 2),
                    "select": f"top{k}/day", "trades": len(picks),
                    "base_cleanrate": round(base_clean, 3),
                    "sel_cleanrate": round(float(picks["clean_w"].mean()), 3),
                    "STOPOUT%": round((1 - float(picks["clean_w"].mean())) * 100, 1),
                    "ceil_clean_med": round(float(picks.loc[picks["clean_w"].eq(1), "ceiling"].median()), 3),
                })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n================= STOP-OUT FRONTIER (eval 2024-2026, ranked by P(clean)) =================")
    for side in ("long", "short"):
        print(f"\n----- {side.upper()} -----")
        print(res[res["side"].eq(side)].drop(columns="side").to_string(index=False))
    print("\nTarget: STOPOUT% < 30. Read across to see trades/day cost and loss-per-stop (stop_loss_%).")


if __name__ == "__main__":
    main()
