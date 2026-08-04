"""GO/NO-GO probe: do leakage-safe features at t predict the clean-move + ceiling target?

Reuses the real feature registry + engineered features from the codebase, attaches the
NEW targets (clean @ 0.6xATR, ceiling), trains side-specific P(clean) + E[ceiling] models,
ranks top-N/day by P(clean)*E[ceiling], and compares realized outcomes vs random-within-universe.

Read-only research. Does NOT touch production runs/ or models/.
Run from terminal:  python analysis/clean_move_predictiveness_probe.py
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

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

WINDOW = 5
ATR_MULT = 0.60
TOP_N_PER_DAY = 5          # selected signals per day (both sides pooled)
TRAIN_END = "2023-12-31"
EVAL_YEARS = [2024, 2025, 2026]
TOP_N_UNIVERSE = 100


def compute_new_targets(base: pd.DataFrame) -> pd.DataFrame:
    """Return long-format (date, symbol, side, clean, ceiling, evaluated) with 0.6xATR clean stop."""
    df = base[["date", "symbol", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["entry_open"] = g["open"].shift(-1)
    df["win_high"] = pd.concat([g["high"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).max(axis=1)
    df["win_low"] = pd.concat([g["low"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).min(axis=1)
    df["n_obs"] = pd.concat([g["close"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).notna().sum(axis=1)
    prev_close = g["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr_pct"] = df.assign(tr=tr).groupby("symbol")["tr"].transform(
        lambda s: s.rolling(14, min_periods=14).mean()) / df["close"]
    evaluated = (df["entry_open"].notna() & (df["n_obs"] == WINDOW) & df["win_high"].notna()
                 & df["win_low"].notna() & df["atr_pct"].notna() & (df["entry_open"] > 0))
    tol = ATR_MULT * df["atr_pct"]

    out = []
    long_floor = (df["entry_open"] - df["win_low"]) / df["entry_open"]
    long_ceil = (df["win_high"] - df["entry_open"]) / df["entry_open"]
    out.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"].astype(str), "side": "long",
                             "clean": (long_floor <= tol).astype(int), "ceiling": long_ceil,
                             "evaluated": evaluated}))
    short_floor = (df["win_high"] - df["entry_open"]) / df["entry_open"]
    short_ceil = (df["entry_open"] - df["win_low"]) / df["entry_open"]
    out.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"].astype(str), "side": "short",
                             "clean": (short_floor <= tol).astype(int), "ceiling": short_ceil,
                             "evaluated": evaluated}))
    res = pd.concat(out, ignore_index=True)
    res.loc[~res["evaluated"], ["clean", "ceiling"]] = np.nan
    return res


def _clean(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def fit_side(train: pd.DataFrame, feats: list[str]):
    imp = SimpleImputer(strategy="median").fit(_clean(train, feats))
    Xtr = imp.transform(_clean(train, feats))
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.045, num_leaves=31,
                         subsample=0.85, colsample_bytree=0.85, class_weight="balanced",
                         random_state=17, verbosity=-1).fit(Xtr, train["clean"].astype(int))
    reg = LGBMRegressor(n_estimators=300, learning_rate=0.045, num_leaves=31,
                        subsample=0.85, colsample_bytree=0.85,
                        random_state=17, verbosity=-1).fit(Xtr, train["ceiling"].astype(float))
    return imp, clf, reg


def main() -> None:
    print("loading market data ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    registry.assert_safe()
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=TOP_N_UNIVERSE))
    base = market[market["symbol"].astype(str).isin(set(universe["symbol"].astype(str)))].copy()

    print("building feature dataset (reuses real registry + engineered features) ...")
    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    targets = compute_new_targets(base)
    df = dataset.merge(targets, on=["date", "symbol", "side"], how="inner")
    df = df[df["evaluated"]].copy()
    df["year"] = df["date"].dt.year

    train = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    print(f"train rows: {len(train):,} | feature count: {len(feats)}")

    models = {s: fit_side(train[train["side"].eq(s)], feats) for s in ("long", "short")}

    # Score eval rows
    scored = []
    for s in ("long", "short"):
        part = df[df["side"].eq(s) & (df["date"] > pd.Timestamp(TRAIN_END))].copy()
        imp, clf, reg = models[s]
        X = imp.transform(_clean(part, feats))
        part["p_clean"] = clf.predict_proba(X)[:, 1]
        part["e_ceiling"] = np.maximum(reg.predict(X), 0)
        part["score"] = part["p_clean"] * part["e_ceiling"]
        scored.append(part)
    ev = pd.concat(scored, ignore_index=True)

    rng = np.random.default_rng(0)
    print("\n===== TOP-N SELECTED vs RANDOM-WITHIN-UNIVERSE (per day) =====")
    rows = []
    for yr in EVAL_YEARS:
        y = ev[ev["year"].eq(yr)]
        if y.empty:
            continue
        sel_clean, sel_ceil, rnd_clean, rnd_ceil, n_sel = [], [], [], [], 0
        for _, day in y.groupby("date"):
            top = day.nlargest(TOP_N_PER_DAY, "score")
            n_sel += len(top)
            sel_clean.append(top["clean"].mean()); sel_ceil.append(top["ceiling"].mean())
            k = min(TOP_N_PER_DAY, len(day))
            r = day.iloc[rng.choice(len(day), size=k, replace=False)]
            rnd_clean.append(r["clean"].mean()); rnd_ceil.append(r["ceiling"].mean())
        rows.append({
            "year": yr, "n_selected": n_sel,
            "sel_clean_rate": round(np.nanmean(sel_clean), 4),
            "rnd_clean_rate": round(np.nanmean(rnd_clean), 4),
            "sel_mean_ceiling": round(np.nanmean(sel_ceil), 4),
            "rnd_mean_ceiling": round(np.nanmean(rnd_ceil), 4),
            "ceiling_lift": round(np.nanmean(sel_ceil) - np.nanmean(rnd_ceil), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== BY SIDE (eval pooled, top-N/day) =====")
    rows = []
    for s in ("long", "short"):
        sub = ev[ev["side"].eq(s)]
        sel_clean, sel_ceil = [], []
        for _, day in sub.groupby("date"):
            top = day.nlargest(max(1, TOP_N_PER_DAY // 2), "score")
            sel_clean.append(top["clean"].mean()); sel_ceil.append(top["ceiling"].mean())
        rows.append({"side": s, "sel_clean_rate": round(np.nanmean(sel_clean), 4),
                     "base_clean_rate": round(sub["clean"].mean(), 4),
                     "sel_mean_ceiling": round(np.nanmean(sel_ceil), 4),
                     "base_mean_ceiling": round(sub["ceiling"].mean(), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nGO if sel_clean_rate >> rnd/base and sel_mean_ceiling >> rnd/base across all eval years.")


if __name__ == "__main__":
    main()
