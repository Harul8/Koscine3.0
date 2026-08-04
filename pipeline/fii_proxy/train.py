"""
Train regime-specific LightGBM proxy models that predict FII stock-level
accumulation from observable price/volume/OI/flow features.

Three models are trained — one each for bull, bear, and range regimes —
so each model learns the patterns that specifically co-occur with FII
accumulation under that market condition.  A fallback 'all' model trained
on the full dataset is also saved and used when the regime is unknown.

Target (5-day rolling consensus):
    For each (symbol, date): target = 1 if rolling 5-day net FII value > 0.
    Computed from silver/fii_stock_trades.parquet (ground truth 2012-2025 Mar).

Train / val split:
    Train:  up to 2023-12-31
    Val:    2024-01-01 to 2025-03-31

Output: models/fii_proxy_lgbm.pkl

Run:
    python -m pipeline.fii_proxy.train
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

SILVER_FII    = Path(r"C:\Users\rahul\Koscine 3.0\data\silver\fii_stock_trades.parquet")
GOLD_FEATURES = Path(r"C:\Users\rahul\Koscine 3.0\gold\features.parquet")
MODEL_OUT     = Path(r"C:\Users\rahul\Koscine 3.0\models\fii_proxy_lgbm.pkl")

FII_DATA_END  = "2025-03-31"   # last date with ground-truth FII trades
TRAIN_END     = "2023-12-31"
VAL_END       = "2024-12-31"   # 2025-01-01 to FII_DATA_END held out for testing

REGIMES = ["bull", "bear", "range"]

_EXCLUDE_COLS = {
    "date", "symbol", "split", "regime", "market_phase",
    "stock_phase", "combined_phase",
    "upside_t5", "downside_t5", "is_conflict",
    "up", "dn", "wild", "up_3", "dn_3",
}

_LGBM_PARAMS = dict(
    objective        = "binary",
    metric           = ["binary_logloss", "auc"],
    n_estimators     = 1500,
    learning_rate    = 0.02,
    num_leaves       = 31,
    min_child_samples= 50,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    reg_lambda       = 2.0,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)


def _build_target(trades: pd.DataFrame) -> pd.DataFrame:
    """5-day backward-looking rolling net FII value per (symbol, date)."""
    trades = trades.sort_values(["symbol", "date"])
    trades["roll5_net"] = (
        trades.groupby("symbol")["net_value"]
        .transform(lambda s: s.rolling(5, min_periods=2).sum())
    )
    trades["target"] = (trades["roll5_net"] > 0).astype(np.int8)
    return trades[["date", "symbol", "target"]].dropna()


def _train_one(X_tr, y_tr, X_va, y_va, label: str) -> tuple:
    """Train one LGBMClassifier + isotonic calibrator. Returns (model, cal, auc)."""
    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    model = LGBMClassifier(scale_pos_weight=pos_weight, **_LGBM_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])

    val_prob = model.predict_proba(X_va)[:, 1]
    raw_auc  = roc_auc_score(y_va, val_prob)

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(val_prob, y_va)
    cal_prob = cal.predict(val_prob)
    cal_auc  = roc_auc_score(y_va, cal_prob)

    print(f"    [{label}]  train={len(y_tr):,}  val={len(y_va):,}  "
          f"bal={y_tr.mean():.1%}  AUC={raw_auc:.4f}  AUC(cal)={cal_auc:.4f}")
    return model, cal, round(cal_auc, 4)


def train() -> None:
    print("[fii_proxy.train] Loading data ...")

    trades = pd.read_parquet(SILVER_FII)
    trades["date"] = pd.to_datetime(trades["date"])

    feats_df = pd.read_parquet(GOLD_FEATURES)
    feats_df["date"] = pd.to_datetime(feats_df["date"])

    target_df = _build_target(trades)

    merged = feats_df.merge(target_df, on=["date", "symbol"], how="inner")
    merged = merged[merged["date"] <= FII_DATA_END].copy()

    print(f"  Merged rows: {len(merged):,}  |  date range: "
          f"{merged['date'].min().date()} - {merged['date'].max().date()}")
    print(f"  Target balance: {merged['target'].mean():.1%} positive")
    print(f"  Regime distribution:\n{merged['regime'].value_counts().to_string()}")

    feature_cols = [c for c in feats_df.columns if c not in _EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols
                    if c in merged.columns
                    and pd.api.types.is_numeric_dtype(merged[c])
                    and merged[c].notna().any()]
    print(f"  Features: {len(feature_cols)}")

    trained_symbols = sorted(merged["symbol"].unique().tolist())

    train_mask = merged["date"] <= TRAIN_END
    val_mask   = (merged["date"] > TRAIN_END) & (merged["date"] <= VAL_END)

    # ── Fallback model: trained on all regimes ────────────────────────────────
    print("\n  Training fallback (all regimes) ...")
    m_all, cal_all, auc_all = _train_one(
        merged.loc[train_mask, feature_cols], merged.loc[train_mask, "target"],
        merged.loc[val_mask,   feature_cols], merged.loc[val_mask,   "target"],
        "all",
    )

    # ── Regime-specific models ────────────────────────────────────────────────
    print("\n  Training regime-specific models ...")
    regime_models = {}
    regime_aucs   = {"all": auc_all}

    for regime in REGIMES:
        r_mask_tr = train_mask & (merged["regime"] == regime)
        r_mask_va = val_mask   & (merged["regime"] == regime)

        if r_mask_tr.sum() < 500 or r_mask_va.sum() < 100:
            print(f"    [{regime}]  insufficient data — using fallback")
            regime_models[regime] = None
            continue

        m, cal, auc = _train_one(
            merged.loc[r_mask_tr, feature_cols], merged.loc[r_mask_tr, "target"],
            merged.loc[r_mask_va, feature_cols], merged.loc[r_mask_va, "target"],
            regime,
        )
        regime_models[regime] = {"model": m, "calibrator": cal}
        regime_aucs[regime] = auc

    print(f"\n  AUC summary: {regime_aucs}")

    artifact = {
        # Fallback
        "model":           m_all,
        "calibrator":      cal_all,
        # Regime-specific
        "regime_models":   regime_models,   # {regime: {"model":..,"calibrator":..} | None}
        # Shared
        "features":        feature_cols,
        "trained_symbols": trained_symbols,
        "train_end":       TRAIN_END,
        "fii_data_end":    FII_DATA_END,
        "val_auc":         regime_aucs,
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved -> {MODEL_OUT}")


if __name__ == "__main__":
    train()
