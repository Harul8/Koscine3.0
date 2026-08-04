"""
Evaluate the FII proxy model on the held-out 2025 test set
(2025-01-01 to 2025-03-31 -- dates the model never saw during training or val).

Metrics reported per regime and overall:
  - AUC
  - Accuracy at threshold 0.5
  - Top-decile precision  (of stocks ranked in top 10% by prob, what fraction
                           actually had net FII accumulation that day?)
  - Brier score (lower is better)

Run:
    python -m pipeline.fii_proxy.evaluate
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

SILVER_FII    = Path(r"C:\Users\rahul\Koscine 3.0\data\silver\fii_stock_trades.parquet")
GOLD_FEATURES = Path(r"C:\Users\rahul\Koscine 3.0\gold\features.parquet")
MODEL_PATH    = Path(r"C:\Users\rahul\Koscine 3.0\models\fii_proxy_lgbm.pkl")

TEST_START = "2025-01-01"
TEST_END   = "2025-03-31"


def _build_target(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.sort_values(["symbol", "date"])
    trades["roll5_net"] = (
        trades.groupby("symbol")["net_value"]
        .transform(lambda s: s.rolling(5, min_periods=2).sum())
    )
    trades["target"] = (trades["roll5_net"] > 0).astype(int)
    return trades[["date", "symbol", "target"]].dropna()


def _score_subset(model, calibrator, features, X) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw)


def _metrics(y_true, y_prob, label: str) -> dict:
    if len(y_true) < 20 or y_true.nunique() < 2:
        print(f"  [{label}]  insufficient data (n={len(y_true)}) -- skipped")
        return {}

    auc    = roc_auc_score(y_true, y_prob)
    acc    = ((y_prob >= 0.5) == y_true).mean()
    brier  = brier_score_loss(y_true, y_prob)

    # Top-decile precision
    cutoff = np.percentile(y_prob, 90)
    top_mask = y_prob >= cutoff
    td_prec  = y_true[top_mask].mean() if top_mask.sum() > 0 else float("nan")

    print(f"  [{label:<10}]  n={len(y_true):>6,}  AUC={auc:.4f}  "
          f"Acc={acc:.3f}  Top10%Prec={td_prec:.3f}  Brier={brier:.4f}")
    return {"label": label, "n": len(y_true), "auc": auc,
            "acc": acc, "top10_prec": td_prec, "brier": brier}


def evaluate() -> pd.DataFrame:
    print("[fii_proxy.evaluate] Loading model and data ...")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    features       = artifact["features"]
    regime_models  = artifact.get("regime_models", {})

    # Features for test period
    feats = pd.read_parquet(GOLD_FEATURES)
    feats["date"] = pd.to_datetime(feats["date"])
    test_feats = feats[(feats["date"] >= TEST_START) & (feats["date"] <= TEST_END)].copy()

    # Ground-truth FII targets for test period
    trades = pd.read_parquet(SILVER_FII)
    trades["date"] = pd.to_datetime(trades["date"])
    target_df = _build_target(trades)
    test_targets = target_df[
        (target_df["date"] >= TEST_START) & (target_df["date"] <= TEST_END)
    ]

    # Merge
    df = test_feats.merge(test_targets, on=["date", "symbol"], how="inner")
    print(f"  Test rows: {len(df):,}  |  dates: {df['date'].nunique()}  "
          f"|  symbols: {df['symbol'].nunique()}")
    print(f"  Target balance: {df['target'].mean():.1%} positive")
    print(f"  Regime breakdown:\n{df['regime'].value_counts().to_string()}")

    X_all = df.reindex(columns=features)

    results = []

    # ── Overall with fallback model ───────────────────────────────────────────
    print(f"\n  === Overall (fallback model) ===")
    prob_all = _score_subset(artifact["model"], artifact["calibrator"], features, X_all)
    results.append(_metrics(df["target"], pd.Series(prob_all, index=df.index), "all(fallback)"))

    # ── Regime routing: each row scored by its own regime model ──────────────
    print(f"\n  === Regime-routed scoring ===")
    prob_routed = np.full(len(df), np.nan)
    for regime in df["regime"].unique():
        mask = (df["regime"] == regime).values
        rm = regime_models.get(regime)
        if rm is not None:
            p = _score_subset(rm["model"], rm["calibrator"], features,
                              X_all[mask])
        else:
            p = _score_subset(artifact["model"], artifact["calibrator"], features,
                              X_all[mask])
        prob_routed[mask] = p

    results.append(_metrics(df["target"],
                            pd.Series(prob_routed, index=df.index),
                            "all(routed)"))

    # ── Per-regime breakdown ──────────────────────────────────────────────────
    print(f"\n  === Per-regime breakdown (routed) ===")
    for regime in sorted(df["regime"].unique()):
        mask = df["regime"] == regime
        results.append(_metrics(df.loc[mask, "target"],
                                pd.Series(prob_routed[mask.values], index=df.index[mask.values]),
                                regime))

    # ── Per-month breakdown ───────────────────────────────────────────────────
    print(f"\n  === Per-month breakdown (routed) ===")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    for month in sorted(df["month"].unique()):
        mask = df["month"] == month
        results.append(_metrics(df.loc[mask, "target"],
                                pd.Series(prob_routed[mask.values], index=df.index[mask.values]),
                                month))

    return pd.DataFrame([r for r in results if r])


if __name__ == "__main__":
    evaluate()
