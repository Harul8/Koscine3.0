"""
Identify price/volume/OI/flow patterns associated with FII accumulation vs distribution.

For each regime model:
  1. Top features by LightGBM gain importance
  2. Conditional means: average feature value when FII accumulating (target=1) vs distributing (target=0)
  3. Direction: does high value of this feature -> more accumulation or more distribution?

Run:
    python -m pipeline.fii_proxy.patterns
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SILVER_FII    = Path(r"C:\Users\rahul\Koscine 3.0\data\silver\fii_stock_trades.parquet")
GOLD_FEATURES = Path(r"C:\Users\rahul\Koscine 3.0\gold\features.parquet")
MODEL_PATH    = Path(r"C:\Users\rahul\Koscine 3.0\models\fii_proxy_lgbm.pkl")

# Human-readable feature group labels
_GROUPS = {
    "ret_":         "Price/Return",
    "above_":       "Price/Return",
    "dist_52":      "Price/Return",
    "rsi":          "Price/Return",
    "macd":         "Price/Return",
    "close_pos":    "Price/Return",
    "atr":          "Price/Return",
    "hv_":          "Price/Return",
    "vol_exp":      "Volume",
    "vol_ratio":    "Volume",
    "amihud":       "Volume",
    "delivery":     "Volume",
    "turnover":     "Volume",
    "oi_":          "OI/Derivatives",
    "pcr":          "OI/Derivatives",
    "iv_":          "OI/Derivatives",
    "skew":         "OI/Derivatives",
    "call_":        "OI/Derivatives",
    "put_":         "OI/Derivatives",
    "fut_":         "OI/Derivatives",
    "basis":        "OI/Derivatives",
    "rollover":     "OI/Derivatives",
    "flow_":        "Flow",
    "sector_":      "Flow",
    "mkt_":         "Market Context",
    "nifty":        "Market Context",
    "vix":          "Market Context",
    "breadth":      "Market Context",
    "adv_dec":      "Market Context",
    "index_":       "Market Context",
}


def _group(feat: str) -> str:
    for prefix, grp in _GROUPS.items():
        if feat.startswith(prefix):
            return grp
    return "Other"


def _analyse_model(model, features: list[str], data: pd.DataFrame,
                   label: str, top_n: int = 20) -> pd.DataFrame:
    # Feature importances (gain = contribution to reducing loss)
    imp = dict(zip(features, model.booster_.feature_importance(importance_type="gain")))
    imp_series = pd.Series(imp).sort_values(ascending=False)

    top_feats = imp_series.head(top_n).index.tolist()

    # Conditional means: accum (target=1) vs distrib (target=0)
    sub = data[["target"] + [f for f in top_feats if f in data.columns]].dropna()
    cond = sub.groupby("target")[top_feats].mean()

    rows = []
    for feat in top_feats:
        if feat not in cond.columns:
            continue
        val_accum = cond.loc[1, feat] if 1 in cond.index else np.nan
        val_distr = cond.loc[0, feat] if 0 in cond.index else np.nan
        direction = "HIGH -> accum" if val_accum > val_distr else "LOW -> accum"
        delta_pct = (val_accum - val_distr) / (abs(val_distr) + 1e-9) * 100
        rows.append({
            "feature":    feat,
            "group":      _group(feat),
            "importance": round(imp_series[feat]),
            "accum_mean": round(val_accum, 4),
            "distr_mean": round(val_distr, 4),
            "delta_pct":  round(delta_pct, 1),
            "direction":  direction,
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  MODEL: {label}  (top {top_n} features by gain importance)")
    print(f"{'='*70}")
    print(f"{'Feature':<30} {'Group':<18} {'Direction':<15} {'Delta%':>7}  {'Accum':>8}  {'Distr':>8}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"  {r['feature']:<28} {r['group']:<18} {r['direction']:<15} "
              f"{r['delta_pct']:>6.1f}%  {r['accum_mean']:>8.4f}  {r['distr_mean']:>8.4f}")

    # Group summary
    grp_imp = df.groupby("group")["importance"].sum().sort_values(ascending=False)
    print(f"\n  Group importance share:")
    total = grp_imp.sum()
    for g, v in grp_imp.items():
        bar = "#" * int(v / total * 40)
        print(f"    {g:<20} {v/total:>5.1%}  {bar}")

    return df


def analyse() -> None:
    print("[fii_proxy.patterns] Loading data ...")

    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)

    features      = art["features"]
    regime_models = art.get("regime_models", {})

    # Build labelled dataset (train + val period only for ground truth)
    trades = pd.read_parquet(SILVER_FII)
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values(["symbol", "date"])
    trades["roll5_net"] = (
        trades.groupby("symbol")["net_value"]
        .transform(lambda s: s.rolling(5, min_periods=2).sum())
    )
    trades["target"] = (trades["roll5_net"] > 0).astype(int)
    target_df = trades[["date", "symbol", "target"]].dropna()

    feats = pd.read_parquet(GOLD_FEATURES)
    feats["date"] = pd.to_datetime(feats["date"])
    merged = feats.merge(target_df, on=["date", "symbol"], how="inner")
    merged = merged[merged["date"] <= art["fii_data_end"]].copy()

    print(f"  Ground-truth rows: {len(merged):,}  |  "
          f"Accumulation: {merged['target'].mean():.1%}")

    results = {}

    # Fallback / all-regime model
    results["all"] = _analyse_model(
        art["model"], features, merged[features + ["target"]], "ALL REGIMES"
    )

    # Regime-specific
    for regime, rm in regime_models.items():
        if rm is None:
            print(f"\n  [{regime}] using fallback — skipping separate analysis")
            continue
        sub = merged[merged["regime"] == regime]
        results[regime] = _analyse_model(
            rm["model"], features,
            sub[features + ["target", "regime"]].drop(columns=["regime"]),
            f"{regime.upper()} REGIME  (n={len(sub):,})"
        )

    # Cross-regime comparison: which features change direction across regimes?
    print(f"\n{'='*70}")
    print("  CROSS-REGIME: Features that flip direction")
    print(f"{'='*70}")
    if len(results) > 1:
        ref = results["all"].set_index("feature")["direction"]
        for regime, df_r in results.items():
            if regime == "all":
                continue
            for _, row in df_r.iterrows():
                feat = row["feature"]
                if feat in ref.index and ref[feat] != row["direction"]:
                    print(f"  {feat:<30} all={ref[feat]}  {regime}={row['direction']}")


if __name__ == "__main__":
    analyse()
