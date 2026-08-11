"""Walk-forward evaluation for a NIFTY intraday MAGNITUDE regressor -- predicts the actual
forward move size (fwd_move_mag_H, already computed continuous by labels.py::add_forward_extremes),
not a binarized big-move-or-not label like train.py's classifier. Built for the live NIFTY
poller's "only fire if >=0.25% move is expected" gate (per the approved plan), which needs a
genuine magnitude forecast -- the classifier's P(big move) isn't the same thing and was never
meant to answer "how big."

Same walk-forward harness as train.py (expanding, monthly refit, K-calibration N/A here since
there's no threshold to calibrate -- the target is used raw), same no-look-ahead embargo
discipline. h=6 (30-min horizon, the finest granularity labels.py computes) is the one intended
for live use -- closest to a "next few 5-min bars" reading for gating individual signals; other
horizons are still evaluated for comparison, matching train.py's structure.

Usage:
    python experiments/nifty_intraday_breakout_v1/train_magnitude.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Same regularization posture as train.py's classifier -- noisy 5-min-bar rows, want early
# generalization over tree depth/count. objective/metric swapped for a continuous target.
LGBM_PARAMS = dict(
    objective="regression", metric="mae", n_estimators=600, learning_rate=0.03,
    num_leaves=31, min_child_samples=200, reg_lambda=2.0, reg_alpha=0.5,
    feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
    n_jobs=-1, verbose=-1, random_state=42,
)
EMBARGO_BARS = {6: 30, 12: 60, 24: 120}   # same generous multiple of the label horizon as train.py


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("feature_"))


def walk_forward(df: pd.DataFrame, horizon: int, feats: list[str], min_train_rows: int = 5000) -> pd.DataFrame:
    import lightgbm as lgb

    target_col = f"fwd_move_mag_{horizon}"
    need = [target_col] + feats
    d = df.dropna(subset=need, how="any").copy()
    d["month"] = pd.PeriodIndex(d["timestamp"], freq="M")
    months = sorted(d["month"].unique())
    embargo = pd.Timedelta(minutes=5 * EMBARGO_BARS[horizon])

    rows = []
    for mo in months[1:]:   # skip the first month -- no prior training data yet
        eval_mask = d["month"] == mo
        if not eval_mask.any():
            continue
        eval_start = d.loc[eval_mask, "timestamp"].min()
        cut = eval_start - embargo
        train = d[d["timestamp"] < cut]
        ev = d[eval_mask]
        if len(train) < min_train_rows or ev.empty:
            continue
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(train[feats], train[target_col])
        pred = model.predict(ev[feats])
        rows.append(pd.DataFrame({
            # .reset_index(drop=True), NOT .values -- see train.py's identical comment on why
            "timestamp": ev["timestamp"].reset_index(drop=True),
            "y": ev[target_col].reset_index(drop=True),
            "pred": pred,
            "month": str(mo),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(preds: pd.DataFrame, label: str) -> dict:
    if preds.empty:
        return {"label": label, "n": 0, "note": "no data"}
    err = preds["pred"] - preds["y"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    ic, _ = spearmanr(preds["pred"], preds["y"])
    # decile check: does ranking by predicted magnitude actually separate real outcomes?
    preds = preds.copy()
    preds["decile"] = pd.qcut(preds["pred"], 10, labels=False, duplicates="drop")
    decile_means = preds.groupby("decile")["y"].mean()
    top_bottom_ratio = (decile_means.iloc[-1] / decile_means.iloc[0]) if decile_means.iloc[0] > 0 else None
    return {
        "label": label, "n": int(len(preds)), "mean_actual_mag": round(float(preds["y"].mean()), 5),
        "mae": round(float(mae), 5), "rmse": round(float(rmse), 5),
        "spearman_ic": round(float(ic), 4) if ic is not None else None,
        "top_decile_vs_bottom_decile": round(float(top_bottom_ratio), 2) if top_bottom_ratio else None,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    v1_path = DATA_DIR / "v1_price_only.parquet"
    if not v1_path.exists():
        raise SystemExit(f"{v1_path} not found -- run dataset.py first")
    v1 = pd.read_parquet(v1_path)
    v1["timestamp"] = pd.to_datetime(v1["timestamp"])
    price_feats = feature_columns(v1)
    for h in labels.HORIZONS:
        preds = walk_forward(v1, h, price_feats)
        results.append(summarize(preds, f"v1_price_only_full_history_h{h}"))
        if not preds.empty:
            preds.to_csv(RESULTS_DIR / f"v1_price_only_magnitude_h{h}_preds.csv", index=False)
        print(results[-1])

    v2_path = DATA_DIR / "v2_option_chain.parquet"
    if v2_path.exists():
        v2 = pd.read_parquet(v2_path)
        v2["timestamp"] = pd.to_datetime(v2["timestamp"])
        option_feats = feature_columns(v2)
        matched_price_feats = [c for c in option_feats if c in price_feats]
        for h in labels.HORIZONS:
            preds_price = walk_forward(v2, h, matched_price_feats)
            results.append(summarize(preds_price, f"v1_price_only_matched_window_magnitude_h{h}"))
            print(results[-1])
            preds_opt = walk_forward(v2, h, option_feats)
            results.append(summarize(preds_opt, f"v2_with_option_chain_magnitude_h{h}"))
            print(results[-1])
            if not preds_opt.empty:
                preds_opt.to_csv(RESULTS_DIR / f"v2_option_chain_magnitude_h{h}_preds.csv", index=False)
    else:
        print(f"[train_magnitude] {v2_path} not found -- skipping v2 comparison")

    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_DIR / "summary_magnitude.csv", index=False)
    print("\n" + summary.to_string(index=False))
    _write_findings(summary)


def _write_findings(summary: pd.DataFrame) -> None:
    lines = [
        "# NIFTY intraday magnitude regressor -- findings",
        "",
        "Walk-forward (expanding, monthly refit). `spearman_ic` is the number that matters --",
        "rank correlation between predicted and realized move magnitude; close to 0 means no",
        "real skill regardless of how low MAE looks (MAE alone can look fine on a model that",
        "just predicts the average every time). `top_decile_vs_bottom_decile` is a plainer",
        "sanity check: does the top 10% by predicted magnitude actually see bigger realized",
        "moves than the bottom 10%.",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (RESULTS_DIR / "FINDINGS_magnitude.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
