"""Walk-forward evaluation for the NIFTY intraday breakout labels.

Runs v1 (price-only, full 2022-2026 history) and, on the Oct2024+ window only, a
MATCHED-WINDOW comparison: v1's price-only features vs v2 (price + option-chain
features) on the exact same rows -- so any lift is attributable to the option-chain
features, not to having more training data.

Expanding walk-forward, monthly refit, imitates experiments/megacap_direction_v1/models.py::wf_index.
K (the breakout threshold) is calibrated fresh inside each fold from that fold's training
rows only (see labels.py). The embargo below is a defense-in-depth margin, not a strict
requirement -- every feature/label window in this dataset is already scoped to its own
trading day (see features.py/labels.py), so there is no cross-day leakage by construction;
the embargo just guards against any edge-case timestamp misalignment.

Usage:
    python experiments/nifty_intraday_breakout_v1/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Smaller / more heavily regularized than pipeline/config.py's daily-bar LGBM_CLEAN_PARAMS --
# 5-min bars are far noisier per-row, and a single monthly fold here has many more rows than
# a daily-bar fold does, so early generalization matters more than tree depth/count.
LGBM_PARAMS = dict(
    objective="binary", metric="average_precision", n_estimators=600, learning_rate=0.03,
    num_leaves=31, min_child_samples=200, reg_lambda=2.0, reg_alpha=0.5,
    feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
    n_jobs=-1, verbose=-1, random_state=42,
)
EMBARGO_BARS = {6: 30, 12: 60, 24: 120}   # generous multiple of the label horizon itself


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("feature_"))


def walk_forward(df: pd.DataFrame, horizon: int, feats: list[str], min_train_rows: int = 5000) -> pd.DataFrame:
    import lightgbm as lgb

    need = [f"fwd_move_mag_{horizon}", labels.DEFAULT_VOL_COL] + feats
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
        k = labels.calibrate_k(train, horizon)
        y_tr = labels.apply_threshold(train, horizon, k)
        y_ev = labels.apply_threshold(ev, horizon, k)
        valid_tr = y_tr.notna()
        valid_ev = y_ev.notna()
        if valid_tr.sum() < min_train_rows or y_tr[valid_tr].nunique() < 2 or valid_ev.sum() < 20:
            continue
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(train.loc[valid_tr, feats], y_tr[valid_tr].astype(int))
        p = model.predict_proba(ev.loc[valid_ev, feats])[:, 1]
        rows.append(pd.DataFrame({
            # .reset_index(drop=True), NOT .values -- .values on a tz-aware datetime Series
            # silently strips the Asia/Kolkata offset (converts to naive UTC), which then
            # fails to match against tz-aware timestamps anywhere downstream (same bug
            # caught and fixed in straddle_path.py).
            "timestamp": ev.loc[valid_ev, "timestamp"].reset_index(drop=True),
            "y": y_ev[valid_ev].astype(int).reset_index(drop=True),
            "p": p,
            "month": str(mo),
            "train_base_rate": float(y_tr[valid_tr].mean()),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(preds: pd.DataFrame, label: str) -> dict:
    if preds.empty or preds["y"].nunique() < 2:
        return {"label": label, "n": int(len(preds)), "note": "insufficient data / single-class"}
    auc = roc_auc_score(preds["y"], preds["p"])
    ap = average_precision_score(preds["y"], preds["p"])
    base_rate = preds["y"].mean()
    top10 = preds.nlargest(max(1, len(preds) // 10), "p")
    prec_at_10 = top10["y"].mean()
    return {
        "label": label, "n": int(len(preds)), "base_rate": round(float(base_rate), 4),
        "auc": round(float(auc), 4), "avg_precision": round(float(ap), 4),
        "precision_at_top10pct": round(float(prec_at_10), 4),
        "lift_over_base_rate": round(float(prec_at_10 / base_rate), 2) if base_rate > 0 else None,
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
            preds.to_csv(RESULTS_DIR / f"v1_price_only_h{h}_preds.csv", index=False)
        print(results[-1])

    v2_path = DATA_DIR / "v2_option_chain.parquet"
    if v2_path.exists():
        v2 = pd.read_parquet(v2_path)
        v2["timestamp"] = pd.to_datetime(v2["timestamp"])
        option_feats = feature_columns(v2)               # superset: price_feats + option feature_* cols
        matched_price_feats = [c for c in option_feats if c in price_feats]
        for h in labels.HORIZONS:
            preds_price = walk_forward(v2, h, matched_price_feats)
            results.append(summarize(preds_price, f"v1_price_only_matched_window_h{h}"))
            print(results[-1])
            preds_opt = walk_forward(v2, h, option_feats)
            results.append(summarize(preds_opt, f"v2_with_option_chain_h{h}"))
            print(results[-1])
            if not preds_opt.empty:
                preds_opt.to_csv(RESULTS_DIR / f"v2_option_chain_h{h}_preds.csv", index=False)
    else:
        print(f"[train] {v2_path} not found -- skipping v2 comparison")

    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    _write_findings(summary)


def _write_findings(summary: pd.DataFrame) -> None:
    lines = [
        "# NIFTY intraday breakout v1 -- findings",
        "",
        "Walk-forward (expanding, monthly refit), K calibrated per fold from that fold's",
        "training rows only. `base_rate` is what a model with zero skill would get at the",
        "top-10% cut; `lift_over_base_rate` is the number that actually matters -- close to",
        "1.0x means no real edge, regardless of how good AUC/avg_precision look on their own.",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (RESULTS_DIR / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
