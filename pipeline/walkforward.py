"""
Walk-forward evaluation — train on expanding window, test on next N months,
slide forward. Reports per-window metrics + aggregate mean ± std.

Why this matters for production:
  - A single train/val split rewards models that overfit one period.
  - 6 rolling windows reveal which models work across regime changes.
  - mean ± std is what should drive promotion decisions, not peak.

Usage:
    # Walk-forward over last 18 months in 3-month windows for one target
    python -m pipeline.walkforward --model tabm --targets dn_5_xl \\
        --n_windows 6 --window_months 3

    # All targets, both model types
    python -m pipeline.walkforward --model lgbm tabm --n_windows 6

Outputs:
    models/walkforward/{model}_{target}_wf.csv  — per-window metrics
    models/walkforward/{model}_{target}_wf.json — aggregate stats
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from dateutil.relativedelta import relativedelta

from .config import (
    GOLD_FEATURES, GOLD_LABELS, MODEL_DIR,
    MODEL_TARGETS, TARGET_FEATURE_COLS, MODEL_FEATURE_OVERRIDES,
    TABM_TARGET_PARAMS, LGBM_TARGET_PARAMS,
)
from .models import get_model_class


WF_DIR = MODEL_DIR / "walkforward"


# ── Window definition ──────────────────────────────────────────────────────────

def _build_windows(
    panel_dates: pd.Series,
    n_windows: int,
    window_months: int,
    min_train_years: float = 3.0,
) -> list[dict]:
    """
    Build n_windows rolling (train, val, test) windows ending at the most recent date.
    Each window has:
      - train: from start of panel up to train_end
      - val:   window_months/2 immediately after train_end (for early stopping)
      - test:  window_months immediately after val_end (the OOS measurement)
    """
    max_date = panel_dates.max()
    min_date = panel_dates.min()
    half = max(1, window_months // 2)

    windows = []
    for i in range(n_windows):
        # window i ends at max_date - i × window_months
        test_end = max_date - relativedelta(months=window_months * i)
        test_start = test_end - relativedelta(months=window_months) + pd.Timedelta(days=1)
        val_end = test_start - pd.Timedelta(days=1)
        val_start = val_end - relativedelta(months=half) + pd.Timedelta(days=1)
        train_end = val_start - pd.Timedelta(days=1)

        # Skip if not enough training history
        train_months = (train_end - min_date).days / 30.44
        if train_months < min_train_years * 12:
            print(f"[wf] skipping window {i}: only {train_months:.1f} train months")
            continue

        windows.append({
            "i":          i,
            "train_end":  train_end,
            "val_start":  val_start,  "val_end":  val_end,
            "test_start": test_start, "test_end": test_end,
        })
    windows.reverse()   # oldest first
    return windows


# ── Per-window training + eval ─────────────────────────────────────────────────

def _eval_window(
    model_type: str,
    target: str,
    panel: pd.DataFrame,
    feat_cols: list[str],
    win: dict,
    t_params: dict,
    wf_seeds: int = 1,
) -> dict:
    """Train on win['train_end'], eval on win['test_start'..test_end'].

    wf_seeds: # of seeds per window. 1 is enough to detect regime stability
              (seed noise ~1.5% prec@10 is much smaller than window-to-window
              regime variance). The 5-seed ensemble is for deployment-time
              variance reduction, not walk-forward measurement.
    """
    dates = panel["date"]

    tr = panel[dates <= win["train_end"]]
    va = panel[(dates >= win["val_start"]) & (dates <= win["val_end"])]
    te = panel[(dates >= win["test_start"]) & (dates <= win["test_end"])]

    yt = tr[target].astype(float).values
    yv = va[target].astype(float).values
    yz = te[target].astype(float).values
    tr_ok, va_ok, te_ok = ~np.isnan(yt), ~np.isnan(yv), ~np.isnan(yz)

    X_train = tr[["date", "symbol"] + feat_cols]
    X_val   = va[["date", "symbol"] + feat_cols]
    X_test  = te[["date", "symbol"] + feat_cols]

    cls = get_model_class(model_type)
    # Per-window seed count: CLI wf_seeds overrides TABM_TARGET_PARAMS.n_seeds.
    # Default to 1 for cheap regime-stability check.
    t_params.pop("n_seeds", None)
    n_seeds = wf_seeds if model_type == "tabm" else 1

    if model_type == "tabm" and n_seeds > 1:
        probs = []
        for s in range(n_seeds):
            m = cls(target=target, feature_cols=feat_cols, seed=s, **t_params)
            m.fit(X_train[tr_ok], yt[tr_ok], X_val[va_ok], yv[va_ok])
            probs.append(m.predict_proba(X_test[te_ok]))
        test_probs = np.stack(probs).mean(axis=0)
    else:
        m = cls(target=target, feature_cols=feat_cols, **t_params)
        m.fit(X_train[tr_ok], yt[tr_ok], X_val[va_ok], yv[va_ok])
        test_probs = m.predict_proba(X_test[te_ok])

    yt_test = yz[te_ok]
    p10_n = max(1, int(len(test_probs) * 0.10))
    top   = np.argsort(test_probs)[::-1][:p10_n]
    base  = float(yt_test.mean())
    p10   = float(yt_test[top].mean())

    # Save test predictions for later blend analysis
    test_df = te[te_ok].reset_index(drop=True)[["date", "symbol"]].copy()
    test_df["y_true"] = yt_test
    test_df["proba"]  = test_probs
    pred_dir = WF_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(pred_dir / f"{model_type}_{target}_win{win['i']}.parquet")

    return {
        "window_i":      win["i"],
        "train_end":     win["train_end"].strftime("%Y-%m-%d"),
        "test_start":    win["test_start"].strftime("%Y-%m-%d"),
        "test_end":      win["test_end"].strftime("%Y-%m-%d"),
        "n_test":        int(te_ok.sum()),
        "base_rate":     base,
        "test_auc":      float(roc_auc_score(yt_test, test_probs)),
        "test_ap":       float(average_precision_score(yt_test, test_probs)),
        "test_prec10":   p10,
        "test_lift":     p10 / base if base > 0 else float("nan"),
    }


# ── Driver ─────────────────────────────────────────────────────────────────────

def run_walkforward(
    model_type: str,
    target: str,
    n_windows: int = 6,
    window_months: int = 3,
    wf_seeds: int = 1,
) -> None:
    print(f"\n{'='*70}\n[wf] {model_type} / {target}\n{'='*70}")

    feat_df = pd.read_parquet(GOLD_FEATURES)
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    labs_df = pd.read_parquet(GOLD_LABELS)
    labs_df["date"] = pd.to_datetime(labs_df["date"])

    panel = feat_df.merge(labs_df[["date", "symbol", target]],
                          on=["date", "symbol"], how="inner")

    feat_cols = (
        MODEL_FEATURE_OVERRIDES.get(model_type, {}).get(target)
        or TARGET_FEATURE_COLS.get(target)
    )
    feat_cols = [c for c in feat_cols if c in panel.columns]
    print(f"[wf] features: {len(feat_cols)}")

    windows = _build_windows(panel["date"], n_windows, window_months)
    print(f"[wf] windows: {len(windows)}")

    t_params_base = (
        TABM_TARGET_PARAMS.get(target, {}) if model_type == "tabm"
        else LGBM_TARGET_PARAMS.get(target, {})
    )

    rows = []
    for win in windows:
        t0 = time.time()
        try:
            r = _eval_window(model_type, target, panel, feat_cols, win,
                             dict(t_params_base), wf_seeds=wf_seeds)
            r["elapsed_sec"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  win{r['window_i']}  test {r['test_start']}→{r['test_end']}  "
                  f"AP={r['test_ap']:.3f}  P@10={r['test_prec10']:.2%}  "
                  f"lift={r['test_lift']:.2f}x  ({r['elapsed_sec']:.0f}s)")
        except Exception as e:
            print(f"  win{win['i']} FAILED: {e}")

    if not rows:
        print("[wf] no windows succeeded — aborting")
        return

    df = pd.DataFrame(rows)
    agg = {
        "n_windows":       len(df),
        "ap_mean":         float(df["test_ap"].mean()),
        "ap_std":          float(df["test_ap"].std()),
        "ap_min":          float(df["test_ap"].min()),
        "prec10_mean":     float(df["test_prec10"].mean()),
        "prec10_std":      float(df["test_prec10"].std()),
        "prec10_min":      float(df["test_prec10"].min()),
        "lift_mean":       float(df["test_lift"].mean()),
        "lift_min":        float(df["test_lift"].min()),
        "windows_above_2x": int((df["test_lift"] >= 2.0).sum()),
    }

    WF_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = WF_DIR / f"{model_type}_{target}_wf.csv"
    json_path = WF_DIR / f"{model_type}_{target}_wf.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(agg, indent=2))

    print(f"\n[wf] {model_type}/{target} SUMMARY (n={agg['n_windows']} windows)")
    print(f"  AP      : {agg['ap_mean']:.3f} ± {agg['ap_std']:.3f}   "
          f"(min {agg['ap_min']:.3f})")
    print(f"  prec@10%: {agg['prec10_mean']:.2%} ± {agg['prec10_std']:.2%}   "
          f"(min {agg['prec10_min']:.2%})")
    print(f"  lift    : {agg['lift_mean']:.2f}x   (min {agg['lift_min']:.2f}x)")
    print(f"  windows ≥ 2x lift: {agg['windows_above_2x']} / {agg['n_windows']}")
    print(f"  → {csv_path.name}, {json_path.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Walk-forward evaluation")
    p.add_argument("--model", nargs="+", default=["lgbm", "tabm"],
                   choices=["lgbm", "ae_mlp", "tabm"])
    p.add_argument("--targets", nargs="+", default=None,
                   choices=list(MODEL_TARGETS),
                   help="Targets to evaluate (default: all 6)")
    p.add_argument("--n_windows",     type=int, default=6)
    p.add_argument("--window_months", type=int, default=3)
    p.add_argument("--wf_seeds",      type=int, default=1,
                   help="Seeds per window for TabM (default 1 — regime detection. "
                        "Use 5 for high-precision deployment-model validation.)")
    args = p.parse_args()

    targets = args.targets or MODEL_TARGETS
    for mt in args.model:
        for tgt in targets:
            try:
                run_walkforward(mt, tgt, args.n_windows, args.window_months,
                                wf_seeds=args.wf_seeds)
            except Exception as e:
                print(f"[wf] {mt}/{tgt} FAILED: {e}")
