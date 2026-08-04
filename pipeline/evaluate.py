"""
Evaluation — trader-meaningful metrics + ML quality check.

Primary output: bucket performance table.
  For each (model_type, direction, bucket):
    - N signals
    - Hit rate  : % of signals where label = 1  (did it actually move?)
    - Avg peak return T+3  : mean ret_up_3 / ret_dn_3  (entry T+1 open → peak within 3 days)
    - Avg peak return T+5  : mean ret_up_5 / ret_dn_5  (entry T+1 open → peak within 5 days)
    - Lift : hit_rate / base_rate

Bucket assignment uses the SAME logic as predict.py (Z-score + abs prob threshold),
so the number you see here is what you'd get in production.

Secondary outputs: AP / AUC per model (for model selection), calibration curves.

Outputs to gold/eval/:
    bucket_perf.csv          primary — bucket hit rate + peak returns
    ap_summary.csv           per-model-type AP / AUC / prec@10%
    calibration.csv          predicted prob bin vs actual hit rate

Usage:
    python -m pipeline.evaluate                    # val split, prod models
    python -m pipeline.evaluate --split test
    python -m pipeline.evaluate --run_dir models/runs/lgbm_20260521_120000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    GOLD_FEATURES, GOLD_LABELS, GOLD_DIR, MODEL_DIR,
    MODEL_TARGETS, SEARCH_AP_WEIGHTS,
    Z_BUCKET_THRESH, ABS_PROB_FLOOR_MULT,
)
from .models import get_model_class
from .models.ensemble import load_prod_models, predict_ensemble, load_weights


EVAL_DIR = GOLD_DIR / "eval"
PROD_DIR = MODEL_DIR / "prod"

# All label + return columns that come from labels.parquet
_LABEL_COLS = MODEL_TARGETS + ["ret_up_3", "ret_dn_3", "ret_up_5", "ret_dn_5"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_split(split: str, labels_path: Path) -> pd.DataFrame:
    feat = pd.read_parquet(GOLD_FEATURES)
    feat["date"] = pd.to_datetime(feat["date"])

    labs = pd.read_parquet(labels_path)
    labs["date"] = pd.to_datetime(labs["date"])

    keep_lab = ["date", "symbol", "split"] + _LABEL_COLS
    keep_lab = [c for c in keep_lab if c in labs.columns]

    panel = feat.merge(labs[keep_lab], on=["date", "symbol"], how="inner")
    out   = panel[panel["split"] == split].drop(columns=["split"]).reset_index(drop=True)
    print(f"[evaluate] split={split}  rows={len(out):,}  "
          f"symbols={out['symbol'].nunique()}  dates={out['date'].nunique()}")
    return out


# ── Per-model inference ────────────────────────────────────────────────────────

def _load_models_from_dir(models_dir: Path) -> dict[str, dict]:
    """
    Scan models_dir for pkl files, load each model.
    Returns {target: {model_type: model}}.
    """
    out: dict[str, dict] = {}
    for pkl in sorted(models_dir.glob("*.pkl")):
        stem = pkl.stem
        for mt in ["lgbm", "ae_mlp", "tabm"]:
            if stem.endswith(f"_{mt}"):
                target = stem[: -(len(mt) + 1)]
                cls = get_model_class(mt)
                try:
                    m = cls.load(pkl)
                    out.setdefault(target, {})[mt] = m
                except Exception as e:
                    print(f"  [evaluate] could not load {pkl.name}: {e}")
                break
    return out


def _load_models_from_dirs(run_dirs: dict[str, Path]) -> dict[str, dict]:
    """
    Load models from multiple run dirs — one per model type.
    run_dirs: {model_type: Path}
    Returns {target: {model_type: model}}.
    """
    out: dict[str, dict] = {}
    for mt, d in run_dirs.items():
        for pkl in sorted(Path(d).glob(f"*_{mt}.pkl")):
            stem   = pkl.stem
            target = stem[: -(len(mt) + 1)]
            cls    = get_model_class(mt)
            try:
                m = cls.load(pkl)
                out.setdefault(target, {})[mt] = m
            except Exception as e:
                print(f"  [evaluate] could not load {pkl.name}: {e}")
    return out


def _infer_all(models: dict, X: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Run predict_proba for every (target, model_type).
    Returns {target: {model_type: probs_array}}.
    """
    probs: dict[str, dict[str, np.ndarray]] = {}
    for target, type_models in models.items():
        for mt, model in type_models.items():
            try:
                p = model.predict_proba(X)
                probs.setdefault(target, {})[mt] = p
            except Exception as e:
                print(f"  [evaluate] infer {target}/{mt}: {e}")
    return probs


# ── Bucket assignment (mirrors predict.py logic) ───────────────────────────────

def _z_score(arr: np.ndarray) -> np.ndarray:
    mu, sd = arr.mean(), arr.std()
    return (arr - mu) / (sd + 1e-9)


def _bucket_series(
    scores: np.ndarray,
    raw_probs: np.ndarray,
    base_rate: float,
) -> np.ndarray:
    """
    Returns string array: "STRONG", "MOD", or "RANGE" per row.
    Mirrors the two-condition gate in predict.py.
    """
    z     = _z_score(scores)
    fires = (z >= Z_BUCKET_THRESH) & (raw_probs >= ABS_PROB_FLOOR_MULT * base_rate)
    return np.where(fires, "SIGNAL", "RANGE")


# ── Bucket performance table ───────────────────────────────────────────────────

def _bucket_perf(
    data: pd.DataFrame,
    all_probs: dict[str, dict[str, np.ndarray]],
    all_scores: dict[str, np.ndarray] | None = None,   # ensemble scores
    ensemble_name: str = "ensemble",
) -> pd.DataFrame:
    """
    Build the primary bucket-performance table.

    For each (model_type, direction target, bucket):
        n, hit_rate, avg_ret_t3, avg_ret_t5, lift vs base rate

    Parameters
    ----------
    all_probs  : {target: {model_type: raw_prob_array}}
    all_scores : optional ensemble rank scores {target: array} — if provided,
                 bucket assignment uses ensemble score; otherwise uses raw prob
    """
    rows = []

    # base rates from training-split data aren't available here;
    # use the empirical rate from the eval split itself as denominator for lift
    base_rates = {t: float(data[t].mean()) for t in MODEL_TARGETS if t in data.columns}

    direction_map = {
        "up_3":    ("up",   "ret_up_3", "ret_up_5", "up_5"),
        "dn_3":    ("dn",   "ret_dn_3", "ret_dn_5", "dn_5"),
        "up_5":    ("up",   "ret_up_3", "ret_up_5", "up_5"),
        "dn_5":    ("dn",   "ret_dn_3", "ret_dn_5", "dn_5"),
        "up_5_xl": ("up",   "ret_up_3", "ret_up_5", "up_5"),
        "dn_5_xl": ("dn",   "ret_dn_3", "ret_dn_5", "dn_5"),
    }

    def _one_model(model_label: str, target: str, probs: np.ndarray,
                   scores: np.ndarray | None):
        if target not in data.columns:
            return
        direction, ret3_col, ret5_col, hit_target = direction_map[target]
        # Use each target's own empirical positive rate as the lift denominator.
        # xl targets have ~9% base rate; using hit_target's rate (~18%) made xl
        # lifts appear negative when they were actually 1.4–1.9x.
        base = base_rates.get(target, 0.20)
        y    = data[target].astype(float).values
        ok   = ~np.isnan(y)

        # Use ensemble scores for bucket gate if provided; fall back to raw probs
        gate_scores = scores if scores is not None else probs
        buckets = _bucket_series(gate_scores[ok], probs[ok], base)

        y_ok   = y[ok]
        ret3   = data[ret3_col].values[ok] if ret3_col in data.columns else None
        ret5   = data[ret5_col].values[ok] if ret5_col in data.columns else None

        for bkt in ["SIGNAL", "RANGE"]:
            mask = buckets == bkt
            n = mask.sum()
            if n == 0:
                continue
            hit   = float(y_ok[mask].mean())
            lift  = hit / base if base > 0 else float("nan")
            r3    = float(np.nanmean(ret3[mask])) if ret3 is not None else float("nan")
            r5    = float(np.nanmean(ret5[mask])) if ret5 is not None else float("nan")
            rows.append({
                "model": model_label, "target": target,
                "direction": direction, "bucket": bkt,
                "n": int(n), "hit_rate": round(hit, 4),
                "base_rate": round(base, 4), "lift": round(lift, 2),
                "avg_peak_ret_t3": round(r3, 4),
                "avg_peak_ret_t5": round(r5, 4),
            })

    # ── Per model-type per target ──────────────────────────────────────────────
    all_model_types: set[str] = set()
    for target_probs in all_probs.values():
        all_model_types.update(target_probs.keys())

    for mt in sorted(all_model_types):
        for target in MODEL_TARGETS:
            if target not in all_probs or mt not in all_probs[target]:
                continue
            probs  = all_probs[target][mt]
            ens_sc = (all_scores or {}).get(target)
            _one_model(mt, target, probs, ens_sc)

    # ── Ensemble (uses ensemble scores as both gate and prob estimate) ─────────
    if all_scores:
        for target in MODEL_TARGETS:
            if target not in all_scores:
                continue
            # Average raw probs across model types as the displayed probability
            raw_list = [all_probs[target][mt]
                        for mt in all_probs.get(target, {})
                        if mt in all_probs.get(target, {})]
            if not raw_list:
                continue
            avg_raw = np.mean(raw_list, axis=0)
            _one_model(ensemble_name, target, avg_raw, all_scores[target])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["target", "model", "bucket"]).reset_index(drop=True)


# ── AP / AUC table ─────────────────────────────────────────────────────────────

def _ap_table(all_probs: dict, data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    all_model_types: set[str] = set()
    for tp in all_probs.values():
        all_model_types.update(tp.keys())

    for mt in sorted(all_model_types):
        for target in MODEL_TARGETS:
            if target not in all_probs or mt not in all_probs[target]:
                continue
            probs  = all_probs[target][mt]
            y_true = data[target].astype(float).values
            ok     = ~np.isnan(y_true)
            y_ok, p_ok = y_true[ok], probs[ok]
            n = ok.sum()
            if n < 10 or len(np.unique(y_ok)) < 2:
                continue
            ap  = float(average_precision_score(y_ok, p_ok))
            auc = float(roc_auc_score(y_ok, p_ok))
            top = max(1, int(0.10 * n))
            idx = np.argsort(p_ok)[::-1]
            p10 = float(y_ok[idx[:top]].mean())
            base = float(y_ok.mean())
            rows.append({
                "model_type": mt, "target": target,
                "ap": round(ap, 4), "auc": round(auc, 4),
                "prec_at_10pct": round(p10, 4),
                "lift": round(p10 / base, 2) if base > 0 else float("nan"),
                "base_rate": round(base, 4), "n": n,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Add composite AP row per model type
    extras = []
    for mt, grp in df.groupby("model_type"):
        total = sum(SEARCH_AP_WEIGHTS.get(r["target"], 0) * r["ap"]
                    for _, r in grp.iterrows())
        extras.append({"model_type": mt, "target": "COMPOSITE_AP",
                        "ap": round(total, 4), "auc": float("nan"),
                        "prec_at_10pct": float("nan"), "lift": float("nan"),
                        "base_rate": float("nan"), "n": float("nan")})
    return pd.concat([df, pd.DataFrame(extras)], ignore_index=True)


# ── Calibration curve ──────────────────────────────────────────────────────────

def _calibration(all_probs: dict, data: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for mt, target_map in _invert(all_probs).items():
        for target, probs in target_map.items():
            if target not in data.columns:
                continue
            y_true = data[target].astype(float).values
            ok     = ~np.isnan(y_true)
            y_ok, p_ok = y_true[ok], probs[ok]
            bins = np.percentile(p_ok, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            for i in range(len(bins) - 1):
                mask = (p_ok >= bins[i]) & (p_ok < bins[i + 1])
                if mask.sum() < 5:
                    continue
                rows.append({
                    "model_type": mt, "target": target,
                    "prob_bin_lo": round(bins[i],     4),
                    "prob_bin_hi": round(bins[i + 1], 4),
                    "mean_pred":  round(float(p_ok[mask].mean()), 4),
                    "actual_rate": round(float(y_ok[mask].mean()), 4),
                    "n": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def _invert(all_probs: dict) -> dict[str, dict[str, np.ndarray]]:
    """Invert {target: {mt: arr}} → {mt: {target: arr}}."""
    out: dict[str, dict] = {}
    for target, tp in all_probs.items():
        for mt, arr in tp.items():
            out.setdefault(mt, {})[target] = arr
    return out


# ── Full evaluation ────────────────────────────────────────────────────────────

def run_evaluation(
    split: str = "val",
    models_dir: Path | None = None,
    run_dirs: dict[str, Path] | None = None,   # {model_type: run_dir} — used by automl
    labels_path: Path | None = None,
    save_dir: Path | None = None,              # override output dir (default: EVAL_DIR)
) -> dict:
    """
    Run full evaluation. Returns a metrics dict for programmatic comparison.

    Returns:
        {
          "composite_ap":         float,              # weighted AP across all models/targets
          "composite_ap_by_mt":   {model_type: float},
          "signal_lift_up":       float,              # avg SIGNAL lift on UP targets
          "signal_lift_dn":       float,
          "auc_by_target":        {target: {mt: float}},
        }
    """
    if run_dirs:
        # Multi-dir mode (automl per-config eval): one run dir per model type
        models_obj = _load_models_from_dirs(run_dirs)
        ens_dir    = None   # no ensemble weights yet during per-config eval
    else:
        models_dir = models_dir or PROD_DIR
        models_obj = _load_models_from_dir(models_dir)
        ens_dir    = models_dir

    labels_path = labels_path or GOLD_LABELS
    out_dir     = Path(save_dir) if save_dir else EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_split(split, labels_path)
    if data.empty:
        print(f"[evaluate] no data for split={split}")
        return {}

    feat_cols = [c for c in data.columns
                 if c not in (["date", "symbol"] + MODEL_TARGETS + _LABEL_COLS)]
    X = data[["date", "symbol"] + feat_cols]

    print("[evaluate] loading models …")
    all_probs = _infer_all(models_obj, X)

    if not all_probs:
        print("[evaluate] no models produced predictions")
        return {}

    # ── Ensemble scores (if weights present) ─────────────────────────────────
    all_scores = None
    if ens_dir:
        weights_file = ens_dir / "ensemble_weights.json"
        if weights_file.exists():
            weights = load_weights(ens_dir)
            all_scores = predict_ensemble(models_obj, X, weights=weights)
            print(f"[evaluate] ensemble weights loaded: {weights}")

    # ── Bucket performance ────────────────────────────────────────────────────
    print("\n[evaluate] computing bucket performance …")
    bp = _bucket_perf(data, all_probs, all_scores)
    if not bp.empty:
        bp.to_csv(out_dir / "bucket_perf.csv", index=False)
        _print_bucket_perf(bp)

    # ── AP / AUC ──────────────────────────────────────────────────────────────
    print("\n[evaluate] computing AP / AUC …")
    ap_df = _ap_table(all_probs, data)
    if not ap_df.empty:
        ap_df.to_csv(out_dir / "ap_summary.csv", index=False)
        comp = ap_df[ap_df["target"] == "COMPOSITE_AP"][["model_type", "ap"]]
        print("\nComposite AP by model type:")
        print(comp.to_string(index=False))

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n[evaluate] computing calibration …")
    cal = _calibration(all_probs, data)
    if not cal.empty:
        cal.to_csv(out_dir / "calibration.csv", index=False)
        print(f"Calibration saved → {out_dir / 'calibration.csv'}")

    print(f"\n[evaluate] results saved → {out_dir}")

    # ── Build return metrics dict ─────────────────────────────────────────────
    metrics: dict = {}

    if not ap_df.empty:
        comp_rows = ap_df[ap_df["target"] == "COMPOSITE_AP"]
        metrics["composite_ap_by_mt"] = dict(
            zip(comp_rows["model_type"], comp_rows["ap"])
        )
        vals = [v for v in metrics["composite_ap_by_mt"].values() if not np.isnan(v)]
        metrics["composite_ap"] = float(np.mean(vals)) if vals else float("nan")

        # AUC by target × model_type
        auc_rows = ap_df[ap_df["target"] != "COMPOSITE_AP"]
        metrics["auc_by_target"] = {}
        for _, row in auc_rows.iterrows():
            metrics["auc_by_target"].setdefault(row["target"], {})[row["model_type"]] = row["auc"]

    if not bp.empty:
        sig = bp[bp["bucket"] == "SIGNAL"]
        up_sig = sig[sig["direction"] == "up"]["lift"].replace([np.inf, -np.inf], np.nan)
        dn_sig = sig[sig["direction"] == "dn"]["lift"].replace([np.inf, -np.inf], np.nan)
        metrics["signal_lift_up"] = float(up_sig.mean()) if not up_sig.empty else float("nan")
        metrics["signal_lift_dn"] = float(dn_sig.mean()) if not dn_sig.empty else float("nan")

    return metrics


def _print_bucket_perf(df: pd.DataFrame) -> None:
    print(f"\n{'='*85}")
    print(f"{'MODEL':<12} {'TARGET':<10} {'BUCKET':<8} "
          f"{'N':>6} {'HIT%':>7} {'BASE%':>7} {'LIFT':>6} "
          f"{'RET_T3':>8} {'RET_T5':>8}")
    print("-"*85)
    for _, r in df.iterrows():
        print(f"{r['model']:<12} {r['target']:<10} {r['bucket']:<8} "
              f"{r['n']:>6,} {r['hit_rate']:>6.1%} {r['base_rate']:>6.1%} "
              f"{r['lift']:>6.2f} "
              f"{r['avg_peak_ret_t3']:>7.1%} {r['avg_peak_ret_t5']:>7.1%}")
    print("="*85)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Model evaluation")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--run_dir", default=None,
                   help="Evaluate a specific run dir instead of prod/")
    args = p.parse_args()

    models_dir = Path(args.run_dir) if args.run_dir else None
    run_evaluation(split=args.split, models_dir=models_dir)
