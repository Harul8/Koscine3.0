"""
Daily inference — ensemble prediction and directional bucket assignment.

Public API (used by api/main.py):
    run(date_str, fetch_data=True)    end-to-end: fetch → silver → features → predict
    get_predictions(date_str)         load saved predictions CSV
    list_dates()                      list dates with saved predictions
    list_symbols()                    universe symbols seen across predictions
    list_active_symbols()             active watchlist
    get_stock_history(symbol, start, end)

Bucket assignment logic (v5 tiered):
  For each (symbol, direction):
    1. Ensemble z-score ≥ Z_BUCKET_THRESH  (unusually strong relative to universe)
    2. Prob ≥ ABS_PROB_FLOOR_MULT × model base rate  (absolute signal floor)
  STRONG  : z ≥ Z_BUCKET_THRESH + STRONG_Z_BONUS  AND  prob ≥ STRONG_PROB_MULT × base_rate
  MOD     : z ≥ Z_BUCKET_THRESH  AND  prob ≥ ABS_PROB_FLOOR_MULT × base_rate
  RANGE   : neither condition met

Usage:
    python -m pipeline.predict 2026-05-21
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    PREDICTIONS_DIR, MODEL_DIR, GOLD_DIR, GOLD_LABELS, GOLD_FEATURES,
    MODEL_TARGETS, Z_BUCKET_THRESH, ABS_PROB_FLOOR_MULT,
    PREDICT_TICKERS, INDEX_TICKERS,
    TIERED_MODEL_TARGETS, TIERED_OVERLAY_TARGETS, OVERLAY_ALPHA, LIQUID_TIER_SIZE,
    STRONG_Z_BONUS, STRONG_PROB_MULT,
    CLEAN_MODEL_TARGETS,
)
from .models.ensemble import load_prod_models, predict_ensemble, load_weights


PROD_DIR = MODEL_DIR / "prod"


# ── Bucket assignment ──────────────────────────────────────────────────────────

def _base_rates(models: dict) -> dict[str, float]:
    """Extract base rates from prod models {target: rate}."""
    rates: dict[str, float] = {}
    for target, type_models in models.items():
        for model in type_models.values():
            if hasattr(model, "base_rate") and not np.isnan(model.base_rate):
                rates[target] = model.base_rate
                break
    return rates


def _assign_buckets(
    scores: dict[str, np.ndarray],
    raw_probs: dict[str, np.ndarray],
    base_rates: dict[str, float],
    symbols: pd.Index,
) -> pd.DataFrame:
    """
    Assign UP/DN bucket for each symbol.

    Returns DataFrame with columns:
        symbol, bucket_up, bucket_dn,
        score_up_5, score_dn_5, score_up_5_xl, score_dn_5_xl,
        prob_up_5, prob_dn_5, prob_up_5_xl, prob_dn_5_xl
    """
    def _z_score(arr: np.ndarray) -> np.ndarray:
        mu, sd = arr.mean(), arr.std()
        return (arr - mu) / (sd + 1e-9)

    def _fires(target: str) -> np.ndarray:
        """Boolean array: does this target fire for each symbol?"""
        s = scores.get(target)
        p = raw_probs.get(target)
        br = base_rates.get(target, 0.20)
        if s is None or p is None:
            return np.zeros(len(symbols), dtype=bool)
        z = _z_score(s)
        return (z >= Z_BUCKET_THRESH) & (p >= ABS_PROB_FLOOR_MULT * br)

    up_fires  = _fires("up_5")
    dn_fires  = _fires("dn_5")
    xl_up     = _fires("up_5_xl")
    xl_dn     = _fires("dn_5_xl")

    bucket_up = np.where(up_fires & xl_up, "STRONG",
                np.where(up_fires,          "MOD", "RANGE"))
    bucket_dn = np.where(dn_fires & xl_dn, "STRONG",
                np.where(dn_fires,          "MOD", "RANGE"))

    rows: dict[str, object] = {"symbol": symbols}
    rows["bucket_up"] = bucket_up
    rows["bucket_dn"] = bucket_dn

    for t in MODEL_TARGETS:
        if t in scores:
            rows[f"score_{t}"] = np.round(scores[t], 4)
        if t in raw_probs:
            rows[f"prob_{t}"]  = np.round(raw_probs[t], 4)

    return pd.DataFrame(rows)


def _assign_buckets_tiered(
    scores: dict[str, np.ndarray],
    raw_probs: dict[str, np.ndarray],
    base_rates: dict[str, float],
    symbols: pd.Index,
    liquid_set: set[str],
) -> pd.DataFrame:
    """
    Tiered bucket assignment for v5 two-family models.

    Per symbol:
      - Liquid symbols use up_liq / dn_liq scores + bad_up_liquid / bad_dn_liquid overlay
      - Rest   symbols use up_rest / dn_rest scores + bad_up_rest  / bad_dn_rest  overlay

    Overlay penalty: final_score = base_prob × (1 − OVERLAY_ALPHA × overlay_prob)
    Then z-score the final scores across the full universe for bucket assignment.

    Bucket tiers (no XL in v5):
      STRONG : z >= Z_BUCKET_THRESH + STRONG_Z_BONUS  AND  prob >= STRONG_PROB_MULT × base_rate
               (defaults: z >= 1.70  AND  prob >= 2.0× base_rate)
      MOD    : z >= Z_BUCKET_THRESH  AND  prob >= ABS_PROB_FLOOR_MULT × base_rate
               (defaults: z >= 0.70  AND  prob >= 1.25× base_rate)
      RANGE  : everything else
    """
    n = len(symbols)
    sym_arr = np.array(symbols)
    is_liq  = np.array([s in liquid_set for s in sym_arr])

    def _get(key: str) -> np.ndarray:
        return scores.get(key, raw_probs.get(key, np.full(n, np.nan)))

    def _br(key: str, fallback: float = 0.07) -> float:
        return base_rates.get(key, fallback)

    # ── Composite scores: pick liq or rest per symbol ─────────────────────────
    up_base = np.where(is_liq, _get("up_liq"),  _get("up_rest"))
    dn_base = np.where(is_liq, _get("dn_liq"),  _get("dn_rest"))

    # Overlay probabilities (bad-close risk)
    ov_up = np.where(is_liq, _get("bad_up_liquid"), _get("bad_up_rest"))
    ov_dn = np.where(is_liq, _get("bad_dn_liquid"), _get("bad_dn_rest"))

    # Replace NaN overlays with 0 (no penalty if overlay missing)
    ov_up = np.where(np.isnan(ov_up), 0.0, ov_up)
    ov_dn = np.where(np.isnan(ov_dn), 0.0, ov_dn)

    # Apply overlay: final = base × (1 − alpha × overlay)
    up_final = up_base * (1.0 - OVERLAY_ALPHA * ov_up)
    dn_final = dn_base * (1.0 - OVERLAY_ALPHA * ov_dn)

    # ── Z-score across full universe ──────────────────────────────────────────
    def _z(arr: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(arr)
        mu = arr[valid].mean() if valid.any() else 0.0
        sd = arr[valid].std()  if valid.any() else 1.0
        return (arr - mu) / (sd + 1e-9)

    z_up = _z(up_final)
    z_dn = _z(dn_final)

    # Base rates for abs-prob threshold
    br_up_liq  = _br("up_liq",  0.30)
    br_up_rest = _br("up_rest", 0.14)
    br_dn_liq  = _br("dn_liq",  0.30)
    br_dn_rest = _br("dn_rest", 0.14)

    br_up = np.where(is_liq, br_up_liq,  br_up_rest)
    br_dn = np.where(is_liq, br_dn_liq,  br_dn_rest)

    STRONG_Z = Z_BUCKET_THRESH + STRONG_Z_BONUS   # e.g. 0.70 + 1.0 = 1.70
    MOD_Z    = Z_BUCKET_THRESH                    # e.g. 0.70

    up_strong = (z_up >= STRONG_Z) & (up_base >= STRONG_PROB_MULT * br_up)
    up_mod    = (z_up >= MOD_Z)    & (up_base >= ABS_PROB_FLOOR_MULT * br_up)
    dn_strong = (z_dn >= STRONG_Z) & (dn_base >= STRONG_PROB_MULT * br_dn)
    dn_mod    = (z_dn >= MOD_Z)    & (dn_base >= ABS_PROB_FLOOR_MULT * br_dn)

    bucket_up = np.where(up_strong, "STRONG", np.where(up_mod, "MOD", "RANGE"))
    bucket_dn = np.where(dn_strong, "STRONG", np.where(dn_mod, "MOD", "RANGE"))

    rows: dict = {
        "symbol":    sym_arr,
        "tier":      np.where(is_liq, "liquid", "rest"),
        "bucket_up": bucket_up,
        "bucket_dn": bucket_dn,
        "score_up":  np.round(up_final,  4),
        "score_dn":  np.round(dn_final,  4),
        "prob_up":   np.round(up_base,   4),
        "prob_dn":   np.round(dn_base,   4),
        "overlay_up": np.round(ov_up,    4),
        "overlay_dn": np.round(ov_dn,    4),
    }
    return pd.DataFrame(rows)


def _is_tiered(models: dict) -> bool:
    """Return True if prod models are the v5 tiered family."""
    return any(t in models for t in TIERED_MODEL_TARGETS)


def _score_clean_models(
    X: pd.DataFrame,
    scores: dict[str, np.ndarray],
    liquid_set: set[str],
) -> pd.DataFrame:
    """
    Score clean directional models and apply Layer-3 consensus gating.

    Tier-asymmetric: 4 models (up/dn × liq/rest) but the output is unified
    into 2 columns — each symbol gets scored by its own tier's model.

      liquid-30 symbols → clean_up_5_liq  / clean_dn_5_liq   (4% threshold)
      rest-35   symbols → clean_up_5_rest / clean_dn_5_rest  (7% threshold)

    Layer-3 gate (bull example, liquid):
      clean_bull_signal = 1  if:
        p_clean >= val_p95
        AND prod up_liq  score >= local p80
        AND prod bad_up_liquid <= local p50
    (rest version uses up_rest / bad_up_rest base scores.)

    Returns DataFrame with columns:
        symbol, p_clean_up, p_clean_dn, clean_bull_signal, clean_bear_signal
    """
    import pickle as _pkl
    import json as _json

    sym_arr = np.array(X["symbol"])
    is_liq  = np.array([s in liquid_set for s in sym_arr])
    n       = len(sym_arr)

    result = pd.DataFrame({
        "symbol":           sym_arr,
        "p_clean_up":       np.full(n, np.nan),
        "p_clean_dn":       np.full(n, np.nan),
        "clean_bull_signal": np.zeros(n, dtype=np.int8),
        "clean_bear_signal": np.zeros(n, dtype=np.int8),
    })

    # Per-target prod-base-score keys (Layer-3 consensus inputs)
    BASE_KEYS = {
        "clean_up_5_liq":  ("up_liq",  "bad_up_liquid"),
        "clean_dn_5_liq":  ("dn_liq",  "bad_dn_liquid"),
        "clean_up_5_rest": ("up_rest", "bad_up_rest"),
        "clean_dn_5_rest": ("dn_rest", "bad_dn_rest"),
    }

    for target in CLEAN_MODEL_TARGETS:
        pkl_path = PROD_DIR / f"{target}_clean_lgbm.pkl"
        thr_path = PROD_DIR / f"{target}_thresholds.json"

        if not pkl_path.exists():
            print(f"[predict] clean model {pkl_path.name} not found — skipping")
            continue

        # Pick rows for the model's tier
        target_is_liq = target.endswith("_liq")
        tier_mask     = is_liq if target_is_liq else (~is_liq)
        tier_idx      = np.where(tier_mask)[0]
        if len(tier_idx) == 0:
            continue
        X_tier = X.iloc[tier_idx].reset_index(drop=True)

        # Load + score
        try:
            with open(pkl_path, "rb") as f:
                model = _pkl.load(f)
            p_clean = model.predict_proba(X_tier)
        except Exception as e:
            print(f"[predict] clean model {target} score failed: {e}")
            continue

        # Val-tuned p95 threshold (saved at train time)
        p95_thr = None
        if thr_path.exists():
            try:
                thr = _json.loads(thr_path.read_text())
                p95_thr = thr.get("p95")
            except Exception:
                p95_thr = None
        if p95_thr is None:
            p95_thr = float(np.percentile(p_clean, 95)) if len(p_clean) > 0 else 1.0

        # Write probabilities into the unified output column
        is_bull  = "_up_" in target
        prob_col = "p_clean_up" if is_bull else "p_clean_dn"
        result.loc[tier_idx, prob_col] = p_clean

        # Layer-3 consensus gate — base scores from the right tier
        dir_key, bad_key = BASE_KEYS[target]
        prod_dir_scores  = scores.get(dir_key, np.zeros(n))
        prod_bad_scores  = scores.get(bad_key, np.zeros(n))

        dir_tier = prod_dir_scores[tier_idx]
        bad_tier = prod_bad_scores[tier_idx]

        # Compute gates vs local (in-tier) distribution on this day
        dir_p80 = (
            float(np.percentile(dir_tier[~np.isnan(dir_tier)], 80))
            if (~np.isnan(dir_tier)).any() else 0.0
        )
        bad_p50 = (
            float(np.percentile(bad_tier[~np.isnan(bad_tier)], 50))
            if (~np.isnan(bad_tier)).any() else 1.0
        )

        clean_fires = p_clean >= p95_thr
        gate = clean_fires & (dir_tier >= dir_p80) & (bad_tier <= bad_p50)

        sig_col = "clean_bull_signal" if is_bull else "clean_bear_signal"
        result.loc[tier_idx[gate], sig_col] = 1

    return result


# ── Core prediction ────────────────────────────────────────────────────────────

def predict_date(date_str: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Run ensemble inference for date_str.

    panel : optional pre-loaded features DataFrame (all dates).  When supplied,
            features are sliced from it instead of calling build_for_date() —
            use this for backtests to avoid rebuilding features 260× per year.

    Returns a DataFrame with one row per active symbol.
    """
    if panel is not None:
        d = pd.Timestamp(date_str)
        X = panel[panel["date"] == d].reset_index(drop=True)
        if X.empty:
            raise ValueError(f"No features in panel for {date_str}")
    else:
        from .features import build_for_date
        print(f"[predict] building features for {date_str} …")
        X = build_for_date(date_str)

    # ── Symbol filtering: use Universe file for tiered mode, PREDICT_TICKERS legacy ──
    # Load models first to detect tiered vs legacy, then apply correct universe filter.
    print(f"[predict] loading prod models from {PROD_DIR} …")
    models  = load_prod_models(PROD_DIR)
    weights = load_weights(PROD_DIR)

    if _is_tiered(models):
        # Tiered mode: filter to hand-curated Universe file (liquid-30 + rest-35)
        try:
            from .universe import load_predict_universe
            _univ_syms, liquid_set = load_predict_universe()
            _active_symbols = set(_univ_syms)
        except Exception:
            _active_symbols = set(PREDICT_TICKERS)
            liquid_set = set(PREDICT_TICKERS) - set(INDEX_TICKERS)
        X = X[X["symbol"].isin(_active_symbols)].reset_index(drop=True)
        # Load per-target blend weights (LGBM vs CatBoost ratio per target)
        _blend_path = PROD_DIR / "tiered_blend_weights.json"
        per_target_weights = (
            json.loads(_blend_path.read_text()) if _blend_path.exists() else None
        )
    else:
        # Legacy mode: filter to hardcoded PREDICT_TICKERS
        X = X[X["symbol"].isin(PREDICT_TICKERS)].reset_index(drop=True)
        liquid_set          = None   # resolved below in legacy branch
        per_target_weights  = None

    if X.empty:
        raise ValueError(f"No active symbols with features for {date_str}")

    print("[predict] running ensemble …")
    scores    = predict_ensemble(models, X, weights=weights,
                                 per_target_weights=per_target_weights)
    base_rates_ = _base_rates(models)

    # Also collect raw (uncalibrated-rank) probs for bucket threshold checks
    raw_probs: dict[str, np.ndarray] = {}
    for target, type_models in models.items():
        probs_list = []
        for mt, model in type_models.items():
            try:
                p = model.predict_proba(X)
                probs_list.append(p)
            except Exception:
                pass
        if probs_list:
            raw_probs[target] = np.mean(probs_list, axis=0)

    if _is_tiered(models):
        # ── v5 tiered bucket assignment ───────────────────────────────────────
        # liquid_set already loaded above during symbol filtering
        out = _assign_buckets_tiered(scores, raw_probs, base_rates_,
                                     X["symbol"], liquid_set)
        sort_col = "score_up"

        # ── Clean directional model scoring (Layer 1+2+3) ─────────────────────
        clean_df = _score_clean_models(X, scores, liquid_set)
        out = out.merge(clean_df[["symbol", "p_clean_up", "p_clean_dn",
                                   "clean_bull_signal", "clean_bear_signal"]],
                        on="symbol", how="left")
    else:
        # ── legacy bucket assignment ──────────────────────────────────────────
        out = _assign_buckets(scores, raw_probs, base_rates_, X["symbol"])
        # is_liquid: all non-index PREDICT_TICKERS are liquid (legacy behaviour)
        _liquid = set(PREDICT_TICKERS) - set(INDEX_TICKERS)
        out["is_liquid"] = out["symbol"].isin(_liquid)
        sort_col = "score_up_5"

    out.insert(0, "date", date_str)

    # target_date: 5 business days after date_str
    _target = pd.bdate_range(date_str, periods=6)[-1]
    out["target_date"] = _target.strftime("%Y-%m-%d")

    # Sort: STRONG → MOD → RANGE, then primary score descending
    _TIER_ORDER = {"STRONG": 0, "MOD": 1, "RANGE": 2}
    out["_up_tier"] = out["bucket_up"].map(_TIER_ORDER)
    out = (
        out.sort_values(["_up_tier", sort_col], ascending=[True, False])
           .drop(columns=["_up_tier"])
           .reset_index(drop=True)
    )
    return out


def _save_predictions(df: pd.DataFrame, date_str: str) -> Path:
    # Ensure run_date column exists (API reads this from CSVs)
    if "run_date" not in df.columns:
        df = df.copy()
        df.insert(0, "run_date", date_str)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    df.to_csv(path, index=False)
    print(f"[predict] saved → {path}  ({len(df)} symbols)")
    return path


# ── End-to-end daily run ───────────────────────────────────────────────────────

def _ensure_data(start: str | None, end: str) -> None:
    """Fetch raw + rebuild silver. If `start` is None, fetch only `end` day.
    Otherwise fetches the inclusive [start, end] range."""
    from . import fetch, silver
    if start is None or start == end:
        print(f"[predict] fetching {end} …")
        fetch.fetch_day(end)
    else:
        print(f"[predict] fetching {start} → {end} …")
        fetch.backfill(start, end)
    print(f"[predict] rebuilding silver tables …")
    silver.main()


def run(date_str: str, fetch_data: bool = True) -> pd.DataFrame:
    """Single-date end-to-end: fetch → silver → features → predict → save CSV.
    Called by api/main.py and CLI single-date mode."""
    if fetch_data:
        _ensure_data(None, date_str)
    df = predict_date(date_str)
    _save_predictions(df, date_str)
    return df


def run_range(start: str, end: str, fetch_data: bool = True,
              build_features: bool = True,
              force: bool = False, progress_cb=None) -> list[str]:
    """Range end-to-end: fetch ONCE for whole [start, end], then predict
    each business day in the range. Skips days that already have a CSV
    unless `force=True`. Returns list of date strings successfully predicted.

    build_features=False : load gold/features.parquet once and slice per date.
                           Use this for backtests over already-built data
                           (avoids rebuilding features 260× per year).
    """
    if fetch_data:
        _ensure_data(start, end)

    # Load feature panel once if skipping per-date feature builds
    panel: pd.DataFrame | None = None
    if not build_features:
        print(f"[predict] loading pre-built features panel from {GOLD_FEATURES} …")
        panel = pd.read_parquet(GOLD_FEATURES)
        panel["date"] = pd.to_datetime(panel["date"])
        print(f"[predict] panel: {len(panel):,} rows  "
              f"dates {panel['date'].min().date()} → {panel['date'].max().date()}")

    done: list[str] = []
    dates = pd.bdate_range(start, end)
    for i, ts in enumerate(dates, 1):
        d = ts.strftime("%Y-%m-%d")
        path = PREDICTIONS_DIR / f"predictions_{d}.csv"
        if path.exists() and not force:
            if progress_cb:
                progress_cb(f"skip {d} ({i}/{len(dates)}) — exists")
            print(f"[predict] {d} ({i}/{len(dates)}) already done — skip")
            continue
        if progress_cb:
            progress_cb(f"predict {d} ({i}/{len(dates)})")
        print(f"[predict] {d} ({i}/{len(dates)}) …")
        try:
            df = predict_date(d, panel=panel)
            _save_predictions(df, d)
            done.append(d)
        except Exception as e:
            print(f"[predict] {d} FAILED: {e}")
    return done


# ── Query helpers (for API) ────────────────────────────────────────────────────

def get_predictions(date_str: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No predictions for {date_str} — run predict first")
    return pd.read_csv(path)


def list_dates() -> list[str]:
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(
        p.stem.replace("predictions_", "")
        for p in PREDICTIONS_DIR.glob("predictions_*.csv")
    )


def list_symbols() -> list[str]:
    dates = list_dates()
    if not dates:
        return []
    df = get_predictions(dates[-1])
    return sorted(df["symbol"].unique().tolist())


def list_active_symbols() -> list[str]:
    return list(PREDICT_TICKERS)


def batch_predict(
    start: str,
    end: str,
    progress_cb=None,
) -> None:
    """
    Run predict_date for every business day in [start, end].
    Skips dates that already have a predictions CSV.
    progress_cb: optional callable(msg: str) for progress reporting.
    """
    dates = pd.bdate_range(start, end)
    for i, ts in enumerate(dates, 1):
        date_str = ts.strftime("%Y-%m-%d")
        path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
        if path.exists():
            if progress_cb:
                progress_cb(f"skip {date_str} (already done)")
            continue
        if progress_cb:
            progress_cb(f"predict {i}/{len(dates)}: {date_str}")
        try:
            df = predict_date(date_str)
            _save_predictions(df, date_str)
        except Exception as e:
            print(f"[predict] batch: {date_str} failed: {e}")


def get_stock_history(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return prediction history for a symbol across saved CSVs, with actuals joined."""
    dfs = []
    for ts in pd.bdate_range(start, end):
        d = ts.strftime("%Y-%m-%d")
        path = PREDICTIONS_DIR / f"predictions_{d}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                row = df[df["symbol"] == symbol]
                if not row.empty:
                    dfs.append(row)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    hist = pd.concat(dfs, ignore_index=True)
    hist["run_date"] = pd.to_datetime(hist["run_date"]).dt.normalize()

    # Join actuals from labels parquet
    # GOLD_LABELS has ret_up_5 / ret_dn_5 (peak return vs T+1 open);
    # frontend expects upside_t5 / downside_t5 — rename on load.
    try:
        labs = pd.read_parquet(
            GOLD_LABELS,
            columns=["date", "symbol", "ret_up_5", "ret_dn_5"],
        )
        labs = labs.rename(columns={"ret_up_5": "upside_t5",
                                    "ret_dn_5": "downside_t5"})
        labs["date"] = pd.to_datetime(labs["date"]).dt.normalize()
        labs = labs[labs["symbol"] == symbol]
        hist = hist.merge(
            labs, left_on=["run_date", "symbol"],
            right_on=["date", "symbol"], how="left",
        ).drop(columns=["date"], errors="ignore")
    except Exception as e:
        print(f"[predict] warn: actuals join failed: {e}")

    hist["run_date"] = hist["run_date"].dt.strftime("%Y-%m-%d")
    return hist.sort_values("run_date", ascending=False).reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_strong_signals(df: pd.DataFrame, date_str: str) -> None:
    """Print STRONG UP / STRONG DN signals for one prediction CSV."""
    su = df[df["bucket_up"] == "STRONG"]
    sd = df[df["bucket_dn"] == "STRONG"]
    print(f"\n=== {date_str} ===")

    # Support both tiered (v5) and legacy column names
    tiered = "score_up" in df.columns
    up_cols = (["symbol", "tier", "prob_up", "overlay_up", "score_up"]
               if tiered else ["symbol", "prob_up_5", "prob_up_5_xl", "score_up_5"])
    dn_cols = (["symbol", "tier", "prob_dn", "overlay_dn", "score_dn"]
               if tiered else ["symbol", "prob_dn_5", "prob_dn_5_xl", "score_dn_5"])
    up_cols = [c for c in up_cols if c in df.columns]
    dn_cols = [c for c in dn_cols if c in df.columns]

    print(f"STRONG UP ({len(su)}):")
    if not su.empty:
        print(su[up_cols].to_string(index=False))
    print(f"STRONG DN ({len(sd)}):")
    if not sd.empty:
        print(sd[dn_cols].to_string(index=False))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Daily prediction pipeline (single date or range)",
        epilog=(
            "Examples:\n"
            "  python -m pipeline.predict 2026-05-21                # single date\n"
            "  python -m pipeline.predict 2026-05-15 2026-05-21     # range\n"
            "  python -m pipeline.predict 2026-05-21 --no_fetch     # skip data fetch\n"
            "  python -m pipeline.predict 2026-05-21 --force        # re-run existing\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("start", help="Prediction date YYYY-MM-DD (or start of range)")
    p.add_argument("end", nargs="?", default=None,
                   help="End date for range mode (optional, inclusive)")
    p.add_argument("--no_fetch", action="store_true",
                   help="Skip data fetch + silver rebuild (use existing data)")
    p.add_argument("--no_features", action="store_true",
                   help="Skip per-date feature build — use gold/features.parquet "
                        "(fast backtest mode, range only)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if predictions CSV already exists")
    args = p.parse_args()

    fetch_data     = not args.no_fetch
    build_features = not args.no_features

    if args.end is None:
        # ── Single date mode ────────────────────────────────────────────────
        path = PREDICTIONS_DIR / f"predictions_{args.start}.csv"
        if path.exists() and not args.force:
            print(f"[predict] {path.name} exists — loading (use --force to re-run)")
            df = pd.read_csv(path)
        else:
            df = run(args.start, fetch_data=fetch_data)
        _print_strong_signals(df, args.start)
    else:
        # ── Range mode ─────────────────────────────────────────────────────
        done = run_range(args.start, args.end, fetch_data=fetch_data,
                         build_features=build_features,
                         force=args.force)
        dates = pd.bdate_range(args.start, args.end)

        # Aggregate summary
        print(f"\n{'='*70}")
        print(f"RANGE SUMMARY  {args.start} → {args.end}  "
              f"({len(dates)} business days)")
        print('='*70)
        print(f"Predicted this run : {len(done)}")
        print(f"Already on disk    : {len(dates) - len(done)}")

        total_up = total_dn = 0
        per_day_rows = []
        for ts in dates:
            d = ts.strftime("%Y-%m-%d")
            path = PREDICTIONS_DIR / f"predictions_{d}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            nu = int((df["bucket_up"] == "STRONG").sum())
            nd = int((df["bucket_dn"] == "STRONG").sum())
            total_up += nu
            total_dn += nd
            per_day_rows.append((d, nu, nd))

        print(f"\n{'Date':<12} {'STRONG UP':>10} {'STRONG DN':>10}")
        for d, nu, nd in per_day_rows:
            print(f"{d:<12} {nu:>10} {nd:>10}")
        print(f"{'TOTAL':<12} {total_up:>10} {total_dn:>10}")
