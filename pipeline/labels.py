"""
Label builder — v4.  Per-stock ATR-adaptive thresholds with data-driven K.

thresh(stock, T) = K × ATR_20d(stock, T), clipped to [CLIP_LO, CLIP_HI].

K is NOT a tunable constant.  It is derived from training data each run:

    K = percentile(forward_return / ATR_20d,  100 × (1 − target_rate))

This guarantees label base rates equal LABEL_TARGET_RATE_BASE (≈20%) and
LABEL_TARGET_RATE_XL (≈10%) automatically — no manual re-calibration needed
after new data arrives or ATR levels shift.

Up and down directions are calibrated separately: K_up ≠ K_dn because equity
markets have a positive drift, so the up distribution has fatter right tails.

Output: gold/labels.parquet  →  date, symbol, split,
                                up_3, dn_3, up_5, dn_5, up_5_xl, dn_5_xl,
                                ret_up_3, ret_dn_3, ret_up_5, ret_dn_5

Usage:
    python -m pipeline.labels
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    SILVER_TABLES, GOLD_LABELS, GOLD_DIR,
    LABEL_HORIZON_3D, LABEL_HORIZON_5D,
    LABEL_ATR_PERIOD,
    LABEL_TARGET_RATE_BASE, LABEL_TARGET_RATE_XL,
    LABEL_ATR_CLIP_LO, LABEL_ATR_CLIP_HI,
    TRAIN_END, VAL_END,
    GOLD_LABELS_TIERED,
    LIQUID_THRESHOLD, REST_THRESHOLD, BAD_CLOSE_THRESH,
    LIQUID_TIER_SIZE,
    CLEAN_UP_THRESH_LIQ,  CLEAN_DN_THRESH_LIQ,
    CLEAN_UP_THRESH_REST, CLEAN_DN_THRESH_REST,
    CLEAN_NOISE_THRESH,
)

GOLD_DIR.mkdir(parents=True, exist_ok=True)


def _split(date: pd.Series) -> pd.Series:
    d = pd.to_datetime(date)
    out = pd.Series("test", index=date.index)
    out[d <= TRAIN_END] = "train"
    out[(d > TRAIN_END) & (d <= VAL_END)] = "val"
    return out


def _compute_atr(stock: pd.DataFrame, period: int = 20) -> np.ndarray:
    """
    ATR as fraction of close, per (symbol, date).
    stock must be sorted by (symbol, date) with reset index.
    """
    g = stock.groupby("symbol", sort=False, group_keys=False)
    prev_close = g["close"].transform(lambda s: s.shift(1))
    tr_pct = (pd.concat([
        stock["high"] - stock["low"],
        (stock["high"] - prev_close).abs(),
        (stock["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1) / stock["close"])

    tmp = pd.DataFrame({"symbol": stock["symbol"], "_tr": tr_pct})
    atr = (tmp.groupby("symbol", sort=False)["_tr"]
              .transform(lambda s: s.rolling(period, min_periods=max(5, period // 2)).mean()))
    return atr.to_numpy()


def _compute_forward_extremes(
    stock: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward extremes used for both K calibration and binary labeling.

    Returns (O_next, max_H3, min_L3, max_H5, min_L5) as numpy arrays.
      O_next  — T+1 open (entry price reference)
      max_H3  — max(high) over T+1 … T+3
      min_L3  — min(low)  over T+1 … T+3
      max_H5  — max(high) over T+1 … T+5
      min_L5  — min(low)  over T+1 … T+5
    """
    n = len(stock)
    g = stock.groupby("symbol", sort=False, group_keys=False)
    O_next = g["open"].transform(lambda s: s.shift(-1)).to_numpy()

    max_H3 = np.full(n, np.nan)
    min_L3 = np.full(n, np.nan)
    max_H5 = np.full(n, np.nan)
    min_L5 = np.full(n, np.nan)

    for k in range(1, LABEL_HORIZON_5D + 1):
        Hk = g["high"].transform(lambda s, _k=k: s.shift(-_k)).to_numpy()
        Lk = g["low"].transform(lambda s, _k=k: s.shift(-_k)).to_numpy()
        if k <= LABEL_HORIZON_3D:
            max_H3 = np.fmax(max_H3, Hk)
            min_L3 = np.fmin(min_L3, Lk)
        max_H5 = np.fmax(max_H5, Hk)
        min_L5 = np.fmin(min_L5, Lk)

    return O_next, max_H3, min_L3, max_H5, min_L5


def _calibrate_k(
    ret: np.ndarray,
    atr: np.ndarray,
    tr_mask: np.ndarray,
    target_rate: float,
) -> float:
    """
    K = (1 − target_rate) quantile of (forward_return / ATR_20d),
    computed on training rows with valid ATR and forward return.

    Example: target_rate=0.20 → K is the 80th percentile of (ret/ATR),
    so thresh = K × ATR will be exceeded by exactly 20% of training rows
    (before CLIP_LO/CLIP_HI are applied).
    """
    mask = tr_mask & ~np.isnan(ret) & ~np.isnan(atr) & (atr > 0) & (ret >= 0)
    ratio = ret[mask] / atr[mask]
    return float(np.nanpercentile(ratio, 100.0 * (1.0 - target_rate)))


def _make_thresh(k: float, atr: np.ndarray) -> np.ndarray:
    t = np.clip(k * atr, LABEL_ATR_CLIP_LO, LABEL_ATR_CLIP_HI)
    return np.where(np.isnan(t), LABEL_ATR_CLIP_LO, t)


def _binary_labels(
    O_next: np.ndarray,
    extreme: np.ndarray,
    thresh: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    direction='up'  → label=1 if extreme (max_H) >= O_next × (1 + thresh)
    direction='dn'  → label=1 if extreme (min_L) <= O_next × (1 - thresh)
    """
    ref_valid = ~np.isnan(O_next)
    if direction == "up":
        return (ref_valid & (extreme >= O_next * (1.0 + thresh))).astype(np.int8)
    else:
        return (ref_valid & (extreme <= O_next * (1.0 - thresh))).astype(np.int8)


def build() -> pd.DataFrame:
    stock = pd.read_parquet(
        SILVER_TABLES["eod_stock"],
        columns=["date", "symbol", "open", "close", "high", "low"],
    )
    deriv = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"], columns=["date", "symbol"])

    fo_syms = set(deriv["symbol"].unique())
    stock   = stock[stock["symbol"].isin(fo_syms)].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ── ATR (per stock, per date, backward-looking) ────────────────────
    atr = _compute_atr(stock, period=LABEL_ATR_PERIOD)

    # ── Forward extremes (computed once, shared by calibration + labels) ─
    O_next, max_H3, min_L3, max_H5, min_L5 = _compute_forward_extremes(stock)

    # Returns relative to T+1 open (fraction)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_up_3 = np.where(O_next > 0, (max_H3 - O_next) / O_next, np.nan)
        ret_dn_3 = np.where(O_next > 0, (O_next - min_L3) / O_next, np.nan)
        ret_up_5 = np.where(O_next > 0, (max_H5 - O_next) / O_next, np.nan)
        ret_dn_5 = np.where(O_next > 0, (O_next - min_L5) / O_next, np.nan)

    # ── Data-driven K calibration (training rows only) ─────────────────
    tr_mask = (stock["date"] <= pd.Timestamp(TRAIN_END)).to_numpy()

    K_3D_up = _calibrate_k(ret_up_3, atr, tr_mask, LABEL_TARGET_RATE_BASE)
    K_3D_dn = _calibrate_k(ret_dn_3, atr, tr_mask, LABEL_TARGET_RATE_BASE)
    K_5D_up = _calibrate_k(ret_up_5, atr, tr_mask, LABEL_TARGET_RATE_BASE)
    K_5D_dn = _calibrate_k(ret_dn_5, atr, tr_mask, LABEL_TARGET_RATE_BASE)
    K_XL_up = _calibrate_k(ret_up_5, atr, tr_mask, LABEL_TARGET_RATE_XL)
    K_XL_dn = _calibrate_k(ret_dn_5, atr, tr_mask, LABEL_TARGET_RATE_XL)

    print(f"[labels] Data-driven K (target_base={LABEL_TARGET_RATE_BASE:.0%}, "
          f"target_xl={LABEL_TARGET_RATE_XL:.0%}):")
    print(f"  K_3D: up={K_3D_up:.3f}  dn={K_3D_dn:.3f}")
    print(f"  K_5D: up={K_5D_up:.3f}  dn={K_5D_dn:.3f}")
    print(f"  K_XL: up={K_XL_up:.3f}  dn={K_XL_dn:.3f}")

    # ── Threshold arrays ──────────────────────────────────────────────
    thresh_3d_up = _make_thresh(K_3D_up, atr)
    thresh_3d_dn = _make_thresh(K_3D_dn, atr)
    thresh_5d_up = _make_thresh(K_5D_up, atr)
    thresh_5d_dn = _make_thresh(K_5D_dn, atr)
    thresh_xl_up = _make_thresh(K_XL_up, atr)
    thresh_xl_dn = _make_thresh(K_XL_dn, atr)

    tr_stats = {}
    for lbl, arr in [("thresh_3d_up", thresh_3d_up[tr_mask]),
                     ("thresh_3d_dn", thresh_3d_dn[tr_mask]),
                     ("thresh_5d_up", thresh_5d_up[tr_mask]),
                     ("thresh_5d_dn", thresh_5d_dn[tr_mask]),
                     ("thresh_xl_up", thresh_xl_up[tr_mask]),
                     ("thresh_xl_dn", thresh_xl_dn[tr_mask])]:
        tr_stats[lbl] = (np.nanpercentile(arr, 5),
                         np.nanmedian(arr),
                         np.nanpercentile(arr, 95))
    print("[labels] Threshold stats (train, p5/median/p95):")
    for lbl, (p5, med, p95) in tr_stats.items():
        print(f"  {lbl}: {p5:.3f}  {med:.3f}  {p95:.3f}")

    # ── Binary labels ─────────────────────────────────────────────────
    stock["up_3"]    = _binary_labels(O_next, max_H3, thresh_3d_up, "up")
    stock["dn_3"]    = _binary_labels(O_next, min_L3, thresh_3d_dn, "dn")
    stock["up_5"]    = _binary_labels(O_next, max_H5, thresh_5d_up, "up")
    stock["dn_5"]    = _binary_labels(O_next, min_L5, thresh_5d_dn, "dn")
    stock["up_5_xl"] = _binary_labels(O_next, max_H5, thresh_xl_up, "up")
    stock["dn_5_xl"] = _binary_labels(O_next, min_L5, thresh_xl_dn, "dn")

    # ── Raw forward returns (peak-to-entry, fraction of T+1 open) ─────
    # These are the actual hold returns if you entered at T+1 open and
    # exited at the intraday peak within the window — used by evaluate.py.
    stock["ret_up_3"] = ret_up_3   # (max_H over T+1..T+3 − open[T+1]) / open[T+1]
    stock["ret_dn_3"] = ret_dn_3   # (open[T+1] − min_L over T+1..T+3) / open[T+1]
    stock["ret_up_5"] = ret_up_5   # (max_H over T+1..T+5 − open[T+1]) / open[T+1]
    stock["ret_dn_5"] = ret_dn_5   # (open[T+1] − min_L over T+1..T+5) / open[T+1]

    # ── Drop rows where forward prices are unavailable ────────────────
    g = stock.groupby("symbol", sort=False, group_keys=False)
    incomplete = (g["open"].transform(lambda s: s.shift(-1)).isna() |
                  g["high"].transform(lambda s: s.shift(-LABEL_HORIZON_5D)).isna())
    label_cols = ("up_3", "dn_3", "up_5", "dn_5", "up_5_xl", "dn_5_xl")
    ret_cols   = ("ret_up_3", "ret_dn_3", "ret_up_5", "ret_dn_5")
    for col in label_cols + ret_cols:
        stock.loc[incomplete, col] = pd.NA

    stock["split"] = _split(stock["date"])
    out = (stock[["date", "symbol", "split",
                  "up_3", "dn_3", "up_5", "dn_5", "up_5_xl", "dn_5_xl",
                  "ret_up_3", "ret_dn_3", "ret_up_5", "ret_dn_5"]]
           .dropna(subset=list(label_cols))
           .reset_index(drop=True))

    out.to_parquet(GOLD_LABELS, index=False)
    _report(out)
    return out


def _report(df: pd.DataFrame) -> None:
    print(f"[labels] saved {len(df):,} rows -> {GOLD_LABELS}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        print(f"  [{split}] rows={len(sub):,}  "
              f"up_3={sub['up_3'].mean():.2%}  dn_3={sub['dn_3'].mean():.2%}  "
              f"up_5={sub['up_5'].mean():.2%}  dn_5={sub['dn_5'].mean():.2%}  "
              f"up_5_xl={sub['up_5_xl'].mean():.2%}  dn_5_xl={sub['dn_5_xl'].mean():.2%}")


def load() -> pd.DataFrame:
    return pd.read_parquet(GOLD_LABELS)


def build_with_rates(
    target_rate_base: float,
    target_rate_xl: float,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build labels with custom target rates and save to out_path.
    Used by experiment.py to build per-trial label files without
    touching the canonical GOLD_LABELS file.

    If out_path is None, saves to GOLD_LABELS (same as build()).
    """
    stock = pd.read_parquet(
        SILVER_TABLES["eod_stock"],
        columns=["date", "symbol", "open", "close", "high", "low"],
    )
    deriv = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"], columns=["date", "symbol"])

    fo_syms = set(deriv["symbol"].unique())
    stock   = stock[stock["symbol"].isin(fo_syms)].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values(["symbol", "date"]).reset_index(drop=True)

    atr = _compute_atr(stock, period=LABEL_ATR_PERIOD)
    O_next, max_H3, min_L3, max_H5, min_L5 = _compute_forward_extremes(stock)

    with np.errstate(invalid="ignore", divide="ignore"):
        ret_up_3 = np.where(O_next > 0, (max_H3 - O_next) / O_next, np.nan)
        ret_dn_3 = np.where(O_next > 0, (O_next - min_L3) / O_next, np.nan)
        ret_up_5 = np.where(O_next > 0, (max_H5 - O_next) / O_next, np.nan)
        ret_dn_5 = np.where(O_next > 0, (O_next - min_L5) / O_next, np.nan)

    tr_mask = (stock["date"] <= pd.Timestamp(TRAIN_END)).to_numpy()

    K_3D_up = _calibrate_k(ret_up_3, atr, tr_mask, target_rate_base)
    K_3D_dn = _calibrate_k(ret_dn_3, atr, tr_mask, target_rate_base)
    K_5D_up = _calibrate_k(ret_up_5, atr, tr_mask, target_rate_base)
    K_5D_dn = _calibrate_k(ret_dn_5, atr, tr_mask, target_rate_base)
    K_XL_up = _calibrate_k(ret_up_5, atr, tr_mask, target_rate_xl)
    K_XL_dn = _calibrate_k(ret_dn_5, atr, tr_mask, target_rate_xl)

    thresh_3d_up = _make_thresh(K_3D_up, atr)
    thresh_3d_dn = _make_thresh(K_3D_dn, atr)
    thresh_5d_up = _make_thresh(K_5D_up, atr)
    thresh_5d_dn = _make_thresh(K_5D_dn, atr)
    thresh_xl_up = _make_thresh(K_XL_up, atr)
    thresh_xl_dn = _make_thresh(K_XL_dn, atr)

    stock["up_3"]    = _binary_labels(O_next, max_H3, thresh_3d_up, "up")
    stock["dn_3"]    = _binary_labels(O_next, min_L3, thresh_3d_dn, "dn")
    stock["up_5"]    = _binary_labels(O_next, max_H5, thresh_5d_up, "up")
    stock["dn_5"]    = _binary_labels(O_next, min_L5, thresh_5d_dn, "dn")
    stock["up_5_xl"] = _binary_labels(O_next, max_H5, thresh_xl_up, "up")
    stock["dn_5_xl"] = _binary_labels(O_next, min_L5, thresh_xl_dn, "dn")

    stock["ret_up_3"] = ret_up_3
    stock["ret_dn_3"] = ret_dn_3
    stock["ret_up_5"] = ret_up_5
    stock["ret_dn_5"] = ret_dn_5

    g = stock.groupby("symbol", sort=False, group_keys=False)
    incomplete = (g["open"].transform(lambda s: s.shift(-1)).isna() |
                  g["high"].transform(lambda s: s.shift(-LABEL_HORIZON_5D)).isna())
    label_cols = ("up_3", "dn_3", "up_5", "dn_5", "up_5_xl", "dn_5_xl")
    ret_cols   = ("ret_up_3", "ret_dn_3", "ret_up_5", "ret_dn_5")
    for col in label_cols + ret_cols:
        stock.loc[incomplete, col] = pd.NA

    stock["split"] = _split(stock["date"])
    out = (stock[["date", "symbol", "split",
                  "up_3", "dn_3", "up_5", "dn_5", "up_5_xl", "dn_5_xl",
                  "ret_up_3", "ret_dn_3", "ret_up_5", "ret_dn_5"]]
           .dropna(subset=list(label_cols))
           .reset_index(drop=True))

    dest = Path(out_path) if out_path else GOLD_LABELS
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"[labels] saved {len(out):,} rows → {dest}  "
          f"(base={target_rate_base:.0%}  xl={target_rate_xl:.0%})")
    return out


def build_tiered() -> pd.DataFrame:
    """
    Build tiered labels (v5) for the two-model architecture.

    Label logic
    -----------
    Liquid tier  (first LIQUID_TIER_SIZE symbols in ESN1.0/Universe):
        up_liq = 1  if  max(high[T+1..T+5]) >= open[T+1] × (1 + LIQUID_THRESHOLD)
        dn_liq = 1  if  min(low [T+1..T+5]) <= open[T+1] × (1 - LIQUID_THRESHOLD)

    Rest tier (all other F&O stocks):
        up_rest = 1  if  max(high[T+1..T+5]) >= open[T+1] × (1 + REST_THRESHOLD)
        dn_rest = 1  if  min(low [T+1..T+5]) <= open[T+1] × (1 - REST_THRESHOLD)

    Overlay labels (both tiers — used by bad-close overlay models):
        bad_up = 1  if  (close[T+5] - open[T+1]) / open[T+1]  < -BAD_CLOSE_THRESH
        bad_dn = 1  if  (close[T+5] - open[T+1]) / open[T+1]  >  BAD_CLOSE_THRESH

    Output: gold/labels_tiered.parquet
        date, symbol, split, tier,
        up_liq, dn_liq,       (NaN for rest-tier rows)
        up_rest, dn_rest,     (NaN for liquid-tier rows)
        bad_up, bad_dn,       (non-NaN for all rows with valid forward data)
        fwd_close_ret         (close[T+5] return vs T+1 open — used by evaluate)
    """
    from .universe import load_predict_universe

    stock = pd.read_parquet(
        SILVER_TABLES["eod_stock"],
        columns=["date", "symbol", "open", "close", "high", "low"],
    )
    deriv = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"], columns=["date", "symbol"])

    fo_syms = set(deriv["symbol"].unique())
    stock   = stock[stock["symbol"].isin(fo_syms)].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ── Identify liquid tier and filter to Universe stocks ────────────────────
    try:
        all_universe_syms, liquid_set = load_predict_universe()
        universe_set = set(all_universe_syms)
    except FileNotFoundError:
        print(f"[labels_tiered] WARNING: Universe file not found — "
              f"using empty liquid_set (all F&O stocks treated as rest tier)")
        liquid_set   = set()
        universe_set = fo_syms   # fall back to all F&O stocks

    # Restrict training data to Universe stocks only (liquid-30 + rest-35).
    # This keeps the REST model trained exclusively on the curated high-quality
    # stocks — removes ~280 low-quality F&O stocks from training.
    stock = stock[stock["symbol"].isin(universe_set)].copy()
    stock = stock.reset_index(drop=True)

    rest_syms = universe_set - liquid_set
    print(f"[labels_tiered] liquid tier: {len(liquid_set)} symbols  "
          f"({LIQUID_THRESHOLD:.0%} threshold)")
    print(f"[labels_tiered] rest   tier: {len(rest_syms)} symbols  "
          f"({REST_THRESHOLD:.0%} threshold)")
    print(f"[labels_tiered] total universe: {len(universe_set)} symbols  "
          f"(filtered from {len(fo_syms)} F&O stocks)")
    stock["tier"] = np.where(stock["symbol"].isin(liquid_set), "liquid", "rest")

    # ── Forward extremes (reuse existing helper) ───────────────────────────────
    O_next, _max_H3, _min_L3, max_H5, min_L5 = _compute_forward_extremes(stock)

    # Forward close return: (close[T+5] − open[T+1]) / open[T+1]
    g = stock.groupby("symbol", sort=False, group_keys=False)
    close_T5 = g["close"].transform(lambda s: s.shift(-LABEL_HORIZON_5D)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd_close_ret = np.where(O_next > 0,
                                 (close_T5 - O_next) / O_next,
                                 np.nan)

    # ── Tiered binary labels ──────────────────────────────────────────────────
    liq_mask  = (stock["tier"] == "liquid").to_numpy()
    rest_mask = ~liq_mask

    # Liquid: 4% threshold
    up_liq_arr = np.where(
        liq_mask,
        _binary_labels(O_next, max_H5, np.full(len(stock), LIQUID_THRESHOLD), "up"),
        np.nan,
    )
    dn_liq_arr = np.where(
        liq_mask,
        _binary_labels(O_next, min_L5, np.full(len(stock), LIQUID_THRESHOLD), "dn"),
        np.nan,
    )

    # Rest: 8% threshold
    up_rest_arr = np.where(
        rest_mask,
        _binary_labels(O_next, max_H5, np.full(len(stock), REST_THRESHOLD), "up"),
        np.nan,
    )
    dn_rest_arr = np.where(
        rest_mask,
        _binary_labels(O_next, min_L5, np.full(len(stock), REST_THRESHOLD), "dn"),
        np.nan,
    )

    # ── Forward returns (fraction of T+1 open) — needed for clean labels ────────
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_up_5 = np.where(O_next > 0, (max_H5 - O_next) / O_next, np.nan)
        ret_dn_5 = np.where(O_next > 0, (O_next - min_L5) / O_next, np.nan)

    # ── Clean directional labels (tier-asymmetric thresholds) ─────────────────
    # Liquid-30 (4% threshold):
    #   clean_up_5_liq = 1  if  ret_up_5 > 4%  AND  ret_dn_5 < 0.91%
    #   clean_dn_5_liq = 1  if  ret_dn_5 > 4%  AND  ret_up_5 < 0.91%
    # Rest-35 (7% threshold):
    #   clean_up_5_rest = 1  if  ret_up_5 > 7%  AND  ret_dn_5 < 0.91%
    #   clean_dn_5_rest = 1  if  ret_dn_5 > 7%  AND  ret_up_5 < 0.91%
    # NaN outside the relevant tier.
    clean_up_liq_arr = np.where(
        liq_mask,
        ((ret_up_5 > CLEAN_UP_THRESH_LIQ) & (ret_dn_5 < CLEAN_NOISE_THRESH)).astype(np.int8),
        np.nan,
    )
    clean_dn_liq_arr = np.where(
        liq_mask,
        ((ret_dn_5 > CLEAN_DN_THRESH_LIQ) & (ret_up_5 < CLEAN_NOISE_THRESH)).astype(np.int8),
        np.nan,
    )
    clean_up_rest_arr = np.where(
        rest_mask,
        ((ret_up_5 > CLEAN_UP_THRESH_REST) & (ret_dn_5 < CLEAN_NOISE_THRESH)).astype(np.int8),
        np.nan,
    )
    clean_dn_rest_arr = np.where(
        rest_mask,
        ((ret_dn_5 > CLEAN_DN_THRESH_REST) & (ret_up_5 < CLEAN_NOISE_THRESH)).astype(np.int8),
        np.nan,
    )

    # ── Overlay labels (bad close — direction-agnostic setup risk) ────────────
    ref_valid  = ~np.isnan(O_next) & ~np.isnan(fwd_close_ret)
    bad_up_arr = np.where(ref_valid, (fwd_close_ret < -BAD_CLOSE_THRESH).astype(np.int8), np.nan)
    bad_dn_arr = np.where(ref_valid, (fwd_close_ret >  BAD_CLOSE_THRESH).astype(np.int8), np.nan)

    # ── Assemble output ───────────────────────────────────────────────────────
    stock["up_liq"]          = up_liq_arr.astype(object)
    stock["dn_liq"]          = dn_liq_arr.astype(object)
    stock["up_rest"]         = up_rest_arr.astype(object)
    stock["dn_rest"]         = dn_rest_arr.astype(object)
    stock["bad_up"]          = bad_up_arr.astype(object)
    stock["bad_dn"]          = bad_dn_arr.astype(object)
    stock["clean_up_5_liq"]  = clean_up_liq_arr.astype(object)
    stock["clean_dn_5_liq"]  = clean_dn_liq_arr.astype(object)
    stock["clean_up_5_rest"] = clean_up_rest_arr.astype(object)
    stock["clean_dn_5_rest"] = clean_dn_rest_arr.astype(object)
    stock["fwd_close_ret"]   = fwd_close_ret

    # Drop rows missing forward prices (last LABEL_HORIZON_5D rows per symbol)
    incomplete = (
        g["open"].transform(lambda s: s.shift(-1)).isna() |
        g["high"].transform(lambda s: s.shift(-LABEL_HORIZON_5D)).isna()
    )
    tiered_label_cols = ("up_liq", "dn_liq", "up_rest", "dn_rest", "bad_up", "bad_dn",
                         "clean_up_5_liq",  "clean_dn_5_liq",
                         "clean_up_5_rest", "clean_dn_5_rest")
    for col in tiered_label_cols + ("fwd_close_ret",):
        stock.loc[incomplete, col] = pd.NA

    stock["split"] = _split(stock["date"])

    # Convert object columns back to float/Int8 for clean parquet storage
    for col in tiered_label_cols:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    out = (stock[["date", "symbol", "split", "tier",
                  "up_liq", "dn_liq", "up_rest", "dn_rest",
                  "bad_up", "bad_dn",
                  "clean_up_5_liq",  "clean_dn_5_liq",
                  "clean_up_5_rest", "clean_dn_5_rest",
                  "fwd_close_ret"]]
           .dropna(subset=["bad_up"])   # keep rows with valid forward close
           .reset_index(drop=True))

    GOLD_LABELS_TIERED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GOLD_LABELS_TIERED, index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"[labels_tiered] saved {len(out):,} rows → {GOLD_LABELS_TIERED}")
    for split in ["train", "val", "test"]:
        sub = out[out["split"] == split]
        if sub.empty:
            continue
        liq  = sub[sub["tier"] == "liquid"]
        rest = sub[sub["tier"] == "rest"]
        print(f"  [{split}] rows={len(sub):,}  "
              f"liquid={len(liq):,}  rest={len(rest):,}")
        if len(liq):
            print(f"    liquid → up_liq={liq['up_liq'].mean():.2%}  "
                  f"dn_liq={liq['dn_liq'].mean():.2%}  "
                  f"bad_up={liq['bad_up'].mean():.2%}  "
                  f"bad_dn={liq['bad_dn'].mean():.2%}  "
                  f"clean_up={liq['clean_up_5_liq'].mean():.2%}  "
                  f"clean_dn={liq['clean_dn_5_liq'].mean():.2%}")
        if len(rest):
            print(f"    rest   → up_rest={rest['up_rest'].mean():.2%}  "
                  f"dn_rest={rest['dn_rest'].mean():.2%}  "
                  f"bad_up={rest['bad_up'].mean():.2%}  "
                  f"bad_dn={rest['bad_dn'].mean():.2%}  "
                  f"clean_up_7%={rest['clean_up_5_rest'].mean():.2%}  "
                  f"clean_dn_7%={rest['clean_dn_5_rest'].mean():.2%}")
    return out


def load_tiered() -> pd.DataFrame:
    """Load the tiered labels parquet built by build_tiered()."""
    return pd.read_parquet(GOLD_LABELS_TIERED)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tiered":
        build_tiered()
    else:
        build()
