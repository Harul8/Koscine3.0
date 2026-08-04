"""
Quick spike: can we predict "clean" directional days with a single LGBM?

Targets:
  clean_up_5 = (ret_up_5 > 4%) AND (ret_dn_5 < 0.91%)
  clean_dn_5 = (ret_dn_5 > 4%) AND (ret_up_5 < 0.91%)

Scope:
  - Liquid-30 stocks only (top-30 of Universe file)
  - Same feature set as up_liq / dn_liq (SHARED + UP / SHARED + DN)
  - One LGBM model per target, no ensemble, no overlay
  - Train on data ≤ TRAIN_END (default 2024-12-31)
  - Use val (H1 2025) for early stopping
  - Report precision @ top-K% on test (H2 2025) — the true holdout

If precision @ top-2% on test ≥ 55-60%, the relabel is learnable and we
proceed to Layer 2 (features) + Layer 3 (consensus gating).

Run:
    python -m pipeline.clean_spike
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import (
    GOLD_FEATURES, GOLD_LABELS,
    SILVER_TABLES,
    TARGET_FEATURE_COLS,
    LGBM_BASE_PARAMS, LGBM_TIERED_TARGET_PARAMS,
)
from .universe import load_predict_universe


# ── Layer 2: path-asymmetry features ───────────────────────────────────────────
# All backward-looking — computed at end of T using bars [T-N+1 .. T].
# Used to predict the forward window T+1..T+5.

def _add_path_features(df: pd.DataFrame, liquid_30: list[str]
                       ) -> tuple[pd.DataFrame, list[str]]:
    """Compute one-sided path features from silver eod_stock; merge into df."""
    s = pd.read_parquet(
        SILVER_TABLES["eod_stock"],
        columns=["date", "symbol", "open", "high", "low", "close",
                 "volume", "deliv_pct"],
    )
    s = s[s["symbol"].isin(liquid_30)].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = s.groupby("symbol", sort=False, group_keys=False)
    prev_close = g["close"].shift(1)

    body       = s["close"] - s["open"]
    abs_body   = body.abs()
    up_body    = body.clip(lower=0)
    dn_body    = (-body).clip(lower=0)
    upper_wick = s["high"] - np.maximum(s["open"], s["close"])
    lower_wick = np.minimum(s["open"], s["close"]) - s["low"]
    rng        = s["high"] - s["low"]
    abs_ret_1d = ((s["close"] - prev_close) / prev_close).abs()
    tr = pd.concat([rng,
                    (s["high"] - prev_close).abs(),
                    (s["low"]  - prev_close).abs()], axis=1).max(axis=1)

    s["_up_body"]    = up_body
    s["_dn_body"]    = dn_body
    s["_abs_body"]   = abs_body
    s["_upper_wick"] = upper_wick
    s["_lower_wick"] = lower_wick
    s["_range"]      = rng
    s["_abs_ret_1d"] = abs_ret_1d
    s["_tr"]         = tr
    s["_is_green"]   = ((s["close"] > s["open"]) & (s["close"] > prev_close)).astype(int)
    s["_is_red"]     = ((s["close"] < s["open"]) & (s["close"] < prev_close)).astype(int)

    g = s.groupby("symbol", sort=False, group_keys=False)
    def roll_sum(col: str, n: int, mp: int) -> pd.Series:
        return g[col].transform(lambda x: x.rolling(n, min_periods=mp).sum())
    def roll_mean(col: str, n: int, mp: int) -> pd.Series:
        return g[col].transform(lambda x: x.rolling(n, min_periods=mp).mean())

    # 5d rolling sums
    up_body_5d   = roll_sum("_up_body",    5, 3)
    dn_body_5d   = roll_sum("_dn_body",    5, 3)
    abs_body_5d  = roll_sum("_abs_body",   5, 3)
    upper_w_5d   = roll_sum("_upper_wick", 5, 3)
    lower_w_5d   = roll_sum("_lower_wick", 5, 3)
    range_5d     = roll_sum("_range",      5, 3)
    abs_ret_5d   = roll_sum("_abs_ret_1d", 5, 3)
    atr20        = roll_mean("_tr",       20, 10)
    close_lag5   = g["close"].shift(5)
    net_ret_5d   = (s["close"] - close_lag5) / close_lag5

    eps = 1e-9
    # Body asymmetry — fraction of recent body that was bullish vs bearish
    s["up_pressure_5d"] = up_body_5d / (abs_body_5d + eps)
    s["dn_pressure_5d"] = dn_body_5d / (abs_body_5d + eps)
    # Wick asymmetry — sellers on top vs buyers on bottom
    s["upper_wick_pressure_5d"] = upper_w_5d / (range_5d + eps)
    s["lower_wick_pressure_5d"] = lower_w_5d / (range_5d + eps)
    # Directional purity — net move / total path (1=monotonic, 0=pure noise)
    s["directional_purity_5d"]        = net_ret_5d.abs() / (abs_ret_5d + eps)
    s["directional_purity_5d_signed"] = net_ret_5d       / (abs_ret_5d + eps)
    # Range purity — net move per unit of intraday range
    s["range_purity_5d_signed"] = (s["close"] - close_lag5) / (range_5d + eps)
    # Whipsaw count: days where range > 1.5×ATR20
    is_whip = (s["_range"] > 1.5 * atr20).astype(int)
    s["whipsaw_count_5d"] = is_whip.groupby(s["symbol"]).transform(
        lambda x: x.rolling(5, min_periods=3).sum())
    # Clean run counts
    s["clean_runup_5d"] = roll_sum("_is_green", 5, 3)
    s["clean_rundn_5d"] = roll_sum("_is_red",   5, 3)

    # 10d versions for longer regime context
    abs_body_10d  = roll_sum("_abs_body",   10, 5)
    up_body_10d   = roll_sum("_up_body",    10, 5)
    abs_ret_10d   = roll_sum("_abs_ret_1d", 10, 5)
    range_10d     = roll_sum("_range",      10, 5)
    close_lag10   = g["close"].shift(10)
    net_ret_10d   = (s["close"] - close_lag10) / close_lag10
    s["up_pressure_10d"]               = up_body_10d / (abs_body_10d + eps)
    s["directional_purity_10d_signed"] = net_ret_10d / (abs_ret_10d  + eps)
    s["range_purity_10d_signed"]       = (s["close"] - close_lag10) / (range_10d + eps)

    # ── Volume asymmetry — fraction of recent volume on green vs red days ────
    s["_vol_green"] = s["volume"] * s["_is_green"]
    s["_vol_red"]   = s["volume"] * s["_is_red"]
    vol_green_5d = roll_sum("_vol_green", 5, 3)
    vol_red_5d   = roll_sum("_vol_red",   5, 3)
    vol_total_5d = roll_sum("volume",     5, 3)
    s["vol_up_ratio_5d"] = vol_green_5d / (vol_total_5d + eps)
    s["vol_dn_ratio_5d"] = vol_red_5d   / (vol_total_5d + eps)

    # ── Gap persistence — gap up that held (close > open after a gap up) ────
    gap_up = ((s["open"] - prev_close) / prev_close > 0.005).astype(int)
    gap_dn = ((prev_close - s["open"]) / prev_close > 0.005).astype(int)
    gap_up_held = (gap_up & (s["close"] > s["open"])).astype(int)
    gap_dn_held = (gap_dn & (s["close"] < s["open"])).astype(int)
    s["_gap_up_held"] = gap_up_held
    s["_gap_dn_held"] = gap_dn_held
    s["gap_up_persistence_5d"] = roll_sum("_gap_up_held", 5, 3)
    s["gap_dn_persistence_5d"] = roll_sum("_gap_dn_held", 5, 3)

    # ── Clear-air streak — close > prior 5d max-high (no overhead supply) ─
    g = s.groupby("symbol", sort=False, group_keys=False)
    prior_5d_high = g["high"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).max())
    prior_5d_low  = g["low" ].transform(lambda x: x.shift(1).rolling(5, min_periods=3).min())
    s["clear_air_up"] = (s["close"] > prior_5d_high).astype(int)
    s["clear_air_dn"] = (s["close"] < prior_5d_low ).astype(int)
    s["clear_air_up_streak_5d"] = roll_sum("clear_air_up", 5, 3)
    s["clear_air_dn_streak_5d"] = roll_sum("clear_air_dn", 5, 3)

    new_feats = [
        "up_pressure_5d", "dn_pressure_5d",
        "upper_wick_pressure_5d", "lower_wick_pressure_5d",
        "directional_purity_5d", "directional_purity_5d_signed",
        "range_purity_5d_signed",
        "whipsaw_count_5d",
        "clean_runup_5d", "clean_rundn_5d",
        "up_pressure_10d",
        "directional_purity_10d_signed", "range_purity_10d_signed",
        # Volume asymmetry
        "vol_up_ratio_5d", "vol_dn_ratio_5d",
        # Gap persistence
        "gap_up_persistence_5d", "gap_dn_persistence_5d",
        # Clear-air structure
        "clear_air_up_streak_5d", "clear_air_dn_streak_5d",
    ]
    out = s[["date", "symbol"] + new_feats]
    df  = df.merge(out, on=["date", "symbol"], how="left")

    # ── NIFTY regime path features (broadcast to all symbols on date) ─────────
    n = pd.read_parquet(
        SILVER_TABLES["indices"],
        columns=["date", "index_name", "open", "high", "low", "close"],
    )
    n = n[n["index_name"] == "Nifty 50"].copy()
    n["date"] = pd.to_datetime(n["date"])
    n = n.sort_values("date").reset_index(drop=True)
    n_prev_close = n["close"].shift(1)
    n_body       = n["close"] - n["open"]
    n_up_body    = n_body.clip(lower=0)
    n_abs_body   = n_body.abs()
    n_range      = n["high"] - n["low"]
    n_abs_ret    = ((n["close"] - n_prev_close) / n_prev_close).abs()

    n["_up_body"]    = n_up_body
    n["_abs_body"]   = n_abs_body
    n["_range"]      = n_range
    n["_abs_ret_1d"] = n_abs_ret

    n_up_body_5d  = n["_up_body"].rolling(5,  min_periods=3).sum()
    n_abs_body_5d = n["_abs_body"].rolling(5,  min_periods=3).sum()
    n_abs_ret_5d  = n["_abs_ret_1d"].rolling(5,  min_periods=3).sum()
    n_range_5d    = n["_range"].rolling(5,  min_periods=3).sum()
    n_close_lag5  = n["close"].shift(5)
    n_net_5d      = (n["close"] - n_close_lag5) / n_close_lag5

    n["nifty_up_pressure_5d"]               = n_up_body_5d / (n_abs_body_5d + eps)
    n["nifty_directional_purity_5d_signed"] = n_net_5d     / (n_abs_ret_5d  + eps)
    n["nifty_range_purity_5d_signed"]       = (n["close"] - n_close_lag5) / (n_range_5d + eps)
    n["nifty_directional_purity_5d"]        = n_net_5d.abs() / (n_abs_ret_5d + eps)

    nifty_feats = [
        "nifty_up_pressure_5d",
        "nifty_directional_purity_5d_signed",
        "nifty_range_purity_5d_signed",
        "nifty_directional_purity_5d",
    ]
    df = df.merge(n[["date"] + nifty_feats], on="date", how="left")

    new_feats = new_feats + nifty_feats
    print(f"[spike] Layer-2 features added: {len(new_feats)} cols "
          f"({len(nifty_feats)} NIFTY regime)")
    return df, new_feats


# Clean-day thresholds (same as the bull/bear analysis we ran)
UP_THRESH        = 0.04    # forward upside > 4%
DN_THRESH        = 0.04    # forward downside > 4%
NOISE_THRESH     = 0.0091  # opposite-side noise < 0.91%  (STRICT)
NOISE_LOOSE      = 0.015   # opposite-side noise < 1.50%  (LOOSE)

EVAL_PCTS = [1, 2, 3, 5, 10, 20]


def _load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load features + labels, filter to liquid-30, build clean targets,
    augment with Layer-2 path features. Returns (df, new_path_feature_cols)."""
    _all_syms, liquid_set = load_predict_universe()
    liquid_30 = sorted(liquid_set)

    feats  = pd.read_parquet(GOLD_FEATURES)
    labels = pd.read_parquet(GOLD_LABELS,
                             columns=["date", "symbol", "split",
                                      "ret_up_5", "ret_dn_5"])

    feats["date"]  = pd.to_datetime(feats["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    feats  = feats[feats["symbol"].isin(liquid_30)]
    labels = labels[labels["symbol"].isin(liquid_30)]

    df = feats.merge(labels, on=["date", "symbol"], how="inner")
    df = df.dropna(subset=["ret_up_5", "ret_dn_5"])

    df["clean_up_5"] = (
        (df["ret_up_5"] > UP_THRESH) & (df["ret_dn_5"] < NOISE_THRESH)
    ).astype(int)
    df["clean_dn_5"] = (
        (df["ret_dn_5"] > DN_THRESH) & (df["ret_up_5"] < NOISE_THRESH)
    ).astype(int)
    # Loose variant: up>4% AND dn<1.5% (vs strict <0.91%)
    df["clean_up_5_loose"] = (
        (df["ret_up_5"] > UP_THRESH) & (df["ret_dn_5"] < NOISE_LOOSE)
    ).astype(int)
    df["clean_dn_5_loose"] = (
        (df["ret_dn_5"] > DN_THRESH) & (df["ret_up_5"] < NOISE_LOOSE)
    ).astype(int)

    print(f"[spike] liquid universe: {len(liquid_30)} stocks")
    print(f"[spike] joined rows: {len(df):,}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        print(f"  [{split}] n={len(sub):,}  "
              f"clean_up_5={sub['clean_up_5'].mean():.2%}  "
              f"clean_dn_5={sub['clean_dn_5'].mean():.2%}  "
              f"({sub['date'].min().date()} -> {sub['date'].max().date()})")

    df, new_feats = _add_path_features(df, liquid_30)
    return df, new_feats


def _train_one(df: pd.DataFrame, target: str, feat_cols: list[str],
               n_seeds: int = 8, saddle_filter: bool = False) -> None:
    """Train n_seeds LGBMs, average predictions, report precision@top-K% on test.

    saddle_filter : if True, drop any seed whose best_iter < 50 (model that
                    bailed out before learning anything). Helpful for the bull
                    target which is overfit-prone; harmful for the bear target
                    which benefits from weak-learner averaging.
    """
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"\n=== {target} ===  ({len(feat_cols)} features, {n_seeds} seeds, "
          f"saddle_filter={saddle_filter})")

    def _slice(split: str):
        m = df["split"] == split
        X = df.loc[m, feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        y = df.loc[m, target].astype(int).values
        return X, y

    Xtr, ytr = _slice("train")
    Xv,  yv  = _slice("val")
    Xte, yte = _slice("test")

    if len(Xte) == 0 or yte.sum() == 0:
        print(f"  [{target}] no test data — skipping")
        return

    # Custom params tuned for clean-direction targets.
    # Both targets have ~12% positive rate. AUC + scale_pos_weight makes the
    # model saddle out at iter 2-3; AP without rebalance trains cleanly.
    base_p = dict(
        objective         = "binary",
        metric            = "average_precision",
        n_estimators      = 4000,
        learning_rate     = 0.02,
        num_leaves        = 63,
        min_child_samples = 60,
        reg_lambda        = 1.5,
        reg_alpha         = 0.3,
        feature_fraction  = 0.75,
        bagging_fraction  = 0.80,
        bagging_freq      = 5,
        n_jobs            = -1,
        verbose           = -1,
    )

    raw_models: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_train = n_seeds + 3 if saddle_filter else n_seeds   # over-sample for filter
    for s in range(n_train):
        p = dict(base_p, random_state=42 + s)
        m = lgb.LGBMClassifier(**p)
        m.fit(
            Xtr, ytr,
            eval_set=[(Xv, yv)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        raw_models.append((
            m.best_iteration_,
            m.predict_proba(Xv )[:, 1],
            m.predict_proba(Xte)[:, 1],
        ))
    if saddle_filter:
        kept = [r for r in raw_models if r[0] >= 50]
        if len(kept) < 2:
            kept = raw_models
    else:
        kept = raw_models
    best_iters = [r[0] for r in kept]
    p_val_sum  = np.sum([r[1] for r in kept], axis=0)
    p_test_sum = np.sum([r[2] for r in kept], axis=0)
    p_val  = p_val_sum  / len(kept)
    p_test = p_test_sum / len(kept)
    if saddle_filter:
        print(f"  seeds used: {len(kept)}/{len(raw_models)}  "
              f"(dropped {len(raw_models) - len(kept)} saddled)")
    # placeholder model attr for the print line below
    class _M:
        pass
    model = _M()
    model.best_iteration_ = best_iters

    base_val  = float(yv.mean())
    base_test = float(yte.mean())
    print(f"  base rate    val={base_val:.2%}   test={base_test:.2%}")
    print(f"  best_iter    {model.best_iteration_}")

    print(f"  {'top%':>5}  {'n_val':>5}  {'p_val':>7}  {'lift_v':>6}  "
          f"{'n_test':>6}  {'p_test':>7}  {'lift_t':>6}")
    for pct in EVAL_PCTS:
        # val
        k_v = max(1, int(len(p_val) * pct / 100))
        idx_v = np.argsort(p_val)[::-1][:k_v]
        prec_v = float(yv[idx_v].mean())
        lift_v = prec_v / base_val if base_val > 0 else float("nan")
        # test
        k_t = max(1, int(len(p_test) * pct / 100))
        idx_t = np.argsort(p_test)[::-1][:k_t]
        prec_t = float(yte[idx_t].mean())
        lift_t = prec_t / base_test if base_test > 0 else float("nan")
        print(f"  {pct:>4}%  {k_v:>5d}  {prec_v:>6.1%}  {lift_v:>5.2f}x  "
              f"{k_t:>6d}  {prec_t:>6.1%}  {lift_t:>5.2f}x")


def _train_and_return(df: pd.DataFrame, target: str, feat_cols: list[str],
                      n_seeds: int = 8, saddle_filter: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same as _train_one but returns (p_val, p_test, yv, yte) for Layer-3 use."""
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"\n=== {target} ===  ({len(feat_cols)} features, {n_seeds} seeds, "
          f"saddle_filter={saddle_filter})")

    def _slice(split: str):
        m = df["split"] == split
        X = df.loc[m, feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        y = df.loc[m, target].astype(int).values
        return X, y

    Xtr, ytr = _slice("train")
    Xv,  yv  = _slice("val")
    Xte, yte = _slice("test")

    base_p = dict(
        objective="binary", metric="average_precision",
        n_estimators=4000, learning_rate=0.02, num_leaves=63,
        min_child_samples=60, reg_lambda=1.5, reg_alpha=0.3,
        feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
        n_jobs=-1, verbose=-1,
    )

    raw_models = []
    n_train = n_seeds + 3 if saddle_filter else n_seeds
    for s in range(n_train):
        p = dict(base_p, random_state=42 + s)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False),
                         lgb.log_evaluation(period=0)])
        raw_models.append((m.best_iteration_,
                           m.predict_proba(Xv )[:, 1],
                           m.predict_proba(Xte)[:, 1]))
    kept = ([r for r in raw_models if r[0] >= 50] if saddle_filter else raw_models)
    if len(kept) < 2:
        kept = raw_models

    p_val  = np.mean([r[1] for r in kept], axis=0)
    p_test = np.mean([r[2] for r in kept], axis=0)
    return p_val, p_test, yv, yte


def _report_topk(name: str, p: np.ndarray, y: np.ndarray) -> None:
    base = float(y.mean())
    print(f"  [{name}] base={base:.2%}")
    print(f"  {'top%':>5}  {'n':>5}  {'prec':>7}  {'lift':>6}")
    for pct in EVAL_PCTS:
        k = max(1, int(len(p) * pct / 100))
        idx = np.argsort(p)[::-1][:k]
        prec = float(y[idx].mean())
        lift = prec / base if base > 0 else float("nan")
        print(f"  {pct:>4}%  {k:>5d}  {prec:>6.1%}  {lift:>5.2f}x")


def _within_day_rank(series: pd.Series, dates: pd.Series) -> np.ndarray:
    """Percentile rank [0,1] within each date group."""
    df = pd.DataFrame({"v": series.values, "d": dates.values})
    df["r"] = df.groupby("d")["v"].rank(pct=True)
    return df["r"].to_numpy()


def _eval_one_gate(label: str, mask: np.ndarray, y: np.ndarray,
                   base: float) -> tuple[int, int, float, float]:
    n = int(mask.sum())
    pos = int(y[mask].sum())
    prec = pos / max(n, 1)
    lift = prec / base if base > 0 else float("nan")
    return n, pos, prec, lift


def _layer3_full(
    df: pd.DataFrame,
    p_val_up: np.ndarray, p_test_up: np.ndarray,
    p_val_dn: np.ndarray, p_test_dn: np.ndarray,
    yv_up: np.ndarray, yte_up: np.ndarray,
    yv_dn: np.ndarray, yte_dn: np.ndarray,
    target_label: str = "STRICT",
) -> None:
    """
    Full Layer-3 evaluation: persistence, differential, z-score, meta-blend,
    val-tuned thresholds applied to test.

    target_label : "STRICT" or "LOOSE" — just for printing.
    """
    from .models.ensemble import load_prod_models, predict_ensemble
    from .config import OUT_ROOT
    from pathlib import Path

    PROD_DIR = OUT_ROOT / "models" / "prod"
    print(f"\n[layer3 {target_label}] loading prod models...")
    models = load_prod_models(PROD_DIR)

    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values
    X_val   = df.loc[val_mask].reset_index(drop=True)
    X_test  = df.loc[test_mask].reset_index(drop=True)

    print(f"[layer3 {target_label}] scoring val+test with prod models...")
    val_prod  = predict_ensemble(models, X_val)
    test_prod = predict_ensemble(models, X_test)

    def _arr(d: dict, key: str, n: int) -> np.ndarray:
        return np.asarray(d.get(key, np.zeros(n)), dtype=float)

    # ── Build feature panels for both val and test ────────────────────────────
    panels: dict[str, pd.DataFrame] = {}
    for split_label, X_, p_up, p_dn, prod, y_up, y_dn in [
        ("val",  X_val,  p_val_up,  p_val_dn,  val_prod,  yv_up,  yv_dn),
        ("test", X_test, p_test_up, p_test_dn, test_prod, yte_up, yte_dn),
    ]:
        n = len(X_)
        panel = pd.DataFrame({
            "date":   X_["date"].values,
            "symbol": X_["symbol"].values,
            "p_up":   p_up,
            "p_dn":   p_dn,
            "up_liq":  _arr(prod, "up_liq",        n),
            "dn_liq":  _arr(prod, "dn_liq",        n),
            "bad_up":  _arr(prod, "bad_up_liquid", n),
            "bad_dn":  _arr(prod, "bad_dn_liquid", n),
            "nifty_dp": X_["nifty_directional_purity_5d_signed"].fillna(0.0).values,
            "y_up":   y_up,
            "y_dn":   y_dn,
        })
        panel["diff_up"] = panel["p_up"] - panel["p_dn"]
        panel["diff_dn"] = panel["p_dn"] - panel["p_up"]
        panels[split_label] = panel

    # ── Lag features (per symbol, across val+test concatenated) ──────────────
    combined = pd.concat([panels["val"], panels["test"]], ignore_index=True)
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = combined.groupby("symbol", sort=False, group_keys=False)
    combined["p_up_lag1"] = g["p_up"].shift(1)
    combined["p_dn_lag1"] = g["p_dn"].shift(1)
    combined["p_up_lag2"] = g["p_up"].shift(2)
    combined["p_dn_lag2"] = g["p_dn"].shift(2)
    combined["p_up_2d_max"] = np.fmax(combined["p_up"], combined["p_up_lag1"])
    combined["p_dn_2d_max"] = np.fmax(combined["p_dn"], combined["p_dn_lag1"])
    combined["p_up_2d_min"] = np.fmin(combined["p_up"], combined["p_up_lag1"])
    combined["p_dn_2d_min"] = np.fmin(combined["p_dn"], combined["p_dn_lag1"])

    # ── Per-stock z-score (using val distribution as reference) ──────────────
    val_only = combined.iloc[:len(panels["val"])] if False else combined[
        combined["date"].isin(panels["val"]["date"].unique())].copy()
    sym_stats_up = val_only.groupby("symbol")["p_up"].agg(["mean", "std"]).reset_index()
    sym_stats_dn = val_only.groupby("symbol")["p_dn"].agg(["mean", "std"]).reset_index()
    sym_stats_up.columns = ["symbol", "up_mean", "up_std"]
    sym_stats_dn.columns = ["symbol", "dn_mean", "dn_std"]
    combined = combined.merge(sym_stats_up, on="symbol", how="left")
    combined = combined.merge(sym_stats_dn, on="symbol", how="left")
    combined["p_up_z"] = (combined["p_up"] - combined["up_mean"]) / (combined["up_std"] + 1e-9)
    combined["p_dn_z"] = (combined["p_dn"] - combined["dn_mean"]) / (combined["dn_std"] + 1e-9)

    # Split back to val/test
    n_val = len(panels["val"])
    # combined is sorted by symbol/date — need to re-extract by date set
    val_dates  = set(pd.to_datetime(panels["val"]["date"]).unique())
    test_dates = set(pd.to_datetime(panels["test"]["date"]).unique())
    combined["_dt"] = pd.to_datetime(combined["date"])
    val_panel  = combined[combined["_dt"].isin(val_dates)].copy().reset_index(drop=True)
    test_panel = combined[combined["_dt"].isin(test_dates)].copy().reset_index(drop=True)

    base_up = float(test_panel["y_up"].mean())
    base_dn = float(test_panel["y_dn"].mean())

    # ── Helper: tune a threshold on val, apply to test ────────────────────────
    def _val_thresh(arr_val: np.ndarray, q: float) -> float:
        return float(np.percentile(arr_val[~np.isnan(arr_val)], 100 * q))

    # ── Gate sweep ────────────────────────────────────────────────────────────
    print(f"\n[layer3 {target_label}] BULL gate sweep "
          f"(thresholds from VAL, eval on TEST; test base={base_up:.2%}):")
    print(f"  {'gate':<70}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")

    val_p_up = val_panel["p_up"].values
    val_diff_up = val_panel["diff_up"].values
    val_2dmin_up = val_panel["p_up_2d_min"].values
    val_z_up = val_panel["p_up_z"].values
    val_up_liq = val_panel["up_liq"].values
    val_bad_up = val_panel["bad_up"].values

    t_pu_99 = _val_thresh(val_p_up,    0.99)
    t_pu_98 = _val_thresh(val_p_up,    0.98)
    t_pu_97 = _val_thresh(val_p_up,    0.97)
    t_pu_95 = _val_thresh(val_p_up,    0.95)
    t_pu_90 = _val_thresh(val_p_up,    0.90)
    t_du_95 = _val_thresh(val_diff_up, 0.95)
    t_du_90 = _val_thresh(val_diff_up, 0.90)
    t_du_98 = _val_thresh(val_diff_up, 0.98)
    t_2du95 = _val_thresh(val_2dmin_up, 0.95)
    t_2du90 = _val_thresh(val_2dmin_up, 0.90)
    t_zu_95 = _val_thresh(val_z_up,    0.95)
    t_zu_90 = _val_thresh(val_z_up,    0.90)
    t_ul_90 = _val_thresh(val_up_liq,  0.90)
    t_ul_80 = _val_thresh(val_up_liq,  0.80)
    t_bu_30 = _val_thresh(val_bad_up,  0.30)
    t_bu_50 = _val_thresh(val_bad_up,  0.50)

    # Test arrays
    tp_pu = test_panel["p_up"].values
    tp_du = test_panel["diff_up"].values
    tp_2d = test_panel["p_up_2d_min"].values
    tp_zu = test_panel["p_up_z"].values
    tp_ul = test_panel["up_liq"].values
    tp_bu = test_panel["bad_up"].values
    tp_nf = test_panel["nifty_dp"].values
    y_up  = test_panel["y_up"].values.astype(int)

    gates_up = [
        ("p_up >= val_p95",                                              (tp_pu >= t_pu_95)),
        ("p_up >= val_p98",                                              (tp_pu >= t_pu_98)),
        ("p_up >= val_p99",                                              (tp_pu >= t_pu_99)),
        ("diff_up >= val_p95 (direction agreement)",                     (tp_du >= t_du_95)),
        ("diff_up >= val_p98",                                           (tp_du >= t_du_98)),
        ("2d_min(p_up) >= val_p95 (persistence)",                        (tp_2d >= t_2du95)),
        ("2d_min(p_up) >= val_p90",                                      (tp_2d >= t_2du90)),
        ("p_up_z >= val_p95 (per-stock surprise)",                       (tp_zu >= t_zu_95)),
        ("p_up_z >= val_p90",                                            (tp_zu >= t_zu_90)),
        # Combos
        ("p_up >= p95 + diff_up >= p90",                                 (tp_pu >= t_pu_95) & (tp_du >= t_du_90)),
        ("p_up >= p95 + 2d_min >= p90 (persist + level)",                (tp_pu >= t_pu_95) & (tp_2d >= t_2du90)),
        ("p_up >= p95 + up_liq >= p80",                                  (tp_pu >= t_pu_95) & (tp_ul >= t_ul_80)),
        ("p_up >= p95 + up_liq >= p80 + bad_up <= p50",                  (tp_pu >= t_pu_95) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p95 + diff_up >= p90 + up_liq >= p80 + bad_up <= p50", (tp_pu >= t_pu_95) & (tp_du >= t_du_90) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p97 + diff_up >= p90 + up_liq >= p80 + bad_up <= p50", (tp_pu >= t_pu_97) & (tp_du >= t_du_90) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p97 + diff_up >= p95 + up_liq >= p80 + bad_up <= p50", (tp_pu >= t_pu_97) & (tp_du >= t_du_95) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p95 + 2d_min >= p90 + up_liq >= p80 + bad_up <= p50",  (tp_pu >= t_pu_95) & (tp_2d >= t_2du90) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p95 + p_up_z >= p90 + up_liq >= p80 + bad_up <= p50",  (tp_pu >= t_pu_95) & (tp_zu >= t_zu_90) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p97 + 2d_min >= p90 + diff_up >= p90 + up_liq >= p80 + bad_up <= p50",
                                                                          (tp_pu >= t_pu_97) & (tp_2d >= t_2du90) & (tp_du >= t_du_90) & (tp_ul >= t_ul_80) & (tp_bu <= t_bu_50)),
        ("p_up >= p98 + diff_up >= p90 + up_liq >= p90 + bad_up <= p30", (tp_pu >= t_pu_98) & (tp_du >= t_du_90) & (tp_ul >= t_ul_90) & (tp_bu <= t_bu_30)),
        ("p_up >= p99 + diff_up >= p90 + up_liq >= p90 + bad_up <= p30", (tp_pu >= t_pu_99) & (tp_du >= t_du_90) & (tp_ul >= t_ul_90) & (tp_bu <= t_bu_30)),
    ]
    for label, mask in gates_up:
        n, pos, prec, lift = _eval_one_gate(label, mask, y_up, base_up)
        if n > 0:
            print(f"  {label:<70}  {n:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")

    # ── BEAR side ─────────────────────────────────────────────────────────────
    print(f"\n[layer3 {target_label}] BEAR gate sweep (test base={base_dn:.2%}):")
    print(f"  {'gate':<70}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")
    val_p_dn = val_panel["p_dn"].values
    val_diff_dn = val_panel["diff_dn"].values
    val_2dmin_dn = val_panel["p_dn_2d_min"].values
    val_z_dn = val_panel["p_dn_z"].values
    val_dn_liq = val_panel["dn_liq"].values
    val_bad_dn = val_panel["bad_dn"].values
    t_pd_99 = _val_thresh(val_p_dn,    0.99)
    t_pd_98 = _val_thresh(val_p_dn,    0.98)
    t_pd_97 = _val_thresh(val_p_dn,    0.97)
    t_pd_95 = _val_thresh(val_p_dn,    0.95)
    t_dd_95 = _val_thresh(val_diff_dn, 0.95)
    t_dd_98 = _val_thresh(val_diff_dn, 0.98)
    t_dd_90 = _val_thresh(val_diff_dn, 0.90)
    t_2dd95 = _val_thresh(val_2dmin_dn, 0.95)
    t_2dd90 = _val_thresh(val_2dmin_dn, 0.90)
    t_zd_95 = _val_thresh(val_z_dn,    0.95)
    t_zd_90 = _val_thresh(val_z_dn,    0.90)
    t_dl_90 = _val_thresh(val_dn_liq,  0.90)
    t_dl_80 = _val_thresh(val_dn_liq,  0.80)
    t_bd_30 = _val_thresh(val_bad_dn,  0.30)
    t_bd_50 = _val_thresh(val_bad_dn,  0.50)
    tp_pd = test_panel["p_dn"].values
    tp_dd = test_panel["diff_dn"].values
    tp_2dd = test_panel["p_dn_2d_min"].values
    tp_zd = test_panel["p_dn_z"].values
    tp_dl = test_panel["dn_liq"].values
    tp_bd = test_panel["bad_dn"].values
    y_dn  = test_panel["y_dn"].values.astype(int)
    gates_dn = [
        ("p_dn >= val_p95",                                              (tp_pd >= t_pd_95)),
        ("p_dn >= val_p98",                                              (tp_pd >= t_pd_98)),
        ("p_dn >= val_p99",                                              (tp_pd >= t_pd_99)),
        ("diff_dn >= val_p95",                                           (tp_dd >= t_dd_95)),
        ("diff_dn >= val_p98",                                           (tp_dd >= t_dd_98)),
        ("2d_min(p_dn) >= val_p95",                                      (tp_2dd >= t_2dd95)),
        ("2d_min(p_dn) >= val_p90",                                      (tp_2dd >= t_2dd90)),
        ("p_dn_z >= val_p95",                                            (tp_zd >= t_zd_95)),
        ("p_dn_z >= val_p90",                                            (tp_zd >= t_zd_90)),
        ("p_dn >= p95 + diff_dn >= p90",                                 (tp_pd >= t_pd_95) & (tp_dd >= t_dd_90)),
        ("p_dn >= p95 + 2d_min >= p90",                                  (tp_pd >= t_pd_95) & (tp_2dd >= t_2dd90)),
        ("p_dn >= p95 + dn_liq >= p80",                                  (tp_pd >= t_pd_95) & (tp_dl >= t_dl_80)),
        ("p_dn >= p95 + dn_liq >= p80 + bad_dn <= p50",                  (tp_pd >= t_pd_95) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p95 + diff_dn >= p90 + dn_liq >= p80 + bad_dn <= p50", (tp_pd >= t_pd_95) & (tp_dd >= t_dd_90) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p97 + diff_dn >= p90 + dn_liq >= p80 + bad_dn <= p50", (tp_pd >= t_pd_97) & (tp_dd >= t_dd_90) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p97 + diff_dn >= p95 + dn_liq >= p80 + bad_dn <= p50", (tp_pd >= t_pd_97) & (tp_dd >= t_dd_95) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p95 + 2d_min >= p90 + dn_liq >= p80 + bad_dn <= p50",  (tp_pd >= t_pd_95) & (tp_2dd >= t_2dd90) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p95 + p_dn_z >= p90 + dn_liq >= p80 + bad_dn <= p50",  (tp_pd >= t_pd_95) & (tp_zd >= t_zd_90) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p97 + 2d_min >= p90 + diff_dn >= p90 + dn_liq >= p80 + bad_dn <= p50",
                                                                          (tp_pd >= t_pd_97) & (tp_2dd >= t_2dd90) & (tp_dd >= t_dd_90) & (tp_dl >= t_dl_80) & (tp_bd <= t_bd_50)),
        ("p_dn >= p98 + diff_dn >= p90 + dn_liq >= p90 + bad_dn <= p30", (tp_pd >= t_pd_98) & (tp_dd >= t_dd_90) & (tp_dl >= t_dl_90) & (tp_bd <= t_bd_30)),
        ("p_dn >= p99 + diff_dn >= p90 + dn_liq >= p90 + bad_dn <= p30", (tp_pd >= t_pd_99) & (tp_dd >= t_dd_90) & (tp_dl >= t_dl_90) & (tp_bd <= t_bd_30)),
    ]
    for label, mask in gates_dn:
        n, pos, prec, lift = _eval_one_gate(label, mask, y_dn, base_dn)
        if n > 0:
            print(f"  {label:<70}  {n:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")

    # ── Composite product score ──────────────────────────────────────────────
    # score_up = p_up * up_liq * (1 - bad_up) * max(0, 1 + diff_up_norm)
    # — multiplicative compression: high score requires ALL signals to agree.
    print(f"\n[layer3 {target_label}] COMPOSITE product score (test):")
    print(f"  {'metric':<40}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")

    def _safe_norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rng = np.nanmax(x) - np.nanmin(x)
        if rng <= 0:
            return np.zeros_like(x)
        return (x - np.nanmin(x)) / rng

    diff_up_norm = _safe_norm(test_panel["diff_up"].values)
    diff_dn_norm = _safe_norm(test_panel["diff_dn"].values)

    score_up = (
        tp_pu *
        np.clip(tp_ul, 0, 1) *
        np.clip(1.0 - tp_bu, 0, 1) *
        np.clip(0.5 + diff_up_norm, 0, 2)
    )
    score_dn = (
        tp_pd *
        np.clip(tp_dl, 0, 1) *
        np.clip(1.0 - tp_bd, 0, 1) *
        np.clip(0.5 + diff_dn_norm, 0, 2)
    )

    for direction, score, y, base in [
        ("BULL composite", score_up, y_up, base_up),
        ("BEAR composite", score_dn, y_dn, base_dn),
    ]:
        for pct in [0.5, 1, 2, 3, 5, 10]:
            k = max(1, int(len(score) * pct / 100))
            idx = np.argsort(score)[::-1][:k]
            pos = int(y[idx].sum())
            prec = pos / max(k, 1)
            lift = prec / base if base > 0 else float("nan")
            print(f"  {direction} top-{pct}%                       {k:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")

    # ── Per-day best-signal-only gate ────────────────────────────────────────
    # Take only the highest-composite-score stock per day. Fire only if it
    # also exceeds an absolute threshold (from val).
    print(f"\n[layer3 {target_label}] PER-DAY best signal (test):")

    # Compute val composite for threshold calibration
    val_diff_up_norm = _safe_norm(val_panel["diff_up"].values)
    val_diff_dn_norm = _safe_norm(val_panel["diff_dn"].values)
    val_score_up = (
        val_panel["p_up"].values *
        np.clip(val_panel["up_liq"].values, 0, 1) *
        np.clip(1.0 - val_panel["bad_up"].values, 0, 1) *
        np.clip(0.5 + val_diff_up_norm, 0, 2)
    )
    val_score_dn = (
        val_panel["p_dn"].values *
        np.clip(val_panel["dn_liq"].values, 0, 1) *
        np.clip(1.0 - val_panel["bad_dn"].values, 0, 1) *
        np.clip(0.5 + val_diff_dn_norm, 0, 2)
    )

    test_df = pd.DataFrame({
        "date":   test_panel["date"].values,
        "score_up": score_up,
        "score_dn": score_dn,
        "y_up":   y_up, "y_dn": y_dn,
    })

    print(f"  {'config':<48}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")
    for direction, score_col, y_col, val_scores, base in [
        ("BULL per-day", "score_up", "y_up", val_score_up, base_up),
        ("BEAR per-day", "score_dn", "y_dn", val_score_dn, base_dn),
    ]:
        # Best stock per day
        idx_best = test_df.groupby("date")[score_col].idxmax()
        best = test_df.loc[idx_best].reset_index(drop=True)
        for pct in [99, 98, 95, 90]:
            thr = float(np.percentile(val_scores, pct))
            mask = best[score_col].values >= thr
            n = int(mask.sum())
            pos = int(best.loc[mask, y_col].sum())
            prec = pos / max(n, 1)
            lift = prec / base if base > 0 else float("nan")
            print(f"  {direction}, best/day, score>=val_p{pct}            {n:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")


def _layer3_consensus(
    df: pd.DataFrame,
    clean_up_pred: np.ndarray,
    clean_dn_pred: np.ndarray,
    yte_up: np.ndarray,
    yte_dn: np.ndarray,
) -> None:
    """
    Apply Layer-3 consensus gating using prod models.

    For clean_up_5: clean_up_score AND up_liq score AND (1 - bad_up) AND nifty-clean-up.
    For clean_dn_5: mirror.

    Gates are applied as within-day percentile ranks so the system adapts to
    daily score distributions.
    """
    from pathlib import Path
    from .models.ensemble import load_prod_models, predict_ensemble
    from .config import OUT_ROOT

    PROD_DIR = OUT_ROOT / "models" / "prod"
    print(f"\n[layer3] loading prod models from {PROD_DIR}")
    models = load_prod_models(PROD_DIR)
    needed = ("up_liq", "dn_liq", "bad_up_liquid", "bad_dn_liquid")
    have = set(models.keys())
    missing = [t for t in needed if t not in have]
    if missing:
        print(f"[layer3] WARN: missing prod targets: {missing}")
        print(f"[layer3] available targets: {sorted(have)}")

    # Run prod models on test data only (saves time).
    test_mask = (df["split"] == "test").values
    X_test = df.loc[test_mask].reset_index(drop=True)
    print(f"[layer3] scoring {len(X_test)} test rows with prod models...")
    prod_scores = predict_ensemble(models, X_test)
    print(f"[layer3] prod scored targets: {sorted(prod_scores.keys())}")

    # Use GLOBAL thresholds (percentile of all test scores) rather than
    # within-day. Within-day picks top-K per day even on days where no setup
    # is strong, which dilutes precision. Global thresholds wait for absolute
    # high-conviction setups.
    def _gpct(arr: np.ndarray, pct: float) -> float:
        """Return value at given percentile (e.g. pct=0.99 → 99th pctile)."""
        return float(np.percentile(arr, 100 * pct))

    up_prod = pd.Series(prod_scores.get("up_liq",        np.zeros(len(X_test)))).values
    dn_prod = pd.Series(prod_scores.get("dn_liq",        np.zeros(len(X_test)))).values
    badup   = pd.Series(prod_scores.get("bad_up_liquid", np.zeros(len(X_test)))).values
    baddn   = pd.Series(prod_scores.get("bad_dn_liquid", np.zeros(len(X_test)))).values
    nifty_dp = X_test["nifty_directional_purity_5d_signed"].fillna(0.0).values

    # Global thresholds (computed once from test distribution)
    cu_p99  = _gpct(clean_up_pred, 0.99)
    cu_p98  = _gpct(clean_up_pred, 0.98)
    cu_p97  = _gpct(clean_up_pred, 0.97)
    cu_p95  = _gpct(clean_up_pred, 0.95)
    cu_p90  = _gpct(clean_up_pred, 0.90)
    cd_p99  = _gpct(clean_dn_pred, 0.99)
    cd_p98  = _gpct(clean_dn_pred, 0.98)
    cd_p97  = _gpct(clean_dn_pred, 0.97)
    cd_p95  = _gpct(clean_dn_pred, 0.95)
    cd_p90  = _gpct(clean_dn_pred, 0.90)
    up_p80  = _gpct(up_prod, 0.80)
    up_p90  = _gpct(up_prod, 0.90)
    dn_p80  = _gpct(dn_prod, 0.80)
    dn_p90  = _gpct(dn_prod, 0.90)
    bu_p50  = _gpct(badup, 0.50)
    bu_p30  = _gpct(badup, 0.30)
    bd_p50  = _gpct(baddn, 0.50)
    bd_p30  = _gpct(baddn, 0.30)

    print("\n[layer3] CLEAN_UP_5 — global consensus gate sweep (test):")
    print(f"  {'gate':<60}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")
    base_up = float(yte_up.mean())
    GATES_UP = [
        ("clean_up_5 >= p90 (global)",                                    clean_up_pred >= cu_p90),
        ("clean_up_5 >= p95",                                             clean_up_pred >= cu_p95),
        ("clean_up_5 >= p98",                                             clean_up_pred >= cu_p98),
        ("clean_up_5 >= p99",                                             clean_up_pred >= cu_p99),
        ("clean p95 + up_liq p80",                                        (clean_up_pred >= cu_p95) & (up_prod >= up_p80)),
        ("clean p95 + up_liq p80 + bad_up<=p50",                          (clean_up_pred >= cu_p95) & (up_prod >= up_p80) & (badup <= bu_p50)),
        ("clean p95 + up_liq p80 + bad_up<=p50 + nifty>0",                (clean_up_pred >= cu_p95) & (up_prod >= up_p80) & (badup <= bu_p50) & (nifty_dp > 0)),
        ("clean p97 + up_liq p80 + bad_up<=p50 + nifty>0.2",              (clean_up_pred >= cu_p97) & (up_prod >= up_p80) & (badup <= bu_p50) & (nifty_dp > 0.2)),
        ("clean p98 + up_liq p90 + bad_up<=p30 + nifty>0",                (clean_up_pred >= cu_p98) & (up_prod >= up_p90) & (badup <= bu_p30) & (nifty_dp > 0)),
        ("clean p98 + up_liq p90 + bad_up<=p30 + nifty>0.2",              (clean_up_pred >= cu_p98) & (up_prod >= up_p90) & (badup <= bu_p30) & (nifty_dp > 0.2)),
        ("clean p99 + up_liq p90 + bad_up<=p30",                          (clean_up_pred >= cu_p99) & (up_prod >= up_p90) & (badup <= bu_p30)),
        ("clean p99 + up_liq p90 + bad_up<=p30 + nifty>0",                (clean_up_pred >= cu_p99) & (up_prod >= up_p90) & (badup <= bu_p30) & (nifty_dp > 0)),
    ]
    for label, mask in GATES_UP:
        n = int(mask.sum())
        pos = int(yte_up[mask].sum())
        prec = pos / max(n, 1)
        lift = prec / base_up if base_up > 0 else float("nan")
        print(f"  {label:<60}  {n:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")

    print("\n[layer3] CLEAN_DN_5 — global consensus gate sweep (test):")
    print(f"  {'gate':<60}  {'n':>4}  {'pos':>4}  {'prec':>6}  {'lift':>6}")
    base_dn = float(yte_dn.mean())
    GATES_DN = [
        ("clean_dn_5 >= p90 (global)",                                    clean_dn_pred >= cd_p90),
        ("clean_dn_5 >= p95",                                             clean_dn_pred >= cd_p95),
        ("clean_dn_5 >= p98",                                             clean_dn_pred >= cd_p98),
        ("clean_dn_5 >= p99",                                             clean_dn_pred >= cd_p99),
        ("clean p95 + dn_liq p80",                                        (clean_dn_pred >= cd_p95) & (dn_prod >= dn_p80)),
        ("clean p95 + dn_liq p80 + bad_dn<=p50",                          (clean_dn_pred >= cd_p95) & (dn_prod >= dn_p80) & (baddn <= bd_p50)),
        ("clean p95 + dn_liq p80 + bad_dn<=p50 + nifty<0",                (clean_dn_pred >= cd_p95) & (dn_prod >= dn_p80) & (baddn <= bd_p50) & (nifty_dp < 0)),
        ("clean p97 + dn_liq p80 + bad_dn<=p50 + nifty<-0.2",             (clean_dn_pred >= cd_p97) & (dn_prod >= dn_p80) & (baddn <= bd_p50) & (nifty_dp < -0.2)),
        ("clean p98 + dn_liq p90 + bad_dn<=p30 + nifty<0",                (clean_dn_pred >= cd_p98) & (dn_prod >= dn_p90) & (baddn <= bd_p30) & (nifty_dp < 0)),
        ("clean p98 + dn_liq p90 + bad_dn<=p30 + nifty<-0.2",             (clean_dn_pred >= cd_p98) & (dn_prod >= dn_p90) & (baddn <= bd_p30) & (nifty_dp < -0.2)),
        ("clean p99 + dn_liq p90 + bad_dn<=p30",                          (clean_dn_pred >= cd_p99) & (dn_prod >= dn_p90) & (baddn <= bd_p30)),
        ("clean p99 + dn_liq p90 + bad_dn<=p30 + nifty<0",                (clean_dn_pred >= cd_p99) & (dn_prod >= dn_p90) & (baddn <= bd_p30) & (nifty_dp < 0)),
    ]
    for label, mask in GATES_DN:
        n = int(mask.sum())
        pos = int(yte_dn[mask].sum())
        prec = pos / max(n, 1)
        lift = prec / base_dn if base_dn > 0 else float("nan")
        print(f"  {label:<60}  {n:>4d}  {pos:>4d}  {prec:>5.1%}  {lift:>5.2f}x")


def main() -> None:
    df, path_feats = _load_data()

    SHARED_PATH = [
        "directional_purity_5d", "directional_purity_5d_signed",
        "range_purity_5d_signed", "whipsaw_count_5d",
        "directional_purity_10d_signed", "range_purity_10d_signed",
        "nifty_up_pressure_5d", "nifty_directional_purity_5d_signed",
        "nifty_range_purity_5d_signed", "nifty_directional_purity_5d",
    ]
    BULL_PATH = SHARED_PATH + [
        "up_pressure_5d", "lower_wick_pressure_5d", "clean_runup_5d",
        "up_pressure_10d",
        "vol_up_ratio_5d", "gap_up_persistence_5d", "clear_air_up_streak_5d",
    ]
    up_feats = TARGET_FEATURE_COLS["up_liq"] + BULL_PATH
    dn_feats = TARGET_FEATURE_COLS["dn_liq"] + path_feats

    # ── STRICT (dn<0.91%) ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STRICT CRITERION: up>4% AND opposite<0.91%")
    print("="*80)
    p_val_up, p_test_up, yv_up, yte_up = _train_and_return(
        df, "clean_up_5", up_feats, saddle_filter=True)
    _report_topk("clean_up_5 STRICT val",  p_val_up,  yv_up)
    _report_topk("clean_up_5 STRICT test", p_test_up, yte_up)

    p_val_dn, p_test_dn, yv_dn, yte_dn = _train_and_return(
        df, "clean_dn_5", dn_feats, saddle_filter=False)
    _report_topk("clean_dn_5 STRICT val",  p_val_dn,  yv_dn)
    _report_topk("clean_dn_5 STRICT test", p_test_dn, yte_dn)

    _layer3_full(df, p_val_up, p_test_up, p_val_dn, p_test_dn,
                 yv_up, yte_up, yv_dn, yte_dn, target_label="STRICT")

    # ── LOOSE (dn<1.5%) ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("LOOSE CRITERION: up>4% AND opposite<1.5%")
    print("="*80)
    print(f"  train clean_up_5_loose rate: {df.loc[df['split']=='train', 'clean_up_5_loose'].mean():.2%}")
    print(f"  val   clean_up_5_loose rate: {df.loc[df['split']=='val',   'clean_up_5_loose'].mean():.2%}")
    print(f"  test  clean_up_5_loose rate: {df.loc[df['split']=='test',  'clean_up_5_loose'].mean():.2%}")
    print(f"  train clean_dn_5_loose rate: {df.loc[df['split']=='train', 'clean_dn_5_loose'].mean():.2%}")
    print(f"  val   clean_dn_5_loose rate: {df.loc[df['split']=='val',   'clean_dn_5_loose'].mean():.2%}")
    print(f"  test  clean_dn_5_loose rate: {df.loc[df['split']=='test',  'clean_dn_5_loose'].mean():.2%}")

    p_val_upL, p_test_upL, yv_upL, yte_upL = _train_and_return(
        df, "clean_up_5_loose", up_feats, saddle_filter=True)
    _report_topk("clean_up_5_loose val",  p_val_upL,  yv_upL)
    _report_topk("clean_up_5_loose test", p_test_upL, yte_upL)

    p_val_dnL, p_test_dnL, yv_dnL, yte_dnL = _train_and_return(
        df, "clean_dn_5_loose", dn_feats, saddle_filter=False)
    _report_topk("clean_dn_5_loose val",  p_val_dnL,  yv_dnL)
    _report_topk("clean_dn_5_loose test", p_test_dnL, yte_dnL)

    _layer3_full(df, p_val_upL, p_test_upL, p_val_dnL, p_test_dnL,
                 yv_upL, yte_upL, yv_dnL, yte_dnL, target_label="LOOSE")


if __name__ == "__main__":
    main()
