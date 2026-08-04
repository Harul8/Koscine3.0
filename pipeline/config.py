"""
Pipeline configuration — single source of truth for all paths and constants.
"""
import os
from pathlib import Path
import datetime as dt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_ROOT = Path(r"C:\Users\rahul\Koscine 3.0\data")
DATA_ROOT = Path(os.environ.get("ESN_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))
OUT_ROOT  = Path(os.environ.get("ESN_OUT_ROOT",  str(ROOT)))

RAW_DIR    = DATA_ROOT / "raw"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR   = OUT_ROOT  / "gold"
MODEL_DIR  = OUT_ROOT  / "models"
MLRUNS_DIR = OUT_ROOT  / "mlruns"        # MLflow local tracking store

SILVER_TABLES = {
    "eod_stock":           SILVER_DIR / "eod_stock.parquet",
    "eod_deriv_daily":     SILVER_DIR / "eod_deriv_daily.parquet",
    "eod_deriv_contracts": SILVER_DIR / "eod_deriv_contracts",
    "lot_size":            SILVER_DIR / "lot_size.parquet",
    "participant_oi":      SILVER_DIR / "participant_oi.parquet",
    "indices":             SILVER_DIR / "indices.parquet",
    "fii_dii_cash":        SILVER_DIR / "fii_dii_cash.parquet",
    "corp_actions":        SILVER_DIR / "corp_actions.parquet",
    "earnings":            SILVER_DIR / "earnings.parquet",
    "fundamentals":        SILVER_DIR / "fundamentals.parquet",
    "analyst_ratings":     SILVER_DIR / "analyst_ratings.parquet",
    "earnings_eps":        SILVER_DIR / "earnings_eps.parquet",
    "fii_sector":          SILVER_DIR / "fii_sector.parquet",
    "eod_deriv":           SILVER_DIR / "eod_deriv.parquet",
    "block_deals":         SILVER_DIR / "block_deals.parquet",
    "bulk_deals":          SILVER_DIR / "bulk_deals.parquet",
    "investing_eps":       SILVER_DIR / "investing_eps.parquet",
}

GOLD_FEATURES   = GOLD_DIR / "features.parquet"   # raw features only — NO label columns
GOLD_LABELS     = GOLD_DIR / "labels.parquet"      # default labels (data-driven K)
GOLD_ZONES      = GOLD_DIR / "zones.parquet"
PREDICTIONS_DIR = GOLD_DIR

YF_DAILY_WATCHLIST = Path(__file__).resolve().parent / "yf_daily_watchlist.txt"

# ── Data format cutovers ────────────────────────────────────────────────────────
UDIFF_CUTOVER = dt.date(2024, 7, 8)
CASH_NEW_FROM = dt.date(2020, 1, 1)

# ── Label config (v4 — data-driven K calibration) ──────────────────────────────
# K is NOT a constant.  It is derived each labels run from training data:
#   K = percentile(forward_return / ATR_20d,  100 × (1 − target_rate))
# This guarantees the base rate equals TARGET_RATE_* on training data automatically.
LABEL_HORIZON    = 5         # kept for features.py compatibility
LABEL_HORIZON_3D = 3
LABEL_HORIZON_5D = 5
LABEL_ATR_PERIOD       = 20
LABEL_TARGET_RATE_BASE = 0.20   # target base rate for up_3/dn_3/up_5/dn_5
LABEL_TARGET_RATE_XL   = 0.10   # target base rate for up_5_xl/dn_5_xl (large-move tier)
LABEL_ATR_CLIP_LO = 0.020       # floor: sub-2% moves are noise
LABEL_ATR_CLIP_HI = 0.120       # cap:  >12% required is unreasonable

# ── Train / val / test split ────────────────────────────────────────────────────
# Overridable via env vars for monthly retrain.
TRAIN_END = os.environ.get("ESN_TRAIN_END", "2024-12-31")
VAL_END   = os.environ.get("ESN_VAL_END",   "2025-06-30")
# test = everything after VAL_END

# ── Model targets ───────────────────────────────────────────────────────────────
MODEL_TARGETS = ["up_3", "dn_3", "up_5", "dn_5", "up_5_xl", "dn_5_xl"]

# ── Tiered model targets (v5 — liquid/rest split with fixed thresholds) ─────────
# Two model families:
#   liquid tier (first LIQUID_TIER_SIZE symbols in Universe file): 4% threshold
#   rest tier   (all other F&O stocks):                            8% threshold
TIERED_MODEL_TARGETS  = ["up_liq", "dn_liq", "up_rest", "dn_rest"]
TIERED_OVERLAY_TARGETS = ["bad_up", "bad_dn"]   # wrong-direction overlay labels

# ── Clean directional model targets (Layer 1 relabeling) ─────────────────────────
# Predict days that will have a CLEAN move: one-sided move, opposite side < 0.91%.
# Tier-asymmetric thresholds (chosen from 6-yr base-rate analysis):
#   liquid-30 → 4% threshold  (base rate ~12%, vol-regime dominant)
#   rest-35   → 7% threshold  (base rate ~9%, directional/PCR/FII dominant)
# LGBM-only, AP metric, no scale_pos_weight. Saddle filter only on liquid bull.
CLEAN_MODEL_TARGETS = [
    "clean_up_5_liq",  "clean_dn_5_liq",   # 4% threshold, liquid-30 only
    "clean_up_5_rest", "clean_dn_5_rest",  # 7% threshold, rest-35 only
]

# Thresholds for clean-day definition (used in labels.py and predict.py)
CLEAN_UP_THRESH_LIQ  = 0.04    # liquid forward upside  > 4%
CLEAN_DN_THRESH_LIQ  = 0.04    # liquid forward downside > 4%
CLEAN_UP_THRESH_REST = 0.07    # rest   forward upside  > 7%
CLEAN_DN_THRESH_REST = 0.07    # rest   forward downside > 7%
CLEAN_NOISE_THRESH   = 0.0091  # opposite side   < 0.91% (strict, both tiers)

# Legacy aliases (older code paths still reference CLEAN_*_THRESH without suffix).
# Point them at the liquid value — those callers were always liquid-only.
CLEAN_UP_THRESH = CLEAN_UP_THRESH_LIQ
CLEAN_DN_THRESH = CLEAN_DN_THRESH_LIQ

GOLD_LABELS_TIERED = GOLD_DIR / "labels_tiered.parquet"

# Fixed move thresholds (fraction of T+1 open)
LIQUID_THRESHOLD  = 0.04   # 4% target for top-30 liquid stocks
REST_THRESHOLD    = 0.07   # 7% target for rest of F&O universe
BAD_CLOSE_THRESH  = 0.02   # stock closed ≥2% against direction = bad signal

# Sample-weight penalty for wrong-direction rows during base model training.
# Rows where the stock closed strongly against the predicted direction receive
# this multiplier so the model is punished harder for those errors.
WRONG_DIR_PENALTY = 4.0

# Per-target overrides for wrong-direction penalty.
# DN liquid has the highest observed wrong-direction rate (24-26%) so it gets
# an extra-heavy penalty to force the model to avoid those errors.
WRONG_DIR_PENALTY_BY_TARGET: dict[str, float] = {
    "dn_liq":  8.0,   # liquid DN: worst wrong-direction rate → max suppression
    "dn_rest": 6.0,   # rest DN:   also elevated, tighter than default
}

# STRONG bucket z-score bonus and probability multiplier.
# STRONG needs a bigger separation from MOD to be meaningfully different.
STRONG_Z_BONUS    = 1.0    # z above Z_BUCKET_THRESH required for STRONG (was 0.5)
STRONG_PROB_MULT  = 2.0    # prob multiple of base_rate required for STRONG (was 1.5)

# Target distribution goals (aspirational; reported in evaluate)
TARGET_HIT_RATE   = 0.65   # 65% of signals should touch target
PARTIAL_WIN_RATE  = 0.25   # 25% right direction, missed target
WRONG_DIR_RATE    = 0.10   # ≤10% wrong direction (bad close)

# ── TabM params ─────────────────────────────────────────────────────────────────
# Per-target overrides applied on top of TabMModel defaults.
# Key levers:
#   lr       — 1e-3 overshoots with 250+ features; 5e-4 converges more stably
#   patience — give more epochs to escape flat regions (checked every 10 epochs)
#   k        — ensemble members; 16 >> 8 for high-precision XL targets
TABM_TARGET_PARAMS: dict[str, dict] = {
    # n_seeds=5 trains 5 TabM models with different seeds and averages their
    # predictions at inference time. Reduces run-to-run variance by ~1/sqrt(N).
    # patience=80: gives warm restart at epoch 50 enough time (3 eval checks
    # after the restart) to find a better basin before early stop fires.
    "up_3":    dict(lr=1e-3, patience=50, k=8, n_seeds=5),
    "dn_3":    dict(lr=1e-3, patience=50, k=8, n_seeds=5),
    "up_5":    dict(lr=1e-3, patience=50, k=8, n_seeds=5),
    "dn_5":    dict(lr=1e-3, patience=50, k=8, n_seeds=5),
    "up_5_xl": dict(lr=1e-3, patience=80, k=8, n_seeds=5),
    "dn_5_xl": dict(lr=1e-3, patience=80, k=8, n_seeds=5),
}

# ── LightGBM params ─────────────────────────────────────────────────────────────
LGBM_BASE_PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 4000,   # cap; actual count set by early_stopping(50)
    learning_rate     = 0.01,
    num_leaves        = 63,
    min_child_samples = 80,
    reg_lambda        = 2.0,
    reg_alpha         = 0.5,
    feature_fraction  = 0.80,
    bagging_fraction  = 0.80,
    bagging_freq      = 5,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

# Per-target overrides applied on top of base params.
# DN models historically overfit more → tighter regularisation.
LGBM_TARGET_PARAMS: dict[str, dict] = {
    "up_3":    dict(learning_rate=0.015, num_leaves=31, min_child_samples=50,  reg_lambda=2.0),
    "dn_3":    dict(learning_rate=0.010, num_leaves=23, min_child_samples=100, reg_lambda=4.0, reg_alpha=1.0),
    "up_5":    dict(learning_rate=0.020, num_leaves=31, min_child_samples=50,  reg_lambda=2.0),
    "dn_5":    dict(learning_rate=0.008, num_leaves=15, min_child_samples=150, reg_lambda=6.0, reg_alpha=2.0,
                    feature_fraction=0.65, bagging_fraction=0.70),
    "up_5_xl": dict(learning_rate=0.010, num_leaves=23, min_child_samples=80,  reg_lambda=3.0, reg_alpha=1.0),
    "dn_5_xl": dict(learning_rate=0.008, num_leaves=23, min_child_samples=100, reg_lambda=4.0, reg_alpha=1.5),
}

# scale_pos_weight = (1 − base_rate) / base_rate
# Based on actual train base rates from v4 labels run:
#   up_3=20.21%  dn_3=20.18%  up_5=20.57%  dn_5=20.47%
#   up_5_xl=11.78%  dn_5_xl=11.27%
LGBM_POS_WEIGHTS: dict[str, float] = {
    "up_3":    3.9,
    "dn_3":    3.9,
    "up_5":    3.9,
    "dn_5":    3.9,
    "up_5_xl": 7.5,
    "dn_5_xl": 7.9,
}

# ── LightGBM params for tiered targets ─────────────────────────────────────────
# Liquid tier has fewer training rows (~30 stocks × history) → smaller leaves.
# DN models use tighter regularisation (historically overfit more).
LGBM_TIERED_TARGET_PARAMS: dict[str, dict] = {
    "up_liq":  dict(learning_rate=0.015, num_leaves=31, min_child_samples=30,  reg_lambda=2.0),
    "dn_liq":  dict(learning_rate=0.010, num_leaves=23, min_child_samples=40,  reg_lambda=4.0, reg_alpha=0.5),
    "up_rest": dict(learning_rate=0.015, num_leaves=31, min_child_samples=80,  reg_lambda=2.0),
    "dn_rest": dict(learning_rate=0.008, num_leaves=23, min_child_samples=120, reg_lambda=5.0, reg_alpha=1.0,
                    feature_fraction=0.70, bagging_fraction=0.75),
    # Overlay: lighter trees — they complement base models, not replace them
    "bad_up":  dict(learning_rate=0.015, num_leaves=23, min_child_samples=50,  reg_lambda=3.0),
    "bad_dn":  dict(learning_rate=0.015, num_leaves=23, min_child_samples=50,  reg_lambda=3.0),
}

# ── LightGBM params for clean directional targets ────────────────────────────────
# CRITICAL: must use average_precision metric — AUC + scale_pos_weight causes iter=2
# saddle collapse for dn target (12% positive rate → trivial all-negative optimum).
# NO scale_pos_weight: removed entirely; let AP guide learning without rebalancing.
# 8-seed ensemble (averaged); bull uses saddle_filter=True (drop iter<50 seeds).
LGBM_CLEAN_PARAMS: dict = dict(
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
LGBM_CLEAN_N_SEEDS       = 8    # seeds kept in final ensemble
LGBM_CLEAN_SEED_BUFFER  = 5    # extra seeds trained to survive saddle filter
LGBM_CLEAN_SADDLE_THRESH = 30  # drop seeds with best_iter < this (all 4 targets)

# ── Self-Paced Ensemble (SPE) for clean models ───────────────────────────────
# When True, replaces fixed-seed training with SPE: K rounds of balanced
# sampling where each round up-weights "hard" negatives based on the previous
# round's predictions (arxiv 1909.03500, ICDE 2020).
# Helps rest-tier clean models which suffer from 9% pos rate + large neg pool.
LGBM_CLEAN_USE_SPE   = True
LGBM_CLEAN_SPE_ROUNDS = 8     # equals N_SEEDS; each SPE round is one ensemble member

# ── Focal loss for clean models ───────────────────────────────────────────────
# When True, replaces standard binary cross-entropy with focal loss for clean
# model LightGBM training.  Focal loss down-weights easy negatives automatically
# (similar effect to scale_pos_weight but without saddle-collapse risk).
# gamma=2.0 is the standard value from the original paper (Lin et al. 2017).
LGBM_CLEAN_USE_FOCAL  = True
LGBM_CLEAN_FOCAL_GAMMA = 2.0

# Per-target param overrides on top of LGBM_CLEAN_PARAMS.
# Liquid targets train well with base params.
# Rest targets need softer trees: lower lr + shallower leaves eliminate saddle
# collapses that appear at ~7-9% base rate with the aggressive base params.
LGBM_CLEAN_TARGET_PARAMS: dict[str, dict] = {
    "clean_up_5_liq":  {},  # base params are fine
    "clean_dn_5_liq":  {},  # base params are fine (saddle filter now on)
    "clean_up_5_rest": dict(
        learning_rate=0.015, num_leaves=31, min_child_samples=80, reg_lambda=2.0,
    ),
    "clean_dn_5_rest": dict(
        learning_rate=0.010, num_leaves=31, min_child_samples=80, reg_lambda=2.5,
    ),
}

# ── CatBoost params ─────────────────────────────────────────────────────────────
CATBOOST_BASE_PARAMS: dict = dict(
    iterations          = 3000,
    learning_rate       = 0.03,
    depth               = 6,
    l2_leaf_reg         = 3.0,
    random_seed         = 42,
    verbose             = 0,
    eval_metric         = "AUC",
    early_stopping_rounds = 100,
    bootstrap_type      = "Bernoulli",
    subsample           = 0.80,
    od_type             = "Iter",     # overfitting detector type
    # rsm (feature fraction) intentionally omitted: not supported on GPU
)

CATBOOST_TARGET_PARAMS: dict[str, dict] = {
    "up_liq":  dict(learning_rate=0.030, depth=6, l2_leaf_reg=3.0),
    # dn_liq: force CPU so AUC is used for early-stopping evaluation.
    # On GPU, CatBoost falls back to Logloss which trivially bottoms out at iter 17
    # when heavy sample-weights (8×) are combined with class_weights — the model
    # finds "predict everything low" as the Logloss optimum immediately.
    # CPU with AUC evaluates correctly and finds a meaningful minimum.
    "dn_liq":  dict(learning_rate=0.025, depth=5, l2_leaf_reg=4.0, iterations=2500,
                    task_type="CPU"),
    "up_rest": dict(learning_rate=0.025, depth=6, l2_leaf_reg=3.0),
    "dn_rest": dict(learning_rate=0.020, depth=5, l2_leaf_reg=5.0, iterations=2500),
    # Overlay models also collapse on GPU (bad_up iter=1).  Force CPU.
    "bad_up":  dict(learning_rate=0.030, depth=5, l2_leaf_reg=3.0, iterations=2000,
                    task_type="CPU"),
    "bad_dn":  dict(learning_rate=0.030, depth=5, l2_leaf_reg=3.0, iterations=2000,
                    task_type="CPU"),
}

# Per-target LGBM/CatBoost blend weights for the tiered ensemble.
# Tuple of (lgbm_weight, catboost_weight) — normalized to 1.0 internally.
# Default for any target not listed = (0.50, 0.50).
#
# After switching dn_liq CatBoost to CPU (AUC eval), CatBoost now outperforms LGBM:
# AP 0.323 vs 0.299, prec@10% 38.3% vs 34.6% → 50/50 blend is correct.
# Adjust if a future retrain shows CatBoost instability on dn_liq.
TIERED_BLEND_WEIGHTS: dict[str, tuple[float, float]] = {
    # No per-target overrides — all targets use 50/50 blend.
}

# Overlay blend: final_score = base_prob × (1 − OVERLAY_ALPHA × overlay_prob)
# 0.0 = overlay disabled, 1.0 = full multiplicative penalty
OVERLAY_ALPHA = 0.60

# Calibration method per target
CALIBRATION_METHOD: dict[str, str] = {
    "up_3":    "beta",       # beta > isotonic on small cal sets (asymmetric sigmoid)
    "dn_3":    "beta",
    "up_5":    "beta",
    "dn_5":    "beta",
    "up_5_xl": "sigmoid",    # low base rate → very narrow spread → Platt still safe
    "dn_5_xl": "sigmoid",
    # Tiered targets — beta calibration replaces isotonic (research: isotonic degrades
    # on < ~3k calibration samples; our tiered val sets are ~3.5k–4.2k → borderline)
    "up_liq":  "beta",
    "dn_liq":  "beta",
    "up_rest": "beta",
    "dn_rest": "beta",
    "bad_up":  "beta",
    "bad_dn":  "beta",
    # Clean directional targets — temperature scaling: 1 parameter, lowest overfitting
    # risk on the small ~400-600 positive calibration samples in clean val sets.
    "clean_up_5_liq":  "temperature",
    "clean_dn_5_liq":  "temperature",
    "clean_up_5_rest": "temperature",
    "clean_dn_5_rest": "temperature",
}

# ── Per-target feature selection ────────────────────────────────────────────────
# Features are assigned to directional groups; TARGET_FEATURE_COLS is composed
# from those groups so each model sees only directionally-coherent signals.
#
# Rule of thumb for group assignment:
#   SHARED  — direction-neutral: volatility, macro regime, sector, path quality,
#             bidirectional derivatives/flows, earnings calendar, beta
#   UP      — bullish-biased: accumulation, breakout, FII inflow, quality, HH/HL
#   DN      — bearish-biased: exhaustion, distribution, smart-money shorts, LH/LL
#   UP_XL   — large-upside-specific: pre-breakout compression, 52w high breakout
#   DN_XL   — large-downside-specific: stretch×beta compounds, macro×stock timing
#
# Target compositions:
#   up_3 / up_5   = SHARED + UP
#   dn_3 / dn_5   = SHARED + DN
#   up_5_xl       = SHARED + UP + UP_XL
#   dn_5_xl       = SHARED + DN + DN_XL
#
# Features listed here that are absent from features.parquet are silently ignored
# by experiment.py — safe to add future features without guard logic.

_F_SHARED: list[str] = [
    # ── Returns — key horizons only ───────────────────────────────────────────
    # ret_3d / ret_10d dropped: covered by neighbours (1d↔5d, 5d↔20d); adding
    # them inflates the return cluster without independent signal.
    # ret_1d_vol_scaled dropped: day-scale vol-normalised return is too noisy
    # to generalise; ret_5d_vol_scaled carries the multi-day signal cleanly.
    "ret_1d", "ret_5d", "ret_20d",
    "ret_5d_vol_scaled", "ret_20d_vol_scaled",

    # ── Volatility context ────────────────────────────────────────────────────
    # vol_ratio_5d dropped: very-short-window ratio is noisy; vol_ratio_20d
    # gives a more stable read of current vs historical activity.
    "atr_14", "hv_20", "vol_expansion", "vol_ratio_20d", "days_since_vol_surge",

    # ── Compression — one metric per concept ─────────────────────────────────
    # bb_width dropped: fully redundant with bb_squeeze (same Bollinger numerics).
    # range_vs_vol_implied dropped: tight_range_10d already captures compression;
    # the vol-implied normalisation adds minimal marginal signal.
    # squeeze_rank retained: rolling 252d percentile of bb_squeeze lets the model
    # compare today's squeeze to historical baseline.
    "bb_squeeze", "squeeze_rank", "tight_range_10d",

    # ── Price position ────────────────────────────────────────────────────────
    # close_in_range_5d dropped: 5d average of close_position; once the model
    # has close_position + ret_5d it can reconstruct this.
    "close_position", "rsi_14", "macd_hist",

    # ── Key level distances (ATR-normalised) ─────────────────────────────────
    # lt_resistance_z / lt_support_z dropped: long-term (>2yr) zone data is
    # sparse and has many NaNs — adds noise more than signal.
    # zone_box_width_pct / weeks_in_box dropped: niche consolidation geometry;
    # low importance across all historical models.
    # dist_from_bb_upper kept: above upper BB = momentum for up, overbought for dn.
    # dist_52w_high_rank kept: cross-sectional stretch rank — useful for both.
    "nearest_resistance_z", "nearest_support_z",
    "dist_from_bb_upper", "dist_52w_high_rank",

    # ── Stock structural state ────────────────────────────────────────────────
    # stock_dd_52w = -dist_52w_high → dropped (exact linear function of it).
    # stock_recovery_52w ≈ function of dist_52w_high + dist_52w_low → dropped.
    "stock_phase", "dist_52w_high", "dist_52w_low",

    # ── Macro regime ─────────────────────────────────────────────────────────
    # regime dropped: market_phase covers the same taxonomy with more granularity
    # (5 states vs 3) — both in SHARED is redundant, keep the richer one.
    # vix_level dropped: vix_rank_252d (percentile vs 1yr history) is a better
    # normalised measure; absolute VIX level is regime-dependent noise.
    # nifty_dist_52w_low dropped: nifty_dist_52w_high + nifty_ret_5d/20d already
    # capture market position; the low-side distance adds minimal signal.
    "market_phase", "regime_duration_days", "regime_changed_5d", "is_expiry_week",
    "nifty_ret_5d", "nifty_ret_20d", "nifty_above_200ma",
    "nifty_dist_52w_high", "vix_rank_252d",

    # ── Sector ───────────────────────────────────────────────────────────────
    # Kept: the three most informative sector signals.
    # Dropped: fii_sector_flow_90d (trend read from 30d+zscore), fii_sector_flow_pct_aum
    # fii_sector_flow_streak, fii_sector_breadth_pos, fii_sector_aum_pct_change_90d
    # — all derived from the same underlying fortnightly NSDL data; marginal once
    # the flow level + z-score + rotation rank are already present.
    "sector_ret_5d", "rel_strength_sector",
    "fii_sector_flow_30d", "fii_sector_flow_zscore", "fii_sector_rotation_rank",

    # ── Derivatives — direction-neutral ───────────────────────────────────────
    # pcr_vol dropped: pcr_oi is more stable (OI builds over days, volume is daily);
    # pcr_oi + pcr_oi_rank_60d + pcr_chg_5d cover level + history + direction.
    # fut_oi_chg_1d dropped: 1-day OI change is noisy; fut_oi_chg_5d gives the
    # same signal with less noise.
    # max_pain_dist dropped: highly correlated with dist_call_wall / dist_put_wall
    # which already appear in UP/DN with direction-specific context.
    "pcr_oi", "pcr_oi_rank_60d", "pcr_chg_5d",
    "fut_oi_chg_5d", "basis_pct", "basis_chg_5d",
    "fut_vol_oi_ratio", "opt_oi_ratio_20d", "days_to_expiry",

    # ── FII/DII flows — bidirectional ────────────────────────────────────────
    # fii_buy_sell_ratio dropped: ratio of fii_cash_net_5d / total — redundant
    # with fii_cash_zscore which already normalises the flow.
    # fii_sector_flow_acceleration dropped: already captured in the sector block.
    "fii_cash_net_5d", "fii_cash_zscore", "fii_cash_net_30d",
    "fii_cash_acceleration", "fii_cash_streak",
    "fii_idx_fut_net_chg", "dii_cash_net_5d", "smart_vs_retail",

    # ── Earnings — bidirectional ─────────────────────────────────────────────
    "days_to_next_earnings", "days_since_last_earnings", "last_eps_surprise_pct",

    # ── Market sensitivity ────────────────────────────────────────────────────
    # beta_nifty_60d dropped: for a 3-5d prediction horizon, 20d beta is more
    # relevant; 60d smooths out the very regime changes we want to capture.
    "beta_nifty_20d",

    # ── Path quality (lagged realized windows — no look-ahead) ───────────────
    "net_swing_lag1", "path_asym_lag1", "net_swing_lag2", "net_swing_roll20",

    # ── Cross-sectional ranks — neutral ───────────────────────────────────────
    # pcr_oi_rank (daily universe rank) dropped: closely correlated with
    # pcr_oi_rank_60d (rolling rank) already in derivatives; two PCR ranks
    # in SHARED is redundant.
    # atr_compression_rank dropped: breakout-compression signal → moved to UP_XL.
    "ret_5d_rank", "ret_20d_rank", "ret_20d_vol_scaled_rank",
    "vol_ratio_20d_rank", "fut_oi_chg_5d_rank",
    "ret_5d_rank_sector", "ret_20d_rank_sector",

    # ── Weekly candles — neutral body/position features ───────────────────────
    # w_body_ratio dropped: linear transform of w_body_pct (body / full range
    # vs body as % of open) — models learn one or the other, not both.
    # w_outside_bar dropped: week that exceeded prior high AND low is rare and
    # correlates strongly with vol_expansion which is already in SHARED.
    "w_body_pct", "w_close_pos", "w_range_pct", "w_inside_bar", "w_is_doji",

    # ── Block deals — net activity (sign carries direction) ───────────────────
    # block_net_qty_5d replaced by block_net_val_5d (crores): value is
    # comparable across stocks; raw share counts are meaningless cross-sectionally.
    # bulk_net_val_5d dropped: block deals more regulated / reliable signal.
    "block_net_val_5d", "block_deal_flag_5d",

    # ── Compound — neutral ────────────────────────────────────────────────────
    # dip_vol_spike_5d dropped from SHARED — down-move on elevated volume is
    # primarily a capitulation/recovery signal → moved to _F_UP.
    "zone_energy", "earnings_vol_setup",
]

_F_UP: list[str] = [
    # ── Trend quality ─────────────────────────────────────────────────────────
    "above_200ma", "win_rate_20d", "sharpe_20d", "max_dd_20d",
    "dist_52w_low_z",                  # near 52w low = dip-buy opportunity

    # ── Breakout & consolidation ──────────────────────────────────────────────
    "breakout_quality", "breakout_vol_confirm", "squeeze_breakout",
    "adl_divergence", "obv_slope_10d",
    "zone_breakout", "zone_breakout_ffill",
    "support_valid_touches", "support_zone_strength",
    "consolidating_at_support", "support_zone_age_weeks",
    "resistance_valid_touches",        # strong resistance = breakout has room to run

    # ── Volume / delivery (accumulation evidence) ─────────────────────────────
    "delivery_pct", "delivery_ratio_20d", "delivery_streak",
    "amihud_illiquidity",
    "avg_trade_value_ratio_20d", "avg_trade_value_zscore_60d",

    # ── Futures — long side ───────────────────────────────────────────────────
    "long_buildup", "long_buildup_streak", "short_covering",
    "dist_put_wall",                   # put-wall support below = floor for longs

    # ── Gaps ─────────────────────────────────────────────────────────────────
    "gap_up_count_20d",

    # ── FII accumulation signals ──────────────────────────────────────────────
    "fii_extreme_inflow",

    # ── Earnings — quality / beat (last_eps_surprise_pct is in SHARED) ────────
    "eps_surprise_3q_avg", "eps_beat_streak",
    "eps_growth_yoy", "days_to_ex_div", "days_since_ex_div",
    "inv_surprise_pct", "inv_beat", "inv_beat_rate_4q",
    "inv_avg_surprise_4q", "inv_rev_surprise_pct",

    # ── Weekly bullish patterns ───────────────────────────────────────────────
    "w_gap", "w_body_expand", "w_range_expand",
    "w_bull_engulf", "w_is_hammer", "w_is_marubozu", "w_lower_wick",

    # ── Higher-high / higher-low structure ────────────────────────────────────
    "higher_high", "higher_low",

    # ── Block deals (buying by institutions/promoters) ────────────────────────
    # block_buy_qty_5d dropped: raw share count meaningless cross-sectionally.
    "block_buy_val_5d",
    "bulk_buy_val_5d", "bulk_buy_flag_5d",

    # ── Compound interactions — bullish ───────────────────────────────────────
    "mega_breakout", "bull_confluence", "quiet_accum",
    "fii_nifty_divergence",            # FII buying into market weakness = accumulation
    "vol_recovery_score", "fii_macro_stock_divergence",
    "dip_vol_spike_5d",                # capitulation down-move + vol spike = dip-buy entry
    "fii_accum_prob",

    # ── Cross-sectional (up-biased) ───────────────────────────────────────────
    "delivery_ratio_20d_rank", "avg_trade_value_ratio_20d_rank",
    "breakout_quality_rank", "breakout_quality_rank_sector",
    "vol_expansion_rank", "vol_expansion_rank_sector",
    "sharpe_20d_rank",
]

_F_UP_XL: list[str] = [
    # Large breakout requires sustained pre-breakout compression — small-move
    # breakouts don't need as many squeeze-specific features.
    "atr_compression", "atr_compression_rank",  # rank moved from SHARED: breakout-specific
    "days_in_squeeze", "price_acceleration",
    "hi52_breakout",                   # 52-week high breakout = momentum continuation
    "double_squeeze",                  # inside_bar × squeeze = coiled spring
    "dist_52w_high_z",                 # less stretched = more room to run
]

_F_DN: list[str] = [
    # ── Overbought / exhaustion ───────────────────────────────────────────────
    "rsi_overbought_days",             # consecutive days RSI ≥ 70 = exhaustion duration
    "dist_above_50ma", "dist_above_50ma_z", "dist_above_50ma_rank",
    "consecutive_green_days", "consecutive_green_days_rank",
    "consecutive_red_days",
    "distribution_days_20d", "red_day_vol_ratio", "days_since_20d_high",
    "gap_down_count_20d",
    "dist_52w_high_z",                 # near high = stretched = reversal risk
    "dist_from_bb_upper_rank",

    # ── Zone resistance ───────────────────────────────────────────────────────
    "consolidating_at_resistance", "resistance_zone_strength",
    "resistance_zone_age_weeks",
    "zone_breakdown", "zone_breakdown_ffill",
    "dist_call_wall",                  # call wall = option dealer resistance above

    # ── Futures — short side ──────────────────────────────────────────────────
    "short_buildup", "short_buildup_streak", "short_buildup_streak_rank",
    "long_unwinding", "long_unwinding_streak",
    "short_buildup_5d_count", "long_unwind_5d_count",

    # ── Options — bearish hedging flow ────────────────────────────────────────
    "put_call_vol_ratio", "put_call_vol_rank_60d", "put_call_vol_ratio_rank",
    "fut_oi_z_60d", "fut_oi_z_60d_rank", "basis_rank_60d",

    # ── Options structure ─────────────────────────────────────────────────────
    "put_oi_pct", "annualized_basis", "wall_compression",

    # ── ATM implied volatility ────────────────────────────────────────────────
    "atm_iv", "put_call_iv_skew", "atm_iv_rank_252d", "put_call_iv_skew_rank_60d",

    # ── FII cash — outflow / reversal signals ────────────────────────────────
    "fii_cash_reversal_flag", "fii_extreme_outflow",

    # ── Participant OI — FII & Client stock futures ───────────────────────────
    "fii_stk_fut_net_chg_5d",   # FII reducing long / adding short in stock futures
    "client_stk_fut_net",        # retail stock futures (contrarian: extreme long = bearish)
    "fii_vs_client_stk",         # FII/Client ratio: negative = FII short, retail long

    # ── Participant OI — put accumulation + pro desk ──────────────────────────
    "fii_put_long_stk_chg_5d",  # FII accelerating put buying in stock options
    "client_put_short_stk",     # retail selling stock puts (complacency)
    "pro_stk_fut_net",          # proprietary desk stock futures net
    "fii_stk_net_opt_dir",      # FII net synthetic short in stock options

    # ── Macro bear timing ─────────────────────────────────────────────────────
    "nifty_momentum_accel", "nifty_hv_20", "vix_chg_5d",

    # ── Earnings — miss ───────────────────────────────────────────────────────
    "big_eps_miss", "eps_miss_streak",

    # ── Block / bulk deals ────────────────────────────────────────────────────
    # block_sell_qty_5d dropped: raw share count meaningless cross-sectionally.
    # block_sell_val_5d (crores) is the right signal.
    "block_sell_val_5d",
    "bulk_sell_val_5d", "bulk_sell_flag_5d",

    # ── Weekly bearish patterns ───────────────────────────────────────────────
    "w_upper_wick", "w_bear_engulf", "w_is_shooting_star",

    # ── Lower-high / lower-low structure ─────────────────────────────────────
    "lower_high", "lower_low",

    # ── Cross-sectional ───────────────────────────────────────────────────────
    "beta_nifty_20d_rank",
]

_F_DN_XL: list[str] = [
    # ── Original technical compounds (keep — collectively critical for neural net) ──
    # Individual permutation importance is near-zero but joint removal caused -13pp
    # regression. Neural networks use these as structural anchors/regularization.
    "stretch_beta",          # overbought × high beta = maximum market exposure
    "stretch_beta_rank",     # cross-sectional vulnerability rank
    "dn_smart_short",        # overbought × short_buildup_streak = smart money leaning short
    "dn_basis_stretch",      # overbought × basis at 60d low = futures market retreating
    "put_activity_stretch",  # overbought × put-flow rank = options market front-running drop
    "iv_vs_hv",              # ATM IV / realized HV (live after silver rebuild)
    "dn_oi_crowded",         # overbought × OI z-score = crowded longs that need to flush
    "dn_exhaustion",         # overbought × green_days_streak = momentum exhaustion
    "dn_macro_stock_timing", # VIX_rising × stretch × beta = right stock at right macro moment

    # ── FII absolute positioning ──────────────────────────────────────────────
    "fii_stk_fut_net",            # FII net in stock futures (level)
    "fii_put_long_stk",           # FII absolute put accumulation in stock options

    # ── FII synthetic options positioning ─────────────────────────────────────
    "fii_stk_put_call_oi_ratio",  # FII long_puts / long_calls: >1 = bearish options bias
    "fii_stk_net_opt_dir_chg_5d", # 5d change in FII total synthetic short

    # ── Client complacency ────────────────────────────────────────────────────
    "client_stk_call_put_net",    # retail net long calls: high = complacent

    # ── Cross-participant compound ────────────────────────────────────────────
    "fii_put_rush_on_stretch",    # FII put acceleration × per-stock overboughtness
]


# ── Group 10 — Path Asymmetry features (Layer 2 for clean directional models) ────
# Backward-looking OHLCV-derived signals computed over 5d and 10d rolling windows.
# Quantify how "clean" and directional recent price action was — the core Layer-2
# signal that separates clean directional days from whipsaw setups.
#
# Stock path features (19):
#   Body direction fraction:  up_pressure_5d, dn_pressure_5d
#   Wick asymmetry:           upper_wick_pressure_5d, lower_wick_pressure_5d
#   Directional purity:       directional_purity_5d, directional_purity_5d_signed, range_purity_5d_signed
#   Noise measure:            whipsaw_count_5d
#   Clean run counts:         clean_runup_5d, clean_rundn_5d
#   10d versions:             up_pressure_10d, directional_purity_10d_signed, range_purity_10d_signed
#   Volume asymmetry:         vol_up_ratio_5d, vol_dn_ratio_5d
#   Gap persistence:          gap_up_persistence_5d, gap_dn_persistence_5d
#   Clear-air structure:      clear_air_up_streak_5d, clear_air_dn_streak_5d
# NIFTY regime path features (4):
#   nifty_up_pressure_5d, nifty_directional_purity_5d_signed,
#   nifty_range_purity_5d_signed, nifty_directional_purity_5d

# Shared path (purity + NIFTY regime) — both bull and bear benefit
_F_PATH_ASYM_SHARED: list[str] = [
    "directional_purity_5d", "directional_purity_5d_signed",
    "range_purity_5d_signed", "whipsaw_count_5d",
    "directional_purity_10d_signed", "range_purity_10d_signed",
    "nifty_up_pressure_5d", "nifty_directional_purity_5d_signed",
    "nifty_range_purity_5d_signed", "nifty_directional_purity_5d",
]

# Bull-specific path extras (bull body direction + lower wick = buyer strength)
_F_PATH_ASYM_BULL: list[str] = _F_PATH_ASYM_SHARED + [
    "up_pressure_5d", "lower_wick_pressure_5d", "clean_runup_5d",
    "up_pressure_10d",
    "vol_up_ratio_5d", "gap_up_persistence_5d", "clear_air_up_streak_5d",
]

# SHAP interaction crosses for clean_up_5_liq — top-10 pairs from
# analysis/shap_interactions.py (mean |SHAP interaction| ranked).
# These are computed in pipeline/features.py _compound_interactions().
_F_SHAP_CROSS_LIQ_BULL: list[str] = [
    "fii_cash_net_30d_x_days_since_last_earn",   # rank 1  |0.00501|
    "regime_duration_days_x_opt_oi_ratio_20d",   # rank 2  |0.00454|
    "opt_oi_ratio_20d_x_days_since_last_earn",   # rank 3  |0.00443|
    "regime_duration_days_x_vix_rank_252d",      # rank 4  |0.00441|
    "nifty_ret_20d_x_vix_rank_252d",             # rank 5  |0.00439|
    "nifty_above_200ma_x_nifty_dist_52w_high",   # rank 6  |0.00424|
    "regime_duration_days_x_days_to_expiry",     # rank 7  |0.00424|
    "regime_duration_days_x_days_since_last_earn", # rank 8 |0.00419|
    "nifty_dist_52w_high_x_vix_rank_252d",       # rank 9  |0.00407|
    "regime_duration_days_x_days_to_next_earning", # rank 10 |0.00379|
]

# Full path feature set (all 23 features) — used by bear model to see bull features
# as contrarian signals; removing them degraded bear precision significantly.
_F_PATH_ASYM: list[str] = _F_PATH_ASYM_SHARED + [
    "up_pressure_5d",    "dn_pressure_5d",
    "upper_wick_pressure_5d", "lower_wick_pressure_5d",
    "clean_runup_5d",    "clean_rundn_5d",
    "up_pressure_10d",
    "vol_up_ratio_5d",   "vol_dn_ratio_5d",
    "gap_up_persistence_5d", "gap_dn_persistence_5d",
    "clear_air_up_streak_5d", "clear_air_dn_streak_5d",
]


# ── Rest-tier directional features (from clean-move 2020-2025 analysis) ─────────
# Driving signals for rest-35 large (>7%) clean moves:
#   PCR / option positioning   → contrarian bullish setups (pcr_vol, put_oi_pct)
#   FII flow                   → strongest contextual signal for rest tier
#   OI dynamics                → long buildup precedes bear (trap), unwound longs
#                                precede bull (clear runway)
#   Premium richness (iv_vs_hv) → puts mispriced before bear moves
#   Walls + delivery           → setup-quality features
_F_REST_DIRECTIONAL: list[str] = [
    # PCR / option positioning
    "pcr_vol", "pcr_oi", "pcr_oi_rank_60d", "put_oi_pct",
    "put_call_vol_ratio", "put_call_vol_rank_60d",
    "put_call_iv_skew", "put_call_iv_skew_rank_60d",
    # FII flow — both stock and sector
    "fii_cash_net_30d", "fii_sector_flow_30d", "fii_sector_flow_90d",
    "fii_sector_flow_streak", "fii_sector_breadth_pos",
    "fii_sector_flow_zscore", "fii_buy_sell_ratio",
    # OI dynamics
    "fut_oi_z_60d", "fut_oi_z_60d_rank", "fut_oi_chg_5d",
    "long_unwind_5d_count", "short_buildup_5d_count",
    "opt_oi_ratio_20d", "opt_total_oi_chg_5d_rank",
    # Premium richness — bear trap signal
    "iv_vs_hv",
    # Option walls — bull room above / bear breakdown of put wall
    "dist_call_wall", "dist_put_wall", "wall_compression",
]

# Rest bull: contrarian/clean-runway setup — delivery_pct higher than bear
_F_REST_BULL: list[str] = _F_REST_DIRECTIONAL + [
    "delivery_pct", "delivery_ratio_20d",
]

# Rest bear: trap setup — same directional core + path-asymmetry bull features
# as contrarian signals (mirrors _F_PATH_ASYM logic for liquid bear)
_F_REST_BEAR: list[str] = _F_REST_DIRECTIONAL


# ── Rest-tier directional features kept for legacy base-model use ────────────────
# (used by up_rest / dn_rest tiered base models — not the clean models)


def _compose_features(*groups: list[str]) -> list[str]:
    """Merge feature groups into a deduplicated list preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for f in g:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


# ── Data-driven feature lists for ALL clean models ──────────────────────────────
# Source: analysis/rank_clean_features.py — per-feature AUC(class vs neutral)
# computed on 2020-2025 data. Top 40 by AUC lift, AUC_MIN=0.55 filter applied.
# AUC direction annotated inline (higher->pos or lower->pos).

# ── Liquid tier (4% threshold) ───────────────────────────────────────────────────
_F_CLEAN_LIQ_BULL: list[str] = [
    "atm_iv",                  # AUC 0.7694  higher->pos
    "atm_ce_iv",               # AUC 0.7659  higher->pos
    "atm_pe_iv",               # AUC 0.7649  higher->pos
    "hv_20",                   # AUC 0.7026  higher->pos
    "vix_level",               # AUC 0.6971  higher->pos
    "w_range_pct",             # AUC 0.6578  higher->pos
    "client_put_short_stk",    # AUC 0.3532  lower->pos
    "bb_width",                # AUC 0.6448  higher->pos
    "delivery_pct",            # AUC 0.3559  lower->pos
    "vix_rank_252d",           # AUC 0.6416  higher->pos
    "nifty_hv_20",             # AUC 0.6401  higher->pos
    "tight_range_10d",         # AUC 0.6397  higher->pos
    "client_stk_fut_net",      # AUC 0.3633  lower->pos
    "days_to_next_earnings",   # AUC 0.6350  higher->pos
    "days_since_ex_div",       # AUC 0.3669  lower->pos
    "gap_up_count_20d",        # AUC 0.6330  higher->pos
    "pro_stk_fut_net",         # AUC 0.3752  lower->pos
    "dist_52w_low",            # AUC 0.6199  higher->pos
    "client_stk_call_put_net", # AUC 0.3849  lower->pos
    "nifty_dist_52w_high",     # AUC 0.6113  higher->pos
    "atm_iv_rank_252d",        # AUC 0.6109  higher->pos
    "max_dd_20d",              # AUC 0.3945  lower->pos
    "days_to_ex_div",          # AUC 0.6052  higher->pos
    "fii_put_long_stk",        # AUC 0.4044  lower->pos
    "beta_nifty_60d",          # AUC 0.5932  higher->pos
    "gap_down_count_20d",      # AUC 0.5914  higher->pos
    "fii_stk_net_opt_dir",     # AUC 0.4149  lower->pos
    "days_since_last_earnings",# AUC 0.4166  lower->pos
    "zone_box_width_pct",      # AUC 0.5770  higher->pos
    "beta_nifty_20d",          # AUC 0.5764  higher->pos
    "dist_put_wall",           # AUC 0.5748  higher->pos
    "pcr_oi_rank",             # AUC 0.5672  higher->pos
    "support_level",           # AUC 0.4346  lower->pos
    "stretch_beta_rank",       # AUC 0.5645  higher->pos
    "nearest_resistance_dist", # AUC 0.5645  higher->pos
    "dist_above_50ma_rank",    # AUC 0.5633  higher->pos
    "beta_nifty_20d_rank",     # AUC 0.5622  higher->pos
    "nifty_dist_52w_low",      # AUC 0.5615  higher->pos
    "fii_vs_client_stk",       # AUC 0.5614  higher->pos
    "amihud_illiquidity",      # AUC 0.5612  higher->pos
]

_F_CLEAN_LIQ_BEAR: list[str] = [
    "atm_iv",                    # AUC 0.8090  higher->pos
    "atm_ce_iv",                 # AUC 0.8056  higher->pos
    "atm_pe_iv",                 # AUC 0.8024  higher->pos
    "hv_20",                     # AUC 0.7223  higher->pos
    "vix_level",                 # AUC 0.6984  higher->pos
    "w_range_pct",               # AUC 0.6755  higher->pos
    "atm_iv_rank_252d",          # AUC 0.6697  higher->pos
    "delivery_pct",              # AUC 0.3325  lower->pos
    "bb_width",                  # AUC 0.6586  higher->pos
    "tight_range_10d",           # AUC 0.6490  higher->pos
    "days_to_next_earnings",     # AUC 0.6456  higher->pos
    "vix_rank_252d",             # AUC 0.6441  higher->pos
    "dist_52w_low",              # AUC 0.6406  higher->pos
    "nifty_hv_20",               # AUC 0.6376  higher->pos
    "days_since_ex_div",         # AUC 0.3645  lower->pos
    "gap_up_count_20d",          # AUC 0.6327  higher->pos
    "client_put_short_stk",      # AUC 0.3692  lower->pos
    "max_dd_20d",                # AUC 0.3759  lower->pos
    "beta_nifty_60d",            # AUC 0.6222  higher->pos
    "client_stk_fut_net",        # AUC 0.3908  lower->pos
    "pro_stk_fut_net",           # AUC 0.3951  lower->pos
    "zone_box_width_pct",        # AUC 0.6023  higher->pos
    "days_to_ex_div",            # AUC 0.5988  higher->pos
    "beta_nifty_20d",            # AUC 0.5913  higher->pos
    "nifty_dist_52w_high",       # AUC 0.5909  higher->pos
    "gap_down_count_20d",        # AUC 0.5901  higher->pos
    "fii_stk_put_call_oi_ratio", # AUC 0.4126  lower->pos
    "client_stk_call_put_net",   # AUC 0.4151  lower->pos
    "fii_put_long_stk",          # AUC 0.4184  lower->pos
    "days_since_last_earnings",  # AUC 0.4195  lower->pos
    "atr_compression",           # AUC 0.5803  higher->pos
    "nearest_support_dist",      # AUC 0.5767  higher->pos
    "nifty_dist_52w_low",        # AUC 0.5766  higher->pos
    "beta_nifty_20d_rank",       # AUC 0.5716  higher->pos
    "dist_put_wall",             # AUC 0.5694  higher->pos
    "gap_up_persistence_5d",     # AUC 0.5685  higher->pos
    "dist_above_50ma_rank",      # AUC 0.5647  higher->pos
    "atr_compression_rank",      # AUC 0.5643  higher->pos
    "amihud_illiquidity",        # AUC 0.5640  higher->pos
    "nearest_resistance_dist",   # AUC 0.5640  higher->pos
]

# ── Rest tier (7% threshold) ──────────────────────────────────────────────────────

_F_CLEAN_REST_BULL: list[str] = [
    "atm_iv",                  # AUC 0.802  higher->pos
    "atm_ce_iv",               # AUC 0.789  higher->pos
    "atm_pe_iv",               # AUC 0.776  higher->pos
    "hv_20",                   # AUC 0.773  higher->pos
    "bb_width",                # AUC 0.750  higher->pos
    "tight_range_10d",         # AUC 0.748  higher->pos
    "fii_sector_flow_zscore",  # AUC 0.276  lower->pos
    "w_range_pct",             # AUC 0.723  higher->pos
    "dist_52w_low",            # AUC 0.713  higher->pos
    "fii_sector_breadth_pos",  # AUC 0.710  higher->pos
    "atr_compression",         # AUC 0.686  higher->pos
    "fii_sector_flow_90d",     # AUC 0.669  higher->pos
    "stretch_beta_rank",       # AUC 0.668  higher->pos
    "dist_above_50ma_rank",    # AUC 0.668  higher->pos
    "pcr_oi_rank_60d",         # AUC 0.663  higher->pos
    "atm_iv_rank_252d",        # AUC 0.661  higher->pos
    "fii_sector_flow_streak",  # AUC 0.657  higher->pos
    "atr_compression_rank",    # AUC 0.656  higher->pos
    "dist_52w_high_z",         # AUC 0.350  lower->pos
    "stock_recovery_52w",      # AUC 0.646  higher->pos
    "vol_expansion",           # AUC 0.646  higher->pos
    "above_200ma",             # AUC 0.643  higher->pos
    "squeeze_rank",            # AUC 0.643  higher->pos
    "dist_above_50ma",         # AUC 0.643  higher->pos
    "bb_squeeze",              # AUC 0.641  higher->pos
    "gap_up_count_20d",        # AUC 0.639  higher->pos
    "ret_20d_rank",            # AUC 0.638  higher->pos
    "ret_20d_rank_sector",     # AUC 0.637  higher->pos
    "dn_basis_stretch",        # AUC 0.636  higher->pos
    "nearest_support_dist",    # AUC 0.635  higher->pos
    "fii_sector_flow_30d",     # AUC 0.635  higher->pos
    "delivery_pct",            # AUC 0.366  lower->pos
    "zone_box_width_pct",      # AUC 0.634  higher->pos
    "put_activity_stretch",    # AUC 0.634  higher->pos
    "stretch_beta",            # AUC 0.634  higher->pos
    "pcr_oi_rank",             # AUC 0.633  higher->pos
    "dist_above_50ma_z",       # AUC 0.629  higher->pos
    "gap_down_count_20d",      # AUC 0.628  higher->pos
    "red_day_vol_ratio",       # AUC 0.372  lower->pos
    "ret_20d",                 # AUC 0.628  higher->pos
]

_F_CLEAN_REST_BEAR: list[str] = [
    "atm_iv",                    # AUC 0.809  higher->pos
    "hv_20",                     # AUC 0.807  higher->pos
    "atm_pe_iv",                 # AUC 0.802  higher->pos
    "bb_width",                  # AUC 0.794  higher->pos
    "tight_range_10d",           # AUC 0.791  higher->pos
    "atm_ce_iv",                 # AUC 0.788  higher->pos
    "w_range_pct",               # AUC 0.763  higher->pos
    "fii_sector_flow_zscore",    # AUC 0.248  lower->pos
    "fii_sector_breadth_pos",    # AUC 0.745  higher->pos
    "fii_sector_flow_90d",       # AUC 0.717  higher->pos
    "dist_52w_low",              # AUC 0.701  higher->pos
    "atm_iv_rank_252d",          # AUC 0.699  higher->pos
    "atr_compression",           # AUC 0.697  higher->pos
    "fii_sector_flow_30d",       # AUC 0.695  higher->pos
    "stretch_beta_rank",         # AUC 0.692  higher->pos
    "fii_sector_flow_streak",    # AUC 0.679  higher->pos
    "dist_above_50ma_rank",      # AUC 0.675  higher->pos
    "squeeze_rank",              # AUC 0.673  higher->pos
    "delivery_pct",              # AUC 0.329  lower->pos
    "bb_squeeze",                # AUC 0.668  higher->pos
    "atr_compression_rank",      # AUC 0.668  higher->pos
    "gap_down_count_20d",        # AUC 0.667  higher->pos
    "vol_expansion",             # AUC 0.667  higher->pos
    "gap_up_count_20d",          # AUC 0.660  higher->pos
    "max_dd_20d",                # AUC 0.346  lower->pos
    "avg_trade_value_rank",      # AUC 0.647  higher->pos
    "vix_rank_252d",             # AUC 0.646  higher->pos
    "ret_20d_rank",              # AUC 0.646  higher->pos
    "ret_20d_rank_sector",       # AUC 0.646  higher->pos
    "dist_52w_high_z",           # AUC 0.355  lower->pos
    "zone_box_width_pct",        # AUC 0.643  higher->pos
    "vol_expansion_rank_sector", # AUC 0.641  higher->pos
    "nearest_support_dist",      # AUC 0.641  higher->pos
    "vol_expansion_rank",        # AUC 0.640  higher->pos
    "sharpe_20d_rank",           # AUC 0.637  higher->pos
    "beta_nifty_60d",            # AUC 0.637  higher->pos
    "ret_20d_vol_scaled_rank",   # AUC 0.636  higher->pos
    "range_compression",         # AUC 0.631  higher->pos
    "pcr_oi_rank_60d",           # AUC 0.631  higher->pos
    "avg_trade_value",           # AUC 0.630  higher->pos
]


# ── dn_5_xl explicit feature list ───────────────────────────────────────────────
# Derived from the 35.80% baseline model (tabm_20260521_151718, 174 of its 187
# feat_cols that are still computed) + all new features added since that run.
# Using an explicit list avoids inadvertent pruning: the neural net treats
# "bullish" features as contrarian signals — do not try to restrict to DN-only.
# Features listed here that are absent from features.parquet are silently ignored.
_F_DN_5_XL: list[str] = [
    # ── 35.80% baseline — 174 features still in parquet ──────────────────────
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "above_200ma", "dist_52w_high", "dist_52w_low", "rsi_14", "macd_hist",
    "close_position", "atr_14", "hv_20", "vol_expansion", "vol_ratio_5d",
    "vol_ratio_20d", "amihud_illiquidity", "delivery_pct", "delivery_ratio_20d",
    "delivery_streak", "avg_trade_value_ratio_20d", "avg_trade_value_zscore_60d",
    "bb_width", "bb_squeeze", "squeeze_rank", "atr_compression", "days_in_squeeze",
    "squeeze_breakout", "breakout_vol_confirm", "breakout_quality", "adl_divergence",
    "obv_slope_10d", "hi52_breakout", "tight_range_10d", "price_acceleration",
    "gap_down_count_20d", "consecutive_red_days", "days_since_20d_high",
    "close_in_range_5d", "dist_above_50ma", "red_day_vol_ratio",
    "ret_1d_vol_scaled", "ret_5d_vol_scaled", "ret_20d_vol_scaled",
    "dist_52w_high_z", "dist_52w_low_z", "dist_above_50ma_z",
    "nearest_resistance_z", "nearest_support_z",
    "resistance_valid_touches", "support_valid_touches",
    "resistance_zone_strength", "support_zone_strength",
    "consolidating_at_resistance", "consolidating_at_support",
    "zone_box_width_pct", "weeks_in_box", "zone_breakout", "zone_breakdown",
    "w_body_pct", "w_body_ratio", "w_upper_wick", "w_lower_wick",
    "w_close_pos", "w_range_pct", "w_gap", "w_body_expand", "w_range_expand",
    "w_inside_bar", "w_outside_bar", "w_bull_engulf", "w_bear_engulf",
    "w_is_hammer", "w_is_shooting_star", "w_is_doji", "w_is_marubozu",
    "fut_oi_chg_1d", "fut_oi_chg_5d",
    "long_buildup", "short_buildup", "short_covering", "long_unwinding",
    "basis_pct", "basis_chg_5d", "pcr_oi", "pcr_vol", "pcr_oi_rank_60d",
    "pcr_chg_5d", "fut_vol_oi_ratio", "opt_oi_ratio_20d",
    "max_pain_dist", "dist_call_wall", "dist_put_wall", "days_to_expiry",
    "fii_cash_net_5d", "fii_cash_zscore", "fii_cash_streak",
    "fii_idx_fut_net_chg", "smart_vs_retail", "fii_cash_net_30d",
    "fii_cash_acceleration", "fii_cash_reversal_flag",
    "fii_extreme_outflow", "fii_extreme_inflow", "fii_buy_sell_ratio",
    "fii_sector_flow_30d", "fii_sector_flow_90d", "fii_sector_flow_zscore",
    "fii_sector_flow_pct_aum", "fii_sector_flow_streak",
    "fii_sector_flow_acceleration", "fii_sector_aum_pct_change_90d",
    "fii_sector_rotation_rank", "fii_sector_breadth_pos",
    "days_to_next_earnings", "days_since_last_earnings", "last_eps_surprise_pct",
    "eps_surprise_3q_avg", "days_to_ex_div", "days_since_ex_div",
    "stock_dd_52w", "stock_recovery_52w",
    "higher_high", "higher_low", "lower_high", "lower_low",
    "nifty_ret_5d", "nifty_ret_20d", "nifty_above_200ma",
    "nifty_dist_52w_high", "nifty_dist_52w_low",
    "nifty_momentum_accel", "nifty_hv_20", "vix_level", "vix_rank_252d",
    "vix_chg_5d", "regime_duration_days", "regime_changed_5d", "is_expiry_week",
    "sector_ret_5d", "rel_strength_sector",
    "mega_breakout", "bull_confluence", "earnings_vol_setup",
    "double_squeeze", "zone_energy", "fii_nifty_divergence", "quiet_accum",
    "fii_accum_prob", "net_swing_lag1", "path_asym_lag1",
    "net_swing_lag2", "net_swing_roll20",
    "ret_5d_rank", "ret_20d_rank", "ret_20d_vol_scaled_rank",
    "vol_expansion_rank", "breakout_quality_rank", "vol_ratio_20d_rank",
    "delivery_ratio_20d_rank", "pcr_oi_rank", "atr_compression_rank",
    "dist_52w_high_rank", "dist_above_50ma_rank", "fut_oi_chg_5d_rank",
    "avg_trade_value_ratio_20d_rank",
    "ret_5d_rank_sector", "ret_20d_rank_sector",
    "breakout_quality_rank_sector", "vol_expansion_rank_sector",
    "range_vs_vol_implied", "vol_recovery_score", "dip_vol_spike_5d",
    "fii_macro_stock_divergence",
    # ── 13 baseline features that were accidentally excluded ─────────────────
    # All still computed in features.py and present in parquet.
    "avg_trade_value", "avg_trade_value_5d_mom", "avg_trade_value_rank",
    "breakout_delivery_confirm", "fii_sector_flow_14d",
    "is_ex_div_week", "is_pre_earnings",
    "nearest_resistance_dist", "nearest_support_dist",
    "polarity_flip", "range_compression", "vix_expanding", "vol_ratio_60d",
    # ── New since baseline: options structure ─────────────────────────────────
    "put_oi_pct", "opt_total_oi_chg_5d_rank", "annualized_basis", "wall_compression",
    # ── New since baseline: ATM IV ────────────────────────────────────────────
    "atm_iv", "atm_ce_iv", "atm_pe_iv",
    "put_call_iv_skew", "atm_iv_rank_252d", "put_call_iv_skew_rank_60d",
    # ── New since baseline: participant OI Tier 1 ─────────────────────────────
    "fii_stk_fut_net", "fii_stk_fut_net_chg_5d",
    "client_stk_fut_net", "fii_vs_client_stk",
    # ── New since baseline: participant OI Tier 2 ─────────────────────────────
    "fii_put_long_stk", "fii_put_long_stk_chg_5d",
    "client_put_short_stk", "pro_stk_fut_net",
    # ── New since baseline: FII synthetic positioning ─────────────────────────
    "fii_stk_put_call_oi_ratio", "fii_stk_net_opt_dir",
    "fii_stk_net_opt_dir_chg_5d", "client_stk_call_put_net",
    # ── New since baseline: DN_XL compounds ──────────────────────────────────
    "stretch_beta", "stretch_beta_rank",
    "dn_smart_short", "dn_basis_stretch", "put_activity_stretch",
    "iv_vs_hv", "dn_oi_crowded", "dn_exhaustion", "dn_macro_stock_timing",
    "fii_put_rush_on_stretch",
    # ── New since baseline: bearish flow & pattern features ───────────────────
    "put_call_vol_ratio", "put_call_vol_rank_60d", "put_call_vol_ratio_rank",
    "fut_oi_z_60d", "fut_oi_z_60d_rank", "basis_rank_60d",
    "consecutive_green_days", "consecutive_green_days_rank",
    "rsi_overbought_days", "distribution_days_20d",
    "short_buildup_streak", "short_buildup_streak_rank",
    "long_unwinding_streak", "short_buildup_5d_count", "long_unwind_5d_count",
    "dist_from_bb_upper", "dist_from_bb_upper_rank",
    "zone_breakdown_ffill", "resistance_zone_age_weeks",
    "beta_nifty_20d", "beta_nifty_20d_rank",
    "big_eps_miss", "eps_miss_streak",
    # ── New since baseline: block/bulk + flows ────────────────────────────────
    # block_sell_qty_5d / block_net_qty_5d EXCLUDED: raw share counts (up to
    #   382M) with IQR≈0 → catastrophic RobustScaler scaling.
    # block_sell_val_5d INCLUDED: already in crores (value_cr source); sparse
    #   p99=0 is handled by the p99.9 fallback winsorization in tabm.py.
    "block_sell_val_5d", "block_deal_flag_5d",
    "bulk_sell_val_5d", "bulk_sell_flag_5d",
    "dii_cash_net_5d", "days_since_vol_surge",
    # ── New since baseline: regime / structure ────────────────────────────────
    "market_phase", "stock_phase",
    "sharpe_20d", "win_rate_20d",
]


# ── LightGBM dn_5_xl — data-driven feature list ─────────────────────────────────
# Built from feature importance (gain) of the 16.35% prec@10% LGBM model.
# 76 of 151 features had ZERO gain and are excluded, including:
#   - ALL DN_XL compound signals (dn_exhaustion, stretch_beta, etc. = 0%)
#   - All traditional bearish signals (short_buildup, zone_breakdown, etc. = 0%)
#   - Weekly patterns, PCR, basis metrics, consecutive day counts = 0%
# The actual LGBM signal is: stock stretched above 50MA + retail long + FII short
# in a high-vol macro regime. Macro + participant positioning dominates.
#
# Feature gain tiers (from importance analysis):
#   Tier 1 (top 24, covers 80% gain): macro regime + overbought + positioning
#   Tier 2 (ranks 25–75, covers remaining 20%): supporting context
#   Excluded: 76 features with zero gain
_F_LGBM_DN_5_XL: list[str] = [
    # ── Tier 1: 80% of signal (top 24 features) ──────────────────────────────
    # Stock overbought
    "dist_above_50ma_z",          # #1  7.12% — primary stretch signal
    "dist_above_50ma",            # #14 2.51% — raw distance (corroborates z)

    # Implied vs realised vol divergence
    "iv_vs_hv",                   # #2  6.42% — options pricing in tail risk

    # Macro volatility regime
    "nifty_hv_20",                # #3  5.12% — market vol environment
    "vix_chg_5d",                 # #8  3.78% — vol direction
    "vix_rank_252d",              # #15 2.36% — vol percentile context
    "nifty_momentum_accel",       # #16 2.18% — market momentum turning
    "nifty_ret_20d",              # #12 2.88% — 20d market return
    "nifty_dist_52w_high",        # #22 1.92% — market stretch
    "nifty_above_200ma",          # #24 1.59% — bull/bear market regime
    "nifty_ret_5d",               # #34 0.87% — short-term market direction
    "regime_duration_days",       # #4  5.11% — how long regime has persisted

    # Participant positioning divergence (the core edge)
    "client_stk_fut_net",         # #5  4.93% — retail long (contrarian short signal)
    "fii_vs_client_stk",          # #13 2.76% — FII/retail divergence ratio
    "client_put_short_stk",       # #18 2.06% — retail complacency (selling puts)
    "client_stk_call_put_net",    # #20 1.98% — retail net call/put bias
    "fii_stk_fut_net",            # #19 1.99% — FII stock futures level
    "fii_stk_fut_net_chg_5d",     # #17 2.14% — FII reducing longs / adding shorts
    "fii_put_long_stk",           # #10 3.16% — FII absolute put accumulation
    "fii_put_long_stk_chg_5d",    # #21 1.95% — FII accelerating put buying
    "fii_stk_put_call_oi_ratio",  # #23 1.70% — FII net bearish options bias
    "pro_stk_fut_net",            # #11 2.97% — proprietary desk positioning

    # FII cash medium-term flow
    "fii_cash_net_30d",           # #7  4.64% — medium-term FII trend

    # Compression + return
    "tight_range_10d",            # #6  4.72% — consolidation before breakdown
    "ret_5d_vol_scaled",          # #9  3.16% — recent vol-adjusted return

    # ── Tier 2: remaining 20% of signal (ranks 25–75) ────────────────────────
    # Price position / structure
    "dist_52w_low",               # #25 — bounce distance (crowded longs floor)
    "dist_52w_high_z",            # #26 — stretched from highs
    "dist_from_bb_upper_rank",    # #27 — Bollinger upper band relative rank
    "dist_from_bb_upper",         # #44 — Bollinger upper band distance
    "dist_52w_high",              # #49 — raw distance from 52w high
    "dist_52w_high_rank",         # #68 — cross-sectional stretch rank
    "nearest_resistance_z",       # #39 — proximity to resistance

    # Volatility structure
    "hv_20",                      # #31 — stock realized vol
    "vol_expansion",              # #33 — volume expansion
    "vol_ratio_20d",              # #35 — vol vs 20d average
    "vol_ratio_20d_rank",         # #52 — cross-sectional vol rank
    "bb_squeeze",                 # #38 — Bollinger squeeze (pre-breakdown)
    "squeeze_rank",               # #41 — rolling percentile of squeeze
    "atr_14",                     # #63 — ATR (position sizing context)

    # FII cash supporting signals
    "fii_cash_streak",            # #28 — FII flow consistency
    "fii_cash_net_5d",            # #32 — short-term FII flow
    "fii_cash_zscore",            # #42 — normalised FII flow
    "fii_cash_reversal_flag",     # #45 — FII turning point signal
    "fii_cash_acceleration",      # #69 — FII flow momentum
    "fii_extreme_outflow",        # #72 — extreme FII selling
    "fii_idx_fut_net_chg",        # #71 — index futures net change
    "fii_stk_net_opt_dir",        # #30 — FII net synthetic short
    "fii_stk_net_opt_dir_chg_5d", # #36 — change in FII synthetic short

    # Returns / momentum
    "ret_20d",                    # #29 — 20d return (momentum context)
    "ret_20d_vol_scaled",         # #53 — vol-adjusted 20d return
    "ret_5d",                     # #50 — 5d return
    "ret_1d",                     # #66 — 1d return
    "ret_5d_rank",                # #60 — cross-sectional 5d rank
    "ret_5d_rank_sector",         # #61 — sector-relative 5d rank

    # Options / derivatives (where they have non-zero gain)
    "atm_iv",                     # #55 — ATM implied vol level
    "atm_iv_rank_252d",           # #67 — ATM IV vs 1yr history
    "put_oi_pct",                 # #58 — put OI as % of total
    "opt_oi_ratio_20d",           # #56 — options OI trend
    "dist_call_wall",             # #47 — distance to call wall
    "wall_compression",           # #70 — call/put wall gap
    "days_to_expiry",             # #40 — expiry timing (options signals need this)
    "is_expiry_week",             # #37 — binary expiry flag
    "fut_oi_chg_5d",              # #48 — futures OI change

    # Macro / regime supporting
    "market_phase",               # #43 — market regime category
    "regime_changed_5d",          # #62 — recent regime transition
    "smart_vs_retail",            # #57 — institutional vs retail flow ratio

    # Earnings calendar
    "days_to_next_earnings",      # #54 — catalyst risk ahead
    "days_since_last_earnings",   # #46 — time since last event

    # Misc non-zero
    "gap_down_count_20d",         # #64 — gap-down frequency
    "close_position",             # #65 — close position in day's range
    "net_swing_lag1",             # #59 — lagged path quality
    "days_since_vol_surge",       # #73 — time since last vol spike
    "consecutive_green_days_rank",# #74 — cross-sectional green streak rank
    "earnings_vol_setup",         # #75 — earnings + vol compound
    "stretch_beta_rank",          # #51 — cross-sectional overbought×beta rank
]


TARGET_FEATURE_COLS: dict[str, list[str] | None] = {
    "up_3":    _compose_features(_F_SHARED, _F_UP),
    "dn_3":    _compose_features(_F_SHARED, _F_DN),
    "up_5":    _compose_features(_F_SHARED, _F_UP),
    "dn_5":    _compose_features(_F_SHARED, _F_DN),
    "up_5_xl": _compose_features(_F_SHARED, _F_UP, _F_UP_XL),
    "dn_5_xl": _compose_features(_F_SHARED, _F_DN, _F_DN_XL),   # fallback / ae_mlp
    # Tiered targets — same directional feature groups as their counterparts
    "up_liq":  _compose_features(_F_SHARED, _F_UP),
    "dn_liq":  _compose_features(_F_SHARED, _F_DN),
    "up_rest": _compose_features(_F_SHARED, _F_UP),
    "dn_rest": _compose_features(_F_SHARED, _F_DN),
    # Overlay: sees full shared + both directional groups so it can detect
    # setups that are dangerous regardless of which way the base model fired
    "bad_up":  _compose_features(_F_SHARED, _F_UP, _F_DN),
    "bad_dn":  _compose_features(_F_SHARED, _F_UP, _F_DN),
    # Clean directional models
    # Liquid: original broad composition — path-asymmetry features carry strong
    # conditional signal that individual-AUC ranking underestimates.
    "clean_up_5_liq":  _compose_features(_F_SHARED, _F_UP, _F_PATH_ASYM_BULL,
                                          _F_SHAP_CROSS_LIQ_BULL),
    "clean_dn_5_liq":  _compose_features(_F_SHARED, _F_DN, _F_PATH_ASYM),
    # Rest: data-driven top-40 from analysis/rank_clean_features.py (AUC vs neutral)
    "clean_up_5_rest": _F_CLEAN_REST_BULL,
    "clean_dn_5_rest": _F_CLEAN_REST_BEAR,
}

# Per-model-type feature overrides.
# Lookup order in train.py: MODEL_FEATURE_OVERRIDES[model_type][target]
#   → falls back to TARGET_FEATURE_COLS[target] if no override defined.
#
# Why separate lists per model type:
#   lgbm  — needs directionally pure bearish features; cannot learn contrarian
#            signals from bullish features with 15-leaf trees.
#   tabm  — benefits from ALL features (bullish, bearish, shared) because the
#            neural net learns globally that e.g. high bull_confluence is a
#            contrarian bearish signal for dn_5_xl.
# Ablation: 187 baseline features only (174 still-in-parquet + 13 restored).
# Excludes the 65 features added since the 35.80% baseline (ATM IV, participant OI,
# DN_XL compounds, etc.) to test whether they dilute TabM's signal.
# Per-target curated lists (built from importance analysis after baseline training).
# The 187-feature baseline matches the original tabm_20260521_151718 model exactly
# and was slightly better than the 252-feature version (21.31% vs 19.57% prec@10%).
_F_DN_5_XL_BASELINE_187: list[str] = _F_DN_5_XL[:187]

# MODEL_FEATURE_OVERRIDES[model_type][target] is consulted FIRST.
# If absent, falls back to TARGET_FEATURE_COLS[target] (group composition).
#
# To curate features for a new (model, target) pair:
#   1. Train baseline: python -m pipeline.train --model <m> --targets <t>
#   2. Importance:     python debug_<m>_importance.py <t>
#   3. Add curated list as _F_<M>_<TARGET> here, populate dict entry below
#   4. Re-train + walk-forward verify
MODEL_FEATURE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "lgbm": {
        # dn_5_xl: NO OVERRIDE — uses full _F_SHARED+_F_DN+_F_DN_XL group composition.
        # Comparison verdict (debug_lgbm_compare.py):
        #   curated 75 features:  val 1.44x / test 0.89x   → overfit val
        #   full 155 features:    val 0.96x / test 1.26x   → generalizes better
        # Lesson: feature curation via gain importance overfits.
        # The 76 "zero-gain" features act as implicit regularization.
        # DO NOT add an override here for dn_5_xl unless multi-window-curated.
        # up_3 / dn_3 / up_5 / dn_5 / up_5_xl: TODO — multi-window curation
    },
    "tabm": {
        # dn_5_xl: 187 baseline features (slightly better than 252)
        "dn_5_xl": _compose_features(_F_DN_5_XL_BASELINE_187),
        # up_3 / dn_3 / up_5 / dn_5 / up_5_xl: TODO — run permutation importance + curate
    },
}

# ── Experiment / search ─────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "esn-search"

# Composite score weights (used by experiment.py objective function).
# XL targets get higher weight — they are the STRONG-tier gate and what
# matters most for signal quality.
SEARCH_AP_WEIGHTS: dict[str, float] = {
    "up_3": 0.15, "dn_3": 0.15,
    "up_5": 0.15, "dn_5": 0.15,
    "up_5_xl": 0.20, "dn_5_xl": 0.20,
}

# Label configs to sweep in the outer search loop
SEARCH_LABEL_CONFIGS = [
    {"target_rate_base": 0.18, "target_rate_xl": 0.09},
    {"target_rate_base": 0.20, "target_rate_xl": 0.10},
    {"target_rate_base": 0.22, "target_rate_xl": 0.12},
]

# ── Bucket assignment ────────────────────────────────────────────────────────────
# A stock gets a directional bucket only when BOTH conditions hold:
#   1. Z-score ≥ Z_BUCKET_THRESH  (stock ranks unusually high in today's universe)
#   2. Prob ≥ ABS_PROB_FLOOR_MULT × model base rate  (absolute strength non-trivial)
# STRONG tier additionally requires the XL model to fire.
Z_BUCKET_THRESH     = 0.70   # ≈ 76th pct of universe z-score
ABS_PROB_FLOOR_MULT = 1.25   # require prob ≥ 1.25× base rate

# ── Zone detection ──────────────────────────────────────────────────────────────
ZONE_LOOKBACK_WEEKS     = 9999  # effectively all available history — see zones.py
ZONE_SWING_BARS         = 3
ZONE_CLUSTER_PCT        = 0.02
ZONE_RETREAT_PCT        = 0.05
ZONE_CONSOLIDATION_DAYS = 5

# ── Active prediction universe ───────────────────────────────────────────────────
# TARGETS / INDEX_TARGETS: legacy must-include lists for build_active().
# Hand-curated universe is now in PREDICT_UNIVERSE_FILE; these are kept as
# empty stubs so universe.py imports don't break.
TARGETS       : list[str] = []
INDEX_TARGETS : list[str] = []

PREDICT_UNIVERSE_FILE = ROOT / "Universe"
LIQUID_TIER_SIZE      = 30
PREDICT_TICKERS = [
    "AXISBANK","BANDHANBNK","BANKBARODA","BANKNIFTY","CHOLAFIN","FINNIFTY",
    "HDFCBANK","ICICIBANK","KOTAKBANK","MANAPPURAM","MUTHOOTFIN","PNB","SBIN",
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","BAJAJ-AUTO","BAJAJFINSV",
    "BAJFINANCE","BHARTIARTL","CIPLA","HCLTECH","HEROMOTOCO","HINDUNILVR","INFY",
    "ITC","M&M","MARUTI","NESTLEIND","NIFTY","RELIANCE","SUNPHARMA","TMPV",
    "TATAPOWER","TATASTEEL","TCS","TECHM","TVSMOTOR","WIPRO",
]
INDEX_TICKERS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

# ── Sector / ticker mappings ─────────────────────────────────────────────────────
SECTOR_INDICES = {
    "banks":    "Nifty Bank",    "auto":    "Nifty Auto",
    "it":       "Nifty IT",      "pharma":  "Nifty Pharma",
    "fmcg":     "Nifty FMCG",   "metal":   "Nifty Metal",
    "realty":   "Nifty Realty",  "energy":  "Nifty Energy",
    "finserv":  "Nifty Financial Services",
    "media":    "Nifty Media",   "psu_bank":"Nifty PSU Bank",
    "pvt_bank": "Nifty Private Bank",
}

TICKER_SECTOR = {
    "AXISBANK":"pvt_bank","BANDHANBNK":"pvt_bank","HDFCBANK":"pvt_bank",
    "ICICIBANK":"pvt_bank","KOTAKBANK":"pvt_bank","BANKBARODA":"psu_bank",
    "PNB":"psu_bank","SBIN":"psu_bank","CHOLAFIN":"finserv","BAJFINANCE":"finserv",
    "BAJAJFINSV":"finserv","MUTHOOTFIN":"finserv","MANAPPURAM":"finserv",
    "BAJAJ-AUTO":"auto","HEROMOTOCO":"auto","M&M":"auto","MARUTI":"auto",
    "TMPV":"auto","TVSMOTOR":"auto",
    "HCLTECH":"it","INFY":"it","TCS":"it","TECHM":"it","WIPRO":"it",
    "CIPLA":"pharma","SUNPHARMA":"pharma","APOLLOHOSP":"pharma",
    "HINDUNILVR":"fmcg","ITC":"fmcg","NESTLEIND":"fmcg","ASIANPAINT":"fmcg",
    "TATASTEEL":"metal","TATAPOWER":"energy","RELIANCE":"energy",
    "BHARTIARTL":"media","ADANIENT":"energy","ADANIPORTS":"energy",
}

TICKER_NSDL_SECTOR = {
    "HDFCBANK":"Financial Services","ICICIBANK":"Financial Services",
    "AXISBANK":"Financial Services","KOTAKBANK":"Financial Services",
    "BANDHANBNK":"Financial Services","INDUSINDBK":"Financial Services",
    "IDFCFIRSTB":"Financial Services","YESBANK":"Financial Services",
    "RBLBANK":"Financial Services","SBIN":"Financial Services",
    "BANKBARODA":"Financial Services","PNB":"Financial Services",
    "CANBK":"Financial Services","BAJFINANCE":"Financial Services",
    "BAJAJFINSV":"Financial Services","CHOLAFIN":"Financial Services",
    "MUTHOOTFIN":"Financial Services","MANAPPURAM":"Financial Services",
    "SHRIRAMFIN":"Financial Services","JIOFIN":"Financial Services",
    "HDFCLIFE":"Financial Services","SBILIFE":"Financial Services",
    "LICHSGFIN":"Financial Services","SBICARD":"Financial Services",
    "PFC":"Financial Services","RECLTD":"Financial Services",
    "TCS":"Information Technology","INFY":"Information Technology",
    "HCLTECH":"Information Technology","WIPRO":"Information Technology",
    "TECHM":"Information Technology","COFORGE":"Information Technology",
    "PERSISTENT":"Information Technology","LTM":"Information Technology",
    "NAUKRI":"Information Technology",
    "SUNPHARMA":"Healthcare","CIPLA":"Healthcare","APOLLOHOSP":"Healthcare",
    "AUROPHARMA":"Healthcare","DIVISLAB":"Healthcare",
    "GLENMARK":"Healthcare","MAXHEALTH":"Healthcare",
    "MARUTI":"Automobile and Auto Components",
    "TMPV":"Automobile and Auto Components",
    "M&M":"Automobile and Auto Components",
    "BAJAJ-AUTO":"Automobile and Auto Components",
    "HEROMOTOCO":"Automobile and Auto Components",
    "TVSMOTOR":"Automobile and Auto Components",
    "EICHERMOT":"Automobile and Auto Components",
    "ASHOKLEY":"Automobile and Auto Components",
    "HYUNDAI":"Automobile and Auto Components",
    "HINDUNILVR":"Fast Moving Consumer Goods",
    "ITC":"Fast Moving Consumer Goods","NESTLEIND":"Fast Moving Consumer Goods",
    "COLPAL":"Fast Moving Consumer Goods","UNITDSPR":"Fast Moving Consumer Goods",
    "VBL":"Fast Moving Consumer Goods","PATANJALI":"Fast Moving Consumer Goods",
    "ASIANPAINT":"Fast Moving Consumer Goods",
    "RELIANCE":"Oil, Gas & Consumable Fuels","ONGC":"Oil, Gas & Consumable Fuels",
    "BPCL":"Oil, Gas & Consumable Fuels","COALINDIA":"Oil, Gas & Consumable Fuels",
    "TATASTEEL":"Metals & Mining","JSWSTEEL":"Metals & Mining",
    "HINDALCO":"Metals & Mining","VEDL":"Metals & Mining",
    "HINDZINC":"Metals & Mining","SAIL":"Metals & Mining","NMDC":"Metals & Mining",
    "TATAPOWER":"Power","NTPC":"Power","POWERGRID":"Power",
    "ADANIGREEN":"Power","JSWENERGY":"Power","ADANIENSOL":"Power",
    "SUZLON":"Power","WAAREEENER":"Power","INOXWIND":"Power",
    "LT":"Construction","RVNL":"Construction","ADANIPORTS":"Construction",
    "ULTRACEMCO":"Construction Materials","AMBUJACEM":"Construction Materials",
    "GRASIM":"Construction Materials","ASTRAL":"Construction Materials",
    "DLF":"Realty","GODREJPROP":"Realty","OBEROIRLTY":"Realty","LODHA":"Realty",
    "BEL":"Capital Goods","HAL":"Capital Goods",
    "BHEL":"Capital Goods","KAYNES":"Capital Goods",
    "TITAN":"Consumer Durables","DIXON":"Consumer Durables",
    "CROMPTON":"Consumer Durables","VOLTAS":"Consumer Durables",
    "ETERNAL":"Consumer Services","SWIGGY":"Consumer Services",
    "JUBLFOOD":"Consumer Services","TRENT":"Consumer Services",
    "DMART":"Consumer Services","INDIGO":"Consumer Services",
    "BHARTIARTL":"Telecommunication","IDEA":"Telecommunication",
    "INDUSTOWER":"Telecommunication",
    "ADANIENT":"Diversified",
}

SECTOR_INTERNAL_TO_NSDL = {
    "pvt_bank": "Financial Services",
    "psu_bank": "Financial Services",
    "finserv":  "Financial Services",
    "auto":     "Automobile and Auto Components",
    "it":       "Information Technology",
    "pharma":   "Healthcare",
    "fmcg":     "Fast Moving Consumer Goods",
    "metal":    "Metals & Mining",
    "energy":   "Oil, Gas & Consumable Fuels",
    "media":    "Media, Entertainment & Publication",
}

FII_SECTOR_PUBLICATION_LAG_DAYS = 10
