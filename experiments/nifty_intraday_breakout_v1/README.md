# NIFTY intraday breakout v1 — research phase

## Objective and scope

Test whether a direction-agnostic "large move coming" flag is predictable on NIFTY 50
index 5-minute candles, over 30min/1hr/2hr forward horizons — not directional sign. This
repo's own prior research (`mover_v2.py`'s rationale, `experiments/megacap_direction_v1/FINDINGS.md`)
found direction ≈ coin flip at daily horizons more than once; magnitude carried the real
edge. 5-minute bars are noisier per-row than daily bars, so this experiment exists to find
out honestly whether that same magnitude-over-direction pattern survives intraday, before
anything is wired into the API or a frontend tab. This experiment is isolated from
production and does not change any locked model or signal selector.

Intended downstream use, if this clears the bar: timing entries into the options-buying
book (buy a same-day/near-day NIFTY option when a large move looks imminent), eventually
surfaced as an "Index Predictions" tab — not built in this pass.

## Data

- `data/intraday/nifty_5m.parquet` — native 5-min NIFTY 50 index OHLC, 2022-01 → 2026-08
  (~85k bars). Volume/OI are always 0 on the raw index — not usable as features.
- `data/intraday/nifty_option_chain_5m/{expiry}.parquet` — full CE/PE chain at 5-min
  resolution, Oct 2024+ only (expired-contract lookup's hard coverage cutoff). Used for the
  v2 feature variant (ATM IV/straddle premium/OI/PCR), on a matched-window basis against v1.

## Label — direction-agnostic breakout flag

For horizons H ∈ {6, 12, 24} bars (30min / 1hr / 2hr), computed **within the same trading
day only** (a forward window that would spill past 15:30 is left NaN and dropped — this
model is meant to time same-day entries, not carry overnight gap risk into the label):

```
move_mag_H = max(fwd_high_H - entry, entry - fwd_low_H) / entry
```

mirrors `src/koscine3/largemove/mover_v2.py::load_book()`'s `up_move`/`down_move`/`move_mag`
construction, on 5-min OHLC instead of daily. The binary flag is **not** a fixed % threshold —
`labels.py::calibrate_k` derives a data-driven K from a given (training-only) slice,
targeting a ~13% positive base rate (between this repo's `LABEL_TARGET_RATE_XL=0.10` and
`LABEL_TARGET_RATE_BASE=0.20`), same philosophy as `pipeline/labels.py::_calibrate_k`.
`labels.py::apply_threshold` then turns `move_mag` into 0/1 for any slice using that K.

These two steps are deliberately split: **K must be calibrated inside each walk-forward
fold, from that fold's training rows only** (`train.py::walk_forward` does this), or the
label leaks later volatility regimes into earlier folds.

## Features (`feature_`-prefixed, per `intraday_exit_v1`'s leakage convention)

- **v1 price-only** (`features.py`): `session_fraction` (time-of-day), `day_of_week`,
  momentum at 1/3/6/12-bar, realized vol at 6/12/24-bar, gap-from-prior-session-close,
  True-Range-% and a 20-bar ATR-style rolling mean of it, plus a range-compression ratio
  (today's TR vs its own rolling ATR). All rolling/return windows reset at the session
  boundary so an overnight gap never contaminates an intraday vol/momentum estimate — the
  gap itself is captured separately.
- **v2 option-chain context** (`option_features.py`): ATM strike's CE/PE implied vol
  (Black-Scholes via Brent's method, same formula as `api/main.py::_bs_price`/`_implied_vol`,
  copied rather than imported since importing `api.main` would construct the live FastAPI
  app as an import side effect), ATM straddle premium and its momentum, ATM OI and its
  change, and chain-wide PCR (OI). Restricted to the Oct 2024+ window.

## Fair comparison

v2 is evaluated against a v1-price-only baseline **restricted to the same Oct 2024+ window**
(`train.py` builds both from `data/v2_option_chain.parquet`, one run using only its
price `feature_*` columns, one using all of them) — not against v1's full-history numbers.
Otherwise any apparent lift from the option-chain features would be confounded with simply
having more training data.

## Model + evaluation

LightGBM binary classifier per horizon, `average_precision` as the primary metric (this
repo already documented a saddle-collapse failure mode using plain AUC + scale_pos_weight
at low base rates — see `pipeline/config.py`'s `LGBM_CLEAN_PARAMS` comment). Expanding
walk-forward, monthly refit, imitates `experiments/megacap_direction_v1/models.py::wf_index`
(closest existing single-series, non-cross-sectional-panel precedent).

Reported per horizon/variant: AUC, average precision, precision@top-10%, and
**lift-over-base-rate** — the number that actually matters. A model can have a
respectable-looking AUC and still be useless; lift close to 1.0× means no real edge
regardless of how the other numbers look.

## Promotion gate (not attempted in this pass)

Only worth promoting to a live signal (production module + API endpoint + an "Index
Predictions" tab, same shape as the Cash Signals build) if lift-over-base-rate is
meaningfully above 1.0× out-of-sample, holds up across more than one sub-period (not just
one lucky month), and the option-chain variant's improvement (if any) survives the
matched-window comparison. If not, this joins the direction-prediction pile of "looked
plausible on paper, didn't survive out-of-sample" already documented elsewhere in this repo.

## Usage

```
python experiments/nifty_intraday_breakout_v1/dataset.py    # -> data/v1_price_only.parquet, data/v2_option_chain.parquet
python experiments/nifty_intraday_breakout_v1/train.py       # -> results/summary.csv, results/FINDINGS.md
```
