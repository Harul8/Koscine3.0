# megacap_direction_v1 — 1-3 day direction for top mega-caps, Nifty & basket; regrouping (COMPLETE)

Mandate: predict 1-3 day movement in the top-10 mega-caps + Nifty/basket; regroup A/B by behaviour. Contained;
no PROD touch.

## Phase 1 — characterization (characterize.py)
- **1-day mega-cap direction ≈ RANDOM**: return autocorr ret_1d→fwd1 ≈ 0.00 every period (pre +0.012, 2025 −0.022,
  2026 +0.002). Predictability BUILDS with horizon → 2026: 2d +0.033, 3d +0.033, **5d +0.107** (momentum, the regime).
- **1-day features = mean-reversion** (2026 IC): breadth −0.078, mkt_adv −0.077, gap −0.057 (fade strength); pcr_oi
  +0.054, skew +0.044 (contrarian). **3-day = momentum**: nifty_ret_5d +0.105, ret_5d +0.086, + contrarian pcr/skew.
- **Aggregate is more predictable than single names**: Nifty has a STABLE ~+0.07 one-day momentum autocorr
  (pre +0.076, 2025 +0.069) vs ~0 for individuals (index positive autocorrelation / lead-lag).
- 2026 H1 tape weak for mega-caps: HDFCBANK up only 38% of days, RELIANCE 43.5%, TCS/INFY 41% (FII outflows).

## Phase 3 inputs — FII flow (the missing driver)
- `silver/fii_dii_cash.parquet`: daily FII net cash 2010→2026-06-19 (complete). **The dominant 2026 driver**
  (web: FIIs net sellers all 2026 on valuations/currency; DIIs cushion). Was NOT in the feature panel → added.
  (DII NaN in this source; per-stock FII trades exist but stale at 2025-03.) FII flipped to selling 06-19 (−1807cr).

## Phase 4 — regrouping (regroup.py)
- Data-driven (KMeans on vol/IV/beta/move5) beats mcap/turnover: **Cluster 0 (n=38, 14 of 15 top mega-caps)** =
  low vol/IV/beta, small moves, **direction unpredictable (mom-IC 0.013)** → AGNOSTIC. **Cluster 1 (n=27)** = high
  beta(1.18)/IV/move, **direction predictable (mom-IC 0.057)** → LEANABLE. k=3 isolates **8 "wild movers"**
  (mom-IC 0.093, biggest moves, highest IV/cost).
- **Direction is strongly SECTORAL/heterogeneous in 2026** (per-stock mom-IC): banks SBIN +0.37, ADANIPORTS +0.35,
  LT +0.25 (momentum) vs RELIANCE −0.13, BHARTIARTL −0.24, M&M −0.23, COALINDIA −0.26, PAYTM −0.31 (mean-revert).
  Web confirms: banks OVERWEIGHT/momentum (RBI hike, Bank Nifty highs), IT underperforming since Feb-2026,
  energy choppy (crude/Hormuz). → a blanket lean averages these out; need sector/factor features (panel has them).

## Phase 2 — models (models.py): individual top-15 direction = COIN FLIP
Expanding monthly WF, full 145 feats, eval top-15. 2026 AUC/hit/IC: 1d 0.482/0.471/−0.029, 2d 0.511/0.494/−0.003,
3d 0.510/0.493/+0.005. **Individual mega-cap 1-3d direction is unpredictable** (≈0.50 every horizon).

## Phase 3 — aggregate + FII (aggregate_fii.py)
Nifty next-day, 2026: BASE AUC 0.502 → **BASE+FII 0.562** (FII lifts 1d); but FII HURTS 2-3d (0.41/0.46). Basket
1d 2026: BASE 0.403 → **BASE+FII 0.508**. `fii_net_20d` & `fii_ratio` in top-10 features. Historically Nifty ~0.54
AUC; **2026 ≈ coin flip**. So: a SMALL 1-day Nifty/basket tilt from FII flow; nothing usable at 2-3d.

## Phase 5 — MAGNITUDE is the predictable part (next_day_outlook.py + latest_snapshot.py)
Next-day move-SIZE rank-IC (eval top-15): ALL +0.191, 2025 +0.210, **2026 +0.152 (model beats atm_iv +0.123)**.
Move size is forecastable; direction is not. Latest snapshot (2026-06-19) expected next-day moves: INFY ±1.84%,
RELIANCE ±1.58, TCS ±1.57, BAJFINANCE ±1.44, … basket avg ±1.28% (vs realized ±1.16%); Nifty ≈ ±0.7-0.8%.
FII flipped to selling 06-19 (−1807cr) after a +2382cr week. Saved next_day_outlook.csv.

## CONCLUSION
- **Direction (1-3d) for individual mega-caps = coin flip** (AUC 0.48-0.51, 2026). Cannot be predicted.
- **Only directional crumbs**: 1-day Nifty tilt from FII flow (AUC→0.56) + sector-momentum tilt (financials up /
  IT down). Both weak; deploy only as a small lean on the high-beta/financial cluster (= existing group-B logic).
- **MAGNITUDE (expected move size) IS predictable** (IC ~0.15-0.21, beats atm_iv in 2026) — and is what options
  need. Deliverable = the next-day expected-move snapshot per name.
- **Regroup**: behaviour split (vol/IV/beta/move) > mcap/turnover → efficient mega-caps trade AGNOSTIC (size by
  expected move); high-beta/financial movers get the small lean. Consistent with [[direction-edge-research]].
