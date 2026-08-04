# macro_direction_v1 — does cross-asset macro add next-day DIRECTION edge?

**Hypothesis (from practitioner advice):** a direction model fed only price-derived features recycles the
same signal. Indian equities react to external drivers the price chart can't see — **USD/INR, the previous
US session, India VIX, crude** — so adding these may lift directional accuracy above the ~50–52% coin flip
we measured in `direction_research_v1`.

**Honest prior:** macro mostly moves the *whole market* (beta), so it can nudge **index** direction but adds
little **cross-sectional** "which name, which way" signal — and daily index direction is itself ~52%. This
experiment exists to put a *number* on that, under correct validation, not to assume the answer.

## On DJIA vs S&P vs Nasdaq
DJIA is price-weighted across 30 names — a narrow, idiosyncratic gauge. S&P 500 is the broad risk-on/off
barometer global markets track; Nasdaq is the right one for Indian IT (ADR-linked). Rather than pre-judge,
**all three are included** (`dji_overnight`, `spx_overnight`, `ndx_overnight`) and `results/feature_importance.csv`
reports their gain share — the data decides which (if any) matters.

## Data sources
| Series | Source | How |
|---|---|---|
| India VIX, Nifty IT | **local** `data/silver/indices.parquet` | read-only (no fetch) |
| USD/INR | yfinance `INR=X` | `fetch_macro.py` |
| US indices (EOD close) | yfinance `^GSPC`, `^IXIC`, `^DJI` | `fetch_macro.py` |
| Crude / Dollar index | yfinance `BZ=F` (Brent), `CL=F` (WTI), `DX-Y.NYB` | `fetch_macro.py` |

## Leak-safety (the part that decides validity)
- Predicting India move **d → d+1** from the info set available at **d+1 pre-open**.
- USD/INR, Brent, WTI, DXY, India VIX: EOD of day **d** (known by d evening) → attached to date d.
- US indices: the overnight session ending the morning of d+1 = US close on calendar date **d**
  (prints ~02:00 IST d+1, before NSE opens) → attached to date d. Set `US_LAG=1` in `macro.py` for an
  ultra-conservative variant that only uses US closes already known at India EOD d.
- Holiday gaps: each series reindexed onto NSE trading dates, forward-filled ≤2 days, **never backfilled**.
- Features are **returns/changes, not levels** (except VIX level, which is stationary-ish).
- Validation: **purged + embargoed** quarterly walk-forward (drop training rows whose label horizon
  overlaps the eval quarter, plus a 5-day embargo). Eval truth is the **full-set** realized sign, so the
  headline isn't inflated by the training-time dead-band.

## Run (from the K3 root, in a terminal — heavy / external, not auto-run)
```bash
# 1) one-time external EOD fetch (needs internet) -> data/macro_raw.parquet
python experiments/macro_direction_v1/fetch_macro.py

# 2) the experiment (GPU XGBoost, purged walk-forward)
set PYTHONPATH=src
python experiments/macro_direction_v1/run_experiment.py
```

## Outputs (`results/`)
- `metrics.json` — pooled + per-quarter acc/AUC for `baseline_price_only` vs `price+macro`, the macro lift
  (ΔAUC, Δacc), and high-conviction (top-decile) precision.
- `feature_importance.csv` — gain share of every feature (answers the DJIA question directly).

**Verdict guide:** ΔAUC < ~0.01 *and* confident-call precision ≈ base rate ⇒ macro adds **no tradeable
direction edge** (confirms keeping the book direction-agnostic). A durable ΔAUC ≥ ~0.02 that holds across
quarters ⇒ worth a deeper look.

## PROD safety
This folder reads only the shared market-data table (`load_market_data`) and the local silver indices,
and the universe list from `locks/prod_largemove_v2/universe_groups.json` (read-only). It **never imports
`koscine3.largemove` and never writes outside this folder**. Verify PROD intact any time:
`python experiments/clone_prod.py --verify` and `python experiments/freeze_v2_prod.py --verify`.
