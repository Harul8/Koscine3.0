# option_gain_study_v1 — where do large option gains come from?

Descriptive study (not a model) on **real option bhavcopy**, to understand which option trades produce large
premium gains and what characterizes them — stock, IV, strike (moneyness), DTE, move, and exit timing.

## Method
For each (stock, entry day d) in the A/B universe (tradeable: close ≥ ₹100, atm_iv present), buy **CALL & PUT**
across an OTM ladder **ATM, ATM+1% … ATM+5%, +7%, +10%**, **enter at the OPEN of day d** (matches the 7 AM
pre-open refresh → trade the open), hold **5 trading days**. Record per trade:
- `high_ratio` = max(option high over the 5d window) / entry_open  — the **peak** (best exit)
- `close_ratio` = option close at day 5 / entry_open  — **held**
- strike_label, otm_pct, dte, entry_open, atm_iv (entry), stock_move, peak_day.

Pennies dropped (entry premium ≥ ₹2). Streaming bhavcopy (each date loads once). 2024-01 → 2026-06.

## Run
```bash
set PYTHONPATH=src
python experiments/option_gain_study_v1/option_gain_study.py            # -> results/option_gain_trades.csv (heavy)
python experiments/option_gain_study_v1/option_gain_analyze.py          # -> the insights (cheap)
```

## Questions answered (option_gain_analyze.py)
1. Gain by strike ladder (peak & held multiples, P(≥2/3/5x)) — which moneyness pays.
2. Best option per stock-day — the convexity actually available.
3. **Top stocks** by large-gain frequency — which 30-40 names give the big gains.
4. What distinguishes big winners (≥3x): IV, move, DTE, strike, peak day.
5. IV regime → gain (are cheap-IV options the bigger multiples?).
6. DTE → gain (near-expiry gamma).
7. Leverage: peak multiple per 1% favorable move, by strike.

Read-only; PROD untouched. Findings in `FINDINGS.md`.
