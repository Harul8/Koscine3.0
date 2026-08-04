# macro_direction_v1 — findings

_Status: complete. Verdict: macro adds **no tradeable** next-day direction edge._

## Setup
- Execution: **~7 AM IST t+1 pre-open refresh** using all of the previous day's global data, then trade the
  **t+1 open** → target = next-day **open→close** (`ENTRY="open"`, the capturable return); training drops the
  ±1% dead-band; eval on full-set sign. (The numbers below were the original *close→close* run — see the kill.)
- Validation: purged + embargoed (5d) quarterly walk-forward, 2024Q1–2026Q2; XGBoost (GPU).
- Features: 148 leak-safe native (forward-looking K2 label columns removed by a corr leak-guard) vs.
  native + 18 macro (USD/INR, S&P/Nasdaq/Dow overnight, Brent/WTI, DXY, India VIX, Nifty IT).
- Macro data fetched 2010→2026 at ~100% coverage (India VIX local from 2014, Nifty IT from 2012).

## Results (purged WF, n_eval = 31,867)
| model | pooled acc | AUC | Brier | conf-precision (top10%) |
|---|---|---|---|---|
| baseline_price_only | 0.5127 | 0.5250 | 0.2524 | 0.5556 |
| price+macro | 0.5435 | 0.5693 | 0.2503 | 0.6281 |

Macro lift: **ΔAUC = +0.044, Δacc = +3.1pp** — driven almost entirely by `spx_overnight` (imp 0.039) and
`dji_overnight` (0.039), ~3× every other macro feature (`ndx_overnight` 0.025, all else ~0.011–0.014).
**These are close→close; on the capturable open→close target the lift is ≈ 0 (see kill).**

## The lift is the un-capturable open gap (diag_macro.py)
`close[d]→close[d+1] = gap (close[d]→open[d+1]) + intraday (open[d+1]→close[d+1])`. Correlation of the US
overnight with each India component, eval 2024–26:

| US feature | → close→close | → overnight gap | → open→close (tradeable) |
|---|---|---|---|
| spx_overnight (LAG0) | 0.124 | **0.187** | **−0.008** |
| dji_overnight (LAG0) | 0.127 | **0.199** | **−0.014** |
| ndx_overnight (LAG0) | 0.105 | **0.164** | **−0.011** |
| spx_prevnight (LAG1, known at EOD d) | 0.007 | 0.008 | 0.000 |
| dji_prevnight (LAG1) | 0.002 | 0.008 | −0.005 |

**Execution: a ~7 AM IST t+1 pre-open refresh then trade the t+1 open.** This makes `US_LAG=0` legitimate —
the US close dated d is settled by ~02:00 IST and fully in hand by 7 AM — so the overnight is **not** a timing
leak here. The kill is the execution *point*, not the information:

**The edge is in the open gap, which entry-at-open cannot capture.** US overnight ↔ India open *gap* ≈ 0.19,
but ↔ **open→close ≈ 0**. By 9:15 the open has already repriced to the overnight; you enter at the open and
nothing is left. The +0.044 above was scored on *close→close*, which credits the `close[d]→open[d+1]` gap you
are not positioned for. Re-scored on the capturable **open→close** return, every macro feature's correlation
collapses to noise: spx 0.008, dji 0.014, ndx 0.011, vix_chg 0.023, brent 0.026, usdinr 0.048 (all < 0.05)
→ ΔAUC ≈ 0 on re-run (`ENTRY="open"`, now the default).

## DJIA vs S&P vs Nasdaq
For predicting the India open *gap*, **Dow was marginally strongest** (0.199 > S&P 0.187 > Nasdaq 0.164) —
the original DJIA instinct had merit. But it's moot: the gap is not capturable, so none of the three yields
a tradeable signal.

## Verdict
Macro/cross-asset features add **no tradeable next-day direction edge**. There is a genuine statistical
US→India relationship, but it is fully priced into the open and absent from the window you actually trade
(open→close from a ~7 AM pre-open refresh). **Keep the book direction-agnostic.** This closes the macro lever
for *direction*. (Open question it does *not* close: whether global overnight data improves next-day
**magnitude/volatility** — which an open-entry long-options book *can* capture. That is a separate experiment.)

## Gap-and-go test (diag_gap.py) — can the gap clarify the capturable direction?
Hypothesis: the overnight gap (seen at the 9:15 open) tells you the day's direction, so enter at the open and
ride the remaining open→close move. **Refuted, and inverted** (A/B universe, eval 2024–26, n=31,867):
`corr(gap, open→close) = −0.03`; P(intraday continues in gap direction) = **0.431** (below coin flip); mean
intraday in gap direction = −6.6 bps. The **fade strengthens with gap size**: 5%+ gaps continue only **33.5%**
of the time, averaging **−108 bps** intraday. Indian large-caps mean-revert the overnight gap intraday.

Implication: following the gap gets ~43% of intraday directions right (wrong side). The mirror (fading) is ~57%
(≈66% on 5%+ gaps) but the magnitude lives only in the rare, event-driven big-gap bucket (n=182), where IV is
elevated — fighting a long-options book's premium. The gap does **not** clarify the capturable direction; the
only thing the overnight reliably predicts is the un-tradeable gap itself.

Do ±5% gaps "give back" the move? Mostly no — they **stick**. Of ±5% gaps: ~61% (2010–26, n=676) /
~67% (2024–26, n=182) give up *some* intraday, but only **~11% / ~3% fully round-trip** past the prior close;
median fraction given back ~12–17% (so a typical such day retains ~85% of the gap). Gap-downs fade slightly
more than gap-ups. The lean is a mild fade, not a reversal.

## Confirming runs (optional)
- `US_LAG=1` in macro.py (only EOD-known US data) → re-run `run_experiment.py`: macro lift collapses to ~0.
- Or switch target to open→close (the tradeable return) → same collapse.
- `diag_leak.py` / `diag_macro.py` are the audit trail for the two leak hunts (forward labels; overnight gap).
