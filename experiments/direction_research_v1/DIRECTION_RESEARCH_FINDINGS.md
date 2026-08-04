# Direction-edge deep research — findings (experiment direction_research_v1)

Goal: find any lever to lift 5-day direction beyond the ~50–52% coin toss. PROD v1/v2 untouched (isolated clone).

## What was tested (broad F&O universe, 2022–26, ~312k rows, univariate AUC = no fitting)
| dimension | result |
|---|---|
| 25+ signals × horizons 1/2/3/5/10d (momentum, reversal, 52wk-high, ADX, IV-spread, OI, delivery, gap) | all **0.46–0.512** |
| RELATIVE / cross-sectional (market-neutral) direction | **0.49–0.51** |
| turnover interaction (reversal in high-turnover, momentum in low-turnover) | real direction but **~0.51** |
| volatility-regime conditioning (low vs high IV) | momentum better in low-IV, but **~0.51** |
| PEAD (post-earnings drift) | **fails** — earnings gaps slightly FADE here (gap→dir5 = 0.479), opposite to US |

No single signal, horizon, or regime beats ~0.512.

## The one real (but decayed) edge: full multivariate model
XGBoost on all ~43 features (kitchen sink) — interactions of OI-positioning + earnings proximity + market regime
+ cross-sectional momentum + options skew — that no single feature shows:

| year (walk-forward) | OOS AUC | extreme-decile directional accuracy |
|---|---|---|
| 2024 | 0.548 | **58.9%** |
| 2025 | 0.524 | 54.3% |
| 2026 | **0.483** | **48.2%** (inverted) |

Recency-adaptive (trailing-1yr) model, by quarter: 2025Q1 **0.563**, Q2 **0.578**, Q3 0.489, Q4 0.519,
2026Q1 0.503, 2026Q2 0.454. **2026 mean AUC = 0.478.**

## Conclusion — exhausted
- A **genuine modest direction edge existed ~2024 through mid-2025** (~56–62% on the most-confident decile),
  driven by multivariate feature *interactions*, strongest on confident picks.
- It has **decayed to nothing / slightly inverted by 2026** — classic alpha decay. Neither expanding-window
  nor recency-adaptive retraining recovers it; it is not a fixable regime shift.
- Going forward, **5-day direction is a coin flip** with the available EOD data. This is now confirmed across
  ~30 signals, 5 horizons, 4 conditioning regimes, multivariate kitchen-sink, walk-forward, and recency models.

## Practitioner price-action (last attempt) — also nothing
Constructed ~15 candlestick / chart patterns (bullish/bearish engulfing, hammer, shooting star, doji, marubozu,
3-up/3-down, gaps, inside/outside bar, new-20d-high/low, close-location) on the broad universe and measured
actual forward up-rate vs base:
- Every pattern's lift is within **±2.4%** of base (~50%). Hammer (cited ~60% in lit) = **+0.2%**; volume-confirmed
  bullish-engulfing was **inverted** (−2.4%). Candle-only ML: OOS AUC **0.503 (h1) / 0.507 (h5)**, deciles flat ~50%.
- Matches rigorous backtest literature: candlestick patterns "as visual formations by themselves do not work" /
  "rarely survive out-of-sample." No edge here.

## The only untested frontier (needs data we don't have)
Intraday order-flow / signed tick data, real-time options flow, and news/earnings-surprise *magnitude* + sentiment.
These are where any residual short-horizon directional information would live; not in the EOD feature set.

## Recommendation
Keep the book **direction-agnostic** (PROD v2 unchanged). Do NOT deploy a directional tilt: the historical edge
is not durable and was negative in 2026. The direction overlay stays informational (grain of salt), as is.
