# option_gain_study_v1 — findings (where large option gains come from)

Real bhavcopy, 2024-01→2026-06, **384,293 trades**, 63 stocks. Entry at OPEN, hold 5d, CALL+PUT × OTM ladder
(ATM…ATM+10%), pennies dropped (entry ≥ ₹2). Descriptive (oracle side + exit) — explains the convexity, doesn't
trade it. `high_ratio`=peak premium/entry, `close_ratio`=day-5 premium/entry.

## 1) Gain by strike (peak vs held)
| strike | med peak | mean peak | P(peak≥2x) | P(peak≥3x) | P(peak≥5x) | med held | P(loss<0.5x) |
|---|---|---|---|---|---|---|---|
| ATM | 1.43 | 1.80 | 0.245 | 0.092 | 0.022 | 0.77 | 0.324 |
| ATM+2% | 1.44 | 1.93 | **0.272** | 0.119 | 0.036 | 0.65 | 0.402 |
| ATM+4% | 1.42 | 1.99 | 0.268 | **0.126** | 0.045 | 0.56 | 0.455 |
| ATM+5% | 1.40 | 1.98 | 0.264 | 0.126 | 0.045 | 0.53 | 0.473 |
| ATM+10% | 1.30 | 1.94 | 0.227 | 0.113 | **0.047** | 0.49 | **0.503** |

- **Near-OTM (ATM+2% to +5%) is the sweet spot**: highest P(peak≥2–3x). Deep OTM (+10%) gives the rare ≥5x
  tail but loses >50% half the time. ATM is best *held* (least theta).
- **Theta is brutal**: median peak ~1.4x but median held 0.49–0.77x. The gain is in the peak, not the close.

## 2) Convexity available (best option per stock-day, oracle side+strike)
Median best peak **2.37x**; P(some option ≥2x) **0.65**, **≥3x 0.34**, ≥5x 0.14. The raw material is plentiful;
harvesting it needs the right side + peak exit.

## 3) Top large-gain stocks (rate best-option-of-day peak ≥3x) — A and B nearly equal
Leaders: AMBER 0.46, CDSL 0.45, HINDUNILVR 0.43, VEDL 0.42, BSE 0.42, BHARTIARTL 0.41, TITAN 0.40, SHRIRAMFIN,
INDIGO, M&M, HAL, COFORGE, AXISBANK, LT, MARUTI, DIXON, KAYNES… **by group A 0.341 vs B 0.354.**
Striking: low-IV mega-caps (HINDUNILVR 1.9% avg move) throw ≥3x gains as often as volatile B (BSE 5.8% move) —
their **cheap options leverage the small move**. Options framing self-hedges the vol regime.

## 4) What distinguishes big winners (peak≥3x) vs all
| | winners | all |
|---|---|---|
| abs stock move % | **4.91** | 2.83 |
| peak day (0–4) | **2.89** | 1.45 |
| entry premium ₹ | 28.7 (cheaper) | 44.2 |
| days-to-expiry | 17.4 | 20.2 |
| entry atm_iv | 0.29 | 0.28 (≈ no diff) |

Winners = a **~5% move that develops over days 3–4**, on a cheaper, nearer-dated option. IV does **not** flag winners.
Winner peak-day mix: day4 43%, day3 24%, day2 18% → **don't exit early.**

## 5) IV regime (terciles) — a wash on the multiple
low_iv med_peak 1.37 / P≥3x 0.109 (move 2.1%); high_iv 1.45 / 0.131 (move 3.8%). High-IV gains come from bigger
moves, not cheaper leverage — net similar multiple. **Don't chase high-IV for bigger multiples.**

## 6) DTE — near-expiry pays
P(peak≥3x): **≤10 DTE 17.8%** vs 11–20d 12.2% vs 21–40d 8.8%. Buy nearer expiry for convexity (gamma).

## 7) Leverage (median (peak−1)/move%, favorable side)
~0.31–0.35 (≈30–35× peak: a 1% favorable move → ~30–35% option gain), peaking at ATM+2–4%; deep OTM less
efficient (0.26). corr(abs move, best peak) = 0.34.

## Actionable structure for the convexity book
Near-OTM **ATM+2% to +5%**, **≤10–15 DTE**, **exit day 3–4** (not held to expiry), stock-agnostic across A+B
(cheap mega-caps qualify). IV regime doesn't matter for the multiple. **Unsolved (by design here):** the side
(direction = coin flip) and disciplined peak exit — that's the realized-edge gap measured in `cheap_convexity_v1`.
Trades: `results/option_gain_trades.csv` (384k rows, any further cut is free).

## 3-day variant (A→ATM+1%, B→ATM+2%, hold 3d; option_gain_3d.py) — does NOT beat 5d
Tested the "shorter window dodges theta" hypothesis. Verdict: **keep 5-day.**
- Per-option held-to-end mean ~flat (3d 0.99x vs 5d 0.95x); median slightly better (less theta) but irrelevant —
  the book lives on the tail, not the median.
- **Shorter window CUTS the winners**: P(some option peaks ≥3x / stock-day) **0.185 (3d) vs 0.34 (5d)**. Big
  winners keep developing — even in the 3d window **50% of ≥3x winners peak on day 3 (the edge)** → many would
  peak day 4–5. Capping the window caps the upside.
- A (ATM+1%) ≈ B (ATM+2%): peak 1.79x / 1.84x, P(peak≥3x) 0.092 / 0.100 (B slightly richer).
- Held-day3 net of 3% cost (per option, random side): A −3.78%, B −3.52% — same as 5d. The binding constraints
  (side + peak exit) are untouched by the window/strike tweak.
CONCLUSION: strike/window are not the levers — direction + exit are. Keep the 5-day window.

## Feature study — what's common among strong gainers (feature_study.py, leak-clean)
Target = option peak ≥3x (11.5% base). Features = option structure + full equity table, **lagged to the prior
trading day** (as-of the ~7AM refresh; entry-day intraday features were a ~0.085-AUC leak — removed).
- OOS AUC (purged WF): pooled **0.655** (was 0.740 leaky), top-decile precision **0.251 = 2.15× lift**.
  calls 0.674 / puts 0.592 — up-excursions more predictable than down (shock-driven).
- Top clean features: `side`, `entry_open` (cheaper→bigger multiple), `nifty_realized_vol_20`, `mkt_pct_above_sma50`
  (breadth), `days_to_month_end`, `dte`, `month`, `otm_pct`, `nifty_ret_5d`, `atr_14`, `atm_iv`, `pcr_oi`,
  `days_to_earnings`, `donchian_width_20`.
- INTERPRETATION: this is a **volatility/excursion + option-structure selector, NOT a direction model**. The peak
  target is a swing measure (predictable ~0.66); the *held* gain needs the side (coin flip). A model can rank
  big-peak options at 2.15× lift — useful for selecting a **direction-agnostic** convexity book — but doesn't
  crack direction. Validates/refines [[cheap-convexity-finding]] (same vol-driven selection edge, clean feature set).
  Saved: results/feature_univariate.csv, feature_importance_gainer.csv.
