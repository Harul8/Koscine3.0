# direction_overlay_v1 — OVERLAY vs COMPLETE-RETRAIN for a 5-day PUT/CALL lean (contained; no PROD touch)

Question (user): to add a direction call to the direction-agnostic v3 book, is a light **recency-trained overlay on
the focused directional features** better, or a **complete retrain on ALL features**? Target y=1[close[t+5]>close[t]],
purged+embargoed quarterly walk-forward 2024Q1–2026Q2, eval on the v3 65-name A/B universe; then REAL ATM±2% CALL/PUT
premium EV on the actual v3 5d book (option_move_v1 forward held/peak ratios), net 3% cost. 3,198/3,229 book rows covered.

## Directional accuracy (OOS AUC / 2026 IC)
| config (feats × window) | ALL AUC | 2024 | 2025 | 2026 AUC | 2026 hit | 2026 IC |
|---|---|---|---|---|---|---|
| **FULL_expand** (all 145, ~1760d) | 0.522 | 0.519 | 0.504 | **0.531** | 0.518 | **+0.077** |
| OVERLAY_recent (focused 46, 365d) | 0.529 | 0.533 | 0.547 | **0.486** | 0.473 | −0.030 |
| FULL_recent (all, 365d) | 0.524 | 0.504 | 0.544 | 0.512 | 0.512 | +0.037 |
| OVERLAY_expand (focused, 1760d) | 0.546 | 0.579 | 0.526 | 0.515 | 0.495 | +0.030 |
| REGIME_rule (OI-regime×vol, 365d) | 0.523 | 0.503 | 0.513 | **0.463** | 0.464 | **−0.081** |

## Option premium EV on the v3 5d book (held 5d, net 3%, bootstrap 95% CI)
| config | ALL held EV | 2026 held EV | group A | group B |
|---|---|---|---|---|
| **FULL_expand** | **+0.027 [−0.012,+0.067]** | +0.038 [−0.052,+0.131] | −0.024 [−0.077,+0.032] | **+0.082 [+0.026,+0.140]** |
| OVERLAY_recent | −0.000 [−0.038,+0.041] | **−0.118 [−0.201,−0.032]** | −0.020 | +0.022 |
| REGIME_rule | −0.050 [−0.088,−0.010] | **−0.119 [−0.203,−0.023]** | −0.034 | −0.066 |
| coin-flip (random leg) | −0.066 [−0.085,−0.047] | −0.095 [−0.138,−0.049] | | |
| anti (FULL_expand opposite) | −0.160 | | | |

## Verdict — COMPLETE RETRAIN wins; the overlay FAILS out-of-sample
- **The full retrain (all features, long window) beats the overlay decisively and is the only robust one.** Its EV
  CI is disjoint from coin-flip (lean adds ~+9pp, anti −0.16 < coin −0.066 < model +0.027 = genuinely right side).
  It **removes the structural theta/VRP bleed** (coin −6.6% → model ~breakeven +2.7%). Model alone is NOT sig >0
  (CI spans 0) — its value is killing the bleed, not strong alpha. **Edge is entirely in group B (movers) +8.2%
  [+2.6,+14.0] significant; group A mega-caps flat/negative.**
- **The recency/focused OVERLAY and the transparent REGIME rule INVERT in 2026** (EV −11.8% / −11.9%, CIs exclude 0;
  AUC 0.486 / 0.463). Reason: a trailing window in 2026 is dominated by 2025 (mean-reversion regime), so they learn
  the dying regime and apply the wrong sign to 2026's momentum. **A regime flip at the year boundary is NOT
  capturable in real-time by a trailing model** — the validate_2026_regimes "OI regimes strong in 2026" was
  IN-SAMPLE/hindsight; OOS it flips against you. This kills the "adaptive overlay" idea.
- **What the surviving edge actually is** (FULL model importances): MARKET regime + breadth + market momentum
  (nifty_realized_vol_20 19%, mkt_pct_above_sma50 13%, mkt_advance_ratio/sma20 14%, nifty_ret_5d 7%) + CALENDAR
  (month 15%, days_to_month_end 11%, is_expiry_week) + days_to_earnings 5%. Stock-level OI/PCR is minor (~5%). So
  it's essentially a **market-beta / breadth momentum tilt** ("lean with the trending market"), NOT stock-specific
  OI alpha — which is why it's regime-stable (market trend) and why the OI-regime overlay (stock-level, flipped) died.

## Group-B MONTHLY-retrain sweep (run_groupB_monthly.py) — windows × feature combos, eval Jan→late-May 2026
Focused on group B (where the edge lives); train→Dec-2025 predict Jan-2026, monthly retrain forward.
**DECISIVE: only the EXPANDING window works; every 3m/6m/9m window INVERTS (neg IC, EV worse than coin).**
| config | univ AUC | IC | book-B held EV (95% CI) |
|---|---|---|---|
| ALL / expand / train-B | 0.565 | +0.146 | **+0.101 [−0.019,+0.224]** (best EV) |
| ALL / expand / train-All | 0.578 | +0.171 | +0.049 (best AUC/IC) |
| NO_CAL / expand / train-All | 0.565 | +0.140 | +0.035 |
| MKT_MOM / expand | 0.557 | +0.122 | −0.026 |
| OI_FLOW / expand | 0.505 | +0.012 | −0.072 (OI/flow direction = DEAD) |
| any 3m/6m/9m window | 0.43–0.48 | −0.05..−0.18 | −0.08..−0.29 |
| REGIME_rule / 9m | 0.415 | −0.177 | −0.221 |

**Why short windows fail / expanding wins:** a 3–9mo window in early-2026 straddles the regime flip (late-2025
mean-reversion + early-2026 momentum) → muddled/inverted mapping. The expanding model learns the STABLE
market-state→direction relationship (breadth/vol/market-momentum = market-timing/beta) over 15y; **monthly retrain
refreshes the INPUTS, not the relationship** — adaptation = current market state fed to a stable mapping, NOT a
short window. Per-month (ALL/expand/trB): leaned PUT into the Jan dip (call-share 0.13, hit 0.65), flipped CALL for
the Apr–May rally (0.65→0.76, hit 0.67/0.60); March the transition-month miss (EV −0.20). NO_CAL≈ALL → edge is NOT
calendar overfit. OI_FLOW alone ≈ 0 → OI/flow carries NO usable 2026 direction even retrained in-regime.
**Caveat:** call-share swings 0.13→0.87 ⇒ it's ~ONE market-direction bet/month on high-beta movers (concentration,
whipsaw risk); EV CI includes 0 (n≈270), AUC/IC (n=3266) is the firmer evidence.

## Caveats / recommendation
- Marginal: ~breakeven overall, group-B-only, 2026 not independently significant (wide CI). Calendar features
  (month/month-end, 26% of importance) are overfit-prone — discount that share.
- Because it's a market-beta call, signals cluster same-side on a given day → the book becomes a leveraged
  market-direction bet (concentration risk), not 6 independent name bets.
- **Recommend:** keep v3 direction-agnostic as the core. Best deployable lean = **monthly-retrained,
  EXPANDING-window, full-feature CatBoost on group B** (2026: AUC 0.565, IC +0.146, held EV +10%), used as a small
  monthly market-direction tilt on the B movers — NOT a short-window/recency overlay and NOT the OI-regime rule
  (both invert). Treat it as market-timing (concentrated monthly bet), size accordingly, refresh monthly.
  See [[direction-edge-research]] (direction non-stationary; the fix is long memory + fresh inputs, not short windows).
