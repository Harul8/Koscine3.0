# Research-backed signal tests — findings

A+B universe (65 stocks), 2010–2026, EOD t → 5-day forward window. 196k stock-days.

## Magnitude — *will* a big move come?  YES (already captured by PROD)
Univariate AUC for P(big move ≥4% in next 5 days), base rate 59%:
| signal | AUC |
|---|---|
| **atm_iv** | **0.726** |
| **atr_pct_14** | **0.706** |
| donchian_width_20 | 0.664 |
| atr_pct_14_rank_60d (relative contraction) | 0.547 |
| volume_dryup_score | 0.524 |
| adx_14 | 0.502 |

- Magnitude is strongly predictable — but it's volatility **LEVEL / clustering** (high IV/ATR → more big moves), **not** the NR7/VCP "contraction precedes expansion" idea. The *relative-contraction* feature is weak here (0.547), and volume dry-up barely registers (0.524).
- PROD already uses `atm_iv`, `atr_pct_14`. **No new magnitude edge.**

## Direction — *which way?*  NOT predictable (triple-confirmed)
Conditional on a big ≥4% move (n=116k, 51.9% up). Univariate AUC — every signal sits in **0.48–0.52** (coin flip):
| signal (research-backed) | AUC dir_peak | AUC dir_close |
|---|---|---|
| fut_chg_oi_ratio_20 | 0.519 | 0.525 |
| pcr_oi | 0.514 | 0.511 |
| put_call_iv_skew | 0.513 | 0.509 |
| vol_spread (call−put IV, Cremers-Weinbaum) | 0.487 | 0.491 |
| iv_skew_ce_minus_pe (Xing-Zhang) | 0.487 | 0.491 |
| oi_buildup_ratio | 0.480 | 0.474 |
| delivery_pct_chg_5 | 0.506 | 0.506 |
| ret_20d / momentum | ~0.50 | ~0.50 |

**Combined direction model (train<2023, test≥2023): OOS AUC = 0.548** (both dir_peak and dir_close). Best signal's tertile lift is non-monotonic (54/49/51) — no real structure.

### Why the academic edges don't transfer
The Cremers-Weinbaum / skew / OI-buildup effects are (a) small (~50 bps/week), (b) **decay over time** (the authors say so), (c) US large-cap cross-section, (d) measured unconditionally. Our task is the hardest case: **binary direction conditional on an already-selected big mover, 5-day horizon, Indian F&O, recent years.** Conditional on a big move, the sign is set by the *next* 5 days' news/flow — not knowable from EOD data. Best achievable ≈ **0.548**, not actionable for side selection.

## This explains the close experiment
Close-above needs magnitude (predictable) **AND** direction-persistence (not predictable). Direction is the binding constraint → close-above is capped (~25%), exactly as `close_persistence_v1` found.

## Strategic implication
Magnitude is predictable; direction is not. The textbook response when you know vol will expand but not which way is a **non-directional** position. Options:
1. **Non-directional structures (straddle/strangle)** on predicted big-movers — capture the move regardless of side; test EV on the real bhavcopy option data.
2. **Single-side + exit-at-peak** (current PROD) — accept coin-flip direction, rely on convexity + peak exit; close the direction quest as data-capped.
3. **Squeeze the marginal 0.548 edge** (oi_buildup/iv_skew/gap as a slight side-tilt) — low value, added complexity.
