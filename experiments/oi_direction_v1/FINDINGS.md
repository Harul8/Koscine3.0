# oi_direction_v1 — does OI positioning (and what else) carry DIRECTION? (contained; no PROD touch)

## 1) OI positioning → direction (oi_direction_study.py + validate_2026_regimes.py)
Tested the 4 regimes (Long Buildup/Short Buildup/Short Covering/Long Unwinding) from price × futures-OI change.
- Multivariate OI direction classifier (trained on history): **OOS AUC 0.521**, and INVERTS in 2026 — but that is a
  STALE-MODEL artifact (it learned the old mean-reversion sign), NOT a dead signal. See the per-regime 2026 recheck:
- **CORRECTION to the earlier "OI broke in 2026":** per-regime, OI gives its **STRONGEST separation on record in 2026.**
  5d regime → fwd5, 2026 (base P(up)=0.432; edge vs base; all significant): Long Buildup **+0.060** (UP/CALL),
  Short Covering **+0.059** (UP/CALL), **Short Buildup −0.082 (DOWN/PUT, strongest cell in the study)**, Long
  Unwinding +0.021 (muddy). Regime spread **0.142** vs pre-2025 0.058 / 2025 0.054 — the WIDEST of any period.
- **What changed = the SIGN flipped on the buildups (mean-reversion → momentum).** Pre-2025 the edge was "fade the
  buildup" (Long Buildup went DOWN −0.034, Long Unwinding bounced UP +0.024). In 2026 buildups CONTINUE. The three
  regimes the user named all keep their ORIGINAL continuation direction in 2026; only Long Unwinding is muddy/flipped.
- **High-vol 2026 amplifies**: Short Buildup −0.142, Long Buildup / Short Covering +0.07; low-vol shrinks to ~±0.03;
  aggressive (top-tercile |OI z|) keeps the pattern (Short Buildup −0.072). **It is a 5-DAY phenomenon** — at 1d/2d
  the same regimes are nearly flat (spread 0.035 / 0.047 vs 0.142 at 5d).
- Caveats: edges are RELATIVE to a down-tilted 2026 base (use to rank PUT vs CALL names, not absolute hit-rate); it
  is ONE ~5.5-month regime episode and the sign flipped once → deploy only recency-trained / regime-adaptive.

## 2) WHY 2026 broke + what works now (direction_2026_diag.py) — THE key result
**Regime FLIP: mean-reversion (2024-25) → MOMENTUM (2026).** `ret_5d` signed IC vs forward return:
| horizon | pre-2025 | 2025 | 2026 |
|---|---|---|---|
| fwd-2d | −0.006 | −0.037 | **+0.053** |
| fwd-5d | −0.014 | −0.075 | **+0.112** |

The OI mean-reversion read inverted because the market turned trending. Drivers (web): SEBI Nov-2024 F&O overhaul
(3× lot sizes, weekly-expiry cut, option-buyer margins, 5% caps) squeezed retail out → OI more institutional, less
crowd to fade; and 2026 is FII-flow-driven/trending.

**What carries direction in 2026 (signed IC):**
| signal | fwd-2d | fwd-5d |
|---|---|---|
| nifty_ret_5d (MARKET momentum) | +0.105 | **+0.165** |
| ret_5d (stock momentum) | +0.053 | +0.112 |
| pcr_oi_chg_5 | +0.067 | +0.109 |
| pcr_oi | +0.064 | +0.097 |
| maxpain_gap | +0.058 | +0.093 |

Momentum (esp. market) is now the signal, IC 0.10–0.17 (>> OI's ~0.02–0.05). **Concentrated in HIGH-vol**
(high-vol 2026 nifty_ret_5d +0.145; low-vol → weak mean-reversion) — consistent with dealer-gamma (short-gamma/
high-vol → momentum; long-gamma/low-vol → pinning).

## Meta-insight
**Direction is NON-STATIONARY — it flips between mean-reversion and momentum regimes.** Every static direction
signal therefore decays (OI, the 2024-mid25 multivariate edge, etc.). The only robust approach is a **regime-adaptive
overlay**: detect the current regime (recent momentum-IC sign / vol-gamma proxy) and lean with it; recency-train.

## Plan (1-2 day directional overlay — to build after the contract-flow experiment)
Regime-adaptive momentum/positioning: market + stock momentum + PCR + maxpain-gap, vol-conditioned, recency-trained,
1-2 day horizon (sharper short-term; regime can shift). Honest: IC ~0.10 @2d in the current momentum regime = a usable
directional TILT, holds only while the regime holds → adaptive not static. Direction-agnostic v3 stays the core.
