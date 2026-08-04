# Direction & magnitude — consolidated research (2 rounds, ~25 signals)

Task framing: A+B universe, EOD t → 5-day forward window. Direction tested **conditional on a big ≥4% move**
(the real decision: gate to movers, then which way / which side). dir_peak = which side spikes more.

## DIRECTION — not predictable (quadruple-confirmed)
Every academically-documented direction signal lands at coin-flip (AUC 0.48–0.52):

| signal | AUC | source |
|---|---|---|
| call−put IV spread (volatility spread) | 0.487 | Cremers-Weinbaum 2010 (JFQA) |
| IV skew (OTM put) | 0.487 | Xing-Zhang 2010 |
| OI build-up / fut ΔOI | 0.48–0.52 | F&O practitioner framework |
| delivery% change | 0.51 | NSE conviction |
| momentum / rel-strength | ~0.50 | Jegadeesh-Titman |
| **52-week-high distance** | **0.499** | George-Hwang 2004 (JF) |
| **Close-Location-Value / Chaikin** | 0.499 / 0.507 | order-flow / accumulation |
| **option-volume skew / pcr_vol** | 0.480 / 0.514 | Pan-Poteshman 2006 (RFS) |

Combined model: OOS AUC **0.548** (round 1) → **0.541** when round-2 signals added (overfits). 
The literature's edges are small (~40–50 bps/week), decay over time (authors say so), are US large-cap
cross-sectional, and **do not survive** the binary direction task conditional on an already-selected big mover,
5-day horizon, Indian F&O, recent years. Credible ML lit also reports ~0.50–0.55 AUC for single-stock daily
direction (the "85–91% accuracy" papers are leakage artifacts). **Conclusion: direction is data-capped.**

## MAGNITUDE — predictable; IV dominates, 52wk-high is a real secondary signal
| signal | AUC for P(big ≥4%) |
|---|---|
| atm_iv (implied vol) | **0.726** |
| atr_pct_14 | 0.706 |
| donchian_width_20 | 0.664 |
| **dist_52wh (nearness to 52-wk high)** | **0.572** |
| atr_pct_14_rank_60d (relative contraction) | 0.547 |
| volume dry-up / CLV / Chaikin | ~0.50 |

The NR7/VCP "contraction precedes expansion" idea is weak cross-sectionally (relative-contraction 0.55,
volume dry-up ~0.52); the dominant magnitude signal is **volatility LEVEL** (IV/ATR), plus a real
**breakout/52-wk-high** effect (0.572) consistent with the breakout literature.

## Implications
- **Direction:** stop trying to predict it. Either trade non-directionally (straddle/strangle on movers) or
  single-side + exit-at-peak, accepting ~50/50 direction.
- **Magnitude / mover precision:** rank by `atm_iv` (no model beats it). `dist_52wh` HURTS as a ranking blend
  (23.7%) but HELPS as a selectivity filter. Precision is raised by SELECTIVITY, not more features.

### Recommended mover-precision selector (vs 31% top-3 / 45% top-5 baseline, "2 every day")
| rule | trades/yr | in_top3 | in_top5 | capture | stable across yrs |
|---|---|---|---|---|---|
| IV-gap top 20% | ~75 | 43.4% | 50.9% | 63.7% | yes (42/42/47) |
| atm_iv top 10% | ~38 | 44.2% | 53.1% | 64.2% | yes (42/43/49) |
| **IV-top20% AND near-52wk-high** | ~34 | 41.7% | **56.3%** | 63.5% | (best top-5) |
| earnings-window | ~14 | 47.6% | 57.1% | 67.1% | sparse |

Selectivity lifts precision ~31%→~44% (top-3), ~45%→~56% (top-5), capture ~58%→~64%, at 34–75 trades/yr.

### Volume frontier (direction-agnostic — metric = % of picks that make a tradeable move EITHER way)
Direction is a coin flip, so you want MANY trades. Reframe to big-move rate (what straddle/peak-exit monetizes):
| operating point | trades/yr | avg move | ≥6% | ≥8% | in_top5 | capture |
|---|---|---|---|---|---|---|
| top-1/grp, IV-gap top 30% | 113 | 8.2% | 58% | 41% | 49% | 62% |
| **top-2/grp, IV-gap top 40%** (recommended) | **301** | **7.8%** | **56%** | **38%** | 46% | 60% |
| top-1/grp, every day | 377 | 7.2% | 51% | 33% | 45% | 58% |
| top-2/grp, every day | 753 | 7.0% | 50% | 30% | 43% | 58% |
| universe base rate | — | 5.0% | 26% | 13% | — | — |

Frontier is FLAT (all picks are high-IV movers): 300→750 trades only drops ≥6% from 56%→50%. **Volume is nearly free.**
Recommended: **top-2/group, IV-gap top ~40% → ~300 trades/yr, 56% move ≥6%, 38% ≥8%, avg 7.8%.**

## Sources
Cremers-Weinbaum (SSRN 968237) · Pan-Poteshman, *Information in Option Volume* (RFS 2006, mit.edu/~junpan/volume.pdf) ·
George-Hwang, *52-Week High & Momentum* (J. Finance 2004) · Xing-Zhang, IV skew & cross-section ·
Chordia-Subrahmanyam, *Order imbalance & individual stock returns* (JFE) · NR7/VCP (Crabel; Minervini).
