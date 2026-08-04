# mover_precision — high-precision large-move signal model (findings)

OBJECTIVE (user): emit ~2-3 direction-agnostic signals/day; maximize the chance they are among the day's actual
TOP-3/TOP-5 movers by 5-day magnitude (either direction = what an option buyer captures; side taken offline);
minimize duds. Low volume, high precision. Quarterly-retrain walk-forward 2024->2026. New experiment, PROD untouched.

Target `move_mag` = max(max(high[t+1..t+5])/close−1, close/min(low[t+1..t+5])−1) (exit-at-peak magnitude).
Universe = tradeable 65 (A_mcap30 ∪ B_turn35); train broad (rolling ~4.5yr), rank within 65. CatBoost, leak-safe
(forward K2 label cols excluded). Baselines: atm_iv-rank, random.

## Levers tested (v1 + v2) — rank precision is CEILING-CAPPED
Every model lands at the same top-3/day precision (~0.21 in-top3, ~0.32 in-top5), only marginally above atm_iv:
| selector | in_top3 | in_top5 | ≥1 in top5/day | hit≥6% | hit≥8% |
|---|---|---|---|---|---|
| random | 0.060 | 0.100 | 0.28 | 0.31 | — |
| atm_iv | 0.203 | 0.315 | 0.694 | 0.614 | 0.408 |
| reg move_mag | 0.210 | 0.321 | 0.689 | 0.621 | 0.402 |
| clf ≥8% | 0.214 | 0.326 | 0.701 | 0.618 | 0.401 |
| YetiRank (LTR) | 0.204 | 0.322 | 0.713 | 0.616 | 0.402 |
| **ensemble (clf+reg+iv)** | 0.213 | 0.327 | 0.710 | 0.619 | 0.407 |

**No model beats atm_iv by more than noise** — the exact top-3/5 RANK is news-driven (data-capped, confirmed yet
again). Lean features, per-group (worse — dilutes), broad-vs-univ train: all same. Everything is **~3.3× random**.

## Two real, usable levers
1. **Dual-gate (model AND atm_iv agree):** both-top-5 → **78% of days have ≥1 signal in the actual top-5** (~3.6/day);
   both-top-3 → highest per-signal precision **in5 0.337** (~1.9/day). Agreement helps.
2. **Conviction gating lifts MAGNITUDE** (what pays an option buyer; rank-precision doesn't lift, hit-rate does):
   top-3 gated ~1/day → **hit≥6% 0.675, hit≥8% 0.468, move 9.2%**; tightest ~0.6/day → **hit≥6% 0.70, hit≥8% 0.49**.

## FINAL operating points (forward 2024-2026, walk-forward; results/mover_book_final.csv)
| tier | signals/yr | in_top5 | ≥1 in top5/day | hit≥6% | hit≥8% | move% | use |
|---|---|---|---|---|---|---|---|
| top-2/day | 469 (~1.9/d) | 0.330 | 0.55 | 0.63 | 0.41 | 8.35 | tight daily core |
| **top-3/day** | 704 (~2.8/d) | 0.326 | **0.71** | 0.62 | 0.41 | 8.45 | **daily core (2-3/day)** |
| dual-gate agree | 890 (~3.6/d) | 0.318 | **0.78** | 0.62 | — | 8.41 | best coverage |
| dual-gate ∩ top-3 | 468 (~1.9/d) | **0.337** | 0.55 | 0.63 | — | 8.40 | highest per-signal precision |
| conviction ~1/day | 284 | 0.32 | — | **0.675** | **0.468** | 9.15 | strong-move tier (~4-5/wk) |
| conviction ~0.6/day | 147 | 0.30 | — | **0.703** | **0.492** | 9.38 | strongest-move tier |

## Verdict
- **Rank precision is data-capped (~0.32 in-top5; ~0.71 ≥1-in-top5/day at 3/day); no model beats atm_iv** — the
  exact biggest movers are news-driven and not predictable from EOD features (re-confirmed across reg/clf/LTR/
  ensemble/dual-gate/per-group/lean). Ensemble + dual-gate are marginally best.
- **The signal IS usable**: at 2-3/day it's ~3.3× random, ~62% hit≥6%, ~40% hit≥8%, mean move 8.4%, only ~5% duds,
  and ~70% of days carry ≥1 genuine top-5 mover.
- **Dial volume↔quality** via conviction: ~1/day → 68% hit≥6% / 47% hit≥8% (the "4-5 strong trades/week" tier).
- Direction-agnostic — book outputs the magnitude signal; user buys CALL/PUT per offline view and exits near the
  day-3/4 peak (see [[option-gain-structure]]). Deliverable: results/mover_book_final.csv (31,930 rows, per day
  ranked, with actual move_mag + hit flags). PROD untouched (clone_prod/freeze_v2 verifiable).
