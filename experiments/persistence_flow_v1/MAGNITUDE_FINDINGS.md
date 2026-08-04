# Large-mover precision — findings (v1–v3)

Objective (direction-agnostic): pick the day's biggest movers (by 5-day |move|, ranked within group)
with high precision. Universe ~28 eligible stocks/group/day. Random baselines: top-3 = 11%, top-5 = 18%.

## Ranker comparison (precision of top-1 pick)
| ranker | in top-3 | in top-5 | capture* |
|---|---|---|---|
| PROD confidence (current) | 26.8 | 41.1 | 56.1 |
| **atm_iv (implied vol) alone** | **31.4** | **45.1** | **58.5** |
| 22-feature P(top-5) classifier | 29.6 | 44.8 | 58.4 |
| LambdaMART learning-to-rank | 28.7 | 44.0 | 58.1 |
| minimal IV+catalyst model | 29.2 | 43.8 | 57.4 |

\*capture = % of the day's single biggest move that the pick captures.

## Conclusion: implied volatility is the ceiling
**Nothing beats ranking by `atm_iv`.** Regression, classification on the exact target, learning-to-rank,
22 features or 5 — all do *worse* than the single raw IV feature. The options market's implied move is
the dominant, near-complete magnitude signal; ML adds only noise.

Lift over random is real but capped: **top pick is a top-3 mover ~31% (2.9×), top-5 ~45% (2.5×), and it is
the actual #1 mover 12% of the time (3.4×).** The move-rank distribution of the #1-IV pick is flat — the
single biggest mover is usually news-driven and not in any EOD signal.

## The crucial caveat for the options book
IV ranks movers best **because the move is priced into the premium**. Ranking by `atm_iv` = buying the most
**expensive** options. So high mover-precision does **not** imply profit — for a long-options book the edge
must come from moves the IV *under*-prices, which an efficient options market mostly removes.
→ Mover-precision (this experiment) and options-EV (premium-adjusted) are different questions; the second
is decided only on the real bhavcopy option data.

## Recommendation
1. `atm_iv`-rank is a simple, robust mover-selector and beats PROD's confidence by ~+4 pts precision — a
   candidate selection change.
2. But before adopting, **validate on real option premiums**: does an IV-ranked book beat the confidence-ranked
   book on premium-adjusted EV? If IV just buys expensive moves, precision won't convert to profit.
