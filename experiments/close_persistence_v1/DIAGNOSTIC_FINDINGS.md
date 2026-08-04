# Close-persistence experiment — Diagnostic findings

Consumes locked PROD OOS predictions read-only. Close = window-end (t+5) signed return in trade direction.

## (a) Baseline reproduced (PROD top-2/group, t+3 cooldown)
| scope | trades | above | small | opp | top3* | peak |
|---|---|---|---|---|---|---|
| A_mcap30 ≥3% | 1038 | 23.3 | 28.2 | 48.5 | 23.5 | 44.4 |
| B_turn35 ≥4% | 1075 | 24.7 | 28.2 | 47.2 | 23.9 | 47.1 |
| **COMBINED** | 2113 | **24.0** | 28.2 | **47.8** | 23.7 | 45.8 |

Matches the cited ~25 / 25 / 50 split. (*top3 = stock among day's 3 biggest movers in group — see note.)

## (b) ORACLE bound — is 50% even reachable?  YES.
Pick the 2 *actual* best-closers from the top-N pool (cheat = upper bound):
| N | pick_by | above | opp | top3 | peak |
|---|---|---|---|---|---|
| 7 | confidence (PROD) | 23.1 | 48.6 | 23.3 | 44.6 |
| 7 | **ORACLE** | **54.2** | **6.8** | 25.0 | 76.9 |
| 5 | ORACLE | 44.6 | 15.7 | 21.4 | 68.7 |
| 10 | ORACLE | 63.3 | 3.0 | 29.0 | 83.5 |

→ On most days, ≥2 names in the top-7 *do* close ≥ thr. The target (≥50% above, ≤35% opp) is physically present in the candidate pool.

## (c) Separability — do simple signals capture it?  BARELY.
Pick top-2 of N=7 by each signal (combined):
| pick_by | above | opp | top3 | peak |
|---|---|---|---|---|
| confidence (=PROD) | 23.1 | 48.6 | 23.3 | 44.6 |
| side_margin | 22.1 | 47.8 | 18.6 | 42.2 |
| trend_sma | 24.9 | 46.9 | 22.0 | 44.8 |
| mom_align | 24.2 | 47.8 | 21.1 | 44.9 |
| blend(margin+mom) | 23.7 | 47.6 | 20.3 | 44.3 |
| **ORACLE** | **54.2** | **6.8** | 25.0 | 76.9 |

Tertile split within the pool (high vs low signal): side_margin opp 51→44, blend opp 51→45 — real but tiny.
**Hand signals capture ~2 of the 31-point above-gap. The persistence signal is not in obvious direction/momentum features.**

## (d) Magnitude tension — none.
Oracle (best closers) keeps top3 ≈ baseline (25 vs 24) and *raises* peak-hit (45→77). Optimizing close is "free" on magnitude — a name that closes ≥ thr necessarily peaked ≥ thr.

## Note on the top-3-mover metric
My definition (stock among the day's 3 biggest movers in its group, by ceiling) gives ~24%, not the cited 75–80%.
Need the intended definition before using it as a constraint. (Under *any* definition, (d) shows close-optimization doesn't hurt it.)

## Verdict / next step
- Headroom is real (oracle 54% @ N=7) and magnitude-free.
- Simple signals miss it → the test is whether a **learned Stage-2 model** (XGBoost, target `close_move ≥ thr`, full 20 feats + engineered, walk-forward) can extract interactions the hand signals can't.
- Honest expectation: weak tertile separation + prior "close-direction ≈ coin-flip" finding temper hopes; likely lands between baseline (23%) and oracle (54%). Build it, measure OOS lift/AUC, report the true frontier.
