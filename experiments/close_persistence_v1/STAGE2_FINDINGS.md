# Stage-2 learned re-ranker — findings

Walk-forward (base<T-1, isotonic-calibrate T-1, predict T). Per group/side XGB on the close target.
Stage 1 (PROD confidence) gates top-N; Stage 2 picks 2 by `p_above − λ·p_opp`.

## Is close-persistence learnable? — Mostly NO.
OOS AUC (0.5 = no signal):
| group | p_above (close ≥ thr) | p_opp (close < 0) |
|---|---|---|
| A_mcap30 | **0.584** | 0.525 |
| B_turn35 | **0.599** | 0.528 |

- `p_above` has a **weak but real** edge (~0.59).
- `p_opp` (close direction) is **essentially a coin flip** (~0.52) — confirms the long-standing finding: *which* big movers reverse by the close is not predictable from EOD features.

## Selection (combined) — marginal gains, far from target
| pick_by | N | above | opp | top3 | peak |
|---|---|---|---|---|---|
| confidence (PROD baseline) | 7 | 23.1 | 48.6 | 23.2 | 44.6 |
| stage2 λ=0 | 10 | **25.4** | **45.6** | 22.7 | 46.4 |
| stage2 λ=0 | 5 | 24.7 | 46.9 | 23.0 | 46.0 |
| **ORACLE** | 7 | **54.2** | **6.8** | 25.0 | 76.9 |

- Best learned config lifts above **23→25%** and cuts opp **48.6→45.6%** — a ~2–3 pt move.
- Penalizing `p_opp` (λ>0) does **not** help — because `p_opp` has no signal.
- top3 / peak preserved (no magnitude tension, as predicted).

## Verdict
Target was above→50%, opp→35%. Achieved ~25% / ~46%. **The re-ranking hypothesis is not supported at target level.**
The oracle proves the headroom physically exists (54%), but it is driven by next-5-day news/flow that EOD features cannot see. Magnitude (peak/ceiling) is mildly predictable; **close direction is not.**

## Credible paths forward (none promote to PROD)
1. **New persistence/flow data** — the only path to higher close-above: post-earnings-drift windows, delivery%/volume persistence, FII/DII flow continuation, options OI-build direction, sector-momentum continuation. Not in current EOD set.
2. **Embrace exit-at-peak** — the convexity book already exits at the favorable peak; peak-hit is 45–77%. Then close-to-t+5 is the wrong KPI; capture-of-move is right. Drop the close objective.
3. **Keep the marginal Stage-2 tweak** (A-group benefits most) as a minor selection improvement — but it does not meet the stated goal.
