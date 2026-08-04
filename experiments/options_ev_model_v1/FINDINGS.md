# options_ev_model — ML program findings

OBJECTIVE: select a daily, direction-agnostic options book (straddle/strangle, entry t+1 OPEN) maximizing real
premium-adjusted EV net of ~3% cost, purged walk-forward, beating atm_iv-rank and cheap_convexity (+1.9% net).
Direction = coin flip (settled) -> model is a volatility/premium-gain selector.

## v1 (held-5d, model_v1.py) — held is a loser; raw-return target ranks badly
Universe held-5d gross EV: straddle **−6.5%**, strangle **−13.5%** (theta + VRP). Net@3% top-K/group:
| structure | best model | atm_iv |
|---|---|---|
| straddle | CLF top-2 **−5.19%** / REG −5.70% | −9.78% |
| strangle | REG top-2 −10.61% | −12.72% |

- Model beats atm_iv by ~4–5pp (consistent with cheap_convexity), but **held-5d is negative for all**.
- **Conviction curves BACKWARDS** (CLF top-5% −14.7%, worse than avg): predicting the *raw* straddle return
  ranks poorly — the model picks high-IV/expensive straddles (the atm_iv trap). Raw return is fat-tailed/noisy.
- straddle >> strangle (drop strangle).

## Lessons -> Phase 2
1. **Managed exit, not held** (held theta-killed). Use the day-by-day path (straddle_paths.csv) -> d3/d4/trailing.
2. **Residualized target, not raw return** — regress the cheap_convexity SURPRISE (realized_cc − implied), a
   well-behaved % target, then rank by it (proven rank-IC +0.166). Raw-return regression mis-ranks.
3. Straddle (ATM) primary.

## v2 (managed-exit + surprise target) — RUNNING/NEXT
Test exits {d3, d4, trail30, trail40, peak} x targets {managed-exit return, surprise} on the path data; select
top-K/quintile; eval net EV vs atm_iv + held + cheap_convexity (+1.9%). _(fill in.)_
