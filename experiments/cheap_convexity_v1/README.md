# cheap_convexity_v1 — find options that out-move what `atm_iv` priced in

**Why:** `atm_iv` ranks the biggest movers, but that just buys the **most expensive** options — the move is
already in the premium (proven in `persistence_flow_v1`: no model beats `atm_iv` at *mover precision*, and
mover precision ≠ profit). The tradeable edge for a long-options book is the **residual**: stocks whose
**realized** move *exceeds* the move their premium implied — i.e., options the market **underpriced**
(cheap convexity / negative vol-risk-premium for the buyer).

**Target (the "surprise"):**
```
implied_move = atm_iv * sqrt(WINDOW/252)          # the ~1σ move the premium priced in
realized_cc  = |close[t+WINDOW] / close[t] - 1|    # what actually happened (close-to-close)
surprise     = realized_cc - implied_move          # > 0  => moved MORE than priced (cheap)
```
A CatBoost regressor predicts `surprise` from **IV-structure / skew / PCR / flow / catalyst** features that
are *orthogonal to the `atm_iv` level* (e.g. `atm_iv_ratio_20` = IV vs its own recent level, `realized_vol_20/atm_iv`
= recent realized vs implied, skew, PCR, delivery, earnings proximity). Predicting the *residual* (not the
magnitude) is the whole point — a model predicting magnitude just rediscovers `atm_iv`.

**Selectors compared (top-k per group/day, purged + embargoed walk-forward 2024Q1–2026Q2):**
| selector | picks by | expectation |
|---|---|---|
| `atm_iv` (PROD baseline) | highest implied vol | buys expensive → surprise/EV likely ≤ 0 (pays the VRP) |
| `cheap_convexity` (model) | highest **predicted surprise** | finds underpriced moves → surprise/EV > baseline if signal exists |

**Profit proxy (premium-adjusted):** `straddle_pnl ≈ realized_cc - 0.8*implied_move` (ATM straddle held to
the horizon; breakeven ≈ 0.8σ√T). `premium_ev.py` re-checks the picks on **real option bhavcopy** (actual
ATM straddle entry/exit premiums) — the gold-standard profit test.

## Run (from K3 root, terminal)
```bash
set PYTHONPATH=src
python experiments/cheap_convexity_v1/run_experiment.py     # core: predict surprise, rank, vs atm_iv (GPU/CPU CatBoost)
python experiments/cheap_convexity_v1/premium_ev.py         # validate top picks on real option premiums (needs bhavcopy)
```

## Outputs (`results/`)
- `metrics.json` — rank-IC(predicted surprise), and for k=3/5: mean surprise, mean straddle-PnL proxy,
  hit-rate P(realized>implied), mean realized move — **model vs `atm_iv` vs universe** — per group + per quarter.
- `feature_importance.csv` — what predicts the surprise (which orthogonal signals flag cheap convexity).
- `picks.csv` — the model's and the baseline's selected book (for `premium_ev.py`).

## Verdict guide
- Model top-k **straddle-PnL ≈ 0 or below `atm_iv`** ⇒ surprise isn't predictable; VRP dominates; `atm_iv` rule stands.
- Model top-k **straddle-PnL > 0 and > `atm_iv`, stable across quarters, confirmed on real premiums** ⇒ a genuine
  cheap-convexity edge → candidate to challenge the PROD selector (via a new lock, never editing v2).

## PROD safety
Reads only `load_market_data` (the shared table) + the universe JSON; the bhavcopy loader is read-only.
Never imports `koscine3.largemove`; writes only inside this folder. Verify: `python experiments/clone_prod.py --verify`,
`python experiments/freeze_v2_prod.py --verify`.
