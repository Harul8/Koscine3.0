# PRD — Large-Move Options Selection Engine (Koscine 3.0)

| | |
|---|---|
| **Version** | v1.0 (draft) |
| **Date** | 2026-06-13 |
| **Owner** | Rahul |
| **Status** | Design locked, pending build |
| **One-line** | Daily, surface a small ranked shortlist of stocks likely to make a large 5-day move, to be traded as long-option (convexity) positions. |

> Design principle for this PRD: **every decision below is backed by an experiment in `analysis/` run on real equity + bhavcopy data.** Where a choice was tested and rejected, that is stated. This is an evidence-grounded spec, not a proposal.

---

## 1. Business Objective

Buy out-of-the-money options on stocks the model predicts will make a **large favorable move (≥4%) within 5 trading days**, entering at next-day open. The economic engine is **convexity**: downside is capped at the premium; upside is leveraged ~12–25× the underlying's % move (measured). We do not need high accuracy — we need a positive-expectancy shortlist plus the option payoff.

**What the product delivers:** each trading day, a ranked **top-3 shortlist** — symbol, direction (CALL/PUT), **calibrated confidence**, and **expected move %** — from which the user selects and executes (option strike/exit managed offline).

**Why this is the objective (and not the prior ones):** the original fixed-threshold "hit/near" engine (Koscine 2/3, v19) produced ~87 calls in 2.5 years and was regime-unstable. The clean-move + options reframe gives an abundant, regime-stable, tradeable signal with a convex payoff.

---

## 2. ML Problem Framing

**Per-day, per-stock binary classification of "favorable move ≥ threshold in next 5 trading days," + a magnitude regression**, scored across the tradeable universe and ranked into a daily top-3.

- **Not** a rare-event touch-probability model (prior framing — wrong target).
- **Not** learning-to-rank — *tested*: LambdaMART **underperformed** the classifier (precision@1 34.9% vs 38.6%), because the predictive signal is largely *absolute* (volatility level), not relative ordering. See §10.
- **Not** deep / sequence models — tabular EOD data; the ablation (§7) shows the signal is point-in-time and saturates with ~11–20 features. GBDT is the correct tool.

---

## 3. Success Criteria (KPIs) — with measured status

| KPI | Target | Measured (top-30 @4%, OOS 2024–26) | Status |
|---|---|---|---|
| Precision@1 (top pick hits ≥4%) | ≥ 40% | **47.1%** (46/46/52 by yr) | ✅ |
| ≥1 actionable idea/day | P(≥1 of top-3 hits) high | **75.4%** | ✅ |
| Regime stability | ≥40% every year incl. 2026 | yes (2026 = 51.6%) | ✅ |
| Confidence calibration | predicted ≈ actual | 0.51→50.8% (bulk) | ✅ |
| Realized option EV (gross) | > 0 | **+38%** peak-close / **+6%** blind-hold (top-20 @4%) | ✅ |
| Net EV after costs | > 0 | **not yet measured** | ⏳ |

---

## 4. Scope

**In (v1):** top-30 F&O equity universe; EOD signal → next-open entry; 5-trading-day horizon; both directions; monthly stock options; daily ranked shortlist output.

**Out (v1, by design):** intraday signals; option strike/exit optimization (managed offline by the user); position sizing & portfolio risk; live execution/OMS; the 21-40 second tier as a *precision* slot (tested — caps at ~24–30%, not viable at 40%; may run opportunistically).

---

## 5. Data

| Dataset | Path | Use | Notes |
|---|---|---|---|
| Equity features | `data/processed/daily_features.parquet` | **Training + features** | 1.31M rows, 447 symbols, 2010-2026, 152 leakage-safe features (built by `koscine/` from `data/silver`) |
| F&O bhavcopy | `data/raw/derivatives_bhavcopy/` | **EV backtest only** (not features) | Strike-wise option OHLC, 2010-2026, two NSE formats (loader: `analysis/options_bhavcopy.py`) |

**Leakage rules (enforced):** block `future_`, `entry_`, `up_move_`, `down_move_`, `fwd_return_`, `*_adverse_`, `label_`. The model trains **only** on the equity table; bhavcopy is touched only by the EV backtest (verified — no training leakage from options data).

**Point-in-time integrity (critical, learned the hard way):** a stock is eligible on date *t* only if it was **optionable** (proxy: `atm_iv` present on *t*) **and** non-penny (`close ≥ ₹100`). Without this, a 2025-turnover universe leaks look-ahead (17 of top-65 weren't in F&O in 2024) and concentrates on untradeable pennies (IDEA was 24–62% of picks in contaminated runs).

---

## 6. Target / Label Definition

For each (symbol, date, side), using the clean-move contract (`src/koscine3/outcomes/clean_move_contract.py`):

- Entry = open at **t+1**; window = **t+1 … t+5**.
- `ceiling` = max favorable move vs entry (long: `(max_high − entry)/entry`; short: `(entry − min_low)/entry`).
- **Classification label:** `ceiling ≥ 0.04` (binary).
- **Regression target:** `ceiling` (clipped 0–0.5), for "expected move."
- **Diagnostics (offline exit aid, not targets):** `days_to_peak`, `reaches_big_by_day`.

No stop in the options framing — the premium *is* the risk, so only the favorable-move target matters. (The 0.6×ATR "clean" stop concept belongs to the equity variant; not used here.)

---

## 7. Feature Engineering

**Decision: lean, level-dominated set (~20 features). Justified by ablation.**

The full table has 152 well-built features (51 options/IV/OI, 32 compression, 46 momentum, earnings/events). An ablation (`analysis/level_vs_timing_ablation.py`) showed:

| Feature set | AUC (≥5% / ≥10%) | precision@1 |
|---|---|---|
| LEVEL only (11) | 0.733 / 0.789 | 42.8 / 22.2 |
| TIMING only (46) | 0.711 / 0.769 | 44.1 / 18.5 |
| ALL + engineered (62) | 0.736 / 0.788 | 44.4 / 22.2 |

**Conclusion: volatility *level* is the signal; it saturates with ~11 features. Timing/engineered features (incl. per-stock IV/ATR z-scores, IV−RV spread, compression×earnings interactions) added ~nothing.** Predictable precision is capped (~47–50% on ≥4%); the residual is news-driven and not in EOD data. → Use a lean set; do not over-engineer (diminishing returns proven).

**Locked feature list (20):**
`atm_iv, atr_pct_14, atm_ce_iv, atm_pe_iv, nifty_realized_vol_20, mkt_pct_above_sma50, days_to_earnings, atr_pct_14_cs_rank, realized_vol_20, atr_pct_14_rank_60d, sector_vol_20, ret_20d_cs_rank, pcr_oi, fut_oi_ratio_20, close_sma50_dist, vol_5v20_ratio, atm_iv_ratio_20, donchian_width_20, mkt_pct_above_sma20, month`

Top importances: `atm_iv` ≫ `atr_pct_14` > `atm_*_iv` > `nifty_realized_vol_20` > `days_to_earnings`.

---

## 8. Universe

**Train BROAD, trade NARROW.**

- **Training universe:** all ~450 eligible stocks. *Justified* (`analysis/train_scope_test.py`): broad training beats focused on AUC + precision for both tiers (357k positives vs 18k; move patterns are cap-agnostic). Removing low-ATR stocks did **not** help (`low_atr_trim_test.py`).
- **Trading universe:** **top-30** by median turnover, point-in-time eligible (§5). top-20 gives slightly higher precision (49.7%) but ~18 names; top-30 gives 47.1% with ~22–25 names (better diversity). **v1 = top-30 @4%.**

---

## 9. Model Architecture & Choice

**Per side (long→CALL, short→PUT), two heads:**

1. **Confidence head** — LightGBM **classifier** `P(ceiling ≥ 0.04)`, `class_weight="balanced"`, then **isotonic calibration** on a held-out year so the probability is meaningful.
2. **Expected-move head** — LightGBM **regressor** on `ceiling`.

**Why LightGBM GBDT:** tabular heterogeneous features, native missing-value handling (option features are absent pre-listing), fast, strong out-of-the-box, transparent gain importances. **Rejected alternatives (with evidence):** LambdaMART ranker (tested — worse); deep/sequence nets (signal is point-in-time + small feature set — unjustified complexity); pure regression-only (loses the calibrated probability the shortlist needs).

**Output:** rank eligible top-30 by calibrated confidence (both directions pooled) → **daily top-3** with {rank, symbol, CALL/PUT, confidence, expected move %}.

---

## 10. Training Agenda

- **Walk-forward, time-ordered, no leakage:** base-fit ≤ 2022-12-31 → **isotonic-calibrate on 2023** → evaluate 2024–2026 out-of-sample. Roll forward annually in production.
- **Retrain cadence:** annually (or quarterly) with expanding window; recalibrate each cycle on the most recent held-out slice.
- **No look-ahead:** universe membership, features, and eligibility all as-of date *t*.
- Reproducibility: seed-fixed, manifest per run (features, splits, config, metric-contract version).

---

## 11. Decision / Selection Layer

Daily pipeline:
1. Compute features for all eligible (optionable, non-penny) top-30 stocks as of EOD *t*.
2. Score each (stock, side): calibrated confidence + expected move.
3. Pool both directions; rank by confidence; emit **top-3**.
4. (Optional) conviction gate: suppress days where rank-1 confidence < threshold (trades volume for precision — needed only if a higher bar is demanded).
5. Persist to `reports/daily_ranked_top3.csv`; the user trades offline (strike ≈ 2% OTM; exit near peak).

---

## 12. Expected Outcome / Performance (measured, OOS)

**Selection (top-30 @4%, calibrated, point-in-time):** precision@1 **47.1%** (≥46% every year, 51.6% in 2026); P(≥1 of top-3) **75.4%**; expected-move ~5%.

**Option economics (real bhavcopy, ~2% strike):**
- Realized **leverage** ~12–25× (median); option ≈ 2.5–3× on a typical 7% move, 5–6× on cheap (low-IV/near-expiry) premiums; leverage is *inverse* to the vol regime (cheap options in calm years self-hedge low big-move frequency).
- **EV per trade (top-20 @4%): +38% peak-close (realistic), +6% blind-hold (floor), stable +32 to +46% across 2024–2026.** top-30 will be close.

**Move supply:** top-20 @4% ≈ 1.7–2/day; combined two-tier ≈ 2.5–3/day — comfortably ≥1/day.

---

## 13. Evaluation Framework

- **OOS test sets:** 2024, 2025, 2026 (each evaluated separately; expanding-window walk-forward).
- **Primary metrics:** precision@1, hit-rate@top-3, P(≥1 of top-3), per year.
- **Calibration:** reliability table (predicted-confidence bin vs actual hit).
- **Economic metric:** realized option EV (peak-close and blind-hold), P(≥2x/3x), per year, on real bhavcopy.
- **Gold gate (ship criteria):** precision@1 ≥ 40% **every** year **and** net option EV > 0 **and** calibration error small in the 0.4–0.7 band.
- **Weak-regime smoke tests:** 2024Q2/Q3, 2025Q2/Q3, low-vol 2026 — reject if precision collapses.

---

## 14. Risks & Limitations (honest)

| Risk | Evidence | Mitigation |
|---|---|---|
| **Precision capped ~50%** (news-driven residual) | ablation: features saturate | Accept; the edge is convexity, not accuracy |
| **Concentration** (~18–25 names; calm large-caps never picked) | EV runs: 18–22 distinct stocks | Optional diversity cap (costs precision); or accept |
| **Exit dependence** — EV lives in peak-close; blind-hold only +6% | EV runs (hold-to-expiry −4% to +6%) | Offline exit discipline; favor fast movers (`days_to_peak`) |
| **Costs/slippage not yet modeled** | — | Phase 6 (NET EV gate) before go-live |
| **Look-ahead / penny contamination** | caught: IDEA 24–62% in bad runs | Point-in-time eligibility (§5) — enforced |
| **Option liquidity on some strikes** | bhavcopy OI/vol checks | Require OI>0, premium floor; prefer near-ATM/2% |

---

## 15. Implementation Plan (phases)

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Outcome/label module (`clean_move_contract.py`) + tests | ✅ done |
| 2 | Lean feature assembly + point-in-time eligibility | prototyped in `analysis/` |
| 3 | Train + isotonic-calibrate (per side) + expected-move regressor | prototyped |
| 4 | Selection/output layer → daily ranked top-3 CSV | prototyped (`top30_4_ranked.py`) |
| 5 | Real option-EV backtest harness | prototyped (`final_option_ev.py`) |
| 6 | **Cost/slippage model → NET EV** (ship gate) | ⏳ next |
| 7 | Productionize: `src/koscine3/` modules, daily job, manifest, monitoring/retrain | ⏳ |

**Productionization target structure:** `data/feature_assembly.py`, `outcomes/clean_move_contract.py` (done), `models/large_move_model.py` (train+calibrate+regress), `selection/daily_ranker.py`, `evaluation/large_move_metrics.py`, `experiments/run_large_move.py`, CLI daily job.

---

## 16. Open Items

1. **NET EV after costs** (option bid-ask ~1–3%, STT, brokerage) — the final ship gate.
2. **Concentration policy** — accept the ~20-name book or impose a diversity cap (quantified trade-off available).
3. **Monitoring** — track live precision@1, calibration drift, EV; auto-alert on regime shift (low-vol starve).
4. **Retrain cadence** — annual expanding-window vs quarterly.

---

## Appendix A — Experiment log (decisions → evidence)

| Decision | Experiment | Finding |
|---|---|---|
| Clean-move + options reframe | base-rate scans | ~30% clean-move base rate, regime-stable 2010-26 |
| Predictable | predictiveness probe | model beats random on clean + ceiling, all years |
| Drop the 0.6×ATR equity stop for options | leverage analysis | downside = premium; ceiling is the only target |
| Lean features, level-dominated | `level_vs_timing_ablation.py` | LEVEL ≈ ALL ≫ TIMING; engineered ~nothing |
| Classifier over LambdaMART | `final_selection_pipeline.py` | ranker worse (34.9 vs 38.6 prec@1) |
| Train broad, trade narrow | `train_scope_test.py` | broad > focused on AUC + precision |
| Keep low-ATR stocks in training | `low_atr_trim_test.py` | trimming hurt AUC |
| top-30 @4% (vs @5%, vs tiers) | `tier_precision_4_7.py`, `top30_4_ranked.py` | 4% → 47% precision; ≥10%/mid-cap tiers cap ~24-30% |
| Point-in-time + non-penny universe | v3 contamination debug | 67% no_chain (look-ahead) + IDEA pennies removed |
| Real option EV positive | `final_option_ev.py` | +38% peak / +6% blind-hold, stable 2024-26 |
| Confidence is trustworthy | `top30_4_ranked.py` calibration | 0.51 predicted → 50.8% actual |
