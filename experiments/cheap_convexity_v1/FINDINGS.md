# cheap_convexity_v1 — findings

_Status: core + real-premium done; exit-timing running. **Result: the cheap-convexity signal is real and
survives real premiums — net-positive after costs when traded selectively.**_

## Setup
- Target `surprise = realized_cc_move − atm_iv·√(5/252)` (the VRP residual; >0 = moved more than priced).
- CatBoost on 35 IV-structure / skew / PCR / flow / catalyst features (orthogonal to the `atm_iv` level) + `group`.
- Purged + embargoed (5d) quarterly walk-forward 2024Q1–2026Q2, n_eval = 31,867. PROD untouched.
- Base rate: only **20.5%** of options out-move their implied (1σ); mean surprise −122 bps (strong positive VRP).

## 1) The surprise is predictable
- **rank-IC(predicted surprise, realized surprise) = +0.166**
- rank-IC(`atm_iv`, realized surprise) = **−0.183** — ranking by IV systematically buys *expensive* (negative surprise).

## 2) Proxy selector comparison (premium-adjusted straddle-PnL, bps)
| selector | k | straddle-PnL | surprise | hit P(real>impl) | realized move |
|---|---|---|---|---|---|
| cheap_convexity | 3 | **+24.3** | −35.7 | 0.328 | 2.64% |
| atm_iv baseline | 3 | −8.6 | −122.0 | 0.294 | 4.45% |
| cheap_convexity | 5 | +17.8 | −43.8 | 0.325 | 2.64% |
| atm_iv baseline | 5 | +2.7 | −104.6 | 0.306 | 4.32% |

Mechanism: the model picks **smaller movers (2.6%) that are cheap**; `atm_iv` picks **bigger movers (4.5%) that
are expensive** → mover-size ≠ profit, exactly as `persistence_flow_v1` warned. Both groups positive (A +23, B +25 bps).
Top predictors: `atm_iv`, `nifty_realized_vol_20` (regime), `days_to_earnings` (catalyst), `atm_iv_ratio_20`
(IV mean-reversion), `delivery_pct`, `oi_buildup_ratio`, `rv_over_iv` (realized-vs-implied).

## 3) Real option premiums (premium_ev.py — ATM straddle, held 5d, close→close)
| selector | n | mean EV | median | win |
|---|---|---|---|---|
| **cheap_convexity** | 3,085 | **+0.93%** | −9.5% | 0.337 |
| atm_iv baseline | 3,111 | −1.17% | −9.8% | 0.344 |

Beats baseline by **~2.1 pp** on real premiums (same sign as the proxy). Long-convexity shape: many small
losses, rare big wins (top-5% of trades ≈ 791% of gross return).

## 4) Selectivity — the edge concentrates and clears costs (analyze_ev.py)
EV is **monotonic in predicted surprise**, and the top quintiles survive realistic cost (~3%):
| pred quintile | mean EV | net @3% | win |
|---|---|---|---|
| Q1 (low) | −6.60% | −9.6% | 0.25 |
| Q2 | −0.96% | −4.0% | 0.34 |
| Q3 | +3.88% | +0.9% | 0.38 |
| Q4 | +3.86% | +0.9% | 0.40 |
| **Q5 (high)** | **+4.45%** | **+1.5%** | 0.32 |

- **Concentrated in A mega-caps**: A +2.73% (net@2% +0.73%) vs B −0.96%.
- **Improving**: 2024 −0.7% → 2025 +1.8% → 2026 +2.9%.
- **Not a penny artifact**: stripping low-premium options slightly *hurts* (real signal, not noise).

## 5) Exit timing (premium_ev_exits.py) — done
Held-to-close is right; **profit-target caps DESTROY the edge** (they clip the fat tail that is the edge).
Oracle peak shows large unrealized convexity. cheap_convexity Q4-Q5:
| exit rule | mean | net@3% | win |
|---|---|---|---|
| held | +4.89% | +1.89% | 0.37 |
| peak (oracle, NOT tradeable) | +24.12% | +21.12% | 0.64 |
| pt25 (profit-target) | −4.69% | −7.69% | 0.41 |
| pt50 (profit-target) | −2.28% | −5.28% | 0.38 |

Lesson: **let winners run** — never cap. The ~22pp held→peak gap is the prize for smarter exit timing (a
trailing stop, not a target).

## 6) Robustness — bootstrap (20k resamples, held-to-close, net@3%)
| book | n | mean | 95% CI | P(EV>0) |
|---|---|---|---|---|
| atm_iv baseline | 3385 | −4.21% | [−5.7%, −2.7%] | 0.00 |
| cheap_convexity ALL | 3361 | −1.18% | [−2.9%, +0.6%] | 0.10 |
| **cheap_convexity Q4-Q5** | 1344 | **+1.89%** | [−1.3%, +5.1%] | **0.87** |
| cheap_convexity Q4-Q5 & A | 885 | +1.20% | [−2.8%, +5.5%] | 0.70 |

`atm_iv` loses decisively after costs; the selective cheap_convexity book is **probably profitable (P≈0.87)**
but its 95% CI dips just below 0 — tail-dependent, not yet bulletproof on held-to-close alone.

## 7) Trailing-stop / shorter-hold exits (premium_ev_paths.py) — done
**None beat simple held-to-close.** cheap_convexity Q4-Q5 net@3%: held **+1.89%**, trail30 +0.82%, trail40
+0.69%, trail20 +0.40%, exit_d3 +0.35%, exit_d1/d2 negative. The +21% oracle peak is **not capturable with
daily-close rules** — winners spike intraday/single-day and collapse by the next close, so trailing stops exit
late and whipsaw. Capturing the peak needs **intraday monitoring/exit** (different infra than the daily 7AM
workflow). **Best realistic exit = hold to the 5-day close; let winners run, never cap.** (Full paths saved to
`results/premium_ev_paths.csv` — any future exit rule is now free to test.)

## Verdict
**Confirmed — a learned model identifies options the market underpriced; the first selector to beat `atm_iv`
on premium-adjusted EV.** `atm_iv` does the opposite of cheap (rank-IC −0.18 with surprise; net **−4.2%** after
costs, CI entirely <0). The cheap_convexity signal is predictable (rank-IC +0.17), monotonic (Q1 −6.6% → Q5
+4.45%), economically sensible (earnings, IV mean-reversion, realized-vs-implied), and beats baseline on proxy
+ real premiums. Best realistic config — **rank by predicted surprise, top-2 quintiles, held to 5-day close** —
is **+1.9% net@3% (P(EV>0)≈0.87)**, i.e. it flips a clearly-losing book (−4.2%) to probably-winning (+1.9%).

**Honest limits (why this is a candidate, not yet a promotion):** the edge is thin and tail-dependent (95% CI
[−1.3%, +5.1%] dips below 0; mean rides on rare winners), weaker in B / 2024, and ~90% of the theoretical
convexity (oracle peak) is unreachable without intraday exit. **Do NOT promote a v3 lock yet** — needs more
out-of-sample data, position-sizing discipline for the fat tail, and (highest upside) an intraday-exit study
to monetize the peak. The concept is proven; the daily-EOD edge is real but marginal.
