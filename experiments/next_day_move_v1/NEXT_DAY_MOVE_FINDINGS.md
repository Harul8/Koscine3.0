# Next-day (t+1) movement-magnitude prediction — findings

Target = largest distance t+1 travels from today's close, either direction:
`next_move = max((high[t+1]-close[t])/close[t], (close[t]-low[t+1])/close[t])`. Decision at EOD t.
Walk-forward (quarterly retrain) XGBoost vs atm_iv baseline. A+B + broad F&O, 2024-26 OOS. PROD untouched.

## Headline: next-day MAGNITUDE is strongly predictable (unlike direction)
Mean next-day move = 2.31% (median 1.89%).

| metric | model | atm_iv baseline |
|---|---|---|
| cross-sectional rank IC (Spearman/day) | 0.381 | **0.399** |
| AUC move ≥2% (base 46%) | 0.700 | 0.699 |
| AUC move ≥3% (base 22%) | 0.712 | 0.708 |
| AUC move ≥4% (base 11%) | 0.736 | 0.728 |
| top-1 pick in day's top-3 movers | 32.0% | 32.3% |
| avg move of top pick / capture | 3.22% / 57% | 3.29% / 58% |

- **IC ≈ 0.40 is a strong magnitude signal** (vs ~0 for direction). We reliably rank which stocks move most tomorrow.
- **atm_iv (implied vol) is the best predictor** — the ML model matches but does NOT beat it (every year). So the
  practical ranker is `atm_iv`; the model's added value is a **calibrated** point-estimate of the move %.

## Calibration — excellent (predicted move% ≈ realized)
| pred decile | predicted | realized |
|---|---|---|
| 0 (low) | 1.41% | 1.46% |
| 5 | 2.29% | 2.29% |
| 9 (high) | 4.10% | 3.56% |
Monotonic; near-diagonal (slight over-prediction only in the extreme decile). Top decile realizes 2.4× the bottom.

Top features: atm_iv (0.30), atr_pct_14, donchian_width_20, realized_vol_20 — pure volatility level.

## Conclusion
Next-day **quantum of movement is predictable and well-calibrated** (IC 0.40, AUC 0.70–0.74). This is the opposite
of direction (coin flip). The deployable predictor is essentially **rank by atm_iv** (model ≈ baseline), with the
regressor giving a calibrated tomorrow-move% estimate.

Caveat vs 5-day: next-day top-pick move ≈ **3.2%** vs 5-day ≈ **7.3%** — more predictable per day but smaller, so
for the options book (theta/premium) the 5-day horizon still monetizes better. This model is a clean standalone
next-day move-size forecaster (useful for sizing, straddle screening, or a 1-day vol play).
