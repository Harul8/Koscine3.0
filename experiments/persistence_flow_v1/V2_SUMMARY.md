# v2 mover-picker — quarterly walk-forward 2024-26, per-group summary

Quarterly-retrained (10 quarters) vs atm_iv rule. Picks ranked within (date, group); NO t+3 cooldown applied
(so concentration is worst-case). closed_opp = spiked one way but 5-day close ended the other (whipsaw vs the
realized dominant side) — NOT the ex-ante directional loss (that's still ~50%, a coin flip → we trade agnostically).

## Ranker: atm_iv still wins (quarterly retrain does NOT help)
top-3 combined: atm_iv move>6% **48.1%** / >8% 29.0% / in_top5 41.6%  vs  model 43.3 / 26.4 / 36.0.
→ Implied vol is the ceiling even with quarterly retraining. **v2 selector = atm_iv rank (rule, no model/retrain needed).**

## Summary — ranker = atm_iv
| group | depth | trades/yr | move>6% | move>8% | in_top3 | in_top5 | closed_opp(whipsaw) | coverage | top5_share |
|---|---|---|---|---|---|---|---|---|---|
| A_mcap30 | top3 | 704 | 38.6% | 23.1% | 28.6% | 40.8% | 13.3% | 30/30 | **77.4%** |
| A_mcap30 | top5 | 1174 | 35.9% | 20.1% | 26.1% | 38.0% | 13.8% | 30/30 | 66.3% |
| B_turn35 | top3 | 704 | **57.5%** | **34.9%** | 27.6% | 42.5% | 13.9% | 30/35 | 50.0% |
| B_turn35 | top5 | 1174 | 56.0% | 33.7% | 25.7% | 39.8% | 13.6% | 31/35 | 46.6% |

Most-picked (top-3 depth):
- **A_mcap30: ADANIENT(425), HAL(397), BEL(329)**, ADANIPORTS(90), INFY(71) — 3 names dominate.
- B_turn35: BSE(202), MCX(166), KAYNES(166), DIXON(160), PFC(153) — spread.

## Two key findings
1. **Group B is the strong convexity book**: 57% of picks move ≥6%, 35% ≥8% (vs A's 39% / 23%). Mega-caps (A) just move less. B is also well-diversified (top-5 = 50%).
2. **Group A is over-concentrated**: ADANIENT/HAL/BEL = ~77% of top-3 picks. atm_iv funnels A to a few perennial high-IV names (Adani, defense PSUs). With t+3 cooldown (not applied here) this roughly halves, but A still leans on a handful of names. The quarterly model de-concentrates A (top-5 55% vs 77%) but at a quality cost (>6% 33% vs 39%).
3. **Whipsaw is low (~13-14%)**: once the big move happens it usually holds to the close on the same side. But *which* side ex-ante is still ~50/50 — hence direction-agnostic (straddle / exit-at-peak).

## Open decisions before locking v2
- A's concentration: apply t+3 cooldown (realistic) and/or a per-stock frequency cap, or rank A by *relative* IV (iv-gap) to spread it?
- Group A weaker (39% ≥6%) vs B (57%): keep A for mega-cap exposure, or weight the book toward B?
