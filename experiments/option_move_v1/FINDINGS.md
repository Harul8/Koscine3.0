# option_move_v1 — model option CONTRACTS directly (vega/gamma, not just the underlying)

Question: can predicting the OPTION contract's move (incl. IV expansion = vega, and gamma/structure) beat the
stock-move approach? Dataset: 602,992 contracts, ATM+/-5% ladder (CE & PE), real bhavcopy 2024-26, per-strike
BS-IV + OI/volume + 5-day forward peak/held. CatBoost, purged WF.

## Result — looks better, but for the WRONG reasons (thesis NOT supported)
- AUC predict big contract move (peak>=2x): stock-only **0.557** -> +option-microstructure **0.637** (+0.08).
- Selection top-3 contracts: model **peak 2.87x** vs atm_iv 1.70x vs random 1.69x.
- **Decomposition kills it:** IV/vega importance **3.0%**, structure 35.3%, flow 5.7%, other-stock 56.1%.
  Top feature = **`is_call` 20.3%** = the model picks CALLS (which paid in the 2024-26 up-market) = a
  DIRECTION/regime bet (regime-fragile; and the user takes direction offline). Rest of lift = `prem_pct`+`dte`/
  gamma = mechanical LEVERAGE of cheap/near-expiry options (already known; amplifies losses too; 2.87x is the
  ORACLE peak, not realized — cheap near-expiry contracts decay to ~0 with no move).
- **Vega prize is small:** cheap-IV contracts peak 1.91x vs priciest 1.59x; IV-mean-reversion (atm_iv_ratio_20) corr ~0.

## Verdict
Modeling the contract directly does NOT unlock a robust direction-agnostic predictive edge over v3. The apparent
gain is (a) is_call market-drift (unusable / regime-fragile) and (b) leverage of cheap/near-expiry options (known,
double-edged). The **IV-expansion / vega thesis did not pan out** (3% importance). **Do not promote a contract
model over v3.** Keep the stack: v3 (direction-agnostic stock-magnitude precision, per-group A/B) for stock
selection + [[option-gain-structure]] rules (cheap, near-expiry, ATM+-2-5%, exit day 3-4) for the contract to buy.
Generalizes/confirms [[cheap-convexity-finding]] (VRP wall) at the contract level. PROD untouched.
