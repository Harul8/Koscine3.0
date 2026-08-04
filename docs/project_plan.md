# Koscine 3.0 Ground-Up Project Plan

## Reset Decision

Koscine 3.0 is a new project rooted at:

`C:\Users\rahul\Koscine 3.0`

Koscine 2.0 is not the base implementation. The new project will not reuse its
models, predictions, overlays, GO labels, promotion gates, report definitions, or
API/UI logic. The only allowed dependency is the ready market data table:

`C:\Users\rahul\Koscine 3.0\data\processed\daily_features.parquet` (consolidated into K3; pipeline rebuilds it from `data/raw` → `data/silver` → here)

Verified data inventory:

- Rows: 1,314,313
- Columns: 193
- Symbols: 447
- Date range: 2010-01-04 through 2026-06-05
- Raw market fields include OHLC, volume, turnover, delivery, futures, options,
  volatility, market, sector, calendar, trend, compression, and breadth features.
- The table also contains future and label columns. Those are outcome fields only
  and must be blocked from model features.

## Objective

Build an AI swing-trading decision tool that answers one question:

Given end-of-day data on date `t`, which small set of long and short trades should
be entered at the next trading day open because they have a favorable 5-trading-day
peak opportunity and low opposite-close risk?

The product output is not a rare-event probability and not a broad watchlist. It is
a small actionable shortlist.

## Non-Negotiable Contract

Signal timing:

- Signal date: EOD on trading day `t`
- Entry: next trading day open
- Evaluation window: 5 trading days after entry

Long outcome:

- Favorable move: max high in the window divided by entry open minus 1
- Opposite close: target not hit and 5-day close is below entry open

Short outcome:

- Favorable move: entry open divided by min low in the window minus 1
- Opposite close: target not hit and 5-day close is above entry open

Verdicts:

- Hit: favorable move reaches the branch threshold
- Near: favorable move reaches at least 80 percent of threshold but not full threshold
- Opposite: target not hit and 5-day signed close is against the signal side
- Small: target not hit, not near, and not opposite
- Pending entry: next open is not available yet
- Pending window: entry exists but the full 5-day window is not complete

Primary gold metric:

- Hit plus near stability across time

Promotion bar:

- Hit plus near >= 60 percent by year and by quarter
- Opposite <= 20 percent preferred
- Opposite <= 25 percent hard maximum
- Enough calls to produce a useful weekly shortlist

Secondary metrics:

- Hit rate
- Near rate
- Opposite rate
- Small rate
- Average favorable move
- Median favorable move
- Average signed 5-day close return
- Net return after simple costs
- Daily signal count
- Side split
- Symbol concentration
- Sector concentration
- Top mover capture

## Data Rules

Feature columns may include only information available at EOD `t`.

Blocked feature prefixes and columns:

- `future_`
- `entry_`
- `up_move_`
- `down_move_`
- `fwd_return_`
- `long_adverse_`
- `short_adverse_`
- `label_`

Allowed uses for blocked columns:

- Outcome verification
- Unit tests
- Backward compatibility checks against recomputed outcomes

The first build must recompute outcomes from OHLC and calendar order rather than
trust old label columns.

## Universe Plan

Koscine 3.0 will derive the tradable universe from the ready data instead of
importing the old liquid/rest lists.

Universe builder requirements:

- Minimum trading history coverage
- Minimum recent turnover
- Minimum recent volume
- Maximum missing OHLC rate
- Symbol-level liquidity/risk band assignment
- Optional cap to the top 65 symbols for the first production target

Initial threshold policy:

- Liquid band: 4 percent favorable peak target
- Wider band: 7 percent favorable peak target

The band assignment must come from data-driven liquidity and volatility rules,
not from old Koscine 2.0 config files.

## Architecture

Planned source layout:

```text
src/koscine3/
  data/
    sources.py
    calendar.py
    universe.py
    feature_registry.py
  outcomes/
    swing_contract.py
  datasets/
    supervised_builder.py
    splits.py
  models/
    baselines.py
    train.py
    calibrate.py
  selection/
    daily_selector.py
    constraints.py
  evaluation/
    gold_metrics.py
    reports.py
  experiments/
    run_experiment.py
  cli.py
tests/
  test_swing_contract.py
  test_feature_registry.py
  test_universe.py
  test_gold_metrics.py
```

## Modeling Direction

The model should optimize trade utility, not target-touch probability alone.

For each side, estimate:

- Probability of hit or near
- Probability of opposite close
- Expected favorable peak move
- Expected signed 5-day close return

Candidate score:

```text
score = P(hit_or_near) * expected_favorable_move
        - opposite_penalty * P(opposite)
        - concentration_penalty
```

Selection is a separate constrained decision layer:

- Per-day max signal count
- Per-side max signal count
- Symbol cooldown
- Sector concentration cap
- Minimum score margin
- Minimum calibrated confidence

No overlays. No rescue labels. No post-hoc GO patches.

## Validation Design

Every experiment must report the same gold contract.

Required splits:

- Train <= 2023-12-31, validate 2024
- Train <= 2024-12-31, validate 2025
- Train <= 2025-12-31, test 2026-01-01 through 2026-06-05

Required slices:

- Aggregate
- Year
- Quarter
- Month
- Side
- Liquidity/risk band
- Symbol
- Sector if available
- Daily signal count

Hard rule:

- A model that works only in Jan-Jun 2026 is rejected.

Weak-regime smoke tests:

- 2024Q2
- 2024Q3
- 2025Q2
- 2025Q3

## Phased Build

Phase 0: Clean root and data contract

- Create project skeleton under Koscine 3.0
- Record data source manifest
- Define blocked outcome columns
- Define accepted raw market columns

Phase 1: Outcome engine

- Recompute next-open entry and 5-day windows from OHLC
- Implement long and short hit, near, opposite, small, pending entry, pending window
- Unit-test edge cases

Phase 2: Universe and feature registry

- Build data-driven universe selection
- Build leakage-proof feature registry
- Produce an audit report listing included and excluded columns

Phase 3: Baseline models

- Train transparent side-specific baselines
- Evaluate 2024, 2025, and Jan-Jun 2026
- Establish naive baselines before using heavier AI models

Phase 4: Robust AI model

- Train calibrated models for hit/near, opposite, and favorable move
- Compare algorithms under the same splits
- Reject models that fail weak-regime smoke tests

Phase 5: Daily selector

- Convert model scores into a small actionable shortlist
- Enforce daily, side, symbol, and sector constraints
- Produce explainable reason fields based on model outputs and constraints

Phase 6: Reporting

- Generate gold reports for every run
- Store manifests with data range, feature list, split policy, model config, and metric contract version
- Make every promoted result reproducible

Phase 7: UI/API

- Build only after the evaluator and selector are stable
- UI shows date, side, entry plan, target, confidence, opposite risk, status, and evaluated verdict
- Research candidates are visually and logically separate from promoted signals

## First Milestone

Build and test the Koscine 3.0 outcome engine.

Deliverables:

- `src/koscine3/outcomes/swing_contract.py`
- `tests/test_swing_contract.py`
- A first report proving recomputed outcomes match the expected next-open and 5-day-window logic on selected examples

This milestone comes before model training because every later decision depends on
the correctness of the outcome contract.

