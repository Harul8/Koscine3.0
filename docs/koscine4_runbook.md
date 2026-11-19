# Koscine 4 runbook

What to run, in order. Every command is run from the project root with `PYTHONPATH=src`.
Nothing here starts by itself: training runs and backtests start only when you run them.

## 1. Data (built on 17 November 2026; rerun after new data arrives)

| Step | Command | Time | State |
|---|---|---|---|
| Exchange report files | `python -m koscine4.sources.exchange_reports --start 2026-09-02 --end <today>` | about 1 s per day | Done to 1 Sep 2026 |
| Store: prices, adjustments, market value, checks | `python -m koscine4.store.build --lots` | under 1 min | Done |
| Monthly stock lists | `python -m koscine4.universe.build` | seconds | Done, 188 months |
| Option surface and quotes | `python -m koscine4.store.surface` | several minutes | **Not run** |
| Features | `python -m koscine4.features.build --time evening` then `--time 1430` | a few minutes each | **Not run** |
| Outcomes | `python -m koscine4.outcomes.build` | about a minute | **Not run** |

The last three must be run once, in that order, before any backtest. Features must be rebuilt
after 18 November 2026: that change added 16 percentile columns (`*_pctile`, F-7).

The 5-minute files under `data/intraday/` (Nifty and Bank Nifty bars and option chains) must be no
more than `[store].max_staleness_days` old. A 14:30 daily run stops when a 15:15 strategy's index
has stale files; an evening run lists them as data warnings (D-10).

## 2. Baselines and the three old ideas (gate 1)

```bash
python -m koscine4.harness.run --strategy baseline_take_all_straddle
python -m koscine4.harness.run --strategy baseline_iv_rank_straddle
python -m koscine4.harness.run --strategy baseline_rv_rank_straddle
python -m koscine4.harness.run --strategy straddle_cheap_vs_realised
python -m koscine4.harness.run --strategy nifty_put_credit_spread_1515
python -m koscine4.harness.run --strategy banknifty_put_credit_spread_1515
```

Each prints a report and saves it under `records/results/`. A normal run stops 12 months
before the last data date; those months stay sealed.

## 3. Models, then the strategies that use them

```bash
python -m koscine4.models.train --model straddle_edge      # needs baseline_take_all_straddle first
python -m koscine4.models.train --model mover_abs_5d
python -m koscine4.models.train --model move_vs_implied_5d
python -m koscine4.harness.run --strategy straddle_model
python -m koscine4.harness.run --strategy stock_mover_book
```

A classifier's probabilities are corrected (M-6, `calibration = "platt"` in the model file; set
`"none"` to keep them raw). Each correction is fitted only on earlier out-of-sample predictions
whose outcomes had ended by that model's cutoff, and is saved beside the model as
`calibration.json`. The model report has a calibration table and says whether every tenth is
within `[models].calibration_tolerance`. Models trained before 18 November 2026 have no
correction; retrain them.

## 4. Index search and direction research

```bash
python -m koscine4.harness.search --grid nifty_credit_spread_grid --fills daily
python -m koscine4.harness.search --grid nifty_credit_spread_grid --fills 5m
python -m koscine4.harness.search --grid nifty_credit_spread_grid --compare
# the same three for nifty_iron_condor_grid, banknifty_credit_spread_grid, banknifty_iron_condor_grid
python -m koscine4.harness.run --strategy nifty_long_straddle
python -m koscine4.models.direction --group NIFTY --horizon 1      # also BANKNIFTY and stocks; horizons 1, 3, 5
```

- `--fills daily` runs every cell on the whole history with fills from the daily file, and
  chooses the setting on the early part. `--fills 5m` runs every cell again with 15:15 fills
  from the 5-minute option chain, over the months those files cover; nothing is chosen there.
  `--compare` puts the two side by side on the same months (IX-5) and writes
  `records/results/search_<grid>_compare.md`. It runs no backtest.
- Every index strategy states `expiry.kind`: `"nearest"`, `"weekly"` or `"monthly"`. The six
  index strategies take the nearest expiry (owner's decision, 2026-11-19), so Nifty trades are
  mostly weeklies and Bank Nifty trades are weeklies until November 2024, when its weeklies
  stopped, and monthlies after. Every run report has a "By expiry kind" table, so the two are
  never pooled silently (IX-7). A grid can still try `"expiry.kind" = ["weekly", "monthly"]`;
  a setting is then chosen for each kind on its own.
- An iron condor or long strangle may have a `[tilt]` section (DR-7): a direction model and
  the strike settings for an "up" and a "down" variant. The trade is decided without the
  forecast; a confident forecast only moves the strikes. The daily run uses the forecast only
  when that direction model has a gate 2 pass.

## 5. Gate decisions (yours)

```bash
python -m koscine4.harness.gate check --run <run_id> --baseline-run <run_id>
python -m koscine4.harness.gate record --strategy <name> --gate 2 --decision pass --runs <run_id> --note "..."
python -m koscine4.harness.gate propose-risk --run <run_id> --capital 1000000 --largest-loss 100000
```

`check` only measures a run against the gate 2 numbers in `configs/settings.toml`. `record`
writes your decision. `propose-risk` suggests risk settings; copy the ones you accept into
the `[risk]` section yourself.

## 6. Daily use (after a strategy has a gate 2 record)

```bash
python -m koscine4.daily.run --date <today> --time evening --paper
python -m koscine4.daily.run --date <today> --time 1430 --paper
```

The list is written to `records/daily/`. Without `--paper` the run needs every `[risk]`
setting and a gate 3 record for the strategy.

- **Expected result (S-5).** Each candidate gets an expected result after costs, as a % of its
  maximum loss: the model's own prediction when the model was trained on that result, otherwise
  the mean result of the strategy's earlier trades with a similar score (the runs behind its
  gate 2 pass, then its closed ledger trades). Candidates from all strategies are ranked by it;
  one with fewer than `[expected_result].min_trades` known trades has none yet and ranks last.
  The harness ranks the same way, so backtests and the daily list agree.
- **Margin (R-6).** The margin shown and used for sizing is a scenario estimate in the manner
  of SPAN: what the position would owe at the worst of a set of price and volatility moves
  (the premium received is not netted off), plus exposure margin on each sold leg. The
  `[margin]` numbers were fitted on 2026-11-19 to Zerodha's F&O calculator: about an 8.75% price
  scan for the indices and 13.5% for RELIANCE, exposure 1.9% (index) and 3.5% (stock). Five
  spreads came out 1% to 7% above Zerodha's total. The old maximum loss x 1.25 was 43% to 82%
  below it. Exposure margin is about as large as SPAN on a spread, and an iron condor pays it
  on both sold legs. Refit if the exchange changes its method. The fallback, when a leg's
  volatility cannot be worked out, is the spread's width x `margin_factor` plus exposure.
- **Costs at expiry.** A trade held to expiry pays no exchange charge or premium tax on the way
  out, but each leg pays the broker's flat brokerage (Zerodha charges it on expired, exercised
  and assigned contracts), and a bought option in the money pays 0.15% of its intrinsic value.
- **Fills (O-6, O-7).** The web app's **Trade List** tab shows the saved lists, the open and
  closed trades, and a form to record each leg's entry and exit fill, mark a recommendation not
  taken, and close a trade from its fills. The same actions are web routes under `/v1`
  (`koscine4/api/routes.py`, included by `api/main.py`).

## Where things are

| What | Where |
|---|---|
| Settings, costs, limits | `configs/settings.toml` |
| Strategy rules | `configs/strategies/*.toml` |
| Model definitions | `configs/models/*.toml` |
| Search grids | `configs/searches/*.toml` |
| Expected result, margin and calibration settings | `[expected_result]`, `[margin]`, `[models]` in `configs/settings.toml` |
| Corporate actions to review | `records/store/adjustments_review.csv`; corrections go in `configs/adjustment_overrides.toml` |
| Data check results | `records/store/check_issues.csv` |
| Monthly lists | `records/universe/top_list.csv` |
| Run log, reports, gate records | `records/run_log/`, `records/results/`, `records/gates/` |
| Tests | `python -m pytest tests/v1` |
