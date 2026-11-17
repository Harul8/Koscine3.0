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

The last three must be run once, in that order, before any backtest.

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

## 4. Index search and direction research

```bash
python -m koscine4.harness.search --grid nifty_credit_spread_grid
python -m koscine4.harness.search --grid nifty_iron_condor_grid
python -m koscine4.harness.search --grid banknifty_credit_spread_grid
python -m koscine4.harness.search --grid banknifty_iron_condor_grid
python -m koscine4.harness.run --strategy nifty_long_straddle
python -m koscine4.models.direction --group NIFTY --horizon 1      # also BANKNIFTY and stocks; horizons 1, 3, 5
```

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
setting and a gate 3 record for the strategy. Fills are recorded in the ledger
(`koscine4.ledger.book.Ledger.record_fill`, or the `/v1/ledger/fills` web route once the
router in `koscine4/api/routes.py` is added to `api/main.py`).

## Where things are

| What | Where |
|---|---|
| Settings, costs, limits | `configs/settings.toml` |
| Strategy rules | `configs/strategies/*.toml` |
| Model definitions | `configs/models/*.toml` |
| Search grids | `configs/searches/*.toml` |
| Corporate actions to review | `records/store/adjustments_review.csv`; corrections go in `configs/adjustment_overrides.toml` |
| Data check results | `records/store/check_issues.csv` |
| Monthly lists | `records/universe/top_list.csv` |
| Run log, reports, gate records | `records/run_log/`, `records/results/`, `records/gates/` |
| Tests | `python -m pytest tests/v1` |
