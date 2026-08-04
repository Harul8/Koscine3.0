# Intraday exit study v1

## Objective and scope

Measure how much of the cheap-convexity option peak can be captured by an
executable intraday exit policy. This experiment is isolated from production
and does not change the signal selector or model lock.

The acquisition universe is intentionally limited to:

- 30 stocks selected from local NSE F&O bhavcopies for liquid, active options
  with stronger single-leg premium movement;
- NIFTY, BANKNIFTY, SENSEX and FINNIFTY.

`universe.py` is the enforced allow-list. A download manifest containing
another underlying fails before any network request.

`rank_universe.py` makes the stock selection reproducible. It uses 80 evenly
sampled sessions from 2025-01-01 through 2026-07-22, requires consistent ATM
CE and PE trading, then scores eligible stocks as:

- 65% liquidity: premium turnover, volume, minimum-leg volume, OI and
  two-leg continuity;
- 35% option-buying movement: balanced ATM CE and PE high/open upside,
  discounted when premium expansion does not persist toward the close.

The movement component deliberately evaluates calls and puts independently.
It does not use synthetic straddle movement, because the intended strategy is
to buy the directionally selected option rather than sell volatility.

## Mixed-resolution execution contract

1. Store the held CE and PE contracts as exact 1-minute candles. These are the
   execution source of truth.
2. Aggregate those synchronized candles into completed 5-minute bars for
   features and model training.
3. The signal uses the completed EOD feature row for trading day `t`, so the
   option trade starts on `t+1`.
4. The default entry decision is made after the first completed 5-minute bar:
   the 09:15--09:19 candles produce a decision timestamp of 09:20 IST.
5. Entry and exit fills use the first available 1-minute open at or after the
   decision timestamp. A 10:05 decision can therefore fill at the 10:05
   1-minute open, never inside the preceding 5-minute candle.
6. Select fixed CE and PE contracts at entry and retain the exact expiry and
   strike throughout the trade. Never backtest a rolling-ATM series as though
   it were a held contract.
7. A trade not exited earlier is closed at the final available 1-minute close,
   normally 15:20 IST on trading day `t+5`.
8. Apply fees and slippage to entry and exit. Cluster evaluation by signal
   date because picks generated on the same day are correlated.

## Canonical data

`data/intraday/options_contract_legs_1m.parquet` is long-form source data:

| column | meaning |
|---|---|
| `trade_id` | stable identifier for one fixed-strike/expiry straddle |
| `option_type` | `CE` or `PE` |
| `instrument_key` | exact expired contract identifier |
| `timestamp` | start of the 1-minute candle in Asia/Kolkata |
| `open`, `high`, `low`, `close`, `volume`, `oi` | contract candle |
| `symbol`, `signal_date`, `expiry`, `strike` | immutable trade metadata |

`prepare.py` produces:

- `options_straddle_1m.parquet`: synchronized CE+PE execution prices;
- `options_straddle_5m.parquet`: completed, end-stamped decision bars;
- `exit_training_5m.parquet`: leakage-safe features plus explicitly prefixed
  forward research targets.

CE and PE highs/lows cannot be summed into a true straddle high/low because
their extrema may occur at different instants. The preparation layer names
them `high_upper_bound` and `low_lower_bound`; the simulator and feature
builder do not use them. The timestamped 1-minute close path supplies the peak
diagnostic.

## Training contract

Features use only information known at each completed 5-minute bar:

- 1/3/6-bar straddle momentum;
- running peak and drawdown;
- rolling realized volatility;
- CE/PE value share and leg-return spread;
- time of day;
- trailing volume and OI changes when available;
- DTE and existing signal metadata.

Forward columns are prefixed `target_` so they cannot accidentally enter the
feature matrix. The initial research target labels whether the next 30 minutes
have neither positive terminal return nor at least 3% additional upside.

All splits must be by trade/signal date with an embargo, never a random
5-minute row split. Recommended evaluation:

- development: 2021--2023;
- validation: 2024;
- final untouched test: 2025--2026.

If the API cannot provide that full span, use expanding walk-forward folds and
leave the latest twelve months untouched.

## Rule families and promotion gate

The transparent benchmark grid contains:

- five-day hold;
- trailing drawdown of 15/20/25/30% after activation at
  +20/+35/+50/+75%.

A learned exit policy is considered only after this benchmark exists. It must:

- beat five-day hold after all costs;
- improve mean net EV without materially reducing right-tail contribution;
- remain positive in the untouched period and at least two subperiods;
- capture materially more of the timestamped 1-minute peak;
- survive one additional minute of execution delay and doubled slippage.

## Upstox acquisition

Status: parked until an authenticated data API is available. The code and
contracts remain ready, but no network acquisition or training run should be
started yet.

Upstox documents exact expired-contract OHLC at `1minute` and `5minute`
intervals. We download `1minute` once and derive 5-minute bars locally so
training and execution stay mathematically consistent.

The downloader accepts a two-row-per-trade manifest (one fixed CE and one fixed
PE), uses `UPSTOX_ACCESS_TOKEN`, and writes the canonical leg file:

```powershell
python experiments/intraday_exit_v1/upstox_1m_download.py `
  --manifest data/intraday/upstox_contract_manifest.csv

python experiments/intraday_exit_v1/prepare.py

python experiments/intraday_exit_v1/study.py
```

A read-only Upstox Analytics Token lasts one year and is suitable for GET
market-data calls. However, Upstox currently documents expired-option contract
lookup and expired historical candles as Upstox Plus-only. The downloader
reports `UDAPI1149` as that specific subscription requirement.
