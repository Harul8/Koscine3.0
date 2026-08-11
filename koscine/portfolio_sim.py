"""Walks the calibrated Condor/Broken-Wing/Skew signal pool chronologically against a fixed
capital pool, 1 lot per position (per user decision -- no risk-based multi-lot sizing), filling
new candidates while capital allows -- fill-and-hold, no rotation (allow_rotation defaults to
False). Rotation (exit the worst held position early for a clearly better new one) was built
and backtested: it fired only 19 times over 2 years and net UNDERPERFORMED fill-and-hold by
~4.5% (cutting a position short loses more foregone decay than the new opportunity gains, on
average, in this signal flow) -- so it's kept as an opt-in parameter, not the shipped default.
Also enforces a `symbol_cooldown_days` gap (default 3 trading days) between entries on the same
stock, measured from its last ENTRY day -- a symbol that fires today won't fire again until at
least `symbol_cooldown_days` trading days later, regardless of when that position exits.

Net of real Zerodha F&O options brokerage (koscine/brokerage.py). See the approved plan
(~/.claude/plans -- "10L capital-allocation engine") for the full design rationale.

Blocked margin: real Zerodha SPAN+exposure margin for a hedged multi-leg spread is NOT the
theoretical max_risk (defined worst-case loss) -- it runs meaningfully higher, since SPAN prices
a scenario-based risk array, not just strike-width-minus-credit. We don't have Zerodha API
access (that needs the account holder's own Kite Connect key/token), but Upstox's margin API
(api.upstox.com/v2/charges/margin -- UPSTOX_ACCESS_TOKEN is already configured in this repo for
the option-chain downloaders) computes real SPAN+exposure margin including hedge benefit for a
multi-leg combo submitted together, so MARGIN_MULTIPLIER below is calibrated against 3 LIVE
queries (real current instrument keys/premiums, 2026-08-11), not a guessed or blog-sourced ratio:
  - Broken-Wing (4-leg, 2%/6%) on RELIANCE: max_risk Rs 29,775 -> final_margin Rs 74,010 = 2.49x
  - Condor (4-leg, symmetric 5%/5%) on RELIANCE: max_risk Rs 23,100 -> Rs 67,582 = 2.93x
  - Skew (2-leg vertical, 2%/6%) on TCS: max_risk Rs 27,338 -> Rs 43,812 = 1.60x
Small sample (3 points, one day, two stocks) -- still an approximation, but a real-data one, not
an assumed one. 4-leg structures (Condor/Broken-Wing) cluster near 2.5-2.9x; the 2-leg Skew
structure needs meaningfully less margin (1.6x) since it only has one side to hedge. Two
multipliers below reflect that split rather than forcing one number on both shapes.

Every leg price (entry AND exit, for both the natural-hold and rotated-out cases) comes from
koscine/../analysis/gen_daily_marks.py's per-trade daily path -- never from the signal-history
CSVs' aggregate credit/exit_value columns, which can't be split back into individual leg
premiums for per-order brokerage costing. A trade's "natural exit date" is DEFINED as the last
date its own daily-marks path covers, so a marks row always exists there by construction (no
exit-price fallback/estimate needed).

Inputs (all already built by prior steps, no new data fetching):
  locks/prod_sell_strategies/{signal_history,broken_wing_signal_history,skew_signal_history}.csv
  locks/prod_sell_strategies/{condor,broken_wing,skew}_daily_marks.parquet
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from koscine import brokerage as bk

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
LOCK_DIR = ROOT / "locks" / "prod_sell_strategies"

DEFAULT_CAPITAL = 1_000_000.0
ROTATION_BUFFER_PCT = 5.0   # a new candidate must beat the held position's remaining upside by
                             # more than (round-trip swap brokerage, as %-of-risk) + this buffer
                             # before we bother rotating -- avoids churning on marginal deltas
MARGIN_MULTIPLIER_4LEG = 2.7   # Condor / Broken-Wing -- see module docstring for the live
                                 # Upstox-margin-API calibration (2.49x, 2.93x averaged)
MARGIN_MULTIPLIER_2LEG = 1.6    # Skew -- see module docstring (live Upstox margin API)


def _margin_multiplier(strategy: str) -> float:
    return MARGIN_MULTIPLIER_2LEG if strategy == "skew" else MARGIN_MULTIPLIER_4LEG
                             # against (~3-9x max_risk observed; 5x is the conservative midpoint)


@dataclass
class Position:
    trade_id: str
    strategy: str          # "condor" | "broken_wing" | "skew"
    tier: str
    symbol: str
    entry_date: pd.Timestamp
    natural_exit_date: pd.Timestamp
    credit: float
    max_risk: float
    lot_size: int
    entry_brokerage: float
    net_entry_ror_pct: float
    exit_date: pd.Timestamp | None = None
    exit_value: float | None = None
    exit_brokerage: float = 0.0
    exit_reason: str = ""   # "natural" | "rotated_out"
    realized_pnl: float | None = None   # net of BOTH entry and exit brokerage, in rupees


def _trade_id(symbol: str, entry_date, tier: str) -> str:
    return f"{symbol}|{pd.Timestamp(entry_date).date().isoformat()}|{tier}"


def _entry_legs_from_marks(strategy: str, marks_row: pd.Series) -> list[tuple[float, bool]]:
    if strategy == "skew":
        return [(marks_row["short_entry"], False), (marks_row["long_entry"], True)]
    return [(marks_row["sc_entry"], False), (marks_row["lc_entry"], True),
            (marks_row["sp_entry"], False), (marks_row["lp_entry"], True)]


def _exit_legs_from_marks(strategy: str, marks_row: pd.Series) -> list[tuple[float, bool]]:
    if strategy == "skew":
        return [(marks_row["short_close"], True), (marks_row["long_close"], False)]
    return [(marks_row["sc_close"], True), (marks_row["lc_close"], False),
            (marks_row["sp_close"], True), (marks_row["lp_close"], False)]


def _mark_value(strategy: str, marks_row: pd.Series) -> float:
    """Combined spread value (cost to close) from one daily_marks row."""
    if strategy == "skew":
        return float(marks_row["short_close"] - marks_row["long_close"])
    return float((marks_row["sc_close"] - marks_row["lc_close"]) + (marks_row["sp_close"] - marks_row["lp_close"]))


def _load_signals() -> pd.DataFrame:
    frames = []
    c = pd.read_csv(LOCK_DIR / "signal_history.csv")
    c["strategy"] = "condor"; c["tier"] = "condor"
    frames.append(c)
    bw = pd.read_csv(LOCK_DIR / "broken_wing_signal_history.csv")
    bw["strategy"] = "broken_wing"
    frames.append(bw)
    sk = pd.read_csv(LOCK_DIR / "skew_signal_history.csv")
    sk["strategy"] = "skew"
    frames.append(sk)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["outcome"] != "pending"].copy()   # can't simulate outcomes we don't have yet
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["trade_id"] = df.apply(lambda r: _trade_id(r["symbol"], r["entry_date"], r["tier"]), axis=1)
    return df


def _load_marks() -> pd.DataFrame:
    frames = []
    for strat, fname in [("condor", "condor_daily_marks.parquet"),
                          ("broken_wing", "broken_wing_daily_marks.parquet"),
                          ("skew", "skew_daily_marks.parquet")]:
        m = pd.read_parquet(LOCK_DIR / fname)
        m["strategy"] = strat
        frames.append(m)
    marks = pd.concat(frames, ignore_index=True, sort=False)
    marks["date"] = pd.to_datetime(marks["date"])
    return marks


def build_candidates(signals: pd.DataFrame, marks: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """One dict per realized trade (everything needed to open/value it), plus a
    trade_id -> natural_exit_date Series (the last date that trade's own marks path covers)."""
    marks_by_trade = {tid: g.sort_values("date") for tid, g in marks.groupby("trade_id")}
    natural_exit_date = marks.groupby("trade_id")["date"].max()
    out = []
    for _, row in signals.iterrows():
        tid = row["trade_id"]
        g = marks_by_trade.get(tid)
        if g is None or g.empty or pd.isna(row["lot_size"]) or row["max_risk"] <= 0:
            continue
        entry_row = g.iloc[0]
        lot_size = int(row["lot_size"])
        entry_legs = _entry_legs_from_marks(row["strategy"], entry_row)
        entry_brokerage = sum(bk.leg_cost(p, lot_size, is_buy) for p, is_buy in entry_legs)
        net_credit_per_lot = row["credit"] - entry_brokerage / lot_size
        net_entry_ror = net_credit_per_lot / row["max_risk"] * 100
        out.append(dict(
            trade_id=tid, strategy=row["strategy"], tier=row["tier"], symbol=row["symbol"],
            entry_date=row["entry_date"], credit=row["credit"], max_risk=row["max_risk"],
            lot_size=lot_size, entry_brokerage=entry_brokerage, net_entry_ror_pct=net_entry_ror,
        ))
    return out, natural_exit_date


def _dedup_daily(candidates: list[dict]) -> list[dict]:
    """One-signal-per-stock-per-day, keep the most profitable (max_profit_per_lot = credit *
    lot_size) -- same rule the frontend applies (App.tsx's maxProfitScore)."""
    best: dict[tuple, dict] = {}
    for c in candidates:
        key = (c["symbol"], c["entry_date"])
        score = c["credit"] * c["lot_size"]
        cur = best.get(key)
        if cur is None or score > cur["credit"] * cur["lot_size"]:
            best[key] = c
    return list(best.values())


def _close_position(pos: Position, today: pd.Timestamp, marks_row: pd.Series, reason: str) -> None:
    pos.exit_date = today
    pos.exit_value = _mark_value(pos.strategy, marks_row)
    exit_legs = _exit_legs_from_marks(pos.strategy, marks_row)
    pos.exit_brokerage = sum(bk.leg_cost(p, pos.lot_size, is_buy) for p, is_buy in exit_legs)
    pos.exit_reason = reason
    pos.realized_pnl = (pos.credit - pos.exit_value) * pos.lot_size - pos.entry_brokerage - pos.exit_brokerage


def simulate(capital: float = DEFAULT_CAPITAL, rotation_buffer_pct: float = ROTATION_BUFFER_PCT,
             allow_rotation: bool = False, symbol_cooldown_days: int = 3) -> dict:
    signals = _load_signals()
    marks = _load_marks()
    marks_idx = marks.set_index(["trade_id", "date"]).sort_index()

    candidates, natural_exit_date = build_candidates(signals, marks)
    candidates = _dedup_daily(candidates)
    by_entry_date: dict[pd.Timestamp, list[dict]] = {}
    for c in candidates:
        by_entry_date.setdefault(c["entry_date"], []).append(c)

    free_capital = capital
    open_positions: dict[str, Position] = {}
    trade_log: list[Position] = []
    equity_curve = []

    all_dates = sorted(marks["date"].unique())
    date_idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    last_entry_idx: dict[str, int] = {}   # symbol -> trading-day index of its last entry

    for today in all_dates:
        today = pd.Timestamp(today)
        today_idx = date_idx[today]

        # 1) natural exits due today
        for tid in [t for t, p in open_positions.items() if p.natural_exit_date == today]:
            pos = open_positions.pop(tid)
            key = (tid, today)
            if key in marks_idx.index:
                _close_position(pos, today, marks_idx.loc[key], "natural")
            else:
                # shouldn't happen (natural_exit_date is defined FROM this trade's own marks),
                # but never silently drop a position -- close it flat rather than leak capital.
                pos.exit_date, pos.exit_value, pos.exit_reason = today, pos.credit, "natural_no_mark"
                pos.realized_pnl = -pos.entry_brokerage
            free_capital += pos.max_risk * pos.lot_size * _margin_multiplier(pos.strategy)
            trade_log.append(pos)

        # 2) today's new candidates, ranked by net entry ROR% (best first)
        for c in sorted(by_entry_date.get(today, []), key=lambda c: c["net_entry_ror_pct"], reverse=True):
            if c["trade_id"] in open_positions:
                continue
            sym = c["symbol"]
            last_idx = last_entry_idx.get(sym)
            if last_idx is not None and today_idx - last_idx <= symbol_cooldown_days:
                continue   # same stock fired within the last symbol_cooldown_days trading days
            nx = natural_exit_date.get(c["trade_id"])
            if pd.isna(nx):
                continue
            required = c["max_risk"] * c["lot_size"] * _margin_multiplier(c["strategy"])
            new_pos = Position(trade_id=c["trade_id"], strategy=c["strategy"], tier=c["tier"],
                               symbol=c["symbol"], entry_date=today, natural_exit_date=nx,
                               credit=c["credit"], max_risk=c["max_risk"], lot_size=c["lot_size"],
                               entry_brokerage=c["entry_brokerage"], net_entry_ror_pct=c["net_entry_ror_pct"])
            if required <= free_capital:
                free_capital -= required
                open_positions[c["trade_id"]] = new_pos
                last_entry_idx[sym] = today_idx
                continue
            if not allow_rotation or not open_positions:
                continue

            # rotation: worst held position = lowest remaining theoretical upside (= today's
            # mark value, as %-of-its-own-risk -- see the plan's rotation-rule rationale)
            worst_tid, worst_remaining_pct, worst_mrow = None, None, None
            for tid, pos in open_positions.items():
                key = (tid, today)
                if key not in marks_idx.index:
                    continue
                mrow = marks_idx.loc[key]
                remaining_pct = _mark_value(pos.strategy, mrow) / pos.max_risk * 100
                if worst_remaining_pct is None or remaining_pct < worst_remaining_pct:
                    worst_tid, worst_remaining_pct, worst_mrow = tid, remaining_pct, mrow
            if worst_tid is None:
                continue
            worst_pos = open_positions[worst_tid]
            exit_legs = _exit_legs_from_marks(worst_pos.strategy, worst_mrow)
            exit_brokerage = sum(bk.leg_cost(p, worst_pos.lot_size, is_buy) for p, is_buy in exit_legs)
            swap_cost_pct = (exit_brokerage + c["entry_brokerage"]) / (worst_pos.max_risk * worst_pos.lot_size) * 100
            if c["net_entry_ror_pct"] > worst_remaining_pct + swap_cost_pct + rotation_buffer_pct:
                _close_position(worst_pos, today, worst_mrow, "rotated_out")
                trade_log.append(worst_pos)
                free_capital += worst_pos.max_risk * worst_pos.lot_size * _margin_multiplier(worst_pos.strategy)
                del open_positions[worst_tid]
                if required <= free_capital:
                    free_capital -= required
                    open_positions[c["trade_id"]] = new_pos
                    last_entry_idx[sym] = today_idx

        # 3) mark-to-market equity for today
        unrealized = 0.0
        for tid, pos in open_positions.items():
            key = (tid, today)
            if key in marks_idx.index:
                value = _mark_value(pos.strategy, marks_idx.loc[key])
                unrealized += (pos.credit - value) * pos.lot_size
        realized_so_far = sum(p.realized_pnl for p in trade_log)
        equity_curve.append(dict(date=today, capital_free=free_capital, n_open=len(open_positions),
                                 equity=capital + realized_so_far + unrealized))

    return dict(trade_log=trade_log, open_at_end=list(open_positions.values()),
                equity_curve=pd.DataFrame(equity_curve), capital=capital)
