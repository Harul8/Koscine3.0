"""The walk-forward trade engine (requirements V-1, V-4 to V-7, V-14, V-15).

Session by session, in this order:

    evening strategies                       14:30 strategies
    1. open: fill yesterday's accepted       1. 15:15: mark open positions, apply exits
       candidates at the opening price       2. 14:30: ask for candidates, select
    2. close: mark positions, apply exits    3. 15:15: fill the accepted candidates
    3. evening: ask for candidates, select

Fills. A leg fills only at a price that traded, on a day with at least the minimum volume.
With daily data the entry is the opening price (evening strategies) or the day's close as a
stand-in for 15:15 (14:30 strategies); both are marked lower trust. Where the 5-minute index
chain exists, 15:15 prices come from it. Prices are recorded without slippage; slippage and
costs are applied when the result is computed, so one run can be re-priced at any slippage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from koscine4.harness.costs import order_cost, rates_on, with_slippage
from koscine4.harness.market import MarketData
from koscine4.selection.select import BookState, select
from koscine4.strategies.base import Candidate, PositionState, Strategy, _row
from koscine4.strategies.structures import BUY, max_loss_per_share

TRADE_COLUMNS = [
    "strategy", "book", "instrument", "decision_date", "entry_date", "exit_date", "sessions_held", "exit_reason",
    "legs", "lot", "entry_prices", "exit_prices", "score", "entry_trust", "exit_trust", "max_loss_per_share", "expiry",
]


@dataclass
class Position:
    cand: Candidate
    entry_date: pd.Timestamp
    entry_prices: list[float]
    entry_trust: str
    risk: float
    sessions_held: int = 0


@dataclass
class RunResult:
    trades: pd.DataFrame
    skips: list[dict] = field(default_factory=list)
    open_positions: list[Position] = field(default_factory=list)


def _leg_price(data: MarketData, instrument: str, leg, date, when: str, min_volume: float):
    """(price, trust) for one leg, or (None, reason). `when` is "open", "close" or "1515"."""
    if when == "1515":
        intraday = data.intraday(instrument)
        if intraday is not None and intraday.has_day(date):
            hit = intraday.price(date, leg.expiry, leg.strike, leg.opt_type)
            if hit is None or not hit[1] > 0:
                return None, "no_trade_at_1515"
            return hit[0], "intraday"
        when = "close"
        trust = "daily_close_for_1515"
    else:
        trust = "daily_open" if when == "open" else "daily_close"
    row = _row(data.chain(instrument, date), leg)
    if row is None:
        return None, "leg_not_listed"
    if not row["volume"] >= max(min_volume, 1):
        return None, "no_volume"
    price = row[when]
    if not price > 0:
        return None, "no_price"
    return price, trust


def _mark(data: MarketData, instrument: str, leg, date, when: str):
    """A price to value an open leg: a traded price if there is one, else the exchange settlement price."""
    price, trust = _leg_price(data, instrument, leg, date, when, 1)
    if price is not None:
        return price, trust
    row = _row(data.chain(instrument, date), leg)
    if row is not None and row["settle"] > 0:
        return row["settle"], "settlement_price"
    return None, "no_mark"


def run_engine(strategy: Strategy, data: MarketData, start, end, limits: dict, groups: dict | None = None,
               apply_weekly_cap: bool = True) -> RunResult:
    spec = strategy.spec
    evening = spec.decision_time == "evening"
    entry_when, mark_when = ("open", "close") if evening else ("1515", "1515")
    min_volume = float(spec.liquidity["min_volume"])
    max_per_day = spec.selection.get("max_per_day")
    days = data.calendar.between(start, end)
    state = BookState()
    # Outcome-building runs set allow_overlap: every signal is taken, even while an earlier trade on the
    # same instrument is open, so each decision day gets a result. Trading runs never set it (S-8).
    overlap = bool(spec.selection.get("allow_overlap", False))
    positions: dict[tuple, Position] = {}
    counter = [0]
    pending: list[Candidate] = []
    trades: list[dict] = []
    skips: list[dict] = []

    def skip(date, instrument, reason, **more):
        skips.append({"date": date, "instrument": instrument, "reason": reason, **more})

    def enter(c: Candidate, date) -> None:
        prices, trusts = [], []
        for leg in c.legs:
            price, trust = _leg_price(data, c.instrument, leg, date, entry_when, min_volume)
            if price is None:
                skip(date, c.instrument, f"entry:{trust}", leg=leg.label())          # V-5: skipped and counted
                state.open_instruments.discard(c.instrument)
                return
            prices.append(price)
            trusts.append(trust)
        risk = max_loss_per_share(c.legs, prices)
        if not risk > 0:
            skip(date, c.instrument, "entry:no_risk_at_entry_prices")
            state.open_instruments.discard(c.instrument)
            return
        counter[0] += 1
        positions[(c.instrument, counter[0])] = Position(c, date, prices, sorted(set(trusts))[0] if len(set(trusts)) == 1 else "mixed", risk)
        state.record_start(date, c.book)

    def manage_all(date) -> None:
        for key in list(positions):
            instrument = key[0]
            pos = positions[key]
            pos.sessions_held += 1
            marks = [_mark(data, instrument, leg, date, mark_when) for leg in pos.cand.legs]
            if any(m[0] is None for m in marks):
                expiry = min(l.expiry for l in pos.cand.legs)
                if date >= expiry:                       # should not happen: the safety exit comes first
                    spot = data.raw_price(instrument, date)
                    marks = [((max(spot - l.strike, 0.0) if l.opt_type == "CE" else max(l.strike - spot, 0.0)), "intrinsic_at_expiry") for l in pos.cand.legs]
                    reason = "expired"
                else:
                    continue                             # no price today: carry and try the next session
            else:
                st = PositionState(instrument, pos.cand.legs, pos.entry_prices, [m[0] for m in marks], pos.risk, pos.sessions_held, date)
                action = strategy.manage(st, None)
                if not action.exit:
                    continue
                reason = action.reason
            trusts = sorted({m[1] for m in marks})
            trades.append({
                "strategy": spec.name, "book": spec.book, "instrument": instrument, "decision_date": pos.cand.decision_date,
                "entry_date": pos.entry_date, "exit_date": date, "sessions_held": pos.sessions_held, "exit_reason": reason,
                "legs": [(str(l.expiry.date()), l.strike, l.opt_type, l.side) for l in pos.cand.legs], "lot": pos.cand.lot,
                "entry_prices": list(pos.entry_prices), "exit_prices": [float(m[0]) for m in marks], "score": pos.cand.score,
                "entry_trust": pos.entry_trust, "exit_trust": trusts[0] if len(trusts) == 1 else "mixed",
                "max_loss_per_share": pos.risk, "expiry": min(l.expiry for l in pos.cand.legs),
            })
            del positions[key]
            state.open_instruments.discard(instrument)

    def decide(date):
        view = data.view(date, spec.decision_time)
        cands, refused = strategy.candidates(view)
        for r in refused:
            skip(date, r["instrument"], f"candidate:{r['reason']}")
        entry_date = date if not evening else None
        accepted, refused = select(cands, state, date if not evening else date + pd.Timedelta(days=1), limits, groups,
                                   apply_weekly_cap=apply_weekly_cap, max_per_day=max_per_day)
        for r in refused:
            skip(date, r["instrument"], f"selection:{r['reason']}")
        if not overlap:
            for c in accepted:
                state.open_instruments.add(c.instrument)
        return accepted

    for date in days:
        if evening:
            for c in pending:
                enter(c, date)
            pending = []
            manage_all(date)
            pending = decide(date)
        else:
            manage_all(date)
            for c in decide(date):
                enter(c, date)

    return RunResult(pd.DataFrame(trades, columns=TRADE_COLUMNS), skips, list(positions.values()))


def price_trades(trades: pd.DataFrame, costs: dict, slippage_pct: float) -> pd.DataFrame:
    """Add the money columns for one slippage level: per-lot profit after costs, and as a share of maximum loss."""
    out = trades.copy()
    pnl, cost_col, risk_col, turn = [], [], [], []
    for t in trades.itertuples(index=False):
        sides = [l[3] for l in t.legs]
        entry = [with_slippage(p, s == BUY, slippage_pct) for p, s in zip(t.entry_prices, sides)]
        exit_ = [with_slippage(p, s != BUY, slippage_pct) for p, s in zip(t.exit_prices, sides)]
        gross = sum(s * (x - e) for s, e, x in zip(sides, entry, exit_)) * t.lot
        r_in, r_out = rates_on(costs, t.entry_date), rates_on(costs, t.exit_date)
        cost = sum(order_cost(e, t.lot, s == BUY, r_in) for e, s in zip(entry, sides))
        cost += sum(order_cost(x, t.lot, s != BUY, r_out) for x, s in zip(exit_, sides) if x > 0)
        legs = [type("L", (), {"expiry": l[0], "strike": l[1], "opt_type": l[2], "side": l[3]})() for l in t.legs]
        risk = max_loss_per_share(legs, entry) * t.lot
        pnl.append(gross - cost)
        cost_col.append(cost)
        risk_col.append(risk)
        turn.append(sum(e + x for e, x in zip(t.entry_prices, t.exit_prices)) * t.lot)
    out["pnl_per_lot"] = pnl
    out["costs_per_lot"] = cost_col
    out["max_loss_per_lot"] = risk_col
    out["turnover_per_lot"] = turn
    out["result_pct"] = [100.0 * p / r if r > 0 else float("nan") for p, r in zip(pnl, risk_col)]
    out["slippage_pct"] = slippage_pct
    return out
