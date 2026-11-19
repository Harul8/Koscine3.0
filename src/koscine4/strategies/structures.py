"""Option structures: which legs, and the most each can lose (requirements S-1, R-1).

Strikes are matched against raw prices, because the exchange quotes strikes in raw terms.
All money figures here are per share; multiply by the lot for rupees per lot.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from koscine4.strategies.spec import SpecError, StrategySpec

BUY, SELL = 1, -1


@dataclass(frozen=True)
class Leg:
    expiry: pd.Timestamp
    strike: float
    opt_type: str           # "CE" or "PE"
    side: int               # +1 bought, -1 sold

    def label(self) -> str:
        return f"{'BUY' if self.side == BUY else 'SELL'} {self.expiry.date()} {self.strike:g} {self.opt_type}"


def choose_expiry(chain: pd.DataFrame, date, min_dte: int, max_dte: int | None = None, kind: str = "nearest") -> pd.Timestamp | None:
    """The nearest option expiry at least `min_dte` calendar days after `date` (and at most `max_dte`).

    kind "nearest" takes either; "monthly" keeps expiries that also have a futures contract; "weekly" keeps the others (IX-7).
    """
    date = pd.Timestamp(date)
    is_opt = chain["opt_type"].isin(["CE", "PE"])
    exp = pd.DatetimeIndex(sorted(chain.loc[is_opt, "expiry"].unique()))
    if kind in ("weekly", "monthly"):
        monthly = set(pd.DatetimeIndex(chain.loc[~is_opt, "expiry"].unique()))
        exp = pd.DatetimeIndex([e for e in exp if (e in monthly) == (kind == "monthly")])
    dte = (exp - date).days
    ok = exp[(dte >= min_dte) & ((dte <= max_dte) if max_dte is not None else True)]
    return ok[0] if len(ok) else None


def expiry_kind(chain: pd.DataFrame, expiry) -> str:
    """'monthly' if the expiry also has a futures contract, else 'weekly' (IX-7)."""
    fut = chain.loc[~chain["opt_type"].isin(["CE", "PE"]), "expiry"]
    return "monthly" if pd.Timestamp(expiry) in set(pd.to_datetime(fut)) else "weekly"


def nearest_strike(strikes, target: float, max_snap_pct: float) -> float | None:
    strikes = sorted(set(float(s) for s in strikes))
    if not strikes:
        return None
    best = min(strikes, key=lambda k: abs(k - target))
    return best if abs(best / target - 1.0) * 100.0 <= max_snap_pct else None


def per_side(strikes, side: str, key: str):
    """A strike setting for one side ("put" or "call"): `put_<key>` if given, else the shared `<key>`."""
    return strikes.get(f"{side}_{key}", strikes.get(key))


def build_legs(spec: StrategySpec, chain: pd.DataFrame, spot: float, date, direction: int | None = None) -> list[Leg] | None:
    """Legs for `spec` from one day's chain, or None if a required expiry or strike is not listed.

    `direction` (+1 up, -1 down) is used only when the spec's side is "model".
    """
    expiry = choose_expiry(chain, date, int(spec.expiry["min_dte"]), spec.expiry.get("max_dte"), str(spec.expiry.get("kind", "nearest")))
    if expiry is None or not spot or spot <= 0:
        return None
    opts = chain[(chain["expiry"] == expiry) & chain["opt_type"].isin(["CE", "PE"])]
    calls = opts.loc[opts["opt_type"] == "CE", "strike"]
    puts = opts.loc[opts["opt_type"] == "PE", "strike"]
    k = spec.strikes
    snap = float(k.get("max_snap_pct", 1.5))
    side = spec.side
    if side == "model":
        if direction not in (1, -1):
            return None
        # a debit spread buys the direction; a credit spread sells the other side
        up = direction == 1
        side = ("call" if up else "put") if spec.structure == "debit_spread" else ("put" if up else "call")

    def strike(pool, pct_from_spot: float):
        return nearest_strike(pool, spot * (1.0 + pct_from_spot), snap)

    s = spec.structure
    if s == "long_straddle":
        atm = nearest_strike(set(calls) & set(puts), spot, snap)
        return None if atm is None else [Leg(expiry, atm, "CE", BUY), Leg(expiry, atm, "PE", BUY)]
    if s == "long_strangle":
        c_otm, p_otm = (float(per_side(k, side, "otm_pct")) / 100.0 for side in ("call", "put"))
        c, p = strike(calls, c_otm), strike(puts, -p_otm)
        return None if c is None or p is None else [Leg(expiry, c, "CE", BUY), Leg(expiry, p, "PE", BUY)]
    if s == "debit_spread":
        near, width = float(k.get("otm_pct", 0.0)) / 100.0, float(k["wing_pct"]) / 100.0
        if side == "call":
            a, b = strike(calls, near), strike(calls, near + width)
            return None if a is None or b is None or b <= a else [Leg(expiry, a, "CE", BUY), Leg(expiry, b, "CE", SELL)]
        a, b = strike(puts, -near), strike(puts, -near - width)
        return None if a is None or b is None or b >= a else [Leg(expiry, a, "PE", BUY), Leg(expiry, b, "PE", SELL)]
    if s in ("credit_spread", "iron_condor"):
        legs: list[Leg] = []
        if s == "iron_condor" or side == "put":
            short, wing = float(per_side(k, "put", "short_otm_pct")) / 100.0, float(per_side(k, "put", "wing_pct")) / 100.0
            a, b = strike(puts, -short), strike(puts, -short - wing)
            if a is None or b is None or b >= a:
                return None
            legs += [Leg(expiry, a, "PE", SELL), Leg(expiry, b, "PE", BUY)]
        if s == "iron_condor" or side == "call":
            short, wing = float(per_side(k, "call", "short_otm_pct")) / 100.0, float(per_side(k, "call", "wing_pct")) / 100.0
            a, b = strike(calls, short), strike(calls, short + wing)
            if a is None or b is None or b <= a:
                return None
            legs += [Leg(expiry, a, "CE", SELL), Leg(expiry, b, "CE", BUY)]
        return legs
    raise SpecError(f"Structure '{s}' is not allowed (S-1).")


def net_debit(legs: list[Leg], prices: list[float]) -> float:
    """Cash paid per share to open: positive for a debit, negative for a credit."""
    return sum(leg.side * p for leg, p in zip(legs, prices))


def max_loss_per_share(legs: list[Leg], prices: list[float]) -> float:
    """The most the position can lose per share, before costs. Always positive for an allowed structure.

    Bought options only: the premium paid. Any structure with a sold leg: the widest gap between
    a sold strike and its bought cover, less the net credit (or plus the net debit).
    """
    debit = net_debit(legs, prices)
    if all(leg.side == BUY for leg in legs):
        return debit
    width = 0.0
    for opt_type in ("CE", "PE"):
        sold = [l for l in legs if l.opt_type == opt_type and l.side == SELL]
        bought = [l for l in legs if l.opt_type == opt_type and l.side == BUY]
        if sold and not bought:
            raise SpecError("A sold option without a bought cover has no fixed maximum loss (A2).")
        for s_leg in sold:
            cover = min(bought, key=lambda b: abs(b.strike - s_leg.strike))
            # a credit spread loses the strike gap; a debit spread's sold leg sits beyond its bought one and adds none
            loses = (cover.strike > s_leg.strike) if opt_type == "CE" else (cover.strike < s_leg.strike)
            width = max(width, abs(cover.strike - s_leg.strike) if loses else 0.0)
    return width + debit


def payoff_at_expiry(legs: list[Leg], prices: list[float], underlying: float) -> float:
    """Profit per share if held to expiry with the underlying at `underlying`. Used to check max loss."""
    total = -net_debit(legs, prices)
    for leg in legs:
        intrinsic = max(underlying - leg.strike, 0.0) if leg.opt_type == "CE" else max(leg.strike - underlying, 0.0)
        total += leg.side * intrinsic
    return total
