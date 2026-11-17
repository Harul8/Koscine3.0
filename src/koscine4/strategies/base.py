"""The strategy interface, and the strategy that a specification file describes.

The harness and the daily run both drive a strategy through `candidates` and `manage`, and
through nothing else. That shared path is what makes a live trade comparable with its backtest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from koscine4.strategies.spec import StrategySpec
from koscine4.strategies.structures import Leg, build_legs, max_loss_per_share, net_debit

SELLS_PREMIUM = ("credit_spread", "iron_condor")


@dataclass
class Candidate:
    strategy: str
    book: str
    instrument: str
    decision_date: pd.Timestamp
    legs: list[Leg]
    score: float
    lot: float
    reference_prices: list[float]          # prices on the chain the decision saw
    max_loss_per_share: float              # at those prices
    reasons: dict = field(default_factory=dict)
    uses_direction: bool = False


@dataclass
class PositionState:
    instrument: str
    legs: list[Leg]
    entry_prices: list[float]
    prices_now: list[float]
    max_loss_per_share: float
    sessions_held: int                     # 1 on the entry session
    date: pd.Timestamp

    @property
    def result_pct(self) -> float:
        """Profit so far as a percentage of the maximum loss, before costs."""
        pnl = sum(l.side * (now - then) for l, now, then in zip(self.legs, self.prices_now, self.entry_prices))
        return 100.0 * pnl / self.max_loss_per_share if self.max_loss_per_share > 0 else 0.0


@dataclass(frozen=True)
class Action:
    exit: bool
    reason: str = ""


class Strategy:
    spec: StrategySpec

    def candidates(self, view) -> tuple[list[Candidate], list[dict]]:
        """(candidates, refusals) for one decision. A refusal is a dict with instrument and reason."""
        raise NotImplementedError

    def manage(self, state: PositionState, view=None) -> Action:
        raise NotImplementedError


_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


class SpecStrategy(Strategy):
    """A strategy fully described by its specification file."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec

    def instruments(self, view) -> list[str]:
        return list(self.spec.instruments) if self.spec.instruments else view.stock_list()

    def score(self, view, instrument: str) -> float:
        s = self.spec.score
        source = str(s["source"])
        if source == "take_all":
            return 1.0
        if source == "feature":
            return float(s.get("sign", 1.0)) * view.feature(instrument, str(s["feature"]))
        if source == "model":
            return view.prediction(str(s["model"]), instrument)
        raise ValueError(f"Unknown score source '{source}'.")

    def direction(self, view, instrument: str) -> int | None:
        if self.spec.side != "model":
            return None
        p = view.prediction(self.spec.direction_model, instrument)
        if p != p:
            return None
        return 1 if p >= 0.5 else -1

    def candidates(self, view):
        spec, out, refused = self.spec, [], []
        liq = spec.liquidity

        def refuse(instrument, reason, **more):
            refused.append({"instrument": instrument, "strategy": spec.name, "reason": reason, **more})

        for instrument in self.instruments(view):
            chain = view.chain(instrument)
            spot = view.spot(instrument)
            if chain.empty or not spot == spot:
                refuse(instrument, "no_option_data")
                continue
            failed = None
            for f in spec.filters:
                value = view.feature(instrument, str(f["feature"]))
                if value != value or not _OPS[str(f["op"])](value, float(f["value"])):
                    failed = f"filter:{f['feature']}{f['op']}{f['value']}"
                    break
            if failed:
                refuse(instrument, failed)
                continue
            hold = int(spec.exit["max_hold_sessions"])
            if spec.book == "stock" and spec.structure in SELLS_PREMIUM and not spec.allow_across_results \
                    and view.results_within(instrument, hold + 1):
                refuse(instrument, "results_in_window")                      # R-8
                continue
            legs = build_legs(spec, chain, spot, view.date, self.direction(view, instrument))
            if not legs:
                refuse(instrument, "no_strikes_for_structure")
                continue
            rows = [_row(chain, leg) for leg in legs]
            if any(r is None for r in rows):
                refuse(instrument, "leg_not_listed")
                continue
            problem = None
            lot0 = rows[0]["lot"] if rows[0]["lot"] == rows[0]["lot"] and rows[0]["lot"] > 0 else float("nan")
            for leg, r in zip(legs, rows):                                    # S-4: every leg
                if not r["volume"] >= float(liq["min_volume"]):
                    problem = "liquidity:volume"
                elif not r["open_int"] / lot0 >= float(liq["min_open_interest"]):      # open interest in lots
                    problem = "liquidity:open_interest"
                elif not r["close"] >= float(liq["min_premium"]):
                    problem = "liquidity:premium"
                if problem:
                    break
            if problem:
                refuse(instrument, problem, leg=leg.label())
                continue
            prices = [float(r["close"]) for r in rows]
            lot = rows[0]["lot"]
            if lot != lot or lot <= 0:
                refuse(instrument, "no_lot_size")
                continue
            risk = max_loss_per_share(legs, prices)
            if not risk > 0:
                refuse(instrument, "no_risk_defined")
                continue
            score = self.score(view, instrument)
            if score != score:
                refuse(instrument, "no_score")
                continue
            if score < float(spec.score["threshold"]):                        # S-6
                refuse(instrument, "below_threshold", score=score)
                continue
            out.append(Candidate(spec.name, spec.book, instrument, view.date, legs, float(score), float(lot), prices, risk,
                                 reasons={"score_source": spec.score["source"], "net_debit": net_debit(legs, prices), "spot": spot},
                                 uses_direction=spec.uses_direction))
        return out, refused

    def manage(self, state: PositionState, view=None) -> Action:
        rules = self.spec.exit
        expiry = min(l.expiry for l in state.legs)
        if (expiry - state.date).days <= int(rules["safety_days"]):
            return Action(True, "safety_before_expiry")                       # R-5
        target, stop = rules.get("target_pct"), rules.get("stop_pct")
        if target is not None and state.result_pct >= float(target):
            return Action(True, "target")
        if stop is not None and state.result_pct <= -abs(float(stop)):
            return Action(True, "stop")
        if state.sessions_held >= int(rules["max_hold_sessions"]):
            return Action(True, "time")
        return Action(False)


def _row(chain: pd.DataFrame, leg: Leg):
    hit = chain[(chain["expiry"] == leg.expiry) & (chain["strike"] == leg.strike) & (chain["opt_type"] == leg.opt_type)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"close": _num(r["close"]), "open": _num(r["open"]), "settle": _num(r.get("settle", float("nan"))),
            "volume": _num(r["volume"]), "open_int": _num(r["open_int"]), "lot": _num(r.get("lot", float("nan")))}


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")
