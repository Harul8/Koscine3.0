"""The strategy interface, and the strategy that a specification file describes.

The harness and the daily run both drive a strategy through `candidates` and `manage`, and
through nothing else. That shared path is what makes a live trade comparable with its backtest.

Every candidate carries one comparable number (S-5): its expected result after costs, as a
percentage of its maximum loss. A model trained on a structure's after-cost result gives it
directly. Any other strategy gets it from its own track record: the mean after-cost result of
its closed trades known at the decision, taken from the trades whose score fell in the same
band. The ranking inside one strategy still follows its specification (a baseline ranks by its
feature, M-1); the expected result is what ranks candidates of different strategies against
each other in the daily list.

Direction may tilt the choice between two neutral variants of one structure (DR-7): the trade
is decided first without it, and only then may a direction forecast move the strikes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import pandas as pd

from koscine4.strategies.spec import StrategySpec
from koscine4.strategies.structures import Leg, build_legs, expiry_kind, max_loss_per_share, net_debit

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
    expected_result_pct: float = float("nan")    # S-5: expected result after costs, % of maximum loss


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

    def expected_result(self, view, score: float) -> tuple[float, dict]:
        """(expected result after costs as % of maximum loss, the inputs it came from) (S-5)."""
        s = self.spec.score
        if s.get("source") == "model" and view.model_predicts_result(str(s["model"])):
            return float(score), {"source": "model", "model": str(s["model"]), "prediction_pct": float(score)}
        return view.expected_from_track_record(self.spec.name, score)

    def _priced_legs(self, spec: StrategySpec, chain: pd.DataFrame, spot: float, date, direction):
        """((legs, prices, lot, max loss per share), "", {}) or (None, reason, details)."""
        liq = spec.liquidity
        legs = build_legs(spec, chain, spot, date, direction)
        if not legs:
            return None, "no_strikes_for_structure", {}
        rows = [_row(chain, leg) for leg in legs]
        if any(r is None for r in rows):
            return None, "leg_not_listed", {}
        lot0 = rows[0]["lot"] if rows[0]["lot"] == rows[0]["lot"] and rows[0]["lot"] > 0 else float("nan")
        for leg, r in zip(legs, rows):                                        # S-4: every leg
            problem = None
            if not r["volume"] >= float(liq["min_volume"]):
                problem = "liquidity:volume"
            elif not r["open_int"] / lot0 >= float(liq["min_open_interest"]):          # open interest in lots
                problem = "liquidity:open_interest"
            elif not r["close"] >= float(liq["min_premium"]):
                problem = "liquidity:premium"
            if problem:
                return None, problem, {"leg": leg.label()}
        prices = [float(r["close"]) for r in rows]
        lot = rows[0]["lot"]
        if lot != lot or lot <= 0:
            return None, "no_lot_size", {}
        risk = max_loss_per_share(legs, prices)
        if not risk > 0:
            return None, "no_risk_defined", {}
        return (legs, prices, float(lot), risk), "", {}

    def _tilt(self, view, instrument: str, chain: pd.DataFrame, spot: float, base: tuple) -> tuple[tuple, dict]:
        """DR-7: move the strikes of an already-decided neutral trade toward a direction forecast.

        Returns the legs to trade and a note. Any problem keeps the untilted trade: the forecast can
        change where the strikes sit, never whether there is a trade.
        """
        t = self.spec.tilt
        note = {"model": str(t["model"]), "variant": "neutral"}
        try:
            p = view.prediction(str(t["model"]), instrument)
        except Exception:                                   # no predictions loaded for the model
            return base, dict(note, reason="no_forecast_available")
        if p != p:
            return base, dict(note, reason="no_forecast_for_this_day")
        note["forecast_up"] = float(p)
        conf = float(t.get("min_confidence", 0.55))
        variant = "up" if p >= conf else "down" if p <= 1.0 - conf else None
        if variant is None:
            return base, dict(note, reason="forecast_not_confident")
        tilted = replace(self.spec, strikes={**self.spec.strikes, **dict(t[variant])})
        got, reason, _ = self._priced_legs(tilted, chain, spot, view.date, None)
        if got is None:
            return base, dict(note, reason=f"{variant}_variant_refused:{reason}")
        return got, dict(note, variant=variant)

    def candidates(self, view):
        spec, out, refused = self.spec, [], []

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
            built, reason, more = self._priced_legs(spec, chain, spot, view.date, self.direction(view, instrument))
            if built is None:
                refuse(instrument, reason, **more)
                continue
            score = self.score(view, instrument)
            if score != score:
                refuse(instrument, "no_score")
                continue
            if score < float(spec.score["threshold"]):                        # S-6
                refuse(instrument, "below_threshold", score=score)
                continue
            reasons = {"score_source": spec.score["source"], "spot": spot}
            if spec.tilt:                                                     # DR-7: only after the trade is decided
                built, reasons["tilt"] = self._tilt(view, instrument, chain, spot, built)
            legs, prices, lot, risk = built
            expected, inputs = self.expected_result(view, float(score))
            reasons.update({"net_debit": net_debit(legs, prices), "expiry_kind": expiry_kind(chain, legs[0].expiry),
                            "expected_result": inputs})
            out.append(Candidate(spec.name, spec.book, instrument, view.date, legs, float(score), lot, prices, risk,
                                 reasons=reasons, uses_direction=spec.uses_direction, expected_result_pct=expected))
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
