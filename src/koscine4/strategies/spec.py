"""Strategy specifications (requirements S-1 to S-3, S-12, N-9).

A strategy is a written rule set, fixed before it is tested: one TOML file under
`configs/strategies/`. The specification is stored with every run that uses it.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from koscine4.settings import Settings

ALLOWED_STRUCTURES = ("long_straddle", "long_strangle", "debit_spread", "credit_spread", "iron_condor")
LEGS = {"long_straddle": 2, "long_strangle": 2, "debit_spread": 2, "credit_spread": 2, "iron_condor": 4}
DIRECTIONAL = ("debit_spread",)                   # a structure that only pays if direction is right
BOOKS = ("index", "stock")
DECISION_TO_ENTRY = {"evening": "next_open", "1430": "1515"}     # the two supported timings (decision A5)
INDEX_BOOK = ("NIFTY", "BANKNIFTY")


class SpecError(ValueError):
    """Raised when a strategy specification breaks a rule."""


@dataclass(frozen=True)
class StrategySpec:
    name: str
    book: str                                   # "index" or "stock"
    structure: str
    decision_time: str                          # "evening" or "1430"
    entry_time: str                             # "next_open" or "1515"
    instruments: tuple[str, ...] = ()           # empty for the stock book = the month's list
    side: str = ""                              # spreads: "put", "call" or "model" (from a direction model)
    family: str = ""                            # runs of one idea share a family; variants are counted per family
    expiry: Mapping[str, Any] = field(default_factory=dict)      # min_dte, max_dte (calendar days)
    strikes: Mapping[str, Any] = field(default_factory=dict)     # otm_pct, short_otm_pct, wing_pct, max_snap_pct
    exit: Mapping[str, Any] = field(default_factory=dict)        # max_hold_sessions, target_pct, stop_pct, safety_days
    liquidity: Mapping[str, Any] = field(default_factory=dict)   # min_volume, min_open_interest, min_premium
    score: Mapping[str, Any] = field(default_factory=dict)       # source, threshold, feature
    filters: tuple[Mapping[str, Any], ...] = ()                  # each: feature, op, value
    selection: Mapping[str, Any] = field(default_factory=dict)   # weekly_cap, max_per_day
    allow_across_results: bool = False          # R-8
    direction_model: str = ""                   # S-12: the direction model this strategy depends on, if any
    notes: str = ""

    @property
    def uses_direction(self) -> bool:
        return self.structure in DIRECTIONAL or self.side == "model" or bool(self.direction_model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instruments"] = list(self.instruments)
        d["filters"] = [dict(f) for f in self.filters]
        for k in ("expiry", "strikes", "exit", "liquidity", "score", "selection"):
            d[k] = dict(d[k])
        return d

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]


def validate(spec: StrategySpec) -> StrategySpec:
    if spec.structure not in ALLOWED_STRUCTURES:
        raise SpecError(f"Structure '{spec.structure}' is not allowed. Allowed: {', '.join(ALLOWED_STRUCTURES)} (S-1).")
    if spec.book not in BOOKS:
        raise SpecError(f"Book must be one of {BOOKS}.")
    if spec.book == "stock" and LEGS[spec.structure] > 2:
        raise SpecError("Stock strategies use one or two legs; four-leg structures are for indices only (S-3).")
    if spec.book == "index" and (not spec.instruments or any(i not in INDEX_BOOK for i in spec.instruments)):
        raise SpecError(f"An index strategy must name its instruments from {INDEX_BOOK} (IX-1).")
    if DECISION_TO_ENTRY.get(spec.decision_time) != spec.entry_time:
        raise SpecError("Supported timings: decision 'evening' with entry 'next_open', or decision '1430' with entry '1515' (A5).")
    if spec.structure in ("debit_spread", "credit_spread") and spec.side not in ("put", "call", "model"):
        raise SpecError("A spread needs side = 'put', 'call' or 'model'.")
    if spec.side == "model" and not spec.direction_model:
        raise SpecError("side = 'model' needs direction_model (S-12).")
    for key in ("max_hold_sessions", "safety_days"):
        if key not in spec.exit:
            raise SpecError(f"Exit rules must state {key} (V-7, R-5).")
    if int(spec.exit["max_hold_sessions"]) > 5:
        raise SpecError("Holding period is at most 5 trading sessions (A4).")
    for key in ("min_volume", "min_open_interest", "min_premium"):
        if key not in spec.liquidity:
            raise SpecError(f"Liquidity check must state {key} (S-4).")
    if "source" not in spec.score or "threshold" not in spec.score:
        raise SpecError("Score must state its source and a threshold fixed before testing (S-5, S-6).")
    if "min_dte" not in spec.expiry:
        raise SpecError("Expiry rule must state min_dte.")
    return spec


def from_dict(d: Mapping[str, Any]) -> StrategySpec:
    d = dict(d)
    d["instruments"] = tuple(d.get("instruments", ()))
    d["filters"] = tuple(dict(f) for f in d.get("filters", ()))
    d.setdefault("family", d.get("name", ""))
    known = set(StrategySpec.__dataclass_fields__)
    extra = set(d) - known
    if extra:
        raise SpecError(f"Unknown fields in strategy specification: {sorted(extra)}")
    return validate(StrategySpec(**d))


def spec_path(settings: Settings, name: str) -> Path:
    return settings.path("strategies") / f"{name}.toml"


def load_spec(settings: Settings, name: str) -> StrategySpec:
    path = spec_path(settings, name)
    if not path.exists():
        raise SpecError(f"No strategy specification at {path}.")
    with path.open("rb") as fh:
        return from_dict(tomllib.load(fh))
