"""The feature registry (requirements F-1, F-2, F-4, F-7 to F-10).

A model can read a feature only if it is registered here. Each entry records the source,
the formula, the look-back and the decision times it is valid for. Features are chosen by
naming them; nothing in this package selects features by excluding column names.
"""
from __future__ import annotations

from dataclasses import dataclass

FAMILIES = (
    "implied_vs_realised", "volatility_state", "events", "regime", "structure",
    "trend", "positioning", "flows", "intraday",
)
MAX_MODEL_FEATURES = 40          # F-10


class UnknownFeature(KeyError):
    """Raised when code asks for a column that is not a registered feature."""


class RegistryError(ValueError):
    """Raised when a registration is malformed or fails its leak test."""


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    group: str                    # the builder that computes it
    source: str                   # store source whose availability governs it
    formula: str
    lookback: int                 # sessions of history the value depends on
    scale_free: bool = True       # F-9: unchanged if every price of the instrument is multiplied by a constant
    decision_times: tuple[str, ...] = ("evening", "1430")
    percentile_of: str | None = None   # F-7: this feature is the trailing percentile of another


class Registry:
    def __init__(self):
        self._features: dict[str, Feature] = {}

    def register(self, feature: Feature) -> Feature:
        if feature.name in self._features:
            raise RegistryError(f"Feature '{feature.name}' is already registered.")
        if feature.family not in FAMILIES:
            raise RegistryError(f"Feature '{feature.name}' has unknown family '{feature.family}'.")
        if not feature.scale_free:
            raise RegistryError(f"Feature '{feature.name}' is a raw level; features must be scale-free (F-9).")
        if not feature.formula.strip() or feature.lookback < 0:
            raise RegistryError(f"Feature '{feature.name}' needs a formula and a non-negative look-back.")
        self._features[feature.name] = feature
        return feature

    def get(self, name: str) -> Feature:
        try:
            return self._features[name]
        except KeyError:
            raise UnknownFeature(f"'{name}' is not a registered feature.") from None

    def require(self, names) -> list[str]:
        """The same names, after checking that every one is registered (F-1)."""
        names = list(names)
        for n in names:
            self.get(n)
        return names

    def names(self, family: str | None = None, decision_time: str | None = None, group: str | None = None) -> list[str]:
        return [f.name for f in self._features.values()
                if (family is None or f.family == family)
                and (decision_time is None or decision_time in f.decision_times)
                and (group is None or f.group == group)]

    def groups(self, names) -> list[str]:
        return sorted({self.get(n).group for n in names})

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)

    def describe(self) -> list[dict]:
        return [vars(f).copy() for f in self._features.values()]


def _f(name, family, group, source, formula, lookback, **kw) -> Feature:
    return Feature(name, family, group, source, formula, lookback, **kw)


def default_registry() -> Registry:
    r = Registry()
    P, O, D, E, FL, PO = "prices_raw", "option_quotes", "option_quotes", "events", "flows_fii_dii", "flows_participant_oi"
    for f in [
        # ---- price trend
        _f("ret_1d", "trend", "price", P, "close / previous close - 1 (adjusted)", 1),
        _f("ret_5d", "trend", "price", P, "close / close 5 sessions ago - 1", 5),
        _f("ret_20d", "trend", "price", P, "close / close 20 sessions ago - 1", 20),
        _f("dist_ma20", "trend", "price", P, "close / 20-session mean close - 1", 20),
        _f("dist_ma50", "trend", "price", P, "close / 50-session mean close - 1", 50),
        _f("dist_ma200", "trend", "price", P, "close / 200-session mean close - 1", 200),
        _f("gap_open", "trend", "price", P, "open / previous close - 1", 1),
        # ---- volatility state
        _f("rv_5", "volatility_state", "price", P, "std of daily log returns over 5 sessions, annualised", 5),
        _f("rv_20", "volatility_state", "price", P, "std of daily log returns over 20 sessions, annualised", 20),
        _f("rv_60", "volatility_state", "price", P, "std of daily log returns over 60 sessions, annualised", 60),
        _f("rv_ratio_5_20", "volatility_state", "price", P, "rv_5 / rv_20", 20),
        _f("parkinson_10", "volatility_state", "price", P, "Parkinson high-low volatility over 10 sessions, annualised", 10),
        _f("atr_pct_14", "volatility_state", "price", P, "14-session mean true range / close", 14),
        _f("range_pct", "volatility_state", "price", P, "(high - low) / close", 0),
        _f("volume_ratio_20", "volatility_state", "price", P, "volume / 20-session mean volume", 20),
        _f("rv_20_pctile", "volatility_state", "price", P, "percentile of rv_20 in its own trailing 252 sessions", 272, percentile_of="rv_20"),
        # ---- implied against realised
        _f("atm_iv", "implied_vs_realised", "option", O, "at-the-money implied volatility of the near expiry", 0),
        _f("iv_rv_ratio", "implied_vs_realised", "option", O, "atm_iv / rv_20", 20),
        _f("iv_minus_rv", "implied_vs_realised", "option", O, "atm_iv - rv_20", 20),
        _f("implied_move_5d", "implied_vs_realised", "option", O, "atm_iv * sqrt(5 / 252)", 0),
        _f("iv_chg_5d", "implied_vs_realised", "option", O, "atm_iv - atm_iv 5 sessions ago", 5),
        _f("iv_pctile", "implied_vs_realised", "option", O, "percentile of atm_iv in its own trailing 252 sessions", 252, percentile_of="atm_iv"),
        _f("term_slope", "implied_vs_realised", "option", O, "next-expiry atm_iv / near-expiry atm_iv - 1", 0),
        # ---- option positioning
        _f("skew", "positioning", "option", O, "IV of the put one s.d. below the forward minus IV of the call one s.d. above", 0),
        _f("pcr_oi", "positioning", "option", D, "put open interest / call open interest", 0),
        _f("pcr_oi_chg_5d", "positioning", "option", D, "pcr_oi - pcr_oi 5 sessions ago", 5),
        _f("pcr_vol", "positioning", "option", D, "put volume / call volume", 0),
        _f("fut_oi_chg_5d", "positioning", "option", D, "futures open interest / its value 5 sessions ago - 1", 5),
        _f("fut_basis", "positioning", "option", D, "near futures close / spot close - 1", 0),
        # ---- trade structure
        _f("straddle_pct", "structure", "option", O, "at-the-money straddle price / forward, near expiry", 0),
        _f("dte_near", "structure", "option", O, "calendar days to the near expiry / 30", 0),
        _f("opt_volume_ratio_20", "structure", "option", D, "option volume / its 20-session mean", 20),
        # ---- scheduled events
        _f("days_to_results", "events", "event", E, "sessions to the next announced result date, capped at 30, / 30", 0),
        _f("results_in_5d", "events", "event", E, "1 if an announced result date falls in the next 5 sessions", 0),
        _f("days_since_results", "events", "event", E, "sessions since the last result date, capped at 60, / 60", 0),
        # ---- market regime (same value for every instrument on a day)
        _f("nifty_ret_5d", "regime", "regime", P, "Nifty close / close 5 sessions ago - 1", 5),
        _f("nifty_rv_20", "regime", "regime", P, "Nifty rv_20", 20),
        _f("nifty_dist_ma200", "regime", "regime", P, "Nifty close / 200-session mean - 1", 200),
        _f("vix_pctile", "regime", "regime", P, "percentile of India VIX in its trailing 252 sessions", 252),
        _f("vix_chg_5d", "regime", "regime", P, "India VIX / its value 5 sessions ago - 1", 5),
        _f("breadth_ma50", "regime", "regime", P, "share of F&O stocks closing above their 50-session mean", 50),
        # ---- flows
        _f("fii_net_5d_z", "flows", "flow", FL, "5-session sum of FII net cash, as a z-score of its trailing 252 sessions", 257),
        _f("dii_net_5d_z", "flows", "flow", FL, "5-session sum of DII net cash, as a z-score of its trailing 252 sessions", 257),
        _f("fii_idx_fut_ratio", "flows", "flow", PO, "FII index futures (long - short) / (long + short)", 0),
        _f("fii_idx_fut_ratio_chg_5d", "flows", "flow", PO, "fii_idx_fut_ratio - its value 5 sessions ago", 5),
        _f("fii_idx_opt_tilt", "flows", "flow", PO, "FII index options (call long + put short - put long - call short) / total", 0),
        _f("fii_net_5d_pctile", "flows", "flow", FL, "percentile of the 5-session FII net cash sum in its trailing 252 sessions", 257, percentile_of="fii_net_5d_z"),
        # ---- intraday state at 14:30 (indices only; not valid for an evening decision)
        _f("intraday_ret_1430", "intraday", "intraday", "intraday", "price at 14:30 / today's open - 1", 0, decision_times=("1430",)),
        _f("intraday_range_1430", "intraday", "intraday", "intraday", "(high - low up to 14:30) / today's open", 0, decision_times=("1430",)),
        _f("gap_open_today", "intraday", "intraday", "intraday", "today's open / previous close - 1", 1, decision_times=("1430",)),
    ]:
        r.register(f)
    return r


REGISTRY = default_registry()
