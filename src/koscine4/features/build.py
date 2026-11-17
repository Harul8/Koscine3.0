"""Build and serve features as of a decision time (requirements F-1, F-5, F-6).

One function, `compute`, makes every feature row, for training, backtest and the daily run
alike. A row for decision date D and time T uses, from each source, only rows that source has
published by (D, T): the builders work on source dates and this module shifts each group by the
number of sessions its source lags the decision.

    python -m koscine4.features.build --time evening      # the owner runs this; it takes a few minutes
    python -m koscine4.features.build --time 1430
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from koscine4.features import builders as B
from koscine4.features.registry import REGISTRY, Registry
from koscine4.settings import Settings, load_settings
from koscine4.store.api import Store
from koscine4.store.availability import DECISION_SLOTS, Decision, _READY, source_availability
from koscine4.store.surface import near_and_next
from koscine4.store.tables import StoreError

KEY = ["date", "instrument"]


def source_lag(settings: Settings, source: str, decision_time: str) -> int:
    """Sessions between a decision at `decision_time` and the newest row of `source` it may read."""
    if source == "intraday":
        return 0
    sessions_after, ready_slot = _READY[source_availability(settings.section("availability"), source)]
    return sessions_after if ready_slot <= DECISION_SLOTS[decision_time] else sessions_after + 1


def all_lags(settings: Settings, decision_time: str) -> dict[str, int]:
    return {s: source_lag(settings, s, decision_time)
            for s in ("prices_raw", "option_quotes", "flows_fii_dii", "flows_participant_oi", "events")}


def load_inputs(store: Store, asof: Decision) -> B.Inputs:
    """Everything visible at `asof`, read through the store."""
    cfg = store.settings.section("features")
    prices = store.prices(asof=asof)
    try:
        surface = near_and_next(store.surface(asof=asof), int(cfg["min_days_to_expiry"]))
        surface = surface[(surface["atm_strike"] / surface["forward"] - 1.0).abs() <= float(cfg["max_atm_distance"])]
    except StoreError:
        surface = pd.DataFrame(columns=KEY + ["atm_iv", "next_iv", "skew", "straddle_pct", "dte"])
    return B.Inputs(prices=prices, surface=surface, deriv=store.deriv_daily(asof=asof), fii_dii=store.fii_dii(asof=asof),
                    participant=store.participant_oi(asof=asof), events=store.all_events(), calendar=store.calendar,
                    intraday=store.intraday_bars(asof=asof) if asof.time == "1430" else None)


def _shift(df: pd.DataFrame, lag: int, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Move rows from their source date to the decision date `lag` sessions later."""
    if lag == 0 or df.empty:
        return df
    pos = pd.Series(np.arange(len(days)), index=days)
    i = df["date"].map(pos) + lag
    keep = i.notna() & (i < len(days))
    out = df[keep].copy()
    out["date"] = days[i[keep].astype(int).to_numpy()]
    return out


def compute(inputs: B.Inputs, decision_time: str, lags: dict[str, int], registry: Registry = REGISTRY,
            decision_date=None) -> pd.DataFrame:
    """Every registered feature valid at `decision_time`, one row per (date, instrument)."""
    days = inputs.calendar.days
    if decision_date is not None and pd.Timestamp(decision_date) > days[-1]:
        days = days.append(pd.DatetimeIndex([pd.Timestamp(decision_date).normalize()]))   # today, before its prices exist
    lag_p, lag_o = lags["prices_raw"], max(lags["prices_raw"], lags["option_quotes"])
    price_src = B.price_features(inputs)
    price = _shift(price_src, lag_p, days)
    out = price
    out = out.merge(_shift(B.option_features(inputs, price_src), lag_o, days), on=KEY, how="left")
    out = out.merge(_shift(B.regime_features(inputs, price_src), lag_p, days), on="date", how="left")
    cash, pos = B.flow_features(inputs)
    out = out.merge(_shift(cash, lags["flows_fii_dii"], days), on="date", how="left")
    out = out.merge(_shift(pos, lags["flows_participant_oi"], days), on="date", how="left")
    out = out.merge(_event_rows(inputs, out[KEY], lags["events"], days), on=KEY, how="left")
    if decision_time == "1430":
        out = out.merge(B.intraday_features(inputs), on=KEY, how="left")
    names = registry.names(decision_time=decision_time)
    for n in names:
        if n not in out.columns:
            out[n] = np.nan
    out = out[KEY + names].copy()
    out[names] = out[names].astype(float).replace([np.inf, -np.inf], np.nan)
    out.insert(2, "decision_time", decision_time)
    return out.sort_values(KEY).reset_index(drop=True)


def _event_rows(inputs: B.Inputs, index: pd.DataFrame, lag: int, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Event features on decision dates: computed on the date the decision can see, then moved forward."""
    if lag == 0:
        return B.event_features(inputs, index)
    pos = pd.Series(np.arange(len(days)), index=days)
    src = index.copy()
    i = src["date"].map(pos) - lag
    src = src[i >= 0]
    src_dates = days[i[i >= 0].astype(int).to_numpy()]
    seen = B.event_features(inputs, pd.DataFrame({"date": src_dates, "instrument": src["instrument"].to_numpy()}))
    seen["date"] = src["date"].to_numpy()
    # `lag` sessions have passed since the visible date
    seen["days_to_results"] = np.clip(seen["days_to_results"] * 30.0 - lag, 0, 30) / 30.0
    seen["results_in_5d"] = (seen["days_to_results"] * 30.0 <= 5).astype(float)
    seen["days_since_results"] = np.clip(seen["days_since_results"] * 60.0 + lag, 0, 60) / 60.0
    return seen


def features_asof(store: Store, asof: Decision, registry: Registry = REGISTRY) -> pd.DataFrame:
    """Feature rows built from what is visible at `asof`. The last date's rows are the live rows (F-6)."""
    return compute(load_inputs(store, asof), asof.time, all_lags(store.settings, asof.time), registry, decision_date=asof.date)


def table_path(settings: Settings, decision_time: str):
    return settings.path("pit") / "features" / f"{decision_time}.parquet"


def build_feature_table(settings: Settings | None = None, decision_time: str = "evening", asof_date=None) -> dict:
    """Compute the whole history for one decision time and store it."""
    settings = settings or load_settings()
    store = Store(settings)
    date = pd.Timestamp(asof_date) if asof_date is not None else store.calendar.days[-1]
    table = features_asof(store, Decision(date, decision_time))
    path = table_path(settings, decision_time)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return {"rows": len(table), "instruments": int(table["instrument"].nunique()), "features": len(table.columns) - 3,
            "first": str(table["date"].min().date()), "last": str(table["date"].max().date())}


def read(names, instruments=None, start=None, end=None, decision_time: str = "evening",
         settings: Settings | None = None, registry: Registry = REGISTRY) -> pd.DataFrame:
    """Stored features by name. Asking for an unregistered column raises `UnknownFeature` (F-1)."""
    settings = settings or load_settings()
    names = registry.require(names)
    for n in names:
        if decision_time not in registry.get(n).decision_times:
            raise ValueError(f"Feature '{n}' is not valid for a {decision_time} decision.")
    path = table_path(settings, decision_time)
    if not path.exists():
        raise StoreError(f"No feature table for '{decision_time}'. Run: python -m koscine4.features.build --time {decision_time}")
    df = pd.read_parquet(path, columns=KEY + names)
    if instruments is not None:
        df = df[df["instrument"].isin(list(instruments))]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time", default="evening", choices=["evening", "1430"])
    args = parser.parse_args()
    for key, value in build_feature_table(decision_time=args.time).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
