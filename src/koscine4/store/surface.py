"""Option surface and per-instrument option quotes (requirements D-5, D-6).

Method, stated so it can be checked:

* Price of an option on a day = its last traded price, used only if it traded that day.
* Forward for an expiry = the futures settlement price of the same expiry when one exists;
  otherwise strike + call - put at the strike where call and put are closest (put-call parity,
  interest ignored, which matters little at these maturities).
* Implied volatility = Black's 1976 formula on the forward, solved by bisection, zero interest.
* At-the-money volatility = average of the call and put volatility at the strike nearest the forward.
* Skew = volatility of the put one standard deviation below the forward minus that of the call
  one standard deviation above, where one standard deviation = forward * atm_iv * sqrt(T).
* Straddle cost = (call + put at the at-the-money strike) / forward.

One pass over the silver contract files writes two things:
`<pit>/option_surface.parquet` for every instrument, and `<pit>/option_quotes/<instrument>/<year>.parquet`
for the instruments the harness trades.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.store.options import available_years, read_contracts
from koscine4.store.symbols import canonicalize, load_changes
from koscine4.store.tables import LOT_SIZES, StoreError, read_table, table_path, write_table

OPTION_SURFACE = "option_surface"
SURFACE_COLUMNS = ["date", "instrument", "expiry", "dte", "forward", "forward_source", "atm_strike",
                   "atm_iv", "call_iv", "put_iv", "skew", "straddle_pct", "atm_volume"]
QUOTE_COLUMNS = ["date", "instrument", "contract_type", "expiry", "strike", "opt_type",
                 "open", "high", "low", "close", "settle", "volume", "open_int", "lot"]
_SQRT2 = math.sqrt(2.0)


def _ncdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * (1.0 + erf(x / _SQRT2))


def black_price(forward, strike, t, sigma, is_call) -> np.ndarray:
    """Black (1976) option price on a forward, zero interest. All arguments broadcast."""
    forward, strike, t, sigma = (np.asarray(a, dtype=float) for a in (forward, strike, t, sigma))
    vol = np.maximum(sigma * np.sqrt(t), 1e-12)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    call = forward * _ncdf(d1) - strike * _ncdf(d2)
    return np.where(np.asarray(is_call, dtype=bool), call, call - forward + strike)


def implied_vol(price, forward, strike, t, is_call, lo: float = 0.005, hi: float = 5.0, steps: int = 50) -> np.ndarray:
    """Implied volatility by bisection. NaN where the price is outside what any volatility in [lo, hi] gives."""
    price, forward, strike, t = (np.asarray(a, dtype=float) for a in (price, forward, strike, t))
    is_call = np.asarray(is_call, dtype=bool)
    low = np.full(price.shape, lo)
    high = np.full(price.shape, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (price > 0) & (forward > 0) & (strike > 0) & (t > 0)
        ok &= (price > black_price(forward, strike, t, low, is_call)) & (price < black_price(forward, strike, t, high, is_call))
        for _ in range(steps):
            mid = 0.5 * (low + high)
            above = black_price(forward, strike, t, mid, is_call) > price
            high = np.where(above, mid, high)
            low = np.where(above, low, mid)
    return np.where(ok, 0.5 * (low + high), np.nan)


def surface_from_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """One row per (instrument, date, expiry) from contract rows (columns as in the silver table,
    with `symbol` already holding the instrument name)."""
    c = contracts
    key = ["symbol", "date", "expiry"]
    fut = c[c["opt_type"].isna() | ~c["opt_type"].isin(["CE", "PE"])]
    fut = fut[fut["settle"] > 0].groupby(key, sort=False)["settle"].last().rename("fut")
    opt = c[c["opt_type"].isin(["CE", "PE"]) & (c["contracts"] > 0) & (c["close"] > 0)]
    opt = opt[(opt["expiry"] - opt["date"]).dt.days >= 1]
    if opt.empty:
        return pd.DataFrame(columns=SURFACE_COLUMNS)
    calls = opt[opt["opt_type"] == "CE"][key + ["strike", "close", "contracts"]].rename(columns={"close": "c", "contracts": "cv"})
    puts = opt[opt["opt_type"] == "PE"][key + ["strike", "close", "contracts"]].rename(columns={"close": "p", "contracts": "pv"})
    both = calls.merge(puts, on=key + ["strike"])
    if both.empty:
        return pd.DataFrame(columns=SURFACE_COLUMNS)

    # forward: futures of the same expiry, else put-call parity at the strike where call and put are closest
    both["cp_gap"] = (both["c"] - both["p"]).abs()
    par = both.loc[both.groupby(key, sort=False)["cp_gap"].idxmin(), key + ["strike", "c", "p"]]
    par["parity"] = par["strike"] + par["c"] - par["p"]
    g = par[key + ["parity"]].merge(fut.reset_index(), on=key, how="left")
    g["forward"] = g["fut"].where(g["fut"] > 0, g["parity"])
    g["forward_source"] = np.where(g["fut"] > 0, "futures", "parity")
    g = g[g["forward"] > 0]
    g["t"] = (g["expiry"] - g["date"]).dt.days / 365.0

    both = both.merge(g[key + ["forward", "forward_source", "t"]], on=key)
    both["dist"] = (both["strike"] - both["forward"]).abs()
    atm = both.loc[both.groupby(key, sort=False)["dist"].idxmin()].copy()
    atm["call_iv"] = implied_vol(atm["c"], atm["forward"], atm["strike"], atm["t"], True)
    atm["put_iv"] = implied_vol(atm["p"], atm["forward"], atm["strike"], atm["t"], False)
    atm["atm_iv"] = atm[["call_iv", "put_iv"]].mean(axis=1)
    atm["straddle_pct"] = (atm["c"] + atm["p"]) / atm["forward"]
    atm["atm_volume"] = atm["cv"] + atm["pv"]
    atm = atm.rename(columns={"strike": "atm_strike"})[key + ["forward", "forward_source", "t", "atm_strike", "call_iv", "put_iv", "atm_iv", "straddle_pct", "atm_volume"]]

    # skew at one standard deviation either side
    sd = atm[key + ["forward", "t", "atm_iv"]].dropna()
    sd["width"] = sd["forward"] * sd["atm_iv"] * np.sqrt(sd["t"])

    def wing(side: pd.DataFrame, price: str, sign: float, is_call: bool, name: str) -> pd.DataFrame:
        w = side.merge(sd, on=key)
        w["dist"] = (w["strike"] - (w["forward"] + sign * w["width"])).abs()
        w = w.loc[w.groupby(key, sort=False)["dist"].idxmin()]
        w = w[w["dist"] <= 0.5 * w["width"]].copy()            # no strike near enough: no skew, rather than a wrong one
        w[name] = implied_vol(w[price], w["forward"], w["strike"], w["t"], is_call)
        return w[key + [name]]

    out = atm.merge(wing(puts, "p", -1.0, False, "put_wing_iv"), on=key, how="left")
    out = out.merge(wing(calls, "c", +1.0, True, "call_wing_iv"), on=key, how="left")
    out["skew"] = out["put_wing_iv"] - out["call_wing_iv"]
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    out = out.rename(columns={"symbol": "instrument"})
    return out[SURFACE_COLUMNS].sort_values(["instrument", "date", "expiry"]).reset_index(drop=True)


def quotes_dir(settings: Settings):
    return settings.path("pit") / "option_quotes"


def build_option_tables(settings: Settings, quote_instruments=None, years=None) -> dict:
    """Scan the contract files once: surface for all instruments, quotes for `quote_instruments`."""
    changes = load_changes(settings)
    try:
        lots = read_table(settings, LOT_SIZES, columns=["instrument", "expiry", "lot"])
    except StoreError:
        lots = None
    wanted = None if quote_instruments is None else set(quote_instruments)
    surfaces, n_quotes = [], 0
    cols = ["date", "symbol", "instrument", "expiry", "strike", "opt_type", "open", "high", "low", "close", "settle", "contracts", "open_int"]
    for year in (years or available_years(settings)):
        c = read_contracts(settings, year, columns=cols)
        c["date"] = pd.to_datetime(c["date"]).dt.normalize()
        c["symbol"] = canonicalize(c["symbol"], c["date"], changes)
        surfaces.append(surface_from_contracts(c))
        q = c if wanted is None else c[c["symbol"].isin(wanted)]
        q = q.rename(columns={"instrument": "contract_type", "symbol": "instrument", "contracts": "volume"})
        if lots is not None:
            q = q.merge(lots, on=["instrument", "expiry"], how="left")
        else:
            q = q.assign(lot=np.nan)
        for instrument, rows in q.groupby("instrument", sort=False):
            folder = quotes_dir(settings) / instrument
            folder.mkdir(parents=True, exist_ok=True)
            rows[QUOTE_COLUMNS].sort_values(["date", "expiry", "opt_type", "strike"]).to_parquet(folder / f"{year}.parquet", index=False)
            n_quotes += len(rows)
    frames = [s for s in surfaces if len(s)]
    surface = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SURFACE_COLUMNS)
    if years is None:
        write_table(settings, OPTION_SURFACE, surface)
    return {"surface_rows": len(surface), "quote_rows": n_quotes}


def read_quotes(settings: Settings, instrument: str) -> pd.DataFrame:
    """Every stored quote of one instrument (all years). Raises if it has not been built."""
    folder = quotes_dir(settings) / instrument
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        raise StoreError(f"No option quotes stored for {instrument}. Run: python -m koscine4.store.build --options")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_surface(settings: Settings) -> pd.DataFrame:
    if not table_path(settings, OPTION_SURFACE).exists():
        raise StoreError("The option surface has not been built. Run: python -m koscine4.store.build --options")
    return read_table(settings, OPTION_SURFACE)


def near_and_next(surface: pd.DataFrame, min_dte: int) -> pd.DataFrame:
    """Per (instrument, date): the first expiry at least `min_dte` days away, and the one after it."""
    s = surface[(surface["dte"] >= min_dte) & surface["atm_iv"].notna()].sort_values(["instrument", "date", "expiry"])
    s["n"] = s.groupby(["instrument", "date"], sort=False).cumcount()
    near = s[s["n"] == 0].drop(columns="n")
    nxt = s[s["n"] == 1][["instrument", "date", "atm_iv", "dte"]].rename(columns={"atm_iv": "next_iv", "dte": "next_dte"})
    return near.merge(nxt, on=["instrument", "date"], how="left")


def traded_instruments(settings: Settings) -> list[str]:
    """Instruments whose quotes the harness needs: the two indices and every stock that was ever in a monthly list."""
    from koscine4.universe.build import INDEX_BOOK, read_all

    months = read_all(settings)
    stocks = sorted(months[months["in_list"]]["instrument"].unique()) if len(months) else []
    return list(INDEX_BOOK) + stocks


def main() -> None:
    """python -m koscine4.store.surface   (the owner runs this; it reads every contract file and takes several minutes)"""
    import argparse

    from koscine4.settings import load_settings

    argparse.ArgumentParser(description=main.__doc__).parse_args()
    settings = load_settings()
    instruments = traded_instruments(settings)
    print(f"quotes for {len(instruments)} instruments; surface for all")
    for key, value in build_option_tables(settings, quote_instruments=instruments).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
