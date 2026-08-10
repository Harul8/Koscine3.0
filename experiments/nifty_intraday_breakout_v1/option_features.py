"""Option-chain-derived 5-min features for the NIFTY intraday breakout model (v2 variant).

Reads data/intraday/nifty_option_chain_5m/{expiry}.parquet (one file per NIFTY expiry,
chain-shaped: one row per (timestamp, strike) with ce_*/pe_* side by side). For each
timestamp, picks the FRONT (nearest unexpired) expiry's chain, finds the ATM strike, and
derives an implied-vol / straddle-premium / OI context.

Coverage: expired-contract lookup only goes back to ~Oct 2024 (see
analysis/download_nifty_option_chain_5m.py's docstring) -- this is why the v2 dataset is
restricted to that window while v1 (price-only) covers the full 2022-2026 history.

The Black-Scholes / implied-vol helpers are copied (not imported) from api/main.py's
_bs_price/_implied_vol -- importing api.main directly would construct the live FastAPI
app as an import side effect, inappropriate for an offline research script.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

_RISK_FREE_RATE = 0.065   # matches api/main.py's flat India risk-free proxy

OPTION_COLUMNS = ["timestamp", "expiry", "strike", "underlying_close",
                   "ce_open", "ce_close", "ce_oi", "pe_open", "pe_close", "pe_oi"]


def _bs_price(spot: float, strike: float, years: float, sigma: float, is_call: bool) -> float:
    if sigma <= 0 or years <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    d1 = (np.log(spot / strike) + (_RISK_FREE_RATE + 0.5 * sigma ** 2) * years) / (sigma * np.sqrt(years))
    d2 = d1 - sigma * np.sqrt(years)
    if is_call:
        return spot * norm.cdf(d1) - strike * np.exp(-_RISK_FREE_RATE * years) * norm.cdf(d2)
    return strike * np.exp(-_RISK_FREE_RATE * years) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def _implied_vol(price: float, spot: float, strike: float, years: float, is_call: bool) -> float | None:
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if not np.isfinite(price) or price <= intrinsic + 1e-6 or years <= 0:
        return None
    try:
        return float(brentq(lambda s: _bs_price(spot, strike, years, s, is_call) - price, 1e-4, 6.0, xtol=1e-4))
    except ValueError:
        return None


def load_chain(option_chain_dir: Path, cache_path: Path | None = None) -> pd.DataFrame:
    """Loads and concats every expiry file under option_chain_dir. Public (not
    underscore-prefixed) because straddle_path.py/credit_spread_path.py also need the raw
    chain to slice a specific contract's own bars forward for a trade, not just the
    per-timestamp ATM snapshot this module builds features from.

    Cached to a single merged parquet, mtime-invalidated against the newest source file
    (same pattern as api/main.py's _sell_sources()/_market_ohlc()) -- re-reading +
    concatenating 97 files (~300MB, ~14.5M rows) from scratch is the dominant cost in
    every script that touches the option chain (dataset.py, entry_exit_backtest.py,
    gen_indices_signals.py, sell_backtest.py all call this), and the source files don't
    change between runs. Rebuilds automatically if any source file is newer than the cache
    (e.g. after a fresh download)."""
    cache_path = cache_path or (option_chain_dir.parent / f"{option_chain_dir.name}_merged_cache.parquet")
    files = sorted(glob.glob(str(option_chain_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {option_chain_dir}")
    newest_source = max(Path(f).stat().st_mtime for f in files)
    if cache_path.exists() and cache_path.stat().st_mtime >= newest_source:
        chain = pd.read_parquet(cache_path)
    else:
        frames = [pd.read_parquet(f, columns=OPTION_COLUMNS) for f in files]
        chain = pd.concat(frames, ignore_index=True)
        chain["timestamp"] = pd.to_datetime(chain["timestamp"])
        chain["expiry"] = pd.to_datetime(chain["expiry"])
        chain.to_parquet(cache_path, index=False)
    chain["timestamp"] = pd.to_datetime(chain["timestamp"])
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    return chain


def build_option_features(option_chain_dir: Path) -> pd.DataFrame:
    """Returns one row per timestamp with feature_atm_*/feature_pcr_oi columns, ready to
    merge onto the price-feature frame on `timestamp`. Note: computes a Black-Scholes IV
    solve per (timestamp, side) after narrowing to one ATM row per timestamp (not per
    strike), but is still a per-row Python loop over ~a few hundred thousand rows -- this
    is a one-time offline dataset-build step, expect it to take a few minutes."""
    chain = load_chain(option_chain_dir).sort_values(["timestamp", "expiry"])

    # front expiry per timestamp = nearest expiry not yet past as of that bar's date
    front_expiry = (chain[chain["expiry"].dt.date >= chain["timestamp"].dt.date]
                     .groupby("timestamp")["expiry"].min())
    chain = chain.merge(front_expiry.rename("front_expiry"), on="timestamp", how="inner")
    front = chain[chain["expiry"] == chain["front_expiry"]].copy()

    # PCR needs the whole front-week chain (all strikes) at that timestamp
    oi_sums = front.groupby("timestamp")[["pe_oi", "ce_oi"]].sum()
    pcr = (oi_sums["pe_oi"] / oi_sums["ce_oi"].replace(0, np.nan)).rename("feature_pcr_oi")

    # ATM strike: nearest |strike - underlying_close| per timestamp
    front["strike_dist"] = (front["strike"] - front["underlying_close"]).abs()
    atm = front.sort_values("strike_dist").groupby("timestamp", as_index=False).first()

    naive_ts = atm["timestamp"].dt.tz_localize(None)
    atm["feature_dte"] = (atm["front_expiry"].dt.normalize() - naive_ts.dt.normalize()).dt.days
    dte_years = (atm["feature_dte"].clip(lower=0) + 0.5) / 365.0   # +0.5d fudge: same-day
                                                                     # expiry still has hours
                                                                     # of time value, not 0
    atm["feature_atm_iv_ce"] = [
        _implied_vol(p, s, k, y, True)
        for p, s, k, y in zip(atm["ce_close"], atm["underlying_close"], atm["strike"], dte_years)
    ]
    atm["feature_atm_iv_pe"] = [
        _implied_vol(p, s, k, y, False)
        for p, s, k, y in zip(atm["pe_close"], atm["underlying_close"], atm["strike"], dte_years)
    ]
    atm["feature_atm_iv_avg"] = atm[["feature_atm_iv_ce", "feature_atm_iv_pe"]].mean(axis=1)
    atm["feature_atm_straddle_premium"] = atm["ce_close"].fillna(0) + atm["pe_close"].fillna(0)
    atm["feature_atm_oi_total"] = atm["ce_oi"].fillna(0) + atm["pe_oi"].fillna(0)

    out = atm[["timestamp", "feature_atm_iv_ce", "feature_atm_iv_pe", "feature_atm_iv_avg",
               "feature_atm_straddle_premium", "feature_atm_oi_total", "feature_dte"]]
    out = out.merge(pcr, on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)

    # momentum on straddle premium / IV / OI, reset at each session boundary
    out["date"] = out["timestamp"].dt.date
    day = out.groupby("date", sort=False, group_keys=False)
    out["feature_atm_straddle_return_1bar"] = day["feature_atm_straddle_premium"].transform(lambda s: s.pct_change())
    out["feature_atm_straddle_return_3bar"] = day["feature_atm_straddle_premium"].transform(lambda s: s.pct_change(3))
    out["feature_atm_iv_chg_3bar"] = day["feature_atm_iv_avg"].transform(lambda s: s.diff(3))
    out["feature_atm_oi_change_1bar"] = day["feature_atm_oi_total"].transform(lambda s: s.pct_change())
    return out.drop(columns=["date"])
