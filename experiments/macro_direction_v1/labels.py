"""Target / label builders for macro_direction_v1.

realized_fwd          : signed close-to-close return over `horizon` days (the EVAL truth — full set).
signed_target         : training label = sign(realized fwd), with a dead-band that DROPS the ambiguous
                        middle from TRAINING (Lopez de Prado-style threshold labeling). Eval still uses the
                        full-set realized sign, so the headline number is not a selection artifact.
triple_barrier_target : vol-scaled upper/lower barriers + time barrier over `horizon` days on daily OHLC;
                        label by first barrier touched (time barrier -> sign of close move). Daily-OHLC
                        caveat: if both barriers touch the SAME day, intraday order is unknown -> that day's
                        close decides (conservative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_fwd(df: pd.DataFrame, horizon: int = 1, entry: str = "open") -> pd.Series:
    """Signed forward return actually capturable by the execution model.
    entry='open'  -> enter at the next session's OPEN (matches a ~7AM IST t+1 pre-open refresh that trades the
                     open); open->close is what you get, the overnight gap is already priced in.
    entry='close' -> enter at signal-day close (credits the un-enterable close[d]->open[d+1] overnight gap)."""
    g = df.groupby("symbol", sort=False)
    exit_px = g["close"].shift(-horizon)
    entry_px = g["open"].shift(-1) if entry == "open" else df["close"]
    return (exit_px - entry_px) / entry_px


def signed_target(fwd_ret: pd.Series, dead_band: float = 0.0) -> np.ndarray:
    f = fwd_ret.to_numpy()
    return np.where(f > dead_band, 1.0, np.where(f < -dead_band, 0.0, np.nan))


def triple_barrier_target(df: pd.DataFrame, horizon: int = 5, k_vol: float = 1.0,
                          vol_col: str = "atm_iv") -> np.ndarray:
    """`df` must be sorted by [symbol, date]; returns labels aligned to df's row order."""
    dv = (df[vol_col].to_numpy() / np.sqrt(252.0))
    med = np.nanmedian(dv)
    dv = np.where(np.isfinite(dv), dv, med)
    bar = k_vol * dv * np.sqrt(horizon)
    close = df["close"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy()
    sym = df["symbol"].to_numpy()
    n = len(df)
    lab = np.full(n, np.nan)
    for i in range(n):
        up = close[i] * (1 + bar[i]); dn = close[i] * (1 - bar[i])
        end = min(i + horizon, n - 1)
        last_valid = -1
        for j in range(i + 1, end + 1):
            if sym[j] != sym[i]:
                break
            last_valid = j
            hi_hit = high[j] >= up; lo_hit = low[j] <= dn
            if hi_hit and lo_hit:
                lab[i] = 1.0 if close[j] >= close[i] else 0.0
                last_valid = -2
                break
            if hi_hit:
                lab[i] = 1.0; last_valid = -2; break
            if lo_hit:
                lab[i] = 0.0; last_valid = -2; break
        if last_valid >= 0:                       # time barrier hit, no touch
            lab[i] = 1.0 if close[last_valid] >= close[i] else 0.0
    return lab
