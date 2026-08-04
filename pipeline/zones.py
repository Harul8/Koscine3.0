"""
Support / Resistance Zone detection — see ARCHITECTURE.md §4 Group 4.

Algorithm:
    1. Resample daily OHLC → weekly (W-FRI).
    2. Find swing highs/lows over the last N weeks using a ±k-bar local extremum.
    3. Cluster swing highs (and lows) within `cluster_pct` band → one zone per cluster.
    4. For each zone, count "valid touches" using the 5% retreat rule:
       price must have moved ≥`retreat_pct` away from the zone since the last touch
       before another touch counts. Consolidation AT the zone for many days does NOT
       inflate the touch count.
    5. Flag `consolidating_at_zone` separately when price stays inside the band for
       ≥`consolidation_days` without a `retreat_pct` retreat.
    6. Polarity flip: a level that has acted as BOTH support and resistance at
       different times is a stronger zone.

Output features per (date, symbol) row — written into gold/zones.parquet:
    resistance_level, support_level                  : price of nearest zone
    nearest_resistance_dist, nearest_support_dist    : (zone - close) / close
    resistance_valid_touches, support_valid_touches  : 5% rule applied
    resistance_zone_strength, support_zone_strength  : touches × log(weeks+1)
    consolidating_at_resistance, consolidating_at_support  : binary flags
    resistance_zone_age_weeks, support_zone_age_weeks: weeks since last swing in zone
    lt_resistance_dist, lt_support_dist              : dist to nearest zone > 2 yrs old
    zone_box_width_pct, weeks_in_box                 : full S/R box geometry
    polarity_flip                                    : level has flipped roles
    zone_breakout, zone_breakdown                    : binary, last 5 trading days

Full price history back to 2010 is used (ZONE_LOOKBACK_WEEKS=9999) so that
4-6 year old support/resistance levels — which can still act as magnets during
aggressive corrections — are visible to the model via lt_support_dist.

This module is called from features.py during the gold-feature build.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from collections import deque

from .config import (
    ZONE_LOOKBACK_WEEKS, ZONE_SWING_BARS, ZONE_CLUSTER_PCT,
    ZONE_RETREAT_PCT, ZONE_CONSOLIDATION_DAYS,
)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Zone:
    """A support or resistance zone built from a cluster of swing extremes."""
    level: float                # median of the cluster
    band_low: float             # level * (1 - cluster_pct/2)
    band_high: float            # level * (1 + cluster_pct/2)
    role: str                   # 'resistance' or 'support'
    first_swing_date: pd.Timestamp
    last_swing_date:  pd.Timestamp
    n_swings: int               # raw count of swing extremes in the cluster


class _ZoneState:
    """
    Mutable zone state used during the O(N) forward pass.

    Each instance represents one zone whose level, touch count, retreat
    eligibility, and polarity data are updated incrementally — one daily
    bar at a time — instead of re-scanning the full history at every
    sample date.

    Zone level uses an online mean (≈median for tight clusters within 2%).
    """
    __slots__ = (
        "level", "band_low", "band_high", "role",
        "first_swing_date", "last_swing_date", "n_swings",
        "touch_count", "eligible", "last_touch_price",
        "first_touch_date", "consec_in_band",
        "bars_above", "bars_below",
    )

    def __init__(self, price: float, role: str, date: pd.Timestamp):
        _h = ZONE_CLUSTER_PCT / 2
        self.level            = price
        self.band_low         = price * (1 - _h)
        self.band_high        = price * (1 + _h)
        self.role             = role
        self.first_swing_date = date
        self.last_swing_date  = date
        self.n_swings         = 1
        # touch tracking
        self.touch_count      = 0
        self.eligible         = True       # cleared after a touch, reset after 5% retreat
        self.last_touch_price = 0.0
        self.first_touch_date: pd.Timestamp | None = None
        self.consec_in_band   = 0
        # polarity flip counters
        self.bars_above       = 0
        self.bars_below       = 0

    def absorb(self, price: float, date: pd.Timestamp) -> None:
        """Merge a new confirmed swing into this zone (online running mean)."""
        self.n_swings += 1
        self.level = self.level + (price - self.level) / self.n_swings
        _h = ZONE_CLUSTER_PCT / 2
        self.band_low  = self.level * (1 - _h)
        self.band_high = self.level * (1 + _h)
        if date > self.last_swing_date:
            self.last_swing_date = date

    def step(self, price: float, date: pd.Timestamp) -> None:
        """Update state for one daily closing price."""
        in_band = self.band_low <= price <= self.band_high

        if in_band:
            if self.eligible:
                self.touch_count += 1
                if self.first_touch_date is None:
                    self.first_touch_date = date
                self.last_touch_price = price
                self.eligible = False
            self.consec_in_band += 1
        else:
            self.consec_in_band = 0
            if self.last_touch_price > 0:
                if self.role == "resistance":
                    if price <= self.last_touch_price * (1 - ZONE_RETREAT_PCT):
                        self.eligible = True
                else:
                    if price >= self.last_touch_price * (1 + ZONE_RETREAT_PCT):
                        self.eligible = True

        if price > self.band_high:
            self.bars_above += 1
        elif price < self.band_low:
            self.bars_below += 1


# ──────────────────────────────────────────────────────────────────────
# Step 1-2: weekly resample + swing detection
# ──────────────────────────────────────────────────────────────────────
def _weekly(stock_one_symbol: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLC → weekly W-FRI for a single symbol."""
    s = stock_one_symbol.set_index("date").sort_index()
    w = s.resample("W-FRI").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna(subset=["close"])
    return w


def _find_swing_highs(weekly: pd.DataFrame, k: int = ZONE_SWING_BARS) -> pd.Series:
    """Indices of weekly bars whose high is the local max over a ±k window."""
    h = weekly["high"].values
    n = len(h)
    flags = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        window = h[i - k:i + k + 1]
        if h[i] == window.max() and (window == h[i]).sum() == 1:
            flags[i] = True
    return pd.Series(flags, index=weekly.index)


def _find_swing_lows(weekly: pd.DataFrame, k: int = ZONE_SWING_BARS) -> pd.Series:
    l = weekly["low"].values
    n = len(l)
    flags = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        window = l[i - k:i + k + 1]
        if l[i] == window.min() and (window == l[i]).sum() == 1:
            flags[i] = True
    return pd.Series(flags, index=weekly.index)


# ──────────────────────────────────────────────────────────────────────
# Step 3: cluster swing extremes into zones
# ──────────────────────────────────────────────────────────────────────
def _cluster_swings(
    weekly: pd.DataFrame,
    swing_mask: pd.Series,
    price_col: str,
    role: str,
    cluster_pct: float = ZONE_CLUSTER_PCT,
) -> list[Zone]:
    """Greedy linkage: walk swings from highest to lowest price, merge any within
    cluster_pct of the median of the current group."""
    swings = weekly.loc[swing_mask, [price_col]].copy()
    swings["date"] = swings.index
    swings = swings.sort_values(price_col, ascending=False).reset_index(drop=True)
    if swings.empty:
        return []

    used = np.zeros(len(swings), dtype=bool)
    zones: list[Zone] = []
    for i in range(len(swings)):
        if used[i]:
            continue
        seed = float(swings.iloc[i][price_col])
        band_lo = seed * (1 - cluster_pct / 2)
        band_hi = seed * (1 + cluster_pct / 2)
        members_idx = []
        for j in range(i, len(swings)):
            if used[j]:
                continue
            p = float(swings.iloc[j][price_col])
            if band_lo <= p <= band_hi:
                members_idx.append(j)
                used[j] = True
        if not members_idx:
            continue
        member_prices = [float(swings.iloc[m][price_col]) for m in members_idx]
        member_dates  = [swings.iloc[m]["date"] for m in members_idx]
        median = float(np.median(member_prices))
        zones.append(Zone(
            level=median,
            band_low=median * (1 - cluster_pct / 2),
            band_high=median * (1 + cluster_pct / 2),
            role=role,
            first_swing_date=min(member_dates),
            last_swing_date=max(member_dates),
            n_swings=len(members_idx),
        ))
    return zones


# ──────────────────────────────────────────────────────────────────────
# Step 4-5: count valid touches with 5% retreat rule + consolidation flag
# ──────────────────────────────────────────────────────────────────────
def _count_valid_touches_and_consolidation(
    daily: pd.DataFrame,
    zone: Zone,
    asof: pd.Timestamp,
    retreat_pct: float = ZONE_RETREAT_PCT,
    consolidation_days: int = ZONE_CONSOLIDATION_DAYS,
) -> tuple[int, bool, int]:
    """
    Walk daily price series for this symbol UP TO `asof`.
    Returns:
      valid_touches : count of approaches qualifying under 5% retreat rule
      consolidating : True if price has been within the zone band for ≥N days as of `asof`
                       without a 5% retreat
      weeks_in_zone : approximate weeks since the FIRST valid touch
    """
    d = daily[daily["date"] <= asof]
    if d.empty:
        return 0, False, 0

    close = d["close"].values
    dates = d["date"].values
    in_band = (close >= zone.band_low) & (close <= zone.band_high)

    valid_touches = 0
    first_touch_date = None
    last_touch_price = None
    eligible_for_new_touch = True       # set False after a touch until 5% retreat clears it
    consec_in_band = 0

    for i in range(len(close)):
        price = close[i]

        if in_band[i]:
            if eligible_for_new_touch:
                valid_touches += 1
                if first_touch_date is None:
                    first_touch_date = dates[i]
                last_touch_price = price
                eligible_for_new_touch = False
            consec_in_band += 1
        else:
            consec_in_band = 0
            if last_touch_price is not None:
                if zone.role == "resistance":
                    if price <= last_touch_price * (1 - retreat_pct):
                        eligible_for_new_touch = True
                else:  # support
                    if price >= last_touch_price * (1 + retreat_pct):
                        eligible_for_new_touch = True

    consolidating = consec_in_band >= consolidation_days

    if first_touch_date is None:
        weeks_in_zone = 0
    else:
        weeks_in_zone = max(1, int(
            (pd.Timestamp(asof) - pd.Timestamp(first_touch_date)).days / 7
        ))
    return valid_touches, consolidating, weeks_in_zone


# ──────────────────────────────────────────────────────────────────────
# Step 6: polarity flip detection
# ──────────────────────────────────────────────────────────────────────
def _detect_polarity_flip(
    daily: pd.DataFrame,
    zone: Zone,
    asof: pd.Timestamp,
    retreat_pct: float = ZONE_RETREAT_PCT,
) -> bool:
    """
    Did this price band serve as BOTH support and resistance historically?

    Heuristic: walk the price series up to `asof`. If price was above the band
    for ≥10 bars AND below the band for ≥10 bars at some point, the level has
    acted as both roles.
    """
    d = daily[daily["date"] <= asof]
    if d.empty:
        return False
    close = d["close"].values
    above = (close > zone.band_high).sum()
    below = (close < zone.band_low).sum()
    return above >= 10 and below >= 10


# ──────────────────────────────────────────────────────────────────────
# Per-symbol orchestrator (run for each as-of date OR once with rolling lookback)
# ──────────────────────────────────────────────────────────────────────
def _zones_for_symbol_asof(
    daily: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_weeks: int = ZONE_LOOKBACK_WEEKS,
) -> dict:
    """
    Compute zone features for one symbol AS-OF a specific date.

    When lookback_weeks >= 9999 (the default via config.ZONE_LOOKBACK_WEEKS) the
    full price history up to `asof` is used — no lower cutoff.  This lets the
    model see 4-6 year old support/resistance levels that a falling stock may
    revisit.  A finite lookback_weeks still works for backward compatibility.

    Returns a dict matching the output column layout. All values nullable.
    """
    if lookback_weeks >= 9999:
        hist = daily[daily["date"] <= asof].copy()
    else:
        cutoff = asof - pd.Timedelta(weeks=lookback_weeks)
        hist = daily[(daily["date"] >= cutoff) & (daily["date"] <= asof)].copy()

    if len(hist) < 30:
        return _empty_row()

    weekly = _weekly(hist)
    if len(weekly) < 2 * ZONE_SWING_BARS + 2:
        return _empty_row()

    sh_mask = _find_swing_highs(weekly)
    sl_mask = _find_swing_lows(weekly)

    res_zones = _cluster_swings(weekly, sh_mask, "high", "resistance")
    sup_zones = _cluster_swings(weekly, sl_mask, "low",  "support")

    close_now = float(hist.iloc[-1]["close"])

    # ── Nearest resistance ABOVE current price ─────────────────────────
    res_above = [z for z in res_zones if z.level >= close_now]
    nearest_res = min(res_above, key=lambda z: z.level - close_now) if res_above else None

    # ── Nearest support BELOW current price ────────────────────────────
    sup_below = [z for z in sup_zones if z.level <= close_now]
    nearest_sup = max(sup_below, key=lambda z: z.level) if sup_below else None

    # ── Long-term zones (last swing > 2 years ago = 104 weeks) ────────
    # A stock in a severe correction may gravitate toward support levels
    # formed 4-6 years ago — capture those separately from the nearest zone.
    _LT_WEEKS = 104
    lt_res_above = [z for z in res_above
                    if (asof - z.last_swing_date).days / 7 > _LT_WEEKS]
    lt_sup_below = [z for z in sup_below
                    if (asof - z.last_swing_date).days / 7 > _LT_WEEKS]
    nearest_lt_res = (min(lt_res_above, key=lambda z: z.level - close_now)
                      if lt_res_above else None)
    nearest_lt_sup = (max(lt_sup_below, key=lambda z: z.level)
                      if lt_sup_below else None)

    row = _empty_row()
    if nearest_res is not None:
        t, cons, w = _count_valid_touches_and_consolidation(hist, nearest_res, asof)
        flip = _detect_polarity_flip(hist, nearest_res, asof)
        age_weeks = (asof - nearest_res.last_swing_date).days / 7
        row.update({
            "resistance_level":            nearest_res.level,
            "nearest_resistance_dist":     (nearest_res.level - close_now) / close_now,
            "resistance_valid_touches":    t,
            "resistance_zone_strength":    t * math.log(w + 1),
            "consolidating_at_resistance": int(cons),
            "polarity_flip_resistance":    int(flip),
            "resistance_zone_age_weeks":   age_weeks,
        })

    if nearest_sup is not None:
        t, cons, w = _count_valid_touches_and_consolidation(hist, nearest_sup, asof)
        flip = _detect_polarity_flip(hist, nearest_sup, asof)
        age_weeks = (asof - nearest_sup.last_swing_date).days / 7
        row.update({
            "support_level":            nearest_sup.level,
            "nearest_support_dist":     (close_now - nearest_sup.level) / close_now,
            "support_valid_touches":    t,
            "support_zone_strength":    t * math.log(w + 1),
            "consolidating_at_support": int(cons),
            "polarity_flip_support":    int(flip),
            "support_zone_age_weeks":   age_weeks,
        })

    # ── Long-term zone distances ────────────────────────────────────────
    if nearest_lt_res is not None:
        row["lt_resistance_dist"] = (nearest_lt_res.level - close_now) / close_now
    if nearest_lt_sup is not None:
        row["lt_support_dist"] = (close_now - nearest_lt_sup.level) / close_now

    # ── Box geometry ───────────────────────────────────────────────────
    if nearest_res is not None and nearest_sup is not None:
        row["zone_box_width_pct"] = (nearest_res.level - nearest_sup.level) / nearest_sup.level
        in_box_mask = (hist["close"] >= nearest_sup.band_low) & (hist["close"] <= nearest_res.band_high)
        row["weeks_in_box"] = int(in_box_mask.sum() / 5)  # ~5 trading days per week

    # ── Breakout / breakdown in last 5 trading days ────────────────────
    last5 = hist.tail(5)
    if nearest_res is not None and len(last5) > 0:
        # any day in the last 5 closed above the resistance band high
        row["zone_breakout"] = int((last5["close"] > nearest_res.band_high).any())
    if nearest_sup is not None and len(last5) > 0:
        row["zone_breakdown"] = int((last5["close"] < nearest_sup.band_low).any())

    row["polarity_flip"] = int(
        bool(row["polarity_flip_resistance"]) or bool(row["polarity_flip_support"])
    )
    return row


def _empty_row() -> dict:
    return {
        "resistance_level":            np.nan,
        "support_level":               np.nan,
        "nearest_resistance_dist":     np.nan,
        "nearest_support_dist":        np.nan,
        "resistance_valid_touches":    0,
        "support_valid_touches":       0,
        "resistance_zone_strength":    0.0,
        "support_zone_strength":       0.0,
        "consolidating_at_resistance": 0,
        "consolidating_at_support":    0,
        "polarity_flip_resistance":    0,
        "polarity_flip_support":       0,
        "polarity_flip":               0,
        "zone_box_width_pct":          np.nan,
        "weeks_in_box":                0,
        "zone_breakout":               0,
        "zone_breakdown":              0,
        # Zone age — weeks since the last swing that defined the nearest zone
        "resistance_zone_age_weeks":   np.nan,
        "support_zone_age_weeks":      np.nan,
        # Long-term zones (last swing > 2 years old) — dist from current price
        "lt_resistance_dist":          np.nan,
        "lt_support_dist":             np.nan,
    }


# ──────────────────────────────────────────────────────────────────────
# Forward-pass helpers (O(N) full rebuild)
# ──────────────────────────────────────────────────────────────────────

def _integrate(states: list, price: float, date: pd.Timestamp, role: str) -> None:
    """
    Online clustering: absorb a new confirmed swing into the nearest matching
    zone, or start a new zone if no existing zone's band contains `price`.

    Equivalent to the batch greedy linkage in _cluster_swings but operates
    on one swing at a time, enabling the forward pass.
    """
    for s in states:
        if s.band_low <= price <= s.band_high:
            s.absorb(price, date)
            return
    states.append(_ZoneState(price, role, date))


def _snapshot_states(
    res_states: list,
    sup_states: list,
    close_now: float,
    asof: pd.Timestamp,
    last5: list,
) -> dict:
    """
    Build an _empty_row()-shaped feature dict from the current running
    _ZoneState lists.  Called at each sample date during the forward pass.
    """
    _LT_WEEKS = 104
    row = _empty_row()

    res_above = [s for s in res_states if s.level >= close_now]
    sup_below = [s for s in sup_states if s.level <= close_now]
    nearest_res = (min(res_above, key=lambda s: s.level - close_now)
                   if res_above else None)
    nearest_sup = (max(sup_below, key=lambda s: s.level)
                   if sup_below else None)

    lt_res = [s for s in res_above
              if (asof - s.last_swing_date).days / 7 > _LT_WEEKS]
    lt_sup = [s for s in sup_below
              if (asof - s.last_swing_date).days / 7 > _LT_WEEKS]
    n_lt_res = (min(lt_res, key=lambda s: s.level - close_now) if lt_res else None)
    n_lt_sup = (max(lt_sup, key=lambda s: s.level) if lt_sup else None)

    if nearest_res is not None:
        s = nearest_res
        w = (max(1, int((asof - s.first_touch_date).days / 7))
             if s.first_touch_date else 0)
        row.update({
            "resistance_level":            s.level,
            "nearest_resistance_dist":     (s.level - close_now) / close_now,
            "resistance_valid_touches":    s.touch_count,
            "resistance_zone_strength":    s.touch_count * math.log(w + 1),
            "consolidating_at_resistance": int(s.consec_in_band >= ZONE_CONSOLIDATION_DAYS),
            "polarity_flip_resistance":    int(s.bars_above >= 10 and s.bars_below >= 10),
            "resistance_zone_age_weeks":   (asof - s.last_swing_date).days / 7,
        })
        if last5:
            row["zone_breakout"] = int(any(c > s.band_high for c in last5))

    if nearest_sup is not None:
        s = nearest_sup
        w = (max(1, int((asof - s.first_touch_date).days / 7))
             if s.first_touch_date else 0)
        row.update({
            "support_level":            s.level,
            "nearest_support_dist":     (close_now - s.level) / close_now,
            "support_valid_touches":    s.touch_count,
            "support_zone_strength":    s.touch_count * math.log(w + 1),
            "consolidating_at_support": int(s.consec_in_band >= ZONE_CONSOLIDATION_DAYS),
            "polarity_flip_support":    int(s.bars_above >= 10 and s.bars_below >= 10),
            "support_zone_age_weeks":   (asof - s.last_swing_date).days / 7,
        })
        if last5:
            row["zone_breakdown"] = int(any(c < s.band_low for c in last5))

    if n_lt_res is not None:
        row["lt_resistance_dist"] = (n_lt_res.level - close_now) / close_now
    if n_lt_sup is not None:
        row["lt_support_dist"] = (close_now - n_lt_sup.level) / close_now

    if nearest_res is not None and nearest_sup is not None:
        row["zone_box_width_pct"] = (
            (nearest_res.level - nearest_sup.level) / nearest_sup.level
        )
        ft_r = nearest_res.first_touch_date
        ft_s = nearest_sup.first_touch_date
        if ft_r and ft_s:
            row["weeks_in_box"] = max(0, int((asof - max(ft_r, ft_s)).days / 7))

    row["polarity_flip"] = int(
        bool(row["polarity_flip_resistance"]) or bool(row["polarity_flip_support"])
    )
    return row


def _build_symbol_zones_forward(
    daily_sym: pd.DataFrame,
    cadence_days: int = 5,
) -> pd.DataFrame:
    """
    Single O(N) chronological forward pass through one symbol's history.

    Why this is faster than _zones_for_symbol_asof called per sample date
    -----------------------------------------------------------------------
    The original approach re-processes the FULL daily history for every
    sample date → O(N²) total.  With 15 years of history and weekly sampling
    that is ~780 full scans per symbol.

    Here instead we do ONE scan:
      • Swing detection: bar wi-k is checked once it has k bars on both sides.
      • Touch counting: each daily bar updates all active zones exactly once.
      • Snapshots: zone state is read out at sample dates (no recomputation).

    Complexity: O(W × Z) where W = weekly bars and Z = active zones (≤ ~20).
    For 15 years: ~780 weeks × 20 zones = ~15 600 state updates vs 780 × 3 500
    daily-bar rescans = ~2.7 M in the old approach — ~170× fewer operations.

    Parameters
    ----------
    daily_sym    : daily OHLCV for one symbol, sorted ascending by date
    cadence_days : sample every N trading days (5 ≈ weekly)

    Returns
    -------
    DataFrame of sampled zone-feature rows (not yet forward-filled to daily).
    """
    k = ZONE_SWING_BARS

    daily_sym = daily_sym.sort_values("date").reset_index(drop=True)
    if len(daily_sym) < 30:
        return pd.DataFrame()

    weekly = _weekly(daily_sym)
    if len(weekly) < 2 * k + 2:
        return pd.DataFrame()

    w_ts     = weekly.index.tolist()          # list[pd.Timestamp] — Fridays
    w_highs  = weekly["high"].values
    w_lows   = weekly["low"].values

    d_dates  = daily_sym["date"].values       # numpy datetime64
    d_closes = daily_sym["close"].values
    n_daily  = len(d_dates)
    n_weekly = len(w_ts)

    res_states: list[_ZoneState] = []
    sup_states: list[_ZoneState] = []
    last5: deque = deque(maxlen=5)            # rolling 5-day close buffer

    results: list[dict] = []
    d_ptr  = 0          # pointer into daily arrays — advanced monotonically

    for wi, w_end in enumerate(w_ts):
        # ── 1. Integrate any newly confirmed swing at candidate wi−k ─────
        # A swing high at position ci is confirmed once ci+k bars exist after it.
        if wi >= 2 * k:
            ci = wi - k
            lo, hi = max(0, ci - k), ci + k + 1

            if (w_highs[ci] == w_highs[lo:hi].max() and
                    (w_highs[lo:hi] == w_highs[ci]).sum() == 1):
                _integrate(res_states, float(w_highs[ci]), w_ts[ci], "resistance")

            if (w_lows[ci] == w_lows[lo:hi].min() and
                    (w_lows[lo:hi] == w_lows[ci]).sum() == 1):
                _integrate(sup_states, float(w_lows[ci]), w_ts[ci], "support")

        # ── 2. Advance daily pointer through bars in this week ────────────
        days_this_week = 0
        while d_ptr < n_daily and pd.Timestamp(d_dates[d_ptr]) <= w_end:
            price = float(d_closes[d_ptr])
            date  = pd.Timestamp(d_dates[d_ptr])
            last5.append(price)
            for s in res_states:
                s.step(price, date)
            for s in sup_states:
                s.step(price, date)
            d_ptr += 1
            days_this_week += 1

        # ── 3. Snapshot at sample dates ───────────────────────────────────
        # wi*5 approximates the cumulative trading-day index.
        if days_this_week > 0 and (
            (wi * 5) % cadence_days == 0 or wi == n_weekly - 1
        ):
            close_now = float(d_closes[d_ptr - 1])
            snap = _snapshot_states(
                res_states, sup_states, close_now, w_end, list(last5)
            )
            snap["date"] = w_end
            results.append(snap)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────
# Public API — compute zone features for a full panel
# ──────────────────────────────────────────────────────────────────────
def compute_zone_features(
    stock: pd.DataFrame,
    asof_dates: list[pd.Timestamp] | None = None,
    sample_every_n_days: int = 5,
) -> pd.DataFrame:
    """
    Compute zone features for every (date, symbol) row in `stock`.

    Performance note: zone computation is O(history × symbols × dates). Re-computing
    every day is expensive — but zones change slowly (weekly cadence). To balance
    accuracy and speed:
      - Compute fresh zones every `sample_every_n_days` (default 5 = weekly)
      - Forward-fill the result for the in-between days

    Args:
        stock: panel df with columns (date, symbol, open, high, low, close)
        asof_dates: optional list of dates to compute on. If None, samples every Nth date.
        sample_every_n_days: if asof_dates is None, sampling cadence.

    Returns:
        DataFrame with (date, symbol) + zone feature columns, forward-filled to daily.
    """
    stock = stock.sort_values(["symbol", "date"]).copy()
    stock["date"] = pd.to_datetime(stock["date"])

    all_dates = sorted(stock["date"].unique())
    if asof_dates is None:
        # take every Nth trading date
        asof_dates = all_dates[::sample_every_n_days]
        # always include the last available date for live inference
        if all_dates[-1] not in asof_dates:
            asof_dates = list(asof_dates) + [all_dates[-1]]

    print(f"[zones] computing on {len(asof_dates)} sampled dates "
          f"× {stock['symbol'].nunique()} symbols")

    out_rows = []
    by_sym = dict(tuple(stock.groupby("symbol", sort=False)))

    for i, asof in enumerate(asof_dates):
        if (i + 1) % 50 == 0:
            print(f"  [zones] {i + 1}/{len(asof_dates)}  asof={asof.date()}")
        for sym, daily in by_sym.items():
            if daily["date"].iloc[0] > asof or daily["date"].iloc[-1] < asof:
                continue
            row = _zones_for_symbol_asof(daily, pd.Timestamp(asof))
            row["date"]   = asof
            row["symbol"] = sym
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    sampled = pd.DataFrame(out_rows)
    # Forward-fill to daily resolution per symbol
    daily_index = stock[["date", "symbol"]].drop_duplicates()
    out = (daily_index.merge(sampled, on=["date", "symbol"], how="left")
                       .sort_values(["symbol", "date"]))
    feat_cols = [c for c in sampled.columns if c not in ("date", "symbol")]
    out[feat_cols] = (out.groupby("symbol", sort=False)[feat_cols].ffill())
    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Standalone builder — run weekly via scheduler or manually
# ──────────────────────────────────────────────────────────────────────

def build_zones(cadence_days: int = 5, full_rebuild: bool = False) -> None:
    """
    Build or incrementally update gold/zones.parquet.

    Strategy
    --------
    Zone features for a given (date, symbol) are computed strictly from
    data up to that date — they are immutable once written.  There is
    therefore no reason to recompute old rows; we only append new ones.

    • First run (no cache)  → full build over entire silver history.
    • Subsequent runs       → load cache, find last cached date, compute
                              only for new trading dates, forward-fill from
                              last known values, append.

    This makes the weekly job fast: typically 5-10 new sample dates instead
    of the full 1,200+ dates in the history.

    Usage
    -----
        python -m pipeline.zones                   # incremental (default)
        python -m pipeline.zones --full            # force full rebuild
        python -m pipeline.zones --cadence 1       # daily cadence
    """
    import time
    from .config import SILVER_TABLES, GOLD_ZONES

    print("[zones] loading silver eod_stock …")
    stock = pd.read_parquet(SILVER_TABLES["eod_stock"])
    stock["date"] = pd.to_datetime(stock["date"])

    # Filter to F&O universe — features.py only uses F&O symbols (filtered via
    # eod_deriv_daily).  Building zones for all 3 700+ NSE stocks wastes ~9×
    # the compute with no benefit to the model.
    try:
        _deriv = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"],
                                 columns=["symbol"])
        fo_syms = set(_deriv["symbol"].unique())
        before = stock["symbol"].nunique()
        stock = stock[stock["symbol"].isin(fo_syms)].reset_index(drop=True)
        print(f"  filtered to F&O universe: {stock['symbol'].nunique()} / {before} symbols")
    except Exception as e:
        print(f"  [zones] could not filter to F&O symbols ({e}) — using all")

    all_dates = sorted(stock["date"].unique())
    print(f"  {len(stock):,} rows  {stock['symbol'].nunique()} symbols  "
          f"{all_dates[0].date()} → {all_dates[-1].date()}")

    GOLD_ZONES.parent.mkdir(parents=True, exist_ok=True)

    # ── Decide: full rebuild or incremental ───────────────────────────────
    existing: pd.DataFrame | None = None
    last_cached: pd.Timestamp | None = None

    if not full_rebuild and GOLD_ZONES.exists():
        try:
            existing = pd.read_parquet(GOLD_ZONES)
            existing["date"] = pd.to_datetime(existing["date"])
            last_cached = existing["date"].max()
            new_dates = [d for d in all_dates if d > last_cached]
            if not new_dates:
                print(f"[zones] already up to date through {last_cached.date()} — nothing to do")
                return
            print(f"[zones] incremental update: {len(new_dates)} new trading days "
                  f"({last_cached.date()} → {new_dates[-1].date()})")
        except Exception as e:
            print(f"[zones] cache unreadable ({e}) — falling back to full rebuild")
            existing = None
            last_cached = None

    if existing is None:
        # ── Full rebuild — O(N) forward-pass per symbol ───────────────────
        # One chronological scan per symbol; zone state is maintained
        # incrementally.  ~170× fewer Python loop iterations vs the old
        # approach of calling _zones_for_symbol_asof at every sample date.
        print("[zones] full rebuild (forward-pass O(N) per symbol) …")
        t0 = time.time()
        by_sym = dict(tuple(stock.groupby("symbol", sort=False)))
        all_rows: list[pd.DataFrame] = []

        for i, (sym, daily_sym) in enumerate(by_sym.items()):
            if (i + 1) % 50 == 0 or (i + 1) == len(by_sym):
                print(f"  [zones] {i+1}/{len(by_sym)}  ({sym})")
            sym_rows = _build_symbol_zones_forward(daily_sym, cadence_days=cadence_days)
            if not sym_rows.empty:
                sym_rows["symbol"] = sym
                all_rows.append(sym_rows)

        if not all_rows:
            print("[zones] no rows computed — check silver/eod_stock.parquet")
            return

        sampled   = pd.concat(all_rows, ignore_index=True)
        feat_cols = [c for c in sampled.columns if c not in ("date", "symbol")]

        # Forward-fill sampled values to every trading day per symbol
        daily_idx = (stock[["date", "symbol"]]
                     .drop_duplicates()
                     .sort_values(["symbol", "date"]))
        out = (daily_idx
               .merge(sampled, on=["date", "symbol"], how="left")
               .sort_values(["symbol", "date"]))
        out[feat_cols] = out.groupby("symbol", sort=False)[feat_cols].ffill()
        out = out.reset_index(drop=True)
        out.to_parquet(GOLD_ZONES, index=False)
        print(f"[zones] saved {len(out):,} rows → {GOLD_ZONES}  ({time.time()-t0:.0f}s)")
        return

    # ── Incremental update ────────────────────────────────────────────────
    # 1. Pick sample dates from the new window only.
    #    Always include the last available date for live inference accuracy.
    new_dates = [d for d in all_dates if d > last_cached]
    new_sample_dates = new_dates[::cadence_days]
    if new_dates[-1] not in new_sample_dates:
        new_sample_dates = list(new_sample_dates) + [new_dates[-1]]

    print(f"[zones] computing {len(new_sample_dates)} sample dates "
          f"× {stock['symbol'].nunique()} symbols …")

    t0 = time.time()
    by_sym = dict(tuple(stock.groupby("symbol", sort=False)))
    new_rows = []
    for i, asof in enumerate(new_sample_dates):
        if (i + 1) % 10 == 0:
            print(f"  [zones] {i+1}/{len(new_sample_dates)}  asof={pd.Timestamp(asof).date()}")
        for sym, daily in by_sym.items():
            if daily["date"].iloc[0] > asof or daily["date"].iloc[-1] < asof:
                continue
            row = _zones_for_symbol_asof(daily, pd.Timestamp(asof))
            row["date"]   = asof
            row["symbol"] = sym
            new_rows.append(row)

    if not new_rows:
        print("[zones] no new rows computed")
        return

    new_sampled = pd.DataFrame(new_rows)
    feat_cols   = [c for c in new_sampled.columns if c not in ("date", "symbol")]

    # 2. Build a daily index for every new trading day.
    new_daily_idx = (
        stock[stock["date"] > last_cached][["date", "symbol"]]
        .drop_duplicates()
        .sort_values(["symbol", "date"])
    )

    # 3. Left-join sampled values onto daily index (NaN on non-sample days).
    new_out = new_daily_idx.merge(new_sampled, on=["date", "symbol"], how="left")
    new_out = new_out.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 4. Forward-fill within new window, seeded by last known value per symbol
    #    from the existing cache.  Without seeding, days before the first new
    #    sample date would have NaN even though the previous zone is still valid.
    last_known = (
        existing.sort_values("date")
        .groupby("symbol", sort=False)
        .last()
        .reset_index()[["symbol"] + feat_cols]
    )
    # Prepend seed rows (dated at last_cached so sort order is correct)
    seed = last_known.copy()
    seed["date"] = last_cached
    combined = (
        pd.concat([seed, new_out], ignore_index=True)
        .sort_values(["symbol", "date"])
    )
    combined[feat_cols] = (
        combined.groupby("symbol", sort=False)[feat_cols].ffill()
    )
    # Drop the seed rows — we only want genuinely new dates
    new_out_filled = combined[combined["date"] > last_cached].reset_index(drop=True)

    # 5. Append to existing cache and save.
    result = pd.concat([existing, new_out_filled], ignore_index=True)
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    result.to_parquet(GOLD_ZONES, index=False)

    elapsed = time.time() - t0
    print(f"[zones] appended {len(new_out_filled):,} rows "
          f"(total {len(result):,}) → {GOLD_ZONES}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Build / incrementally update gold/zones.parquet. "
                    "Run weekly — appends only new dates, preserving full history."
    )
    p.add_argument("--cadence", type=int, default=5,
                   help="Sample every N trading days (default 5 = weekly)")
    p.add_argument("--full", action="store_true",
                   help="Force full rebuild from scratch (ignores existing cache)")
    args = p.parse_args()
    build_zones(cadence_days=args.cadence, full_rebuild=args.full)
