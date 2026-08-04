import numpy as np
import pandas as pd

from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes

# 22 business days. A flat baseline (O=100,H=101,L=99,C=100) for the first rows
# establishes ATR(14) ~ 2 -> atr_pct ~ 0.02 -> stop_tol = 0.6*0.02 = 0.012 (1.2%).
# We signal at row S=15 (ATR is defined) and override the window rows S+1..S+5.
N = 22
SIGNAL = 15
CONTRACT = CleanMoveContract(atr_window=14, atr_mult=0.6)


def _baseline():
    return {
        "open": [100.0] * N,
        "high": [101.0] * N,
        "low": [99.0] * N,
        "close": [100.0] * N,
    }


def _frame(symbol, cols):
    dates = pd.date_range("2026-01-01", periods=N, freq="B")
    return pd.DataFrame({"date": dates, "symbol": symbol, **cols})


def _set(cols, row, high=None, low=None, close=None, open_=None):
    if high is not None:
        cols["high"][row] = high
    if low is not None:
        cols["low"][row] = low
    if close is not None:
        cols["close"][row] = close
    if open_ is not None:
        cols["open"][row] = open_


def _signal_row(outcomes, symbol, side):
    m = outcomes["symbol"].eq(symbol) & outcomes["side"].eq(side)
    return outcomes[m].sort_values("date").reset_index(drop=True).iloc[SIGNAL]


def test_long_clean_when_low_stays_above_stop():
    c = _baseline()
    for r in range(SIGNAL + 1, SIGNAL + 6):
        _set(c, r, low=100.0)  # no dip below entry -> floor_depth 0
    _set(c, SIGNAL + 2, high=108.0, close=104.0)  # peak +8% on window day 2
    r = _signal_row(compute_clean_move_outcomes(_frame("CLEAN", c), contract=CONTRACT), "CLEAN", "long")
    assert r["status"] == "evaluated"
    assert bool(r["clean"]) is True
    assert r["verdict"] == "clean"
    assert abs(r["ceiling"] - 0.08) < 1e-9
    assert abs(r["floor_depth"] - 0.0) < 1e-9
    assert r["days_to_peak"] == 2
    assert r["reaches_big_by_day"] == 2
    assert bool(r["reaches_big"]) is True


def test_long_stopped_when_low_breaches_stop():
    c = _baseline()
    for r in range(SIGNAL + 1, SIGNAL + 6):
        _set(c, r, low=100.0)
    _set(c, SIGNAL + 1, low=90.0, close=95.0)  # -10% dip day 1 breaches the stop
    _set(c, SIGNAL + 2, high=108.0, close=104.0)
    r = _signal_row(compute_clean_move_outcomes(_frame("STOP", c), contract=CONTRACT), "STOP", "long")
    assert r["status"] == "evaluated"
    assert bool(r["clean"]) is False
    assert r["verdict"] == "stopped"
    assert abs(r["floor_depth"] - 0.10) < 1e-9


def test_short_ceiling_uses_window_low():
    c = _baseline()
    # highs stay near entry (<=101.2) so the short is clean; lows drop to 93 -> ceiling 0.07
    _set(c, SIGNAL + 1, low=99.0)
    _set(c, SIGNAL + 2, low=93.0, close=94.0)
    for r in range(SIGNAL + 3, SIGNAL + 6):
        _set(c, r, low=98.0)
    r = _signal_row(compute_clean_move_outcomes(_frame("SH", c), contract=CONTRACT), "SH", "short")
    assert bool(r["clean"]) is True
    assert abs(r["ceiling"] - 0.07) < 1e-9
    assert r["days_to_peak"] == 2  # min low on window day 2


def test_pending_entry_on_last_row():
    out = compute_clean_move_outcomes(_frame("P", _baseline()), contract=CONTRACT)
    last = out["date"].max()
    r = out[out["date"].eq(last) & out["side"].eq("long")].iloc[0]
    assert r["status"] == "pending_entry"
    assert np.isnan(r["ceiling"])


def test_clean_label_is_exact_from_min_low():
    # floor_depth depends only on the window's min low vs the stop, independent of
    # intraday high-vs-low ordering -- the key property that makes the label trustworthy.
    c = _baseline()
    _set(c, SIGNAL + 1, low=99.7)
    _set(c, SIGNAL + 2, low=99.6, high=106.0, close=105.0)  # min low 99.6 -> floor_depth 0.004
    for r in range(SIGNAL + 3, SIGNAL + 6):
        _set(c, r, low=100.0)
    r = _signal_row(compute_clean_move_outcomes(_frame("AMB", c), contract=CONTRACT), "AMB", "long")
    assert abs(r["floor_depth"] - 0.004) < 1e-9
    assert bool(r["clean"]) is True  # 0.004 <= 0.012 stop tolerance
