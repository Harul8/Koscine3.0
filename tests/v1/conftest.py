"""A small synthetic market for the store tests: five stocks, two indices, known corporate actions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from koscine4.settings import load_settings, project_root

DAYS = pd.bdate_range("2024-01-01", periods=60)
EX = DAYS[30]            # the day every synthetic corporate action takes effect

# symbol -> (start price, factor applied from EX onwards)
STOCKS = {
    "AAA": (1000.0, 0.5),    # 1:1 bonus, declared in the vendor file
    "BBB": (500.0, 0.2),     # face-value split 10 -> 2, declared in the silver table
    "CCC": (800.0, 0.6),     # demerger: a 40% gap with no declared event
    "DDD": (5.0, 0.7),       # penny stock: a 30% gap that must be listed, not applied
    "EEE": (300.0, 1.0),     # a bonus is declared but the price never shows it
    "FFF": (400.0, 0.5),     # split that the silver layer has already back-adjusted (wrongly)
    "GGG": (600.0, 0.6),     # demerger hidden inside 8 missing sessions
    "HHH": (200.0, 1.6),     # a genuine 60% rise across 30 missing sessions: must not be applied
    "ZZZ": (100.0, 1.0),     # never in F&O: must be left out
}
MISSING = {"GGG": (30, 38), "HHH": (20, 50)}      # sessions [from, to) with no row in the silver table
SILVER_ADJUSTED = {"FFF"}
HOLES = {"FFF": (12, 13)}                          # one session with no row, inside adjusted history


def _stock_rows(symbol: str, start: float, factor: float) -> pd.DataFrame:
    i = np.arange(len(DAYS))
    close = start * (1 + 0.01 * np.sin(i / 3.0))
    close = np.where(i >= 30, close * factor, close)
    prev = np.roll(close, 1); prev[0] = close[0]
    open_ = np.where(i == 30, prev * factor * 1.002, prev * 1.001)
    if symbol in MISSING:                              # the move happens while rows are missing
        a, b = MISSING[symbol]
        close = start * (1 + 0.01 * np.sin(i / 3.0)) * np.where(i >= a, factor, 1.0)
        prev = np.roll(close, 1); prev[0] = close[0]
        open_ = prev * 1.001
    df = pd.DataFrame({
        "date": DAYS, "symbol": symbol, "open": open_,
        "high": np.maximum(open_, close) * 1.004, "low": np.minimum(open_, close) * 0.996, "close": close,
        "prev_close": prev, "volume": 1_000_000.0, "turnover": 1.0e9, "n_trades": 5000.0,
        "deliv_qty": 400_000.0, "deliv_pct": 40.0,
    })
    if symbol in SILVER_ADJUSTED:
        # what pipeline/silver.py does: scale history by ex-day close / previous close, leave prev_close alone
        k = df.loc[30, "close"] / df.loc[29, "close"]
        df.loc[:29, ["open", "high", "low", "close"]] = (df.loc[:29, ["open", "high", "low", "close"]] * k).round(4)
        df.loc[:29, "volume"] = (df.loc[:29, "volume"] / k).round(0)
    if symbol in MISSING:
        a, b = MISSING[symbol]
        df = df.drop(index=range(a, b)).reset_index(drop=True)
    if symbol in HOLES:
        a, b = HOLES[symbol]
        df = df.drop(index=range(a, b)).reset_index(drop=True)
    return df


def true_raw(symbol: str) -> pd.DataFrame:
    """The prices the exchange actually printed, before any silver back-adjustment."""
    start, factor = STOCKS[symbol]
    i = np.arange(len(DAYS))
    base = start * (1 + 0.01 * np.sin(i / 3.0))
    return pd.DataFrame({"date": DAYS, "close": np.where(i >= 30, base * factor, base)})


def _index_rows(name: str, days, start: float) -> pd.DataFrame:
    i = np.arange(len(days))
    close = start * (1 + 0.005 * np.sin(i / 4.0))
    return pd.DataFrame({"date": days, "index_name": name, "open": close * 0.999, "high": close * 1.003,
                         "low": close * 0.997, "close": close, "volume": 0.0, "turnover": 0.0})


def write_world(root: Path, min_symbols: int = 3) -> None:
    silver = root / "data" / "silver"; silver.mkdir(parents=True)
    (root / "configs").mkdir()
    text = (project_root() / "configs" / "settings.toml").read_text(encoding="utf-8")
    text = text.replace("min_symbols_for_session = 50", f"min_symbols_for_session = {min_symbols}")
    (root / "configs" / "settings.toml").write_text(text, encoding="utf-8")

    pd.concat([_stock_rows(s, p, f) for s, (p, f) in STOCKS.items()], ignore_index=True).to_parquet(silver / "eod_stock.parquet")
    pd.DataFrame({"date": DAYS[0], "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "NIFTY"]}).to_parquet(silver / "eod_deriv_daily.parquet")
    pd.concat([
        _index_rows("CNX Nifty", DAYS[:20], 20000.0),        # old exchange name
        _index_rows("Nifty 50", DAYS[20:], 20100.0),         # new name, same instrument
        _index_rows("Nifty Bank", DAYS, 45000.0),
        _index_rows("India VIX", DAYS, 14.0),
        _index_rows("Nifty IT", DAYS, 30000.0),              # not one of ours
    ], ignore_index=True).to_parquet(silver / "indices.parquet")

    pd.DataFrame({
        "date": [EX, DAYS[10], EX, EX], "symbol": ["BBB", "AAA", "FFF", "FFF"],
        "action_type": ["Face Value Split From Rs.10/- To Rs.2/-", "Dividend - Rs 5 Per Share", "SPLIT", "SPLIT"],
        "amount": [np.nan, 5.0, 2.0, 2.0], "ex_date": [EX, DAYS[10], EX, EX], "record_date": [pd.NaT] * 4,
    }).to_parquet(silver / "corp_actions.parquet")

    yf = root / "data" / "raw" / "corp_actions_yf"; yf.mkdir(parents=True)
    stamp = lambda d: f"{d.date()} 00:00:00+05:30"
    (yf / "yf_AAA.json").write_text(json.dumps([
        {"Date": stamp(DAYS[10]), "Dividends": 5.0, "Stock Splits": 0.0},
        {"Date": stamp(EX), "Dividends": 0.0, "Stock Splits": 2.0},
    ]))
    (yf / "yf_EEE.json").write_text(json.dumps([{"Date": stamp(DAYS[40]), "Dividends": 0.0, "Stock Splits": 2.0}]))

    pd.DataFrame({"date": DAYS, "fii_buy": 1.0, "fii_sell": 1.0, "fii_net": np.arange(len(DAYS), dtype=float),
                  "dii_buy": 1.0, "dii_sell": 1.0, "dii_net": 0.0}).to_parquet(silver / "fii_dii_cash.parquet")
    pd.DataFrame({"date": DAYS, "participant": "FII", "fut_idx_long": 10.0, "fut_idx_short": 5.0}).to_parquet(silver / "participant_oi.parquet")
    pd.DataFrame({"date": [DAYS[45]], "symbol": ["AAA"], "event_type": ["results"], "period": ["Q3"]}).to_parquet(silver / "earnings.parquet")


@pytest.fixture()
def world(tmp_path):
    """Settings rooted at a temp folder holding the synthetic silver layer."""
    write_world(tmp_path)
    return load_settings(root=tmp_path)


@pytest.fixture()
def built(world):
    """The same world with the store tables built."""
    from koscine4.store.build import build_store

    summary = build_store(world, raise_on_error=False)
    return world, summary


# ---- the larger synthetic market used by the tests of phases 2 to 7 (see synthetic.py) -----------------------
@pytest.fixture(scope="session")
def big_root(tmp_path_factory):
    """Built once per test session: silver layer, store, monthly lists, option tables, features, outcomes."""
    import synthetic

    root = tmp_path_factory.mktemp("bigworld")
    synthetic.write_big_world(root)
    synthetic.build_all(root)
    return root


@pytest.fixture(scope="session")
def big(big_root):
    """Settings for the shared market. Tests using this must not write to it."""
    return load_settings(root=big_root)


@pytest.fixture()
def fresh(big_root, tmp_path):
    """A private copy of the market, for tests that write run logs, gate records or the ledger."""
    import shutil

    dest = tmp_path / "world"
    shutil.copytree(big_root, dest)
    return load_settings(root=dest)
