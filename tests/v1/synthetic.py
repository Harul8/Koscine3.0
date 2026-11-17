"""A larger synthetic market for the tests of phases 2 to 7: 420 sessions, nine stocks, two indices,
option contracts priced by formula, flows, events and share counts. Everything is generated
from a fixed seed, so every test sees the same market.

What is planted, and which test relies on it:
    AAA   traded as AAAOLD for its first 100 sessions (symbol change)
    EEE   has share counts only late in the history (market value is back-cast before that)
    FFF   2-for-1 split on session 200, declared in the exchange file
    GGG   lists on session 250 (a new listing must not appear early)
    HHH   its options stop trading on session 300 (a faded stock must drop out)
    NOP   the largest company, with futures but no options (must never qualify)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from koscine4.settings import load_settings, project_root
from koscine4.store.surface import black_price

N = 420
DAYS = pd.bdate_range("2022-01-03", periods=N)
RENAME_AT, SPLIT_AT, LIST_AT, FADE_AT = 100, 200, 250, 300
TOP_N = 3

#        start price, shares, first session
STOCKS = {
    "AAA": (1000.0, 1.0e9, 0),
    "BBB": (500.0, 3.0e9, 0),
    "CCC": (2500.0, 2.0e8, 0),
    "DDD": (150.0, 4.0e9, 0),
    "EEE": (300.0, 1.0e9, 0),
    "FFF": (800.0, 5.0e8, 0),
    "GGG": (600.0, 5.0e9, LIST_AT),
    "HHH": (900.0, 2.0e9, 0),
    "NOP": (2000.0, 5.0e9, 0),
}
INDICES = {"NIFTY": ("Nifty 50", 18000.0, 0.007), "BANKNIFTY": ("Nifty Bank", 40000.0, 0.009)}
OPTION_SPECS = {  # instrument -> (strike low, strike high, step, lot, implied volatility)
    "NIFTY": (13000.0, 24000.0, 100.0, 50, 0.14),
    "AAA": (500.0, 1700.0, 20.0, 500, 0.26),
    "BBB": (250.0, 850.0, 10.0, 1000, 0.26),
}
RESULT_SESSIONS = (60, 125, 190, 255, 320, 385)
ANNOUNCE_AHEAD = 10


def _path(seed: int, start: float, vol: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0002, vol, n)))
    prev = np.concatenate([[start], close[:-1]])
    open_ = prev * np.exp(rng.normal(0.0, vol / 3.0, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3.0, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3.0, n)))
    return open_, high, low, close


def symbol_then(name: str, i: int) -> str:
    return "AAAOLD" if name == "AAA" and i < RENAME_AT else name


def stock_frame(name: str) -> pd.DataFrame:
    """Raw prices as the exchange printed them (FFF halves on its split day)."""
    start, _, first = STOCKS[name]
    o, h, l, c = _path(sum(map(ord, name)), start, 0.012, N)
    i = np.arange(N)
    if name == "FFF":
        k = np.where(i >= SPLIT_AT, 0.5, 1.0)
        prev_raw = np.concatenate([[start], c[:-1]]) * np.concatenate([[1.0], k[:-1]])
        o, h, l, c = o * k, h * k, l * k, c * k
    else:
        prev_raw = np.concatenate([[start], c[:-1]])
    df = pd.DataFrame({"date": DAYS, "symbol": [symbol_then(name, j) for j in i], "open": o, "high": h, "low": l, "close": c,
                       "prev_close": prev_raw, "volume": 1.0e6, "turnover": 1.0e9, "n_trades": 5000.0,
                       "deliv_qty": 4.0e5, "deliv_pct": 40.0})
    return df[i >= first].reset_index(drop=True)


def index_frame(name: str) -> pd.DataFrame:
    label, start, vol = INDICES[name]
    o, h, l, c = _path(sum(map(ord, name)), start, vol, N)
    return pd.DataFrame({"date": DAYS, "index_name": label, "open": o, "high": h, "low": l, "close": c, "volume": 0.0, "turnover": 0.0})


def vix_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    v = 15.0 + np.cumsum(rng.normal(0, 0.3, N)) * 0.5
    v = np.clip(v, 9.0, 40.0)
    return pd.DataFrame({"date": DAYS, "index_name": "India VIX", "open": v, "high": v * 1.02, "low": v * 0.98, "close": v, "volume": 0.0, "turnover": 0.0})


def expiries() -> pd.DatetimeIndex:
    """Last Thursday of each month."""
    months = pd.period_range(DAYS[0], DAYS[-1] + pd.Timedelta(days=100), freq="M")
    out = []
    for m in months:
        d = m.end_time.normalize()
        while d.weekday() != 3:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return pd.DatetimeIndex(out)


def contracts_frame(name: str, prices: pd.DataFrame) -> pd.DataFrame:
    """Option and futures rows for one instrument, priced by Black's formula at the day's open and close."""
    lo, hi, step, lot, vol = OPTION_SPECS[name]
    strikes = np.arange(lo, hi + step / 2, step)
    exps = expiries()
    kind_opt, kind_fut = ("OPTIDX", "FUTIDX") if name in INDICES else ("OPTSTK", "FUTSTK")
    parts = []
    for i, (d, o, c) in enumerate(zip(prices["date"], prices["open"], prices["close"])):
        sym = symbol_then(name, DAYS.get_loc(d))
        live = exps[exps >= d][:3]
        for e in live:
            t = max((e - d).days, 0.5) / 365.0
            for is_call, ot in ((True, "CE"), (False, "PE")):
                close = np.maximum(np.round(black_price(c, strikes, t, vol, is_call) / 0.05) * 0.05, 0.05)
                open_ = np.maximum(np.round(black_price(o, strikes, t, vol, is_call) / 0.05) * 0.05, 0.05)
                traded = np.abs(strikes / c - 1.0) <= 0.12
                parts.append(pd.DataFrame({
                    "date": d, "symbol": sym, "instrument": kind_opt, "expiry": e, "strike": strikes, "opt_type": ot,
                    "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close, "settle": close,
                    "contracts": np.where(traded, 500, 0), "open_int": (lot * (1 + np.arange(len(strikes)) % 7)).astype(np.int64),
                }))
            parts.append(pd.DataFrame({"date": [d], "symbol": [sym], "instrument": [kind_fut], "expiry": [e], "strike": [np.nan],
                                       "opt_type": ["nan"], "open": [o], "high": [max(o, c)], "low": [min(o, c)], "close": [c],
                                       "settle": [c], "contracts": [1000], "open_int": [np.int64(lot * 200)]}))
    out = pd.concat(parts, ignore_index=True)
    out["contracts"] = out["contracts"].astype(np.int64)
    return out


def write_big_world(root: Path) -> None:
    root = Path(root)
    silver = root / "data" / "silver"
    silver.mkdir(parents=True)
    (root / "configs").mkdir()
    text = (project_root() / "configs" / "settings.toml").read_text(encoding="utf-8")
    for old, new in (("min_symbols_for_session = 50", "min_symbols_for_session = 3"), ("top_n = 30 ", f"top_n = {TOP_N} "),
                     ("sealed_months = 12", "sealed_months = 3"), ('default_start = "2012-01-01"', 'default_start = "2022-06-01"')):
        assert old in text, old
        text = text.replace(old, new)
    (root / "configs" / "settings.toml").write_text(text, encoding="utf-8")
    for sub in ("strategies", "models", "searches"):
        (root / "configs" / sub).mkdir()

    stocks = {name: stock_frame(name) for name in STOCKS}
    pd.concat(stocks.values(), ignore_index=True).to_parquet(silver / "eod_stock.parquet")
    idx = {name: index_frame(name) for name in INDICES}
    pd.concat(list(idx.values()) + [vix_frame()], ignore_index=True).to_parquet(silver / "indices.parquet")

    # per-day derivative totals: who has options trading, and when
    rows = []
    for name, df in {**stocks, **idx}.items():
        pos = DAYS.get_indexer(df["date"])
        has_opt = np.ones(len(df), dtype=bool)
        if name == "NOP":
            has_opt[:] = False
        if name == "HHH":
            has_opt = pos < FADE_AT
        wave = 1 + 0.2 * np.sin(pos / 9.0)
        rows.append(pd.DataFrame({
            "date": df["date"].to_numpy(), "symbol": [symbol_then(name, j) for j in pos], "fut_close": df["close"].to_numpy() * 1.001,
            "fut_oi": 1.0e6 * wave, "opt_call_oi": np.where(has_opt, 5.0e5 * wave, 0.0), "opt_put_oi": np.where(has_opt, 4.0e5 / wave, 0.0),
            "opt_call_vol": np.where(has_opt, 2000.0 * wave, 0.0), "opt_put_vol": np.where(has_opt, 1500.0 / wave, 0.0)}))
    pd.concat(rows, ignore_index=True).to_parquet(silver / "eod_deriv_daily.parquet")

    # option contracts for the instruments the harness trades
    cframes = [contracts_frame("NIFTY", idx["NIFTY"]), contracts_frame("AAA", stocks["AAA"]), contracts_frame("BBB", stocks["BBB"])]
    allc = pd.concat(cframes, ignore_index=True)
    for year, g in allc.groupby(allc["date"].dt.year):
        folder = silver / "eod_deriv_contracts" / f"year={year}"
        folder.mkdir(parents=True)
        g.to_parquet(folder / "data.parquet", index=False)

    rng = np.random.default_rng(11)
    pd.DataFrame({"date": DAYS, "fii_buy": 1.0, "fii_sell": 1.0, "fii_net": rng.normal(0, 1000, N), "dii_buy": 1.0, "dii_sell": 1.0,
                  "dii_net": rng.normal(0, 800, N)}).to_parquet(silver / "fii_dii_cash.parquet")
    part = []
    for who in ("FII", "Client"):
        long_ = 1.0e5 * (1 + 0.3 * np.sin(np.arange(N) / 15.0 + (who == "Client")))
        part.append(pd.DataFrame({"date": DAYS, "participant": who, "fut_idx_long": long_, "fut_idx_short": 1.5e5 - long_ * 0.5,
                                  "opt_idx_call_long": 4.0e5, "opt_idx_put_long": 5.0e5 + 1.0e4 * np.cos(np.arange(N) / 7.0),
                                  "opt_idx_call_short": 3.0e5, "opt_idx_put_short": 2.0e5}))
    pd.concat(part, ignore_index=True).to_parquet(silver / "participant_oi.parquet")
    pd.DataFrame({"date": [DAYS[150]], "symbol": ["CCC"], "event_type": ["Results"], "period": ["Financial results"]}).to_parquet(silver / "earnings.parquet")
    pd.DataFrame({"date": [DAYS[5]], "symbol": ["CCC"], "action_type": ["Dividend - Rs 5 Per Share"], "amount": [5.0],
                  "ex_date": [DAYS[5]], "record_date": [pd.NaT]}).to_parquet(silver / "corp_actions.parquet")

    # what koscine4.sources.exchange_reports would have downloaded
    rep = root / "data" / "raw" / "exchange_reports"
    for t in ("corp_actions", "board_meetings", "share_counts"):
        (rep / t).mkdir(parents=True)
    pd.DataFrame({"file_date": [DAYS[SPLIT_AT - 3], DAYS[40]], "series": ["EQ", "EQ"], "symbol": ["FFF", "DDD"],
                  "record_date": [pd.NaT, pd.NaT], "ex_date": [DAYS[SPLIT_AT], DAYS[40]],
                  "purpose": ["FV SPLIT RS.10 TO RS.5", "DIVIDEND RS 2 PER SHARE"]}).to_parquet(rep / "corp_actions" / "2022.parquet")
    bm = [{"file_date": DAYS[s - ANNOUNCE_AHEAD], "symbol": symbol_then(name, s - ANNOUNCE_AHEAD), "meeting_date": DAYS[s],
           "purpose": "Financial Results To consider and approve the financial results"} for name in ("AAA", "BBB") for s in RESULT_SESSIONS]
    pd.DataFrame(bm).to_parquet(rep / "board_meetings" / "2022.parquet")
    counts = []
    for name, (_, shares, first) in STOCKS.items():
        for i in list(range(0, 30)) + list(range(320, N)):
            if i < first or (name == "EEE" and i < 320):
                continue
            n = shares * (2.0 if name == "FFF" and i >= SPLIT_AT else 1.0)
            counts.append({"date": DAYS[i], "symbol": symbol_then(name, i), "shares": n, "close": 0.0, "source": "ix" if i < 30 else "mcap"})
    pd.DataFrame(counts).to_parquet(rep / "share_counts" / "2022.parquet")
    (rep / "symbol_changes.csv").write_text(
        f"NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\nAaa Limited,AAAOLD,AAA,{DAYS[RENAME_AT].strftime('%d-%b-%Y').upper()}\n", encoding="utf-8")


def build_all(root: Path):
    """Build every data table the later phases need, in the order the owner would."""
    from koscine4.features.build import build_feature_table
    from koscine4.outcomes.build import build_moves
    from koscine4.store.build import build_store
    from koscine4.store.surface import build_option_tables
    from koscine4.universe.build import build_universe

    settings = load_settings(root=root)
    summary = {"store": build_store(settings, lots=True, raise_on_error=False)}
    summary["universe"] = build_universe(settings)
    summary["options"] = build_option_tables(settings, quote_instruments=["NIFTY", "AAA", "BBB"])
    summary["features_evening"] = build_feature_table(settings, "evening")
    summary["features_1430"] = build_feature_table(settings, "1430")
    summary["outcomes"] = build_moves(settings)
    return settings, summary
