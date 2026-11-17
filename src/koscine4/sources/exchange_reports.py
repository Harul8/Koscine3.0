"""The exchange's daily report bundle, one zip per trading day, from its public archive.

Each bundle holds several small files. Five are parsed here and stored by year under
`<raw>/exchange_reports/<table>/<year>.parquet`:

    corp_actions    ex-date and purpose of each corporate action (file `Bc`)
    board_meetings  meeting date and purpose, with the file date = the day it was announced (`Bm`)
    index_prices    open, high, low, close of Nifty, Bank Nifty and India VIX (index rows of `Pd`)
    other_series    price rows of F&O stocks trading outside the main series, e.g. trade-for-trade (`Pd`)
    share_counts    shares in issue per stock (`MCAP` from 2024, `Ix` and `ffix` in the early years)

    python -m koscine4.sources.exchange_reports --start 2010-01-01 --end 2026-09-01

A date already in the log is skipped, so the command can be stopped and run again.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from koscine4.settings import Settings, load_settings

TABLES = ("corp_actions", "board_meetings", "index_prices", "other_series", "share_counts")
INDEX_ROW_NAMES = {
    "s&p cnx nifty": "NIFTY", "cnx nifty": "NIFTY", "nifty 50": "NIFTY",
    "bank nifty": "BANKNIFTY", "cnx bank": "BANKNIFTY", "nifty bank": "BANKNIFTY",
    "india vix": "INDIAVIX",
}
OTHER_SERIES = ("BE", "BZ")
_BM_LINE = re.compile(
    r"^(?P<name>.+?)\s+(?P<symbol>[A-Z0-9&\-\.\*]+)\s*:\s*(?P<date>\d{1,2}-[A-Za-z]{3}-\d{4})\s*:\s*(?P<purpose>.*)$"
)


def reports_dir(settings: Settings) -> Path:
    return settings.path("raw") / "exchange_reports"


# ---------------------------------------------------------------- parsing (no network)

def _rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if row]


def _num(value: str) -> float | None:
    value = (value or "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def _date(value: str):
    """Dates are dd/mm/yyyy in files up to October 2025 and yyyy-mm-dd after."""
    value = (value or "").strip()
    if not value:
        return pd.NaT
    return pd.to_datetime(value, format="%Y-%m-%d" if value[:4].isdigit() and "-" in value else "%d/%m/%Y", errors="coerce")


def parse_corp_actions(text: str, file_date: pd.Timestamp) -> pd.DataFrame:
    rows = _rows(text)
    out = []
    for r in rows[1:]:
        if len(r) < 10 or not r[1]:
            continue
        out.append({
            "file_date": file_date, "series": r[0], "symbol": r[1].upper(),
            "record_date": _date(r[3]),
            "ex_date": _date(r[6]),
            "purpose": ",".join(r[9:]).strip(),
        })
    return pd.DataFrame(out, columns=["file_date", "series", "symbol", "record_date", "ex_date", "purpose"])


def parse_board_meetings(text: str, file_date: pd.Timestamp) -> pd.DataFrame:
    out: list[dict] = []
    for line in text.splitlines()[1:]:
        m = _BM_LINE.match(line.strip())
        if m:
            out.append({
                "file_date": file_date, "symbol": m["symbol"].upper(),
                "meeting_date": pd.to_datetime(m["date"], format="%d-%b-%Y", errors="coerce"),
                "purpose": m["purpose"].strip()[:300],
            })
        elif out and line.strip():
            out[-1]["purpose"] = (out[-1]["purpose"] + " " + line.strip())[:300]     # a notice can run over lines
    return pd.DataFrame(out, columns=["file_date", "symbol", "meeting_date", "purpose"])


def parse_price_file(text: str, file_date: pd.Timestamp, symbols: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index rows, and stock rows in the series listed in OTHER_SERIES (for `symbols` only, if given)."""
    idx, other = [], []
    for r in _rows(text)[1:]:
        if len(r) < 14:
            continue
        series, symbol, security = r[1], r[2].upper(), r[3]
        base = {"date": file_date, "prev_close": _num(r[4]), "open": _num(r[5]), "high": _num(r[6]),
                "low": _num(r[7]), "close": _num(r[8]), "turnover": _num(r[9]), "volume": _num(r[10])}
        if not symbol and security.lower() in INDEX_ROW_NAMES:
            idx.append({"instrument": INDEX_ROW_NAMES[security.lower()], **base})
        elif series in OTHER_SERIES and symbol and (symbols is None or symbol in symbols):
            other.append({"instrument": symbol, "series": series, **base, "n_trades": _num(r[13])})
    cols = ["date", "instrument", "prev_close", "open", "high", "low", "close", "turnover", "volume"]
    return (pd.DataFrame(idx, columns=cols),
            pd.DataFrame(other, columns=cols[:2] + ["series"] + cols[2:] + ["n_trades"]))


def parse_market_cap(text: str, file_date: pd.Timestamp, symbols: set[str] | None) -> pd.DataFrame:
    """`MCAP` file: Trade Date, Symbol, Series, Name, Category, Last Trade Date, Face Value, Issue Size, Close, Market Cap."""
    out = []
    for r in _rows(text)[1:]:
        if len(r) < 10 or r[2] not in ("EQ",) + OTHER_SERIES:
            continue
        symbol = r[1].upper()
        if symbols is not None and symbol not in symbols:
            continue
        shares = _num(r[7])
        if shares and shares > 0:
            out.append({"date": file_date, "symbol": symbol, "shares": shares, "close": _num(r[8]), "source": "mcap"})
    return pd.DataFrame(out, columns=["date", "symbol", "shares", "close", "source"])


def parse_index_members(text: str, file_date: pd.Timestamp, symbols: set[str] | None, source: str) -> pd.DataFrame:
    """`Ix` and `ffix` files: index constituents with ISSUE_CAP (shares in issue) in column 5."""
    rows = _rows(text)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "shares", "close", "source"])
    header = [h.upper() for h in rows[0]]
    i_cap = header.index("ISSUE_CAP") if "ISSUE_CAP" in header else 4
    i_close = header.index("CLOSE_PRIC") if "CLOSE_PRIC" in header else 5
    seen, out = set(), []
    for r in rows[1:]:
        if len(r) <= max(i_cap, i_close) or not r[1] or r[1].upper() in seen:
            continue
        symbol = r[1].upper()
        shares = _num(r[i_cap])
        if not shares or shares <= 0 or (symbols is not None and symbol not in symbols):
            continue
        seen.add(symbol)
        out.append({"date": file_date, "symbol": symbol, "shares": shares, "close": _num(r[i_close]), "source": source})
    return pd.DataFrame(out, columns=["date", "symbol", "shares", "close", "source"])


def parse_bundle(content: bytes, file_date, symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """All five tables from one day's zip. A member that is absent gives an empty table."""
    file_date = pd.Timestamp(file_date).normalize()
    out: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            name = info.filename.lower().rsplit("/", 1)[-1]
            if not name.endswith((".csv", ".txt")):
                continue
            text = zf.read(info).decode("latin-1")
            if name.startswith("bc"):
                out["corp_actions"].append(parse_corp_actions(text, file_date))
            elif name.startswith("bm"):
                out["board_meetings"].append(parse_board_meetings(text, file_date))
            elif name.startswith("pd"):
                idx, other = parse_price_file(text, file_date, symbols)
                out["index_prices"].append(idx)
                out["other_series"].append(other)
            elif name.startswith("mcap"):
                out["share_counts"].append(parse_market_cap(text, file_date, symbols))
            elif name.startswith("ffix"):
                out["share_counts"].append(parse_index_members(text, file_date, symbols, "ffix"))
            elif name.startswith("ix"):
                out["share_counts"].append(parse_index_members(text, file_date, symbols, "ix"))
    result = {}
    for table, frames in out.items():
        frames = [f for f in frames if len(f)]
        result[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(result["share_counts"]):
        order = {"mcap": 0, "ix": 1, "ffix": 2}
        sc = result["share_counts"].assign(_o=lambda d: d["source"].map(order)).sort_values(["symbol", "_o"])
        result["share_counts"] = sc.drop_duplicates("symbol").drop(columns="_o").reset_index(drop=True)
    return result


# ---------------------------------------------------------------- storage

def read_reports(settings: Settings, table: str) -> pd.DataFrame:
    """Every stored year of one table, or an empty frame if nothing has been downloaded."""
    folder = reports_dir(settings) / table
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _log_path(settings: Settings) -> Path:
    return reports_dir(settings) / "_log.csv"


def read_log(settings: Settings) -> pd.DataFrame:
    path = _log_path(settings)
    if not path.exists():
        return pd.DataFrame(columns=["date", "status"])
    return pd.read_csv(path, parse_dates=["date"])


def _append_year(settings: Settings, year: int, parsed: list[dict[str, pd.DataFrame]], log_rows: list[dict]) -> None:
    root = reports_dir(settings)
    for table in TABLES:
        frames = [p[table] for p in parsed if len(p.get(table, []))]
        if not frames:
            continue
        path = root / table / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            frames.insert(0, pd.read_parquet(path))
        df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        df.to_parquet(path, index=False)
    log = pd.concat([read_log(settings), pd.DataFrame(log_rows)], ignore_index=True)
    log = log.drop_duplicates("date", keep="last").sort_values("date")
    _log_path(settings).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(_log_path(settings), index=False)


# ---------------------------------------------------------------- download

def _fetch(url: str, headers: dict, timeout: float, retries: int) -> tuple[str, bytes]:
    import requests

    last = "error"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content[:2] == b"PK":
                return "ok", r.content
            if r.status_code == 404:
                return "missing", b""
            last = f"http_{r.status_code}"
        except Exception as exc:  # network errors are retried, then recorded
            last = f"error_{type(exc).__name__}"
        time.sleep(1.5 * (attempt + 1))
    return last, b""


def download(settings: Settings | None = None, start=None, end=None, dates: list | None = None,
             workers: int | None = None, retry_failed: bool = True, progress: bool = True, force: bool = False) -> dict:
    """Download and parse the bundle for each date not yet in the log. Returns counts by status."""
    settings = settings or load_settings()
    cfg = settings.section("sources")
    if dates is None:
        dates = list(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end)))
    dates = sorted({pd.Timestamp(d).normalize() for d in dates})
    log = read_log(settings)
    done = set(log["date"]) if not retry_failed else set(log[log["status"].isin(["ok", "missing"])]["date"])
    todo = dates if force else [d for d in dates if d not in done]
    symbols = fo_symbols(settings)
    headers = {"User-Agent": str(cfg["user_agent"]), "Accept": "*/*"}
    template = str(cfg["report_bundle_url"])
    counts: dict[str, int] = {}

    def one(d: pd.Timestamp):
        status, content = _fetch(template.format(ddmmyy=d.strftime("%d%m%y")), headers,
                                 float(cfg["timeout_seconds"]), int(cfg["retries"]))
        parsed = {}
        if status == "ok":
            try:
                parsed = parse_bundle(content, d, symbols)
            except Exception as exc:
                status = f"parse_{type(exc).__name__}"
        return d, status, parsed

    for year in sorted({d.year for d in todo}):
        batch = [d for d in todo if d.year == year]
        with ThreadPoolExecutor(max_workers=int(workers or cfg["workers"])) as pool:
            results = list(pool.map(one, batch))
        _append_year(settings, year, [p for _, s, p in results if s == "ok"],
                     [{"date": d, "status": s} for d, s, _ in results])
        for _, s, _ in results:
            counts[s] = counts.get(s, 0) + 1
        if progress:
            print(f"{year}: {len(batch)} dates, " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    return counts


def download_symbol_changes(settings: Settings | None = None) -> int:
    """The exchange's list of symbol changes (old symbol, new symbol, effective date), kept as received."""
    settings = settings or load_settings()
    cfg = settings.section("sources")
    import requests

    r = requests.get(str(cfg["symbol_changes_url"]), headers={"User-Agent": str(cfg["user_agent"]), "Accept": "*/*"},
                     timeout=float(cfg["timeout_seconds"]))
    r.raise_for_status()
    path = reports_dir(settings) / "symbol_changes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return len(read_symbol_changes(settings))


def read_symbol_changes(settings: Settings) -> pd.DataFrame:
    """Columns old, new, date. Empty when the file has not been downloaded."""
    path = reports_dir(settings) / "symbol_changes.csv"
    cols = ["old", "new", "date"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    rows = _rows(path.read_text(encoding="latin-1"))
    out = [{"old": r[-3].upper(), "new": r[-2].upper(), "date": pd.to_datetime(r[-1], format="%d-%b-%Y", errors="coerce")}
           for r in rows[1:] if len(r) >= 4]
    df = pd.DataFrame(out, columns=cols)
    return df[df["date"].notna() & (df["old"] != df["new"])].drop_duplicates().sort_values("date").reset_index(drop=True)


def fo_symbols(settings: Settings) -> set[str] | None:
    """Every symbol that ever had derivatives, or None when the silver table is absent (keep all)."""
    path = settings.path("silver") / "eod_deriv_daily.parquet"
    if not path.exists():
        return None
    return set(pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str).str.upper().unique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    print("symbol changes:", download_symbol_changes())
    print(download(start=args.start, end=args.end, workers=args.workers))


if __name__ == "__main__":
    main()
