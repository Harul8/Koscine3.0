"""
FII/DII silver builder — parses NSDL monthly archive files.

NSDL publishes monthly "Archive_Data_*.xls" files containing daily FPI cash
and derivative trades.  Despite the .xls extension they are HTML; we use
pandas.read_html with header=[0, 1] to flatten the multi-row header.

NSDL only publishes FII/FPI data — DII columns are written as NaN.  To get
DII, either:
  (a) Add a CDSL parser alongside this one and merge by date.
  (b) Use NSE's combined fii_dii_trade JSON API (older infrastructure exists
      in pipeline/fetch.py — see _url_fii_dii).
Both produce the same silver schema; this module owns the NSDL path only.

How to use
----------
1. Download monthly archives from
       https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx
   The default download name is `Archive_Data_<m>_<d>_<yyyy>_<hh>_<mm>_<ss>.xls`.
   You can rename them however you like — the parser accepts any *.xls or
   *.html file in the directory.
2. Drop them all in:  <DATA_ROOT>/raw/nsdl_fpi/
3. Run:  python -m pipeline.fetch_fiidii
   Writes:  <DATA_ROOT>/silver/fii_dii_cash.parquet

Schema written
--------------
date (datetime), fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
                 [all crore INR; DII cols all NaN if only NSDL is used]
"""
from __future__ import annotations
import datetime as dt
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from .config import RAW_DIR, SILVER_TABLES

# Folder where you drop downloaded NSDL Archive_Data_*.xls files.
# Defaults to RAW_DIR/FII_cash to match the directory you've set up; the older
# 'nsdl_fpi' name is also scanned for back-compat.
NSDL_RAW_DIRS = [RAW_DIR / "FII_cash", RAW_DIR / "nsdl_fpi"]
NSDL_ARCHIVE_URL = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"


# ─────────────────────────────────────────────────────────────────────────
# NSDL Archive_Data_*.xls parser
# ─────────────────────────────────────────────────────────────────────────
def _to_num(v) -> float:
    """Parse NSDL crore amounts: '6188.90', '(0.80)' -> -0.80, blank -> NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().replace(",", "")
    if not s or s in {"-", "nan", "NaN"}:
        return np.nan
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header — keep the most specific level."""
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for c in df.columns:
            # Walk from the most-specific level upward, take the first
            # non-"Unnamed" string we find
            chosen = None
            for level in reversed(c):
                s = str(level)
                if s and not s.startswith("Unnamed"):
                    chosen = s
                    break
            cols.append(chosen if chosen else str(c[-1]))
        df = df.copy()
        df.columns = cols
    return df


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_col(df: pd.DataFrame, *needles: str) -> str | None:
    """Find a column whose normalised name contains any of `needles`."""
    norm_to_col = {_normalise(c): c for c in df.columns}
    for needle in needles:
        n = _normalise(needle)
        for k, v in norm_to_col.items():
            if n in k:
                return v
    return None


def _find_cash_table(fp: Path) -> pd.DataFrame:
    tables = pd.read_html(fp)
    if not tables:
        raise ValueError(f"No tables found in {fp.name}")
    errors = []
    for table in tables:
        raw = _flatten_cols(table)
        c_date = _pick_col(raw, "ReportingDate", "Date")
        c_de   = _pick_col(raw, "DebtEquity")
        c_rt   = _pick_col(raw, "InvestmentRoute")
        c_buy  = _pick_col(raw, "GrossPurchases")
        c_sell = _pick_col(raw, "GrossSales")
        c_net  = _pick_col(raw, "NetInvestmentRsCrore", "NetInvestmentRsCrores", "NetInvestment")
        missing = [n for n, v in [("date", c_date), ("debt/equity", c_de),
                                   ("route", c_rt), ("buy", c_buy), ("sell", c_sell),
                                   ("net", c_net)] if v is None]
        if not missing:
            return raw
        errors.append(missing)
    raise ValueError(f"{fp.name}: no FII cash table found; missing candidates: {errors[:3]}")


def parse_nsdl_archive(fp: Path) -> pd.DataFrame:
    """Parse one NSDL archive file -> DataFrame[date, fii_buy, fii_sell, fii_net]."""
    raw = _find_cash_table(fp)

    # Find the columns we care about (NSDL has fancy headers; do a fuzzy match)
    c_date = _pick_col(raw, "ReportingDate", "Date")
    c_de   = _pick_col(raw, "DebtEquity")
    c_rt   = _pick_col(raw, "InvestmentRoute")
    c_buy  = _pick_col(raw, "GrossPurchases")
    c_sell = _pick_col(raw, "GrossSales")
    c_net  = _pick_col(raw, "NetInvestmentRsCrore", "NetInvestmentRsCrores", "NetInvestment")
    missing = [n for n, v in [("date", c_date), ("debt/equity", c_de),
                               ("route", c_rt), ("buy", c_buy), ("sell", c_sell),
                               ("net", c_net)] if v is None]
    if missing:
        raise ValueError(f"{fp.name}: missing columns {missing}; have: {list(raw.columns)}")

    # NSDL emits 7 rows per date: Equity-SE / Equity-Primary / Equity-Subtotal
    # / Debt-SE / Debt-Primary / Debt-Subtotal / Debt-Total.  We want the
    # Equity Sub-total row — that's the daily FPI net for equities including
    # both stock exchange and primary market.
    df = raw[(raw[c_de].astype(str).str.strip() == "Equity") &
             (raw[c_rt].astype(str).str.strip() == "Sub-total")].copy()

    out = pd.DataFrame({
        "date":     pd.to_datetime(df[c_date], format="%d-%b-%Y", errors="coerce"),
        "fii_buy":  df[c_buy].apply(_to_num),
        "fii_sell": df[c_sell].apply(_to_num),
        "fii_net":  df[c_net].apply(_to_num),
    }).dropna(subset=["date"]).reset_index(drop=True)
    return out


def _make_nsdl_session():
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
    except Exception:
        import requests
        session = requests.Session()
    session.verify = False
    return session


def _post_payload(html: str, as_of: dt.date) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    payload = {}
    for inp in soup.find_all("input"):
        name = inp.get("name")
        if name:
            payload[name] = inp.get("value", "")
    date_text = as_of.strftime("%d-%b-%Y")
    payload.update({
        "__EVENTTARGET": "btnSubmit1",
        "__EVENTARGUMENT": "",
        "txtDate": date_text,
        "hdnDate": date_text,
        "hdnFlag": "",
        "HdnValexceldata": "",
    })
    return payload


def fetch_nsdl_archive(as_of: dt.date, force: bool = False) -> Path:
    """Fetch NSDL Archive.aspx cash/FPI report up to as_of and save as raw/FII_cash."""
    out_dir = RAW_DIR / "FII_cash"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"Archive_Data_auto_{as_of.strftime('%Y%m%d')}.xls"
    if dest.exists() and dest.stat().st_size > 1000 and not force:
        print(f"[fii_dii] cached {dest.name}")
        return dest

    headers = {
        "Referer": NSDL_ARCHIVE_URL,
        "Origin": "https://www.fpi.nsdl.co.in",
    }
    last_error = None
    for attempt in range(1, 5):
        try:
            session = _make_nsdl_session()
            page = session.get(NSDL_ARCHIVE_URL, timeout=30)
            page.raise_for_status()
            response = session.post(
                NSDL_ARCHIVE_URL,
                data=_post_payload(page.text, as_of),
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            print(f"[fii_dii] NSDL fetch retry {attempt}/4 for {as_of}: {exc}")
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"NSDL archive fetch failed for {as_of}: {last_error}") from last_error
    content = response.content
    text = content.decode("utf-8", errors="ignore")
    if "Daily Trends in FPI Investments" not in text or "Gross Purchases" not in text:
        raise RuntimeError(f"NSDL archive response did not contain FII cash table for {as_of}")
    dest.write_bytes(content)
    print(f"[fii_dii] fetched {as_of} -> {dest}")
    return dest


def _month_end_or_end(cur: dt.date, end: dt.date) -> dt.date:
    next_month = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    month_end = next_month - dt.timedelta(days=1)
    return min(month_end, end)


def fetch_nsdl_archives(start: dt.date, end: dt.date, force: bool = False) -> list[Path]:
    """Fetch one NSDL archive per month in [start, end]."""
    cur = dt.date(start.year, start.month, 1)
    paths = []
    while cur <= end:
        as_of = _month_end_or_end(cur, end)
        paths.append(fetch_nsdl_archive(as_of, force=force))
        cur = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return paths


# ─────────────────────────────────────────────────────────────────────────
# Build silver table
# ─────────────────────────────────────────────────────────────────────────
def build_fii_dii_cash(extra_paths: list[Path] | None = None) -> pd.DataFrame:
    """Aggregate all NSDL archives under NSDL_RAW_DIRS (+ any extra_paths)
    and write silver/fii_dii_cash.parquet.  DII cols are NaN."""
    files: list[Path] = []
    for d in NSDL_RAW_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.xls")))
            files.extend(sorted(d.glob("*.html")))
    if extra_paths:
        files.extend(extra_paths)

    if not files:
        print(f"[fii_dii] no archives in any of {NSDL_RAW_DIRS}")
        print(f"  drop NSDL Archive_Data_*.xls files there and re-run.")
        return pd.DataFrame()

    print(f"[fii_dii] parsing {len(files)} NSDL archive(s)...")
    chunks: list[pd.DataFrame] = []
    n_ok = 0
    for fp in files:
        try:
            df = parse_nsdl_archive(fp)
            chunks.append(df)
            n_ok += 1
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
    print(f"  parsed OK: {n_ok}/{len(files)}")

    if not chunks:
        return pd.DataFrame()

    combined = (pd.concat(chunks, ignore_index=True)
                  .drop_duplicates(subset=["date"])
                  .sort_values("date")
                  .reset_index(drop=True))

    # DII columns — schema-compat placeholder.  Fill from CDSL later if available.
    combined["dii_buy"]  = np.nan
    combined["dii_sell"] = np.nan
    combined["dii_net"]  = np.nan
    combined = combined[["date", "fii_buy", "fii_sell", "fii_net",
                         "dii_buy", "dii_sell", "dii_net"]]

    silver_path = SILVER_TABLES["fii_dii_cash"]
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(silver_path)

    print(f"\n[fii_dii] saved {len(combined):,} rows -> {silver_path}")
    print(f"  date range: {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print(f"  fii_net  mean={combined['fii_net'].mean():+.1f}  "
          f"std={combined['fii_net'].std():.1f}  "
          f"(in Rs Crore)")
    print(f"  positive flow days: {(combined['fii_net'] > 0).sum()} / {len(combined)}")
    return combined


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    force = "--force" in args
    if "--fetch" in args:
        idx = args.index("--fetch")
        start = dt.date.fromisoformat(args[idx + 1])
        end = dt.date.fromisoformat(args[idx + 2]) if len(args) > idx + 2 and not args[idx + 2].startswith("--") else start
        fetch_nsdl_archives(start, end, force=force)
        build_fii_dii_cash()
        return
    extra = [Path(a) for a in args if Path(a).exists()]
    build_fii_dii_cash(extra_paths=extra if extra else None)


if __name__ == "__main__":
    main()
