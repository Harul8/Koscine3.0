"""
Unified NSE + Yahoo Finance data fetcher.

Two public entry points:
    backfill(start, end)   â€” download all feeds for a date range
    fetch_day(date)        â€” download all feeds for one trading day

NSE daily feeds (one file per trading day):
    cash_bhavcopy     OHLCV + delivery pct
    derivatives       F&O OI / strikes (UDIFF 2024+ or legacy zip)
    participant_oi    FII/DII/Client/Pro OI
    indices           All indices + India VIX

NSE bulk feeds (month-chunked JSON, fetched on backfill or monthly):
    fii_dii_cash      Daily cash net buy/sell
    corp_actions      Dividends, splits, bonus, mergers
    earnings_calendar Results announcement dates

NSE one-off:
    lot_size          F&O lot sizes (quarterly snapshot)

Yahoo Finance (quarterly, run on backfill only):
    fundamentals      Revenue, margins, ROE, D/E
    earnings_eps      Quarterly EPS history
    corp_actions_yf   Dividends + splits (supplement to NSE)

Usage:
    python -m pipeline.fetch                          # fetch today only
    python -m pipeline.fetch 2026-01-01               # backfill from date to today
    python -m pipeline.fetch 2026-01-01 2026-04-24    # backfill date range
"""
import sys
import json
import time
import random
import datetime as dt
from pathlib import Path
from typing import Optional

import requests
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

from .config import RAW_DIR, GOLD_DIR, SILVER_TABLES, UDIFF_CUTOVER, CASH_NEW_FROM, YF_DAILY_WATCHLIST

YF_RETRY_TRACKER = GOLD_DIR / "yf_eps_retry.json"
_RETRY_MAX_DAYS  = 30   # give up after this many calendar days

SLEEP_SEC   = 0.8
MAX_RETRIES = 3
TIMEOUT     = 30



def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

def _make_yf_session():
    """Create a Yahoo Finance session that works behind local SSL interception."""
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        session.verify = False
        return session
    except Exception:
        return None

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# NSE session
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.verify = False
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=TIMEOUT)
        time.sleep(0.3)
        s.get("https://www.nseindia.com/all-reports", timeout=TIMEOUT)
    except Exception as e:
        print(f"  [warn] NSE cookie handshake: {e}")
    return s


# ---------------------------------------------------------------------------
# core download helpers
# ---------------------------------------------------------------------------

def _download(session: requests.Session, url: str, dest: Path) -> str:
    """Returns 'ok' | 'skip' | '404' | 'fail'."""
    if dest.exists() and dest.stat().st_size > 200:
        return "skip"
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200 and len(r.content) > 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                return "ok"
            if r.status_code == 404:
                return "404"
            time.sleep(2 + attempt * 2)
        except Exception:
            time.sleep(2 + attempt * 2)
    return "fail"


def _download_json(session: requests.Session, url: str, dest: Path,
                   warmup_url: Optional[str] = None) -> str:
    """Download NSE JSON API endpoint."""
    if dest.exists() and dest.stat().st_size > 100:
        return "skip"
    hdrs = {**HEADERS, "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest"}
    if warmup_url:
        try:
            session.get("https://www.nseindia.com", timeout=TIMEOUT)
            time.sleep(0.3)
            session.get(warmup_url, timeout=TIMEOUT)
            time.sleep(0.3)
        except Exception:
            pass
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT, headers=hdrs)
            if r.status_code == 200 and len(r.content) > 40:
                try:
                    payload = r.json()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(payload), encoding="utf-8")
                    return "ok"
                except Exception:
                    time.sleep(2 + attempt * 2)
            elif r.status_code == 404:
                return "404"
            else:
                time.sleep(2 + attempt * 2)
        except Exception:
            time.sleep(2 + attempt * 2)
    return "fail"


# ---------------------------------------------------------------------------
# URL builders â€” NSE daily feeds
# ---------------------------------------------------------------------------

def _url_cash_bhav(d: dt.date) -> str:
    return (f"https://archives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv")

def _url_cash_bhav_legacy_ohlc(d: dt.date):
    fn  = f"cm{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
    url = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
           f"{d.year}/{d.strftime('%b').upper()}/{fn}")
    return url, fn

def _url_cash_bhav_legacy_delivery(d: dt.date):
    fn  = f"MTO_{d.strftime('%d%m%Y')}.DAT"
    url = f"https://archives.nseindia.com/archives/equities/mto/{fn}"
    return url, fn

def _url_participant_oi(d: dt.date) -> str:
    return (f"https://archives.nseindia.com/content/nsccl/"
            f"fao_participant_oi_{d.strftime('%d%m%Y')}.csv")

def _url_indices(d: dt.date) -> str:
    return (f"https://archives.nseindia.com/content/indices/"
            f"ind_close_all_{d.strftime('%d%m%Y')}.csv")

def _url_derivatives(d: dt.date):
    """Returns (url, filename, format_kind)."""
    if d >= UDIFF_CUTOVER:
        fn  = f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        url = f"https://archives.nseindia.com/content/fo/{fn}"
        return url, fn, "udiff"
    fn  = f"fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
    url = (f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
           f"{d.year}/{d.strftime('%b').upper()}/{fn}")
    return url, fn, "legacy"


# ---------------------------------------------------------------------------
# URL builders â€” NSE bulk (JSON) feeds
# ---------------------------------------------------------------------------

def _url_block_deals(d: dt.date) -> str:
    return "https://www.nseindia.com/api/block-deal"


def _url_block_deals_warmup() -> str:
    return "https://www.nseindia.com/market-data/block-deal"


def _url_fii_dii(s: dt.date, e: dt.date) -> str:
    return (f"https://www.nseindia.com/api/historical/fiidiiTradeReact"
            f"?from={s.strftime('%d-%m-%Y')}&to={e.strftime('%d-%m-%Y')}")

def _url_corp_actions(s: dt.date, e: dt.date) -> str:
    return (f"https://www.nseindia.com/api/corporates-corporateActions"
            f"?index=equities&from_date={s.strftime('%d-%m-%Y')}"
            f"&to_date={e.strftime('%d-%m-%Y')}")

def _url_earnings(s: dt.date, e: dt.date) -> str:
    return (f"https://www.nseindia.com/api/event-calendar"
            f"?from_date={s.strftime('%d-%m-%Y')}&to_date={e.strftime('%d-%m-%Y')}")

_BULK_SPECS = [
    ("fii_dii_cash",    _url_fii_dii,        "fii_dii_cash",
     "https://www.nseindia.com/reports/fii-dii"),
    ("corp_actions",    _url_corp_actions,   "corporate_actions",
     "https://www.nseindia.com/companies-listing/corporate-filings-actions"),
    ("earnings",        _url_earnings,       "earnings_calendar",
     "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar"),
]


# ---------------------------------------------------------------------------
# URL builder â€” lot size
# ---------------------------------------------------------------------------

_LOT_URLS = [
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
]


# ---------------------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------------------

def _weekdays(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)

def _month_chunks(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        yield max(start, cur), min(end, nxt - dt.timedelta(days=1))
        cur = nxt


# ---------------------------------------------------------------------------
# NSE daily feed download for one date
# ---------------------------------------------------------------------------

def _nse_day(session: requests.Session, d: dt.date, counts: dict):
    ystr = str(d.year)

    # cash bhavcopy
    if d >= CASH_NEW_FROM:
        url  = _url_cash_bhav(d)
        dest = RAW_DIR / "cash_bhavcopy" / ystr / url.rsplit("/", 1)[-1]
        counts[_download(session, url, dest)] += 1
        time.sleep(SLEEP_SEC + random.random() * 0.4)
    else:
        for url, fn in [_url_cash_bhav_legacy_ohlc(d),
                        _url_cash_bhav_legacy_delivery(d)]:
            sub = ("cash_bhavcopy_legacy_ohlc"
                   if "bhav" in fn.lower() else "cash_bhavcopy_legacy_delivery")
            dest = RAW_DIR / sub / ystr / fn
            counts[_download(session, url, dest)] += 1
            time.sleep(SLEEP_SEC + random.random() * 0.4)

    # participant OI
    url  = _url_participant_oi(d)
    dest = RAW_DIR / "participant_oi" / ystr / url.rsplit("/", 1)[-1]
    counts[_download(session, url, dest)] += 1
    time.sleep(SLEEP_SEC + random.random() * 0.4)

    # indices
    url  = _url_indices(d)
    dest = RAW_DIR / "indices" / ystr / url.rsplit("/", 1)[-1]
    counts[_download(session, url, dest)] += 1
    time.sleep(SLEEP_SEC + random.random() * 0.4)

    # derivatives
    url, fn, _ = _url_derivatives(d)
    dest = RAW_DIR / "derivatives_bhavcopy" / ystr / fn
    counts[_download(session, url, dest)] += 1
    time.sleep(SLEEP_SEC + random.random() * 0.4)


# ---------------------------------------------------------------------------
# NSE bulk (JSON) feeds for a date range
# ---------------------------------------------------------------------------

# Feeds that carry forward-looking data (board meetings, ex-dates announced weeks ahead).
# These must be re-fetched for future months daily â€” new announcements appear continuously.
_FORWARD_SPECS = {"corp_actions", "earnings"}

def _nse_bulk(session: requests.Session, start: dt.date, end: dt.date, counts: dict,
              force: bool = False):
    for label, url_fn, folder, warmup in _BULK_SPECS:
        for cs, ce in _month_chunks(start, end):
            tag  = cs.strftime("%Y%m")
            dest = RAW_DIR / folder / tag[:4] / f"{tag}.json"
            if not force and dest.exists() and dest.stat().st_size > 100:
                counts["skip"] += 1
                continue
            status = _download_json(session, url_fn(cs, ce), dest, warmup)
            counts[status] += 1
            if status not in ("skip", "ok"):
                print(f"  [warn] {label} {tag}: {status}")
            time.sleep(SLEEP_SEC * 1.5 + random.random())
        session = make_session()   # refresh cookies between feeds


def _nse_bulk_forward(session: requests.Session, base: dt.date, counts: dict,
                      days_ahead: int = 90):
    """Fetch earnings + corp_actions for the next `days_ahead` days.

    Always overwrites existing files â€” board meetings and ex-dates are announced
    on a rolling basis, so cached future-month files go stale within days.
    Only fetches the two forward-looking feeds, not FII/DII.
    """
    end = base + dt.timedelta(days=days_ahead)
    forward_specs = [s for s in _BULK_SPECS if s[0] in _FORWARD_SPECS]
    for label, url_fn, folder, warmup in forward_specs:
        for cs, ce in _month_chunks(base, end):
            tag  = cs.strftime("%Y%m")
            dest = RAW_DIR / folder / tag[:4] / f"{tag}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            status = _download_json(session, url_fn(cs, ce), dest, warmup)
            counts[status] += 1
            if status not in ("skip", "ok"):
                print(f"  [warn] {label} fwd/{tag}: {status}")
            time.sleep(SLEEP_SEC * 1.5 + random.random())
        session = make_session()
    return session


# ---------------------------------------------------------------------------
# Lot size (quarterly one-off)
# ---------------------------------------------------------------------------

def _fetch_deal_day(session: requests.Session, d: dt.date, counts: dict,
                    deal_type: str = "block_deals"):
    """Fetch one day's NSE block or bulk deals via the historical API.

    Uses /api/historicalOR/bulk-block-short-deals which works for any date
    going back to 2010.  Saves as raw/<deal_type>/{YYYY}/{DDMMYYYY}.json.

    The API caps at 70 rows per request.  For block deals this is safe â€”
    typical days have 2â€“25 deals.  Bulk deals hit the cap every day (70
    rows = largest deals by value for that session).

    deal_type : "block_deals" (default) or "bulk_deals"
    """
    dest = RAW_DIR / deal_type / str(d.year) / f"{d.strftime('%d%m%Y')}.json"
    if dest.exists() and dest.stat().st_size > 40:
        counts["skip"] += 1
        return

    ds   = d.strftime("%d-%m-%Y")
    url  = (f"https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"
            f"?from={ds}&to={ds}&optionType={deal_type}")
    hdrs = {**HEADERS, "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"}

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT, headers=hdrs)
            if r.status_code == 200:
                try:
                    payload = r.json()
                    data = payload.get("data", []) if isinstance(payload, dict) else []
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(data), encoding="utf-8")
                    if data:
                        print(f"  [{deal_type}] {d}: {len(data)} deals")
                    counts["ok"] += 1
                    return
                except Exception:
                    pass
            if r.status_code in (404, 503):
                # Holiday / endpoint temporarily down â€” save empty so we skip next time
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps([]), encoding="utf-8")
                counts["404"] += 1
                return
            time.sleep(2 + attempt * 2)
        except Exception:
            time.sleep(2 + attempt * 2)

    print(f"  [warn] {deal_type} {d}: failed after {MAX_RETRIES} attempts")
    counts["fail"] += 1


# Keep old name as alias for backward compat
def _fetch_block_deals(session: requests.Session, d: dt.date, counts: dict,
                       deal_type: str = "block_deals"):
    _fetch_deal_day(session, d, counts, deal_type)


def _fetch_bulk_deals(session: requests.Session, d: dt.date, counts: dict):
    _fetch_deal_day(session, d, counts, deal_type="bulk_deals")


def _backfill_deal_type(deal_type: str, start: dt.date, end: dt.date):
    """Common backfill loop for block_deals or bulk_deals with monthly progress."""
    label = "Block" if deal_type == "block_deals" else "Bulk"
    print(f"\n{label} deal backfill: {start} to {end}")
    session = make_session()
    try:
        session.get("https://www.nseindia.com/report-detail/display-bulk-and-block-deals",
                    timeout=TIMEOUT)
        time.sleep(0.5)
    except Exception:
        pass

    counts     = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    days       = list(_weekdays(start, end))
    prev_month = None

    for i, d in enumerate(days, 1):
        _fetch_deal_day(session, d, counts, deal_type)
        if prev_month is not None and d.month != prev_month:
            # Month boundary â€” print monthly summary and refresh session
            prev_year = d.year if d.month > 1 else d.year - 1
            ym = f"{prev_year}-{prev_month:02d}"
            print(f"  [{ym}] [{i}/{len(days)}] {d}  "
                  f"ok={counts['ok']} skip={counts['skip']} "
                  f"404={counts['404']} fail={counts['fail']}")
            session = make_session()
            try:
                session.get("https://www.nseindia.com/report-detail/display-bulk-and-block-deals",
                            timeout=TIMEOUT)
                time.sleep(0.5)
            except Exception:
                pass
        time.sleep(SLEEP_SEC + random.random() * 0.3)
        prev_month = d.month

    print(f"{label} deal backfill done: ok={counts['ok']} skip={counts['skip']} "
          f"404={counts['404']} fail={counts['fail']}")


def backfill_block_deals(start: str | dt.date, end: str | dt.date | None = None):
    """Backfill historical block deal data from NSE for startâ€¦end.

    Uses the historical API endpoint â€” works from 2010 to today.
    Skips dates already on disk.  Runs day-by-day so the 70-row cap is
    never hit (typical block deal volume: 2â€“25 per day).
    Prints monthly progress.

    Usage:
        python -m pipeline.fetch --block_deals 2010-01-01
        python -m pipeline.fetch --block_deals 2020-01-01 2024-12-31
    """
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    if end is None:
        end = dt.date.today()
    elif isinstance(end, str):
        end = dt.date.fromisoformat(end)
    _backfill_deal_type("block_deals", start, end)


def backfill_bulk_deals(start: str | dt.date, end: str | dt.date | None = None):
    """Backfill historical bulk deal data from NSE for startâ€¦end.

    The API returns up to 70 rows per day (the day's largest deals by value).
    Prints monthly progress.

    Usage:
        python -m pipeline.fetch --bulk_deals 2010-01-01
        python -m pipeline.fetch --bulk_deals 2020-01-01 2024-12-31
    """
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    if end is None:
        end = dt.date.today()
    elif isinstance(end, str):
        end = dt.date.fromisoformat(end)
    _backfill_deal_type("bulk_deals", start, end)


def _fetch_lot_size(session: requests.Session, counts: dict):
    tag  = dt.date.today().strftime("%Y-%m")
    dest = RAW_DIR / "lot_size" / tag / "fo_mktlots.csv"
    if dest.exists() and dest.stat().st_size > 200:
        counts["skip"] += 1
        return
    for url in _LOT_URLS:
        status = _download(session, url, dest)
        counts[status] += 1
        if status in ("ok", "skip"):
            return
        time.sleep(1)


# ---------------------------------------------------------------------------
# Yahoo Finance feeds
# ---------------------------------------------------------------------------

def load_yf_daily_watchlist() -> list[str]:
    """Return symbols from pipeline/yf_daily_watchlist.txt (comments/blanks stripped)."""
    if not YF_DAILY_WATCHLIST.exists():
        return []
    symbols = []
    for line in YF_DAILY_WATCHLIST.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            symbols.append(s.upper())
    return symbols


def _fetch_yf(symbols: list[str] | None = None, force: bool = False, progress_cb=None):
    """Fetch fundamentals, EPS, and corp actions from Yahoo Finance.

    symbols: if provided, fetch only these symbols; otherwise uses training universe.
    force:   if True, re-fetch even when a local file already exists.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[warn] yfinance not installed â€” skipping YF feeds")
        return

    if symbols is None:
        from .universe import load_training
        symbols = load_training()
    total = len(symbols)
    if total == 0:
        print("  [yf] watchlist is empty â€” skipping YF feeds")
        return
    print(f"\n  YF feeds for {total} symbols (force={force}) ...")
    yf_session = _make_yf_session()

    for i, sym in enumerate(symbols, 1):
        if progress_cb:
            progress_cb(f"yf {i}/{total} {sym}")

        ticker = yf.Ticker(f"{sym}.NS", session=yf_session) if yf_session is not None else yf.Ticker(f"{sym}.NS")

        def _needs_write(dest) -> bool:
            return force or not (dest.exists() and dest.stat().st_size > 100)

        # --- fundamentals (quarterly balance sheet / income) ---
        try:
            dest = RAW_DIR / "fundamentals" / f"yf_{sym}.json"
            if _needs_write(dest):
                info = ticker.info or {}
                q_fin = ticker.quarterly_financials
                q_bs  = ticker.quarterly_balance_sheet
                payload = {
                    "info": {k: info.get(k) for k in (
                        "totalRevenue","grossProfit","ebitda","netIncome",
                        "totalDebt","totalAssets","returnOnEquity",
                        "debtToEquity","profitMargins",
                    )},
                    "quarterly_financials": (q_fin.to_dict()
                                             if q_fin is not None else {}),
                    "quarterly_balance_sheet": (q_bs.to_dict()
                                                if q_bs is not None else {}),
                }
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
        except Exception as ex:
            print(f"    [warn] YF fundamentals {sym}: {ex}")

        # --- EPS history ---
        try:
            dest = RAW_DIR / "earnings_eps" / f"yf_{sym}.json"
            if _needs_write(dest):
                ed = ticker.get_earnings_dates(limit=80)
                payload = ed.reset_index().to_dict(orient="records") if ed is not None else []
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
        except Exception as ex:
            print(f"    [warn] YF EPS {sym}: {ex}")

        # --- corporate actions (dividends + splits) ---
        try:
            dest = RAW_DIR / "corp_actions_yf" / f"yf_{sym}.json"
            if _needs_write(dest):
                actions = ticker.actions
                payload = (actions.reset_index().to_dict(orient="records")
                           if actions is not None and not actions.empty else [])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
        except Exception as ex:
            print(f"    [warn] YF actions {sym}: {ex}")

        time.sleep(0.3)

    print("  YF feeds done.")


# ---------------------------------------------------------------------------
# YF retry tracker
# ---------------------------------------------------------------------------

def _load_retry_tracker() -> dict:
    """Load {symbol: {earn_date, added, last_tried}} from disk."""
    if YF_RETRY_TRACKER.exists():
        try:
            return json.loads(YF_RETRY_TRACKER.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_retry_tracker(tracker: dict):
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    YF_RETRY_TRACKER.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def _yf_has_latest_eps(sym: str, earn_date_str: str) -> bool:
    """Return True if the raw YF EPS file has eps_reported for earn_date_str or later."""
    path = RAW_DIR / "earnings_eps" / f"yf_{sym}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return False
        earn_dt = dt.date.fromisoformat(earn_date_str)
        for row in data:
            raw_d = str(row.get("earnings_date", ""))[:10]
            try:
                row_dt = dt.date.fromisoformat(raw_d)
            except ValueError:
                continue
            if row_dt >= earn_dt and row.get("eps_reported") not in (None, ""):
                return True
    except Exception:
        pass
    return False


def get_yf_daily_symbols(date_str: str) -> tuple[list[str], list[str]]:
    """Return (split_check_syms, eps_refresh_syms) for a targeted daily YF fetch.

    split_check_syms : symbols with a SPLIT/BONUS ex_date within the next 3 calendar days
                       â€” needed for patch_splits_from_yf to detect unannounced splits
    eps_refresh_syms : symbols with a board-meeting (earnings) date == date_str,
                       plus any symbols still pending in the retry tracker
    """
    import pandas as pd

    today = dt.date.fromisoformat(date_str)
    watchlist = set(load_yf_daily_watchlist())

    # --- split / bonus check ---
    split_syms: set[str] = set()
    corp_path = SILVER_TABLES.get("corp_actions")
    if corp_path and corp_path.exists():
        corp = pd.read_parquet(corp_path, columns=["symbol", "action_type", "ex_date"])
        corp = corp[corp["action_type"].str.upper().str.contains("SPLIT|BONUS", na=False)]
        corp["ex_date"] = pd.to_datetime(corp["ex_date"], errors="coerce").dt.date
        window_end = today + dt.timedelta(days=3)
        mask = (corp["ex_date"] >= today) & (corp["ex_date"] <= window_end)
        split_syms = set(corp.loc[mask, "symbol"]) & watchlist

    # --- earnings date today ---
    earn_syms: set[str] = set()
    earn_path = SILVER_TABLES.get("earnings")
    if earn_path and earn_path.exists():
        earn = pd.read_parquet(earn_path, columns=["date", "symbol"])
        earn["date"] = pd.to_datetime(earn["date"]).dt.date
        earn_syms = set(earn.loc[earn["date"] == today, "symbol"]) & watchlist

    # --- retry tracker ---
    tracker = _load_retry_tracker()
    retry_syms = set(tracker.keys()) & watchlist

    eps_syms = sorted(earn_syms | retry_syms)
    split_syms = sorted(split_syms)

    if split_syms:
        print(f"  [yf-targeted] split/bonus check: {split_syms}")
    if eps_syms:
        print(f"  [yf-targeted] EPS refresh ({len(earn_syms)} earnings today, "
              f"{len(retry_syms & earn_syms)} overlap, {len(retry_syms - earn_syms)} retry-only): "
              f"{eps_syms}")

    return split_syms, eps_syms


def update_yf_retry_tracker(date_str: str, earn_syms_today: list[str]):
    """Add symbols whose earnings hit today but YF hasn't published yet.
       Remove symbols that are now filled or older than _RETRY_MAX_DAYS.
    """
    tracker = _load_retry_tracker()
    today_str = date_str

    # add new pending symbols
    for sym in earn_syms_today:
        if not _yf_has_latest_eps(sym, date_str):
            if sym not in tracker:
                tracker[sym] = {"earn_date": date_str, "added": today_str, "last_tried": today_str}
                print(f"  [yf-retry] added {sym} (earnings {date_str}, YF not yet updated)")
            else:
                tracker[sym]["last_tried"] = today_str

    # prune: filled or stale
    today_d = dt.date.fromisoformat(date_str)
    to_remove = []
    for sym, info in tracker.items():
        if _yf_has_latest_eps(sym, info["earn_date"]):
            print(f"  [yf-retry] {sym} now filled â€” removing from tracker")
            to_remove.append(sym)
            continue
        added_d = dt.date.fromisoformat(info.get("added", date_str))
        if (today_d - added_d).days > _RETRY_MAX_DAYS:
            print(f"  [yf-retry] {sym} stale ({(today_d - added_d).days}d) â€” dropping")
            to_remove.append(sym)
    for sym in to_remove:
        tracker.pop(sym, None)

    _save_retry_tracker(tracker)


def _fetch_yf_one_eps(ticker, sym: str):
    """Fetch EPS history + fundamentals for one symbol into raw files."""
    dest = RAW_DIR / "earnings_eps" / f"yf_{sym}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ed = ticker.get_earnings_dates(limit=80)
        payload = ed.reset_index().to_dict(orient="records") if ed is not None else []
        dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
    except Exception as ex:
        print(f"    [warn] YF EPS {sym}: {ex}")

    dest = RAW_DIR / "fundamentals" / f"yf_{sym}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        info  = ticker.info or {}
        q_fin = ticker.quarterly_financials
        q_bs  = ticker.quarterly_balance_sheet
        payload = {
            "info": {k: info.get(k) for k in (
                "totalRevenue", "grossProfit", "ebitda", "netIncome",
                "totalDebt", "totalAssets", "returnOnEquity",
                "debtToEquity", "profitMargins",
            )},
            "quarterly_financials":    (q_fin.to_dict() if q_fin is not None else {}),
            "quarterly_balance_sheet": (q_bs.to_dict()  if q_bs  is not None else {}),
        }
        dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
    except Exception as ex:
        print(f"    [warn] YF fundamentals {sym}: {ex}")


def fetch_yf_eps_targeted(symbols: list[str], date_str: str, progress_cb=None):
    """Fetch EPS + fundamentals for a targeted symbol list, then update retry tracker."""
    try:
        import yfinance as yf
    except ImportError:
        print("[warn] yfinance not installed â€” skipping targeted EPS fetch")
        return

    if not symbols:
        return

    print(f"  [yf-eps] targeted fetch for {len(symbols)} symbols ...")
    yf_session = _make_yf_session()
    for i, sym in enumerate(symbols, 1):
        if progress_cb:
            progress_cb(f"yf_eps {i}/{len(symbols)} {sym}")
        _fetch_yf_one_eps(yf.Ticker(f"{sym}.NS", session=yf_session) if yf_session is not None else yf.Ticker(f"{sym}.NS"), sym)
        time.sleep(0.4)

    print("  [yf-eps] done")
    update_yf_retry_tracker(date_str, symbols)


def fetch_yf_actions_daily(symbols: list[str], progress_cb=None) -> None:
    """
    Lightweight YF fetch used in daily runs: only downloads ticker.actions
    (splits + dividends) for the given symbols.  Overwrites existing files
    so the split-patch step always sees fresh data.
    Skips fundamentals and EPS to keep the daily run fast (~0.2 s per symbol).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[warn] yfinance not installed â€” skipping YF split check")
        return

    print(f"  [yf-actions] refreshing actions for {len(symbols)} symbols ...")
    yf_session = _make_yf_session()
    for i, sym in enumerate(symbols, 1):
        if progress_cb:
            progress_cb(f"yf_actions {i}/{len(symbols)} {sym}")
        try:
            ticker = yf.Ticker(f"{sym}.NS", session=yf_session) if yf_session is not None else yf.Ticker(f"{sym}.NS")
            actions = ticker.actions
            payload = (actions.reset_index().to_dict(orient="records")
                       if actions is not None and not actions.empty else [])
            dest = RAW_DIR / "corp_actions_yf" / f"yf_{sym}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(_json_safe(payload), default=str), encoding="utf-8")
        except Exception as ex:
            print(f"    [warn] YF actions {sym}: {ex}")
        time.sleep(0.2)
    print("  [yf-actions] done")


# ---------------------------------------------------------------------------
# Investing.com â€” quarterly EPS estimates (analyst consensus)
# ---------------------------------------------------------------------------

INVESTING_PAIR_IDS = Path(__file__).resolve().parent / "investing_pair_ids.json"

_INV_HDRS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/html, */*",
    "Origin":  "https://in.investing.com",
    "Referer": "https://in.investing.com/",
}


def _load_pair_ids() -> dict[str, int]:
    if INVESTING_PAIR_IDS.exists():
        try:
            return json.loads(INVESTING_PAIR_IDS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_pair_ids(ids: dict) -> None:
    INVESTING_PAIR_IDS.write_text(
        json.dumps(ids, indent=2, sort_keys=True), encoding="utf-8"
    )


def _search_investing_pair_id(session: requests.Session, symbol: str) -> int | None:
    """Search Investing.com India for NSE symbol â†’ pair_ID integer (or None)."""
    hdrs = {**_INV_HDRS, "Accept": "application/json"}
    # Try "SYMBOL NSE" first for precision, fall back to plain symbol
    for q in [f"{symbol} NSE", symbol]:
        try:
            r = session.get(
                "https://api.investing.com/api/search/v2/search",
                params={"q": q, "lang_ID": "1", "domain_ID": "68",
                        "category": "equities"},
                headers=hdrs, timeout=TIMEOUT,
            )
            if r.status_code != 200:
                time.sleep(1)
                continue
            data = r.json()
            # Two known response shapes: quotes[] or hits.hits[]
            candidates = data.get("quotes") or [
                h.get("_source", {})
                for h in data.get("hits", {}).get("hits", [])
            ]
            for c in candidates:
                sym_f = str(c.get("symbol", c.get("ticker", ""))).upper()
                exch  = str(c.get("exchange", "")).upper()
                if sym_f == symbol.upper() and "NSE" in exch:
                    pid = c.get("pair_ID") or c.get("pairId") or c.get("id")
                    if pid:
                        return int(pid)
            time.sleep(0.3)
        except Exception:
            time.sleep(1)
    return None


def _parse_investing_earnings_html(html: str, sym: str) -> list[dict]:
    """Parse Investing.com earnings-history HTML fragment â†’ list of record dicts."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [warn] pip install beautifulsoup4 for Investing.com EPS parsing")
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        tbl  = soup.find("table")
        if tbl is None:
            return []
        all_rows = tbl.find_all("tr")
        if not all_rows:
            return []

        # Detect column positions from header row
        hdr = [c.get_text(" ", strip=True).lower()
               for c in all_rows[0].find_all(["th", "td"])]

        def _ci(*kws):
            for i, h in enumerate(hdr):
                if any(kw in h for kw in kws):
                    return i
            return None

        ci_date    = _ci("date", "release")    or 0
        ci_eps_est = _ci("eps forecast", "eps estimate", "forecast")
        ci_eps_act = _ci("eps actual", "reported eps", "actual")
        ci_surp    = _ci("surprise")
        ci_rev_est = _ci("revenue forecast", "rev forecast", "revenue estimate")
        ci_rev_act = _ci("revenue actual", "rev actual", "revenue reported")

        rows = []
        for tr in all_rows[1:]:
            tds = tr.find_all("td")
            if not tds:
                continue

            def _get(idx):
                if idx is None or idx >= len(tds):
                    return ""
                return tds[idx].get_text(strip=True)

            date_str = _get(ci_date)
            if not date_str:
                continue

            rows.append({
                "symbol":       sym,
                "date":         date_str,
                "eps_estimate": _get(ci_eps_est) if ci_eps_est is not None else "",
                "eps_reported": _get(ci_eps_act) if ci_eps_act is not None else _get(2),
                "surprise_pct": _get(ci_surp)    if ci_surp    is not None else "",
                "rev_estimate": _get(ci_rev_est) if ci_rev_est is not None else "",
                "rev_reported": _get(ci_rev_act) if ci_rev_act is not None else "",
            })
        return rows
    except Exception as e:
        print(f"  [warn] inv_eps HTML parse {sym}: {e}")
        return []


def _fetch_investing_eps_one(session: requests.Session, sym: str,
                              pair_id: int, force: bool = False) -> bool:
    """Fetch quarterly earnings history for one symbol from Investing.com.

    Saves as raw/investing_eps/{sym}.json (list of quarter records).
    Skips if file is <7 days old (unless force=True).
    Returns True on success.
    """
    dest = RAW_DIR / "investing_eps" / f"{sym}.json"
    if not force and dest.exists():
        age = (dt.date.today() -
               dt.date.fromtimestamp(dest.stat().st_mtime)).days
        if age < 7:
            return True  # fresh enough

    hdrs = {**_INV_HDRS,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "text/html, */*; q=0.01"}

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                "https://in.investing.com/instruments/Financials/earnings-history",
                params={"pair_ID": pair_id, "period_type": "Quarterly",
                        "action": "change_period_type"},
                headers=hdrs, timeout=TIMEOUT,
            )
            if r.status_code == 200:
                rows = _parse_investing_earnings_html(r.text, sym)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(rows), encoding="utf-8")
                if rows:
                    print(f"  [inv_eps] {sym}: {len(rows)} quarters")
                return True
            if r.status_code in (403, 404):
                return False
            time.sleep(2 + attempt * 2)
        except Exception:
            time.sleep(2 + attempt * 2)

    print(f"  [warn] inv_eps {sym}: failed after {MAX_RETRIES} attempts")
    return False


def _make_inv_session() -> requests.Session:
    """Create a fresh requests.Session for Investing.com with cookies warmed up."""
    s = requests.Session()
    s.verify = False
    for k, v in _INV_HDRS.items():
        s.headers[k] = v
    try:
        s.get("https://in.investing.com/", timeout=TIMEOUT)
        time.sleep(0.5 + random.random() * 0.5)
    except Exception:
        pass
    return s


def fetch_investing_eps(symbols: list[str] | None = None,
                        force: bool = False,
                        n_workers: int = 4) -> None:
    """Fetch quarterly EPS estimates + actuals from Investing.com.

    Two-phase execution:
      Phase 1 (sequential) â€” build symbol â†’ pair_ID map for any unknown symbols.
                             The search API is rate-sensitive so this stays serial.
      Phase 2 (parallel)   â€” fetch earnings history with n_workers threads.
                             Each thread has its own session; writes go to separate
                             per-symbol files so there are no race conditions.

    Skips symbols whose raw file is <7 days old (use --force to override).

    Usage:
        python -m pipeline.fetch --investing_eps
        python -m pipeline.fetch --investing_eps --force
        python -m pipeline.fetch --investing_eps --workers 6
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .universe import load_training
    if symbols is None:
        symbols = load_training()

    pair_ids = _load_pair_ids()

    # â”€â”€ Phase 1: sequential pair_ID lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    missing = [s for s in symbols if s not in pair_ids]
    if missing:
        print(f"\n[investing_eps] phase 1: looking up {len(missing)} new pair_IDs â€¦")
        search_sess = _make_inv_session()
        for i, sym in enumerate(missing, 1):
            pid = _search_investing_pair_id(search_sess, sym)
            pair_ids[sym] = pid or 0
            time.sleep(0.4 + random.random() * 0.3)
            if i % 50 == 0:
                _save_pair_ids(pair_ids)
                found = sum(1 for p in pair_ids.values() if p)
                print(f"  [{i}/{len(missing)}] pair_ids found={found}")
        _save_pair_ids(pair_ids)

    found = sum(1 for p in pair_ids.values() if p)
    print(f"[investing_eps] pair_ids: {found} found / {len(symbols)} total")

    # â”€â”€ Phase 2: parallel earnings fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fetchable = [(sym, pair_ids[sym]) for sym in symbols if pair_ids.get(sym)]
    if not fetchable:
        print("[investing_eps] no pair_IDs found â€” check search results")
        return

    print(f"[investing_eps] phase 2: fetching {len(fetchable)} symbols "
          f"with {n_workers} workers â€¦")

    completed_lock = threading.Lock()
    completed = [0]

    def _worker(sym: str, pid: int) -> tuple[str, bool]:
        # Each thread gets its own session â€” avoids cookie/connection contention
        sess = _make_inv_session()
        # Stagger start so all workers don't slam the server simultaneously
        time.sleep(random.uniform(0, 2.0))
        ok = _fetch_investing_eps_one(sess, sym, pid, force=force)
        with completed_lock:
            completed[0] += 1
            n = completed[0]
        if n % 25 == 0:
            print(f"  [{n}/{len(fetchable)}] workers={n_workers}")
        return sym, ok

    ok_count = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, sym, pid): sym for sym, pid in fetchable}
        for fut in as_completed(futures):
            _, ok = fut.result()
            if ok:
                ok_count += 1

    print(f"[investing_eps] done â€” fetched={ok_count}/{len(fetchable)}")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def backfill(start: str | dt.date, end: str | dt.date | None = None,
             skip_yf: bool = False, daily_yf: bool = False, progress_cb=None):
    """
    Download all feeds for startâ€¦end (inclusive).
    start / end: 'YYYY-MM-DD' string or date object.
    skip_yf: set True to skip Yahoo Finance (useful for quick re-runs).
    """
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    if end is None:
        end = dt.date.today()
    elif isinstance(end, str):
        end = dt.date.fromisoformat(end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBackfill: {start} to {end}", flush=True)

    counts  = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    session = make_session()
    days    = list(_weekdays(start, end))

    print(f"\n--- Phase 1: NSE daily feeds ({len(days)} trading days) ---", flush=True)
    for i, d in enumerate(days, 1):
        if progress_cb:
            progress_cb(f"nse_daily {i}/{len(days)} {d}")
        print(f"  [daily {i}/{len(days)}] {d} ...", flush=True)
        _nse_day(session, d, counts)
        if i % 5 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {d}  "
                  f"ok={counts['ok']} skip={counts['skip']} "
                  f"404={counts['404']} fail={counts['fail']}", flush=True)
            session = make_session()

    print("\n--- Phase 2: NSE bulk JSON feeds ---", flush=True)
    if progress_cb:
        progress_cb("nse_bulk")
    _nse_bulk(session, start, end, counts)

    print("\n--- Phase 3: Lot sizes ---", flush=True)
    _fetch_lot_size(session, counts)

    if not skip_yf:
        print("\n--- Phase 4: Yahoo Finance feeds ---", flush=True)
        if daily_yf:
            watchlist = load_yf_daily_watchlist()
            if watchlist:
                print(f"  daily mode: {len(watchlist)} symbols from yf_daily_watchlist.txt")
                _fetch_yf(symbols=watchlist, force=True, progress_cb=progress_cb)
            else:
                print("  daily mode: yf_daily_watchlist.txt is empty â€” skipping YF")
        else:
            from .universe import load as load_active_universe, build as build_active_universe
            active_symbols = load_active_universe()
            if not active_symbols:
                active_symbols = build_active_universe()
            index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
            active_symbols = [s for s in active_symbols if s not in index_symbols]
            print(f"  active stock universe mode: {len(active_symbols)} symbols")
            _fetch_yf(symbols=active_symbols, force=True, progress_cb=progress_cb)

    print(f"\nBackfill done: ok={counts['ok']} skip={counts['skip']} "
          f"404={counts['404']} fail={counts['fail']}", flush=True)


def fetch_day(date: str | dt.date, progress_cb=None):
    """
    Download all daily NSE feeds for one trading day.
    Also refreshes the current month's bulk JSON feeds if file is missing.
    """
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)

    if date.weekday() >= 5:
        print(f"[skip] {date} is a weekend")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nFetch day: {date}")

    counts  = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    session = make_session()

    if progress_cb:
        progress_cb(f"nse_daily {date}")
    _nse_day(session, date, counts)

    # block + bulk deals for today
    _fetch_deal_day(session, date, counts, "block_deals")
    time.sleep(SLEEP_SEC)
    _fetch_deal_day(session, date, counts, "bulk_deals")
    time.sleep(SLEEP_SEC)

    # refresh bulk feeds for current month if any are missing
    month_start = date.replace(day=1)
    _nse_bulk(session, month_start, date, counts)

    # forward-looking: earnings + corp_actions for the next 90 days
    # always overwrite so newly-announced board meetings are captured daily
    session = _nse_bulk_forward(session, date, counts, days_ahead=90)

    print(f"  fetch_day done: ok={counts['ok']} skip={counts['skip']} "
          f"404(holiday?)={counts['404']} fail={counts['fail']}")
    return counts


def main():
    args = sys.argv[1:]
    if not args:
        fetch_day(dt.date.today())
    elif args[0] == "--block_deals":
        start = args[1] if len(args) > 1 else "2010-01-01"
        end   = args[2] if len(args) > 2 else None
        backfill_block_deals(start, end)
    elif args[0] == "--bulk_deals":
        start = args[1] if len(args) > 1 else "2010-01-01"
        end   = args[2] if len(args) > 2 else None
        backfill_bulk_deals(start, end)
    elif args[0] == "--investing_eps":
        force = "--force" in args
        n_workers = 4
        for j, a in enumerate(args):
            if a == "--workers" and j + 1 < len(args):
                try:
                    n_workers = int(args[j + 1])
                except ValueError:
                    pass
        fetch_investing_eps(force=force, n_workers=n_workers)
    elif len(args) == 1:
        backfill(args[0])
    else:
        backfill(args[0], args[1])


if __name__ == "__main__":
    main()







