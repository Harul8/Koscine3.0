"""
NSDL Fortnightly Sectoral FII Investment scraper.

Source URL pattern:
  https://www.fpi.nsdl.co.in/web/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/
    FIIInvestSector_<Mon><DD><YYYY>.html

  Example: FIIInvestSector_Apr302026.html  → 30-Apr-2026 fortnight report

Each report covers one fortnight (1-15 of month, or 16-end of month).  Two
reports per month → ~24 per year × 14 years (2012-now) ≈ ~336 files total.

Stages
------
This module is fetch-only.  Parsing the per-sector flow tables into a silver
table is a separate step (`build_silver_fii_sector`) that we wire up once we
have local HTML samples and know the actual table structure.

Usage
-----
    python -m pipeline.fetch_fii_sector --test       # fetch Apr-2026 only (smoke)
    python -m pipeline.fetch_fii_sector --year 2012  # backfill one year
    python -m pipeline.fetch_fii_sector --all        # full backfill 2012->today
    python -m pipeline.fetch_fii_sector --recent     # last 90 days only
"""
from __future__ import annotations
import argparse
import calendar
import datetime as dt
import random
import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import urllib3

from .config import RAW_DIR, SILVER_DIR

# Squelch the noisy InsecureRequestWarning — NSDL's cert chain is iffy on some
# Windows boxes.  Site is public, no auth, no secrets.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAW_FII_SECTOR_DIR = RAW_DIR / "fii_sector"
# (Legacy single-format constant removed — see _BASE_URL_TEMPLATE below)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────────────────────────────────────────────────────
# Date generation: last day of each month only.
#
# Each NSDL end-of-month report contains BOTH fortnights for that month
# (Net Apr 1-15 AND Net Apr 16-30 columns), so mid-month files are redundant.
# Restricting to EOM cuts our fetch surface in half (173 dates instead of 346).
# Already-cached mid-month files are kept on disk — the parser will still
# pick them up and dedupe at (fortnight_end, sector) granularity.
# ─────────────────────────────────────────────────────────────────────────
def month_end_dates(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    """Yield the last day of each month in [start, end]."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        last_n = calendar.monthrange(cur.year, cur.month)[1]
        d_last = dt.date(cur.year, cur.month, last_n)
        if start <= d_last <= end:
            yield d_last
        # Advance one month
        cur = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)


# Back-compat alias — older callers may import the old name
fortnight_dates = month_end_dates


# ─────────────────────────────────────────────────────────────────────────
# URL stub generation — NSDL changed conventions several times.
# Confirmed historical formats:
#   2020+:      FIIInvestSector_Apr302026.html         (3-letter month, .html)
#   2018-2019:  FIIInvestSector_October312018.html     (full month name, .html)
#   2014:       FIIInvestSector_Oct312014.htm          (.htm extension)
#               FIIInestSector_Oct15302014.htm         (typo 'Inest', combined fortnight)
#   2012:       FIIInvestSecor_Jan312012.html          (typo 'Secor' — missing 't')
#               FIIInvestSector_Mar302012.html         (last day reported as 30 even though 31)
#   2020-04:    FIIInvestSectorApril302020.html        (no underscore before month name)
#
# Strategy:
#   - try standard, no-underscore, and two name-typo prefixes
#   - try 3-letter and full month names
#   - try both .html and .htm
#   - for EOM dates, also try (last_day - 1) as the day stub (some 31-day
#     months are reported with day=30)
#   - dedupe — many variants collapse when day >= 10
# ─────────────────────────────────────────────────────────────────────────
_URL_DIR = ("https://www.fpi.nsdl.co.in/web/StaticReports/"
            "Fortnightly_Sector_wise_FII_Investment_Data/")

# (prefix, separator) pairs that we've seen in the wild.
# Most likely first so a successful fetch returns quickly.
_PREFIX_VARIANTS = [
    ("FIIInvestSector", "_"),    # standard
    ("FIIInvestSector", ""),     # no underscore (seen Apr 2020)
    ("FIIInestSector",  "_"),    # 'Inest' typo (seen 2014)
    ("FIIInvestSecor",  "_"),    # 'Secor' typo (seen Jan 2012)
]


def url_candidates(d: dt.date) -> list[str]:
    mon3     = d.strftime("%b")        # Apr
    mon_full = d.strftime("%B")        # April
    yr       = d.strftime("%Y")        # 2026
    last_n   = calendar.monthrange(d.year, d.month)[1]
    is_eom   = (d.day == last_n)

    # Days to try.  For end-of-month requests we also try (last_n - 1) since
    # NSDL sometimes ends the fortnight on the 30th of a 31-day month.
    days = [d.day]
    if is_eom and last_n > 1:
        days.append(last_n - 1)

    # Build date-stub variants
    stubs: list[str] = []
    for day in days:
        for mon_token in (mon3, mon_full):
            stubs.append(f"{mon_token}{day}{yr}")
            if day < 10:
                stubs.append(f"{mon_token}{day:02d}{yr}")

    # Combined-fortnight stub — only for EOM dates.  Try both last_n and last_n-1.
    if is_eom:
        for mon_token in (mon3, mon_full):
            stubs.append(f"{mon_token}15{last_n}{yr}")
            if last_n > 1:
                stubs.append(f"{mon_token}15{last_n - 1}{yr}")

    # Dedupe while preserving order
    seen = set(); ordered = []
    for s in stubs:
        if s not in seen:
            seen.add(s); ordered.append(s)

    # Cross-product with prefix and extension variants
    urls: list[str] = []
    for stub in ordered:
        for prefix, sep in _PREFIX_VARIANTS:
            for ext in ("html", "htm"):
                urls.append(f"{_URL_DIR}{prefix}{sep}{stub}.{ext}")
    return urls


# ─────────────────────────────────────────────────────────────────────────
# Polite fetcher
# ─────────────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_one(session: requests.Session, d: dt.date,
              save_dir: Path, retries: int = 2,
              verify_ssl: bool = False) -> tuple[Path | None, str | None]:
    """Fetch a single fortnight report.  Returns (saved_path, error_msg).

    Saves RAW bytes from the HTTP response so the original encoding (UTF-8
    or UTF-16 LE/BE) is preserved.  The parser uses an encoding-sniffing
    reader to decode at parse time.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{d.isoformat()}.html"

    # Skip if already cached and non-trivial in size
    if save_path.exists() and save_path.stat().st_size > 2_000:
        return save_path, "cached"

    last_err = None
    for url in url_candidates(d):
        for attempt in range(retries + 1):
            try:
                r = session.get(url, timeout=20, verify=verify_ssl)
                if r.status_code == 200 and len(r.content) > 2_000:
                    save_path.write_bytes(r.content)
                    return save_path, None
                if r.status_code == 404:
                    last_err = "404"
                    break   # try next URL variant
                last_err = f"http {r.status_code}"
            except Exception as e:
                last_err = type(e).__name__
                time.sleep(1.0 + random.random())
        # backoff before trying the next URL variant
        time.sleep(0.4)
    return None, last_err


def _read_html_text(fp: Path) -> str:
    """Read an HTML file with encoding sniffing — handles UTF-8 + UTF-16 LE/BE."""
    raw = fp.read_bytes()
    # BOM-based detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    # Heuristic: lots of null bytes in the first chunk = UTF-16 without BOM
    if raw[:200].count(b"\x00") > 50:
        # Decide LE vs BE: in ASCII content, LE has nulls in odd positions, BE in even
        odd_nulls  = sum(1 for i in range(1, min(200, len(raw)), 2) if raw[i] == 0)
        even_nulls = sum(1 for i in range(0, min(200, len(raw)), 2) if raw[i] == 0)
        encoding = "utf-16-le" if odd_nulls > even_nulls else "utf-16-be"
        return raw.decode(encoding, errors="replace")
    # Default UTF-8
    return raw.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────
# Public driver
# ─────────────────────────────────────────────────────────────────────────
def fetch_range(start: dt.date, end: dt.date,
                pause: float = 0.4, verify_ssl: bool = False) -> dict:
    """Fetch end-of-month NSDL sectoral reports in [start, end].
    Each EOM file covers both fortnights of its month, so a single file
    per month is sufficient.  Saves HTML to RAW_FII_SECTOR_DIR."""
    session = make_session()
    dates = list(month_end_dates(start, end))
    n_total = len(dates)
    n_ok = n_cached = n_fail = 0
    failures: list[tuple[dt.date, str]] = []

    print(f"[fii_sector] {n_total} end-of-month reports from {start} to {end} "
          f"(each EOM file covers both fortnights of its month)")
    print(f"[fii_sector] saving to {RAW_FII_SECTOR_DIR}")

    for i, d in enumerate(dates, 1):
        path, err = fetch_one(session, d, RAW_FII_SECTOR_DIR, verify_ssl=verify_ssl)
        if path is None:
            n_fail += 1
            failures.append((d, err or "unknown"))
        elif err == "cached":
            n_cached += 1
        else:
            n_ok += 1
        if i % 25 == 0 or i == n_total:
            print(f"  {i:>4}/{n_total}  {d}  ok={n_ok}  cached={n_cached}  fail={n_fail}")
        # Polite pacing — random jitter so we don't hammer in lockstep
        time.sleep(pause + random.uniform(0, 0.2))

    print(f"\n[fii_sector] done: ok={n_ok}, cached={n_cached}, failed={n_fail} / {n_total}")
    if failures and n_fail <= 30:
        print("  failures (date, err):")
        for d, err in failures[:30]:
            print(f"    {d}  {err}")
    elif failures:
        print(f"  ({n_fail} failures — first 10:)")
        for d, err in failures[:10]:
            print(f"    {d}  {err}")
    return {"ok": n_ok, "cached": n_cached, "failed": n_fail,
            "total": n_total, "failures": failures}


# ─────────────────────────────────────────────────────────────────────────
# Parser: NSDL fortnightly sectoral report → tidy rows
# ─────────────────────────────────────────────────────────────────────────
# NSDL's table layout has evolved across four eras:
#   2012-2015: 26 total cols, 6 cols per major group (Equity, Debt, Total × INR Cr & USD Mn)
#   2018-2021: ~42 cols, ~10 cols per group (added Debt VRR, Hybrid)
#   2022+:     98 cols, 24 cols per group (added Debt-FAR, MF sub-classes, AIF)
#
# The ONE invariant: in every era, the FIRST column of each major group
# is "Equity in INR Cr" (asset class = Equity, units = INR).  We discover
# the major-group boundaries by scanning row 0 for unique header strings,
# then take the first column of each group as our Equity-INR data column.
#
# Sector list also evolved (42 → 32 → 35 → 24).  We normalise every name
# and map it to the current 24-sector canonical taxonomy.  Names that
# don't map (10 individual issuer entities in 2012 like NABARD, JP Morgan)
# are dropped.

# Current 24 canonical sectors (the 2022+ NSDL taxonomy)
CANONICAL_SECTORS = {
    "Automobile and Auto Components", "Capital Goods", "Chemicals",
    "Construction", "Construction Materials", "Consumer Durables",
    "Consumer Services", "Diversified", "Fast Moving Consumer Goods",
    "Financial Services", "Forest Materials", "Healthcare",
    "Information Technology", "Media, Entertainment & Publication",
    "Metals & Mining", "Oil, Gas & Consumable Fuels", "Power",
    "Realty", "Services", "Telecommunication", "Textiles",
    "Utilities", "Sovereign", "Others",
}

# Historical name → canonical (key = lower-cased + whitespace-collapsed + digit-stripped)
SECTOR_ALIAS_MAP = {
    # Renames where canonical name is also valid input
    "automobile and auto components":     "Automobile and Auto Components",
    "automobiles & auto components":      "Automobile and Auto Components",
    "capital goods":                      "Capital Goods",
    "general industrials":                "Capital Goods",
    "chemicals":                          "Chemicals",
    "chemicals & petrochemicals":         "Chemicals",
    "construction":                       "Construction",
    "construction materials":             "Construction Materials",
    "consumer durables":                  "Consumer Durables",
    "consumer services":                  "Consumer Services",
    "diversified consumer services":      "Consumer Services",
    "hotels, restaurants & tourism":      "Consumer Services",
    "diversified":                        "Diversified",
    "fast moving consumer goods":         "Fast Moving Consumer Goods",
    "food, beverages & tobacco":          "Fast Moving Consumer Goods",
    "household & personal products":      "Fast Moving Consumer Goods",
    "food & drugs retailing":             "Fast Moving Consumer Goods",
    "financial services":                 "Financial Services",
    "total financial services":           "Financial Services",
    "insurance":                          "Financial Services",
    "forest materials":                   "Forest Materials",
    "healthcare":                         "Healthcare",
    "pharmaceuticals & biotechnology":    "Healthcare",
    "healthcare equipment & supplies":    "Healthcare",
    "healthcare services":                "Healthcare",
    "information technology":             "Information Technology",
    "hardware technology & equipment":    "Information Technology",
    "software & services":                "Information Technology",
    "media, entertainment & publication": "Media, Entertainment & Publication",
    "media":                              "Media, Entertainment & Publication",
    "metals & mining":                    "Metals & Mining",
    "oil, gas & consumable fuels":        "Oil, Gas & Consumable Fuels",
    "oil & gas":                          "Oil, Gas & Consumable Fuels",
    "coal":                               "Oil, Gas & Consumable Fuels",
    "power":                              "Power",
    "realty":                             "Realty",
    "real estate investment":             "Realty",
    "services":                           "Services",
    "commercial services & supplies":     "Services",
    "retailing":                          "Services",
    "transportation":                     "Services",   # user-confirmed mapping
    "telecommunication":                  "Telecommunication",
    "telecom services":                   "Telecommunication",
    "telecommunications equipment":       "Telecommunication",
    "textiles":                           "Textiles",
    "textiles, apparels & accessories":   "Textiles",
    "utilities":                          "Utilities",
    "sovereign":                          "Sovereign",
    "others":                             "Others",
}

# Single-cell minimum table dimensions (defensive — older reports vary)
_MIN_TABLE_ROWS = 8
_MIN_TABLE_COLS = 10


_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]


def _parse_month_name(s: str) -> int | None:
    """Map a month-name token (3+ chars) to 1..12, tolerating typos.
    Examples: 'September' -> 9, 'Septemebr' -> 9 (typo in 2013-09 NSDL data)."""
    s = str(s).lower().strip()
    # Direct prefix match handles correct names + most typos (e.g. 'septemebr'
    # starts with 'sep' so still resolves to 9).
    for i, m in enumerate(_MONTH_NAMES):
        if s.startswith(m):
            return i + 1
    return None


def _parse_date_token(s: str) -> dt.date | None:
    """'AUC as on April 15, 2026' or 'Apr 15 2026' -> date(2026, 4, 15)."""
    m = re.search(r"([A-Za-z]{3,})\s+(\d{1,2})[,\s]+(\d{4})", str(s))
    if not m:
        return None
    mon_str, day, year = m.groups()
    mn = _parse_month_name(mon_str)
    if mn is None:
        return None
    try:
        return dt.date(int(year), mn, int(day))
    except Exception:
        return None


def _parse_fortnight_range(s: str) -> tuple[dt.date | None, dt.date | None]:
    """'Net Investment April 01-15, 2026' -> (Apr 1 2026, Apr 15 2026)."""
    m = re.search(r"([A-Za-z]{3,})\s+(\d{1,2})\s*-\s*(\d{1,2})[,\s]+(\d{4})", str(s))
    if not m:
        return None, None
    mon_str, d1, d2, year = m.groups()
    mn = _parse_month_name(mon_str)
    if mn is None:
        return None, None
    try:
        return (dt.date(int(year), mn, int(d1)),
                dt.date(int(year), mn, int(d2)))
    except Exception:
        return None, None


def _to_num(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in {"-", "nan", "na"}:
        return np.nan
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def _normalize_sector_for_lookup(name: str) -> str:
    """Normalise a sector name for alias lookup.
       lower-case, collapse whitespace, strip trailing footnote digits."""
    s = re.sub(r"\s+", " ", str(name)).strip()
    s = re.sub(r"\d+$", "", s).strip()
    return s.lower()


def _find_major_groups(t: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Discover the major column groups by scanning row 0 for unique
    AUC / Net-Investment headers.  Returns list of (col_start, col_end, header).

    Each group's FIRST column is the Equity-in-INR-Cr column we want.
    Headers are whitespace-normalised before comparison so older reports
    with double-spaced text (e.g. 2012's "Net  Investment") match.
    """
    n_cols = t.shape[1]
    row0 = []
    for c in range(n_cols):
        v = t.iloc[0, c]
        # Normalise whitespace: 2012 reports often have "Net  Investment" (double space)
        row0.append("" if pd.isna(v) else re.sub(r"\s+", " ", str(v)).strip())

    groups: list[tuple[int, int, str]] = []
    cur_start = None
    cur_text = None
    for c, text in enumerate(row0):
        u = text.upper()
        is_relevant = ("AUC" in u) or ("NET INVEST" in u)
        if is_relevant:
            if text != cur_text:
                if cur_start is not None:
                    groups.append((cur_start, c - 1, cur_text))
                cur_start = c
                cur_text = text
            # else: continuation of same group via colspan
        else:
            if cur_start is not None:
                groups.append((cur_start, c - 1, cur_text))
                cur_start = None
                cur_text = None
    if cur_start is not None:
        groups.append((cur_start, n_cols - 1, cur_text))
    return groups


def _parse_nsdl_table(t: pd.DataFrame) -> pd.DataFrame:
    """Core parser: given a DataFrame where row 0 has the major-group headers
    ("AUC as on...", "Net Investment...") and sector rows follow, extract
    fortnight × canonical-sector data.  Used by both the HTML parser and the
    Excel parser."""
    if t.shape[0] < _MIN_TABLE_ROWS or t.shape[1] < _MIN_TABLE_COLS:
        raise ValueError(f"table too small: {t.shape}")

    # ── Discover major groups + classify each as AUC or Net Investment ──
    # Net groups are further classified by their date range:
    #   F1  = starts on day 1, ends on or before the 15th
    #   F2  = starts on day >= 16
    #   full-month / other = skipped (some 2012 reports show monthly aggregates
    #     like "Net Investment May 1-31" which are F1+F2 combined — ambiguous)
    auc_groups: list[tuple[dt.date, int, str]] = []  # (date, equity_col, header)
    f1_groups:  list[tuple[dt.date, int, str]] = []  # (end_date, equity_col, header)
    f2_groups:  list[tuple[dt.date, int, str]] = []  # (end_date, equity_col, header)
    for col_start, _, hdr in _find_major_groups(t):
        u = hdr.upper()
        if "AUC" in u:
            d = _parse_date_token(hdr)
            if d:
                auc_groups.append((d, col_start, hdr))
        elif "NET INVEST" in u:
            start, end = _parse_fortnight_range(hdr)
            if not (start and end):
                continue
            if start.day == 1 and end.day <= 15:
                f1_groups.append((end, col_start, hdr))
            elif start.day >= 16:
                f2_groups.append((end, col_start, hdr))
            # else: full-month / ambiguous — skip

    if len(auc_groups) < 1:
        raise ValueError(f"no AUC groups: {[g[2][:30] for g in _find_major_groups(t)]}")
    if not (f1_groups or f2_groups):
        raise ValueError("no valid F1 / F2 Net Investment groups (only full-month or empty)")

    auc_groups.sort()
    f1_groups.sort()
    f2_groups.sort()
    aum_mid_col = auc_groups[0][1]                                     # AUC at mid-month
    aum_end_col = auc_groups[-1][1] if len(auc_groups) > 1 else None   # AUC at month-end

    # ── Iterate sector rows ──
    rows = []
    for ri in range(t.shape[0]):
        sr = str(t.iloc[ri, 0]).strip()
        if not sr.isdigit():
            continue
        raw_name = str(t.iloc[ri, 1]).strip()
        if not raw_name or raw_name.lower() in ("nan", "grand total"):
            continue
        canonical = SECTOR_ALIAS_MAP.get(_normalize_sector_for_lookup(raw_name))
        if canonical is None:
            continue

        # Emit one row per fortnight that this report covers.
        # F1's AUM = mid-month AUC (= end-of-F1 = start-of-F2)
        # F2's AUM = month-end AUC (= end-of-F2)
        for f_end, f_col, _ in f1_groups:
            rows.append({
                "fortnight_end": f_end, "sector": canonical, "sector_raw": raw_name,
                "fii_equity_net_cr": _to_num(t.iloc[ri, f_col]),
                "fii_equity_aum_cr": _to_num(t.iloc[ri, aum_mid_col]),
            })
        for f_end, f_col, _ in f2_groups:
            rows.append({
                "fortnight_end": f_end, "sector": canonical, "sector_raw": raw_name,
                "fii_equity_net_cr": _to_num(t.iloc[ri, f_col]),
                "fii_equity_aum_cr": (_to_num(t.iloc[ri, aum_end_col])
                                       if aum_end_col is not None else float("nan")),
            })
    return pd.DataFrame(rows)


def parse_nsdl_sector_report(fp: Path) -> pd.DataFrame:
    """Parse one NSDL fortnightly sectoral HTML file via the core table parser."""
    tables = pd.read_html(StringIO(_read_html_text(fp)))
    if not tables:
        raise ValueError("no tables in HTML")
    return _parse_nsdl_table(tables[0])


def _excel_sheet_to_nsdl_table(fp: Path, sheet: str) -> pd.DataFrame:
    """Read one Excel sheet (hand-prepared dump for a missing fortnight) and
    reshape it so the existing _parse_nsdl_table() can consume it.

    Handles two layout quirks:
      - Excel may have leading rows above the actual NSDL header — trim them
      - Some sheets put "Net Investment" in row 0 and the date in row 1 (the
        Sep 2013 sheet); merge those rows so row 0 ends up self-contained.
    """
    df = pd.read_excel(fp, sheet_name=sheet, header=None)

    # Locate the row containing "AUC" markers
    auc_row = None
    for i in range(min(8, len(df))):
        if any("AUC" in str(v) for v in df.iloc[i].values if pd.notna(v)):
            auc_row = i
            break
    if auc_row is None:
        raise ValueError(f"{sheet}: no AUC row found")
    df = df.iloc[auc_row:].reset_index(drop=True)

    # If Net-Investment cells in row 0 lack a year token (e.g. just say
    # "Net Investment"), augment from row 1.
    def _has_year(s: str) -> bool:
        return bool(re.search(r"\d{4}", s))

    if len(df) >= 2:
        needs_merge = any(
            pd.notna(v) and "Net Investment" in str(v) and not _has_year(str(v))
            for v in df.iloc[0].values
        )
        if needs_merge:
            for c in range(df.shape[1]):
                v0 = df.iloc[0, c]
                v1 = df.iloc[1, c]
                if (pd.notna(v0) and "Net Investment" in str(v0)
                        and not _has_year(str(v0)) and pd.notna(v1)):
                    df.iloc[0, c] = f"{str(v0).strip()} {str(v1).strip()}"
            df = df.drop(index=1).reset_index(drop=True)
    return df


def parse_nsdl_sector_excel(fp: Path) -> pd.DataFrame:
    """Parse all sheets of a hand-prepared Excel of missing NSDL reports.
    Each sheet = one fortnight report.  Returns one combined DataFrame."""
    chunks = []
    xls = pd.ExcelFile(fp)
    for sheet in xls.sheet_names:
        try:
            t = _excel_sheet_to_nsdl_table(fp, sheet)
            chunks.append(_parse_nsdl_table(t))
        except Exception as e:
            print(f"  [skip] {fp.name}::{sheet}: {e}")
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_silver_fii_sector(verbose: bool = True) -> pd.DataFrame:
    """Parse all cached HTML reports → silver/fii_sector.parquet.

    Reports overlap: each EOM file shows 2 fortnights (Apr 1-15 + Apr 16-30),
    so the Apr 15 mid-month file (if cached) duplicates the Apr 1-15 fortnight
    that's also in Apr 30 EOM file.  Within the same fortnight, multiple
    historical sectors may also map to the same canonical (e.g. Telecom Services
    + Telecommunications Equipment both → Telecommunication).  We groupby
    (fortnight_end, sector) and SUM the values to merge cleanly across both
    types of overlap.
    """
    html_files  = sorted(RAW_FII_SECTOR_DIR.glob("*.html"))
    excel_files = sorted(RAW_FII_SECTOR_DIR.glob("*.xlsx"))
    # Skip Excel lockfiles (~$missing files.xlsx etc.)
    excel_files = [f for f in excel_files if not f.name.startswith("~$")]

    if not html_files and not excel_files:
        print(f"[fii_sector] no HTML/XLSX files in {RAW_FII_SECTOR_DIR} - run fetch first")
        return pd.DataFrame()

    chunks, n_ok, errors = [], 0, []
    for fp in html_files:
        try:
            df = parse_nsdl_sector_report(fp)
            chunks.append(df)
            n_ok += 1
        except Exception as e:
            errors.append((fp.name, str(e)[:120]))

    # Hand-prepared Excel fill-ins (each sheet = one missing fortnight report)
    for fp in excel_files:
        try:
            df = parse_nsdl_sector_excel(fp)
            if not df.empty:
                chunks.append(df)
                n_ok += 1
                print(f"[fii_sector] ingested Excel {fp.name}: {len(df):,} rows")
        except Exception as e:
            errors.append((fp.name, str(e)[:120]))

    if not chunks:
        print("[fii_sector] no parseable files")
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)

    # Save a diagnostic dump of all raw sector names + their canonical mapping
    sector_diag = (raw.groupby(["sector_raw", "sector"], dropna=False)
                       .size().reset_index(name="rows")
                       .sort_values("rows", ascending=False))
    diag_path = SILVER_DIR / "fii_sector_alias_audit.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    sector_diag.to_csv(diag_path, index=False)

    # First, when the SAME raw sector appears twice for one fortnight (mid-month +
    # EOM file both contain that fortnight's data), drop_duplicates.  Then
    # groupby (fortnight_end, canonical_sector) and sum across merged sectors.
    deduped = (raw.dropna(subset=["fortnight_end", "sector"])
                  .drop_duplicates(subset=["fortnight_end", "sector_raw"]))
    combined = (deduped.groupby(["fortnight_end", "sector"], as_index=False)
                       .agg(fii_equity_net_cr=("fii_equity_net_cr", "sum"),
                            fii_equity_aum_cr=("fii_equity_aum_cr", "sum"),
                            n_source_sectors=("sector_raw", "nunique"))
                       .sort_values(["fortnight_end", "sector"])
                       .reset_index(drop=True))
    combined["fortnight_end"] = pd.to_datetime(combined["fortnight_end"])

    silver_path = SILVER_DIR / "fii_sector.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(silver_path)

    if verbose:
        print(f"\n[fii_sector] parsed OK: {n_ok}/{len(html_files) + len(excel_files)} sources")
        if errors:
            print(f"  parse errors ({len(errors)}):")
            for n, e in errors[:8]:
                print(f"    {n}: {e}")
            if len(errors) > 8:
                print(f"    ... and {len(errors)-8} more")
        print(f"[fii_sector] saved {len(combined):,} rows -> {silver_path}")
        print(f"  date range:    {combined['fortnight_end'].min().date()} -> {combined['fortnight_end'].max().date()}")
        print(f"  canonical sectors used: {combined['sector'].nunique()}")
        print(f"  fortnights covered:     {combined['fortnight_end'].nunique()}")
        print(f"  flow stats:    mean={combined['fii_equity_net_cr'].mean():+.1f}  "
              f"std={combined['fii_equity_net_cr'].std():.1f}  (Rs Cr per sector-fortnight)")
        print(f"  alias audit:   {diag_path}")
    return combined


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true",
                   help="fetch just April 2026 (2 files) — smoke test")
    g.add_argument("--year", type=int,
                   help="fetch one calendar year (e.g. --year 2012)")
    g.add_argument("--recent", action="store_true",
                   help="fetch the last 90 days")
    g.add_argument("--all", action="store_true",
                   help="fetch full backfill 2012-01-15 to today")
    g.add_argument("--build-silver", action="store_true",
                   help="skip fetch; parse all cached HTML files and write silver/fii_sector.parquet")
    p.add_argument("--also-build-silver", action="store_true",
                   help="after a fetch run, also build the silver table")
    p.add_argument("--verify-ssl", action="store_true",
                   help="enforce SSL cert verification (default: off — NSDL cert chain is unreliable on Windows)")
    args = p.parse_args()

    if args.build_silver:
        build_silver_fii_sector()
        return

    today = dt.date.today()
    if args.test:
        start, end = dt.date(2026, 4, 1), dt.date(2026, 4, 30)
    elif args.recent:
        start, end = today - dt.timedelta(days=90), today
    elif args.year:
        start, end = dt.date(args.year, 1, 1), dt.date(args.year, 12, 31)
    else:  # --all
        start, end = dt.date(2012, 1, 15), today

    fetch_range(start, end, verify_ssl=args.verify_ssl)

    if args.also_build_silver:
        build_silver_fii_sector()


if __name__ == "__main__":
    sys.exit(main())
