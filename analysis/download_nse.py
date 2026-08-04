"""Download NSE FO + cash bhavcopy for a date range into ESN Data\\raw (to extend the feature data)."""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(r"C:\Users\rahul\Koscine 3.0\data\raw")
DERIV = RAW / "derivatives_bhavcopy" / "2026"
CASH = RAW / "cash_bhavcopy" / "2026"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
       "Referer": "https://www.nseindia.com/"}


def fetch(url: str, dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1000:
        return True, f"exists ({dest.stat().st_size:,}B)"
    try:
        req = urllib.request.Request(url, headers=HDR)
        data = urllib.request.urlopen(req, timeout=40).read()
        if len(data) < 500:
            return False, f"too small ({len(data)}B)"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True, f"downloaded ({len(data):,}B)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def main(start: str, end: str) -> None:
    days = pd.bdate_range(start, end)  # business days only
    print(f"Fetching FO + cash bhavcopy for {[d.date().isoformat() for d in days]}\n")
    ok = 0
    for d in days:
        ymd, dmy = d.strftime("%Y%m%d"), d.strftime("%d%m%Y")
        fo = DERIV / f"BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
        cash = CASH / f"sec_bhavdata_full_{dmy}.csv"
        s1, m1 = fetch(f"https://nsearchives.nseindia.com/content/fo/{fo.name}", fo)
        s2, m2 = fetch(f"https://nsearchives.nseindia.com/products/content/{cash.name}", cash)
        # validate FO zip
        zinfo = ""
        if s1 and zipfile.is_zipfile(fo):
            with zipfile.ZipFile(fo) as z:
                n = sum(1 for _ in z.open(z.namelist()[0]))
            zinfo = f" | FO rows~{n}"
        elif s1:
            s1 = False; zinfo = " | FO NOT a valid zip"
        crows = ""
        if s2:
            try:
                crows = f" | cash rows {sum(1 for _ in cash.open(encoding='utf-8', errors='ignore')) - 1}"
            except Exception:
                pass
        print(f"  {d.date()}  FO: {m1}{zinfo}   CASH: {m2}{crows}")
        ok += int(s1 and s2)
    print(f"\n{ok}/{len(days)} days fully fetched (FO+cash).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-06-08",
         sys.argv[2] if len(sys.argv) > 2 else "2026-06-12")
