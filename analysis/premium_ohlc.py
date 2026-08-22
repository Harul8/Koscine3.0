"""Near-ATM option premium OHLC over the 5-day window for each v2 book pick.

For each historical pick (signal t -> entry t+1, window t+1..t+5): pick the nearest-to-spot strike
on the nearest monthly expiry that survives the window, then record the CE & PE premium OHLC across
the window (entry open, window high/low, window-end close) + best multiple (window_high / entry).

Writes locks/prod_largemove_v2/book_premiums.csv. Heavy bhavcopy I/O — run in background.

Optional incremental mode: `python analysis/premium_ohlc.py --append` only recomputes picks
whose signal date falls in the trailing LOOKBACK_DAYS window (2026-08, explicit user decision --
was rebuilding all ~646 historical days' bhavcopy on every daily Refresh run, even though the
overwhelming majority of picks' 5-day forward windows resolved long ago and can never change).
Every older, already-resolved pick is read straight from the existing book_premiums.csv and left
untouched. Full rebuild (no args) is still there for a one-time/manual re-run of the whole book.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from options_bhavcopy import load_bhavcopy  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
OUT_PATH = LOCK_V2 / "book_premiums.csv"
LOOKBACK_DAYS = 10   # trailing CALENDAR days of signal dates to recompute in --append mode --
# comfortably covers any pick whose 5-day (trading-day) forward window could still be
# incomplete or only just closed, with margin for weekends/holidays.
APPEND = len(sys.argv) > 1 and sys.argv[1] == "--append"


def main():
    book = pd.read_csv(LOCK_V2 / "book_2024_26.csv", parse_dates=["date"])
    book["symbol"] = book["symbol"].astype(str)
    universe = set(book["symbol"])

    existing = pd.read_csv(OUT_PATH, parse_dates=["date"]) if APPEND and OUT_PATH.exists() else None
    if APPEND:
        cutoff = book["date"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
        book = book[book["date"] >= cutoff].copy()
        print(f"--append: only recomputing picks with signal date >= {cutoff.date()} "
              f"({len(book)} of the full book)", flush=True)

    mk = load_market_data(columns=["date", "symbol", "close"])
    mk["symbol"] = mk["symbol"].astype(str)
    cal = sorted(pd.Timestamp(d) for d in mk["date"].unique())
    cidx = {d: i for i, d in enumerate(cal)}
    spot = mk.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["close"]

    # build pick windows (need full 5-day forward window)
    picks = []
    for row in book.itertuples():
        i = cidx.get(pd.Timestamp(row.date))
        if i is None or i + 5 >= len(cal):
            continue
        entry = cal[i + 1]
        window = [cal[i + 1 + k] for k in range(5)]
        picks.append((row.Index, row.symbol, pd.Timestamp(row.date), entry, window))

    day_picks = defaultdict(list)
    for p in picks:
        for d in p[4]:
            day_picks[d].append(p)

    state: dict = {}
    days = sorted(day_picks)
    for n, d in enumerate(days):
        chain = load_bhavcopy(d)
        if chain.empty:
            continue
        chain = chain[chain["symbol"].isin(universe)]
        if chain.empty:
            continue
        by_sym = {s: g for s, g in chain.groupby("symbol")}
        for pid, sym, sig, entry, window in day_picks[d]:
            sub = by_sym.get(sym)
            if sub is None:
                continue
            st = state.setdefault(pid, {"symbol": sym, "sig": sig})
            if d == entry:
                spt = spot.get((sig, sym))
                if spt is None or not np.isfinite(spt):
                    spt = spot.get((entry, sym))
                if spt is None or not np.isfinite(spt):
                    continue
                exps = sorted(pd.Timestamp(e) for e in sub["expiry"].dropna().unique())
                wend = window[-1]
                valid = [e for e in exps if e >= wend]
                exp = valid[0] if valid else (exps[-1] if exps else None)
                if exp is None:
                    continue
                ce = sub[(sub["expiry"] == exp) & (sub["opt_type"] == "CE")]
                strikes = ce["strike"].dropna().unique()
                if not len(strikes):
                    continue
                strike = float(strikes[int(np.argmin(np.abs(strikes - float(spt))))])
                st.update(strike=strike, expiry=exp, spot=float(spt))
            if "strike" not in st:
                continue
            for ot in ("CE", "PE"):
                r = sub[(sub["expiry"] == st["expiry"]) & (sub["opt_type"] == ot)
                        & (sub["strike"] == st["strike"])]
                if r.empty:
                    continue
                o, h, l, c = (float(r.iloc[0][k]) for k in ("open", "high", "low", "close"))
                k = ot.lower()
                if d == entry:
                    st[f"{k}_entry"], st[f"{k}_h"], st[f"{k}_l"], st[f"{k}_c"] = o, h, l, c
                else:
                    st[f"{k}_h"] = max(st.get(f"{k}_h", h), h)
                    st[f"{k}_l"] = min(st.get(f"{k}_l", l), l)
                    st[f"{k}_c"] = c
        if n % 50 == 0:
            print(f"  processed {n}/{len(days)} days, {len(state)} picks touched", flush=True)

    rows = []
    meta = book.loc[[p[0] for p in picks], ["date", "group", "symbol"]]
    for pid, sym, sig, entry, window in picks:
        st = state.get(pid)
        if not st or "strike" not in st or "ce_entry" not in st:
            continue
        rows.append({
            "date": sig, "group": book.at[pid, "group"], "symbol": sym,
            "strike": st["strike"], "expiry": pd.Timestamp(st["expiry"]).date(), "spot": round(st.get("spot", np.nan), 2),
            "ce_entry": st.get("ce_entry"), "ce_high": st.get("ce_h"), "ce_low": st.get("ce_l"), "ce_close": st.get("ce_c"),
            "pe_entry": st.get("pe_entry"), "pe_high": st.get("pe_h"), "pe_low": st.get("pe_l"), "pe_close": st.get("pe_c"),
        })
    out = pd.DataFrame(rows)
    for leg in ("ce", "pe"):
        out[f"{leg}_mult_best"] = (out[f"{leg}_high"] / out[f"{leg}_entry"]).round(2)
        out[f"{leg}_mult_close"] = (out[f"{leg}_close"] / out[f"{leg}_entry"]).round(2)
    if existing is not None and len(existing):
        # every existing row OUTSIDE the recomputed window is kept exactly as-is (long-since
        # resolved, can never change); rows inside it are replaced by the fresh recompute above.
        keep_old = existing[existing["date"] < cutoff]
        out = pd.concat([keep_old, out], ignore_index=True).sort_values(["date", "symbol"])
    out.to_csv(OUT_PATH, index=False)
    print(f"\nsaved {len(out)} pick-premiums -> {OUT_PATH}")
    print(f"CE best-multiple median {out.ce_mult_best.median():.2f} | PE best-multiple median {out.pe_mult_best.median():.2f}")


if __name__ == "__main__":
    main()
