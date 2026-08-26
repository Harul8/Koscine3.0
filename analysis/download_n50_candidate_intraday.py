"""Candidate-level intraday download for the NIFTY 50 sell-strategy universe.

WHY: the 2026-08 pilot (analysis/pilot_intraday_entry_test.py) showed the sell backtest's
entry price -- the 09:15 opening print -- is not fillable for ~55% of trades (thinnest leg
traded ZERO in the first 5 minutes), and that ~75-85% of recorded PnL disappears once entry is
re-priced to something transactable. That pilot could only RE-PRICE the trades that already
fired; it could not test a genuinely different system, because a 14:30-signal/15:15-entry
strategy would SELECT DIFFERENT TRADES. This download provides what that needs: intraday
prices for EVERY structurally-valid candidate leg, not just the ones the EOD backtest picked.

SCOPE (deliberately bounded, per explicit user decision -- NIFTY 50 only, not the full A/B
universe): every distinct option contract appearing as a leg of any Broken-Wing or Skew
candidate on a NIFTY50 underlying, from Upstox's expired-instrument cutoff (~Oct 2024) on.
~30,800 contracts + ~1,070 contract-list lookups.

STORAGE: raw 1-minute candles for this set would be ~100M rows. Since the question is "what
could you transact at a given moment", each day is reduced on the fly to a fixed set of MARKS
(the last print at or before each), plus cumulative traded volume up to that mark -- which is
what makes a fillability gate possible at all. ~3M rows total.

Also fetches NIFTY50 EQUITY 5-minute bars: a real 14:30 system picks its strikes off the
underlying's 14:30 price, not the previous close, so strike selection can't be replayed
without them.

Resumable: contracts already in the output are skipped. Safe to interrupt and re-run.

Usage:
    python analysis/download_n50_candidate_intraday.py
    python analysis/download_n50_candidate_intraday.py --limit 200   # smoke run
    python analysis/download_n50_candidate_intraday.py --equity-only
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import socket
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))
LOCK = ROOT / "locks" / "prod_sell_strategies"
OUT_OPT = ROOT / "data" / "intraday" / "n50_candidate_marks.parquet"
OUT_EQ = ROOT / "data" / "intraday" / "n50_equity_5m.parquet"

V2 = "https://api.upstox.com/v2"
V3 = "https://api.upstox.com/v3/historical-candle"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")

RATE_SLEEP = 0.4          # ~2.5 req/s -- the sustained-quota pacing proven out by the NIFTY
                           # option-chain run; faster tripped 429s for hours
NET_RETRIES, RL_RETRIES = 4, 6
MAX_CHUNK_DAYS = 20
CUTOFF = "2024-10-01"     # Upstox expired-instrument availability
FWD_PAD_DAYS = 12         # carry past the last entry date so a 5-day hold's exits are covered

# what you could actually transact at, through the session
MARKS = ["09:15", "09:20", "09:30", "10:00", "11:00", "12:00",
         "13:00", "14:00", "14:30", "15:00", "15:15", "15:25"]


def _get(url: str, token: str) -> dict:
    req = Request(url, headers={"Accept": "application/json",
                                "Authorization": f"Bearer {token}", "User-Agent": UA})
    attempt = rl = 0
    while attempt < NET_RETRIES and rl < RL_RETRIES:
        try:
            with urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in (401, 403):
                # Upstox access tokens expire daily (~03:30 IST). Without this, an overnight run
                # that outlives its token would rip through every remaining contract, "succeed"
                # with almost everything marked unavailable, and bury the cause in the log.
                # Abort loudly instead -- the run is resumable, so a fresh token just continues.
                raise SystemExit(f"\n*** AUTH FAILED (HTTP {e.code}) -- the access token has most "
                                 f"likely expired.\n*** Refresh UPSTOX_ACCESS_TOKEN and re-run; "
                                 f"already-downloaded contracts are skipped.\n{body[:200]}")
            if e.code == 429:
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.isdigit() else min(3 * (2 ** rl), 60)
                rl += 1
                if rl < RL_RETRIES:
                    time.sleep(wait)
                    continue
            raise RuntimeError(f"HTTP {e.code}: {body[:180]}") from e
        except (URLError, ConnectionResetError, TimeoutError, socket.timeout, OSError) as e:
            attempt += 1
            if attempt < NET_RETRIES:
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"network: {e}") from e
    raise RuntimeError("unreachable")


def instrument_master() -> dict[str, str]:
    req = Request("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
                  headers={"User-Agent": UA})
    with urlopen(req, timeout=180) as r:
        raw = gzip.GzipFile(fileobj=io.BytesIO(r.read())).read()
    out = {}
    for i in json.loads(raw):
        if i.get("segment") == "NSE_EQ" and i.get("instrument_type") == "EQ":
            ts = i.get("trading_symbol") or i.get("tradingsymbol")
            if ts:
                out[ts] = i["instrument_key"]
    return out


def wanted_contracts() -> pd.DataFrame:
    """Every distinct option contract that appears as a leg of any NIFTY50 candidate, with the
    date window it needs to cover."""
    from koscine.liquid_universe import NIFTY50
    n50 = set(NIFTY50)
    frames = []
    bw = pd.read_parquet(LOCK / "broken_wing_candidates_raw.parquet")
    bw = bw[bw["symbol"].isin(n50) & (bw["entry_date"] >= CUTOFF)]
    for col, ot in [("short_ce", "CE"), ("long_ce", "CE"), ("short_pe", "PE"), ("long_pe", "PE")]:
        frames.append(bw[["symbol", "expiry", "entry_date"]].assign(strike=bw[col], opt_type=ot))
    sk = pd.read_parquet(LOCK / "skew_candidates_raw.parquet")
    sk = sk[sk["symbol"].isin(n50) & (sk["entry_date"] >= CUTOFF)]
    for col in ("short_strike", "long_strike"):
        frames.append(sk[["symbol", "expiry", "entry_date"]].assign(strike=sk[col], opt_type=sk["side"]))
    legs = pd.concat(frames, ignore_index=True)
    legs["strike"] = legs["strike"].astype(float)
    g = legs.groupby(["symbol", "expiry", "strike", "opt_type"])["entry_date"].agg(["min", "max"])
    return g.reset_index().rename(columns={"min": "d_from", "max": "d_to"})


def _chunks(d_from: date, d_to: date) -> list[tuple[date, date]]:
    out, cur = [], d_from
    while cur <= d_to:
        end = min(d_to, cur + timedelta(days=MAX_CHUNK_DAYS - 1))
        out.append((cur, end))
        cur = end + timedelta(days=1)
    return out


def to_marks(candles: list) -> pd.DataFrame:
    """1-min candles -> one row per (date, mark): the last print at or before that mark, plus
    volume traded up to it. An option that has not printed yet by a mark yields no row for it --
    absence is itself the signal that it was untradeable at that moment."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("timestamp")
    df["d"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hhmm"] = df["timestamp"].dt.strftime("%H:%M")
    rows = []
    for d, g in df.groupby("d", sort=True):
        g = g.sort_values("hhmm")
        cum = g["volume"].cumsum()
        for m in MARKS:
            sel = g["hhmm"] <= m
            if not sel.any():
                continue
            i = sel.values.nonzero()[0][-1]
            rows.append({"date": d, "mark": m, "price": float(g["close"].iloc[i]),
                         "cum_vol": float(cum.iloc[i]), "oi": float(g["oi"].iloc[i])})
    return pd.DataFrame(rows)


def download_options(token: str, limit: int | None) -> None:
    eq = instrument_master()
    want = wanted_contracts()
    print(f"[n50] {len(want):,} distinct contracts to fetch "
          f"({want['symbol'].nunique()} symbols, {want['expiry'].nunique()} expiries)", flush=True)

    done: set[tuple] = set()
    frames: list[pd.DataFrame] = []
    if OUT_OPT.exists():
        prev = pd.read_parquet(OUT_OPT)
        frames.append(prev)
        done = set(map(tuple, prev[["symbol", "expiry", "strike", "opt_type"]]
                       .drop_duplicates().astype(str).values))
        print(f"[n50] resuming -- {len(done):,} contracts already done", flush=True)

    want = want[~want.apply(lambda r: (r["symbol"], str(r["expiry"]), str(r["strike"]),
                                       r["opt_type"]) in done, axis=1)]
    if limit:
        want = want.head(limit)
    print(f"[n50] {len(want):,} remaining this run", flush=True)

    ckey: dict[tuple, dict] = {}

    def key_for(sym, exp, strike, ot):
        k = (sym, exp)
        if k not in ckey:
            ik = eq.get(sym)
            if not ik:
                ckey[k] = {}
            else:
                try:
                    r = _get(f"{V2}/expired-instruments/option/contract"
                             f"?instrument_key={quote(ik, safe='')}&expiry_date={exp}", token)
                    ckey[k] = {(float(c["strike_price"]), c["instrument_type"]): c["instrument_key"]
                               for c in (r.get("data") or [])}
                except RuntimeError as e:
                    print(f"    contract-list {sym} {exp}: {e}", flush=True)
                    ckey[k] = {}
                time.sleep(RATE_SLEEP)
        return ckey[k].get((float(strike), ot))

    t0, n_ok, n_miss, n_incomplete = time.time(), 0, 0, 0
    for i, (_, r) in enumerate(want.iterrows()):
        sym, exp = r["symbol"], str(r["expiry"])
        ik = key_for(sym, exp, r["strike"], r["opt_type"])
        if not ik:
            n_miss += 1
            continue
        d_from = date.fromisoformat(str(r["d_from"]))
        d_to = min(date.fromisoformat(str(r["d_to"])) + timedelta(days=FWD_PAD_DAYS),
                   date.fromisoformat(exp))
        got = []
        chunk_failed = False
        for a, b in _chunks(d_from, d_to):
            try:
                resp = _get(f"{V2}/expired-instruments/historical-candle/{quote(ik, safe='')}"
                            f"/1minute/{b.isoformat()}/{a.isoformat()}", token)
                got += (resp.get("data") or {}).get("candles") or []
            except RuntimeError as e:
                chunk_failed = True
                print(f"    candles {sym} {r['strike']}{r['opt_type']} {a}..{b}: {e}", flush=True)
            time.sleep(RATE_SLEEP)
        if chunk_failed:
            # A contract whose chunk failed (e.g. a 429 that outlived its retry budget) but whose
            # other chunk succeeded would otherwise be written with MISSING DAYS -- and since the
            # resume check keys on "is this contract present in the output", it would look
            # complete and never be retried. Silent gaps are worse than absence, so write nothing
            # and let a re-run fetch it whole. Same discipline as the NIFTY chain downloader's
            # failure-rate threshold.
            n_incomplete += 1
            continue
        m = to_marks(got)
        if m.empty:
            n_miss += 1
            continue
        m["symbol"], m["expiry"] = sym, exp
        m["strike"], m["opt_type"] = float(r["strike"]), r["opt_type"]
        frames.append(m)
        n_ok += 1
        if (i + 1) % 250 == 0:
            pd.concat(frames, ignore_index=True).to_parquet(OUT_OPT, index=False)
            el = (time.time() - t0) / 60
            rate = (i + 1) / el if el else 0
            eta = (len(want) - i - 1) / rate / 60 if rate else 0
            print(f"  [{i+1:,}/{len(want):,}] ok={n_ok:,} miss={n_miss:,} incomplete={n_incomplete:,} "
                  f"| {el:.0f}min elapsed, ETA {eta:.1f}h -> checkpoint", flush=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(OUT_OPT, index=False)
    print(f"[n50] options done: {n_ok:,} contracts, {n_miss:,} unavailable, "
          f"{n_incomplete:,} skipped as incomplete (re-run to fetch them whole) "
          f"-> {OUT_OPT}", flush=True)


def download_equity(token: str) -> None:
    """NIFTY50 equity 5-min bars -- needed to replay strike selection off the 14:30 underlying."""
    from koscine.liquid_universe import NIFTY50
    eq = instrument_master()
    start, end = date.fromisoformat(CUTOFF), date.today()
    frames = []
    done = set()
    if OUT_EQ.exists():
        prev = pd.read_parquet(OUT_EQ)
        frames.append(prev)
        done = set(prev["symbol"].unique())
        print(f"[eq] resuming -- {len(done)} symbols done", flush=True)
    todo = [s for s in NIFTY50 if s not in done]
    print(f"[eq] {len(todo)} symbols to fetch", flush=True)
    for i, sym in enumerate(todo):
        ik = eq.get(sym)
        if not ik:
            print(f"    no instrument_key for {sym}", flush=True)
            continue
        got = []
        for a, b in _chunks(start, end):
            try:
                resp = _get(f"{V3}/{quote(ik, safe='')}/minutes/5/"
                            f"{b.isoformat()}/{a.isoformat()}", token)
                got += (resp.get("data") or {}).get("candles") or []
            except RuntimeError as e:
                print(f"    {sym} {a}..{b}: {e}", flush=True)
            time.sleep(RATE_SLEEP)
        if not got:
            continue
        df = pd.DataFrame(got, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        df["symbol"] = sym
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(OUT_EQ, index=False)
        print(f"  [{i+1}/{len(todo)}] {sym}: {len(df):,} bars", flush=True)
    print(f"[eq] done -> {OUT_EQ}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--equity-only", action="store_true")
    ap.add_argument("--options-only", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not set")

    if not args.options_only:
        download_equity(token)
    if not args.equity_only:
        download_options(token, args.limit)


if __name__ == "__main__":
    main()
