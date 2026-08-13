"""Live 5-min NIFTY poller: every 5 minutes during market hours, fetches the current session's
spot 5-min OHLC bars (Upstox v3 intraday-candle -- gives real high/low, not just an LTP sample,
which the ATR/true-range price features need) and the live option chain (NSE primary, Upstox
fallback -- per explicit instruction), recomputes price features on the spot bars, scores the
persisted magnitude-regression model (locks/prod_nifty_intraday/magnitude_model.txt, h=6/30-min,
walk-forward Spearman IC 0.53 -- see train_magnitude.py's FINDINGS), and fires a signal when the
predicted move clears MOVE_THRESHOLD_PCT -- with a non-overlap cooldown (no new signal while one
is still running) and a PATTERN/ZONE-based exit, not a fixed-bar hold: see _check_exit()'s
docstring -- direction is inferred from what the market actually does after entry (the model
predicts magnitude, not direction), then the position is closed once it gives back a chunk of
its peak favorable move or reaches a support/resistance zone (koscine/nifty_zones.py, the SAME
zone engine already used for stocks -- see that module for how it's built off NIFTY's own daily
EOD history), with a 2hr/session-close hard backstop.

This is operational, not research -- lives under production/, not experiments/, per the repo's
EXPERIMENT_POLICY.md convention. You run this yourself (Task Scheduler 9:14->15:16, or manual
start) -- same "you run long-running/production things" boundary as every backtest this session.

Price features only (no option-chain features): train_magnitude.py's matched-window comparison
showed the option-chain-augmented model adds ~nothing over price-only (IC 0.4784 vs 0.4756 at
h6) while needing live option-chain data the scoring path would otherwise not depend on. The
option chain is still fetched (per your original ask, and for eventual live trade pricing/
display), just not fed into the model.

Outputs:
  data/intraday/nifty_5m_live.parquet          -- today's spot bars, appended each poll
  data/intraday/nifty_option_chain_live.parquet -- today's chain snapshots, appended each poll
                                                    (source column: "nse" or "upstox")
  locks/prod_nifty_intraday/signals_live.json  -- fired/resolved signals, today's session
  locks/prod_nifty_intraday/live_state.json    -- cooldown/open-window bookkeeping

Usage:
    python production/nifty_live_poller.py            # continuous loop, 9:15-15:15 IST weekdays
    python production/nifty_live_poller.py --once      # single fetch+score cycle, any time (dry run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "nifty_intraday_breakout_v1"))
import features as feat_mod  # noqa: E402
from pipeline.fetch import make_session, HEADERS  # noqa: E402
from koscine import nifty_zones  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 15)
POLL_SECONDS = 300

SPOT_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
NIFTY_SYMBOL = "NIFTY"

MOVE_THRESHOLD_PCT = 0.25    # fire only if predicted forward move >= this, per your instruction
DTE_MIN_LIVE = 2             # never quote/recommend off an expiry with <2 days left -- same
                              # "skip the dead zone right before expiry" rule the offline
                              # backtest (gen_sell_signal_history.py) uses, applied to live
                              # chain fetches so a weekly-expiry-day poll doesn't pick the
                              # about-to-be-worthless front contract
HORIZON_BARS = 6             # matches the finalized magnitude model (magnitude_config.json)

# Exit is pattern/zone-based, NOT a fixed-bar hold -- see _check_exit()'s docstring.
DIRECTION_LOCK_PCT = 0.05    # %% move needed to "lock in" a tracked direction (model predicts
                              # magnitude, not direction -- the market tells us which way after entry)
EROSION_GIVEBACK_FRAC = 0.5  # exit once this fraction of the peak favorable move has been given back
ZONE_BUFFER_PCT = 0.001      # treat "entering the zone band" loosely (0.1%) rather than exact
MAX_HOLD_BARS = 24           # 2hr hard backstop (matches labels.py's same-day horizons) -- a
                              # safety cap, not the primary exit trigger
SESSION_FORCE_EXIT_TIME = dtime(15, 25)   # never carry a signal past the session close

# Regime read on EVERY poll (not a "fired signal" -- consolidation is the default/background
# state, not an event; only the big-move case gets a discrete non-overlapping signal):
#   predicted move <  MOVE_THRESHOLD_PCT -> "consolidation": defined-risk premium SELLING is the
#     appropriate structure (reuse the already-proven Condor/Broken-Wing/Skew engine, adapted to
#     NIFTY -- see experiments/nifty_intraday_breakout_v1/eod_credit_spread.py from earlier this
#     session), since capped-risk selling wins on the majority no-big-move outcome.
#   predicted move >= MOVE_THRESHOLD_PCT -> "big_move_expected": buy a naked option in the
#     DIRECTION_HINT direction if given, or treat as a neutral (direction-agnostic) signal --
#     the direction hint is a cheap momentum heuristic, NOT a validated edge. This project's own
#     research (direction_edge_research memory, and repeated findings throughout this session)
#     shows short-horizon NIFTY/stock direction is close to a coin flip -- use it cautiously,
#     exactly as you framed it, not as a confident call.
MOMENTUM_LOOKBACK_BARS = 12   # ~1hr, for the direction hint

DATA_DIR = ROOT / "data" / "intraday"
LOCK_DIR = ROOT / "locks" / "prod_nifty_intraday"
SPOT_LIVE = DATA_DIR / "nifty_5m_live.parquet"
CHAIN_LIVE = DATA_DIR / "nifty_option_chain_live.parquet"
SIGNALS_LIVE = LOCK_DIR / "signals_live.json"
STATE_FILE = LOCK_DIR / "live_state.json"
REGIME_LIVE = LOCK_DIR / "regime_live.json"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


# --------------------------------------------------------------------------- market hours
def is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


# --------------------------------------------------------------------------- spot bars (Upstox)
def fetch_spot_bars_upstox(token: str) -> pd.DataFrame:
    """Today's 5-min OHLC bars so far, real high/low (not an LTP sample)."""
    import urllib.parse
    key = urllib.parse.quote(SPOT_INSTRUMENT_KEY, safe="")
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/5"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    candles = r.json()["data"]["candles"]
    if not candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Kolkata")   # string, not the
    # IST ZoneInfo object -- pandas treats zoneinfo- and pytz-backed tz columns as different
    # dtypes for concat purposes even when both say "Asia/Kolkata", which silently demotes any
    # later pd.concat() with the (pytz-backed, from parquet) historical data to dtype=object.
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- option chain (NSE primary)
def fetch_chain_nse(session: requests.Session, symbol: str = NIFTY_SYMBOL) -> pd.DataFrame | None:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:   # noqa: BLE001 -- any NSE failure falls back to Upstox, never crash the poll
        print(f"[nse] fetch failed: {e}")
        return None
    records = payload.get("records", {})
    rows = records.get("data", [])
    underlying = records.get("underlyingValue")
    ts = pd.Timestamp.now(tz="Asia/Kolkata").floor("5min")
    out = []
    for r in rows:
        expiry = r.get("expiryDate")
        strike = r.get("strikePrice")
        ce, pe = r.get("CE") or {}, r.get("PE") or {}
        out.append({
            "timestamp": ts, "expiry": pd.to_datetime(expiry, format="%d-%b-%Y", errors="coerce"),
            "strike": strike, "underlying_close": underlying,
            "ce_close": ce.get("lastPrice"), "ce_oi": ce.get("openInterest"), "ce_volume": ce.get("totalTradedVolume"),
            "pe_close": pe.get("lastPrice"), "pe_oi": pe.get("openInterest"), "pe_volume": pe.get("totalTradedVolume"),
            "source": "nse",
        })
    df = pd.DataFrame(out)
    if df.empty:
        return None
    # NSE returns EVERY expiry's strikes in one payload -- filter down to a single front expiry
    # (earliest that clears DTE_MIN_LIVE, same "skip the dead zone right before expiry" rule as
    # fetch_chain_upstox and the offline backtest) or nearest-strike lookups could silently match
    # a strike from the wrong expiry.
    today = datetime.now(IST).date()
    expiries = sorted(df["expiry"].dropna().unique())
    front = next((e for e in expiries if (pd.Timestamp(e).date() - today).days >= DTE_MIN_LIVE),
                 expiries[-1] if expiries else None)
    if front is None:
        return None
    return df[df["expiry"] == front].reset_index(drop=True)


def fetch_chain_upstox(token: str, symbol_underlying_key: str = "NSE_INDEX|Nifty 50") -> pd.DataFrame | None:
    """Fallback only -- fires when NSE fails. Lists live contracts + LTPs for the front expiry."""
    try:
        import urllib.parse
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": UA}
        key = urllib.parse.quote(symbol_underlying_key, safe="")
        r = requests.get(f"https://api.upstox.com/v2/option/contract?instrument_key={key}",
                         headers=headers, timeout=15)
        r.raise_for_status()
        contracts = r.json()["data"]
        if not contracts:
            return None
        today = datetime.now(IST).date()
        # earliest expiry that clears DTE_MIN_LIVE, not just the nearest one -- the nearest
        # expiry can be TODAY (weekly expiry day), which has ~zero time value in its last
        # minutes and would poison the credit-spread recommendation with a bogus negative
        # credit rather than a real one. Same "roll forward past the dead zone" pattern
        # analysis/gen_sell_signal_history.py already uses for the offline backtest.
        expiries = sorted({c["expiry"] for c in contracts})
        nearest = next((e for e in expiries
                        if (pd.Timestamp(e).date() - today).days >= DTE_MIN_LIVE), expiries[-1])
        front = [c for c in contracts if c["expiry"] == nearest]
        keys = [c["instrument_key"] for c in front]
        ltps = {}
        for i in range(0, len(keys), 500):   # LTP endpoint has a batch-size cap
            chunk = keys[i:i + 500]
            q = urllib.parse.quote(",".join(chunk), safe=",|")
            resp = requests.get(f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={q}",
                                headers=headers, timeout=15).json()
            for v in resp.get("data", {}).values():
                ltps[v["instrument_token"]] = v["last_price"]
        ts = pd.Timestamp.now(tz="Asia/Kolkata").floor("5min")
        by_strike: dict[float, dict] = {}
        for c in front:
            row = by_strike.setdefault(c["strike_price"], {"timestamp": ts, "expiry": nearest,
                                                            "strike": c["strike_price"], "source": "upstox"})
            side = "ce" if c["instrument_type"] == "CE" else "pe"
            row[f"{side}_close"] = ltps.get(c["instrument_key"])
        return pd.DataFrame(by_strike.values())
    except Exception as e:   # noqa: BLE001 -- both sources failed; caller treats None as "skip this poll"
        print(f"[upstox] chain fetch failed: {e}")
        return None


# --------------------------------------------------------------------------- state
def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"open_signal": None}


def _save_state(state: dict) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _load_signals() -> list[dict]:
    if SIGNALS_LIVE.exists():
        return json.loads(SIGNALS_LIVE.read_text(encoding="utf-8")).get("signals", [])
    return []


def _save_signals(signals: list[dict]) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_LIVE.write_text(json.dumps({"as_of": datetime.now(IST).isoformat(), "signals": signals},
                                       indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- regime + direction hint
def _direction_hint(bars: pd.DataFrame, lookback_bars: int = MOMENTUM_LOOKBACK_BARS) -> dict:
    """Cheap momentum-sign heuristic for the big-move regime's "which way" flag -- NOT a
    validated directional edge. This project has repeatedly found short-horizon NIFTY/stock
    direction to be close to a coin flip (see the direction_edge_research memory); this exists
    only because you explicitly asked for a direction hint "to be used cautiously", not because
    it's been backtested as predictive. Always ship it labeled as low-confidence."""
    same_day = bars[bars["timestamp"].dt.date == bars["timestamp"].iloc[-1].date()]
    if len(same_day) <= lookback_bars:
        return {"direction": None, "basis": "insufficient same-day bars"}
    ref = float(same_day.iloc[-lookback_bars - 1]["close"])
    now = float(same_day.iloc[-1]["close"])
    momentum_pct = (now - ref) / ref * 100
    direction = "up" if momentum_pct > 0 else "down" if momentum_pct < 0 else None
    return {"direction": direction, "momentum_pct": round(momentum_pct, 3),
            "basis": f"sign of {lookback_bars}-bar (~1hr) momentum", "confidence": "low"}


# NIFTY EOD Broken-Wing config from experiments/nifty_intraday_breakout_v1/eod_grid_search.py's
# best-found ("max median") structure: short ~0.75% OTM, asymmetric 2%/4% wings -- median
# Rs 3,760/lot in the historical backtest. Reused here for the LIVE consolidation-regime
# recommendation rather than re-deriving a new structure.
SELL_SHORT_OTM, SELL_WING_CE, SELL_WING_PE = 0.0075, 0.02, 0.04
NIFTY_LOT_SIZE = 65   # current lot as of Aug 2026 -- see eod_credit_spread.py


def _nearest_strike(chain: pd.DataFrame, target: float) -> pd.Series | None:
    if chain is None or chain.empty or "strike" not in chain.columns:
        return None
    valid = chain.dropna(subset=["strike"])
    if valid.empty:
        return None
    return valid.loc[(valid["strike"] - target).abs().idxmin()]


def recommend_sell_spread(chain: pd.DataFrame, spot: float) -> dict | None:
    """Broken-Wing credit spread off the LIVE chain (SELL_SHORT_OTM/WING_CE/WING_PE above) --
    for the 'consolidation' regime. Returns None (not a crash) if the chain can't support it
    this cycle -- a missing recommendation is always safer than a wrong one."""
    if chain is None or chain.empty:
        return None
    sc = _nearest_strike(chain, spot * (1 + SELL_SHORT_OTM))
    lc = _nearest_strike(chain, spot * (1 + SELL_SHORT_OTM + SELL_WING_CE))
    sp = _nearest_strike(chain, spot * (1 - SELL_SHORT_OTM))
    lp = _nearest_strike(chain, spot * (1 - SELL_SHORT_OTM - SELL_WING_PE))
    if any(x is None for x in (sc, lc, sp, lp)):
        return None
    sc_p, lc_p, sp_p, lp_p = sc.get("ce_close"), lc.get("ce_close"), sp.get("pe_close"), lp.get("pe_close")
    if any(v is None or pd.isna(v) for v in (sc_p, lc_p, sp_p, lp_p)):
        return None
    credit = (sc_p + sp_p) - (lc_p + lp_p)
    width = max(float(lc["strike"]) - float(sc["strike"]), float(sp["strike"]) - float(lp["strike"]))
    risk = width - credit
    if credit <= 0 or risk <= 0 or lc["strike"] <= sc["strike"] or lp["strike"] >= sp["strike"]:
        return None
    return {
        "structure": "broken_wing_credit_spread", "expiry": str(pd.Timestamp(chain["expiry"].iloc[0]).date()),
        "short_ce": float(sc["strike"]), "long_ce": float(lc["strike"]),
        "short_pe": float(sp["strike"]), "long_pe": float(lp["strike"]),
        "credit": round(credit, 2), "max_risk": round(risk, 2), "lot_size": NIFTY_LOT_SIZE,
        "max_profit_per_lot": round(credit * NIFTY_LOT_SIZE, 1),
        "max_risk_per_lot": round(risk * NIFTY_LOT_SIZE, 1),
        "entry_ror_pct": round(credit / risk * 100, 1),
    }


def recommend_buy(chain: pd.DataFrame, spot: float, direction: dict | None) -> dict | None:
    """Naked ATM CE or PE off the LIVE chain per direction_hint -- for the 'big_move_expected'
    regime. direction=None (or unresolved) -> no specific leg recommended; the caller still
    sees the predicted move and can treat it as a neutral, direction-agnostic signal."""
    if chain is None or chain.empty or direction is None or direction.get("direction") is None:
        return None
    side = "ce" if direction["direction"] == "up" else "pe"
    row = _nearest_strike(chain, spot)
    if row is None:
        return None
    premium = row.get(f"{side}_close")
    if premium is None or pd.isna(premium):
        return None
    return {"structure": "naked_option", "side": side.upper(), "strike": float(row["strike"]),
            "expiry": str(pd.Timestamp(chain["expiry"].iloc[0]).date()), "premium": round(float(premium), 2),
            "lot_size": NIFTY_LOT_SIZE, "premium_per_lot": round(float(premium) * NIFTY_LOT_SIZE, 1),
            "direction_confidence": "low"}


def _save_regime(regime: str, pred_move_pct: float, spot: float, ts: pd.Timestamp,
                 direction: dict | None, chain: pd.DataFrame | None) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"as_of": datetime.now(IST).isoformat(), "bar_ts": ts.isoformat(), "spot": spot,
               "predicted_move_pct": round(pred_move_pct, 3), "threshold_pct": MOVE_THRESHOLD_PCT,
               "regime": regime}
    if regime == "big_move_expected":
        payload["direction_hint"] = direction
        payload["note"] = ("Direction is a low-confidence momentum heuristic, not a validated "
                           "edge -- use cautiously, or treat as a neutral (direction-agnostic) signal.")
        payload["recommended_trade"] = recommend_buy(chain, spot, direction)
    else:
        payload["recommended_trade"] = recommend_sell_spread(chain, spot)
    REGIME_LIVE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- pattern/zone exit
def _check_exit(open_sig: dict, bars: pd.DataFrame, zones: dict) -> dict | None:
    """Pattern/zone-based exit -- NOT a fixed-bar hold. Mutates open_sig in place with the
    running direction/peak-price tracking state; returns an exit dict once triggered, else None.

    1. Direction is unknown at entry (the model predicts MAGNITUDE, not direction) -- once
       price has moved >=DIRECTION_LOCK_PCT away from entry in either direction, that becomes
       the tracked direction for the rest of this signal's life (the market tells us which way,
       we don't guess it at entry).
    2. Once a direction is locked, track the best (most favorable) price reached since entry.
       Exit once price has given back more than EROSION_GIVEBACK_FRAC of that peak favorable
       move (the move is stalling/reversing -- "erode our profits", per your framing) OR price
       has entered the nearest support/resistance zone band on the side the move is heading
       toward (koscine/nifty_zones.py -- the same zone engine used for stocks; zones are where
       a move statistically becomes more likely to stall or reverse, not just an arbitrary %).
    3. MAX_HOLD_BARS (2hr) and the session close are hard backstops, not the primary trigger --
       a signal that never resolves a clear direction or reaches a zone still eventually closes
       rather than running forever or carrying overnight risk.
    """
    entry_ts = pd.Timestamp(open_sig["entry_ts"])
    since = bars[bars["timestamp"] > entry_ts].reset_index(drop=True)
    if since.empty:
        return None
    entry_price = open_sig["entry_underlying"]
    latest = since.iloc[-1]
    current = float(latest["close"])
    bars_elapsed = len(since)

    direction = open_sig.get("direction")
    peak_price = open_sig.get("peak_price") or entry_price
    if direction is None:
        move_pct = (current - entry_price) / entry_price * 100
        if abs(move_pct) >= DIRECTION_LOCK_PCT:
            direction = 1 if move_pct > 0 else -1
            peak_price = current
    elif direction > 0:
        peak_price = max(peak_price, current)
    else:
        peak_price = min(peak_price, current)
    open_sig["direction"], open_sig["peak_price"] = direction, peak_price

    exit_reason = None
    if direction is not None:
        peak_move = (peak_price - entry_price) * direction     # positive once locked
        giveback = (peak_price - current) * direction           # positive = giving gains back
        if peak_move > 0 and giveback >= EROSION_GIVEBACK_FRAC * peak_move:
            exit_reason = "erosion"
        elif direction > 0 and zones.get("resistance_level") and \
                current >= zones["resistance_level"] * (1 - ZONE_BUFFER_PCT):
            exit_reason = "resistance_zone"
        elif direction < 0 and zones.get("support_level") and \
                current <= zones["support_level"] * (1 + ZONE_BUFFER_PCT):
            exit_reason = "support_zone"

    if exit_reason is None and latest["timestamp"].time() >= SESSION_FORCE_EXIT_TIME:
        exit_reason = "session_close"
    if exit_reason is None and bars_elapsed >= MAX_HOLD_BARS:
        exit_reason = "max_hold"
    if exit_reason is None:
        return None

    realized_pct = (current - entry_price) / entry_price * 100
    return {"exit_ts": latest["timestamp"].isoformat(), "exit_underlying": current,
            "realized_move_pct": round(realized_pct, 3), "exit_reason": exit_reason}


def _with_prior_session_context(bars: pd.DataFrame, lookback_bars: int = 100) -> pd.DataFrame:
    """features.py's feature_gap_from_prev_close needs the PRIOR trading day's close (via
    day_summary.shift(1) across the `date` groupby) -- a single day of live bars alone can
    never produce it (nothing to shift from), so every live score would silently NaN out on
    that one feature otherwise. Prepends the tail of the existing offline historical dataset
    (already downloaded, no new fetch) so the feature builder sees at least one real prior
    session before today's live bars, then drops that prepended context before scoring --
    only used to make the rolling/day-boundary features correct for TODAY's last row."""
    hist_path = DATA_DIR / "nifty_5m.parquet"
    if not hist_path.exists():
        return bars
    hist = pd.read_parquet(hist_path)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist[hist["timestamp"] < bars["timestamp"].min()].tail(lookback_bars)
    if hist.empty:
        return bars
    return pd.concat([hist, bars], ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- score + fire
def score_latest(bars: pd.DataFrame, booster, feature_cols: list[str]) -> tuple[pd.Timestamp, float] | None:
    if len(bars) < 25:   # need enough today-bars for the longest same-day rolling window (atr_20bar)
        return None
    scoring_buffer = _with_prior_session_context(bars)
    feats = feat_mod.build_price_features(scoring_buffer)
    feats = feats[feats["timestamp"].isin(bars["timestamp"])]   # score only today's own bars
    # .iloc[[-1]] (list index), NOT .iloc[-1] -- the latter collapses the row to a Series,
    # which pandas gives a single dtype (object) once the source columns are heterogeneous,
    # and LightGBM's predict() rejects object-dtype input outright.
    last_row = feats.iloc[[-1]]
    if last_row[feature_cols].isna().any(axis=None):
        return None
    pred = float(booster.predict(last_row[feature_cols])[0])
    return last_row["timestamp"].iloc[0], pred


def _beep_alert(pattern: list[tuple[int, int]]) -> None:
    """Audible system alert -- winsound.Beep plays through the system's default audio output
    regardless of which window has focus (this repo always runs on Windows; a terminal bell
    character wouldn't reliably sound if the console isn't focused). `pattern` is a list of
    (frequency_hz, duration_ms) tuples played in sequence. Never let a beep failure (e.g. no
    audio device) crash the poll loop -- an alert is a nice-to-have, not the poller's job."""
    try:
        import winsound
        for freq, dur in pattern:
            winsound.Beep(freq, dur)
    except Exception as e:  # noqa: BLE001
        print(f"[poll] beep alert failed: {e}")


# Entry (new signal fired): higher-pitched, 3 beeps -- the actionable moment, most attention-
# grabbing. Exit (signal resolved): lower-pitched, single beep -- informational, less urgent.
_BEEP_ENTRY = [(1200, 250), (1200, 250), (1200, 250)]
_BEEP_EXIT = [(700, 300)]


def poll_once() -> None:
    import os
    import lightgbm as lgb

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not set")

    booster = lgb.Booster(model_file=str(LOCK_DIR / "magnitude_model.txt"))
    cfg = json.loads((LOCK_DIR / "magnitude_config.json").read_text(encoding="utf-8"))
    feature_cols = cfg["features"]

    bars = fetch_spot_bars_upstox(token)
    if bars.empty:
        print("[poll] no spot bars returned, skipping")
        return
    bars.to_parquet(SPOT_LIVE, index=False)

    session = make_session()
    chain = fetch_chain_nse(session)
    if chain is None:
        print("[poll] NSE chain fetch failed, falling back to Upstox")
        chain = fetch_chain_upstox(token)
    if chain is not None:
        chain.to_parquet(CHAIN_LIVE, index=False)
    else:
        print("[poll] both NSE and Upstox chain fetch failed this cycle -- spot/scoring still proceeds")

    result = score_latest(bars, booster, feature_cols)
    if result is None:
        print("[poll] not enough bar history yet to score")
        return
    ts, pred_move_pct = result
    pred_move_pct *= 100
    spot = float(bars.iloc[-1]["close"])
    print(f"[poll] {ts} spot={spot:.1f} predicted_move={pred_move_pct:.3f}%")

    # regime read every poll (not a fired signal -- see REGIME_LIVE's comment above)
    if pred_move_pct >= MOVE_THRESHOLD_PCT:
        direction = _direction_hint(bars)
        _save_regime("big_move_expected", pred_move_pct, spot, ts, direction, chain)
        print(f"[poll] regime=big_move_expected direction_hint={direction.get('direction')} "
              f"(low confidence, {direction.get('basis')})")
    else:
        _save_regime("consolidation", pred_move_pct, spot, ts, None, chain)

    state = _load_state()
    signals = _load_signals()
    open_sig = state.get("open_signal")

    # resolve an open signal via pattern/zone-based exit (not a fixed-bar hold) -- see _check_exit
    if open_sig is not None:
        zones = nifty_zones.latest_zones()
        exit_info = _check_exit(open_sig, bars, zones)
        if exit_info is not None:
            for s in signals:
                if s["entry_ts"] == open_sig["entry_ts"]:
                    s.update(exit_info)
                    break
            state["open_signal"] = None
            print(f"[poll] resolved signal from {open_sig['entry_ts']}: "
                  f"realized {exit_info['realized_move_pct']:.3f}% ({exit_info['exit_reason']})")
            _beep_alert(_BEEP_EXIT)
        else:
            state["open_signal"] = open_sig   # persist updated direction/peak_price tracking

    # fire a new signal: predicted move clears the threshold AND nothing currently open
    # (non-overlap -- no new signal while one is still running, per your instruction)
    if state.get("open_signal") is None and pred_move_pct >= MOVE_THRESHOLD_PCT:
        new_sig = {"entry_ts": ts.isoformat(), "entry_underlying": spot,
                   "predicted_move_pct": round(pred_move_pct, 3), "exit_ts": None,
                   "exit_underlying": None, "realized_move_pct": None, "exit_reason": None,
                   "direction": None, "peak_price": None}
        signals.append(new_sig)
        state["open_signal"] = new_sig
        print(f"[poll] FIRED: predicted {pred_move_pct:.3f}% >= {MOVE_THRESHOLD_PCT}% threshold")
        _beep_alert(_BEEP_ENTRY)

    _save_signals(signals)   # always, even unchanged -- gives a live "as_of" heartbeat every
                              # poll rather than only writing the file on a firing/resolving day
    _save_state(state)


def run_loop() -> None:
    print(f"[nifty_live_poller] starting, polling every {POLL_SECONDS}s during "
          f"{MARKET_OPEN}-{MARKET_CLOSE} IST weekdays", flush=True)
    while True:
        now = datetime.now(IST)
        if is_market_open(now):
            try:
                poll_once()
            except Exception as e:   # noqa: BLE001 -- one bad poll must not kill the whole session
                print(f"[poll] ERROR: {e}")
            time.sleep(POLL_SECONDS)
        else:
            print(f"[nifty_live_poller] market closed ({now.time()}), sleeping 60s")
            time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single fetch+score cycle, any time (dry run)")
    args = ap.parse_args()
    if args.once:
        poll_once()
    else:
        run_loop()
