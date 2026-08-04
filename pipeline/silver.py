"""
Silver layer parsers
====================
Parse raw NSE files into normalised, type-clean parquet tables.

Tables produced (long format):
  eod_stock      : (date, symbol, open, high, low, close, last, prev_close,
                    avg_price, volume, turnover, n_trades, deliv_qty, deliv_pct)
  eod_deriv      : (date, symbol, instrument, expiry, strike, opt_type,
                    open, high, low, close, settle, contracts, val_lakh,
                    open_int, chg_in_oi)
  participant_oi : (date, participant, fut_idx_long, fut_idx_short,
                    fut_stk_long, fut_stk_short, opt_idx_call_long,
                    opt_idx_put_long, opt_idx_call_short, opt_idx_put_short,
                    opt_stk_call_long, opt_stk_put_long, opt_stk_call_short,
                    opt_stk_put_short, total_long, total_short)
  indices        : (date, index_name, open, high, low, close, points_chg,
                    pct_chg, volume, turnover, pe, pb, div_yield)
  fii_dii_cash   : (date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net)
  corp_actions   : (date, symbol, action_type, ex_date, record_date, ratio_text)
  earnings       : (date, symbol, event_type, period)

Idempotent: re-running just upserts new days.

Usage:
  python -m pipeline.silver           # parse all feeds, all years
  python -m pipeline.silver eod_stock # parse only one feed
"""
import sys, json, zipfile, io
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

from .config import (RAW_DIR, SILVER_DIR, SILVER_TABLES,
                     UDIFF_CUTOVER, CASH_NEW_FROM)

SILVER_DIR.mkdir(parents=True, exist_ok=True)

# ── Black-Scholes IV helpers ────────────────────────────────────────────────
_BS_R_FREE = 0.065  # India approximate risk-free rate (91-day T-bill)


def _bs_price(S: float, K: float, T: float, r: float,
              sigma: float, is_call: bool) -> float:
    """Black-Scholes option price (call or put)."""
    from math import log, sqrt, exp
    from scipy.special import ndtr
    if T <= 0 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = exp(-r * T)
    if is_call:
        return S * ndtr(d1) - K * disc * ndtr(d2)
    return K * disc * ndtr(-d2) - S * ndtr(-d1)


def _solve_iv(price: float, S: float, K: float, T: float,
              r: float, is_call: bool) -> float:
    """Invert Black-Scholes for IV via Brent's method. Returns NaN on failure."""
    from scipy.optimize import brentq
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic * 0.99:        # below intrinsic → stale quote
        return float("nan")
    try:
        return brentq(
            lambda s: _bs_price(S, K, T, r, s, is_call) - price,
            1e-4, 5.0, xtol=1e-5, maxiter=50,
        )
    except Exception:
        return float("nan")


def _compute_atm_iv(ce: pd.DataFrame, pe: pd.DataFrame,
                    spot: float, trade_date, expiry) -> dict:
    """
    ATM implied vol for one symbol's near-month options chain.

    Returns dict: atm_ce_iv, atm_pe_iv, atm_iv (mean), put_call_iv_skew (PE-CE).
    All values are annualised decimal vol (0.30 = 30%).  NaN when unavailable.
    """
    nan4 = {"atm_ce_iv": np.nan, "atm_pe_iv": np.nan,
            "atm_iv": np.nan, "put_call_iv_skew": np.nan}

    T = (pd.Timestamp(expiry) - pd.Timestamp(trade_date)).days / 365.0
    if T <= 0 or spot <= 0 or np.isnan(spot):
        return nan4

    r = _BS_R_FREE
    # restrict to contracts with actual volume (live quotes, not stale)
    ce_v = ce[ce["contracts"].fillna(0) > 0]
    pe_v = pe[pe["contracts"].fillna(0) > 0]

    active_strikes = sorted(
        set(ce_v["strike"].dropna()) | set(pe_v["strike"].dropna()),
        key=lambda k: abs(k - spot),
    )
    if not active_strikes:
        return nan4

    atm_k = float(active_strikes[0])
    ce_iv = pe_iv = float("nan")

    for side, grp, is_call in [("CE", ce_v, True), ("PE", pe_v, False)]:
        row = grp[grp["strike"] == atm_k]
        if row.empty:
            continue
        px = float(row["settle"].iloc[0])
        if np.isnan(px) or px <= 0:
            px = float(row["close"].iloc[0])
        if px > 0 and not np.isnan(px):
            iv = _solve_iv(px, spot, atm_k, T, r, is_call)
            if is_call:
                ce_iv = iv
            else:
                pe_iv = iv

    ivs = [v for v in [ce_iv, pe_iv] if not np.isnan(v)]
    atm_iv  = float(np.mean(ivs)) if ivs else float("nan")
    skew    = (pe_iv - ce_iv) if not (np.isnan(pe_iv) or np.isnan(ce_iv)) else float("nan")

    return {"atm_ce_iv": ce_iv, "atm_pe_iv": pe_iv,
            "atm_iv": atm_iv, "put_call_iv_skew": skew}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _strip_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _date_from_ddmmyyyy(s):
    return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")

def _date_from_ddmmmyyyy(s):
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")

def _safe_read_csv(path, **kw):
    # Some NSE files are XLSX (ZIP) despite a .csv extension — detect by magic bytes
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"PK":
        try:
            # strip encoding kwarg — read_excel doesn't accept it
            kw.pop("encoding", None)
            return pd.read_excel(path, **kw)
        except Exception as e:
            print(f"  [warn] could not read {path.name} as Excel: {e}")
            return None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [warn] could not read {path.name}: {e}")
            return None
    print(f"  [warn] {path.name}: no encoding worked")
    return None


# ----------------------------------------------------------------------
# 1. CASH BHAVCOPY
# ----------------------------------------------------------------------
def parse_cash_modern(fp: Path) -> pd.DataFrame:
    """sec_bhavdata_full_DDMMYYYY.csv (2020+)."""
    df = _safe_read_csv(fp)
    if df is None or len(df) == 0: return None
    df = _strip_cols(df)
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    df["date"]   = _date_from_ddmmmyyyy(df["DATE1"].str.strip())
    out = pd.DataFrame({
        "date":       df["date"],
        "symbol":     df["SYMBOL"].astype(str).str.strip().str.upper(),
        "open":       _to_num(df["OPEN_PRICE"]),
        "high":       _to_num(df["HIGH_PRICE"]),
        "low":        _to_num(df["LOW_PRICE"]),
        "close":      _to_num(df["CLOSE_PRICE"]),
        "last":       _to_num(df["LAST_PRICE"]),
        "prev_close": _to_num(df["PREV_CLOSE"]),
        "avg_price":  _to_num(df["AVG_PRICE"]),
        "volume":     _to_num(df["TTL_TRD_QNTY"]),
        "turnover":   _to_num(df["TURNOVER_LACS"]),
        "n_trades":   _to_num(df["NO_OF_TRADES"]),
        "deliv_qty":  _to_num(df["DELIV_QTY"]),
        "deliv_pct":  _to_num(df["DELIV_PER"]),
    })
    return out.dropna(subset=["date","symbol","close"])


def parse_cash_legacy(fp_ohlc: Path, fp_mto: Path|None) -> pd.DataFrame:
    """cmDDMMMYYYYbhav.csv.zip (OHLC) + MTO_DDMMYYYY.DAT (delivery)."""
    try:
        with zipfile.ZipFile(fp_ohlc) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                df = pd.read_csv(f)
    except Exception as e:
        print(f"  [warn] cm zip {fp_ohlc.name}: {e}"); return None
    df = _strip_cols(df)
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    df["date"] = _date_from_ddmmmyyyy(df["TIMESTAMP"].astype(str).str.strip())
    out = pd.DataFrame({
        "date":       df["date"],
        "symbol":     df["SYMBOL"].astype(str).str.strip().str.upper(),
        "open":       _to_num(df["OPEN"]),
        "high":       _to_num(df["HIGH"]),
        "low":        _to_num(df["LOW"]),
        "close":      _to_num(df["CLOSE"]),
        "last":       _to_num(df["LAST"]),
        "prev_close": _to_num(df["PREVCLOSE"]),
        "avg_price":  np.nan,
        "volume":     _to_num(df["TOTTRDQTY"]),
        "turnover":   _to_num(df["TOTTRDVAL"]) / 1e5,  # to lakhs
        "n_trades":   _to_num(df.get("TOTALTRADES", np.nan)),
        "deliv_qty":  np.nan,
        "deliv_pct":  np.nan,
    })
    # join MTO delivery
    if fp_mto is not None and fp_mto.exists():
        try:
            mto = pd.read_csv(fp_mto, skiprows=4, header=None,
                              names=["rec_type","sl","SYMBOL","SERIES",
                                     "deliv_qty","traded_qty","deliv_pct"])
            mto = mto[mto["SERIES"].astype(str).str.strip()=="EQ"]
            mto["symbol"] = mto["SYMBOL"].astype(str).str.strip().str.upper()
            mto["deliv_qty"] = _to_num(mto["deliv_qty"])
            mto["deliv_pct"] = _to_num(mto["deliv_pct"])
            out = out.drop(columns=["deliv_qty","deliv_pct"]).merge(
                mto[["symbol","deliv_qty","deliv_pct"]], on="symbol", how="left")
        except Exception as e:
            print(f"  [warn] MTO {fp_mto.name}: {e}")
    return out.dropna(subset=["date","symbol","close"])


def _split_adjust_eod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-adjust OHLC prices for stock splits recorded in corp_actions.

    Derives the adjustment factor empirically from the actual close-price ratio
    on the ex-date vs the last pre-ex-date close.  This is more reliable than
    the 'amount' field because some events combine a split with a bonus issue,
    making the real price discontinuity larger than the stated ratio.

    Logic
    -----
    For each (symbol, ex_date) split event:
      factor = close[ex_date] / close[ex_date - 1]   (always < 1 for splits)
      All closes before ex_date are multiplied by factor  → series becomes continuous.
      Volume/n_trades are divided by factor              → share count adjusts inversely.

    A factor > 0.75 (< 25% price drop) is ignored — that's normal market noise,
    not a split.  corp_actions is used purely to identify candidate dates; the
    factor is always derived from the price data.
    """
    ca_path = SILVER_TABLES["corp_actions"]
    if not ca_path.exists():
        return df  # corp_actions not built yet — skip silently

    splits = pd.read_parquet(ca_path, columns=["symbol", "ex_date", "action_type"])
    splits = splits[splits["action_type"] == "SPLIT"].copy()
    splits["ex_date"] = pd.to_datetime(splits["ex_date"]).dt.normalize()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    price_cols = [c for c in ("open", "high", "low", "close", "last")
                  if c in df.columns]  # exclude prev_close — it comes verbatim from NSE
    vol_cols   = [c for c in ("volume", "n_trades") if c in df.columns]

    n_applied = 0
    for sym, sym_splits in splits.groupby("symbol"):
        sym_mask = df["symbol"] == sym
        if not sym_mask.any():
            continue
        sym_df = df.loc[sym_mask].sort_values("date")

        for ex_date in sorted(sym_splits["ex_date"].unique()):
            before      = sym_df[sym_df["date"] <  ex_date]
            on_or_after = sym_df[sym_df["date"] >= ex_date]
            if before.empty or on_or_after.empty:
                continue

            pre_close = float(before.iloc[-1]["close"])
            ex_close  = float(on_or_after.iloc[0]["close"])
            if pre_close <= 0 or ex_close <= 0:
                continue

            factor = ex_close / pre_close
            if factor >= 1.0:                       # price went up — not a split
                continue

            pre_idx = before.index
            for col in price_cols:
                df.loc[pre_idx, col] = (df.loc[pre_idx, col] * factor).round(4)
            for col in vol_cols:
                # Use float division; avoid int64 cast which crashes on NaN volumes
                df.loc[pre_idx, col] = (df.loc[pre_idx, col] / factor).round(0)
            n_applied += 1

    print(f"  [split-adj] {n_applied} split events applied across "
          f"{splits['symbol'].nunique()} symbols")
    return df


def patch_splits_from_yf(run_date: str | None = None) -> list[str]:
    """
    Incremental split adjustment for daily runs.

    Scans raw/corp_actions_yf/yf_*.json for split events within the last
    14 calendar days of run_date.  For each one found, checks whether the
    price series in eod_stock.parquet still shows a discontinuity (factor<1).
    If it does, the split has not yet been applied by the NSE-based
    _split_adjust_eod and we patch it here.

    Idempotency: the empirical factor check is the guard — once prices are
    continuous the factor ≈ 1.0 and the symbol is skipped automatically.
    No separate tracking file is needed.

    Returns list of symbols that were adjusted.
    """
    eod_path = SILVER_TABLES["eod_stock"]
    if not eod_path.exists():
        return []

    yf_dir = RAW_DIR / "corp_actions_yf"
    if not yf_dir.exists():
        return []

    ref_date     = pd.Timestamp(run_date) if run_date else pd.Timestamp.today()
    window_start = ref_date - pd.Timedelta(days=14)

    # collect split events in the window from all YF action files
    candidate_splits: list[tuple[str, pd.Timestamp]] = []
    for fpath in yf_dir.glob("yf_*.json"):
        sym = fpath.stem[3:]   # strip "yf_"
        try:
            records = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in records:
            split_val = rec.get("Stock Splits") or 0
            if split_val <= 0:
                continue
            try:
                ex_dt = pd.Timestamp(rec["Date"]).tz_localize(None).normalize()
            except Exception:
                continue
            if window_start <= ex_dt <= ref_date:
                candidate_splits.append((sym, ex_dt))

    if not candidate_splits:
        print("  [split-patch] no new YF splits in last 14 days")
        return []

    print(f"  [split-patch] {len(candidate_splits)} candidate split(s) in window: "
          f"{[(s, str(d.date())) for s, d in candidate_splits]}")

    eod        = pd.read_parquet(eod_path)
    eod["date"] = pd.to_datetime(eod["date"]).dt.normalize()
    price_cols  = [c for c in ("open", "high", "low", "close", "last") if c in eod.columns]
    vol_cols    = [c for c in ("volume", "n_trades") if c in eod.columns]

    adjusted: list[str] = []
    for sym, ex_date in sorted(candidate_splits, key=lambda x: x[1]):
        sym_mask = eod["symbol"] == sym
        if not sym_mask.any():
            continue

        sym_df      = eod.loc[sym_mask].sort_values("date")
        before      = sym_df[sym_df["date"] <  ex_date]
        on_or_after = sym_df[sym_df["date"] >= ex_date]
        if before.empty or on_or_after.empty:
            continue

        pre_close = float(before.iloc[-1]["close"])
        ex_close  = float(on_or_after.iloc[0]["close"])
        if pre_close <= 0 or ex_close <= 0:
            continue

        factor = ex_close / pre_close
        if factor >= 1.0:
            print(f"    [skip] {sym} {ex_date.date()}: factor={factor:.4f} (price up, not a split)")
            continue
        # near-parity means already adjusted by NSE path — threshold 0.98 avoids
        # re-adjusting for a ~1% dividend that happened to coincide with the window
        if factor > 0.98:
            print(f"    [skip] {sym} {ex_date.date()}: factor={factor:.4f} (already adjusted)")
            continue

        pre_idx = before.index
        for col in price_cols:
            eod.loc[pre_idx, col] = (eod.loc[pre_idx, col] * factor).round(4)
        for col in vol_cols:
            eod.loc[pre_idx, col] = (
                (eod.loc[pre_idx, col] / factor).round(0).astype("int64")
            )
        adjusted.append(sym)
        print(f"    [adj] {sym} {ex_date.date()}  factor={factor:.4f}  "
              f"rows_adjusted={len(pre_idx)}")

    if adjusted:
        eod.to_parquet(eod_path)
        print(f"  [split-patch] eod_stock updated for: {adjusted}")

    return adjusted


def build_eod_stock() -> pd.DataFrame:
    print("[silver] building eod_stock ...")
    frames = []
    # modern
    for f in sorted((RAW_DIR / "cash_bhavcopy").rglob("sec_bhavdata_full_*.csv")):
        df = parse_cash_modern(f)
        if df is not None: frames.append(df)
    # legacy
    legacy_dir = RAW_DIR / "cash_bhavcopy_legacy_ohlc"
    if legacy_dir.exists():
        for f in sorted(legacy_dir.rglob("cm*bhav.csv.zip")):
            # find matching MTO file
            try:
                d = dt.datetime.strptime(f.stem.replace("bhav.csv","")[2:],
                                         "%d%b%Y").date()
                mto = (RAW_DIR / "cash_bhavcopy_legacy_delivery" /
                       str(d.year) / f"MTO_{d.strftime('%d%m%Y')}.DAT")
            except Exception:
                mto = None
            df = parse_cash_legacy(f, mto)
            if df is not None: frames.append(df)
    if not frames:
        print("  [empty] no cash files found"); return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["date","symbol"])
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    df = _split_adjust_eod(df)
    df.to_parquet(SILVER_TABLES["eod_stock"])
    print(f"  rows={len(df):,}  uniq_symbols={df['symbol'].nunique()}  "
          f"days={df['date'].nunique()}")
    return df


# ----------------------------------------------------------------------
# 2. DERIVATIVES BHAVCOPY (legacy + UDIFF)
# ----------------------------------------------------------------------
def parse_deriv_legacy(fp: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(fp) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                df = pd.read_csv(f)
    except Exception as e:
        print(f"  [warn] deriv zip {fp.name}: {e}"); return None
    df = _strip_cols(df)
    df["date"]   = _date_from_ddmmmyyyy(df["TIMESTAMP"].astype(str).str.strip())
    df["expiry"] = _date_from_ddmmmyyyy(df["EXPIRY_DT"].astype(str).str.strip())
    out = pd.DataFrame({
        "date":       df["date"],
        "symbol":     df["SYMBOL"].astype(str).str.strip().str.upper(),
        "instrument": df["INSTRUMENT"].astype(str).str.strip(),  # FUTIDX/FUTSTK/OPTIDX/OPTSTK
        "expiry":     df["expiry"],
        "strike":     _to_num(df["STRIKE_PR"]),
        "opt_type":   df["OPTION_TYP"].astype(str).str.strip(),  # CE/PE/XX
        "open":             _to_num(df["OPEN"]),
        "high":             _to_num(df["HIGH"]),
        "low":              _to_num(df["LOW"]),
        "close":            _to_num(df["CLOSE"]),
        "settle":           _to_num(df["SETTLE_PR"]),
        "contracts":        _to_num(df["CONTRACTS"]),
        "val_lakh":         _to_num(df["VAL_INLAKH"]),
        "open_int":         _to_num(df["OPEN_INT"]),
        "chg_in_oi":        _to_num(df["CHG_IN_OI"]),
        "underlying_price": np.nan,   # not in legacy format
    })
    return out.dropna(subset=["date","symbol"])


def parse_deriv_udiff(fp: Path) -> pd.DataFrame:
    """UDIFF format (post 2024-07-08). Handles both .csv and .csv.zip."""
    try:
        if fp.suffix.lower() == ".zip":
            with zipfile.ZipFile(fp) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                with z.open(inner) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(fp)
    except Exception as e:
        print(f"  [warn] deriv UDIFF {fp.name}: {e}"); return None
    df = _strip_cols(df)
    inst_map = {"STO":"OPTSTK","IDO":"OPTIDX","STF":"FUTSTK","IDF":"FUTIDX"}
    out = pd.DataFrame({
        "date":       pd.to_datetime(df["TradDt"], errors="coerce"),
        "symbol":     df["TckrSymb"].astype(str).str.strip().str.upper(),
        "instrument": df["FinInstrmTp"].map(inst_map).fillna(df["FinInstrmTp"]),
        "expiry":     pd.to_datetime(df["XpryDt"], errors="coerce"),
        "strike":     _to_num(df["StrkPric"]),
        "opt_type":   df["OptnTp"].astype(str).str.strip(),
        "open":             _to_num(df["OpnPric"]),
        "high":             _to_num(df["HghPric"]),
        "low":              _to_num(df["LwPric"]),
        "close":            _to_num(df["ClsPric"]),
        "settle":           _to_num(df["SttlmPric"]),
        "contracts":        _to_num(df["TtlTradgVol"]),
        "val_lakh":         _to_num(df["TtlTrfVal"]) / 1e5,
        "open_int":         _to_num(df["OpnIntrst"]),
        "chg_in_oi":        _to_num(df["ChngInOpnIntrst"]),
        "underlying_price": _to_num(df["UndrlygPric"]),   # UDIFF carries spot price
    })
    return out.dropna(subset=["date","symbol"])


# --- per-day aggregator: turns one bhavcopy day into (date, symbol) rows ----
def _aggregate_one_day(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a single day's raw F&O rows into (date, symbol) aggregates.
    Futures: nearest-expiry row per symbol.
    Options: per-expiry chain aggregates (total/call/put OI+vol, max-pain, walls)
             taken from the near-month expiry only to keep it bounded."""
    if df is None or df.empty: return None
    out_rows = []
    # nearest expiry per (symbol, instrument)
    df = df.copy()
    df["is_fut"] = df["instrument"].isin(["FUTIDX","FUTSTK"])
    df["is_opt"] = df["instrument"].isin(["OPTIDX","OPTSTK"])

    # --- FUTURES: pick nearest non-expired expiry per symbol -------------
    fut = df[df["is_fut"]]
    if not fut.empty:
        fut = fut.sort_values(["symbol","expiry"])
        near_fut = fut.groupby("symbol", as_index=False).first()
    else:
        near_fut = pd.DataFrame(columns=["symbol"])

    # pre-build futures settle lookup for IV spot fallback
    _fut_settle = {}
    if not near_fut.empty and "settle" in near_fut.columns:
        for _, _r in near_fut.iterrows():
            _s = _r.get("settle", float("nan"))
            _fut_settle[_r["symbol"]] = float(_s) if pd.notna(_s) and _s > 0 else float("nan")

    # --- OPTIONS: near-month chain aggregates per symbol -----------------
    opt = df[df["is_opt"] & df["strike"].notna()]
    opt_rows = []
    if not opt.empty:
        # near-month expiry per symbol
        opt = opt.sort_values(["symbol","expiry"])
        near_exp = opt.groupby("symbol")["expiry"].min().rename("near_exp")
        opt = opt.merge(near_exp, on="symbol")
        nm = opt[opt["expiry"] == opt["near_exp"]]
        for sym, g in nm.groupby("symbol", sort=False):
            ce = g[g["opt_type"].str.upper() == "CE"]
            pe = g[g["opt_type"].str.upper() == "PE"]
            call_oi = ce.groupby("strike")["open_int"].sum()
            put_oi  = pe.groupby("strike")["open_int"].sum()
            strikes = np.array(sorted(set(call_oi.index) | set(put_oi.index)))
            co = call_oi.reindex(strikes, fill_value=0).values.astype("float64")
            po = put_oi.reindex(strikes, fill_value=0).values.astype("float64")
            # vectorised max-pain: for each candidate K, sum writer losses
            if len(strikes) >= 2:
                diff_c = np.maximum(strikes[:, None] - strikes[None, :], 0)  # K - Ki
                diff_p = np.maximum(strikes[None, :] - strikes[:, None], 0)  # Ki - K
                pain = diff_c @ co + diff_p @ po
                max_pain = float(strikes[int(np.argmin(pain))])
            else:
                max_pain = float(strikes[0]) if len(strikes) else np.nan
            # walls (top-3 OI strike)
            top_c = call_oi.sort_values(ascending=False).head(3).index.tolist()
            top_p = put_oi .sort_values(ascending=False).head(3).index.tolist()
            top_c += [np.nan]*(3-len(top_c)); top_p += [np.nan]*(3-len(top_p))
            # ── ATM implied volatility ────────────────────────────────
            # Underlying price: use UndrlygPric from UDIFF when available;
            # fall back to near-month futures settle (very close to spot).
            _spot_vals = (g["underlying_price"].replace(0, np.nan).dropna()
                          if "underlying_price" in g.columns else pd.Series([], dtype=float))
            spot = float(_spot_vals.iloc[0]) if not _spot_vals.empty else _fut_settle.get(sym, float("nan"))
            trade_dt = g["date"].iloc[0]
            expiry_dt = g["expiry"].iloc[0]
            # ATM IV only: on expiry day the near-month series has T=0 (IV
            # unsolvable → all-NaN). Roll IV to the next expiry, matching what
            # the day after expiry already does. OI aggregates above stay on the
            # near (expiring) month.
            iv_ce, iv_pe, iv_expiry = ce, pe, expiry_dt
            if (pd.Timestamp(expiry_dt) - pd.Timestamp(trade_dt)).days <= 0:
                sym_all = opt[opt["symbol"] == sym]
                future_exp = sorted(
                    e for e in sym_all["expiry"].dropna().unique()
                    if pd.Timestamp(e) > pd.Timestamp(trade_dt)
                )
                if future_exp:
                    nxt = future_exp[0]
                    gx = sym_all[sym_all["expiry"] == nxt]
                    iv_ce = gx[gx["opt_type"].str.upper() == "CE"]
                    iv_pe = gx[gx["opt_type"].str.upper() == "PE"]
                    iv_expiry = nxt
            iv_dict = _compute_atm_iv(iv_ce, iv_pe, spot, trade_dt, iv_expiry)

            opt_rows.append({
                "symbol": sym,
                "opt_near_expiry": expiry_dt,
                "opt_call_oi":     ce["open_int"].sum(),
                "opt_put_oi":      pe["open_int"].sum(),
                "opt_call_vol":    ce["contracts"].sum(),
                "opt_put_vol":     pe["contracts"].sum(),
                "pcr_oi":          (pe["open_int"].sum() /
                                    max(ce["open_int"].sum(), 1)),
                "pcr_vol":         (pe["contracts"].sum() /
                                    max(ce["contracts"].sum(), 1)),
                "max_pain":        max_pain,
                "call_wall_1":     top_c[0], "call_wall_2": top_c[1], "call_wall_3": top_c[2],
                "put_wall_1":      top_p[0], "put_wall_2":  top_p[1], "put_wall_3":  top_p[2],
                **iv_dict,         # atm_ce_iv, atm_pe_iv, atm_iv, put_call_iv_skew
            })
    opt_agg = pd.DataFrame(opt_rows) if opt_rows else pd.DataFrame(columns=["symbol"])

    # --- merge futures + options -----------------------------------------
    base = near_fut.rename(columns={
        "expiry":"fut_near_expiry","close":"fut_close","settle":"fut_settle",
        "open_int":"fut_oi","chg_in_oi":"fut_chg_oi","contracts":"fut_vol",
    })[["symbol","fut_near_expiry","fut_close","fut_settle",
        "fut_oi","fut_chg_oi","fut_vol"]] if not near_fut.empty else \
        pd.DataFrame(columns=["symbol"])
    daily = base.merge(opt_agg, on="symbol", how="outer")
    daily.insert(0, "date", df["date"].iloc[0])
    return daily


def build_eod_deriv_daily() -> pd.DataFrame:
    """Stream file-by-file, aggregate on the fly, write (date, symbol) table.
    This replaces the old row-level eod_deriv for modelling — no OOM."""
    print("[silver] building eod_deriv_daily (streaming) ...")
    deriv_dir = RAW_DIR / "derivatives_bhavcopy"
    files = []
    # legacy zips in year subdirs
    files += [("legacy", p) for p in sorted(deriv_dir.rglob("fo*bhav.csv.zip"))]
    # UDIFF — BOTH zipped (in subdirs) and unzipped (flat in folder root or subdirs)
    files += [("udiff", p) for p in sorted(deriv_dir.rglob("BhavCopy_NSE_FO_*.csv.zip"))]
    files += [("udiff", p) for p in sorted(deriv_dir.rglob("BhavCopy_NSE_FO_*.csv"))
              if p.suffix.lower() == ".csv"]
    print(f"  {len(files)} files to process")
    chunks, n_done, n_err = [], 0, 0
    for kind, fp in files:
        day = parse_deriv_legacy(fp) if kind == "legacy" else parse_deriv_udiff(fp)
        if day is None or day.empty:
            n_err += 1; continue
        agg = _aggregate_one_day(day)
        if agg is not None and not agg.empty:
            chunks.append(agg)
        n_done += 1
        if n_done % 100 == 0:
            print(f"    processed {n_done}/{len(files)} ...")
    if not chunks:
        print("  [empty]"); return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["date","symbol"]).drop_duplicates(["date","symbol"])
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["eod_deriv_daily"])
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"days={df['date'].nunique()}  errors={n_err}")
    return df


# keep the legacy row-level builder around but no-op by default
def build_eod_deriv() -> pd.DataFrame:
    print("[silver] eod_deriv (row-level) skipped — using eod_deriv_daily instead")
    return pd.DataFrame()


def build_eod_deriv_contracts() -> pd.DataFrame:
    """Per-contract rows (every strike/expiry) with OI > 0,
    filtered to the active F&O universe (silver/active_universe.txt).
    Written partitioned by year."""
    from . import universe as _u
    allow = set(_u.load())
    if not allow:
        print("[contracts] no active_universe.txt — run `python -m pipeline.universe` first")
        return pd.DataFrame()
    print(f"[silver] building eod_deriv_contracts (OI>0, {len(allow)} symbols) ...")
    deriv_dir = RAW_DIR / "derivatives_bhavcopy"
    out_dir   = SILVER_TABLES["eod_deriv_contracts"]
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    files += [("legacy", p) for p in sorted(deriv_dir.rglob("fo*bhav.csv.zip"))]
    files += [("udiff",  p) for p in sorted(deriv_dir.rglob("BhavCopy_NSE_FO_*.csv.zip"))]
    files += [("udiff",  p) for p in sorted(deriv_dir.rglob("BhavCopy_NSE_FO_*.csv"))
              if p.suffix.lower() == ".csv"]
    print(f"  {len(files)} files to process")

    # buffer rows by year; flush every ~50 files to limit memory
    buf_by_year: dict[int, list[pd.DataFrame]] = {}
    n_done = 0
    total_rows = 0
    def _flush(year=None):
        nonlocal total_rows
        years = [year] if year else list(buf_by_year.keys())
        for y in years:
            if not buf_by_year.get(y): continue
            df = pd.concat(buf_by_year[y], ignore_index=True)
            fp = out_dir / f"year={y}" / "data.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            if fp.exists():
                df = pd.concat([pd.read_parquet(fp), df], ignore_index=True)
                df = df.drop_duplicates(["date","symbol","instrument",
                                         "expiry","strike","opt_type"])
            df.to_parquet(fp)
            total_rows += len(df)
            buf_by_year[y] = []

    for kind, fp in files:
        day = parse_deriv_legacy(fp) if kind == "legacy" else parse_deriv_udiff(fp)
        if day is None or day.empty: continue
        day = day[(day["open_int"].fillna(0) > 0) &
                  (day["symbol"].isin(allow))]            # <<<< OI>0 + universe
        if day.empty: continue
        y = int(pd.to_datetime(day["date"].iloc[0]).year)
        buf_by_year.setdefault(y, []).append(day)
        n_done += 1
        if n_done % 50 == 0:
            print(f"    processed {n_done}/{len(files)} ...")
            _flush()
    _flush()
    print(f"  total rows written: {total_rows:,}  partitions: {len(list(out_dir.glob('year=*')))}")
    return pd.DataFrame()


# ----------------------------------------------------------------------
# 3. PARTICIPANT OI
# ----------------------------------------------------------------------
def parse_participant_oi(fp: Path) -> pd.DataFrame:
    expected = [
        "Client Type",
        "Future Index Long",
        "Future Index Short",
        "Future Stock Long",
        "Future Stock Short",
        "Option Index Call Long",
        "Option Index Put Long",
        "Option Index Call Short",
        "Option Index Put Short",
        "Option Stock Call Long",
        "Option Stock Put Long",
        "Option Stock Call Short",
        "Option Stock Put Short",
        "Total Long Contracts",
        "Total Short Contracts",
    ]
    try:
        # row 0 = title, row 1 = header, rows 2-5 = Client/DII/FII/Pro, row 6 = TOTAL
        df = pd.read_csv(fp, skiprows=1)
    except Exception as e:
        print(f"  [warn] {fp.name}: {e}"); return None
    df = _strip_cols(df)
    if "Future Index Long" not in df.columns:
        try:
            alt = _strip_cols(pd.read_csv(fp))
            if "Future Index Long" in alt.columns:
                df = alt
            elif len(alt.columns) >= len(expected):
                df = alt.iloc[:, :len(expected)].copy()
                df.columns = expected
        except Exception as e:
            print(f"  [warn] participant_oi fallback {fp.name}: {e}")
    elif len(df.columns) >= len(expected):
        df = df.iloc[:, :len(expected)].copy()

    df.columns = [str(col).strip() for col in df.columns]
    if "Client Type" not in df.columns and len(df.columns) >= len(expected):
        df = df.iloc[:, :len(expected)].copy()
        df.columns = expected
    # filter participant rows
    df = df[df.iloc[:,0].isin(["Client","DII","FII","Pro"])].copy()
    if df.empty: return None
    # extract date from filename: fao_participant_oi_DDMMYYYY.csv
    try:
        ds = fp.stem.replace("fao_participant_oi_","")
        d  = dt.datetime.strptime(ds, "%d%m%Y").date()
    except Exception:
        d = pd.NaT
    def _col(name: str):
        if name in df.columns:
            return df[name]
        stripped = {str(col).strip(): col for col in df.columns}
        if name in stripped:
            return df[stripped[name]]
        raise KeyError(name)

    out = pd.DataFrame({
        "date":               pd.to_datetime(d),
        "participant":        df.iloc[:,0].values,
        "fut_idx_long":       _to_num(_col("Future Index Long")),
        "fut_idx_short":      _to_num(_col("Future Index Short")),
        "fut_stk_long":       _to_num(_col("Future Stock Long")),
        "fut_stk_short":      _to_num(_col("Future Stock Short")),
        "opt_idx_call_long":  _to_num(_col("Option Index Call Long")),
        "opt_idx_put_long":   _to_num(_col("Option Index Put Long")),
        "opt_idx_call_short": _to_num(_col("Option Index Call Short")),
        "opt_idx_put_short":  _to_num(_col("Option Index Put Short")),
        "opt_stk_call_long":  _to_num(_col("Option Stock Call Long")),
        "opt_stk_put_long":   _to_num(_col("Option Stock Put Long")),
        "opt_stk_call_short": _to_num(_col("Option Stock Call Short")),
        "opt_stk_put_short":  _to_num(_col("Option Stock Put Short")),
        "total_long":         _to_num(_col("Total Long Contracts")),
        "total_short":        _to_num(_col("Total Short Contracts")),
    })
    return out


def build_participant_oi() -> pd.DataFrame:
    print("[silver] building participant_oi ...")
    frames = []
    for f in sorted((RAW_DIR / "participant_oi").rglob("fao_participant_oi_*.csv")):
        df = parse_participant_oi(f)
        if df is not None: frames.append(df)
    if not frames:
        print("  [empty]"); return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["date","participant"])
    df = df.sort_values(["date","participant"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["participant_oi"])
    print(f"  rows={len(df):,}  days={df['date'].nunique()}")
    return df


# ----------------------------------------------------------------------
# 4. INDICES (incl. India VIX, sectorals)
# ----------------------------------------------------------------------
def parse_indices(fp: Path) -> pd.DataFrame:
    df = _safe_read_csv(fp)
    if df is None or df.empty: return None
    df = _strip_cols(df)
    df["date"] = _date_from_ddmmyyyy(df["Index Date"].astype(str).str.strip())
    out = pd.DataFrame({
        "date":       df["date"],
        "index_name": df["Index Name"].astype(str).str.strip(),
        "open":       _to_num(df["Open Index Value"]),
        "high":       _to_num(df["High Index Value"]),
        "low":        _to_num(df["Low Index Value"]),
        "close":      _to_num(df["Closing Index Value"]),
        "points_chg": _to_num(df["Points Change"]),
        "pct_chg":    _to_num(df["Change(%)"]),
        "volume":     _to_num(df["Volume"]),
        "turnover":   _to_num(df["Turnover (Rs. Cr.)"]),
        "pe":         _to_num(df["P/E"]),
        "pb":         _to_num(df["P/B"]),
        "div_yield":  _to_num(df["Div Yield"]),
    })
    return out.dropna(subset=["date","index_name"])


def build_indices() -> pd.DataFrame:
    print("[silver] building indices ...")
    frames = []
    for f in sorted((RAW_DIR / "indices").rglob("ind_close_all_*.csv")):
        df = parse_indices(f)
        if df is not None: frames.append(df)
    if not frames:
        print("  [empty]"); return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["date","index_name"])
    df = df.sort_values(["index_name","date"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["indices"])
    print(f"  rows={len(df):,}  uniq_idx={df['index_name'].nunique()}  "
          f"days={df['date'].nunique()}")
    return df


# ----------------------------------------------------------------------
# LOT SIZE — from UDIFF bhavcopies (NewBrdLotQty) + fo_mktlots.csv snapshots
# ----------------------------------------------------------------------
def _lot_from_udiff_files() -> pd.DataFrame:
    """Every UDIFF bhavcopy row carries NewBrdLotQty — pull distinct
    (symbol, expiry_month, lot_size) without touching price/OI columns."""
    rows = []
    files = list((RAW_DIR/"derivatives_bhavcopy").rglob("BhavCopy_NSE_FO_*.csv.zip")) + \
            [p for p in (RAW_DIR/"derivatives_bhavcopy").rglob("BhavCopy_NSE_FO_*.csv")
             if p.suffix.lower() == ".csv"]
    print(f"  scanning {len(files)} UDIFF files for lot sizes ...")
    for i, fp in enumerate(files):
        try:
            if fp.suffix.lower() == ".zip":
                with zipfile.ZipFile(fp) as z:
                    with z.open([n for n in z.namelist() if n.lower().endswith(".csv")][0]) as f:
                        df = pd.read_csv(f, usecols=["TckrSymb","XpryDt","NewBrdLotQty"])
            else:
                df = pd.read_csv(fp, usecols=["TckrSymb","XpryDt","NewBrdLotQty"])
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        df["symbol"] = df["TckrSymb"].astype(str).str.strip().str.upper()
        df["expiry"] = pd.to_datetime(df["XpryDt"], errors="coerce")
        df["lot"]    = _to_num(df["NewBrdLotQty"])
        df = df.dropna(subset=["symbol","expiry","lot"]).drop_duplicates()
        rows.append(df[["symbol","expiry","lot"]])
        if (i+1) % 200 == 0: print(f"    scanned {i+1}/{len(files)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _lot_from_mktlots_files() -> pd.DataFrame:
    """Parse any raw/lot_size/<YYYY-MM>/fo_mktlots.csv snapshots."""
    rows = []
    for fp in sorted((RAW_DIR/"lot_size").rglob("fo_mktlots.csv")):
        snapshot = fp.parent.name  # YYYY-MM
        try:
            df = pd.read_csv(fp, skiprows=1)    # first row is disclaimer
        except Exception as e:
            print(f"  [warn] {fp}: {e}"); continue
        df.columns = [c.strip() for c in df.columns]
        sym_col = [c for c in df.columns if "SYMBOL" in c.upper()]
        if not sym_col: continue
        df["symbol"] = df[sym_col[0]].astype(str).str.strip().str.upper()
        # remaining columns are month-labels like 'Apr-25'; melt into long form
        month_cols = [c for c in df.columns if c != sym_col[0] and c != "symbol"
                      and ("-" in c or "/" in c)]
        for mc in month_cols:
            try:
                expiry_m = pd.to_datetime("01-" + mc, format="%d-%b-%y", errors="coerce")
            except Exception:
                expiry_m = pd.NaT
            sub = pd.DataFrame({
                "symbol": df["symbol"],
                "expiry_month": expiry_m,
                "lot": _to_num(df[mc]),
                "snapshot": snapshot,
            }).dropna(subset=["symbol","expiry_month","lot"])
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_lot_size() -> pd.DataFrame:
    print("[silver] building lot_size ...")
    udiff = _lot_from_udiff_files()
    if not udiff.empty:
        udiff["expiry_month"] = udiff["expiry"].dt.to_period("M").dt.to_timestamp()
        udiff = udiff[["symbol","expiry_month","lot"]].assign(source="udiff")
    mkt = _lot_from_mktlots_files()
    if not mkt.empty:
        mkt = mkt[["symbol","expiry_month","lot"]].assign(source="mktlots")
    frames = [d for d in (udiff, mkt) if not d.empty]
    if not frames:
        print("  [empty]"); return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # one row per (symbol, expiry_month) — prefer UDIFF (more authoritative, tied to actual trading)
    df = df.sort_values(["symbol","expiry_month","source"]).drop_duplicates(
            ["symbol","expiry_month"], keep="first")
    df.to_parquet(SILVER_TABLES["lot_size"])
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"months={df['expiry_month'].nunique()}")
    return df


# ----------------------------------------------------------------------
# 5-7. BULK JSON FEEDS (fii_dii_cash, corp_actions, earnings)
# ----------------------------------------------------------------------
def _load_jsons(folder: Path) -> list[dict]:
    out = []
    if not folder.exists(): return out
    for f in sorted(folder.rglob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    return out


def build_fii_dii_cash() -> pd.DataFrame:
    print("[silver] building fii_dii_cash ...")
    rows = []
    for j in _load_jsons(RAW_DIR / "fii_dii_cash"):
        # NSE returns either {"data":[{...}]} or list-of-dicts; tolerate both
        items = j.get("data", j) if isinstance(j, dict) else j
        if not isinstance(items, list): continue
        for it in items:
            d = it.get("date") or it.get("trDate") or it.get("Date")
            rows.append({
                "date":    pd.to_datetime(d, errors="coerce"),
                "fii_buy":  _to_num(pd.Series([it.get("fiiBuyValue") or it.get("buyValue")])).iloc[0],
                "fii_sell": _to_num(pd.Series([it.get("fiiSellValue") or it.get("sellValue")])).iloc[0],
                "fii_net":  _to_num(pd.Series([it.get("fiiNetValue")  or it.get("netValue")])).iloc[0],
                "dii_buy":  _to_num(pd.Series([it.get("diiBuyValue")])).iloc[0],
                "dii_sell": _to_num(pd.Series([it.get("diiSellValue")])).iloc[0],
                "dii_net":  _to_num(pd.Series([it.get("diiNetValue")])).iloc[0],
            })
    if not rows:
        print("  [empty json feed; using NSDL archive builder]")
        try:
            from .fetch_fiidii import build_fii_dii_cash as build_nsdl_fii_dii_cash
            return build_nsdl_fii_dii_cash()
        except Exception as e:
            print(f"  [warn] NSDL FII/DII builder failed: {e}")
            return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["fii_dii_cash"])
    print(f"  rows={len(df):,}")
    return df


def _nse_corp_rows() -> list[dict]:
    rows = []
    for j in _load_jsons(RAW_DIR / "corporate_actions"):
        items = j if isinstance(j, list) else j.get("data", [])
        for it in items:
            rows.append({
                "date":        pd.to_datetime(it.get("exDate") or it.get("ex_date"), errors="coerce"),
                "symbol":      str(it.get("symbol", "")).strip().upper(),
                "action_type": str(it.get("subject") or it.get("purpose") or ""),
                "amount":      float("nan"),
                "ex_date":     pd.to_datetime(it.get("exDate"), errors="coerce"),
                "record_date": pd.to_datetime(it.get("recDate"), errors="coerce"),
                "_src":        "nse",
            })
    return rows


def _yf_corp_rows() -> list[dict]:
    rows = []
    corp_yf_dir = RAW_DIR / "corp_actions_yf"
    if not corp_yf_dir.exists():
        return rows
    for fp in sorted(corp_yf_dir.glob(_YF_JSON_GLOB)):
        try:
            items = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")
            continue
        for it in items:
            rows.append({
                "date":        pd.to_datetime(it.get("ex_date"), errors="coerce"),
                "symbol":      str(it.get("symbol", "")).strip().upper(),
                "action_type": str(it.get("action_type", "")),
                "amount":      float(it["amount"]) if it.get("amount") is not None else float("nan"),
                "ex_date":     pd.to_datetime(it.get("ex_date"), errors="coerce"),
                "record_date": pd.NaT,
                "_src":        "yf",
            })
    return rows


def build_corp_actions() -> pd.DataFrame:
    """Merge NSE raw corporate actions (if present) with yfinance corp actions.
    NSE wins on (symbol, date, action_type) duplicates; yfinance covers the rest.
    """
    print("[silver] building corp_actions ...")
    rows = _nse_corp_rows() + _yf_corp_rows()
    if not rows:
        print("  [empty]"); return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["date", "symbol"])
    df["symbol"] = df["symbol"].str.strip().str.upper()

    # NSE wins: sort nse last (ord=1), keep last on (symbol, date, action_type)
    df["_src_ord"] = df["_src"].map({"yf": 0, "nse": 1}).fillna(0)
    df = (df.sort_values(["symbol", "date", "action_type", "_src_ord"])
            .drop_duplicates(["symbol", "date", "action_type"], keep="last")
            .drop(columns=["_src", "_src_ord"])
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))

    df.to_parquet(SILVER_TABLES["corp_actions"])
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}")
    return df


def build_earnings() -> pd.DataFrame:
    print("[silver] building earnings ...")
    rows = []
    for j in _load_jsons(RAW_DIR / "earnings_calendar"):
        items = j if isinstance(j, list) else j.get("data", [])
        for it in items:
            rows.append({
                "date":       pd.to_datetime(it.get("date") or it.get("bm_desc"), errors="coerce"),
                "symbol":     str(it.get("symbol","")).strip().upper(),
                "event_type": it.get("purpose") or it.get("subject") or "RESULTS",
                "period":     it.get("bm_desc",""),
            })
    if not rows:
        print("  [empty]"); return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["date","symbol"]).drop_duplicates()
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["earnings"])
    print(f"  rows={len(df):,}")
    return df


# ----------------------------------------------------------------------
# 8. QUARTERLY FUNDAMENTALS (yfinance fetch_yf_fundamentals.py)
# ----------------------------------------------------------------------
_YF_JSON_GLOB = "yf_*.json"

_FUND_NUM_COLS = [
    "revenue", "net_income", "ebitda", "gross_profit", "total_debt",
    "total_equity", "total_assets", "pat_margin", "rev_growth_yoy",
    "roe_annualized", "de_ratio", "ebitda_margin",
]


def _load_fund_source(fund_dir: Path, pattern: str, source: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(fund_dir.glob(pattern)):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend(payload)
            else:
                # Newer yfinance snapshots are raw nested statements and do
                # not have the normalized avail_from/period_end row schema.
                # Keep them in raw for future parsing, but do not crash silver.
                continue
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    required = {"symbol", "avail_from", "period_end"}
    if not required.issubset(df.columns):
        print(f"  [warn] {source} fundamentals missing normalized columns; skipping")
        return pd.DataFrame()
    df["avail_from"] = pd.to_datetime(df["avail_from"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["symbol", "avail_from"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    for c in _FUND_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_src"] = source
    return df


def build_fundamentals() -> pd.DataFrame:
    """Merge screener.in (historical baseline) with yfinance (recent overlay).
    yfinance wins on overlapping (symbol, period_end) — it has gross_profit + total_assets.
    Derived ratios (margins, ROE, D/E) are unit-independent so both sources are comparable.
    """
    print("[silver] building fundamentals ...")
    fund_dir = RAW_DIR / "fundamentals"
    if not fund_dir.exists():
        print("  [skip] raw/fundamentals not found — run fetch_screener/yf_fundamentals.py first")
        return pd.DataFrame()

    screener = _load_fund_source(fund_dir, "screener_*.json", "screener")
    yf       = _load_fund_source(fund_dir, _YF_JSON_GLOB,       "yf")
    print(f"  screener rows={len(screener):,}  yf rows={len(yf):,}")

    frames = [df for df in (screener, yf) if not df.empty]
    if not frames:
        print("  [empty]"); return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # yfinance wins on same (symbol, period_end): sort yf last, keep last on dedup
    combined["_src_ord"] = combined["_src"].map({"screener": 0, "yf": 1}).fillna(0)
    combined = (combined
                .sort_values(["symbol", "period_end", "_src_ord"])
                .drop_duplicates(["symbol", "period_end"], keep="last")
                .drop(columns=["_src", "_src_ord"])
                .sort_values(["symbol", "avail_from"])
                .reset_index(drop=True))

    combined.to_parquet(SILVER_TABLES["fundamentals"])
    print(f"  merged rows={len(combined):,}  symbols={combined['symbol'].nunique()}")
    return combined


def build_analyst_ratings() -> pd.DataFrame:
    print("[silver] building analyst_ratings ...")
    rat_dir = RAW_DIR / "analyst_ratings"
    if not rat_dir.exists():
        print("  [skip] raw/analyst_ratings not found — run fetch_yf_fundamentals.py first")
        return pd.DataFrame()
    rows = []
    for fp in sorted(rat_dir.glob(_YF_JSON_GLOB)):
        try:
            rows.extend(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")
    if not rows:
        print("  [empty]"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df.drop_duplicates(["symbol", "date", "firm"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["analyst_ratings"])
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}")
    return df


_EPS_NUM_COLS = ["eps_estimate", "eps_reported", "surprise_pct", "eps_growth_yoy", "eps_ttm"]


def _parse_investing_eps_record(rec: dict) -> dict | None:
    """Parse one Investing.com earnings record from raw/investing_eps/{sym}.json."""
    sym = str(rec.get("symbol", "") or "").strip().upper()
    if not sym:
        return None

    date_raw = str(rec.get("date", "") or "").strip()
    # Common formats from Investing.com: "Nov 01, 2023", "01/11/2023", "2023-11-01"
    date = None
    for fmt in ("%b %d, %Y", "%b %d,%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            date = pd.to_datetime(date_raw, format=fmt)
            break
        except Exception:
            pass
    if date is None:
        date = pd.to_datetime(date_raw, errors="coerce")
    if pd.isna(date):
        return None

    def _num(v):
        s = str(v or "").replace(",", "").replace("%", "").replace("+", "").strip()
        if s in ("", "-", "--", "N/A", "n/a"):
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return {
        "date":         date,
        "symbol":       sym,
        "eps_estimate": _num(rec.get("eps_estimate")),
        "eps_reported": _num(rec.get("eps_reported")),
        "surprise_pct": _num(rec.get("surprise_pct")),
        "rev_estimate": _num(rec.get("rev_estimate")),
        "rev_reported": _num(rec.get("rev_reported")),
    }


def build_investing_eps() -> pd.DataFrame:
    """Consolidate raw/investing_eps/{sym}.json → silver/investing_eps.parquet.

    Schema: date, symbol, eps_estimate, eps_reported, surprise_pct,
            rev_estimate, rev_reported  (one row per stock per quarter)
    """
    print("[silver] building investing_eps ...")
    eps_dir = RAW_DIR / "investing_eps"
    if not eps_dir.exists():
        print("  [skip] run: python -m pipeline.fetch --investing_eps")
        return pd.DataFrame()

    rows = []
    for fp in sorted(eps_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for rec in data:
                parsed = _parse_investing_eps_record(rec)
                if parsed:
                    rows.append(parsed)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    if not rows:
        print("  [empty]")
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .dropna(subset=["date", "symbol"])
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))

    out_path = SILVER_TABLES["investing_eps"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    n_est  = df["eps_estimate"].notna().sum()
    n_surp = df["surprise_pct"].notna().sum()
    n_rev  = df["rev_reported"].notna().sum()
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"with_estimate={n_est:,}  with_surprise={n_surp:,}  "
          f"with_revenue={n_rev:,}")
    return df


def build_earnings_eps() -> pd.DataFrame:
    """Quarterly EPS history from fetch_yf_eps.py.
    earnings_date is the actual announcement date (used as avail_from in features).
    """
    print("[silver] building earnings_eps ...")
    eps_dir = RAW_DIR / "earnings_eps"
    if not eps_dir.exists():
        print("  [skip] raw/earnings_eps not found — run fetch_yf_eps.py first")
        return pd.DataFrame()
    rows = []
    for fp in sorted(eps_dir.glob(_YF_JSON_GLOB)):
        try:
            rows.extend(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")
    if not rows:
        print("  [empty]"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df = df.dropna(subset=["symbol", "earnings_date"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    for c in _EPS_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)
    df.to_parquet(SILVER_TABLES["earnings_eps"])
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}")
    return df


def _parse_block_deal_record(rec: dict) -> dict | None:
    """Parse one block deal record from either API format.

    Historical API (BD_* fields):
        BD_SYMBOL, BD_BUY_SELL, BD_QTY_TRD, BD_TP_WATP, BD_DT_DATE, BD_CLIENT_NAME

    Legacy today-only API (camelCase fields — kept for backward-compat):
        symbol, dealType, quantity, price, tradeDate
    """
    # Historical API format
    if "BD_SYMBOL" in rec:
        sym  = str(rec.get("BD_SYMBOL", "") or "").strip().upper()
        bs   = str(rec.get("BD_BUY_SELL", "") or "").strip().upper()
        qty  = pd.to_numeric(rec.get("BD_QTY_TRD",  0), errors="coerce") or 0
        price= pd.to_numeric(rec.get("BD_TP_WATP",  0), errors="coerce") or 0
        date_raw = str(rec.get("BD_DT_DATE", "") or "")
        client   = str(rec.get("BD_CLIENT_NAME", "") or "").strip()
        # BD_DT_DATE is "DD-MON-YYYY" (e.g. "05-JAN-2024")
        date = pd.to_datetime(date_raw, format="%d-%b-%Y", errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(date_raw, errors="coerce")
    else:
        # Legacy format
        sym  = str(rec.get("symbol",   "") or "").strip().upper()
        bs   = str(rec.get("dealType", "") or "").strip().upper()
        qty  = pd.to_numeric(rec.get("quantity", 0), errors="coerce") or 0
        price= pd.to_numeric(rec.get("price",    0), errors="coerce") or 0
        date_raw = str(rec.get("tradeDate", "") or "")
        client   = ""
        date = pd.to_datetime(date_raw, format="%d-%b-%Y", errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(date_raw, errors="coerce")

    if not sym or pd.isna(date):
        return None
    return {
        "date":      date,
        "symbol":    sym,
        "deal_type": bs,      # "BUY" or "SELL"
        "client":    client,
        "quantity":  qty,
        "price":     price,
        "value_cr":  round(qty * price / 1e7, 4),   # crores
    }


def build_block_deals() -> pd.DataFrame:
    """Build silver/block_deals.parquet from raw/block_deals/{YYYY}/*.json files.

    Raw files are produced by fetch.py using the NSE historical API:
        /api/historicalOR/bulk-block-short-deals?from=DD-MM-YYYY&to=DD-MM-YYYY&optionType=block_deals

    Each file is a JSON list of deal records (BD_* fields).

    Output schema (one row per deal):
        date, symbol, deal_type, client, quantity, price, value_cr
    """
    print("[silver] building block_deals ...")
    bd_dir = RAW_DIR / "block_deals"
    if not bd_dir.exists():
        print("  [skip] raw/block_deals not found — run: python -m pipeline.fetch --block_deals 2010-01-01")
        return pd.DataFrame()

    rows = []
    for fp in sorted(bd_dir.rglob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not data:
                continue
            for rec in data:
                parsed = _parse_block_deal_record(rec)
                if parsed:
                    rows.append(parsed)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    if not rows:
        print("  [empty]")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "symbol"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    out_path = SILVER_TABLES["block_deals"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    sell_cr = df.loc[df["deal_type"] == "SELL", "value_cr"].sum()
    buy_cr  = df.loc[df["deal_type"] == "BUY",  "value_cr"].sum()
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"dates={df['date'].nunique()}  "
          f"sell_cr={sell_cr:.0f}  buy_cr={buy_cr:.0f}")
    return df


def build_bulk_deals() -> pd.DataFrame:
    """Build silver/bulk_deals.parquet from raw/bulk_deals/{YYYY}/*.json files.

    Raw files are produced by fetch.py using the NSE historical API:
        /api/historicalOR/bulk-block-short-deals?from=DD-MM-YYYY&to=DD-MM-YYYY&optionType=bulk_deals

    The API caps at 70 rows per day (the day's largest deals by value).
    Same output schema as block_deals: date, symbol, deal_type, client, quantity, price, value_cr
    """
    print("[silver] building bulk_deals ...")
    bd_dir = RAW_DIR / "bulk_deals"
    if not bd_dir.exists():
        print("  [skip] raw/bulk_deals not found — run: python -m pipeline.fetch --bulk_deals 2010-01-01")
        return pd.DataFrame()

    rows = []
    for fp in sorted(bd_dir.rglob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not data:
                continue
            for rec in data:
                parsed = _parse_block_deal_record(rec)
                if parsed:
                    rows.append(parsed)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    if not rows:
        print("  [empty]")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "symbol"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    out_path = SILVER_TABLES["bulk_deals"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    sell_cr = df.loc[df["deal_type"] == "SELL", "value_cr"].sum()
    buy_cr  = df.loc[df["deal_type"] == "BUY",  "value_cr"].sum()
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"dates={df['date'].nunique()}  "
          f"sell_cr={sell_cr:.0f}  buy_cr={buy_cr:.0f}")
    return df


# ----------------------------------------------------------------------
# incremental append helpers
# ----------------------------------------------------------------------

def _upsert_parquet(path, new_df: pd.DataFrame, keys: list[str]):
    """Append new rows to an existing parquet, deduplicating on keys."""
    if new_df is None or new_df.empty:
        return
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    combined = (combined
                .drop_duplicates(keys, keep="last")
                .sort_values(keys)
                .reset_index(drop=True))
    combined.to_parquet(path)


def append_day(date_str: str):
    """
    Parse only the raw files for one trading date and upsert into each
    silver table.  Much faster than a full rebuild for a daily workflow.
    Tables that come from bulk/periodic sources (fundamentals, corp_actions,
    earnings, analyst_ratings, earnings_eps) are left unchanged — they are
    only updated during a periodic full rebuild.
    """
    date_d    = dt.date.fromisoformat(date_str)
    ddmmyyyy  = date_d.strftime("%d%m%Y")
    yyyymmdd  = date_d.strftime("%Y%m%d")
    year      = str(date_d.year)

    def _find(base: Path, *names) -> Path | None:
        for name in names:
            for candidate in [base / name, base / year / name]:
                if candidate.exists():
                    return candidate
        return None

    # ── eod_stock ──────────────────────────────────────────────────────
    fp = _find(RAW_DIR / "cash_bhavcopy", f"sec_bhavdata_full_{ddmmyyyy}.csv")
    if fp:
        df = parse_cash_modern(fp)
        if df is not None:
            _upsert_parquet(SILVER_TABLES["eod_stock"], df, ["date", "symbol"])
            print(f"[silver] eod_stock  +{len(df)} rows for {date_str}")
    else:
        print(f"[silver] eod_stock: no file found for {date_str}")

    # ── eod_deriv_daily + eod_deriv_contracts ──────────────────────────
    fp = _find(RAW_DIR / "derivatives_bhavcopy",
               f"BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip",
               f"BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv",
               f"BhavCopy_NSE_FO_{yyyymmdd}.csv.zip",
               f"BhavCopy_NSE_FO_{yyyymmdd}.csv")
    # also try legacy format fo<DDMMMYYYY>bhav.csv.zip
    if fp is None:
        ddmmmyyyy = date_d.strftime("%d%b%Y").upper()
        fp = _find(RAW_DIR / "derivatives_bhavcopy",
                   f"fo{ddmmmyyyy}bhav.csv.zip")

    if fp:
        kind = "legacy" if "bhav.csv.zip" in fp.name.lower() and fp.name.lower().startswith("fo") else "udiff"
        raw_day = parse_deriv_legacy(fp) if kind == "legacy" else parse_deriv_udiff(fp)
        if raw_day is not None and not raw_day.empty:
            # eod_deriv_daily
            agg = _aggregate_one_day(raw_day)
            if agg is not None and not agg.empty:
                _upsert_parquet(SILVER_TABLES["eod_deriv_daily"], agg, ["date", "symbol"])
                print(f"[silver] eod_deriv_daily +{len(agg)} rows for {date_str}")
            # eod_deriv_contracts (OI>0, universe filtered, partitioned by year)
            from . import universe as _u
            allow = set(_u.load())
            if allow:
                contracts = raw_day[(raw_day["open_int"].fillna(0) > 0) &
                                    (raw_day["symbol"].isin(allow))]
                if not contracts.empty:
                    out_dir = Path(SILVER_TABLES["eod_deriv_contracts"])
                    part = out_dir / f"year={date_d.year}" / "data.parquet"
                    part.parent.mkdir(parents=True, exist_ok=True)
                    if part.exists():
                        existing = pd.read_parquet(part)
                        contracts = pd.concat([existing, contracts], ignore_index=True)
                        contracts = contracts.drop_duplicates(
                            ["date", "symbol", "instrument", "expiry", "strike", "opt_type"])
                    contracts.to_parquet(part)
                    print(f"[silver] eod_deriv_contracts +rows for {date_str}")
    else:
        print(f"[silver] eod_deriv: no file found for {date_str}")

    # ── participant_oi ─────────────────────────────────────────────────
    fp = _find(RAW_DIR / "participant_oi", f"fao_participant_oi_{ddmmyyyy}.csv")
    if fp:
        df = parse_participant_oi(fp)
        if df is not None:
            _upsert_parquet(SILVER_TABLES["participant_oi"], df, ["date", "participant"])
            print(f"[silver] participant_oi +{len(df)} rows for {date_str}")

    # ── indices ────────────────────────────────────────────────────────
    fp = _find(RAW_DIR / "indices", f"ind_close_all_{ddmmyyyy}.csv")
    if fp:
        df = parse_indices(fp)
        if df is not None:
            _upsert_parquet(SILVER_TABLES["indices"], df, ["date", "index_name"])
            print(f"[silver] indices +{len(df)} rows for {date_str}")

    # ── fii_dii_cash — rebuild from all JSONs (fast) ───────────────────
    # FII cash is refreshed once per pipeline via pipeline.fetch_fiidii.
    # Avoid rebuilding all NSDL archives inside every append-day call.

    # ── block_deals / bulk_deals — upsert today's deals ──────────────
    for _deal_key in ("block_deals", "bulk_deals"):
        _deal_file = RAW_DIR / _deal_key / year / f"{ddmmyyyy}.json"
        if _deal_file.exists():
            try:
                data = json.loads(_deal_file.read_text(encoding="utf-8"))
                if data:
                    rows = [_parse_block_deal_record(rec) for rec in data]
                    rows = [r for r in rows if r is not None]
                    if rows:
                        new_df = pd.DataFrame(rows)
                        _upsert_parquet(SILVER_TABLES[_deal_key], new_df,
                                        ["date", "symbol", "deal_type", "client"])
                        print(f"[silver] {_deal_key} +{len(rows)} deals for {date_str}")
            except Exception as e:
                print(f"[silver] {_deal_key} parse error for {date_str}: {e}")



# ----------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------
BUILDERS = {
    # corp_actions must come before eod_stock — _split_adjust_eod reads it
    "corp_actions":       build_corp_actions,
    "eod_stock":          build_eod_stock,
    "eod_deriv":          build_eod_deriv,          # no-op, kept for compat
    "eod_deriv_daily":    build_eod_deriv_daily,    # real F&O output now
    "eod_deriv_contracts": build_eod_deriv_contracts,  # per-contract, OI>0
    "lot_size":           build_lot_size,
    "participant_oi":     build_participant_oi,
    "indices":            build_indices,
    "fii_dii_cash":       build_fii_dii_cash,
    "earnings":           build_earnings,
    "fundamentals":       build_fundamentals,
    "analyst_ratings":    build_analyst_ratings,
    "earnings_eps":       build_earnings_eps,
    "block_deals":        build_block_deals,
    "bulk_deals":         build_bulk_deals,
    "investing_eps":      build_investing_eps,
}

def main(tables=None):
    # default run: everything except legacy row-level and heavy contracts table
    default = [k for k in BUILDERS.keys()
               if k not in ("eod_deriv","eod_deriv_contracts")]
    # When called as a library function (e.g. from daily_run.py), `tables` is
    # passed explicitly.  When invoked as __main__ use CLI argv instead.
    cli = tables if tables is not None else sys.argv[1:]
    if cli and cli[0] == "append-day":
        if len(cli) < 2:
            raise SystemExit("Usage: python -m pipeline.silver append-day YYYY-MM-DD [YYYY-MM-DD ...]")
        for date_str in cli[1:]:
            append_day(date_str)
        return
    targets = cli or default
    for t in targets:
        if t in BUILDERS:
            BUILDERS[t]()
        else:
            print(f"  [skip] unknown table: {t}")

if __name__ == "__main__":
    main()
