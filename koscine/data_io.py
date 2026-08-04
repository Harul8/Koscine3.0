from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from koscine.config import RAW_DATA_ROOT, SILVER_DATA_ROOT, SYMBOL_ALIASES, TARGET_UNIVERSE


TARGET_SET = set(TARGET_UNIVERSE)


def load_universe_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    symbols = {
        normalize_symbol(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return symbols


def load_training_universe(silver_root: Path = SILVER_DATA_ROOT) -> set[str]:
    symbols = load_universe_file(silver_root / "training_universe.txt")
    return (symbols or set()) | TARGET_SET


def normalize_symbol(symbol: object) -> str:
    value = str(symbol).strip().upper()
    return SYMBOL_ALIASES.get(value, value)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().strip(",") for col in df.columns]
    return df


def _read_legacy_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as fh:
            return pd.read_csv(fh)


def _read_modern_cash_file(path: Path) -> pd.DataFrame:
    try:
        df = _clean_columns(pd.read_csv(path))
    except UnicodeDecodeError:
        try:
            df = _clean_columns(pd.read_csv(path, encoding="latin1"))
        except Exception:
            with path.open("rb") as fh:
                prefix = fh.read(2)
            if prefix != b"PK":
                raise
            df = _clean_columns(pd.read_excel(path))
    except Exception:
        with path.open("rb") as fh:
            prefix = fh.read(2)
        if prefix != b"PK":
            raise
        df = _clean_columns(pd.read_excel(path))
    df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y"),
            "symbol": df["SYMBOL"].map(normalize_symbol),
            "open": pd.to_numeric(df["OPEN_PRICE"], errors="coerce"),
            "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
            "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
            "last": pd.to_numeric(df["LAST_PRICE"], errors="coerce"),
            "prev_close": pd.to_numeric(df["PREV_CLOSE"], errors="coerce"),
            "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
            "turnover_lacs": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce"),
            "trades": pd.to_numeric(df["NO_OF_TRADES"], errors="coerce"),
            "delivery_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
            "delivery_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
        }
    )
    return out[out["symbol"].isin(TARGET_SET)]


def _read_legacy_ohlc_file(path: Path) -> pd.DataFrame:
    df = _clean_columns(_read_legacy_zip(path))
    df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y"),
            "symbol": df["SYMBOL"].map(normalize_symbol),
            "open": pd.to_numeric(df["OPEN"], errors="coerce"),
            "high": pd.to_numeric(df["HIGH"], errors="coerce"),
            "low": pd.to_numeric(df["LOW"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
            "last": pd.to_numeric(df["LAST"], errors="coerce"),
            "prev_close": pd.to_numeric(df["PREVCLOSE"], errors="coerce"),
            "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
            "turnover_lacs": pd.to_numeric(df["TOTTRDVAL"], errors="coerce") / 100000.0,
        }
    )
    return out[out["symbol"].isin(TARGET_SET)]


def _read_legacy_delivery_file(path: Path) -> pd.DataFrame:
    rows = []
    trade_date = None
    trade_date_pattern = re.compile(r"Trade Date <([^>]+)>")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            match = trade_date_pattern.search(line)
            if match:
                trade_date = pd.to_datetime(match.group(1), format="%d-%b-%Y")
                continue
            if not line.startswith("20,"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "date": trade_date,
                    "symbol": normalize_symbol(parts[2]),
                    "delivery_qty": pd.to_numeric(parts[5], errors="coerce"),
                    "delivery_pct": pd.to_numeric(parts[6], errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "delivery_qty", "delivery_pct"])
    out = pd.DataFrame(rows)
    return out[out["symbol"].isin(TARGET_SET)]


def load_cash_daily(raw_root: Path = RAW_DATA_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    legacy_dir = raw_root / "cash_bhavcopy_legacy_ohlc"
    for path in sorted(legacy_dir.glob("*/*.zip")):
        try:
            frame = _read_legacy_ohlc_file(path)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"warn: failed legacy OHLC {path}: {exc}")

    modern_dir = raw_root / "cash_bhavcopy"
    for path in sorted(modern_dir.glob("*/*.csv")):
        try:
            frame = _read_modern_cash_file(path)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"warn: failed modern cash {path}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No cash bhavcopy rows found under {raw_root}")

    cash = pd.concat(frames, ignore_index=True)
    cash = cash.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    cash = cash.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    delivery_frames: list[pd.DataFrame] = []
    delivery_dir = raw_root / "cash_bhavcopy_legacy_delivery"
    for path in sorted(delivery_dir.glob("*/*.DAT")):
        try:
            frame = _read_legacy_delivery_file(path)
            if not frame.empty:
                delivery_frames.append(frame)
        except Exception as exc:
            print(f"warn: failed delivery {path}: {exc}")

    if delivery_frames:
        delivery = pd.concat(delivery_frames, ignore_index=True)
        delivery = delivery.dropna(subset=["date", "symbol"])
        cash = cash.merge(
            delivery,
            on=["date", "symbol"],
            how="left",
            suffixes=("", "_legacy"),
        )
        for col in ("delivery_qty", "delivery_pct"):
            legacy_col = f"{col}_legacy"
            if legacy_col in cash:
                cash[col] = cash[col].fillna(cash[legacy_col])
                cash = cash.drop(columns=[legacy_col])

    return cash.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_silver_stock_daily(
    silver_root: Path = SILVER_DATA_ROOT,
    universe: set[str] | None = None,
) -> pd.DataFrame:
    path = silver_root / "eod_stock.parquet"
    df = pd.read_parquet(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if universe is not None:
        df = df[df["symbol"].isin(universe)]
    out = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "last": pd.to_numeric(df["last"], errors="coerce"),
            "prev_close": pd.to_numeric(df["prev_close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "turnover_lacs": pd.to_numeric(df["turnover"], errors="coerce"),
            "trades": pd.to_numeric(df["n_trades"], errors="coerce"),
            "delivery_qty": pd.to_numeric(df["deliv_qty"], errors="coerce"),
            "delivery_pct": pd.to_numeric(df["deliv_pct"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "symbol", "open", "high", "low", "close"]).sort_values(
        ["date", "symbol"]
    )


def load_silver_derivatives_daily(
    silver_root: Path = SILVER_DATA_ROOT,
    universe: set[str] | None = None,
) -> pd.DataFrame:
    path = silver_root / "eod_deriv_daily.parquet"
    df = pd.read_parquet(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if universe is not None:
        df = df[df["symbol"].isin(universe)]
    keep = [
        "date",
        "symbol",
        "fut_close",
        "fut_settle",
        "fut_oi",
        "fut_chg_oi",
        "fut_vol",
        "opt_call_oi",
        "opt_put_oi",
        "opt_call_vol",
        "opt_put_vol",
        "pcr_oi",
        "pcr_vol",
        "max_pain",
        "call_wall_1",
        "call_wall_2",
        "call_wall_3",
        "put_wall_1",
        "put_wall_2",
        "put_wall_3",
        "atm_ce_iv",
        "atm_pe_iv",
        "atm_iv",
        "put_call_iv_skew",
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].copy()
    numeric_cols = [col for col in out.columns if col not in {"date", "symbol"}]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def load_silver_index_daily(
    silver_root: Path = SILVER_DATA_ROOT,
    index_name: str = "NIFTY 50",
) -> pd.DataFrame:
    df = pd.read_parquet(silver_root / "indices.parquet")
    df = df.copy()
    df["index_key"] = df["index_name"].astype(str).str.upper().str.strip()
    wanted = {index_name.upper(), "S&P CNX NIFTY", "CNX NIFTY", "NIFTY 50"}
    df = df[df["index_key"].isin(wanted)]
    if df.empty:
        return pd.DataFrame(columns=["date"])
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "nifty_close": pd.to_numeric(df["close"], errors="coerce"),
            "nifty_open": pd.to_numeric(df["open"], errors="coerce"),
            "nifty_high": pd.to_numeric(df["high"], errors="coerce"),
            "nifty_low": pd.to_numeric(df["low"], errors="coerce"),
            "nifty_volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    ).sort_values("date").drop_duplicates("date")


def load_silver_market_daily(silver_root: Path = SILVER_DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_training_universe(silver_root)
    cash = load_silver_stock_daily(silver_root, universe=universe)
    deriv = load_silver_derivatives_daily(silver_root, universe=universe)
    if not deriv.empty:
        cash = cash.merge(deriv, on=["date", "symbol"], how="left")
    indices = load_silver_index_daily(silver_root)
    return cash, indices


def load_index_daily(raw_root: Path = RAW_DATA_ROOT, index_name: str = "Nifty 50") -> pd.DataFrame:
    frames = []
    for path in sorted((raw_root / "indices").glob("*/*.csv")):
        try:
            df = _clean_columns(pd.read_csv(path))
        except Exception as exc:
            print(f"warn: failed index file {path}: {exc}")
            continue
        name_col = "Index Name"
        if name_col not in df:
            continue
        mask = df[name_col].astype(str).str.upper().isin({index_name.upper(), "S&P CNX NIFTY"})
        if not mask.any():
            continue
        part = df[mask].copy()
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(part["Index Date"], dayfirst=True, errors="coerce"),
                    "nifty_close": pd.to_numeric(part["Closing Index Value"], errors="coerce"),
                    "nifty_open": pd.to_numeric(part["Open Index Value"], errors="coerce"),
                    "nifty_high": pd.to_numeric(part["High Index Value"], errors="coerce"),
                    "nifty_low": pd.to_numeric(part["Low Index Value"], errors="coerce"),
                    "nifty_volume": pd.to_numeric(part["Volume"], errors="coerce"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
