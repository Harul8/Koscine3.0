"""
Parse raw SEBI FII delivery trade disclosure zips into a silver parquet.

Each zip contains one CSV or XLSX file with individual FII equity delivery trades.
We filter to equity buy/sell (TR_TYPE 1/4, RFDE_INSTR_TYPE REG_DL_INSTR_EQ) and
aggregate per (ISIN, trade_date): net_value, buy_value, sell_value, n_buyers, n_sellers.

Output: silver/fii_stock_trades.parquet  — columns:
    date, isin, symbol, net_value, buy_value, sell_value, n_buyers, n_sellers

Run:
    python -m pipeline.fii_proxy.parse
"""
from __future__ import annotations
import os
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = Path(r"C:\Users\rahul\Koscine 3.0\data\raw\fii_trades")
OUT_FILE = Path(r"C:\Users\rahul\Koscine 3.0\data\silver\fii_stock_trades.parquet")

# ISIN -> NSE symbol for our F&O universe.
# ISINs can change due to mergers / face-value splits; list the most recent one.
# Older ISINs for the same company are mapped to the same symbol so historical
# data is preserved.
ISIN_TO_SYMBOL: dict[str, str] = {
    # ── Large-cap Financials ──────────────────────────────────────────────────
    "INE002A01018": "RELIANCE",
    "INE040A01034": "HDFCBANK",      # post-merger FV1
    "INE040A01026": "HDFCBANK",      # pre-merger FV2
    "INE090A01021": "ICICIBANK",
    "INE090A01013": "ICICIBANK",     # older ISIN
    "INE062A01020": "SBIN",
    "INE062A01012": "SBIN",          # older ISIN
    "INE238A01034": "AXISBANK",
    "INE238A01026": "AXISBANK",      # older ISIN
    "INE237A01028": "KOTAKBANK",
    "INE545U01014": "BANDHANBNK",
    "INE296A01024": "BAJFINANCE",
    "INE918I01026": "BAJAJFINSV",
    "INE118A01012": "BAJAJHLDNG",
    "INE121A01024": "CHOLAFIN",
    "INE414G01012": "MUTHOOTFIN",
    "INE522D01027": "MANAPPURAM",
    "INE160A01022": "PNB",
    "INE028A01039": "BANKBARODA",
    "INE476A01022": "CANBK",
    "INE084A01016": "BANKINDIA",
    "INE483A01010": "CENTRALBK",
    "INE562A01011": "INDIANB",
    "INE692A01016": "UNIONBANK",
    "INE171A01029": "FEDERALBNK",
    "INE774D01024": "M&MFIN",
    "INE092T01019": "IDFCFIRSTB",
    "INE976G01028": "RBLBANK",
    "INE528G01035": "YESBANK",
    "INE949L01017": "AUBANK",
    "INE063P01018": "EQUITAS",
    "INE491A01021": "CUB",
    "INE503A01015": "DCBBANK",
    "INE614B01018": "KTKBANK",
    "INE683A01023": "SOUTHBANK",
    "INE551W01018": "UJJIVAN",
    # Insurance / AMC
    "INE765G01017": "ICICIGI",
    "INE726G01019": "ICICIPRULI",
    "INE795G01014": "HDFCLIFE",
    "INE127D01025": "HDFCAMC",
    "INE123W01016": "SBILIFE",
    "INE018E01016": "SBICARD",
    "INE298J01013": "NAM-INDIA",
    "INE575P01011": "STAR",
    "INE0J1Y01017": "LICI",
    # NBFCs / Housing
    "INE115A01026": "LICHSGFIN",
    "INE572E01012": "PNBHOUSING",
    "INE477A01020": "CANFINHOME",
    "INE148I01020": "IBULHSGFIN",
    "INE511C01022": "POONAWALLA",
    "INE498L01015": "L&TFH",
    "INE530B01024": "IIFL",
    "INE674K01013": "ABCAPITAL",
    # Exchanges / Fintech
    "INE736A01011": "CDSL",
    "INE118H01025": "BSE",
    "INE745G01035": "MCX",
    "INE022Q01020": "IEX",
    "INE596I01012": "CAMS",
    "INE138Y01010": "KFINTECH",
    "INE531F01015": "NUVAMA",
    "INE417T01026": "POLICYBZR",
    "INE732I01013": "ANGELONE",
    # ── IT / Technology ──────────────────────────────────────────────────────
    "INE467B01029": "TCS",
    "INE009A01021": "INFY",
    "INE075A01022": "WIPRO",
    "INE860A01027": "HCLTECH",
    "INE669C01036": "TECHM",
    "INE262H01021": "PERSISTENT",
    "INE591G01017": "COFORGE",
    "INE356A01018": "MPHASIS",
    "INE010V01017": "LTTS",
    "INE04I401011": "KPIT",
    "INE881D01027": "OFSS",
    "INE136B01020": "CYIENT",
    "INE836A01035": "BSOFT",
    "INE670A01012": "TATAELXSI",
    "INE306R01017": "INTELLECT",
    # ── Auto / Ancillaries ───────────────────────────────────────────────────
    "INE585B01010": "MARUTI",
    "INE101A01026": "M&M",
    "INE158A01026": "HEROMOTOCO",
    "INE917I01010": "BAJAJ-AUTO",
    "INE494B01023": "TVSMOTOR",
    "INE155A01022": "TATAMOTORS",
    "INE066A01021": "EICHERMOT",
    "INE208A01029": "ASHOKLEY",
    "INE042A01014": "ESCORTS",
    "INE787D01026": "BALKRISIND",
    "INE482A01020": "CEATLTD",
    "INE438A01022": "APOLLOTYRE",
    "INE883A01011": "MRF",
    "INE302A01020": "EXIDEIND",
    "INE775A01035": "MOTHERSON",
    "INE405E01023": "UNOMINDA",
    "INE974X01010": "TIINDIA",
    # ── FMCG / Consumer ──────────────────────────────────────────────────────
    "INE030A01027": "HINDUNILVR",
    "INE154A01025": "ITC",
    "INE239A01024": "NESTLEIND",
    "INE021A01026": "ASIANPAINT",
    "INE016A01026": "DABUR",
    "INE196A01026": "MARICO",
    "INE259A01022": "COLPAL",
    "INE216A01030": "BRITANNIA",
    "INE200M01039": "VBL",
    "INE102D01028": "GODREJCP",
    "INE192A01025": "TATACONSUM",
    "INE854D01024": "MCDOWELL-N",
    "INE686F01025": "UBL",
    "INE761H01022": "PAGEIND",
    "INE647O01011": "ABFRL",
    "INE301A01014": "RAYMOND",
    "INE463A01038": "BERGEPAINT",
    "INE172A01027": "CASTROLIND",
    "INE233A01035": "GODREJIND",
    "INE180A01020": "MFSL",
    # ── Pharma / Healthcare ──────────────────────────────────────────────────
    "INE044A01036": "SUNPHARMA",
    "INE059A01026": "CIPLA",
    "INE437A01024": "APOLLOHOSP",
    "INE089A01031": "DRREDDY",
    "INE326A01037": "LUPIN",
    "INE361B01024": "DIVISLAB",
    "INE406A01037": "AUROPHARMA",
    "INE376G01013": "BIOCON",
    "INE935A01035": "GLENMARK",
    "INE571A01038": "IPCALAB",
    "INE031B01049": "AJANTPHARM",
    "INE540L01014": "ALKEM",
    "INE634S01028": "MANKIND",
    "INE027H01010": "MAXHEALTH",
    "INE398R01022": "SYNGENE",
    "INE947Q01028": "LAURUSLABS",
    "INE101D01020": "GRANULES",
    "INE600L01024": "LALPATHLAB",
    "INE112L01020": "METROPOLIS",
    "INE939A01011": "STRTECH",
    "INE049B01025": "WOCKPHARMA",
    # ── Energy / Oil & Gas ───────────────────────────────────────────────────
    "INE081A01020": "TATASTEEL",
    "INE081A01012": "TATASTEEL",     # older FV10
    "INE397D01024": "BHARTIARTL",
    "INE733E01010": "NTPC",
    "INE213A01029": "ONGC",
    "INE245A01021": "TATAPOWER",
    "INE742F01042": "ADANIPORTS",
    "INE423A01024": "ADANIENT",
    "INE029A01011": "BPCL",
    "INE242A01010": "IOC",
    "INE094A01015": "HINDPETRO",
    "INE129A01019": "GAIL",
    "INE347G01014": "PETRONET",
    "INE203G01027": "IGL",
    "INE002S01010": "MGL",
    "INE844O01030": "GUJGASLTD",
    "INE246F01010": "GSPL",
    "INE267A01025": "HINDZINC",
    "INE205A01025": "VEDL",
    "INE522F01014": "COALINDIA",
    "INE584A01023": "NMDC",
    "INE490G01020": "MOIL",
    "INE139A01034": "NATIONALUM",
    "INE531E01026": "HINDCOPPER",
    "INE399L01023": "ATGL",
    "INE364U01010": "ADANIGREEN",
    "INE814H01011": "ADANIPOWER",
    "INE931S01010": "ADANIENSOL",
    # ── Power / Utilities ────────────────────────────────────────────────────
    "INE752E01010": "POWERGRID",
    "INE134E01011": "PFC",
    "INE020B01018": "RECLTD",
    "INE848E01016": "NHPC",
    "INE002L01015": "SJVN",
    "INE031A01017": "HUDCO",
    "INE486A01021": "CESC",
    "INE121E01018": "JSWENERGY",
    "INE813H01021": "TORNTPOWER",
    "INE040H01021": "SUZLON",
    "INE066P01011": "INOXWIND",
    "INE377N01017": "WAAREEENER",
    # ── Capital Goods / Engineering ──────────────────────────────────────────
    "INE018A01030": "LT",
    "INE257A01026": "BHEL",
    "INE003A01024": "SIEMENS",
    "INE117A01022": "ABB",
    "INE323A01026": "BOSCHLTD",
    "INE298A01020": "CUMMINSIND",
    "INE176B01034": "HAVELLS",
    "INE226A01021": "VOLTAS",
    "INE472A01039": "BLUESTARCO",
    "INE067A01029": "CGPOWER",
    "INE263A01024": "BEL",
    "INE258A01016": "BEML",
    "INE171Z01026": "BDL",
    "INE066F01020": "HAL",
    "INE249Z01012": "MAZDOCK",
    "INE615H01020": "TITAGARH",
    "INE918Z01012": "KAYNES",
    "INE465A01025": "BHARATFORG",
    "INE661I01014": "BGRENERGY",
    "INE510A01028": "ENGINERSIN",
    # ── Metals / Steel ───────────────────────────────────────────────────────
    "INE038A01020": "HINDALCO",
    "INE019A01038": "JSWSTEEL",
    "INE114A01011": "SAIL",
    "INE749A01030": "JINDALSTEL",
    "INE324A01032": "JINDALSAW",
    "INE191B01025": "WELCORP",
    "INE855B01025": "RAIN",
    # ── Cement / Building Materials ──────────────────────────────────────────
    "INE481G01011": "ULTRACEMCO",
    "INE070A01015": "SHREECEM",
    "INE012A01025": "ACC",
    "INE079A01024": "AMBUJACEM",
    "INE331A01037": "RAMCOCEM",
    "INE00R701025": "DALBHARAT",
    "INE823G01014": "JKCEMENT",
    "INE702C01027": "APLAPOLLO",
    "INE455K01017": "POLYCAB",
    "INE878B01027": "KEI",
    "INE006I01046": "ASTRAL",
    "INE217B01036": "KAJARIACER",
    # ── Chemicals / Agri ─────────────────────────────────────────────────────
    "INE769A01020": "AARTIIND",
    "INE092A01019": "TATACHEM",
    "INE628A01036": "UPL",
    "INE603J01030": "PIIND",
    "INE169A01031": "COROMANDEL",
    "INE085A01013": "CHAMBLFERT",
    "INE026A01025": "GSFC",
    "INE113A01013": "GNFC",
    "INE09N301011": "GUJFLUORO",
    "INE647A01010": "SRF",
    "INE100A01010": "ATUL",
    "INE288B01029": "DEEPAKNTR",
    "INE048G01026": "NAVINFLUOR",
    "INE343H01029": "SOLARINDS",
    # ── Real Estate / Hotels ─────────────────────────────────────────────────
    "INE271C01023": "DLF",
    "INE484J01027": "GODREJPROP",
    "INE093I01010": "OBEROIRLTY",
    "INE811K01011": "PRESTIGE",
    "INE670K01029": "LODHA",
    "INE671H01015": "SOBHA",
    "INE211B01039": "PHOENIXLTD",
    "INE053A01029": "INDHOTEL",
    # ── Infrastructure / Logistics ───────────────────────────────────────────
    "INE111A01025": "CONCOR",
    "INE148O01028": "DELHIVERY",
    "INE415G01027": "RVNL",
    "INE335Y01020": "IRCTC",
    "INE053F01010": "IRFC",
    "INE776C01039": "GMRAIRPORT",
    "INE455F01025": "JPASSOCIAT",
    "INE351F01018": "JPPOWER",
    "INE017A01032": "GESHIP",
    "INE109A01011": "SCI",
    "INE868B01028": "NCC",
    "INE074A01025": "PRAJIND",
    # ── Consumer / Retail / New-age ──────────────────────────────────────────
    "INE192R01011": "DMART",
    "INE933S01016": "INDIAMART",
    "INE663F01024": "NAUKRI",
    "INE758T01015": "ZOMATO",
    "INE982J01020": "PAYTM",
    "INE388Y01029": "NYKAA",
    "INE758E01017": "JIOFIN",
    "INE00H001014": "SWIGGY",
    "INE797F01020": "JUBLFOOD",
    "INE424H01027": "SUNTV",
    "INE256A01028": "ZEEL",
    "INE191H01014": "PVRINOX",
    "INE646L01027": "INDIGO",
    "INE599M01018": "JUSTDIAL",
    "INE935N01020": "DIXON",
    "INE371P01015": "AMBER",
    "INE124G01033": "DELTACORP",
    "INE280A01028": "TITAN",
    "INE849A01020": "TRENT",
    "INE142M01025": "TATATECH",
    # ── Telecom / Media ──────────────────────────────────────────────────────
    "INE153A01019": "MTNL",
    "INE669E01016": "IDEA",
    "INE836F01026": "DISHTV",
    "INE151A01013": "TATACOMM",
    "INE517B01013": "TTML",
    "INE685A01028": "TORNTPHARM",
    # ── Energy / Refining ────────────────────────────────────────────────────
    "INE103A01014": "MRPL",
    "INE178A01016": "CHENNPETRO",
    "INE345A01011": "HINDOILEXP",
    # ── Consumer / Other ─────────────────────────────────────────────────────
    "INE176A01028": "BATAINDIA",
    "INE885A01032": "AMARAJABAT",
    "INE119A01028": "BALRAMCHIN",
    "INE131A01031": "GMDCLTD",
    "INE483S01020": "INFIBEAM",
    "INE612J01015": "REPCOHOME",
    # ── Misc ─────────────────────────────────────────────────────────────────
    "INE466L01038": "360ONE",
    "INE748C01038": "3IINFOTECH",
}


def _read_zip(fpath: Path) -> pd.DataFrame | None:
    try:
        z = zipfile.ZipFile(fpath)
    except Exception:
        return None
    inner = z.namelist()[0]
    try:
        with z.open(inner) as f:
            if inner.lower().endswith(".xlsx"):
                import io
                data = f.read()
                df = pd.read_excel(io.BytesIO(data), engine="calamine")
            else:
                df = pd.read_csv(f, low_memory=False)
    except Exception:
        return None
    return df


def _parse_file(fpath: Path) -> pd.DataFrame:
    df = _read_zip(fpath)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    required = {"TR_TYPE(*)", "RFDE_INSTR_TYPE", "ISIN", "TR_DATE",
                 "VALUE (in Rs)", "FII"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Equity delivery buy/sell only
    df = df[
        df["TR_TYPE(*)"].isin([1, 4]) &
        (df["RFDE_INSTR_TYPE"] == "REG_DL_INSTR_EQ")
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["TR_DATE"], dayfirst=False, errors="coerce")
    df["value"] = pd.to_numeric(df["VALUE (in Rs)"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "ISIN"])
    df["ISIN"] = df["ISIN"].astype(str).str.strip()
    df["fii_id"] = df["FII"].astype(str).str.strip()
    df["is_buy"] = df["TR_TYPE(*)"] == 1

    return df[["date", "ISIN", "fii_id", "is_buy", "value"]]


def build_silver() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.zip"))
    print(f"[fii_proxy.parse] Processing {len(files)} zip files …")

    chunks: list[pd.DataFrame] = []
    skipped = 0
    for fp in files:
        chunk = _parse_file(fp)
        if chunk.empty:
            skipped += 1
        else:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No valid FII trade data found.")

    raw = pd.concat(chunks, ignore_index=True)
    print(f"  Raw rows (equity buy/sell): {len(raw):,}  |  skipped files: {skipped}")

    # Aggregate per (ISIN, date)
    buys  = raw[raw["is_buy"]]
    sells = raw[~raw["is_buy"]]

    b = buys.groupby(["ISIN", "date"]).agg(
        buy_value=("value", "sum"),
        n_buyers=("fii_id", "nunique"),
    ).reset_index()
    s = sells.groupby(["ISIN", "date"]).agg(
        sell_value=("value", "sum"),
        n_sellers=("fii_id", "nunique"),
    ).reset_index()

    agg = b.merge(s, on=["ISIN", "date"], how="outer").fillna(0.0)
    agg["net_value"] = agg["buy_value"] - agg["sell_value"]

    # Map ISIN -> symbol (restrict to our universe)
    agg["symbol"] = agg["ISIN"].map(ISIN_TO_SYMBOL)
    known = agg.dropna(subset=["symbol"]).copy()
    print(f"  After ISIN -> symbol mapping: {len(known):,} rows "
          f"({known['symbol'].nunique()} symbols, {known['date'].nunique()} dates)")

    known = known.sort_values(["symbol", "date"]).reset_index(drop=True)
    known["date"] = known["date"].astype("datetime64[ns]")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    known.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")
    return known


if __name__ == "__main__":
    build_silver()
