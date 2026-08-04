from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = Path(r"C:\Users\rahul\Koscine 3.0\data\raw")
SILVER_DATA_ROOT = Path(r"C:\Users\rahul\Koscine 3.0\data\silver")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
BACKTEST_DIR = PROJECT_ROOT / "backtests"
RUNS_DIR = PROJECT_ROOT / "runs"

HORIZON_DAYS = 5
MOVE_THRESHOLDS = (0.05, 0.07)
PURGE_DAYS = 5
DEFAULT_COST_BPS = 20.0
DEFAULT_TOP_N = 5

TARGET_UNIVERSE = [
    "HDFCBANK",
    "INFY",
    "TCS",
    "RELIANCE",
    "ICICIBANK",
    "HCLTECH",
    "BHARTIARTL",
    "ITC",
    "SBIN",
    "BAJFINANCE",
    "WIPRO",
    "KOTAKBANK",
    "APOLLOHOSP",
    "M&M",
    "AXISBANK",
    "ETERNAL",
    "CHOLAFIN",
    "LT",
    "TMPV",
    "JSWSTEEL",
    "MARUTI",
    "HINDUNILVR",
    "JIOFIN",
    "CIPLA",
    "MUTHOOTFIN",
    "ADANIENT",
    "ASIANPAINT",
    "INDUSINDBK",
    "AMBUJACEM",
    "TITAN",
    "SUNPHARMA",
    "TVSMOTOR",
    "SHRIRAMFIN",
    "INDIGO",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "VEDL",
    "ADANIPORTS",
    "PFC",
    "ULTRACEMCO",
    "RECLTD",
    "ONGC",
    "PATANJALI",
    "LODHA",
    "SWIGGY",
    "INDUSTOWER",
    "BHEL",
    "HDFCLIFE",
    "TATAPOWER",
    "NESTLEIND",
    "EICHERMOT",
    "SAIL",
    "POWERGRID",
    "ADANIGREEN",
    "YESBANK",
    "SUZLON",
    "IDFCFIRSTB",
    "RVNL",
    "UNITDSPR",
    "SBICARD",
    "LICHSGFIN",
    "AUROPHARMA",
    "TECHM",
    "JUBLFOOD",
    "BANKBARODA",
    "ADANIENSOL",
    "KAYNES",
    "PNB",
    "TATASTEEL",
    "GODREJPROP",
    "LTM",
    "VOLTAS",
    "GLENMARK",
    "DIVISLAB",
    "BPCL",
    "HEROMOTOCO",
    "HINDZINC",
    "WAAREEENER",
    "JSWENERGY",
    "MAXHEALTH",
    "SBILIFE",
    "ASTRAL",
    "CANBK",
    "CROMPTON",
    "OBEROIRLTY",
    "DMART",
    "ASHOKLEY",
    "COLPAL",
    "INOXWIND",
    "NAUKRI",
    "HYUNDAI",
    "RBLBANK",
    "IDEA",
    "TRENT",
    "BEL",
    "DIXON",
    "PERSISTENT",
    "COALINDIA",
    "HINDALCO",
    "GRASIM",
    "COFORGE",
    "NMDC",
    "HAL",
    "VBL",
    "NTPC",
    "DLF",
]

TOP30_LIQUID_UNIVERSE = TARGET_UNIVERSE[:30]
REST_UNIVERSE = TARGET_UNIVERSE[30:]


SYMBOL_ALIASES = {
    # NSE renamed Zomato to Eternal in 2025; historical cash data may use either.
    "ZOMATO": "ETERNAL",
    "ETERNAL": "ETERNAL",
    # Tata Motors passenger vehicles symbol changed in 2025; historical data uses TATAMOTORS.
    "TATAMOTORS": "TMPV",
    "TMPV": "TMPV",
    # Naukri's exchange symbol is INFOEDGE.
    "INFOEDGE": "NAUKRI",
    "NAUKRI": "NAUKRI",
    # User's list has LTM; exchange history uses LTI/LTIM around the merger.
    "LTI": "LTM",
    "LTIM": "LTM",
    "LTM": "LTM",
    # Macrotech Developers commonly maps to Lodha.
    "LODHA": "LODHA",
    # Larsen & Toubro is LT in NSE files.
    "L&T": "LT",
}
