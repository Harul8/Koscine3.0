"""
Two universe files:

  training_universe.txt — every symbol ever in eod_deriv_daily with ≥ MIN_TRAINING_DAYS
                          of data; used to maximise cross-sectional breadth during training.

  active_universe.txt   — 43 TARGETS + top-N by avg fut_oi over the last 3 months;
                          used for production inference (only currently active stocks).
"""
import sys
import pandas as pd
from .config import (SILVER_TABLES, SILVER_DIR, TARGETS, INDEX_TARGETS,
                     PREDICT_UNIVERSE_FILE, LIQUID_TIER_SIZE)

UNIVERSE_FILE          = SILVER_DIR / "active_universe.txt"
TRAINING_UNIVERSE_FILE = SILVER_DIR / "training_universe.txt"
LOOKBACK_DAYS  = 90    # ~3 months (active universe)
EXTRA_N        = 57    # 43 targets + 57 extras = 100 (+ 3 indices ~= 103)
MIN_TRAINING_DAYS = 100  # minimum trading-day observations to include in training universe


def build_training(min_days: int = MIN_TRAINING_DAYS) -> list[str]:
    """All-time F&O universe: every symbol with ≥ min_days observations in eod_deriv_daily."""
    dd = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"])
    if dd.empty:
        print("[universe] eod_deriv_daily empty"); return []

    day_counts = dd.groupby("symbol")["date"].nunique()
    eligible   = sorted(day_counts[day_counts >= min_days].index.tolist())

    TRAINING_UNIVERSE_FILE.write_text("\n".join(eligible))
    print(f"[universe] training: {len(eligible)} symbols (≥{min_days} days)")

    # breakdown by era
    for year in [2015, 2018, 2021, 2024]:
        era_syms = dd[dd["date"].dt.year <= year]["symbol"].nunique()
        print(f"  unique symbols through {year}: {era_syms}")

    return eligible


def load_training() -> list[str]:
    if TRAINING_UNIVERSE_FILE.exists():
        return [s.strip() for s in TRAINING_UNIVERSE_FILE.read_text().splitlines() if s.strip()]
    return []


def build(extra_n: int = EXTRA_N, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    dd = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"])
    if dd.empty:
        print("[universe] eod_deriv_daily empty"); return []
    cutoff = dd["date"].max() - pd.Timedelta(days=lookback_days)
    recent = dd[(dd["date"] >= cutoff) & dd["fut_oi"].notna()].copy()

    # join lot sizes so we can rank by OI-in-lots rather than raw contract OI
    lot_path = SILVER_TABLES["lot_size"]
    if lot_path.exists():
        ls = pd.read_parquet(lot_path)[["symbol", "lot"]].drop_duplicates("symbol")
        recent = recent.merge(ls, on="symbol", how="left")
        recent["lot"] = recent["lot"].fillna(1)
        recent["lot_size"] = recent["lot"]
        recent["lots_oi"] = recent["fut_oi"] / recent["lot"]
    else:
        recent["lots_oi"] = recent["fut_oi"]

    agg = recent.groupby("symbol").agg(
        lots_oi_avg  = ("lots_oi", "mean"),
        days_present = ("date",    "nunique"),
    )
    agg = agg[agg["days_present"] >= 20]   # minimum presence over the 3m window

    must = set(TARGETS) | set(INDEX_TARGETS)
    ranked = agg.drop(index=[s for s in must if s in agg.index]) \
                .sort_values("lots_oi_avg", ascending=False)
    extras   = ranked.head(extra_n).index
    universe = sorted(must | set(extras))

    UNIVERSE_FILE.write_text("\n".join(universe))
    print(f"[universe] wrote {len(universe)} symbols -> {UNIVERSE_FILE}")
    print(f"  must-include: {len(must)}")
    print(f"  extras (top-{extra_n} by 3m avg oi in lots):")
    for s, v in ranked.head(extra_n)["lots_oi_avg"].items():
        print(f"    {s:<16} {v:>12,.0f} lots")
    return universe

def load() -> list[str]:
    if UNIVERSE_FILE.exists():
        return [s.strip() for s in UNIVERSE_FILE.read_text().splitlines() if s.strip()]
    return []


def load_predict_universe() -> tuple[list[str], set[str]]:
    """
    Load ESN1.0/Universe — the hand-curated, liquidity-ordered predict universe.

    Returns
    -------
    symbols     : list[str]  — all symbols in file order (blanks stripped)
    liquid_set  : set[str]   — first LIQUID_TIER_SIZE symbols = liquid tier
    """
    if not PREDICT_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Predict universe file not found: {PREDICT_UNIVERSE_FILE}\n"
            "Create ESN1.0/Universe with one symbol per line, most liquid first."
        )
    symbols = [s.strip() for s in PREDICT_UNIVERSE_FILE.read_text().splitlines()
               if s.strip() and not s.strip().startswith("#")]
    liquid_set = set(symbols[:LIQUID_TIER_SIZE])
    return symbols, liquid_set


if __name__ == "__main__":
    # python -m pipeline.universe           -> active universe only
    # python -m pipeline.universe training  -> training universe only
    # python -m pipeline.universe both      -> both
    mode    = sys.argv[1] if len(sys.argv) > 1 else "active"
    extra_n = int(sys.argv[2]) if len(sys.argv) > 2 else EXTRA_N
    if mode in ("training", "both"):
        build_training()
    if mode in ("active", "both"):
        build(extra_n=extra_n)
