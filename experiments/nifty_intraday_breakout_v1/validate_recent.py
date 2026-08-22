"""Single clean train/test split (NOT the walk-forward in train_magnitude.py): fit the
magnitude regressor on data through 2026-03-31 only, then score every bar from 2026-04-01 to
the latest available data, unseen at train time. Answers a different question than the walk-
forward's aggregate IC=0.53 (averaged across ~55 rolling folds spanning 2022-2026, mixing
regimes) -- this checks specifically whether the model still works on the MOST RECENT months,
which is what actually matters for trusting it live today.

No embargo gap needed at the March 31 / April 1 boundary: labels.py's forward-move labels are
STRICTLY same-day-only (a bar's fwd_move_mag_H never looks past that trading day's own close),
so a training row from March 31 can never see into April 1 -- the calendar cutoff alone is
leak-free.

Also runs the SAME fire/non-overlap/pattern-exit state machine production/nifty_live_poller.py
uses (not a naive "every bar where predicted>=threshold counts as a signal", which is what
would produce a new "signal" on every consecutive candle during a volatile stretch) over this
April-to-date window, using the April-onward OUT-OF-SAMPLE predictions -- so the backtested
signal count/hit-rate reflects what the live poller would actually have done, not an inflated
per-bar count.

Usage:
    python experiments/nifty_intraday_breakout_v1/validate_recent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_magnitude as tm  # noqa: E402

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "production"))

DATA = Path(__file__).resolve().parent / "data" / "v1_price_only.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRAIN_CUTOFF = pd.Timestamp("2026-03-31 23:59:59", tz="Asia/Kolkata")
HORIZON = 6
MOVE_THRESHOLD_PCT = 0.25


def fit_and_predict() -> pd.DataFrame:
    import lightgbm as lgb

    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = tm.feature_columns(df)
    target_col = f"fwd_move_mag_{HORIZON}"
    d = df.dropna(subset=[target_col] + feats, how="any")

    train = d[d["timestamp"] <= TRAIN_CUTOFF]
    test = d[d["timestamp"] > TRAIN_CUTOFF]
    print(f"[validate_recent] train: {len(train):,} rows through {train['timestamp'].max()}")
    print(f"[validate_recent] test:  {len(test):,} rows from {test['timestamp'].min()} to {test['timestamp'].max()}")

    model = lgb.LGBMRegressor(**tm.LGBM_PARAMS)
    model.fit(train[feats], train[target_col])
    pred = model.predict(test[feats])

    out = test[["timestamp", "date", "open", "high", "low", "close", target_col]].copy()
    out["pred"] = pred
    out = out.rename(columns={target_col: "y"})
    return out.reset_index(drop=True)


def summarize(preds: pd.DataFrame) -> dict:
    err = preds["pred"] - preds["y"]
    mae = err.abs().mean()
    ic, _ = spearmanr(preds["pred"], preds["y"])
    preds = preds.copy()
    preds["decile"] = pd.qcut(preds["pred"], 10, labels=False, duplicates="drop")
    decile_means = preds.groupby("decile")["y"].mean()
    ratio = (decile_means.iloc[-1] / decile_means.iloc[0]) if decile_means.iloc[0] > 0 else None
    return dict(n=len(preds), mean_actual_mag=round(float(preds["y"].mean()), 5),
                mae=round(float(mae), 5), spearman_ic=round(float(ic), 4),
                top_decile_vs_bottom_decile=round(float(ratio), 2) if ratio else None)


# ------------------------------------------------------------- non-overlap signal backtest
def _zones_by_date() -> dict:
    zones_path = ROOT / "locks" / "prod_nifty_intraday" / "nifty_zones.parquet"
    if not zones_path.exists():
        return {}
    z = pd.read_parquet(zones_path)
    z["date"] = pd.to_datetime(z["date"]).dt.date
    return {row["date"]: row.to_dict() for _, row in z.iterrows()}


def backtest_signals(preds: pd.DataFrame) -> pd.DataFrame:
    """Same fire/non-overlap/pattern-exit logic as production/nifty_live_poller.py's
    poll_once()/_check_exit() -- replayed here bar-by-bar over the historical OOS predictions,
    so 'how many signals fired' reflects the state machine, not a raw per-bar threshold count.
    Uses the SAME nifty_zones.parquet the live poller reads (that day's zone snapshot, looked
    up by date), not an empty/no-zones stub."""
    from nifty_live_poller import _check_exit   # noqa: E402  (needs production/ on sys.path)
    from koscine import nifty_zones as nz       # noqa: E402  (needs ROOT on sys.path)

    bars = preds[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
    zones_by_date = _zones_by_date()
    signals = []
    open_sig = None
    for i in range(len(preds)):
        row = preds.iloc[i]
        pred_pct = float(row["pred"]) * 100
        ts, price = row["timestamp"], float(row["close"])

        if open_sig is not None:
            # both zone timeframes computed from bars.iloc[:i+1] ONLY -- the same leakage-safe
            # slice _check_exit itself gets, so a "signal" here never sees a zone that was only
            # discoverable from bars ahead of the row being evaluated (2026-08, explicit user
            # requirement: past-date signals must be fireable using only data available as of
            # that candle, no future data).
            todays_zones = zones_by_date.get(pd.Timestamp(open_sig["entry_ts"]).date(), {})
            zones_15m = nz.latest_intraday_zones(bars.iloc[:i + 1])
            exit_info = _check_exit(open_sig, bars.iloc[:i + 1], todays_zones, zones_15m)
            if exit_info is not None:
                open_sig.update(exit_info)
                signals.append(open_sig)
                open_sig = None

        if open_sig is None and pred_pct >= MOVE_THRESHOLD_PCT:
            open_sig = {"entry_ts": ts.isoformat(), "entry_underlying": price,
                        "predicted_move_pct": round(pred_pct, 3), "direction": None, "peak_price": None}
    if open_sig is not None:
        signals.append(open_sig)   # still open at the end of the window -- report as unresolved
    return pd.DataFrame(signals)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = fit_and_predict()
    preds.to_csv(RESULTS_DIR / "validate_recent_preds.csv", index=False)

    summary = summarize(preds)
    print("\n[validate_recent] out-of-sample summary (2026-04-01 -> latest):")
    print(f"  {summary}")

    sig_df = backtest_signals(preds)
    n_resolved = int(sig_df["exit_reason"].notna().sum()) if len(sig_df) else 0
    print(f"\n[validate_recent] signals fired (non-overlap state machine): {len(sig_df)} "
          f"({n_resolved} resolved, {len(sig_df) - n_resolved} still open at window end)")
    if len(sig_df) and n_resolved:
        resolved = sig_df[sig_df["exit_reason"].notna()]
        win_rate = (resolved["realized_move_pct"].abs() >= MOVE_THRESHOLD_PCT).mean()
        print(f"  mean realized |move|: {resolved['realized_move_pct'].abs().mean():.3f}%")
        print(f"  hit rate (|realized| >= {MOVE_THRESHOLD_PCT}% threshold): {win_rate:.1%}")
        print(f"  exit reasons: {resolved['exit_reason'].value_counts().to_dict()}")
    sig_df.to_csv(RESULTS_DIR / "validate_recent_signals.csv", index=False)

    (RESULTS_DIR / "FINDINGS_validate_recent.md").write_text(
        "# NIFTY magnitude model -- train<=2026-03-31, validate 2026-04-01 onward\n\n"
        f"```\n{summary}\n```\n\n"
        f"Signals fired (non-overlap state machine): {len(sig_df)} ({n_resolved} resolved)\n",
        encoding="utf-8")
    print(f"\n[validate_recent] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
