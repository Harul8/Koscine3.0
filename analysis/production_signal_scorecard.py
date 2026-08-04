"""Reproducible, target-aware scorecard for the locked production signal books.

This intentionally reports each book against its own realised target.  It does
not declare a winner across incompatible entry rules (v2 starts at t+1 open;
v3 and the 1-day estimators start at t close).  A promotion requires a common
walk-forward experiment with one universe, entry time, liquidity rule and cost
model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "locks"
REPORTS = ROOT / "reports"


def _load(relative: str) -> pd.DataFrame:
    frame = pd.read_csv(LOCKS / relative)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _summary(frame: pd.DataFrame, outcome: str, thresholds: tuple[float, ...]) -> dict[str, object]:
    observed = frame.dropna(subset=[outcome]).copy()
    by_year: dict[str, dict[str, float | int]] = {}
    for year, group in observed.groupby(observed.date.dt.year):
        by_year[str(year)] = {"rows": int(len(group)), "mean_pct": round(float(group[outcome].mean()), 2), **{
            f"hit_ge_{threshold:g}_pct": round(float((group[outcome] >= threshold).mean() * 100), 1)
            for threshold in thresholds
        }}
    return {
        "rows": int(len(observed)),
        "date_range": [observed.date.min().strftime("%Y-%m-%d"), observed.date.max().strftime("%Y-%m-%d")],
        "mean_pct": round(float(observed[outcome].mean()), 2),
        **{f"hit_ge_{threshold:g}_pct": round(float((observed[outcome] >= threshold).mean() * 100), 1)
           for threshold in thresholds},
        "by_year": by_year,
    }


def build_scorecard() -> dict[str, object]:
    v2 = _load("prod_largemove_v2/book_2024_26.csv")
    v3_5 = _load("prod_largemove_v3/mover_v3_book_5d.csv")
    v3_1 = _load("prod_largemove_v3/mover_v3_book_1d.csv")
    next_day = _load("prod_largemove_v2/next_day_book.csv")
    expected = _load("prod_expected_move_v1/expected_move_book.csv")
    direction = _load("prod_direction_v1/direction_v1_book_B.csv")

    v2["realised_peak_pct"] = v2.move_mag * 100
    v3_5["realised_peak_pct"] = v3_5.move_mag * 100
    v3_1["realised_peak_pct"] = v3_1.move_mag * 100
    next_day["absolute_error_pct"] = (next_day.pred_move_pct - next_day.next_move_pct).abs()
    # expected_move_v1 persists both the prediction and realised close move in percent units.
    expected["absolute_error_pct"] = (expected.exp_move_pct - expected.fwd1_abs_pct).abs()

    card: dict[str, object] = {
        "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "decision": {
            "production_selector": "v2 ATM-IV rank remains the locked 5-day selector.",
            "product_role": "v3 is the primary signal presentation and conviction overlay; expected-move and next-day models are sizing estimates, not replacements for the 5-day selector.",
            "promotion_gate": "Do not promote v3 or a new ML selector until it beats v2 in a common walk-forward protocol, including identical universe, entry timestamp, liquidity screen, turnover/cooldown, option-cost and slippage assumptions.",
        },
        "comparability": [
            "v2 realised peak is measured from t+1 open over five sessions; v3 peak is measured from t close.",
            "v3 applies an ATM+2% liquidity gate; v2's locked book uses a different eligibility and cooldown policy.",
            "next-day peak move and next-day close-to-close expected move are distinct targets; their errors are not comparable to five-day peak-move hit rates.",
        ],
        "books": {
            "v2_5d_selector": {"target": "five-day peak excursion from t+1 open", **_summary(v2, "realised_peak_pct", (4, 6, 8))},
            "v3_5d_overlay": {"target": "five-day peak excursion from t close", **_summary(v3_5, "realised_peak_pct", (4, 6, 8))},
            "v3_1d_overlay": {"target": "one-day peak excursion from t close", **_summary(v3_1, "realised_peak_pct", (2, 4, 6))},
            "next_day_peak_estimate": {
                "target": "next-session high/low excursion from t close",
                **_summary(next_day, "next_move_pct", (2, 3, 4)),
                "mae_pct": round(float(next_day.absolute_error_pct.mean()), 2),
                "rank_ic": round(float(next_day[["pred_move_pct", "next_move_pct"]].corr(method="spearman").iloc[0, 1]), 3),
            },
            "expected_move_v1": {
                "target": "absolute next-day close-to-close return",
                **_summary(expected.assign(realised_close_move_pct=expected.fwd1_abs_pct), "realised_close_move_pct", (1, 2, 3)),
                "mae_pct": round(float(expected.absolute_error_pct.mean()), 2),
                "rank_ic": round(float(expected[["exp_move_pct", "fwd1_abs_pct"]].corr(method="spearman").iloc[0, 1]), 3),
            },
            "direction_v1_b_5d": {
                "target": "sign of five-day close return; B group only",
                "rows": int(direction.lean_correct.notna().sum()),
                "accuracy_pct": round(float(direction.lean_correct.mean() * 100), 1),
                "note": "A small tilt only; it must never turn a direction-agnostic magnitude trade into a directional mandate.",
            },
        },
    }
    return card


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    card = build_scorecard()
    (REPORTS / "production_signal_scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    rows = []
    for name, details in card["books"].items():
        rows.append({"signal": name, "target": details.get("target"), "rows": details.get("rows"),
                     "mean_pct": details.get("mean_pct"), "hit_ge_4_pct": details.get("hit_ge_4_pct"),
                     "hit_ge_6_pct": details.get("hit_ge_6_pct"), "rank_ic": details.get("rank_ic"), "mae_pct": details.get("mae_pct")})
    pd.DataFrame(rows).to_csv(REPORTS / "production_signal_scorecard.csv", index=False)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
