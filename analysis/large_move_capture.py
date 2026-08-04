"""Large-move capture for the options-buying strategy (no stop; downside = premium).

Objective: of the daily picks, how often do they land among the day's TOP MOVERS and
reach large favourable moves (>=8/10/12/15%)? Downside is irrelevant (capped at premium).

Model: per side, P(ceiling >= BIG_TRAIN) classifier + E[ceiling] regressor, trained <=2023.
Book : pick top-1 long (call) + top-1 short (put) per day, ranked by P(big). 2 options/day.
Reports big-move hit rates, top-K mover precision, the user's daily-success rate, and a
rough convexity EV at the user's 12%->5x assumption. Baseline = random within universe.
Read-only research.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from koscine3.data.feature_registry import build_feature_registry  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.data.universe import UniverseConfig, build_universe  # noqa: E402
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns  # noqa: E402
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes  # noqa: E402

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-31")
BIG_TRAIN = 0.10            # training target: ceiling >= 10%
THRESHOLDS = [0.06, 0.08, 0.10, 0.12, 0.15]
WIN_MULT = 5.0             # user assumption: a >=12% move makes the option ~5x
WIN_AT = 0.12


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def main() -> None:
    print("loading + building features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=100))
    rank = universe.set_index(universe["symbol"].astype(str))["rank"]
    tier_of = rank.apply(lambda r: "TOP30" if r <= 30 else "MID_31_100")

    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    outc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    outc = outc[outc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling", "days_to_peak"]].copy()
    outc["symbol"] = outc["symbol"].astype(str)
    df = dataset.merge(outc, on=["date", "symbol", "side"], how="inner")
    df["tier"] = df["symbol"].map(tier_of)
    # rank within (date, side): is this stock a top mover today?
    df["mover_rank"] = df.groupby(["date", "side"])["ceiling"].rank(method="min", ascending=False)
    train, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END]

    picks_by_side = {}
    for side in ("long", "short"):
        tr_s, ev_s = train[train["side"].eq(side)], evl[evl["side"].eq(side)].copy()
        imp = SimpleImputer(strategy="median").fit(_clean(tr_s, feats))
        Xtr, Xev = imp.transform(_clean(tr_s, feats)), imp.transform(_clean(ev_s, feats))
        ybig = (tr_s["ceiling"] >= BIG_TRAIN).astype(int)
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, class_weight="balanced", random_state=17,
                             verbosity=-1).fit(Xtr, ybig)
        reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.85,
                            colsample_bytree=0.85, random_state=17, verbosity=-1).fit(Xtr, tr_s["ceiling"])
        ev_s["p_big"] = clf.predict_proba(Xev)[:, 1]
        ev_s["e_ceiling"] = np.maximum(reg.predict(Xev), 0)
        top1 = ev_s.sort_values("p_big", ascending=False).groupby("date").head(1)
        picks_by_side[side] = top1

    def profile(picks: pd.DataFrame, label: str) -> dict:
        c = picks["ceiling"]
        row = {"book": label, "n": len(picks), "mean_fav_%": round(c.mean() * 100, 2),
               "med_fav_%": round(c.median() * 100, 2), "med_day_to_peak": float(picks["days_to_peak"].median())}
        for t in THRESHOLDS:
            row[f">= {int(t*100)}%"] = round(float((c >= t).mean()) * 100, 1)
        for k in (3, 5, 10):
            row[f"top{k}_mover_%"] = round(float((picks["mover_rank"] <= k).mean()) * 100, 1)
        return row

    rng = np.random.default_rng(0)
    rows = []
    for side in ("long", "short"):
        rows.append(profile(picks_by_side[side], f"{side} top1/day (model)"))
        # random baseline within universe/side
        rnd = evl[evl["side"].eq(side)].groupby("date").sample(1, random_state=0)
        rows.append(profile(rnd, f"{side} top1/day (random)"))
    print("\n===== PER-SIDE PICK PROFILE (eval 2024-2026) =====")
    print(pd.DataFrame(rows).to_string(index=False))

    # Daily 2-option book: 1 long call + 1 short put. Day succeeds if EITHER reaches threshold.
    L = picks_by_side["long"].set_index("date")["ceiling"]
    S = picks_by_side["short"].set_index("date")["ceiling"]
    book = pd.DataFrame({"long_fav": L, "short_fav": S}).dropna()
    book["day_max_fav"] = book[["long_fav", "short_fav"]].max(axis=1)
    print("\n===== DAILY BOOK: 1 call + 1 short put (day green if EITHER pick hits threshold) =====")
    drows = []
    for t in THRESHOLDS:
        q = float((book["day_max_fav"] >= t).mean())
        drows.append({"threshold": f">= {int(t*100)}%", "P(day has a winner)_%": round(q * 100, 1)})
    print(pd.DataFrame(drows).to_string(index=False))

    # Convexity EV at user's assumption: a pick that reaches WIN_AT returns WIN_MULT x (else -100%).
    p_win_trade = float(((picks_by_side["long"]["ceiling"] >= WIN_AT).sum()
                         + (picks_by_side["short"]["ceiling"] >= WIN_AT).sum())
                        / (len(picks_by_side["long"]) + len(picks_by_side["short"])))
    ev_trade = p_win_trade * (WIN_MULT - 1) - (1 - p_win_trade) * 1.0
    q_day = float((book["day_max_fav"] >= WIN_AT).mean())
    ev_day = q_day * (WIN_MULT * 1 - 2) + (1 - q_day) * (-2)   # 2 premiums staked, winner->WIN_MULT
    print(f"\n===== CONVEXITY EV  (assume >= {int(WIN_AT*100)}% move -> {WIN_MULT:g}x option, else premium lost) =====")
    print(f"per-trade  P(>= {int(WIN_AT*100)}%) = {p_win_trade*100:.1f}%  -> EV = {ev_trade*100:+.0f}% of premium  "
          f"(breakeven needs P > {100/WIN_MULT:.0f}%)")
    print(f"per-day    P(>=1 winner) = {q_day*100:.1f}%  -> EV = {ev_day:+.2f} premiums/day on 2 staked  "
          f"(breakeven needs P > {2/WIN_MULT*100:.0f}%)")

    # Stability + where winners come from
    bk = book.copy(); bk["year"] = pd.to_datetime(bk.index).year
    print("\n===== BY YEAR: P(day has a >=12% winner) =====")
    print(bk.groupby("year")["day_max_fav"].apply(lambda s: round((s >= 0.12).mean() * 100, 1)).to_string())
    wins = pd.concat([picks_by_side["long"], picks_by_side["short"]])
    wins = wins[wins["ceiling"] >= WIN_AT]
    print(f"\ntier split of >= {int(WIN_AT*100)}% winning picks:")
    print(wins["tier"].value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
