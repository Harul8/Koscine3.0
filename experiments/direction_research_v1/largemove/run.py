"""CLI for the LOCKED Large-Move production engine.

  python -m largemove.run walkforward   # regenerate OOS validation predictions (locks/.../predictions)
  python -m largemove.run train         # fit production models on all data (locks/.../models)
  python -m largemove.run predict [--date YYYY-MM-DD]  # daily ranked shortlist
"""
from __future__ import annotations

import argparse

import pandas as pd

from largemove import pipeline as P
from largemove.config import PROD, PREDICTIONS_DIR


def cmd_walkforward() -> None:
    df = P.load_dataset(PROD)
    out = P.walk_forward(PROD, df)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    allp = []
    for group, p in out.items():
        p.to_csv(PREDICTIONS_DIR / f"group_{group}_predictions.csv", index=False)
        allp.append(p)
        d1 = p[p.rank_in_day == 1]
        by = d1.assign(y=pd.to_datetime(d1.date).dt.year).groupby("y")["hit"].mean().mul(100).round(1).to_dict()
        print(f"{group}: prec@1 {d1['hit'].mean()*100:.1f}% by-year {by} | {len(p)} eligible rows")
    cooled = P.rank_cooldown(pd.concat(allp, ignore_index=True), PROD, n_per_day=2)
    cooled.to_csv(PREDICTIONS_DIR / "combined_shortlist.csv", index=False)
    print(f"saved walk-forward predictions + combined_shortlist to {PREDICTIONS_DIR}")


def cmd_train() -> None:
    P.train_production(PROD)
    print("production models trained + saved.")


def cmd_predict(date: str | None) -> None:
    df = P.load_dataset(PROD)
    if date is None:
        date = str(pd.to_datetime(df.loc[df.eligible, "date"]).max().date())
    preds = P.predict(df, PROD, on_date=date)
    shortlist = P.rank_cooldown(preds, PROD, n_per_day=2)
    print(f"=== Daily shortlist {date} (top-2/group, t+{PROD.cooldown_trading_days} cooldown) ===")
    print(shortlist[["group", "symbol", "dir", "confidence", "exp_move_%"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("walkforward"); sub.add_parser("train")
    pp = sub.add_parser("predict"); pp.add_argument("--date", default=None)
    args = ap.parse_args()
    if args.cmd == "walkforward": cmd_walkforward()
    elif args.cmd == "train": cmd_train()
    elif args.cmd == "predict": cmd_predict(args.date)


if __name__ == "__main__":
    main()
