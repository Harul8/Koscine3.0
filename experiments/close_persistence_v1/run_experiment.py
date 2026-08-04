"""Entry point for this experiment. Edit largemove/ freely — PROD is untouched."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # -> the cloned `largemove` package
sys.path.insert(0, str(HERE.parents[1] / "src"))   # -> shared koscine3.data / koscine3.outcomes

import pandas as pd
from largemove import pipeline as P
from largemove.config import PROD, PREDICTIONS_DIR

def main():
    df = P.load_dataset(PROD)
    out = P.walk_forward(PROD, df)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    allp = []
    for group, p in out.items():
        p.to_csv(PREDICTIONS_DIR / f"group_{group}_predictions.csv", index=False)
        allp.append(p)
        d1 = p[p.rank_in_day == 1]
        by = d1.assign(y=pd.to_datetime(d1.date).dt.year).groupby("y")["hit"].mean().mul(100).round(1).to_dict()
        print(f"{group}: prec@1 {d1['hit'].mean()*100:.1f}% by-year {by} | {len(p)} rows")
    P.rank_cooldown(pd.concat(allp, ignore_index=True), PROD, n_per_day=2)\
        .to_csv(PREDICTIONS_DIR / "combined_shortlist.csv", index=False)

if __name__ == "__main__":
    main()
