"""Direction research (requirements DR-1 to DR-5, decision A8).

For one group of instruments and one horizon, compares four ways of calling up or down:

    always_up      no model
    linear         regularised logistic regression on all history
    trees          gradient-boosted trees on all history
    trees_recent   the same trees, trained each month on the most recent 24 months only

The outcome is the move from the next session's open, so the overnight gap before entry is
never part of it (DR-2). Inputs are registered features only (DR-4). Results are shown by
period so a fading edge is visible (DR-5). The predictions of each model are saved, so a
directional strategy can be tested on them through the harness (DR-6).

    python -m koscine4.models.direction --group NIFTY --horizon 1
    python -m koscine4.models.direction --group stocks --horizon 5

This is a research training run. It starts only when the owner runs it (M-8).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from koscine4.features import build as feature_build
from koscine4.features.registry import REGISTRY
from koscine4.harness import records
from koscine4.harness.walkforward import monthly_folds
from koscine4.models.core import ModelConfig, walk_forward
from koscine4.outcomes import build as outcomes
from koscine4.settings import Settings, load_settings
from koscine4.store.api import Store
from koscine4.universe.build import UniverseError, stock_list

FAMILIES = ("trend", "volatility_state", "positioning", "flows", "events", "implied_vs_realised", "regime")
MODELS = {
    "linear": dict(algorithm="logistic", train_window_months=0, params={"C": 0.05}),
    "trees": dict(algorithm="gbt", train_window_months=0, params={"n_estimators": 150, "num_leaves": 7, "min_child_samples": 40}),
    "trees_recent": dict(algorithm="gbt", train_window_months=24, params={"n_estimators": 100, "num_leaves": 7, "min_child_samples": 20}),
}


def candidate_features(decision_time: str = "evening", limit: int = 40) -> list[str]:
    names = [n for fam in FAMILIES for n in REGISTRY.names(family=fam, decision_time=decision_time)]
    return names[:limit]


def score(pred_up: pd.Series, ret: pd.Series) -> dict:
    """Accuracy, and the average return of following the call, before costs."""
    call = np.where(pred_up >= 0.5, 1.0, -1.0)
    ok = ret.notna()
    hit = (np.sign(ret[ok]) == call[ok.to_numpy()])
    return {"rows": int(ok.sum()), "accuracy_pct": float(hit.mean() * 100.0) if ok.any() else float("nan"),
            "mean_return_pct": float((call[ok.to_numpy()] * ret[ok]).mean() * 100.0) if ok.any() else float("nan")}


def by_block(df: pd.DataFrame, periods: int = 3) -> dict:
    years = np.sort(df["date"].dt.year.unique())
    out = {"all": score(df["prediction"], df["ret"])}
    per_year = {int(y): score(g["prediction"], g["ret"]) for y, g in df.groupby(df["date"].dt.year)}
    out["years_better_than_chance"] = f"{sum(v['accuracy_pct'] > 50 for v in per_year.values())} of {len(per_year)}"
    out["by_year"] = per_year
    out["by_period"] = {f"{int(e[0])}-{int(e[-1])}": score(df[df["date"].dt.year.isin(e)]["prediction"], df[df["date"].dt.year.isin(e)]["ret"])
                        for e in np.array_split(years, min(periods, len(years))) if len(e)}
    return out


def research(group: str, horizon: int, settings: Settings | None = None, start=None, end=None) -> dict:
    settings = settings or load_settings()
    hcfg = settings.section("harness")
    store = Store(settings)
    cal = store.calendar
    start = pd.Timestamp(start or hcfg["default_start"])
    end, sealed_note = records.clip_to_seal(settings, end, cal.days[-1], False)
    gap = int(hcfg["gap_trading_days"])
    folds = monthly_folds(cal, start, end, gap, gap)
    names = candidate_features()
    instruments = None if group == "stocks" else (group,)
    feats = feature_build.read(names, instruments, None, end, "evening", settings)
    o = outcomes.read(settings, "next_open", end, horizon, instruments)
    target = o.rename(columns={"fwd_up": "y"})[["date", "instrument", "y", "window_end"]]
    rets = o[["date", "instrument", "fwd_ret"]].rename(columns={"fwd_ret": "ret"})

    def tradeable(date):
        if group != "stocks":
            return [group]
        try:
            return stock_list(settings, date)
        except UniverseError:
            return []

    report = {"group": group, "horizon": horizon, "features": names, "start": str(start.date()), "end": str(end.date()),
              "sealed_note": sealed_note, "models": {}}
    base = None
    for label, spec in MODELS.items():
        name = f"direction_{group}_{horizon}d_{label}"
        cfg = ModelConfig(name=name, kind="classification", algorithm=spec["algorithm"], features=tuple(names),
                          target={"table": "moves", "column": "fwd_up", "horizon": horizon}, instruments=instruments or (),
                          train_window_months=spec["train_window_months"], params=spec["params"], min_train_rows=250).validate()
        preds, _ = walk_forward(cfg, feats, target, folds, cal, gap, weight=outcomes.overlap_weight(horizon), score_instruments=tradeable)
        path = settings.path("pit") / "predictions" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(path, index=False)
        scored = preds.merge(rets, on=["date", "instrument"], how="inner")
        report["models"][label] = by_block(scored)
        base = scored if base is None else base
    if base is not None and len(base):
        report["models"] = {"always_up": by_block(base.assign(prediction=1.0)), **report["models"]}
    records.append_log(settings, {"kind": "direction_research", "group": group, "horizon": horizon,
                                  "data_snapshot": records.data_snapshot(settings), "code_version": records.code_version(settings),
                                  "summary": {k: v["all"] for k, v in report["models"].items()}})
    res = settings.path("records") / "research"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"direction_{group}_{horizon}d.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (res / f"direction_{group}_{horizon}d.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"# Direction research: {report['group']}, {report['horizon']} session(s)", "",
             f"Outcome: move from the next session's open (the overnight gap before entry is excluded). "
             f"Out-of-sample {report['start']} to {report['end']}. {report['sealed_note']}", "",
             "| Model | Rows | Accuracy % | Years better than chance | Mean return per call %, before costs |", "|---|---|---|---|---|"]
    for name, m in report["models"].items():
        a = m["all"]
        lines.append(f"| {name} | {a['rows']} | {a['accuracy_pct']:.1f} | {m['years_better_than_chance']} | {a['mean_return_pct']:.3f} |")
    lines += ["", "## By period", "", "| Model | Period | Accuracy % | Mean return per call % |", "|---|---|---|---|"]
    for name, m in report["models"].items():
        for period, v in m["by_period"].items():
            lines.append(f"| {name} | {period} | {v['accuracy_pct']:.1f} | {v['mean_return_pct']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="NIFTY, BANKNIFTY or stocks")
    parser.add_argument("--horizon", type=int, required=True, choices=[1, 3, 5])
    parser.add_argument("--start")
    args = parser.parse_args()
    print(to_markdown(research(args.group, args.horizon, start=args.start)))


if __name__ == "__main__":
    main()
