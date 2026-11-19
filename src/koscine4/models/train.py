"""Walk-forward training of one model (requirements M-2, M-4 to M-8, U-6).

A model is described by a TOML file under `configs/models/`. Training writes:

    <pit>/predictions/<name>.parquet     one out-of-sample prediction per (date, instrument)
    <model_store>/<name>/<cutoff>/       the fitted model and its record, one per retraining
    records/results/model_<name>.md      how well the predictions ranked the outcomes, by year

    python -m koscine4.models.train --model move_vs_implied_5d

This is a training run. It starts only when the owner runs it (M-8).
"""
from __future__ import annotations

import argparse
import json
import tomllib

import numpy as np
import pandas as pd

from koscine4.features import build as feature_build
from koscine4.harness import records
from koscine4.harness.leak import rank_correlation, too_good_columns
from koscine4.harness.walkforward import monthly_folds
from koscine4.models.core import (ModelConfig, calibrate, calibration_check, calibration_table, load_calibration, save_calibration,
                                  walk_forward)
from koscine4.outcomes import build as outcomes
from koscine4.settings import Settings, load_settings
from koscine4.store.api import Store
from koscine4.universe.build import INDEX_BOOK, UniverseError, stock_list


def load_config(settings: Settings, name: str) -> ModelConfig:
    path = settings.root / "configs" / "models" / f"{name}.toml"
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    raw["features"] = tuple(raw["features"])
    raw["instruments"] = tuple(raw.get("instruments", ()))
    return ModelConfig(**raw).validate()


def load_target(settings: Settings, cfg: ModelConfig, cutoff) -> pd.DataFrame:
    """Outcome rows (date, instrument, y, window_end) with windows ended by `cutoff`."""
    t = cfg.target
    if t.get("table") == "structure":
        o = outcomes.read_structure(settings, str(t["strategy"]), cutoff)
        return o.rename(columns={str(t.get("column", "result_pct")): "y"})[["date", "instrument", "y", "window_end"]]
    entry = "next_open" if cfg.decision_time == "evening" else "1515"
    o = outcomes.read(settings, entry, cutoff, int(t["horizon"]))
    return o.rename(columns={str(t["column"]): "y"})[["date", "instrument", "y", "window_end"]]


def train(name: str, settings: Settings | None = None, start=None, end=None, unlock_sealed: bool = False, reason: str = "") -> dict:
    settings = settings or load_settings()
    cfg = load_config(settings, name)
    hcfg = settings.section("harness")
    store = Store(settings)
    cal = store.calendar
    start = pd.Timestamp(start or cfg.start or hcfg["default_start"])
    end, sealed_note = records.clip_to_seal(settings, end, cal.days[-1], unlock_sealed, reason)
    gap = int(hcfg["gap_trading_days"])
    folds = monthly_folds(cal, start, end, gap, gap)
    feats = feature_build.read(cfg.features, cfg.instruments or None, None, end, cfg.decision_time, settings)
    target = load_target(settings, cfg, end)
    horizon = int(cfg.target.get("horizon", 5))

    def tradeable(date):
        if cfg.instruments:
            return list(cfg.instruments)
        try:
            return list(INDEX_BOOK) + stock_list(settings, date)
        except UniverseError:
            return []

    leaks = too_good_columns(feats.merge(target, on=["date", "instrument"])[list(cfg.features)].tail(50000),
                             feats.merge(target, on=["date", "instrument"])["y"].tail(50000))
    if leaks:
        raise RuntimeError(f"Columns that track the outcome too closely to be features (V-13): {leaks}")
    preds, model_records = walk_forward(cfg, feats, target, folds, cal, gap, weight=outcomes.overlap_weight(horizon),
                                        model_dir=settings.path("model_store") / cfg.name, score_instruments=tradeable)
    preds, calibrators = calibrate(cfg, preds, target)                                   # M-6
    for cutoff, c in calibrators.items():
        save_calibration(settings.path("model_store") / cfg.name / cutoff, c)
    out_path = settings.path("pit") / "predictions" / f"{cfg.name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)

    scored = preds.merge(target, on=["date", "instrument"], how="inner")
    by_year = {int(y): {"rows": int(len(g)), "rank_correlation": rank_correlation(g["prediction"], g["y"])}
               for y, g in scored.groupby(scored["date"].dt.year)}
    classifier = cfg.kind == "classification"
    tolerance = float(dict(settings.section("models")).get("calibration_tolerance", 0.05))
    report = {"model": cfg.name, "rows": int(len(preds)), "scored_rows": int(len(scored)), "fits": len(model_records),
              "rank_correlation": rank_correlation(scored["prediction"], scored["y"]), "by_year": by_year,
              "calibration": calibration_table(scored["prediction"], scored["y"]) if classifier else [],
              "calibration_raw": calibration_table(scored["raw_prediction"], scored["y"]) if classifier else [],
              "calibration_method": cfg.calibration if classifier else "not a classifier",
              "corrected_fits": sum(c is not None for c in calibrators.values()),
              "start": str(start.date()), "end": str(end.date()), "sealed_note": sealed_note}
    if classifier:
        report["calibration_check"] = calibration_check(report["calibration"], tolerance)
    records.append_log(settings, {"kind": "train", "model": cfg.name, "features": list(cfg.features), "feature_count": len(cfg.features),
                                  "data_snapshot": records.data_snapshot(settings), "code_version": records.code_version(settings),
                                  "summary": {k: report[k] for k in ("rows", "fits", "rank_correlation")}})
    res = settings.path("records") / "results"
    res.mkdir(parents=True, exist_ok=True)
    lines = [f"# Model {cfg.name}", "", f"{cfg.algorithm}, {cfg.kind}, {len(cfg.features)} features, {len(model_records)} fits. "
             f"Out-of-sample {report['start']} to {report['end']}. {sealed_note}", "",
             f"Rank correlation of prediction with outcome: {report['rank_correlation']:.4f}", "",
             "| Year | Rows | Rank correlation |", "|---|---|---|"]
    lines += [f"| {y} | {v['rows']} | {v['rank_correlation']:.4f} |" for y, v in by_year.items()]
    if classifier:
        chk = report["calibration_check"]
        verdict = "not checked: too few rows" if chk["ok"] is None else ("within" if chk["ok"] else "OUTSIDE") + f" tolerance {chk['tolerance']}"
        lines += ["", f"## Calibration (M-6, {cfg.calibration}; {report['corrected_fits']} of {len(calibrators)} fits corrected)", "",
                  f"Largest gap between predicted and realised rate: {chk['largest_gap']} ({verdict}).", "",
                  "| Tenth | Rows | Raw prediction | Corrected prediction | Realised rate |", "|---|---|---|---|---|"]
        raw = {r["bin"]: r for r in report["calibration_raw"]}
        lines += [f"| {r['bin'] + 1} | {r['rows']} | {raw.get(r['bin'], {}).get('mean_prediction', float('nan')):.3f} | "
                  f"{r['mean_prediction']:.3f} | {r['realised_rate']:.3f} |" for r in report["calibration"]]
    (res / f"model_{cfg.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (res / f"model_{cfg.name}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def predict_latest(settings: Settings, name: str, feature_rows: pd.DataFrame, date) -> pd.DataFrame:
    """Score today's feature rows with the saved model stamped with the latest cutoff (daily run)."""
    from koscine4.models.core import load_latest_model

    from pathlib import Path

    model, record = load_latest_model(settings.path("model_store"), name, date)
    raw = np.asarray(model.predict(feature_rows[record["features"]]), dtype=float)
    cal = load_calibration(Path(record["_folder"]))                                       # M-6: the same correction as the backtest
    return pd.DataFrame({"date": feature_rows["date"].to_numpy(), "instrument": feature_rows["instrument"].to_numpy(),
                         "prediction": cal.apply(raw) if cal is not None else raw, "raw_prediction": raw, "cutoff": record["cutoff"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--unlock-sealed", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    report = train(args.model, start=args.start, end=args.end, unlock_sealed=args.unlock_sealed, reason=args.reason)
    print(json.dumps({k: v for k, v in report.items() if k != "calibration"}, indent=2, default=str))


if __name__ == "__main__":
    main()
