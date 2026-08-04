from __future__ import annotations

import gc
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from koscine.config import MODEL_DIR, PREDICTIONS_DIR, PROCESSED_DIR, PROJECT_ROOT
from koscine.tiered_clean_direction import TierSpec, TieredProdConfig, prepare_tiered_frame, tier_specs
from koscine.training import feature_columns
from scripts.run_clean_close_candidate_experiment import (
    CleanCloseCandidateConfig,
    add_top5_metrics,
    fit_binary_model,
    fit_catboost_model,
    fit_ranker_model,
    metric_row,
    release_memory,
    score_frame,
    stable_seed,
    train_cutoff_for_prediction_start,
    train_window,
)


CRASH_SHORT_ROOT = MODEL_DIR / "crash_short"
CRASH_SHORT_LABEL = "CSGO+"
CRASH_SHORT_PROFILE = "crash_short_v1"
CRASH_SHORT_MODEL_IDS = ("liquid30_down_4pct_5d", "rest35_down_7pct_5d")


@dataclass(frozen=True)
class CrashShortBranch:
    branch_id: str
    model_id: str
    score_variant: str
    gate: dict[str, float | str]


CRASH_SHORT_BRANCHES: tuple[CrashShortBranch, ...] = (
    CrashShortBranch(
        branch_id="LCB_SHORT_liquid_crash",
        model_id="liquid30_down_4pct_5d",
        score_variant="score_rank_product",
        gate={
            "clean_rank_pct_date_min": 0.90,
            "mkt_pct_above_sma20_max": 0.15,
            "cc_side_breadth_min": 0.70,
            "cc_side_ret_20d_min": 0.12,
            "nifty_realized_vol_20_min": 0.009,
            "pcr_oi_min": 0.55,
            "pcr_oi_max": 0.75,
        },
    ),
    CrashShortBranch(
        branch_id="RCB_SHORT_rest_crash",
        model_id="rest35_down_7pct_5d",
        score_variant="score_rank_product",
        gate={
            "clean_rank_pct_date_min": 0.94,
            "cc_side_breadth_min": 0.66,
            "cc_side_ret_20d_min": 0.10,
            "mkt_pct_above_sma20_max": 0.18,
            "nifty_realized_vol_20_min": 0.0125,
            "delivery_pct_rank_252d_min": 0.45,
            "balanced_veto_false_or_pcr_oi_rank_252d_min": 0.45,
        },
    ),
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _branch_by_model_id(model_id: str) -> CrashShortBranch:
    for branch in CRASH_SHORT_BRANCHES:
        if branch.model_id == model_id:
            return branch
    raise KeyError(f"Unknown crash-short model_id: {model_id}")


def _specs_by_model_id() -> dict[str, TierSpec]:
    return {spec.model_id: spec for spec in tier_specs()}


def _crash_short_specs() -> list[TierSpec]:
    specs = _specs_by_model_id()
    return [specs[model_id] for model_id in CRASH_SHORT_MODEL_IDS]


def make_crash_short_config(use_gpu: bool = False) -> CleanCloseCandidateConfig:
    return CleanCloseCandidateConfig(
        train_start_year=2010,
        validation_days=365,
        train_cutoff_day=20,
        good_num_boost_round=300,
        opposite_num_boost_round=300,
        large_num_boost_round=180,
        ranker_num_boost_round=260,
        early_stopping_rounds=30,
        use_gpu=use_gpu,
        use_ranker=True,
        use_catboost=True,
        catboost_iterations=140,
        catboost_weight=0.45,
        train_all_symbols=False,
    )


def _resolve(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else PROJECT_ROOT / out


def _load_frame(dataset_path: Path, specs: list[TierSpec], config: CleanCloseCandidateConfig) -> tuple[pd.DataFrame, list[str]]:
    frame, _base_features, _specs = prepare_tiered_frame(
        dataset_path,
        TieredProdConfig(train_start_year=config.train_start_year, train_cutoff_day=config.train_cutoff_day),
        specs,
    )
    return frame, feature_columns(frame)


def _save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _model_paths(model_dir: Path, model_id: str) -> dict[str, str | None]:
    paths: dict[str, str | None] = {
        "good_lgbm": f"{model_id}.good.lgbm.txt",
        "opposite_lgbm": f"{model_id}.opposite.lgbm.txt",
        "large_lgbm": f"{model_id}.large.lgbm.txt",
        "ranker_lgbm": f"{model_id}.ranker.lgbm.txt",
        "good_catboost": f"{model_id}.good.catboost.cbm",
        "opposite_catboost": f"{model_id}.opposite.catboost.cbm",
    }
    return {name: rel if (model_dir / rel).exists() else None for name, rel in paths.items()}


def train_crash_short_models(
    dataset_path: str | Path = PROCESSED_DIR / "daily_features.parquet",
    prediction_start: str = "2026-01-01",
    prediction_end: str = "2026-03-31",
    run_name: str | None = None,
    use_gpu: bool = False,
    update_current: bool = False,
) -> Path:
    dataset = _resolve(dataset_path)
    pred_start = pd.Timestamp(prediction_start).normalize()
    pred_end = pd.Timestamp(prediction_end).normalize()
    config = make_crash_short_config(use_gpu=use_gpu)
    train_end = train_cutoff_for_prediction_start(pred_start, config.train_cutoff_day)
    run_name = run_name or f"crash_short_{pred_start:%Y%m%d}_{pred_end:%Y%m%d}_{_now_tag()}"
    run_dir = CRASH_SHORT_ROOT / run_name
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    specs = _crash_short_specs()
    frame, base_features = _load_frame(dataset, specs, config)
    model_entries = []
    training_rows = []
    scored_parts = []

    for spec in specs:
        branch = _branch_by_model_id(spec.model_id)
        train, valid, features = train_window(frame, base_features, spec, train_end, config)
        train_pos = {
            "good": int(train["good_target"].sum()),
            "opposite": int(train["opposite_target"].sum()),
            "large": int(train["large_good_target"].sum()),
        }
        valid_pos = {
            "good": int(valid["good_target"].sum()),
            "opposite": int(valid["opposite_target"].sum()),
            "large": int(valid["large_good_target"].sum()),
        }
        print(
            f"\n=== crash-short {spec.model_id}: train through {train_end.date()}, "
            f"predict {pred_start.date()} to {pred_end.date()} ===",
            flush=True,
        )
        good_model = fit_binary_model(
            train,
            valid,
            features,
            "good_target",
            "sample_weight_good",
            spec,
            config,
            stable_seed(f"cc_good_{spec.model_id}", 61000),
            "good",
        )
        opposite_model = fit_binary_model(
            train,
            valid,
            features,
            "opposite_target",
            "sample_weight_opposite",
            spec,
            config,
            stable_seed(f"cc_opposite_{spec.model_id}", 62000),
            "opposite",
        )
        if good_model is None or opposite_model is None:
            raise RuntimeError(f"Missing required crash-short head for {spec.model_id}")
        good_cat_model = fit_catboost_model(
            train,
            valid,
            features,
            "good_target",
            "sample_weight_good",
            stable_seed(f"cc_good_cat_{spec.model_id}", 65000),
            config,
            "good",
        )
        opposite_cat_model = fit_catboost_model(
            train,
            valid,
            features,
            "opposite_target",
            "sample_weight_opposite",
            stable_seed(f"cc_opp_cat_{spec.model_id}", 66000),
            config,
            "opposite",
        )
        large_model = None
        if train_pos["large"] >= config.large_min_train_positive and valid_pos["large"] >= 5:
            large_model = fit_binary_model(
                train,
                valid,
                features,
                "large_good_target",
                "sample_weight_large",
                spec,
                config,
                stable_seed(f"cc_large_{spec.model_id}", 63000),
                "large",
            )
        ranker_model = fit_ranker_model(train, valid, features, spec, config)

        good_model.save_model(str(model_dir / f"{spec.model_id}.good.lgbm.txt"))
        opposite_model.save_model(str(model_dir / f"{spec.model_id}.opposite.lgbm.txt"))
        if large_model is not None:
            large_model.save_model(str(model_dir / f"{spec.model_id}.large.lgbm.txt"))
        if ranker_model is not None:
            ranker_model.save_model(str(model_dir / f"{spec.model_id}.ranker.lgbm.txt"))
        if good_cat_model is not None:
            good_cat_model.save_model(str(model_dir / f"{spec.model_id}.good.catboost.cbm"))
        if opposite_cat_model is not None:
            opposite_cat_model.save_model(str(model_dir / f"{spec.model_id}.opposite.catboost.cbm"))

        prediction_frame = frame[
            frame["date"].between(pred_start, pred_end) & frame["symbol"].isin(spec.predict_symbols)
        ].copy()
        scored = score_frame(
            prediction_frame,
            features,
            spec,
            config,
            good_model,
            opposite_model,
            large_model,
            ranker_model,
            good_cat_model,
            opposite_cat_model,
        )
        scored["crash_short_signal"] = crash_short_signal_mask(scored, branch)
        scored["crash_short_label"] = np.where(scored["crash_short_signal"], CRASH_SHORT_LABEL, "")
        scored["crash_short_branch"] = branch.branch_id
        scored["crash_short_profile"] = CRASH_SHORT_PROFILE
        scored["crash_short_reason"] = np.where(scored["crash_short_signal"], _branch_reason(branch), "")
        scored_parts.append(scored)

        model_entry = {
            "branch_id": branch.branch_id,
            "profile": CRASH_SHORT_PROFILE,
            "model_id": spec.model_id,
            "tier": spec.name,
            "side": spec.side,
            "threshold": spec.threshold,
            "score_variant": branch.score_variant,
            "gate": branch.gate,
            "features": features,
            "train_end": str(train_end.date()),
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_pos": train_pos,
            "valid_pos": valid_pos,
            "models": _model_paths(model_dir, spec.model_id),
            "predict_symbols": list(spec.predict_symbols),
        }
        model_entries.append(model_entry)
        training_rows.append(
            {
                "model_id": spec.model_id,
                "train_end": str(train_end.date()),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "feature_count": len(features),
                **{f"train_{k}_pos": v for k, v in train_pos.items()},
                **{f"valid_{k}_pos": v for k, v in valid_pos.items()},
                "good_best_iteration": good_model.best_iteration,
                "opposite_best_iteration": opposite_model.best_iteration,
                "large_best_iteration": large_model.best_iteration if large_model is not None else np.nan,
                "ranker_best_iteration": ranker_model.best_iteration if ranker_model is not None else np.nan,
                "good_catboost_trees": good_cat_model.tree_count_ if good_cat_model is not None else np.nan,
                "opposite_catboost_trees": opposite_cat_model.tree_count_ if opposite_cat_model is not None else np.nan,
            }
        )
        print(
            f"  {spec.model_id}: train={len(train):,} valid={len(valid):,} "
            f"test={len(scored):,} CSGO+={int(scored['crash_short_signal'].sum()):,}",
            flush=True,
        )
        del train, valid, prediction_frame, scored
        del good_model, opposite_model, large_model, ranker_model, good_cat_model, opposite_cat_model
        release_memory(spec.model_id)

    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    signals = scored_all[scored_all.get("crash_short_signal", pd.Series(False, index=scored_all.index)).fillna(False)].copy()
    scored_all.to_parquet(run_dir / "scored_predictions.parquet", index=False)
    scored_all.to_csv(run_dir / "scored_predictions.csv", index=False)
    signals.to_parquet(run_dir / "signals.parquet", index=False)
    signals.to_csv(run_dir / "signals.csv", index=False)
    summary = summarize_crash_short(scored_all)
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "profile": CRASH_SHORT_PROFILE,
        "label": CRASH_SHORT_LABEL,
        "dataset": str(dataset),
        "prediction_start": str(pred_start.date()),
        "prediction_end": str(pred_end.date()),
        "train_end": str(train_end.date()),
        "config": asdict(config),
        "models": model_entries,
        "prod_current_touched": bool(update_current),
        "locked_core_prod_touched": False,
        "predictions_prod_touched": False,
        "objective": "append sparse crash-short CSGO+ branch after locked production core",
    }
    _save_manifest(run_dir, manifest)

    if update_current:
        current = CRASH_SHORT_ROOT / "current"
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(run_dir, current)
        print(f"updated crash-short current: {current}", flush=True)

    print(f"\nwrote crash-short production overlay models: {run_dir}", flush=True)
    if not summary.empty:
        print(summary.round(2).to_string(index=False), flush=True)
    return run_dir


def _load_catboost(path: Path | None) -> CatBoostClassifier | None:
    if path is None or not path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _load_model_bundle(model_root: Path, entry: dict) -> dict:
    model_dir = model_root / "models"
    models = entry["models"]
    good = models.get("good_lgbm")
    opposite = models.get("opposite_lgbm")
    if not good or not opposite:
        raise ValueError(f"Missing required LightGBM models for {entry['model_id']}")
    large = models.get("large_lgbm")
    ranker = models.get("ranker_lgbm")
    good_cat = models.get("good_catboost")
    opposite_cat = models.get("opposite_catboost")
    return {
        "good_model": lgb.Booster(model_file=str(model_dir / good)),
        "opposite_model": lgb.Booster(model_file=str(model_dir / opposite)),
        "large_model": lgb.Booster(model_file=str(model_dir / large)) if large else None,
        "ranker_model": lgb.Booster(model_file=str(model_dir / ranker)) if ranker else None,
        "good_cat_model": _load_catboost(model_dir / good_cat if good_cat else None),
        "opposite_cat_model": _load_catboost(model_dir / opposite_cat if opposite_cat else None),
    }


def _load_manifest(model_root: str | Path) -> tuple[Path, dict]:
    root = _resolve(model_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Crash-short manifest not found: {manifest_path}")
    return root, json.loads(manifest_path.read_text(encoding="utf-8"))


def crash_short_signal_mask(scored: pd.DataFrame, branch: CrashShortBranch | dict) -> pd.Series:
    if isinstance(branch, dict):
        branch = CrashShortBranch(
            branch_id=branch["branch_id"],
            model_id=branch["model_id"],
            score_variant=branch["score_variant"],
            gate=branch["gate"],
        )
    gate = branch.gate
    base = _num(scored, "clean_rank_pct_date").ge(float(gate["clean_rank_pct_date_min"]))
    base &= _num(scored, "mkt_pct_above_sma20").le(float(gate["mkt_pct_above_sma20_max"]))
    base &= _num(scored, "cc_side_breadth").ge(float(gate["cc_side_breadth_min"]))
    base &= _num(scored, "cc_side_ret_20d").ge(float(gate["cc_side_ret_20d_min"]))
    base &= _num(scored, "nifty_realized_vol_20").ge(float(gate["nifty_realized_vol_20_min"]))
    if branch.model_id == "liquid30_down_4pct_5d":
        base &= _num(scored, "pcr_oi").between(float(gate["pcr_oi_min"]), float(gate["pcr_oi_max"]))
    elif branch.model_id == "rest35_down_7pct_5d":
        veto_false = _num(scored, "cc_prod_balanced_veto_pass", 0.0).fillna(0.0).le(0.0)
        pcr_rank_ok = _num(scored, "pcr_oi_rank_252d").ge(
            float(gate["balanced_veto_false_or_pcr_oi_rank_252d_min"])
        )
        base &= _num(scored, "delivery_pct_rank_252d").ge(float(gate["delivery_pct_rank_252d_min"]))
        base &= veto_false | pcr_rank_ok
    return base.fillna(False).astype(bool)


def _branch_reason(branch: CrashShortBranch | dict) -> str:
    if isinstance(branch, dict):
        branch_id = branch["branch_id"]
    else:
        branch_id = branch.branch_id
    return f"{CRASH_SHORT_PROFILE}:{branch_id}"


def score_crash_short_range(
    dataset_path: str | Path = PROCESSED_DIR / "daily_features.parquet",
    model_root: str | Path = CRASH_SHORT_ROOT / "current",
    start: str = "2026-01-01",
    end: str = "2026-03-31",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    root, manifest = _load_manifest(model_root)
    config = CleanCloseCandidateConfig(**manifest["config"])
    specs_by_id = _specs_by_model_id()
    model_entries = [entry for entry in manifest["models"] if entry["model_id"] in CRASH_SHORT_MODEL_IDS]
    specs = [specs_by_id[entry["model_id"]] for entry in model_entries]
    frame, _base_features = _load_frame(_resolve(dataset_path), specs, config)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    parts = []
    for entry in model_entries:
        spec = specs_by_id[entry["model_id"]]
        branch = {
            "branch_id": entry["branch_id"],
            "model_id": entry["model_id"],
            "score_variant": entry["score_variant"],
            "gate": entry["gate"],
        }
        prediction_frame = frame[
            frame["date"].between(start_ts, end_ts) & frame["symbol"].isin(spec.predict_symbols)
        ].copy()
        bundle = _load_model_bundle(root, entry)
        scored = score_frame(
            prediction_frame,
            entry["features"],
            spec,
            config,
            bundle["good_model"],
            bundle["opposite_model"],
            bundle["large_model"],
            bundle["ranker_model"],
            bundle["good_cat_model"],
            bundle["opposite_cat_model"],
        )
        mask = crash_short_signal_mask(scored, branch)
        scored["crash_short_signal"] = mask
        scored["crash_short_label"] = np.where(mask, CRASH_SHORT_LABEL, "")
        scored["crash_short_branch"] = entry["branch_id"]
        scored["crash_short_profile"] = CRASH_SHORT_PROFILE
        scored["crash_short_reason"] = np.where(mask, _branch_reason(branch), "")
        scored["crash_short_model_root"] = str(root)
        scored["crash_short_train_end"] = manifest.get("train_end")
        parts.append(scored)
        del prediction_frame, scored, bundle
        release_memory(entry["model_id"])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if output_dir is not None:
        output = _resolve(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output / "crash_short_scored.parquet", index=False)
        out.to_csv(output / "crash_short_scored.csv", index=False)
        signals = out[out.get("crash_short_signal", pd.Series(False, index=out.index)).fillna(False)].copy()
        signals.to_parquet(output / "crash_short_signals.parquet", index=False)
        signals.to_csv(output / "crash_short_signals.csv", index=False)
        summarize_crash_short(out).to_csv(output / "crash_short_summary.csv", index=False)
    return out


def combine_with_crash_short_label(label: object) -> str:
    text = str(label or "").strip()
    if not text or text == "WATCH" or text == "nan":
        return CRASH_SHORT_LABEL
    parts = [part.strip() for part in text.split("/") if part.strip() and part.strip() not in {"WATCH", "nan"}]
    if CRASH_SHORT_LABEL not in parts:
        parts.append(CRASH_SHORT_LABEL)
    return "/".join(parts) if parts else CRASH_SHORT_LABEL


def apply_crash_short_overlay_to_frame(frame: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or scored.empty or "model_id" not in frame.columns:
        out = frame.copy()
        if "crash_short_signal" not in out.columns:
            out["crash_short_signal"] = False
        return out
    out = frame.copy()
    overlay_cols = [
        "crash_short_signal",
        "crash_short_label",
        "crash_short_branch",
        "crash_short_profile",
        "crash_short_reason",
        "crash_short_model_root",
        "crash_short_train_end",
        "crash_short_bucket",
        "clean_good_score",
        "clean_opposite_score",
        "clean_large_score",
        "clean_rank_pct_date",
        "score_rank_product",
    ]
    out = out.drop(columns=[col for col in overlay_cols if col in out.columns], errors="ignore")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    signals = scored[scored.get("crash_short_signal", pd.Series(False, index=scored.index)).fillna(False)].copy()
    if signals.empty:
        out["crash_short_signal"] = False
        out["crash_short_label"] = ""
        return out
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    signal_cols = [
        "date",
        "symbol",
        "model_id",
        "crash_short_signal",
        "crash_short_label",
        "crash_short_branch",
        "crash_short_profile",
        "crash_short_reason",
        "crash_short_model_root",
        "crash_short_train_end",
        "clean_good_score",
        "clean_opposite_score",
        "clean_large_score",
        "clean_rank_pct_date",
        "score_rank_product",
    ]
    signal_cols = [col for col in signal_cols if col in signals.columns]
    signals = signals[signal_cols].drop_duplicates(["date", "symbol", "model_id"], keep="first")
    merged = out.reset_index(drop=False).merge(signals, on=["date", "symbol", "model_id"], how="left")
    if "crash_short_signal" not in merged.columns:
        merged["crash_short_signal"] = False
    mask = merged["crash_short_signal"].eq(True)

    defaults = {
        "go_label": "WATCH",
        "go_label_raw": None,
        "production_bucket": "WATCH",
        "final_signal_bucket": "WATCH",
        "production_lock_note": "",
    }
    for col, default in defaults.items():
        if col not in merged.columns:
            merged[col] = merged["go_label"] if col == "go_label_raw" and "go_label" in merged.columns else default

    merged["crash_short_signal"] = mask
    merged["crash_short_label"] = np.where(mask, CRASH_SHORT_LABEL, "")
    for idx in merged.index[mask]:
        current_label = merged.at[idx, "go_label"]
        combined = combine_with_crash_short_label(current_label)
        bucket = "PROD_SHORT_CSGO_PLUS"
        merged.at[idx, "go_label"] = combined
        merged.at[idx, "production_signal"] = True
        merged.at[idx, "passes_min_score"] = True
        merged.at[idx, "production_locked"] = True
        merged.at[idx, "crash_short_bucket"] = bucket
        if str(merged.at[idx, "production_bucket"]) == "WATCH":
            merged.at[idx, "production_bucket"] = bucket
        if str(merged.at[idx, "final_signal_bucket"]) == "WATCH":
            merged.at[idx, "final_signal_bucket"] = bucket
        merged.at[idx, "trade_priority"] = "high"
        merged.at[idx, "trade_bucket"] = "crash_short"
        merged.at[idx, "final_side"] = "down"
        note = str(merged.at[idx, "production_lock_note"])
        suffix = str(merged.at[idx, "crash_short_reason"])
        merged.at[idx, "production_lock_note"] = suffix if not note or note == "nan" else f"{note};{suffix}"
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    return merged


def _prod_files(prod_dir: Path, start: str | None, end: str | None) -> list[Path]:
    files = sorted(prod_dir.glob("prod_predictions_*.parquet"))
    if start:
        start_ts = pd.Timestamp(start).normalize()
        files = [path for path in files if pd.Timestamp(path.stem.replace("prod_predictions_", "")).normalize() >= start_ts]
    if end:
        end_ts = pd.Timestamp(end).normalize()
        files = [path for path in files if pd.Timestamp(path.stem.replace("prod_predictions_", "")).normalize() <= end_ts]
    return files


def apply_crash_short_overlay_to_prod_dir(
    prod_dir: str | Path = PREDICTIONS_DIR / "prod",
    dataset_path: str | Path = PROCESSED_DIR / "daily_features.parquet",
    model_root: str | Path = CRASH_SHORT_ROOT / "current",
    start: str | None = None,
    end: str | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    prod = _resolve(prod_dir)
    files = _prod_files(prod, start, end)
    if not files:
        return {"profile": CRASH_SHORT_PROFILE, "files": 0, "csgoplus_rows": 0, "model_root": str(_resolve(model_root))}
    start_value = start or min(pd.Timestamp(path.stem.replace("prod_predictions_", "")).strftime("%Y-%m-%d") for path in files)
    end_value = end or max(pd.Timestamp(path.stem.replace("prod_predictions_", "")).strftime("%Y-%m-%d") for path in files)
    scored = score_crash_short_range(
        dataset_path=dataset_path,
        model_root=model_root,
        start=start_value,
        end=end_value,
        output_dir=output_dir,
    )
    written = 0
    csgoplus_rows = 0
    for parquet_path in files:
        before = pd.read_parquet(parquet_path)
        date = pd.Timestamp(parquet_path.stem.replace("prod_predictions_", "")).normalize()
        daily = scored[pd.to_datetime(scored["date"]).dt.normalize().eq(date)].copy()
        after = apply_crash_short_overlay_to_frame(before, daily)
        csgoplus_rows += int(after.get("crash_short_signal", pd.Series(False, index=after.index)).fillna(False).sum())
        after.to_parquet(parquet_path, index=False)
        after.to_csv(parquet_path.with_suffix(".csv"), index=False)
        written += 1
    summary = summarize_crash_short(scored)
    return {
        "profile": CRASH_SHORT_PROFILE,
        "label": CRASH_SHORT_LABEL,
        "files": written,
        "csgoplus_rows": csgoplus_rows,
        "model_root": str(_resolve(model_root)),
        "date_range": [start_value, end_value],
        "summary": summary.to_dict("records") if not summary.empty else [],
    }


def load_crash_short_schedule(schedule: str | Path | Iterable[dict]) -> list[dict]:
    if isinstance(schedule, (str, Path)):
        path = _resolve(schedule)
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = list(schedule)
    if not isinstance(raw, list):
        raise ValueError("Crash-short schedule must be a list of date-range entries")
    entries = []
    for entry in raw:
        if not {"start", "end", "model_root"} <= set(entry):
            raise ValueError(f"Invalid crash-short schedule entry: {entry}")
        entries.append(
            {
                "start": str(entry["start"]),
                "end": str(entry["end"]),
                "model_root": str(entry["model_root"]),
                "label": entry.get("label", CRASH_SHORT_LABEL),
            }
        )
    return entries


def apply_crash_short_overlay_schedule_to_prod_dir(
    schedule: str | Path | Iterable[dict],
    prod_dir: str | Path = PREDICTIONS_DIR / "prod",
    dataset_path: str | Path = PROCESSED_DIR / "daily_features.parquet",
    output_dir: str | Path | None = None,
) -> dict:
    entries = load_crash_short_schedule(schedule)
    results = []
    total_files = 0
    total_rows = 0
    for entry in entries:
        result = apply_crash_short_overlay_to_prod_dir(
            prod_dir=prod_dir,
            dataset_path=dataset_path,
            model_root=entry["model_root"],
            start=entry["start"],
            end=entry["end"],
            output_dir=output_dir,
        )
        results.append(result)
        total_files += int(result.get("files", 0))
        total_rows += int(result.get("csgoplus_rows", 0))
        gc.collect()
    return {
        "profile": CRASH_SHORT_PROFILE,
        "label": CRASH_SHORT_LABEL,
        "files": total_files,
        "csgoplus_rows": total_rows,
        "schedule": entries,
        "results": results,
    }


def summarize_crash_short(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scored.empty:
        return pd.DataFrame()
    signals = scored[scored.get("crash_short_signal", pd.Series(False, index=scored.index)).fillna(False)].copy()
    signals_eval = signals[signals.get("evaluated", pd.Series(False, index=signals.index)).fillna(False)].copy()
    rows.append(add_top5_metrics(signals_eval, scored, metric_row(signals_eval, "CSGO+")))
    for branch, group in signals_eval.groupby("crash_short_branch", sort=True):
        rows.append(add_top5_metrics(group, scored, metric_row(group, "CSGO+", {"branch": branch})))
    for model_id, group in signals_eval.groupby("model_id", sort=True):
        rows.append(add_top5_metrics(group, scored, metric_row(group, "CSGO+", {"model_id": model_id})))
    return pd.DataFrame(rows)
