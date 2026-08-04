from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibratorBundle:
    isotonic: IsotonicRegression
    train_min: float
    train_max: float
    n_train: int

    def predict(self, scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(scores, self.train_min, self.train_max)
        return self.isotonic.predict(clipped)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> CalibratorBundle:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) < 50 or labels.sum() < 5:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        return CalibratorBundle(iso, 0.0, 1.0, int(len(scores)))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return CalibratorBundle(iso, float(scores.min()), float(scores.max()), int(len(scores)))


def save_calibrator(bundle: CalibratorBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x_thresholds": bundle.isotonic.X_thresholds_.tolist(),
        "y_thresholds": bundle.isotonic.y_thresholds_.tolist(),
        "train_min": bundle.train_min,
        "train_max": bundle.train_max,
        "n_train": bundle.n_train,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibrator(path: Path) -> CalibratorBundle:
    payload = json.loads(path.read_text(encoding="utf-8"))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.X_thresholds_ = np.asarray(payload["x_thresholds"], dtype=float)
    iso.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=float)
    iso.X_min_ = float(iso.X_thresholds_.min())
    iso.X_max_ = float(iso.X_thresholds_.max())
    iso.f_ = None
    iso.increasing_ = True
    iso._build_f(iso.X_thresholds_, iso.y_thresholds_)
    return CalibratorBundle(iso, payload["train_min"], payload["train_max"], int(payload["n_train"]))


def calibration_report(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0:
        return pd.DataFrame()
    quantiles = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    bins = pd.cut(scores, bins=quantiles, include_lowest=True, duplicates="drop")
    df = pd.DataFrame({"score": scores, "label": labels, "bin": bins})
    grouped = df.groupby("bin", observed=True).agg(
        n=("label", "size"),
        positives=("label", "sum"),
        mean_score=("score", "mean"),
        actual_rate=("label", "mean"),
    ).reset_index(drop=True)
    grouped["gap"] = grouped["mean_score"] - grouped["actual_rate"]
    return grouped
