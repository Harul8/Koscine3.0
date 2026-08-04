"""
Probability calibrators — pickle-stable wrappers used by train + predict.

These classes need a stable module path so pickled models save/load cleanly
across processes.  Defining them in train.py was broken under `python -m
pipeline.train` because pickle would resolve the class to __main__.

Calibrator hierarchy (best → fallback):
  beta      — asymmetric sigmoid; beats isotonic on small cal sets (betacal pkg)
  venn_abers— conformal guarantee; best log-loss reduction (venn-abers pkg)
  sigmoid   — Platt scaling (LogReg on logits); safe, always available
  isotonic  — sklearn IsotonicRegression; can overfit on < 1k cal samples
  none      — identity (no calibration)

Both beta and venn_abers fall back to sigmoid if the required package is absent
so the rest of the pipeline never breaks.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt (sigmoid) calibration: logistic regression on logit-transformed probs.

    Same .fit(probs, y) / .predict(probs) interface as sklearn's
    IsotonicRegression, so it's a drop-in replacement.  Stays smooth + monotonic
    even when the raw probability distribution is narrow (which is where
    isotonic collapses into a near-degenerate step function).
    """
    _EPS = 1e-7

    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs")

    def _logits(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), self._EPS, 1.0 - self._EPS)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr.fit(self._logits(probs), np.asarray(y, dtype=int))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._logits(probs))[:, 1]


class BetaCalibrator:
    """Beta calibration — generalises Platt with an asymmetric sigmoid.

    Allows different slopes on the low-prob and high-prob ends, which
    better corrects tail miscalibration common in gradient-boosted models.
    Requires: pip install betacal
    Falls back to Platt if betacal is not installed.

    Parameters
    ----------
    parameters : 'abm' | 'ab' | 'a' | 'b'
        Which Beta-calibration parameters to fit.  'abm' (default) is the
        fully flexible 3-parameter variant.
    """
    _EPS = 1e-6

    def __init__(self, parameters: str = "abm"):
        self.parameters = parameters
        self._cal = None
        self._is_beta = False

    def _clip(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), self._EPS, 1.0 - self._EPS)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        try:
            from betacal import BetaCalibration
            self._cal = BetaCalibration(parameters=self.parameters)
            self._cal.fit(self._clip(probs).reshape(-1, 1), np.asarray(y, dtype=int))
            self._is_beta = True
        except (ImportError, Exception):
            # Graceful fallback to Platt if betacal unavailable or fit fails
            self._cal = PlattCalibrator()
            self._cal.fit(probs, y)
            self._is_beta = False
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self._is_beta:
            return self._cal.predict(self._clip(probs).reshape(-1, 1))
        return self._cal.predict(probs)


class TemperatureCalibrator:
    """Temperature scaling — single-parameter calibration.

    Divides the logit by a learnable temperature T:
        p_cal = sigmoid(logit(p_raw) / T)

    T > 1 softens predictions (too confident model); T < 1 sharpens them.
    Very low overfitting risk on small calibration sets (one free parameter).
    Competitive with isotonic for AUC, often better for log-loss.
    """
    _EPS = 1e-7

    def __init__(self):
        self.T = 1.0

    def _logits(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), self._EPS, 1.0 - self._EPS)
        return np.log(p / (1.0 - p))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        from scipy.optimize import minimize_scalar
        from scipy.special import expit

        logits = self._logits(probs)
        y_arr  = np.asarray(y, dtype=float)

        def nll(T):
            T = max(T, 1e-3)
            p = expit(logits / T)
            p = np.clip(p, self._EPS, 1.0 - self._EPS)
            return -np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p))

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.T = float(result.x)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        from scipy.special import expit
        return expit(self._logits(probs) / max(self.T, 1e-3))


class VennAbersCalibrator:
    """Inductive Venn-Abers predictor (midpoint calibration).

    Provides conformal coverage guarantees; empirically achieves the best
    log-loss reduction across tabular model families.
    Requires: pip install venn-abers
    Falls back to BetaCalibrator (then Platt) if package unavailable.
    """

    def __init__(self):
        self._scores = None
        self._labels = None
        self._fallback = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "VennAbersCalibrator":
        try:
            # Verify the package is available and the API works
            import venn_abers  # noqa: F401
            self._scores = np.asarray(probs, dtype=float)
            self._labels = np.asarray(y,     dtype=int)
            self._fallback = None
        except ImportError:
            self._fallback = BetaCalibrator()
            self._fallback.fit(probs, y)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(probs)
        from venn_abers import VennAbers
        va = VennAbers()
        va.fit(self._scores, self._labels)
        p_test = np.asarray(probs, dtype=float)
        result = va.predict_proba(p_test)
        # result may be shape (n, 2) [p0, p1] or a tuple — handle both
        if isinstance(result, tuple):
            p0, p1 = result
        else:
            p0, p1 = result[:, 0], result[:, 1]
        return (np.asarray(p0) + np.asarray(p1)) / 2.0


class IdentityCalibrator:
    """No-op calibrator — returns raw probabilities unchanged."""

    def fit(self, probs, y):
        return self

    def predict(self, probs):
        return np.asarray(probs, dtype=float)


def make_calibrator(model_name: str, calibration_method_map: dict):
    """Pick the calibrator type based on the CALIBRATION_METHOD config.

    Looks up by base name (strips _bull / _range / _bear / _sideways regime
    suffix) so regime variants inherit the same calibration method as their
    base model.

    Returns an object exposing .fit(probs, y) and .predict(probs).
    """
    from sklearn.isotonic import IsotonicRegression  # local import keeps the top light

    base = model_name
    for suffix in ("_bull", "_bear", "_range", "_sideways"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    method = calibration_method_map.get(base, "isotonic")

    if method == "beta":
        return BetaCalibrator()
    if method == "temperature":
        return TemperatureCalibrator()
    if method == "venn_abers":
        return VennAbersCalibrator()
    if method == "sigmoid":
        return PlattCalibrator()
    if method == "none":
        return IdentityCalibrator()
    return IsotonicRegression(out_of_bounds="clip")
