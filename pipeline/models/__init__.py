"""
Model zoo — LightGBM, Denoising AE+MLP, TabM ensemble.

All models share a common BaseModel interface:
    fit(X_train, y_train, X_val, y_val) → dict of metrics
    predict_proba(X) → np.ndarray (1D, positive class probabilities)
    save(path) / load(path)

Usage:
    from pipeline.models import get_model_class
    cls = get_model_class("lgbm")          # or "ae_mlp", "tabm"
    model = cls(target="up_5")
    metrics = model.fit(X_tr, y_tr, X_val, y_val)
"""
from .base import BaseModel
from .lgbm_model import LGBMModel
from .ae_mlp import AEMLPModel
from .tabm import TabMModel
from .catboost_model import CatBoostModel

_REGISTRY = {
    "lgbm":     LGBMModel,
    "ae_mlp":   AEMLPModel,
    "tabm":     TabMModel,
    "catboost": CatBoostModel,
}


def get_model_class(model_type: str) -> type[BaseModel]:
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Available: {list(_REGISTRY)}")
    return _REGISTRY[model_type]


__all__ = [
    "BaseModel", "LGBMModel", "AEMLPModel", "TabMModel", "CatBoostModel",
    "get_model_class",
]
