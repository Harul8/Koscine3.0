from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = {"date", "symbol"}
PRICE_COLUMNS = {"open", "high", "low", "close", "last", "prev_close"}
BLOCKED_PREFIXES = (
    "future_",
    "entry_",
    "up_move_",
    "down_move_",
    "fwd_return_",
    "long_adverse_",
    "short_adverse_",
    "label_",
)
BLOCKED_EXACT = {"vol_adj_adverse_limit"}


@dataclass(frozen=True)
class FeatureRegistry:
    identity_columns: list[str]
    price_columns: list[str]
    feature_columns: list[str]
    blocked_columns: list[str]
    non_numeric_columns: list[str]

    def assert_safe(self, requested_features: list[str] | None = None) -> None:
        features = requested_features or self.feature_columns
        blocked = sorted(c for c in features if is_blocked_column(c))
        if blocked:
            raise ValueError(f"Blocked future/outcome columns requested as features: {blocked}")
        missing = sorted(set(features) - set(self.feature_columns))
        if missing:
            raise ValueError(f"Requested features are not in the safe feature registry: {missing}")


def is_blocked_column(column: str) -> bool:
    return column in BLOCKED_EXACT or column.startswith(BLOCKED_PREFIXES)


def build_feature_registry(df: pd.DataFrame) -> FeatureRegistry:
    identity = [c for c in df.columns if c in IDENTITY_COLUMNS]
    price = [c for c in df.columns if c in PRICE_COLUMNS]
    blocked = [c for c in df.columns if is_blocked_column(c)]
    non_numeric = [
        c
        for c in df.columns
        if c not in IDENTITY_COLUMNS and c not in blocked and not pd.api.types.is_numeric_dtype(df[c])
    ]
    features = [
        c
        for c in df.columns
        if c not in IDENTITY_COLUMNS
        and c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return FeatureRegistry(
        identity_columns=identity,
        price_columns=price,
        feature_columns=features,
        blocked_columns=blocked,
        non_numeric_columns=non_numeric,
    )


def write_feature_audit(registry: FeatureRegistry, output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for name, columns in [
        ("identity", registry.identity_columns),
        ("price", registry.price_columns),
        ("feature", registry.feature_columns),
        ("blocked", registry.blocked_columns),
        ("non_numeric", registry.non_numeric_columns),
    ]:
        rows.extend({"category": name, "column": c} for c in columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

