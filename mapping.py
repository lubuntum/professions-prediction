from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from models import PsychTest


@dataclass(frozen=True)
class ActiveFeatures:
    names: tuple[str, ...]
    test_by_feature: dict[str, str]
    default_value: float


def load_active_features(mapping_path: Path) -> ActiveFeatures:
    """Read feature names from tests whose mapping has is_active=true."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    feature_names: list[str] = []
    test_by_feature: dict[str, str] = {}

    for test_name, test_settings in mapping["test_mapping"].items():
        if test_settings.get("is_active") is not True:
            continue
        for feature_name in test_settings["param_names"]:
            if feature_name in test_by_feature:
                raise ValueError(f"Feature is listed more than once: {feature_name}")
            feature_names.append(feature_name)
            test_by_feature[feature_name] = test_name

    return ActiveFeatures(
        names=tuple(feature_names),
        test_by_feature=test_by_feature,
        default_value=float(mapping.get("default_value", 0.0)),
    )


def select_active_features(
    psych_tests: Mapping[str, PsychTest],
    active_features: ActiveFeatures,
    expected_order: Sequence[str] | None = None,
) -> list[float]:
    """Return active PsychTest values in the one order used by all calculations."""
    feature_order = tuple(expected_order or active_features.names)
    if feature_order != active_features.names:
        raise ValueError("Cluster files use a different feature order than the active mapping")

    values: list[float] = []
    for feature_name in feature_order:
        test = psych_tests.get(active_features.test_by_feature[feature_name])
        if test is None:
            values.append(active_features.default_value)
            continue
        value = next(
            (param.value for param in test.psych_params if param.name == feature_name),
            None,
        )
        values.append(
            active_features.default_value if value is None else float(value)
        )
    return values
