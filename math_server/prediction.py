from __future__ import annotations

import logging
from pathlib import Path

from clusters import ClusterState, active_cluster_folder, ensure_clusters_ready
from mapping import ActiveFeatures, load_active_features, missing_active_features
from models import Pupil
from settings import Settings
from math_server.models import MathPrediction, MathPredictionItem
from math_server.calculator import (
    BELBIN_COLS,
    ParamsInitial,
    ParamsRaw,
    calc_pupil_belbin,
    calc_pupil_bennet,
    calc_pupil_eysenck,
    calc_recommendation_complex,
)
from math_server.params import (
    load_params_initial,
    load_params_raw,
    params_paths_for,
)

class IncompletePupilError(RuntimeError):
    def __init__(self, missing: list[str]):
        super().__init__("Pupil data is incomplete")
        self.missing = missing

logger = logging.getLogger(__name__)


def list_available_professions(folder: Path) -> list[str]:
    """Список профессий, для которых есть файлы параметров в папке."""
    params_folder = folder / "math_params"
    if not params_folder.is_dir():
        return []
    professions = []
    for path in sorted(params_folder.glob("*.json")):
        if path.name.endswith("_raw.json"):
            continue
        professions.append(path.stem)
    return professions


def load_params_for_profession(folder: Path, profession_slug: str) -> tuple[ParamsInitial, ParamsRaw]:
    params_path, raw_path = params_paths_for(folder, profession_slug)
    return load_params_initial(params_path), load_params_raw(raw_path)


def _pupil_row(pupil: Pupil) -> dict:
    if pupil.age is None:
        raise ValueError(f"Pupil {pupil.pupil_id} has no age")
    row: dict = {"age": pupil.age}
    for test in pupil.psych_tests.values():
        for param in test.psych_params:
            if param.name in (
                "extrav_introver_score",
                "neirotizm_score",
                "engineering_thinking_level",
                *BELBIN_COLS,
            ):
                row[param.name] = param.value or 0.0
    return row


def _predict_one(
    profession: str,
    params: ParamsInitial,
    params_raw: ParamsRaw,
    row: dict,
) -> MathPredictionItem:
    eysenck_norm = calc_pupil_eysenck(row, params, params_raw)
    belbin_norm = calc_pupil_belbin(row, params, params_raw)
    bennet_norm = calc_pupil_bennet(row, params, params_raw)

    total_score = eysenck_norm + belbin_norm + bennet_norm
    w = params.weights

    additive_utility = (
        eysenck_norm * w["eysenck"]
        + belbin_norm * w["belbin"]
        + bennet_norm * w["bennet"]
    )
    weighted_product = (
        eysenck_norm ** w["eysenck"]
        + belbin_norm ** w["belbin"]
        + bennet_norm ** w["bennet"]
    )

    norm_utility = (1 + additive_utility) / (1 + params_raw.max_utility)
    norm_product = (1 + weighted_product) / (1 + params_raw.max_product)
    final_index = (norm_utility + norm_product) / 2
    percent = min(final_index * 100, 100)

    recommendation = "рекомендуем" if total_score > 75 else "не рекомендуем"
    recommendation_complex = calc_recommendation_complex(row, recommendation)

    return MathPredictionItem(
        profession=profession,
        percentage=round(percent, 1),
        recommendation=recommendation,
        recommendationComplex=recommendation_complex,
        aizenNorm=round(eysenck_norm, 2),
        belbinNorm=round(belbin_norm, 2),
        bennetNorm=round(bennet_norm, 2),
        finalScore=round(total_score, 2),
        utility=round(additive_utility, 4),
    )


def predict_math(pupil: Pupil, settings: Settings) -> MathPrediction:
    active_features = load_active_features(settings.mapping_path)

    missing = missing_active_features(pupil.psych_tests, active_features)
    if missing:
        raise IncompletePupilError(missing)
    
    state = ensure_clusters_ready(settings, active_features)
    folder = active_cluster_folder(settings, state)
    params_folder = folder / "math_params"

    professions = list_available_professions(folder)
    if not professions:
        raise RuntimeError("No math params found for any profession")

    row = _pupil_row(pupil)
    items = [
        _predict_one(slug, *load_params_for_profession(params_folder, slug), row)
        for slug in professions
    ]
    return MathPrediction(pupil_id=pupil.pupil_id, professions=items)