"""Calculate a Prediction for one Pupil from current cluster files."""

from __future__ import annotations

import csv
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from clusters import ClusterState, active_cluster_folder, ensure_clusters_ready
from mapping import ActiveFeatures, load_active_features, missing_active_features, select_active_features
from models import Prediction, Pupil
from settings import Settings
from specialist import validate_cluster_files


logger = logging.getLogger(__name__)


class PredictionError(RuntimeError):
    pass

class IncompletePupilError(PredictionError):
    def __init__(self, missing: list[str]):
        super().__init__("Pupil data is incomplete")
        self.missing = missing


@dataclass(frozen=True)
class LoadedClusters:
    scaler: object
    kmeans: object
    specialist_features: np.ndarray
    professions: np.ndarray
    specialist_ids: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class DistanceCategory:
    minimum: float
    maximum: float
    name: str


loaded_folder: str | None = None
loaded_clusters: LoadedClusters | None = None


def predict_pupil(pupil: Pupil, settings: Settings) -> Prediction:
    """Ensure clusters are current, then find the nearest Specialist for a Pupil."""
    started_at = time.perf_counter()
    logger.info("prediction started pupilId=%s", pupil.pupil_id)
    active_features = load_active_features(settings.mapping_path)

    missing = missing_active_features(pupil.psych_tests, active_features)
    if missing:
        raise IncompletePupilError(missing)

    cluster_state = ensure_clusters_ready(settings, active_features)
    clusters = get_loaded_clusters(settings, active_features, cluster_state)

    pupil_values = select_active_features(
        pupil.psych_tests,
        active_features,
        clusters.feature_names,
    )
    pupil_matrix = np.asarray(pupil_values, dtype=float).reshape(1, -1)
    scaled_pupil = clusters.scaler.transform(pupil_matrix)
    cluster = int(clusters.kmeans.predict(scaled_pupil)[0])
    specialist_positions = np.where(clusters.kmeans.labels_ == cluster)[0]
    if len(specialist_positions) == 0:
        raise PredictionError("Predicted cluster contains no Specialists")

    distances = cdist(
        scaled_pupil,
        clusters.specialist_features[specialist_positions],
    )[0]
    nearest_position = int(specialist_positions[int(np.argmin(distances))])
    distance = float(np.min(distances))
    prediction = Prediction(
        pupil_id=pupil.pupil_id,
        cluster=cluster,
        predicted_profession=str(clusters.professions[nearest_position]),
        nearest_specialist_id=int(clusters.specialist_ids[nearest_position]),
        distance=distance,
        confidence_category=category_for_distance(distance, settings.category_path),
    )
    logger.info(
        "prediction completed pupilId=%s durationMs=%.1f",
        pupil.pupil_id,
        (time.perf_counter() - started_at) * 1000,
    )
    return prediction


def get_loaded_clusters(
    settings: Settings,
    active_features: ActiveFeatures,
    state: ClusterState,
) -> LoadedClusters:
    """Load a new cluster folder once and reuse it until the next update."""
    global loaded_folder, loaded_clusters
    if loaded_clusters is not None and loaded_folder == state.active_folder:
        return loaded_clusters

    folder = active_cluster_folder(settings, state)
    validate_cluster_files(folder, settings.cluster_count, active_features.names)
    with (folder / "scaler.pkl").open("rb") as source:
        scaler = pickle.load(source)
    with (folder / f"kmeans_{settings.cluster_count}.pkl").open("rb") as source:
        kmeans = pickle.load(source)
    specialists = pd.read_pickle(folder / "specialists.pkl")
    loaded_clusters = LoadedClusters(
        scaler=scaler,
        kmeans=kmeans,
        specialist_features=np.load(folder / "specialist_features.npy"),
        professions=np.load(folder / "professions.npy", allow_pickle=True),
        specialist_ids=specialists["specialist_id"].to_numpy(),
        feature_names=tuple(
            str(value)
            for value in np.load(folder / "feature_names.npy", allow_pickle=True).tolist()
        ),
    )
    loaded_folder = state.active_folder
    return loaded_clusters


def category_for_distance(distance: float, category_path: Path) -> str:
    for category in load_distance_categories(category_path):
        if category.minimum <= distance <= category.maximum:
            return category.name
    return "Не определено"


def load_distance_categories(category_path: Path) -> tuple[DistanceCategory, ...]:
    categories: list[DistanceCategory] = []
    with category_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            maximum = float("inf") if row["max"] == "inf" else float(row["max"])
            categories.append(
                DistanceCategory(float(row["min"]), maximum, row["category"])
            )
    if not categories:
        raise ValueError("Distance category configuration is empty")
    return tuple(categories)
