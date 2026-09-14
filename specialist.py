"""Load Specialist data and turn it into cluster files."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mapping import ActiveFeatures, load_active_features, select_active_features
from math_server.calculator import build_params_for_profession
from math_server.params import params_paths_for, save_params_initial, save_params_raw
from models import Specialist
from settings import Settings


@dataclass(frozen=True)
class ClusterBuild:
    specialists_count: int
    feature_count: int
    available_cluster_counts: tuple[int, ...]


def required_cluster_files(cluster_count: int) -> tuple[str, ...]:
    return (
        "specialists.pkl",
        "scaler.pkl",
        f"kmeans_{cluster_count}.pkl",
        "specialist_features.npy",
        "professions.npy",
        "feature_names.npy",
        "cluster_manifest.json",
    )


def load_specialists(settings: Settings, session: requests.Session | None = None) -> list[Specialist]:
    """Load Specialist and PsychTest data from the protected backend endpoint."""
    if not settings.backend_url or not settings.backend_email or not settings.backend_password:
        raise RuntimeError(
            "BACKEND_BASE_URL, BACKEND_SERVICE_EMAIL and BACKEND_SERVICE_PASSWORD are required"
        )

    http = session or requests.Session()
    timeout = (settings.backend_connect_timeout, settings.backend_read_timeout)
    login = http.post(
        urljoin(settings.backend_url.rstrip("/") + "/", "api/auth/login"),
        json={"email": settings.backend_email, "password": settings.backend_password},
        timeout=timeout,
    )
    login.raise_for_status()
    token = login.text.strip()
    if not token:
        raise RuntimeError("Backend returned an empty service token")

    response = http.get(
        urljoin(settings.backend_url.rstrip("/") + "/", "api/specialists/reference-data"),
        headers={"Authorization": token},
        timeout=timeout,
    )
    response.raise_for_status()
    specialists = response.json()
    if not isinstance(specialists, list):
        raise RuntimeError("Backend Specialist response is not a list")
    return [Specialist.model_validate(item) for item in specialists]


def create_cluster_files(
    specialists: list[Specialist],
    output_dir: Path,
    active_features: ActiveFeatures,
    cluster_count: int,
) -> ClusterBuild:
    """Scale Specialist features, run KMeans and write all cluster files."""
    if not specialists:
        raise ValueError("Backend returned no Specialist data")

    rows = []
    for specialist in specialists:
        row = {
            "specialist_id": specialist.specialist_id,
            "profession": specialist.profession,
        }
        row.update(
            zip(
                active_features.names,
                select_active_features(specialist.psych_tests, active_features),
                strict=True,
            )
        )
        rows.append(row)

    specialist_table = pd.DataFrame(rows)
    #filter for specialists
    feature_columns = list(active_features.names)
    has_data = (specialist_table[feature_columns].astype(float) != 0.0).any(axis=1)
    before = len(specialist_table)
    specialist_table = specialist_table[has_data].reset_index(drop=True)
    print(
        "specialists filtered: kept=%d, dropped=%d",
        len(specialist_table), before - len(specialist_table),
    )
    if specialist_table.empty:
        raise ValueError("All Specialists have empty feature vectors")
    #end of filter
    if specialist_table["specialist_id"].duplicated().any():
        raise ValueError("Specialist data contains duplicate IDs")
    if specialist_table["profession"].isna().any() or (
        specialist_table["profession"].str.strip() == ""
    ).any():
        raise ValueError("Every Specialist must have a Profession")

    feature_matrix = specialist_table[list(active_features.names)].astype(float).to_numpy()
    professions = specialist_table["profession"].to_numpy()
    maximum_cluster_count = min(len(np.unique(professions)), len(specialist_table) - 1)
    available_cluster_counts = tuple(range(2, maximum_cluster_count + 1))
    if cluster_count not in available_cluster_counts:
        raise ValueError(
            f"PREDICTION_CLUSTER_COUNT={cluster_count} is unavailable; "
            f"available values: {available_cluster_counts}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "specialists.pkl").open("wb") as target:
        pickle.dump(specialist_table, target)

    scaler = StandardScaler()
    scaled_specialists = scaler.fit_transform(feature_matrix)
    with (output_dir / "scaler.pkl").open("wb") as target:
        pickle.dump(scaler, target)

    metrics: dict[int, dict[str, float]] = {}
    for current_count in available_cluster_counts:
        kmeans = KMeans(n_clusters=current_count, random_state=42, n_init=50)
        labels = kmeans.fit_predict(scaled_specialists)
        with (output_dir / f"kmeans_{current_count}.pkl").open("wb") as target:
            pickle.dump(kmeans, target)
        np.save(output_dir / f"labels_{current_count}.npy", labels)
        metrics[current_count] = {
            "silhouette_score": float(silhouette_score(scaled_specialists, labels)),
            "davies_bouldin_index": float(davies_bouldin_score(scaled_specialists, labels)),
            "calinski_harabasz_index": float(
                calinski_harabasz_score(scaled_specialists, labels)
            ),
        }

    with (output_dir / "cluster_metrics.pkl").open("wb") as target:
        pickle.dump(metrics, target)

    pca_2d = PCA(n_components=2)
    np.save(output_dir / "specialists_pca_2d.npy", pca_2d.fit_transform(scaled_specialists))
    with (output_dir / "pca_2d.pkl").open("wb") as target:
        pickle.dump(pca_2d, target)

    pca_3d = PCA(n_components=3)
    np.save(output_dir / "specialists_pca_3d.npy", pca_3d.fit_transform(scaled_specialists))
    with (output_dir / "pca_3d.pkl").open("wb") as target:
        pickle.dump(pca_3d, target)

    np.save(output_dir / "specialist_features.npy", scaled_specialists)
    np.save(output_dir / "professions.npy", professions)
    np.save(output_dir / "feature_names.npy", np.asarray(active_features.names, dtype=object))
    (output_dir / "cluster_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "cluster_count": cluster_count,
                "available_cluster_counts": list(available_cluster_counts),
                "feature_names": list(active_features.names),
                "specialists_count": len(specialist_table),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    #После обновления кластеров, обновляем мат модели сразу
    #Общее обновление двух систем одновременно
    math_params_folder = output_dir / "math_params"
    math_params_folder.mkdir()
    for profession, group in specialist_table.groupby("profession"):
        rows = group.to_dict("records")
        params, raw = build_params_for_profession(rows)
        params_path, raw_path = params_paths_for(math_params_folder, profession)
        save_params_initial(params, params_path)
        save_params_raw(raw, raw_path)
    validate_cluster_files(output_dir, cluster_count, active_features.names)
    return ClusterBuild(len(specialist_table), len(active_features.names), available_cluster_counts)


def validate_cluster_files(
    cluster_folder: Path,
    cluster_count: int,
    expected_feature_names: tuple[str, ...],
) -> None:
    """Reject incomplete or incompatible cluster folders."""
    missing = [
        name for name in required_cluster_files(cluster_count) if not (cluster_folder / name).is_file()
    ]
    if missing:
        raise ValueError(f"Cluster folder is incomplete; missing files: {missing}")

    feature_names = tuple(
        str(value)
        for value in np.load(cluster_folder / "feature_names.npy", allow_pickle=True).tolist()
    )
    if feature_names != expected_feature_names:
        raise ValueError("Cluster feature order does not match the active mapping")

    scaled_specialists = np.load(cluster_folder / "specialist_features.npy")
    professions = np.load(cluster_folder / "professions.npy", allow_pickle=True)
    specialists = pd.read_pickle(cluster_folder / "specialists.pkl")
    if scaled_specialists.ndim != 2 or scaled_specialists.shape[1] != len(feature_names):
        raise ValueError("Specialist feature matrix has an invalid shape")
    if (
        scaled_specialists.shape[0] == 0
        or len(professions) != scaled_specialists.shape[0]
        or len(specialists) != scaled_specialists.shape[0]
    ):
        raise ValueError("Cluster files contain inconsistent Specialist counts")

    with (cluster_folder / "scaler.pkl").open("rb") as source:
        pickle.load(source)
    with (cluster_folder / f"kmeans_{cluster_count}.pkl").open("rb") as source:
        pickle.load(source)
    #Проверка папок для математики
    math_params_folder = cluster_folder / "math_params"
    if not math_params_folder.is_dir():
        raise ValueError("Cluster folder is missing math_params directory")
    if not any(math_params_folder.glob("*.json")):
        raise ValueError("Cluster folder has no math params files")


def main() -> None:
    settings = Settings.from_env()
    active_features = load_active_features(settings.mapping_path)
    from clusters import update_clusters

    state = update_clusters(settings, active_features, force=True)
    print(
        f"Clusters updated: folder={state.active_folder}, "
        f"specialists={state.specialists_count}, updated={state.updated_at.isoformat()}"
    )


if __name__ == "__main__":
    main()
