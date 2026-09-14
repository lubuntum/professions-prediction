"""Keep generated cluster files current and safe to read."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from mapping import ActiveFeatures
from settings import Settings
from specialist import create_cluster_files, load_specialists, validate_cluster_files


logger = logging.getLogger(__name__)


class ClustersUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClusterState:
    updated_at: datetime
    active_folder: str
    specialists_count: int
    version: int = 1


@dataclass(frozen=True)
class ClusterStatus:
    status: Literal["fresh", "stale", "missing"]
    updated_at: datetime | None


def cluster_status(settings: Settings, active_features: ActiveFeatures) -> ClusterStatus:
    state = read_cluster_state(settings)
    if state is None or not cluster_files_are_valid(settings, active_features, state):
        return ClusterStatus("missing", state.updated_at if state else None)
    return ClusterStatus(
        "stale" if clusters_need_update(settings, active_features, state) else "fresh",
        state.updated_at,
    )


def clusters_need_update(
    settings: Settings,
    active_features: ActiveFeatures,
    state: ClusterState | None = None,
) -> bool:
    """Return true when files or timestamp are missing, invalid or at least TTL hours old."""
    current_state = state or read_cluster_state(settings)
    if current_state is None:
        return True
    if not cluster_files_are_valid(settings, active_features, current_state):
        return True
    age = datetime.now(timezone.utc) - current_state.updated_at
    return age >= timedelta(hours=settings.cluster_update_hours)


def ensure_clusters_ready(settings: Settings, active_features: ActiveFeatures) -> ClusterState:
    """Use fresh clusters, update stale clusters, or fall back to valid old files."""
    current_state = read_cluster_state(settings)
    current_valid = (
        current_state is not None
        and cluster_files_are_valid(settings, active_features, current_state)
    )
    if current_valid and not clusters_need_update(settings, active_features, current_state):
        logger.info("cluster files are fresh")
        return current_state

    try:
        return update_clusters(settings, active_features, force=False)
    except Exception as error:
        logger.exception("cluster update failed")
        latest_state = read_cluster_state(settings)
        latest_valid = (
            latest_state is not None
            and cluster_files_are_valid(settings, active_features, latest_state)
        )
        fallback = latest_state if latest_valid else current_state if current_valid else None
        if fallback is not None:
            logger.warning("using stale cluster files folder=%s", fallback.active_folder)
            return fallback
        raise ClustersUnavailableError("Cluster data is temporarily unavailable") from error


def update_clusters(
    settings: Settings,
    active_features: ActiveFeatures,
    force: bool = True,
) -> ClusterState:
    """Load Specialists and atomically replace generated cluster files."""
    with cluster_update_lock(settings):
        state = read_cluster_state(settings)
        if (
            not force
            and state is not None
            and not clusters_need_update(settings, active_features, state)
        ):
            logger.info("clusters were updated by another request")
            return state
        return create_new_cluster_folder(settings, active_features)


def active_cluster_folder(settings: Settings, state: ClusterState) -> Path:
    folder = (settings.cluster_dir / state.active_folder).resolve()
    if folder.parent != settings.cluster_dir.resolve():
        raise ClustersUnavailableError("Cluster state contains an invalid folder")
    return folder


def cluster_files_are_valid(
    settings: Settings,
    active_features: ActiveFeatures,
    state: ClusterState,
) -> bool:
    try:
        validate_cluster_files(
            active_cluster_folder(settings, state),
            settings.cluster_count,
            active_features.names,
        )
        return True
    except Exception as error:
        logger.exception("cluster validation failed folder=%s: %s", state.active_folder, error)
        return False


def create_new_cluster_folder(
    settings: Settings,
    active_features: ActiveFeatures,
) -> ClusterState:
    logger.info("cluster update started")
    settings.cluster_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    folder_name = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    temporary_folder = settings.cluster_dir / f".updating-{uuid.uuid4().hex}"
    final_folder = settings.cluster_dir / folder_name

    try:
        specialists = load_specialists(settings)
        build = create_cluster_files(
            specialists,
            temporary_folder,
            active_features,
            settings.cluster_count,
        )
        validate_cluster_files(
            temporary_folder,
            settings.cluster_count,
            active_features.names,
        )
        os.replace(temporary_folder, final_folder)
        state = ClusterState(
            updated_at=datetime.now(timezone.utc),
            active_folder=folder_name,
            specialists_count=build.specialists_count,
        )
        write_cluster_state(settings, state)
        logger.info(
            "cluster update completed folder=%s specialists=%s",
            folder_name,
            build.specialists_count,
        )
        return state
    finally:
        if temporary_folder.exists():
            shutil.rmtree(temporary_folder)


def read_cluster_state(settings: Settings) -> ClusterState | None:
    state_path = settings.cache_dir / "cluster_state.json"
    try:
        content = json.loads(state_path.read_text(encoding="utf-8"))
        updated_at = datetime.fromisoformat(content["updated_at"].replace("Z", "+00:00"))
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return ClusterState(
            updated_at=updated_at.astimezone(timezone.utc),
            active_folder=str(content["active_folder"]),
            specialists_count=int(content["specialists_count"]),
            version=int(content.get("version", 1)),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_cluster_state(settings: Settings, state: ClusterState) -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = settings.cache_dir / "cluster_state.json"
    temporary_state = state_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    content = asdict(state)
    content["updated_at"] = state.updated_at.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    temporary_state.write_text(json.dumps(content, indent=2), encoding="utf-8")
    os.replace(temporary_state, state_path)


@contextmanager
def cluster_update_lock(settings: Settings) -> Iterator[None]:
    """Prevent two FastAPI workers from updating the same cluster folder."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = settings.cache_dir / "cluster_update.lock"
    deadline = time.monotonic() + settings.lock_wait_seconds

    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(
                    json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()})
                )
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > settings.abandoned_lock_seconds:
                    lock_path.unlink()
                    logger.warning("removed abandoned cluster lock ageSeconds=%.1f", age)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise ClustersUnavailableError("Timed out waiting for cluster update")
            time.sleep(0.05)

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)
