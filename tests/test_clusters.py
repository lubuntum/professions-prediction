import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import clusters
from clusters import (
    ClusterState,
    ensure_clusters_ready,
    read_cluster_state,
    update_clusters,
    write_cluster_state,
)
from mapping import load_active_features
from specialist import ClusterBuild


def marker_validator(folder: Path, _count: int, _features: tuple[str, ...]) -> None:
    if not (folder / "cluster.ok").is_file():
        raise ValueError("cluster files missing")


def install_state(settings, age_hours: float, with_files: bool = True) -> ClusterState:
    folder = settings.cluster_dir / "existing"
    folder.mkdir(parents=True, exist_ok=True)
    if with_files:
        (folder / "cluster.ok").write_text("valid", encoding="utf-8")
    state = ClusterState(
        datetime.now(timezone.utc) - timedelta(hours=age_hours),
        "existing",
        3,
    )
    write_cluster_state(settings, state)
    return state


def test_fresh_clusters_are_not_updated(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    expected = install_state(settings, age_hours=1)
    monkeypatch.setattr(
        clusters,
        "update_clusters",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not update")),
    )

    assert ensure_clusters_ready(settings, active_features) == expected


def test_stale_clusters_are_updated(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    install_state(settings, age_hours=25)
    updated = ClusterState(datetime.now(timezone.utc), "new", 4)
    calls = []

    def fake_update(*_args, **_kwargs):
        calls.append(1)
        return updated

    monkeypatch.setattr(clusters, "update_clusters", fake_update)

    assert ensure_clusters_ready(settings, active_features) == updated
    assert len(calls) == 1


def test_missing_cluster_files_trigger_update(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    install_state(settings, age_hours=1, with_files=False)
    updated = ClusterState(datetime.now(timezone.utc), "new", 4)
    monkeypatch.setattr(clusters, "update_clusters", lambda *_args, **_kwargs: updated)

    assert ensure_clusters_ready(settings, active_features) == updated


def test_successful_update_changes_timestamp(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    old = install_state(settings, age_hours=25)
    monkeypatch.setattr(clusters, "load_specialists", lambda _settings: [object()])

    def fake_create(_specialists, folder, _features, _count):
        folder.mkdir()
        (folder / "cluster.ok").write_text("valid", encoding="utf-8")
        return ClusterBuild(1, len(active_features.names), (2,))

    monkeypatch.setattr(clusters, "create_cluster_files", fake_create)

    updated = update_clusters(settings, active_features)

    assert updated.updated_at > old.updated_at
    assert read_cluster_state(settings) == updated


def test_failed_update_does_not_change_timestamp(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    old = install_state(settings, age_hours=25)
    monkeypatch.setattr(
        clusters,
        "load_specialists",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
    )

    with pytest.raises(RuntimeError):
        update_clusters(settings, active_features)

    assert read_cluster_state(settings) == old


def test_concurrent_requests_update_clusters_once(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    calls = 0
    monkeypatch.setattr(clusters, "load_specialists", lambda _settings: [object()])

    def fake_create(_specialists, folder, _features, _count):
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        folder.mkdir()
        (folder / "cluster.ok").write_text("valid", encoding="utf-8")
        return ClusterBuild(1, len(active_features.names), (2,))

    monkeypatch.setattr(clusters, "create_cluster_files", fake_create)

    with ThreadPoolExecutor(max_workers=3) as executor:
        states = list(
            executor.map(lambda _: ensure_clusters_ready(settings, active_features), range(3))
        )

    assert calls == 1
    assert len({state.active_folder for state in states}) == 1


def test_lock_timeout_uses_valid_stale_clusters(settings, monkeypatch):
    active_features = load_active_features(settings.mapping_path)
    short_wait = replace(settings, lock_wait_seconds=0.01, abandoned_lock_seconds=3600)
    monkeypatch.setattr(clusters, "validate_cluster_files", marker_validator)
    expected = install_state(short_wait, age_hours=25)
    lock_path = short_wait.cache_dir / "cluster_update.lock"
    lock_path.write_text("{}", encoding="utf-8")

    try:
        assert ensure_clusters_ready(short_wait, active_features) == expected
    finally:
        lock_path.unlink(missing_ok=True)
