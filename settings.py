from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    project_root: Path
    cluster_dir: Path
    cache_dir: Path
    mapping_path: Path
    category_path: Path
    cluster_update_hours: float
    lock_wait_seconds: float
    abandoned_lock_seconds: float
    backend_url: str | None
    backend_email: str | None
    backend_password: str | None
    backend_connect_timeout: float
    backend_read_timeout: float
    cluster_count: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")

        def project_path(name: str, default: str) -> Path:
            path = Path(os.getenv(name, default))
            return path if path.is_absolute() else PROJECT_ROOT / path

        return cls(
            project_root=PROJECT_ROOT,
            cluster_dir=project_path("CLUSTER_DATA_DIR", "data"),
            cache_dir=project_path("CLUSTER_CACHE_DIR", "cache"),
            mapping_path=project_path("TEST_MAPPING_PATH", "test_mapping.json"),
            category_path=project_path("DISTANCE_CATEGORY_PATH", "category.csv"),
            cluster_update_hours=float(os.getenv("CLUSTER_UPDATE_HOURS", "24")),
            lock_wait_seconds=float(os.getenv("CLUSTER_LOCK_WAIT_SECONDS", "180")),
            abandoned_lock_seconds=float(os.getenv("CLUSTER_LOCK_STALE_SECONDS", "1800")),
            backend_url=os.getenv("BACKEND_BASE_URL"),
            backend_email=os.getenv("BACKEND_SERVICE_EMAIL"),
            backend_password=os.getenv("BACKEND_SERVICE_PASSWORD"),
            backend_connect_timeout=float(os.getenv("BACKEND_CONNECT_TIMEOUT_SECONDS", "5")),
            backend_read_timeout=float(os.getenv("BACKEND_READ_TIMEOUT_SECONDS", "60")),
            cluster_count=int(os.getenv("PREDICTION_CLUSTER_COUNT", "5")),
        )
