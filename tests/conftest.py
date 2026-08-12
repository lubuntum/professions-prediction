from dataclasses import replace
import pytest

from settings import Settings


@pytest.fixture
def settings(tmp_path):
    return replace(
        Settings.from_env(),
        cluster_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        cluster_count=2,
        backend_url=None,
        backend_email=None,
        backend_password=None,
    )
