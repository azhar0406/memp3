import pytest

from memp3.core.storage import StorageManager


@pytest.fixture
def storage(tmp_path):
    """Provide a StorageManager backed by a temp directory."""
    sm = StorageManager(base_path=str(tmp_path / "memp3"))
    yield sm
    sm.close()
