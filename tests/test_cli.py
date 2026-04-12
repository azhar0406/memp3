import pytest
from unittest.mock import patch
from typer.testing import CliRunner
from memdio.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_storage(tmp_path):
    """Use tmp_path for all storage to avoid side effects."""
    with patch("memdio.cli.main._get_storage") as mock:
        from memdio.core.storage import StorageManager
        storage = StorageManager(base_path=str(tmp_path / "memdio"))
        mock.return_value = storage
        yield storage


def test_encode_command():
    result = runner.invoke(app, ["encode", "test text"])
    assert result.exit_code == 0
    assert "Memory stored with ID:" in result.stdout


def test_search_command():
    result = runner.invoke(app, ["search", "test"])
    assert result.exit_code == 0


def test_list_command():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0


def test_decode_not_found():
    result = runner.invoke(app, ["decode", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1
