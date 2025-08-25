import pytest
from typer.testing import CliRunner
from memp3.cli.main import app

runner = CliRunner()

def test_encode_command():
    """Test the encode command"""
    result = runner.invoke(app, ["encode", "test text"])
    assert result.exit_code == 0
    assert "Memory stored with ID:" in result.stdout

def test_search_command():
    """Test the search command"""
    result = runner.invoke(app, ["search", "test"])
    assert result.exit_code == 0

def test_list_command():
    """Test the list command"""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0

def test_mcp_command():
    """Test the mcp command"""
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert "Starting MCP server..." in result.stdout