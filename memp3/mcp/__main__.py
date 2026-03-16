"""Direct entry point for MCP server — skips Typer, minimal startup."""
import asyncio
from memp3.mcp.server import run_mcp_server

asyncio.run(run_mcp_server())
