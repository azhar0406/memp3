import typer
import uvicorn
from typing import Optional
from memp3.core.storage import StorageManager
from memp3.mcp.server import app as mcp_app

app = typer.Typer()
storage = StorageManager()

@app.command()
def encode(
    text: str = typer.Argument(..., help="Text to encode to audio memory"),
    tags: Optional[str] = typer.Option(None, "-t", "--tags", help="Tags for the memory")
):
    """Encode text to audio memory"""
    try:
        mem_id = storage.store(text, tags)
        typer.echo(f"Memory stored with ID: {mem_id}")
    except Exception as e:
        typer.echo(f"Error encoding memory: {e}")
        raise typer.Exit(code=1)

@app.command()
def decode(
    mem_id: str = typer.Argument(..., help="Memory ID to decode")
):
    """Decode audio memory to text"""
    try:
        content = storage.retrieve(mem_id)
        typer.echo(f"Content: {content}")
    except KeyError:
        typer.echo(f"Memory {mem_id} not found")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error decoding memory: {e}")
        raise typer.Exit(code=1)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query")
):
    """Search memories"""
    try:
        results = storage.search(query)
        if not results:
            typer.echo("No memories found")
            return
            
        typer.echo(f"Found {len(results)} memory(s):")
        for result in results:
            typer.echo(f"  {result['id']}: {result['content'][:50]}...")
    except Exception as e:
        typer.echo(f"Error searching memories: {e}")
        raise typer.Exit(code=1)

@app.command()
def list():
    """List all memories"""
    try:
        results = storage.list_all()
        if not results:
            typer.echo("No memories found")
            return
            
        typer.echo(f"Found {len(results)} memory(s):")
        for result in results:
            typer.echo(f"  {result['id']}: {result['content'][:50]}...")
    except Exception as e:
        typer.echo(f"Error listing memories: {e}")
        raise typer.Exit(code=1)

@app.command()
def mcp(
    host: str = typer.Option("127.0.0.1", help="Host to bind the MCP server to"),
    port: int = typer.Option(3141, help="Port to bind the MCP server to")
):
    """Start MCP server for Claude Desktop"""
    typer.echo(f"Starting MCP server on {host}:{port}")
    uvicorn.run(mcp_app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app()