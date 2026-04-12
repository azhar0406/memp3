# memdio

**AI memory encoded in sound. Lossless, fast, and hardware-adaptive.**

memdio encodes text into FLAC audio files with Reed-Solomon error correction, stores them in SQLite, and provides semantic search via MCP for Claude Desktop.

## Architecture

```
Text  -->  zlib compress  -->  Reed-Solomon ECC  -->  Frequency encoding (200-4000Hz)
      -->  FLAC compress  -->  SQLite blob storage

Search: FTS5 (word match) + FastEmbed ONNX + sqlite-vector (semantic similarity)
```

Everything lives in one SQLite database per user. No external services, no separate index files.

## Performance

| Operation | Time |
|-----------|------|
| Store memory | 67ms |
| Retrieve memory | 15ms |
| Word search (FTS5) | <1ms |
| Semantic search | 150ms |
| MCP server cold start | 127ms |

## Installation

```bash
pip install -e .

# With semantic search support
pip install -e ".[search]"
```

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memdio": {
      "command": "/path/to/venv/bin/python",
      "args": ["-u", "-m", "memdio.mcp"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Encode text into FLAC audio. Returns memory ID |
| `retrieve_memory` | Decode audio back to text by UUID |
| `search_memories` | FTS5 word-level search (any word matches) |
| `semantic_search` | Meaning-based search via embeddings |
| `list_memories` | List all stored memories |
| `delete_memory` | Delete a memory by UUID |

## CLI

```bash
# Store and retrieve
memdio encode "Meeting at 3pm in Thane" --tags "meeting,thane"
memdio decode <memory-id>

# Search
memdio search "meeting"
memdio semantic-search "schedule appointment"

# Manage
memdio list
memdio delete <memory-id>
memdio info <memory-id>
memdio stats

# Export
memdio export-flac <memory-id>
memdio export-wav <memory-id>

# Servers
memdio mcp          # MCP stdio server (Claude Desktop)
memdio serve        # REST API server
```

## REST API

```bash
# Start server
memdio serve

# Create API key
memdio create-key alice

# Use API
curl -X POST http://localhost:8000/memories \
  -H "Authorization: Bearer memdio_xxxx" \
  -H "Content-Type: application/json" \
  -d '{"content": "Meeting at 3pm", "tags": "meeting"}'

curl http://localhost:8000/memories?query=meeting \
  -H "Authorization: Bearer memdio_xxxx"
```

## How It Works

### Encoding Pipeline

```
"Hello World"
    |
    v
UTF-8 bytes --> zlib compress --> Reed-Solomon ECC (14% redundancy)
    |
    v
CRC32 header (magic + version + length + checksum)
    |
    v
Frequency mapping: each byte -> 200-4000Hz tone (256 slots)
    |
    v
48kHz audio signal with Hann windowing (reduces spectral leakage)
    |
    v
FLAC lossless compression --> SQLite blob
```

### Search

- **FTS5**: Word-level matching. "Lakshya wedding" finds "Lakshya is getting married" because "Lakshya" matches.
- **Semantic**: FastEmbed (ONNX) generates 384-dim embeddings, sqlite-vector does cosine similarity. "motorcycle" finds "bike" (0.837 score).

### Error Correction

Reed-Solomon RS(255,223) recovers the original text even if 5% of audio samples are corrupted.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio encoding | numpy, scipy, soundfile |
| Error correction | reedsolo (Reed-Solomon) |
| Storage | SQLite (WAL mode, FLAC blobs) |
| Word search | SQLite FTS5 |
| Semantic search | FastEmbed (ONNX) + sqlite-vector |
| MCP server | Lightweight JSON-RPC over stdio |
| REST API | FastAPI |
| CLI | Typer |

## Multi-tenant SaaS

Per-user isolation with API key auth:

```
/data/memdio/users/
  alice/index.db    <-- all of Alice's memories
  bob/index.db      <-- all of Bob's memories
```

Delete user = delete their folder. No shared database.

## License

MIT
