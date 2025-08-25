# memp3 - MVP

**AI memory encoded in sound. Lossless, fast, and hardware-adaptive.**

`memp3` is a cutting-edge AI memory system that compresses and stores textual knowledge into FLAC audio files.

## 🚀 MVP Features

- 🔊 **Text-to-FLAC Encoding**: Encodes text to audio memory
- ⚡ **Fast Retrieval**: Retrieves memories by ID
- 📦 **File Storage**: Stores memories as FLAC files
- 🔎 **Basic Search**: Searches memories by content
- 🌐 **MCP Ready**: Integrates with Claude Desktop

## 🔧 Tech Stack (MVP)

- **Languages**: Python 3.11+
- **Audio**: soundfile, numpy, scipy
- **Storage**: File system + SQLite
- **API**: FastAPI, Uvicorn
- **CLI**: Typer

## 📦 Installation

```bash
# Install with pip (development version)
pip install -e .

# Eventually will support:
# uv tool install memp3
```

## 🔧 Claude Desktop Integration (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memp3": {
      "command": "memp3",
      "args": [
        "mcp"
      ]
    }
  }
}
```

Then start the MCP server:
```bash
memp3 mcp
```

## 🔍 CLI Usage

```bash
# Encode text to audio memory
memp3 encode "Hello World" --tags "greeting"

# Decode audio memory to text
memp3 decode <memory-id>

# Search memories
memp3 search "Hello"

# List all memories
memp3 list

# Start MCP server
memp3 mcp
```

## 🌐 MCP API Endpoints

Once the MCP server is running, you can interact with it via REST API:

- `POST /memories` - Create a new memory
- `GET /memories/{id}` - Retrieve a memory by ID
- `GET /memories` - Search memories
- `GET /health` - Health check

Example:
```bash
# Create a memory
curl -X POST http://127.0.0.1:3141/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a test memory", "tags": "test"}'

# Retrieve a memory
curl -X GET http://127.0.0.1:3141/memories/<memory-id>
```

## 📜 License

MIT License. See [`LICENSE`](./LICENSE) for details.

---

### 💡 Why the name "memp3"?

A play on "memory" + "MP3" — `memp3` encodes structured memory into audio.