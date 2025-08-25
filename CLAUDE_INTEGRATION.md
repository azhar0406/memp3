# Using memp3 with Claude Desktop

## Setup

1. Install memp3:
   ```bash
   pip install -e .
   ```

2. Configure Claude Desktop by adding this to your `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "memp3": {
         "command": "memp3",
         "args": ["mcp"]
       }
     }
   }
   ```

3. Start the memp3 MCP server:
   ```bash
   memp3 mcp
   ```

## Usage in Claude Conversations

Once configured, you can use memp3 directly in your conversations with Claude:

### Storing Memories
```
Claude, please store this conversation summary using memp3:
"Today we discussed the memp3 project implementation and tested the MCP integration."
```

### Retrieving Memories
```
Claude, can you retrieve my previous conversation about memp3?
```

### Searching Memories
```
Claude, find all memories related to MCP integration.
```

## API Usage

You can also interact with memp3 programmatically:

### Create a Memory
```bash
curl -X POST http://127.0.0.1:3141/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a test memory", "tags": "test,example"}'
```

### Retrieve a Memory
```bash
curl -X GET http://127.0.0.1:3141/memories/<memory-id>
```

### Search Memories
```bash
curl -X GET "http://127.0.0.1:3141/memories?query=test"
```

## File Storage

memp3 stores memories in:
- Audio files: `~/memp3/memory/*.flac`
- Metadata: `~/memp3/index.db` (SQLite database)

This allows you to access your memories even outside of Claude Desktop.