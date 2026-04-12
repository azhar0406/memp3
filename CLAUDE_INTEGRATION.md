# Using memdio with Claude Desktop

## Setup

1. Install memdio:
   ```bash
   pip install -e .
   ```

2. Configure Claude Desktop by adding this to your `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "memdio": {
         "command": "memdio",
         "args": ["mcp"]
       }
     }
   }
   ```

3. Start the memdio MCP server:
   ```bash
   memdio mcp
   ```

## Usage in Claude Conversations

Once configured, you can use memdio directly in your conversations with Claude:

### Storing Memories
```
Claude, please store this conversation summary using memdio:
"Today we discussed the memdio project implementation and tested the MCP integration."
```

### Retrieving Memories
```
Claude, can you retrieve my previous conversation about memdio?
```

### Searching Memories
```
Claude, find all memories related to MCP integration.
```

## API Usage

You can also interact with memdio programmatically:

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

memdio stores memories in:
- Audio files: `~/memdio/memory/*.flac`
- Metadata: `~/memdio/index.db` (SQLite database)

This allows you to access your memories even outside of Claude Desktop.