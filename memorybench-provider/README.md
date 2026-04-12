# memdio MemoryBench Provider

Evaluates memdio against LongMemEval and other benchmarks using [MemoryBench](https://github.com/supermemoryai/memorybench).

## Setup

### 1. Start memdio API server

```bash
cd /path/to/memdio
source venv/bin/activate
uvicorn memdio.api.server:app --port 8100
```

### 2. Create an API key

```bash
python -c "from memdio.api.auth import create_api_key; print(create_api_key('bench'))"
```

### 3. Set environment variables

```bash
export MEMDIO_API_URL=http://localhost:8100
export MEMDIO_API_KEY=memdio_<your-key>
```

### 4. Install into MemoryBench

Copy the provider into a cloned MemoryBench repo:

```bash
git clone https://github.com/supermemoryai/memorybench.git
cp src/memdio/index.ts memorybench/src/providers/memdio/index.ts
```

Then register in `memorybench/src/providers/index.ts`:

```typescript
import { memdioProvider } from "./memdio/index.js";

export const providers = {
  // ... existing providers
  memdio: memdioProvider,
};
```

### 5. Run benchmark

```bash
cd memorybench
bun run src/index.ts run -p memdio -b longmem -j openai/gpt-4o
```

## How it works

The provider calls the memdio REST API:

| MemoryBench Phase | memdio API |
|---|---|
| `ingest` | `POST /memories` (one per session) |
| `awaitIndexing` | No-op (synchronous) |
| `search` | `GET /memories?query=` + `GET /memories/semantic?query=` |
| `clear` | `DELETE /memories/{id}` per memory |
