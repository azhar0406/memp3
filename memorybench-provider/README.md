# memp3 MemoryBench Provider

Evaluates memp3 against LongMemEval and other benchmarks using [MemoryBench](https://github.com/supermemoryai/memorybench).

## Setup

### 1. Start memp3 API server

```bash
cd /path/to/memp3
source venv/bin/activate
uvicorn memp3.api.server:app --port 8100
```

### 2. Create an API key

```bash
python -c "from memp3.api.auth import create_api_key; print(create_api_key('bench'))"
```

### 3. Set environment variables

```bash
export MEMP3_API_URL=http://localhost:8100
export MEMP3_API_KEY=memp3_<your-key>
```

### 4. Install into MemoryBench

Copy the provider into a cloned MemoryBench repo:

```bash
git clone https://github.com/supermemoryai/memorybench.git
cp src/memp3/index.ts memorybench/src/providers/memp3/index.ts
```

Then register in `memorybench/src/providers/index.ts`:

```typescript
import { memp3Provider } from "./memp3/index.js";

export const providers = {
  // ... existing providers
  memp3: memp3Provider,
};
```

### 5. Run benchmark

```bash
cd memorybench
bun run src/index.ts run -p memp3 -b longmem -j openai/gpt-4o
```

## How it works

The provider calls the memp3 REST API:

| MemoryBench Phase | memp3 API |
|---|---|
| `ingest` | `POST /memories` (one per session) |
| `awaitIndexing` | No-op (synchronous) |
| `search` | `GET /memories?query=` + `GET /memories/semantic?query=` |
| `clear` | `DELETE /memories/{id}` per memory |
