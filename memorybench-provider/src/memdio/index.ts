/**
 * memdio provider for MemoryBench.
 *
 * Connects to a running memdio FastAPI server and implements the MemoryBench
 * Provider interface for benchmarking against LongMemEval and other datasets.
 *
 * Prerequisites:
 *   1. Start memdio API server: uvicorn memdio.api.server:app --port 8100
 *   2. Create an API key: python -c "from memdio.api.auth import create_api_key; print(create_api_key('bench'))"
 *   3. Set env vars: MEMDIO_API_URL, MEMDIO_API_KEY
 */

const MEMDIO_API_URL = process.env.MEMDIO_API_URL || "http://localhost:8100";
const MEMDIO_API_KEY = process.env.MEMDIO_API_KEY || "";

interface MemoryResult {
  id: string;
  content: string;
  created_at?: string;
  score?: number;
  tags?: string;
}

// Track stored memory IDs per container for cleanup
const containerMemories = new Map<string, string[]>();

async function apiCall(
  method: string,
  path: string,
  body?: unknown
): Promise<unknown> {
  const url = `${MEMDIO_API_URL}${path}`;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${MEMDIO_API_KEY}`,
    "Content-Type": "application/json",
  };

  const res = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`memdio API error ${res.status}: ${text}`);
  }

  return res.json();
}

/**
 * memdio MemoryBench Provider
 *
 * This implements the Provider interface from MemoryBench.
 * To use with MemoryBench, copy this file into the memorybench repo at:
 *   src/providers/memdio/index.ts
 * and register it in src/providers/index.ts
 */
export const memdioProvider = {
  name: "memdio",

  concurrency: {
    default: 10,
    ingest: 20,
    indexing: 1,
    search: 10,
    answer: 5,
    evaluate: 5,
  },

  async initialize(): Promise<void> {
    if (!MEMDIO_API_KEY) {
      throw new Error(
        "MEMDIO_API_KEY not set. Create one with: python -c \"from memdio.api.auth import create_api_key; print(create_api_key('bench'))\""
      );
    }
    // Health check
    const res = await fetch(`${MEMDIO_API_URL}/health`);
    if (!res.ok) {
      throw new Error(
        `memdio API not reachable at ${MEMDIO_API_URL}. Start with: uvicorn memdio.api.server:app --port 8100`
      );
    }
    console.log(`memdio provider initialized (${MEMDIO_API_URL})`);
  },

  async ingest(
    sessions: Array<{ messages: Array<{ role: string; content: string; timestamp?: string }> }>,
    options: { containerTag: string }
  ): Promise<{ documentIds: string[] }> {
    const documentIds: string[] = [];
    const tag = options.containerTag;

    for (const session of sessions) {
      // Format session as text block (matching Python benchmark format)
      const lines: string[] = [];
      for (const msg of session.messages) {
        if (msg.timestamp) {
          lines.push(`[Date: ${msg.timestamp}]`);
        }
        lines.push(`${msg.role}: ${msg.content}`);
      }
      const content = lines.join("\n");

      if (content.trim()) {
        const result = (await apiCall("POST", "/memories", {
          content,
          tags: tag,
        })) as { id: string };
        documentIds.push(result.id);
      }
    }

    // Track for cleanup
    containerMemories.set(tag, documentIds);
    return { documentIds };
  },

  async awaitIndexing(
    _result: { documentIds: string[] },
    _containerTag: string
  ): Promise<void> {
    // memdio indexes synchronously during ingest — no-op
  },

  async search(
    query: string,
    _options: { containerTag: string; topK?: number }
  ): Promise<MemoryResult[]> {
    const topK = _options.topK || 10;
    const results: Map<string, MemoryResult> = new Map();

    // FTS5 word search
    try {
      const ftsRes = (await apiCall(
        "GET",
        `/memories?query=${encodeURIComponent(query)}`
      )) as { results: MemoryResult[] };
      for (const r of ftsRes.results.slice(0, topK)) {
        results.set(r.id, r);
      }
    } catch {
      // FTS may fail on empty queries
    }

    // Semantic search
    try {
      const semRes = (await apiCall(
        "GET",
        `/memories/semantic?query=${encodeURIComponent(query)}&top_k=${topK}`
      )) as { results: MemoryResult[] };
      for (const r of semRes.results) {
        if (!results.has(r.id)) {
          results.set(r.id, r);
        }
      }
    } catch {
      // Semantic search may not be available
    }

    // Sort by score descending
    return Array.from(results.values())
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .slice(0, topK);
  },

  async clear(containerTag: string): Promise<void> {
    const ids = containerMemories.get(containerTag) || [];
    for (const id of ids) {
      try {
        await apiCall("DELETE", `/memories/${id}`);
      } catch {
        // Ignore errors on cleanup
      }
    }
    containerMemories.delete(containerTag);
  },
};

export default memdioProvider;
