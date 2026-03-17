"""Storage manager for memp3 memories.

Architecture — everything in one SQLite database:
  - FLAC audio blobs (lossless compressed)
  - FTS5 full-text search (word-level matching)
  - sqlite-vector embeddings (semantic similarity, COSINE distance)
  - Proper indexes on all query paths

No FAISS. No separate index files. One .db file per user.
"""

import io
import json
import logging
import os
import sqlite3
import uuid

import numpy as np

from memp3.core.validators import (
    validate_content,
    validate_memory_id,
    validate_query,
    validate_tags,
)

logger = logging.getLogger(__name__)

VECTOR_DIM = 384  # BAAI/bge-small-en-v1.5 output dimension


def _signal_to_flac_blob(signal: np.ndarray, sample_rate: int) -> bytes:
    """Encode signal as FLAC bytes in memory."""
    import soundfile as sf
    import tempfile

    fp = tempfile.mktemp(suffix=".flac")
    try:
        sf.write(fp, signal, sample_rate)
        with open(fp, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(fp):
            os.remove(fp)


def _flac_blob_to_signal(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode FLAC bytes back to signal + sample_rate."""
    import soundfile as sf

    return sf.read(io.BytesIO(blob))


def _load_vector_extension(conn):
    """Load sqlite-vector extension if available."""
    try:
        import sqlite_vector
        ext_dir = os.path.join(os.path.dirname(sqlite_vector.__file__), "binaries", "vector")
        conn.enable_load_extension(True)
        conn.load_extension(ext_dir)
        return True
    except Exception as e:
        logger.debug("sqlite-vector not available: %s", e)
        return False


class StorageManager:
    """SQLite + FLAC + FTS5 + sqlite-vector — one DB, zero external files."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "index.db")

        os.makedirs(base_path, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")
        self._conn.execute("PRAGMA mmap_size=67108864")

        self._has_vector = _load_vector_extension(self._conn)
        self._embedder = None
        self._init_db()

    def _init_db(self):
        c = self._conn.cursor()

        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        table_exists = c.fetchone() is not None

        if table_exists:
            c.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in c.fetchall()]

            if "flac_blob" not in columns:
                c.execute("ALTER TABLE memories ADD COLUMN flac_blob BLOB")
                logger.info("Migrated: added flac_blob column")

            if "embedding" not in columns and self._has_vector:
                c.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
                logger.info("Migrated: added embedding column")

            if "filename" in columns:
                new_cols = "id, content, tags, encoder_version, flac_blob, created_at"
                extra = ", embedding" if "embedding" in columns or (self._has_vector and "embedding" not in columns) else ""
                c.execute(f"""
                    CREATE TABLE IF NOT EXISTS memories_new (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        tags TEXT,
                        encoder_version INTEGER DEFAULT 2,
                        flac_blob BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        {",' embedding BLOB'" if self._has_vector else ""}
                    )
                """.replace("'", ""))
                c.execute(f"""
                    INSERT OR IGNORE INTO memories_new ({new_cols})
                    SELECT {new_cols} FROM memories
                """)
                c.execute("DROP TABLE memories")
                c.execute("ALTER TABLE memories_new RENAME TO memories")
                logger.info("Migrated: rebuilt table, dropped filename column")
        else:
            c.execute(f"""
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT,
                    encoder_version INTEGER DEFAULT 2,
                    flac_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    {",' embedding BLOB'" if self._has_vector else ""}
                )
            """.replace("'", ""))

        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")

        # FTS5 for word-level search
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, content, tags, content=memories, content_rowid=rowid
            )
        """)
        c.execute("INSERT OR IGNORE INTO memories_fts(memories_fts) VALUES('rebuild')")

        # sqlite-vector init
        if self._has_vector:
            try:
                c.execute("PRAGMA table_info(memories)")
                cols = [row[1] for row in c.fetchall()]
                if "embedding" not in cols:
                    c.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
                self._conn.execute(
                    f"SELECT vector_init('memories', 'embedding', 'dimension={VECTOR_DIM},type=FLOAT32,distance=COSINE')"
                )
                logger.info("sqlite-vector initialized (dim=%d, COSINE)", VECTOR_DIM)
            except Exception as e:
                logger.warning("sqlite-vector init failed: %s", e)
                self._has_vector = False

        self._conn.commit()

    def _get_embedder(self):
        """Lazy-load FastEmbed ONNX model — no PyTorch, 284MB RAM."""
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
            logger.info("Loaded embedding model: BAAI/bge-small-en-v1.5 (ONNX)")
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        model = self._get_embedder()
        vec = list(model.embed([text]))[0]
        return vec.tolist()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def store(self, content, tags=None, encoder_version=2):
        """Store content as FLAC audio blob + optional vector embedding."""
        content = validate_content(content)
        tags = validate_tags(tags)

        if encoder_version == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder()
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder()

        signal = encoder.encode(content)
        flac_blob = _signal_to_flac_blob(signal, encoder.sample_rate)
        mem_id = str(uuid.uuid4())

        if self._has_vector and self._embedder is not None:
            embedding = self._embed(content)
            self._conn.execute(
                "INSERT INTO memories (id, content, tags, encoder_version, flac_blob, embedding) VALUES (?,?,?,?,?,vector_as_f32(?))",
                (mem_id, content, tags, encoder_version, flac_blob, json.dumps(embedding)),
            )
        else:
            self._conn.execute(
                "INSERT INTO memories (id, content, tags, encoder_version, flac_blob) VALUES (?,?,?,?,?)",
                (mem_id, content, tags, encoder_version, flac_blob),
            )

        # Sync FTS
        rowid = self._conn.execute("SELECT rowid FROM memories WHERE id=?", (mem_id,)).fetchone()[0]
        self._conn.execute(
            "INSERT INTO memories_fts(rowid, id, content, tags) VALUES (?,?,?,?)",
            (rowid, mem_id, content, tags),
        )
        self._conn.commit()
        logger.info("Stored memory %s (%d bytes text, %d bytes FLAC)", mem_id, len(content), len(flac_blob))
        return mem_id

    def retrieve(self, mem_id):
        """Retrieve and decode by PRIMARY KEY — O(1)."""
        mem_id = validate_memory_id(mem_id)

        row = self._conn.execute(
            "SELECT flac_blob, encoder_version FROM memories WHERE id=?", (mem_id,),
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        flac_blob, enc_ver = row[0], row[1] or 2

        if flac_blob is None:
            raise KeyError(f"Memory {mem_id} has no audio data")

        signal, sample_rate = _flac_blob_to_signal(flac_blob)

        if enc_ver == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder(sample_rate=sample_rate)
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder(sample_rate)

        return encoder.decode(signal)

    def search(self, query):
        """FTS5 word search — matches any word from the query."""
        query = validate_query(query)
        words = query.strip().split()
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        rows = self._conn.execute(
            "SELECT m.id, m.content, m.created_at FROM memories_fts f JOIN memories m ON f.id = m.id WHERE memories_fts MATCH ? ORDER BY rank LIMIT 100",
            (fts_query,),
        ).fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def list_all(self, limit=100, offset=0):
        """List with pagination — uses idx_memories_created."""
        rows = self._conn.execute(
            "SELECT id, content, created_at FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def delete(self, mem_id):
        """Delete by PRIMARY KEY — O(1)."""
        mem_id = validate_memory_id(mem_id)
        row = self._conn.execute("SELECT rowid FROM memories WHERE id=?", (mem_id,)).fetchone()
        if not row:
            logger.info("Memory %s already deleted or not found", mem_id)
            return False
        self._conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES('delete', ?, ?, (SELECT content FROM memories WHERE id=?), (SELECT tags FROM memories WHERE id=?))",
            (row[0], mem_id, mem_id, mem_id),
        )
        self._conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self._conn.commit()
        logger.info("Deleted memory %s", mem_id)

    def get_info(self, mem_id):
        """Get metadata by PRIMARY KEY — O(1)."""
        mem_id = validate_memory_id(mem_id)
        row = self._conn.execute(
            "SELECT id, LENGTH(content), created_at, tags, encoder_version, LENGTH(flac_blob) FROM memories WHERE id=?",
            (mem_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        return {
            "id": row[0],
            "content_length": row[1],
            "created_at": row[2],
            "tags": row[3],
            "encoder_version": row[4] or 2,
            "flac_bytes": row[5] or 0,
            "storage": "flac_blob",
        }

    def export_flac(self, mem_id, output_path=None):
        """Export a memory as a FLAC file."""
        mem_id = validate_memory_id(mem_id)
        row = self._conn.execute("SELECT flac_blob FROM memories WHERE id=?", (mem_id,)).fetchone()
        if not row or not row[0]:
            raise KeyError(f"Memory {mem_id} not found")

        if output_path is None:
            os.makedirs(os.path.join(self.base_path, "exports"), exist_ok=True)
            output_path = os.path.join(self.base_path, "exports", f"{mem_id}.flac")

        with open(output_path, "wb") as f:
            f.write(row[0])
        return output_path

    def export_wav(self, mem_id, output_path=None):
        """Export a memory as a WAV file."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)
        row = self._conn.execute("SELECT flac_blob FROM memories WHERE id=?", (mem_id,)).fetchone()
        if not row or not row[0]:
            raise KeyError(f"Memory {mem_id} not found")

        signal, sample_rate = _flac_blob_to_signal(row[0])

        if output_path is None:
            os.makedirs(os.path.join(self.base_path, "exports"), exist_ok=True)
            output_path = os.path.join(self.base_path, "exports", f"{mem_id}.wav")

        sf.write(output_path, signal, sample_rate, format="WAV", subtype="FLOAT")
        return output_path

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search using sqlite-vector (no FAISS, no sync bugs)."""
        query = validate_query(query)

        if not self._has_vector:
            raise RuntimeError("sqlite-vector extension not available")

        # Embed query
        query_vec = json.dumps(self._embed(query))

        # Backfill/re-embed: rows without embeddings OR stale embeddings from old model
        # Check if embeddings need rebuild (model switch detection)
        meta_table = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memp3_meta'"
        ).fetchone()
        current_model = "BAAI/bge-small-en-v1.5"
        needs_reembed = False

        if not meta_table:
            self._conn.execute("CREATE TABLE memp3_meta (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.execute("INSERT INTO memp3_meta VALUES ('embed_model', ?)", (current_model,))
            self._conn.commit()
            needs_reembed = True
        else:
            row = self._conn.execute("SELECT value FROM memp3_meta WHERE key='embed_model'").fetchone()
            if not row or row[0] != current_model:
                self._conn.execute("INSERT OR REPLACE INTO memp3_meta VALUES ('embed_model', ?)", (current_model,))
                self._conn.commit()
                needs_reembed = True

        if needs_reembed:
            rows_without = self._conn.execute("SELECT id, content FROM memories").fetchall()
        else:
            rows_without = self._conn.execute(
                "SELECT id, content FROM memories WHERE embedding IS NULL"
            ).fetchall()
        if rows_without:
            for row_id, row_content in rows_without:
                vec = self._embed(row_content)
                self._conn.execute(
                    "UPDATE memories SET embedding = vector_as_f32(?) WHERE id = ?",
                    (json.dumps(vec), row_id),
                )
            self._conn.commit()
            logger.info("Backfilled %d embeddings", len(rows_without))

        # Vector similarity search
        rows = self._conn.execute(
            f"""SELECT m.id, m.content, m.created_at, v.distance
                FROM vector_full_scan('memories', 'embedding', ?, ?) v
                JOIN memories m ON m.rowid = v.rowid""",
            (query_vec, top_k),
        ).fetchall()

        return [
            {"id": r[0], "content": r[1], "created_at": r[2], "score": 1.0 - r[3]}
            for r in rows
        ]

    def stats(self):
        """Aggregate stats — single query."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0), COALESCE(SUM(LENGTH(flac_blob)),0) FROM memories"
        ).fetchone()
        return {
            "total_memories": row[0],
            "total_content_bytes": row[1],
            "total_flac_bytes": row[2],
        }
