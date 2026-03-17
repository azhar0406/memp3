"""Storage manager for memp3 memories.

Architecture:
  SQLite database with FLAC audio blobs — no separate files.
  Properly indexed for O(1) ID lookup, fast substring search.
  All audio stored as FLAC (lossless compressed) inside the DB.
"""

import io
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


class StorageManager:
    """SQLite + FLAC blob storage with proper indexing."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "index.db")

        os.makedirs(base_path, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8MB page cache
        self._conn.execute("PRAGMA mmap_size=67108864")  # 64MB mmap
        self._init_db()

        self._semantic = None

    def _init_db(self):
        c = self._conn.cursor()

        # Check if table exists and needs migration
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        table_exists = c.fetchone() is not None

        if table_exists:
            c.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in c.fetchall()]

            if "flac_blob" not in columns:
                c.execute("ALTER TABLE memories ADD COLUMN flac_blob BLOB")
                logger.info("Migrated schema: added flac_blob column")

            # Rebuild table to drop old NOT NULL constraints on filename
            if "filename" in columns:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS memories_new (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        tags TEXT,
                        encoder_version INTEGER DEFAULT 2,
                        flac_blob BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                c.execute("""
                    INSERT OR IGNORE INTO memories_new (id, content, tags, encoder_version, flac_blob, created_at)
                    SELECT id, content, tags, encoder_version, flac_blob, created_at FROM memories
                """)
                c.execute("DROP TABLE memories")
                c.execute("ALTER TABLE memories_new RENAME TO memories")
                logger.info("Migrated schema: rebuilt table, dropped filename column")
        else:
            c.execute("""
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT,
                    encoder_version INTEGER DEFAULT 2,
                    flac_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")

        # FTS5 virtual table for proper word-level search
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, content, tags, content=memories, content_rowid=rowid
            )
        """)
        # Sync FTS with existing data (idempotent)
        c.execute("INSERT OR IGNORE INTO memories_fts(memories_fts) VALUES('rebuild')")
        self._conn.commit()

    def _get_semantic(self, required=False):
        if self._semantic is None:
            if not required:
                return None
            try:
                from memp3.core.search import SemanticSearch
                self._semantic = SemanticSearch(self.base_path)
            except ImportError:
                if required:
                    raise RuntimeError(
                        "Semantic search requires sentence-transformers and faiss-cpu"
                    )
                return None
        return self._semantic

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def store(self, content, tags=None, encoder_version=2):
        """Store content as FLAC audio blob in SQLite."""
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

        self._conn.execute(
            "INSERT INTO memories (id, content, tags, encoder_version, flac_blob) VALUES (?,?,?,?,?)",
            (mem_id, content, tags, encoder_version, flac_blob),
        )
        # Sync FTS index
        rowid = self._conn.execute("SELECT rowid FROM memories WHERE id=?", (mem_id,)).fetchone()[0]
        self._conn.execute(
            "INSERT INTO memories_fts(rowid, id, content, tags) VALUES (?,?,?,?)",
            (rowid, mem_id, content, tags),
        )
        self._conn.commit()
        logger.info("Stored memory %s (%d bytes text, %d bytes FLAC)", mem_id, len(content), len(flac_blob))

        sem = self._get_semantic(required=False)
        if sem is not None:
            sem.add(mem_id, content)

        return mem_id

    def retrieve(self, mem_id):
        """Retrieve and decode a memory by ID. Uses PRIMARY KEY index — O(1)."""
        mem_id = validate_memory_id(mem_id)

        row = self._conn.execute(
            "SELECT flac_blob, encoder_version FROM memories WHERE id=?",
            (mem_id,),
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        flac_blob, enc_ver = row[0], row[1] or 2

        # Fallback for legacy rows without flac_blob
        if flac_blob is None:
            # Try old file-based approach
            legacy_row = self._conn.execute(
                "SELECT filename FROM memories WHERE id=?", (mem_id,)
            ).fetchone()
            if legacy_row and legacy_row[0]:
                import soundfile as sf
                filepath = os.path.join(self.base_path, "memory", legacy_row[0])
                signal, sample_rate = sf.read(filepath)
                if enc_ver == 2:
                    from memp3.core.encoder import BinaryEncoder
                    return BinaryEncoder(sample_rate=sample_rate).decode(signal)
                else:
                    from memp3.core.encoder import SimpleEncoder
                    return SimpleEncoder(sample_rate).decode(signal)
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
        """Search using FTS5 word matching. Finds any word from the query."""
        query = validate_query(query)
        # Split query into words, match ANY word (OR logic)
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
        """List memories with pagination. Uses idx_memories_created index."""
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
        # Remove from FTS first (needs rowid)
        self._conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES('delete', ?, ?, (SELECT content FROM memories WHERE id=?), (SELECT tags FROM memories WHERE id=?))",
            (row[0], mem_id, mem_id, mem_id),
        )
        self._conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self._conn.commit()
        logger.info("Deleted memory %s", mem_id)

        sem = self._get_semantic()
        if sem is not None:
            sem.remove(mem_id)

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
        row = self._conn.execute(
            "SELECT flac_blob FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        if not row or not row[0]:
            raise KeyError(f"Memory {mem_id} not found")

        if output_path is None:
            os.makedirs(os.path.join(self.base_path, "exports"), exist_ok=True)
            output_path = os.path.join(self.base_path, "exports", f"{mem_id}.flac")

        with open(output_path, "wb") as f:
            f.write(row[0])
        logger.info("Exported %s to %s", mem_id, output_path)
        return output_path

    def export_wav(self, mem_id, output_path=None):
        """Export a memory as a WAV file."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)
        row = self._conn.execute(
            "SELECT flac_blob FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        if not row or not row[0]:
            raise KeyError(f"Memory {mem_id} not found")

        signal, sample_rate = _flac_blob_to_signal(row[0])

        if output_path is None:
            os.makedirs(os.path.join(self.base_path, "exports"), exist_ok=True)
            output_path = os.path.join(self.base_path, "exports", f"{mem_id}.wav")

        sf.write(output_path, signal, sample_rate, format="WAV", subtype="FLOAT")
        return output_path

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic similarity search."""
        query = validate_query(query)
        sem = self._get_semantic(required=True)

        # Sync FAISS index with DB — rebuild if counts don't match
        db_count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if sem._index.ntotal != db_count:
            sem._index.reset()
            sem._ids.clear()
            rows = self._conn.execute("SELECT id, content FROM memories").fetchall()
            for row in rows:
                sem.add(row[0], row[1])
            logger.info("Rebuilt FAISS index: %d entries", db_count)

        results = sem.search(query, top_k=top_k)
        enriched = []
        for r in results:
            row = self._conn.execute(
                "SELECT content, created_at FROM memories WHERE id=?", (r["id"],)
            ).fetchone()
            if row:
                enriched.append({
                    "id": r["id"],
                    "content": row[0],
                    "created_at": row[1],
                    "score": r["score"],
                })
        return enriched

    def stats(self):
        """Aggregate stats — single query, no loops."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0), COALESCE(SUM(LENGTH(flac_blob)),0) FROM memories"
        ).fetchone()
        return {
            "total_memories": row[0],
            "total_content_bytes": row[1],
            "total_flac_bytes": row[2],
        }
