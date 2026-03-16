"""Storage manager for memp3 memories."""

import logging
import os
import sqlite3
import uuid
from datetime import datetime

from memp3.core.validators import (
    validate_content,
    validate_memory_id,
    validate_query,
    validate_tags,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class StorageManager:
    """Manages audio memory storage with SQLite metadata."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        self.memory_path = os.path.join(base_path, "memory")
        self.db_path = os.path.join(base_path, "index.db")

        os.makedirs(self.memory_path, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

        self._semantic = None  # lazy-loaded

    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                encoder_version INTEGER DEFAULT 1
            )
        """)
        # Migration: add encoder_version if missing
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]
        if "encoder_version" not in columns:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN encoder_version INTEGER DEFAULT 1"
            )
            logger.info("Migrated schema: added encoder_version column")
        self._conn.commit()

    def _get_conn(self):
        return self._conn

    def _get_semantic(self):
        if self._semantic is None:
            try:
                from memp3.core.search import SemanticSearch
                self._semantic = SemanticSearch(self.base_path)
            except ImportError:
                logger.debug("sentence-transformers/faiss not installed; semantic search disabled")
                return None
        return self._semantic

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def store(self, content, tags=None, encoder_version=2):
        """Store content as audio memory."""
        import soundfile as sf

        content = validate_content(content)
        tags = validate_tags(tags)

        mem_id = str(uuid.uuid4())
        filename = f"{mem_id}.flac"
        filepath = os.path.join(self.memory_path, filename)

        if encoder_version == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder()
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder()

        signal = encoder.encode(content)
        sf.write(filepath, signal, encoder.sample_rate)

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO memories (id, filename, content, tags, encoder_version) VALUES (?, ?, ?, ?, ?)",
            (mem_id, filename, content, tags, encoder_version),
        )
        conn.commit()
        logger.info("Stored memory %s (%d bytes)", mem_id, len(content))

        sem = self._get_semantic()
        if sem is not None:
            sem.add(mem_id, content)

        return mem_id

    def retrieve(self, mem_id):
        """Retrieve and decode a memory by ID."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)

        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename, encoder_version FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filename, enc_ver = row[0], row[1] or 1
        filepath = os.path.join(self.memory_path, filename)

        signal, sample_rate = sf.read(filepath)

        if enc_ver == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder(sample_rate=sample_rate)
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder(sample_rate)

        return encoder.decode(signal)

    def search(self, query):
        """Search memories by content substring."""
        query = validate_query(query)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories WHERE content LIKE ? ORDER BY created_at DESC",
            (f"%{query}%",),
        ).fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def list_all(self):
        """List all memories."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories ORDER BY created_at DESC"
        ).fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def delete(self, mem_id):
        """Delete a memory by ID."""
        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filepath = os.path.join(self.memory_path, row[0])
        if os.path.exists(filepath):
            os.remove(filepath)

        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        conn.commit()
        logger.info("Deleted memory %s", mem_id)

        sem = self._get_semantic()
        if sem is not None:
            sem.remove(mem_id)

    def get_info(self, mem_id):
        """Get metadata for a memory."""
        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, filename, content, created_at, tags, encoder_version FROM memories WHERE id = ?",
            (mem_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filepath = os.path.join(self.memory_path, row[1])
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        return {
            "id": row[0],
            "filename": row[1],
            "content_length": len(row[2]),
            "created_at": row[3],
            "tags": row[4],
            "encoder_version": row[5] or 1,
            "file_size_bytes": file_size,
        }

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search memories by semantic similarity."""
        query = validate_query(query)
        sem = self._get_semantic()
        if sem is None:
            raise RuntimeError("Semantic search requires sentence-transformers and faiss-cpu")
        results = sem.search(query, top_k=top_k)
        # Enrich with content from DB
        conn = self._get_conn()
        enriched = []
        for r in results:
            row = conn.execute(
                "SELECT content, created_at FROM memories WHERE id = ?", (r["id"],)
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
        """Get overall statistics."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*), SUM(LENGTH(content)) FROM memories").fetchone()
        total_memories = row[0] or 0
        total_content_bytes = row[1] or 0

        total_audio_bytes = 0
        if total_memories > 0:
            for fname in os.listdir(self.memory_path):
                fpath = os.path.join(self.memory_path, fname)
                if os.path.isfile(fpath):
                    total_audio_bytes += os.path.getsize(fpath)

        return {
            "total_memories": total_memories,
            "total_content_bytes": total_content_bytes,
            "total_audio_bytes": total_audio_bytes,
        }
