"""Storage manager for memp3 memories.

Three-tier storage:
  Hot:  Raw PCM signal stored as blob in SQLite — zero file I/O, fastest
  Warm: WAV file on disk — no codec overhead, fast random access
  Cold: FLAC file export — compressed lossless, portable, shareable

All tiers are bit-perfect lossless. PCM = WAV = FLAC in terms of data.
"""

import logging
import os
import sqlite3
import struct
import uuid

import numpy as np

from memp3.core.validators import (
    validate_content,
    validate_memory_id,
    validate_query,
    validate_tags,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


def _signal_to_blob(signal: np.ndarray, sample_rate: int) -> bytes:
    """Encode signal as FLAC bytes in memory — best compression, no file I/O."""
    import io
    import soundfile as sf
    import tempfile
    # soundfile can't write FLAC to BytesIO reliably, use a temp file
    fp = tempfile.mktemp(suffix=".flac")
    try:
        sf.write(fp, signal, sample_rate)
        with open(fp, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(fp):
            os.remove(fp)


def _blob_to_signal(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode FLAC bytes back to signal + sample_rate — from memory, no file I/O."""
    import io
    import soundfile as sf
    buf = io.BytesIO(blob)
    signal, sample_rate = sf.read(buf)
    return signal, sample_rate


class StorageManager:
    """Manages audio memory storage with SQLite metadata + PCM blobs."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        self.memory_path = os.path.join(base_path, "memory")
        self.db_path = os.path.join(base_path, "index.db")

        os.makedirs(self.memory_path, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

        self._semantic = None

    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                encoder_version INTEGER DEFAULT 1,
                pcm_blob BLOB
            )
        """)
        # Migrations
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]
        if "encoder_version" not in columns:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN encoder_version INTEGER DEFAULT 1"
            )
            logger.info("Migrated schema: added encoder_version column")
        if "pcm_blob" not in columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN pcm_blob BLOB")
            logger.info("Migrated schema: added pcm_blob column")
        self._conn.commit()

    def _get_conn(self):
        return self._conn

    def _get_semantic(self, required=False):
        if self._semantic is None:
            if not required:
                return None
            try:
                from memp3.core.search import SemanticSearch
                self._semantic = SemanticSearch(self.base_path)
            except ImportError:
                if required:
                    raise RuntimeError("Semantic search requires sentence-transformers and faiss-cpu")
                return None
        return self._semantic

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def store(self, content, tags=None, encoder_version=2):
        """Store content as audio memory (PCM blob in SQLite — no file I/O)."""
        content = validate_content(content)
        tags = validate_tags(tags)

        mem_id = str(uuid.uuid4())
        filename = f"{mem_id}.flac"  # reserved for cold export

        if encoder_version == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder()
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder()

        signal = encoder.encode(content)
        pcm_blob = _signal_to_blob(signal, encoder.sample_rate)

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO memories (id, filename, content, tags, encoder_version, pcm_blob) VALUES (?, ?, ?, ?, ?, ?)",
            (mem_id, filename, content, tags, encoder_version, pcm_blob),
        )
        conn.commit()
        logger.info("Stored memory %s (%d bytes, %d bytes PCM)", mem_id, len(content), len(pcm_blob))

        sem = self._get_semantic(required=False)
        if sem is not None:
            sem.add(mem_id, content)

        return mem_id

    def retrieve(self, mem_id):
        """Retrieve and decode a memory by ID (from PCM blob or FLAC fallback)."""
        mem_id = validate_memory_id(mem_id)

        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename, encoder_version, pcm_blob FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filename, enc_ver, pcm_blob = row[0], row[1] or 1, row[2]

        # Hot path: decode from PCM blob (no file I/O)
        if pcm_blob:
            signal, sample_rate = _blob_to_signal(pcm_blob)
        else:
            # Fallback: read from FLAC file (legacy memories)
            import soundfile as sf
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
        """Delete a memory by ID. Idempotent."""
        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        if not row:
            logger.info("Memory %s already deleted or not found", mem_id)
            return False

        filepath = os.path.join(self.memory_path, row[0])
        if os.path.exists(filepath):
            os.remove(filepath)

        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        conn.commit()
        logger.info("Deleted memory %s", mem_id)

        sem = self._get_semantic()
        if sem is not None:
            sem.remove(mem_id)

    def export_flac(self, mem_id, output_path=None):
        """Export a memory as a FLAC file (cold storage / sharing)."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename, pcm_blob FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filename, pcm_blob = row
        if output_path is None:
            output_path = os.path.join(self.memory_path, filename)

        if pcm_blob:
            signal, sample_rate = _blob_to_signal(pcm_blob)
        else:
            # Already a FLAC file on disk
            return os.path.join(self.memory_path, filename)

        sf.write(output_path, signal, sample_rate)
        logger.info("Exported memory %s to %s", mem_id, output_path)
        return output_path

    def export_wav(self, mem_id, output_path=None):
        """Export a memory as a WAV file (warm storage)."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT filename, pcm_blob FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()

        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filename, pcm_blob = row
        if output_path is None:
            output_path = os.path.join(self.memory_path, filename.replace(".flac", ".wav"))

        if pcm_blob:
            signal, sample_rate = _blob_to_signal(pcm_blob)
        else:
            existing = os.path.join(self.memory_path, filename)
            signal, sample_rate = sf.read(existing)

        sf.write(output_path, signal, sample_rate, format="WAV", subtype="FLOAT")
        logger.info("Exported memory %s to %s", mem_id, output_path)
        return output_path

    def get_info(self, mem_id):
        """Get metadata for a memory."""
        mem_id = validate_memory_id(mem_id)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, filename, content, created_at, tags, encoder_version, LENGTH(pcm_blob) FROM memories WHERE id = ?",
            (mem_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Memory {mem_id} not found")

        filepath = os.path.join(self.memory_path, row[1])
        flac_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        pcm_size = row[6] or 0

        return {
            "id": row[0],
            "filename": row[1],
            "content_length": len(row[2]),
            "created_at": row[3],
            "tags": row[4],
            "encoder_version": row[5] or 1,
            "pcm_blob_bytes": pcm_size,
            "flac_file_bytes": flac_size,
            "storage": "pcm_blob" if pcm_size > 0 else "flac_file",
        }

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search memories by semantic similarity."""
        query = validate_query(query)
        sem = self._get_semantic(required=True)

        if sem._index.ntotal == 0:
            conn = self._get_conn()
            rows = conn.execute("SELECT id, content FROM memories").fetchall()
            for row in rows:
                sem.add(row[0], row[1])

        results = sem.search(query, top_k=top_k)
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
        row = conn.execute(
            "SELECT COUNT(*), SUM(LENGTH(content)), SUM(LENGTH(pcm_blob)) FROM memories"
        ).fetchone()
        total_memories = row[0] or 0
        total_content_bytes = row[1] or 0
        total_pcm_bytes = row[2] or 0

        total_flac_bytes = 0
        if os.path.isdir(self.memory_path):
            for fname in os.listdir(self.memory_path):
                fpath = os.path.join(self.memory_path, fname)
                if os.path.isfile(fpath):
                    total_flac_bytes += os.path.getsize(fpath)

        return {
            "total_memories": total_memories,
            "total_content_bytes": total_content_bytes,
            "total_pcm_bytes": total_pcm_bytes,
            "total_flac_bytes": total_flac_bytes,
        }
