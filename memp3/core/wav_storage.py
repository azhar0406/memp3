"""WAV append storage engine — per-user isolated files, mmap for fast reads.

Architecture:
  memory.wav   — append-only WAV file (raw PCM float32)
  index.json   — memory metadata (offsets, lengths, content, tags)

The WAV file is memory-mapped for retrieval, so reads come from RAM
after the first access — no disk I/O penalty.
"""

import json
import logging
import mmap
import os
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


class WavStreamStorage:
    """Per-user WAV append storage with mmap reads."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.wav_path = os.path.join(base_path, "memory.wav")
        self.index_path = os.path.join(base_path, "index.json")
        self.sample_rate = 48000

        self._index = self._load_index()
        self._current_offset = sum(m["length"] for m in self._index)
        self._ensure_wav()
        self._data_offset = self._find_data_offset()
        self._mmap = None
        self._mmap_file = None
        self._semantic = None

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                return json.load(f)
        return []

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, separators=(",", ":"))

    def _ensure_wav(self):
        if not os.path.exists(self.wav_path):
            import soundfile as sf
            sf.write(
                self.wav_path,
                np.array([], dtype=np.float64),
                self.sample_rate,
                format="WAV",
                subtype="FLOAT",
            )

    def _find_data_offset(self):
        """Find the byte offset where audio data starts in the WAV file."""
        with open(self.wav_path, "rb") as f:
            header = f.read(200)
            idx = header.find(b"data")
            if idx >= 0:
                return idx + 8  # skip 'data' + 4-byte size
        return 80  # fallback for FLOAT WAV

    def _invalidate_mmap(self):
        """Close mmap so it gets re-opened on next read with new data."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._mmap_file is not None:
            self._mmap_file.close()
            self._mmap_file = None

    def _get_mmap(self):
        """Get or create memory-mapped view of the WAV file."""
        if self._mmap is None:
            file_size = os.path.getsize(self.wav_path)
            if file_size <= self._data_offset:
                return None
            self._mmap_file = open(self.wav_path, "rb")
            self._mmap = mmap.mmap(
                self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        return self._mmap

    def _append_wav(self, signal):
        """Append PCM samples to WAV file by updating header + appending data."""
        f32 = signal.astype(np.float32)
        raw = f32.tobytes()

        with open(self.wav_path, "r+b") as f:
            # Read current sizes
            f.seek(4)
            old_riff = struct.unpack("<I", f.read(4))[0]
            f.seek(self._data_offset - 4)
            old_data = struct.unpack("<I", f.read(4))[0]

            # Append audio data at end
            f.seek(0, 2)
            f.write(raw)

            # Update RIFF chunk size
            f.seek(4)
            f.write(struct.pack("<I", old_riff + len(raw)))

            # Update data chunk size
            f.seek(self._data_offset - 4)
            f.write(struct.pack("<I", old_data + len(raw)))

        # Invalidate mmap so next read picks up new data
        self._invalidate_mmap()

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
        self._invalidate_mmap()

    def store(self, content, tags=None, encoder_version=2):
        """Store content by appending encoded audio to the WAV stream."""
        content = validate_content(content)
        tags = validate_tags(tags)

        if encoder_version == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder()
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder()

        signal = encoder.encode(content)
        mem_id = str(uuid.uuid4())

        self._append_wav(signal)

        self._index.append({
            "id": mem_id,
            "offset": self._current_offset,
            "length": len(signal),
            "content": content,
            "tags": tags,
            "encoder_version": encoder_version,
        })
        self._current_offset += len(signal)
        self._save_index()

        logger.info("Stored memory %s (%d bytes)", mem_id, len(content))

        sem = self._get_semantic(required=False)
        if sem is not None:
            sem.add(mem_id, content)

        return mem_id

    def retrieve(self, mem_id):
        """Retrieve a memory by seeking into the mmap'd WAV file."""
        mem_id = validate_memory_id(mem_id)

        entry = next((m for m in self._index if m["id"] == mem_id), None)
        if not entry:
            raise KeyError(f"Memory {mem_id} not found")

        mm = self._get_mmap()
        if mm is None:
            raise KeyError(f"Memory {mem_id} not found (empty WAV)")

        byte_offset = self._data_offset + entry["offset"] * 4  # float32 = 4 bytes
        byte_length = entry["length"] * 4

        raw = mm[byte_offset : byte_offset + byte_length]
        signal = np.frombuffer(raw, dtype=np.float32).astype(np.float64)

        enc_ver = entry.get("encoder_version", 2)
        if enc_ver == 2:
            from memp3.core.encoder import BinaryEncoder
            encoder = BinaryEncoder(sample_rate=self.sample_rate)
        else:
            from memp3.core.encoder import SimpleEncoder
            encoder = SimpleEncoder(self.sample_rate)

        return encoder.decode(signal)

    def search(self, query):
        """Search memories by content substring."""
        query = validate_query(query)
        q_lower = query.lower()
        return [
            {"id": m["id"], "content": m["content"], "created_at": m.get("created_at", "")}
            for m in self._index
            if q_lower in m["content"].lower()
        ]

    def list_all(self):
        """List all memories."""
        return [
            {"id": m["id"], "content": m["content"], "created_at": m.get("created_at", "")}
            for m in self._index
        ]

    def delete(self, mem_id):
        """Delete a memory from the index (audio data stays in WAV, reclaimed on compact)."""
        mem_id = validate_memory_id(mem_id)
        before = len(self._index)
        self._index = [m for m in self._index if m["id"] != mem_id]
        if len(self._index) == before:
            logger.info("Memory %s already deleted or not found", mem_id)
            return False
        self._save_index()
        logger.info("Deleted memory %s", mem_id)

        sem = self._get_semantic()
        if sem is not None:
            sem.remove(mem_id)

    def get_info(self, mem_id):
        """Get metadata for a memory."""
        mem_id = validate_memory_id(mem_id)
        entry = next((m for m in self._index if m["id"] == mem_id), None)
        if not entry:
            raise KeyError(f"Memory {mem_id} not found")

        return {
            "id": entry["id"],
            "content_length": len(entry["content"]),
            "created_at": entry.get("created_at", ""),
            "tags": entry.get("tags"),
            "encoder_version": entry.get("encoder_version", 2),
            "pcm_blob_bytes": entry["length"] * 4,
            "flac_file_bytes": 0,
            "storage": "wav_stream",
        }

    def stats(self):
        """Get overall statistics."""
        wav_size = os.path.getsize(self.wav_path) if os.path.exists(self.wav_path) else 0
        idx_size = os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0
        total_content = sum(len(m["content"]) for m in self._index)

        return {
            "total_memories": len(self._index),
            "total_content_bytes": total_content,
            "total_pcm_bytes": wav_size,
            "total_flac_bytes": 0,
            "wav_file_bytes": wav_size,
            "index_file_bytes": idx_size,
        }

    def export_flac(self, mem_id=None, output_path=None):
        """Export a single memory or entire stream as FLAC."""
        import soundfile as sf

        if mem_id is not None:
            mem_id = validate_memory_id(mem_id)
            entry = next((m for m in self._index if m["id"] == mem_id), None)
            if not entry:
                raise KeyError(f"Memory {mem_id} not found")
            mm = self._get_mmap()
            byte_offset = self._data_offset + entry["offset"] * 4
            raw = mm[byte_offset : byte_offset + entry["length"] * 4]
            signal = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
            if output_path is None:
                output_path = os.path.join(self.base_path, f"{mem_id}.flac")
            sf.write(output_path, signal, self.sample_rate)
        else:
            if output_path is None:
                output_path = os.path.join(self.base_path, "memory.flac")
            data, sr = sf.read(self.wav_path)
            sf.write(output_path, data, sr)

        logger.info("Exported to %s", output_path)
        return output_path

    def export_wav(self, mem_id, output_path=None):
        """Export a single memory as WAV."""
        import soundfile as sf

        mem_id = validate_memory_id(mem_id)
        entry = next((m for m in self._index if m["id"] == mem_id), None)
        if not entry:
            raise KeyError(f"Memory {mem_id} not found")

        mm = self._get_mmap()
        byte_offset = self._data_offset + entry["offset"] * 4
        raw = mm[byte_offset : byte_offset + entry["length"] * 4]
        signal = np.frombuffer(raw, dtype=np.float32).astype(np.float64)

        if output_path is None:
            output_path = os.path.join(self.base_path, f"{mem_id}.wav")
        sf.write(output_path, signal, self.sample_rate, format="WAV", subtype="FLOAT")
        return output_path

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search memories by semantic similarity."""
        query = validate_query(query)
        sem = self._get_semantic(required=True)

        if sem._index.ntotal == 0:
            for m in self._index:
                sem.add(m["id"], m["content"])

        results = sem.search(query, top_k=top_k)
        enriched = []
        for r in results:
            entry = next((m for m in self._index if m["id"] == r["id"]), None)
            if entry:
                enriched.append({
                    "id": r["id"],
                    "content": entry["content"],
                    "created_at": entry.get("created_at", ""),
                    "score": r["score"],
                })
        return enriched

    def compact(self):
        """Rewrite WAV file removing deleted memories, reclaiming space."""
        import soundfile as sf

        if not self._index:
            # All deleted, reset
            sf.write(self.wav_path, np.array([], dtype=np.float64), self.sample_rate, format="WAV", subtype="FLOAT")
            self._current_offset = 0
            self._invalidate_mmap()
            self._save_index()
            return

        self._invalidate_mmap()
        mm_file = open(self.wav_path, "rb")
        mm = mmap.mmap(mm_file.fileno(), 0, access=mmap.ACCESS_READ)

        segments = []
        new_offset = 0
        for entry in self._index:
            byte_off = self._data_offset + entry["offset"] * 4
            raw = mm[byte_off : byte_off + entry["length"] * 4]
            segments.append(np.frombuffer(raw, dtype=np.float32))
            entry["offset"] = new_offset
            new_offset += entry["length"]

        mm.close()
        mm_file.close()

        combined = np.concatenate(segments).astype(np.float64)
        sf.write(self.wav_path, combined, self.sample_rate, format="WAV", subtype="FLOAT")

        self._current_offset = new_offset
        self._data_offset = self._find_data_offset()
        self._save_index()
        logger.info("Compacted WAV: %d memories", len(self._index))
