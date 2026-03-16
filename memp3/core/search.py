"""Semantic search for memp3 using sentence-transformers and FAISS."""

import logging
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticSearch:
    """FAISS-backed semantic search over memory content."""

    def __init__(self, base_path: str, model_name: str = DEFAULT_MODEL):
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim)  # inner product (cosine after normalize)
        self._ids: list[str] = []
        self._index_path = os.path.join(base_path, "faiss.index")
        self._ids_path = os.path.join(base_path, "faiss_ids.npy")
        self._load()

    def _load(self):
        if os.path.exists(self._index_path) and os.path.exists(self._ids_path):
            self._index = faiss.read_index(self._index_path)
            self._ids = list(np.load(self._ids_path, allow_pickle=True))
            logger.info("Loaded FAISS index with %d entries", len(self._ids))

    def _save(self):
        faiss.write_index(self._index, self._index_path)
        np.save(self._ids_path, np.array(self._ids, dtype=object))

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def add(self, mem_id: str, content: str):
        """Add a memory to the semantic index."""
        vec = self._embed([content])
        self._index.add(vec)
        self._ids.append(mem_id)
        self._save()

    def remove(self, mem_id: str):
        """Remove a memory from the index (rebuilds index)."""
        if mem_id not in self._ids:
            return
        idx = self._ids.index(mem_id)
        self._ids.pop(idx)
        # Rebuild index without the removed vector
        if self._index.ntotal > 0:
            all_vecs = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._dim
            ).reshape(self._index.ntotal, self._dim).copy()
            new_vecs = np.delete(all_vecs, idx, axis=0)
            self._index.reset()
            if len(new_vecs) > 0:
                self._index.add(new_vecs)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for semantically similar memories.

        Returns list of {"id": str, "score": float} sorted by relevance.
        """
        if self._index.ntotal == 0:
            return []
        vec = self._embed([query])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({"id": self._ids[idx], "score": float(score)})
        return results
