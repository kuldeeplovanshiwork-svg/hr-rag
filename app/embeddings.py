"""Optional semantic search: sentence embeddings from all-MiniLM-L6-v2 (open weights, ~90 MB).

Embeddings are used ONLY to add a similarity term to the BM25 score inside retrieval.search.
They are computed on sanitised passages of documents the caller may already read, by code.
Nothing in a document can set or change a score.

If sentence-transformers is not installed or the model cannot be loaded, the app logs a
warning and falls back to keyword-only (BM25) search.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol

log = logging.getLogger("hr_rag")


class Embedder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return one L2-normalised vector per text."""
        ...


class MiniLMEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._cache: Dict[str, List[float]] = {}  # text -> vector (texts are already-permitted passages)

    def encode(self, texts: List[str]) -> List[List[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = [float(x) for x in v]
        return [self._cache[t] for t in texts]


def load_embedder(model_name: str) -> Optional[Embedder]:
    """Try to load the embedding model; return None (keyword-only search) on any failure."""
    try:
        return MiniLMEmbedder(model_name)
    except Exception as exc:  # missing package, no internet on first run, out of memory...
        log.warning("Embeddings disabled, using keyword search only: %s", exc)
        return None
