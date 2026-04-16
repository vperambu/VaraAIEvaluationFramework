"""DocumentStore: manages documents, embeddings, FAISS index and persistence.

Features:
- Add documents with id, text, metadata
- Build FAISS index (if available) using provided embed_fn
- Persist index and metadata to disk
- Fallback to linear scan when FAISS not available

This module keeps concerns separated: embedding logic is provided by the
caller via `embed_fn` to remain model-agnostic.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self, embed_fn: Callable[[str], Sequence[float]], index_path: Optional[str] = None):
        self.embed_fn = embed_fn
        self.index_path = index_path
        self._docs: List[Dict[str, Any]] = []
        self._vectors = None
        self._index = None
        try:
            import faiss  # optional
        except Exception:
            faiss = None
            logger.debug("FAISS not available; will use linear scan fallback")
        self._faiss = faiss

        # optional numpy; if unavailable we fall back to pure-python arrays
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None
            logger.debug("numpy not available; using python fallback for vector ops")
        self._np = np

    def add_documents(self, docs: Sequence[Dict[str, Any]]) -> None:
        """Docs: sequence of {'id': str, 'text': str, 'meta': {...}}"""
        for d in docs:
            assert "id" in d and "text" in d
            self._docs.append(d)

    def _compute_vectors(self) -> None:
        vectors = []
        for d in self._docs:
            try:
                v = self.embed_fn(d["text"])
            except Exception as e:
                logger.exception("embed_fn failed for doc %s: %s", d.get("id"), e)
                v = [0.0]
            vectors.append(list(v))

        if self._np is not None:
            self._vectors = self._np.array(vectors, dtype="float32")
        else:
            # keep as list-of-lists for fallback
            self._vectors = vectors

    def build_index(self) -> None:
        """Build or rebuild the FAISS index (or prepare fallback vectors)."""
        if not self._docs:
            logger.warning("No documents to index")
            return
        self._compute_vectors()

        if self._faiss is not None and self._np is not None:
            dim = self._vectors.shape[1]
            idx = self._faiss.IndexFlatL2(dim)
            idx.add(self._vectors)
            self._index = idx
            logger.info("Built FAISS index with %d vectors (dim=%d)", len(self._docs), dim)
            # Optionally persist
            if self.index_path:
                try:
                    self._faiss.write_index(self._index, self.index_path)
                    # save metadata
                    meta_path = self.index_path + ".meta"
                    with open(meta_path, "wb") as f:
                        pickle.dump(self._docs, f)
                    logger.info("Persisted FAISS index and metadata to %s", self.index_path)
                except Exception as e:
                    logger.exception("Failed to persist FAISS index: %s", e)
        else:
            logger.info("FAISS not available or numpy missing; prepared vectors for linear-scan fallback")

    def save(self, path: str) -> None:
        """Persist vectors and docs without relying on FAISS write (fallback)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(self._docs, f, ensure_ascii=False, indent=2)
            if self._vectors is None:
                self._compute_vectors()
            if self._np is not None:
                self._np.save(path + ".npy", self._vectors)
            else:
                # save python list as pickle
                with open(path + ".npy.pickle", "wb") as vf:
                    pickle.dump(self._vectors, vf)
            logger.info("Saved DocumentStore metadata and vectors to %s.*", path)
        except Exception as e:
            logger.exception("Failed to save DocumentStore: %s", e)

    def load(self, path: str) -> None:
        """Load persisted metadata and vectors (non-FAISS fallback)."""
        try:
            with open(path + ".meta.json", "r", encoding="utf-8") as f:
                self._docs = json.load(f)
            if self._np is not None:
                self._vectors = self._np.load(path + ".npy")
            else:
                with open(path + ".npy.pickle", "rb") as vf:
                    self._vectors = pickle.load(vf)
            logger.info("Loaded DocumentStore from %s.*", path)
        except Exception as e:
            logger.exception("Failed to load DocumentStore: %s", e)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k documents (full doc dicts)."""
        try:
            qv = self.embed_fn(query)
        except Exception as e:
            logger.exception("embed_fn failed for query: %s", e)
            return []

        # FAISS path
        if self._faiss is not None and self._index is not None and self._np is not None:
            q = self._np.array([qv], dtype="float32")
            D, I = self._index.search(q, k)
            ids = I[0].tolist()
            docs = [self._docs[i] for i in ids if i < len(self._docs)]
            return docs

        # fallback linear scan (L2)
        try:
            if self._vectors is None:
                self._compute_vectors()
            # numpy path
            if self._np is not None:
                q = self._np.array(qv, dtype="float32")
                diffs = self._vectors - q
                dists = (diffs ** 2).sum(axis=1)
                idxs = dists.argsort()[:k]
                return [self._docs[int(i)] for i in idxs.tolist()]
            # pure-python fallback
            q = [float(x) for x in qv]
            def l2(a, b):
                return sum((ai - bi) ** 2 for ai, bi in zip(a, b))
            dists = [l2(v, q) for v in self._vectors]
            idxs = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
            return [self._docs[i] for i in idxs]
        except Exception as e:
            logger.exception("Linear-scan retrieval failed: %s", e)
            # deterministic fallback: return first k docs
            return self._docs[:k]
