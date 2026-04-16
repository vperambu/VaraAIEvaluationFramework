import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


class FaissUnavailable(Exception):
    pass


class FaissRetriever:
    """A thin FAISS wrapper that can map indices to documents using a DocumentStore.

    If `docstore` is provided it will return full document dicts from the store.
    Otherwise it returns placeholder ids (same behavior as before).
    """

    def __init__(
        self,
        embed_fn,
        index: Optional[object] = None,
        index_path: Optional[str] = None,
        docstore: Optional[Any] = None,
    ):
        self.embed_fn = embed_fn
        self.index = index
        self.index_path = index_path
        self.docstore = docstore
        try:
            import faiss  # optional dependency
        except Exception:
            logger.warning(
                "FAISS not available; FaissRetriever will fallback to CPU scan."
            )
            self._faiss = None
        else:
            self._faiss = faiss

    def build_index(self, vectors: Sequence[Sequence[float]]):
        if self._faiss is None:
            raise FaissUnavailable("FAISS not installed")
        dim = len(vectors[0]) if vectors else 0
        idx = self._faiss.IndexFlatL2(dim)
        import numpy as np

        idx.add(np.array(vectors, dtype="float32"))
        self.index = idx
        return idx

    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """Return top-k document dicts (if docstore available) or ids/placeholder strings.

        Uses `embed_fn` to vectorize the query. Falls back to a deterministic
        placeholder list when retrieval fails.
        """
        try:
            vec = self.embed_fn(query)
        except Exception as e:
            logger.exception("embed_fn failed during retrieve: %s", e)
            return []

        # Use FAISS index if available
        if self._faiss and self.index is not None:
            try:
                import numpy as np

                q = np.array([vec], dtype="float32")
                D, I = self.index.search(q, k)
                indices = I[0].tolist()
                # If a docstore is supplied, map indices -> docs
                if self.docstore is not None:
                    docs = []
                    for i in indices:
                        try:
                            # docstore should provide access by integer index or id mapping
                            docs.append(self.docstore._docs[int(i)])
                        except Exception:
                            logger.debug("Failed to map index %s to doc; skipping", i)
                    return docs
                # otherwise return id strings
                return [f"doc-{i}" for i in indices]
            except Exception as e:
                logger.exception("FAISS search failed: %s", e)

        # Fallback behavior: use docstore linear-retrieval if available
        if self.docstore is not None:
            try:
                return self.docstore.retrieve(query, k=k)
            except Exception as e:
                logger.exception("Docstore retrieval failed: %s", e)

        # Deterministic placeholder fallback
        logger.debug("FAISS not available or no index; returning placeholder docs.")
        return [f"placeholder-doc-{i}" for i in range(min(k, 3))]
