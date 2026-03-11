"""Vector store module backed by FAISS for similarity search over code embeddings."""

import logging
import os
import pickle

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store for cosine-similarity search.

    Uses ``IndexFlatIP`` (inner-product) on L2-normalised vectors so that
    scores correspond to cosine similarity.  Metadata dicts are stored in
    parallel with the indexed vectors and returned alongside search results.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialise an empty vector store.

        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        """Add vectors and their associated metadata to the store.

        Args:
            embeddings: Array of shape ``(n, dimension)`` containing
                L2-normalised embedding vectors.
            metadata_list: List of metadata dicts, one per vector.  Typical
                keys include *file_path*, *function_name*, *snippet*,
                *start_line*, and *end_line*.

        Raises:
            ValueError: If the number of metadata entries does not match the
                number of embedding vectors, or if the embedding dimension
                does not match the store dimension.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )
        if len(metadata_list) != embeddings.shape[0]:
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) must equal "
                f"the number of embeddings ({embeddings.shape[0]})"
            )

        self._index.add(embeddings.astype(np.float32))
        self._metadata.extend(metadata_list)
        logger.debug("Added %d vectors (total: %d)", embeddings.shape[0], len(self))

    def clear(self) -> None:
        """Reset the index and metadata to an empty state."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = []
        logger.debug("Vector store cleared")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the *top_k* most similar vectors.

        Args:
            query_embedding: Query vector of shape ``(dimension,)`` or
                ``(1, dimension)``.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts ordered by descending similarity.  Each dict
            contains all fields from the stored metadata plus a ``score``
            key with the inner-product similarity value.

        Raises:
            ValueError: If *query_embedding* has an incompatible shape.
        """
        if self._index.ntotal == 0:
            logger.debug("Search called on empty index; returning empty list")
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            if query.shape[0] != self.dimension:
                raise ValueError(
                    f"Query dimension ({query.shape[0]}) does not match "
                    f"store dimension ({self.dimension})"
                )
            query = query.reshape(1, -1)
        elif query.ndim == 2:
            if query.shape != (1, self.dimension):
                raise ValueError(
                    f"Expected query shape (1, {self.dimension}), got {query.shape}"
                )
        else:
            raise ValueError(
                f"query_embedding must be 1-D or 2-D, got {query.ndim}-D"
            )

        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, effective_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = {**self._metadata[idx], "score": float(score)}
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index and metadata to disk.

        Creates *path* (and parent directories) if it does not exist.  The
        FAISS index is written to ``{path}/faiss.index`` and metadata to
        ``{path}/metadata.pkl``.

        Args:
            path: Directory in which to store the files.
        """
        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "metadata.pkl")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "wb") as fh:
            pickle.dump(self._metadata, fh)
        logger.info("Saved vector store (%d vectors) to %s", len(self), path)

    def load(self, path: str) -> None:
        """Load a previously saved index and metadata from disk.

        If the expected files do not exist a warning is logged and the
        current (empty) state is preserved.

        Args:
            path: Directory containing ``faiss.index`` and ``metadata.pkl``.
        """
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(
                "Vector store files not found in %s; keeping empty state", path
            )
            return

        self._index = faiss.read_index(index_path)
        with open(meta_path, "rb") as fh:
            self._metadata = pickle.load(fh)
        logger.info("Loaded vector store (%d vectors) from %s", len(self), path)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of vectors currently in the index."""
        return self._index.ntotal
