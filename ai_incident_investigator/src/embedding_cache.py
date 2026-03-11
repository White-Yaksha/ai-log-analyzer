"""Embedding cache for avoiding redundant embedding computations during re-indexing."""

import hashlib
import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache layer for code embeddings keyed by content hash.

    Uses SHA-256 hashes of source code content as cache keys so that
    unchanged files are never re-embedded during incremental re-indexing.
    The cache can be persisted to disk as a pickle file and restored on
    subsequent runs.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialise the embedding cache.

        Args:
            cache_dir: Directory used for on-disk persistence.  Defaults to
                ``data/embedding_cache/`` relative to the project root.
        """
        if cache_dir is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            cache_dir = os.path.join(project_root, "data", "embedding_cache")

        self.cache_dir: str = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self._cache: Dict[str, np.ndarray] = {}
        logger.info("EmbeddingCache initialised (cache_dir=%s)", self.cache_dir)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return the SHA-256 hex digest of *content*.

        Args:
            content: The source code (or any string) to hash.

        Returns:
            A 64-character lowercase hex string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core lookup / store
    # ------------------------------------------------------------------

    def get(self, content_hash: str) -> Optional[np.ndarray]:
        """Look up a cached embedding by its content hash.

        Args:
            content_hash: SHA-256 hex digest used as cache key.

        Returns:
            The cached numpy embedding array, or ``None`` if not present.
        """
        embedding = self._cache.get(content_hash)
        if embedding is not None:
            logger.debug("Cache hit for hash %s", content_hash[:12])
        else:
            logger.debug("Cache miss for hash %s", content_hash[:12])
        return embedding

    def put(self, content_hash: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache.

        Args:
            content_hash: SHA-256 hex digest used as cache key.
            embedding: The embedding vector to cache.
        """
        self._cache[content_hash] = embedding
        logger.debug("Cached embedding for hash %s", content_hash[:12])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the cache to disk as a pickle file.

        Args:
            path: Destination file path.  Defaults to
                ``<cache_dir>/cache.pkl``.
        """
        if path is None:
            path = os.path.join(self.cache_dir, "cache.pkl")

        with open(path, "wb") as fh:
            pickle.dump(self._cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d cache entries to %s", len(self._cache), path)

    def load(self, path: Optional[str] = None) -> None:
        """Load the cache from a pickle file on disk.

        If the file does not exist the cache is left empty and no error is
        raised.

        Args:
            path: Source file path.  Defaults to ``<cache_dir>/cache.pkl``.
        """
        if path is None:
            path = os.path.join(self.cache_dir, "cache.pkl")

        if not os.path.exists(path):
            logger.info("No cache file found at %s; starting with empty cache", path)
            return

        with open(path, "rb") as fh:
            self._cache = pickle.load(fh)
        logger.info("Loaded %d cache entries from %s", len(self._cache), path)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, content_hash: str) -> None:
        """Remove a single entry from the cache.

        Args:
            content_hash: The key to remove.  No error is raised if the
                key is not present.
        """
        if self._cache.pop(content_hash, None) is not None:
            logger.debug("Invalidated cache entry %s", content_hash[:12])

    def clear(self) -> None:
        """Wipe the entire cache, both in-memory and on disk."""
        self._cache.clear()

        cache_file = os.path.join(self.cache_dir, "cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info("Removed cache file %s", cache_file)

        logger.info("Cache cleared")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def __contains__(self, content_hash: str) -> bool:
        """Support the ``in`` operator for cache membership checks."""
        return content_hash in self._cache
