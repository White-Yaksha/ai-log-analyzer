"""Tests for the EmbeddingCache module."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """Return an EmbeddingCache backed by a temp directory."""
    return EmbeddingCache(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# compute_hash
# ---------------------------------------------------------------------------

class TestComputeHash:
    def test_consistent_sha256(self):
        h1 = EmbeddingCache.compute_hash("hello")
        h2 = EmbeddingCache.compute_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_different_content_different_hash(self):
        h1 = EmbeddingCache.compute_hash("hello")
        h2 = EmbeddingCache.compute_hash("world")
        assert h1 != h2


# ---------------------------------------------------------------------------
# put / get
# ---------------------------------------------------------------------------

class TestPutGet:
    def test_store_and_retrieve(self, cache):
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        h = EmbeddingCache.compute_hash("test content")
        cache.put(h, embedding)
        result = cache.get(h)
        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    def test_cache_miss_returns_none(self, cache):
        result = cache.get("nonexistent_hash")
        assert result is None


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_persist_and_restore(self, tmp_path):
        cache1 = EmbeddingCache(cache_dir=str(tmp_path))
        embedding = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        h = EmbeddingCache.compute_hash("persist me")
        cache1.put(h, embedding)
        cache1.save()

        cache2 = EmbeddingCache(cache_dir=str(tmp_path))
        cache2.load()
        result = cache2.get(h)
        assert result is not None
        np.testing.assert_array_equal(result, embedding)


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_removes_entry(self, cache):
        embedding = np.array([7.0, 8.0], dtype=np.float32)
        h = EmbeddingCache.compute_hash("remove me")
        cache.put(h, embedding)
        assert cache.get(h) is not None

        cache.invalidate(h)
        assert cache.get(h) is None

    def test_invalidate_nonexistent_is_noop(self, cache):
        # Should not raise
        cache.invalidate("does_not_exist")


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_cache(self, cache):
        cache.put("h1", np.array([1.0]))
        cache.put("h2", np.array([2.0]))
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("h1") is None


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------

class TestDunderMethods:
    def test_len(self, cache):
        assert len(cache) == 0
        cache.put("a", np.array([1.0]))
        assert len(cache) == 1
        cache.put("b", np.array([2.0]))
        assert len(cache) == 2

    def test_contains(self, cache):
        h = "test_hash"
        assert h not in cache
        cache.put(h, np.array([1.0]))
        assert h in cache
