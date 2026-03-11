"""Tests for the VectorStore module."""

import os
import pickle
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Build mock faiss module before importing VectorStore
# ---------------------------------------------------------------------------

class _MockIndexFlatIP:
    """Mock FAISS IndexFlatIP that tracks added vectors."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.ntotal = 0
        self._vectors = []

    def add(self, vectors):
        self._vectors.append(vectors.copy())
        self.ntotal += vectors.shape[0]

    def search(self, query, k):
        n_results = min(k, self.ntotal)
        distances = np.array([[0.9 - 0.1 * i for i in range(n_results)]], dtype=np.float32)
        indices = np.array([[i for i in range(n_results)]], dtype=np.int64)
        return distances, indices


sys.modules.setdefault("faiss", MagicMock())

from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 384


@pytest.fixture(autouse=True)
def _patch_faiss():
    """Patch faiss references inside src.vector_store for every test."""
    mock_faiss = MagicMock()
    mock_faiss.IndexFlatIP = _MockIndexFlatIP
    mock_faiss.write_index = MagicMock()
    mock_faiss.read_index = MagicMock(return_value=_MockIndexFlatIP(DIM))
    with patch("src.vector_store.faiss", mock_faiss):
        yield mock_faiss


@pytest.fixture
def store():
    return VectorStore(dimension=DIM)


def _random_embeddings(n, dim=DIM):
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_stores_vectors_and_metadata(self, store):
        embs = _random_embeddings(3)
        meta = [{"file_path": f"f{i}.py"} for i in range(3)]
        store.add(embs, meta)
        assert len(store) == 3
        assert len(store._metadata) == 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_results_with_metadata(self, store):
        embs = _random_embeddings(5)
        meta = [{"file_path": f"file{i}.py", "snippet": f"code {i}"} for i in range(5)]
        store.add(embs, meta)

        query = _random_embeddings(1)[0]
        results = store.search(query, top_k=3)

        assert len(results) == 3
        for r in results:
            assert "score" in r
            assert "file_path" in r

    def test_search_empty_index_returns_empty(self, store):
        query = _random_embeddings(1)[0]
        results = store.search(query, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_calls_faiss_write(self, store, tmp_path, _patch_faiss):
        embs = _random_embeddings(2)
        meta = [{"file_path": "a.py"}, {"file_path": "b.py"}]
        store.add(embs, meta)
        store.save(str(tmp_path))
        _patch_faiss.write_index.assert_called()

    def test_load_calls_faiss_read(self, store, tmp_path, _patch_faiss):
        index_path = os.path.join(str(tmp_path), "faiss.index")
        meta_path = os.path.join(str(tmp_path), "metadata.pkl")
        open(index_path, "w").close()
        with open(meta_path, "wb") as f:
            pickle.dump([{"file_path": "loaded.py"}], f)

        store.load(str(tmp_path))
        _patch_faiss.read_index.assert_called()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_state(self, store):
        embs = _random_embeddings(3)
        meta = [{"file_path": f"f{i}.py"} for i in range(3)]
        store.add(embs, meta)
        assert len(store) == 3

        store.clear()
        assert len(store) == 0
        assert store._metadata == []


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_len_reflects_additions(self, store):
        assert len(store) == 0
        store.add(_random_embeddings(2), [{"a": 1}, {"a": 2}])
        assert len(store) == 2
