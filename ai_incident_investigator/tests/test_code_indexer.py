"""Tests for the CodeIndexer module."""

import os
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock sentence_transformers before any src imports that depend on it
_mock_st_module = MagicMock()
sys.modules.setdefault("sentence_transformers", _mock_st_module)

# Mock faiss (required by vector_store, imported transitively by code_indexer)
_mock_faiss = MagicMock()
sys.modules.setdefault("faiss", _mock_faiss)

from src.code_indexer import CodeIndexer, CodeChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed_batch.return_value = np.random.randn(3, 384).astype(np.float32)
    return model


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.add = MagicMock()
    store.save = MagicMock()
    store.clear = MagicMock()
    return store


@pytest.fixture
def mock_embedding_cache():
    cache = MagicMock()
    cache.get.return_value = None  # all cache misses by default
    cache.put = MagicMock()
    cache.save = MagicMock()
    cache.compute_hash = MagicMock(side_effect=lambda text: f"hash_{abs(hash(text))}")
    return cache


@pytest.fixture
def indexer(mock_embedding_model, mock_vector_store, mock_embedding_cache):
    return CodeIndexer(
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store,
        embedding_cache=mock_embedding_cache,
        chunk_size=50,
        chunk_overlap=5,
    )


# ---------------------------------------------------------------------------
# chunk_code
# ---------------------------------------------------------------------------

SAMPLE_PYTHON = """\
import os

def hello():
    print("hello")

def world():
    print("world")

class MyClass:
    def method(self):
        pass
"""


class TestChunkCode:
    def test_chunks_contain_function_names(self, indexer):
        """Multiple functions should produce chunks with function_name set."""
        chunks = indexer.chunk_code("sample.py", SAMPLE_PYTHON)
        assert len(chunks) > 0
        func_names = [c.function_name for c in chunks if c.function_name]
        assert "hello" in func_names or "world" in func_names or "MyClass" in func_names

    def test_chunks_have_metadata(self, indexer):
        chunks = indexer.chunk_code("sample.py", SAMPLE_PYTHON)
        for chunk in chunks:
            assert chunk.file_path == "sample.py"
            assert isinstance(chunk.start_line, int)
            assert isinstance(chunk.end_line, int)
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert len(chunk.snippet) > 0

    def test_chunk_respects_size_limits(self):
        """Small chunk_size should produce more, smaller chunks."""
        model = MagicMock()
        store = MagicMock()
        small_indexer = CodeIndexer(
            embedding_model=model,
            vector_store=store,
            chunk_size=10,
            chunk_overlap=2,
        )
        long_content = "\n".join(f"line_{i} = {i} * 2  # comment" for i in range(100))
        chunks = small_indexer.chunk_code("big.py", long_content)
        assert len(chunks) > 1
        for chunk in chunks:
            token_count = len(chunk.snippet.split())
            # Allow some tolerance because boundary splitting may not be exact
            assert token_count <= 20, f"Chunk has {token_count} tokens, expected ≤20"

    def test_empty_content_returns_empty(self, indexer):
        chunks = indexer.chunk_code("empty.py", "")
        assert chunks == []


# ---------------------------------------------------------------------------
# index_repository (mocked)
# ---------------------------------------------------------------------------

class TestIndexRepository:
    @patch("src.code_indexer.GitHubRepoManager")
    def test_full_pipeline(self, MockRepoManager, indexer):
        """Verify index_repository scans, chunks, embeds, and stores."""
        mock_manager = MagicMock()
        mock_manager.scan_files.return_value = ["/repo/main.py"]
        mock_manager.read_file.return_value = SAMPLE_PYTHON
        MockRepoManager.return_value = mock_manager

        # Make embed_batch return correct shape for produced chunks
        chunks = indexer.chunk_code("/repo/main.py", SAMPLE_PYTHON)
        n_chunks = len(chunks)
        indexer.embedding_model.embed_batch.return_value = np.random.randn(
            n_chunks, 384
        ).astype(np.float32)

        result = indexer.index_repository("/repo")

        assert result == n_chunks
        indexer.vector_store.add.assert_called_once()
        indexer.vector_store.save.assert_called_once()
