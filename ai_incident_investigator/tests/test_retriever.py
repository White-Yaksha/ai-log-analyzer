"""Tests for the Retriever module."""

import os
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed_text.return_value = np.random.randn(384).astype(np.float32)
    return model


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def retriever(mock_embedding_model, mock_vector_store):
    return Retriever(
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store,
        boost_factor=1.5,
    )


def _sample_results():
    """Return sample raw search results from the vector store."""
    return [
        {
            "file_path": "/repo/event_producer.py",
            "function_name": "send_event",
            "snippet": "def send_event(): ...",
            "start_line": 10,
            "end_line": 20,
            "score": 0.8,
        },
        {
            "file_path": "/repo/utils.py",
            "function_name": "validate",
            "snippet": "def validate(): ...",
            "start_line": 1,
            "end_line": 5,
            "score": 0.6,
        },
        {
            "file_path": "/repo/config.py",
            "function_name": "load_config",
            "snippet": "def load_config(): ...",
            "start_line": 1,
            "end_line": 15,
            "score": 0.4,
        },
    ]


# ---------------------------------------------------------------------------
# Basic retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_basic_retrieve_output_format(self, retriever, mock_vector_store):
        mock_vector_store.search.return_value = _sample_results()
        results = retriever.retrieve("KafkaTimeoutException", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "file_path" in r
            assert "function_name" in r
            assert "snippet" in r
            assert "score" in r
            assert "line_numbers" in r
            assert "boosted" in r


# ---------------------------------------------------------------------------
# Stack trace priority boosting
# ---------------------------------------------------------------------------

class TestPriorityBoosting:
    def test_matching_file_is_boosted(self, retriever, mock_vector_store):
        """Results matching priority_files should have boosted=True and higher scores."""
        mock_vector_store.search.return_value = _sample_results()
        priority = [{"file": "event_producer.py", "line": 15}]
        results = retriever.retrieve("error", top_k=3, priority_files=priority)

        boosted = [r for r in results if r["boosted"]]
        assert len(boosted) >= 1
        assert any("event_producer" in r["file_path"] for r in boosted)

        # Boosted result should have score > original 0.8
        for r in boosted:
            if "event_producer" in r["file_path"]:
                assert r["score"] > 0.8

    def test_non_matching_files_not_boosted(self, retriever, mock_vector_store):
        mock_vector_store.search.return_value = _sample_results()
        priority = [{"file": "event_producer.py", "line": 15}]
        results = retriever.retrieve("error", top_k=3, priority_files=priority)

        non_boosted = [r for r in results if not r["boosted"]]
        for r in non_boosted:
            assert "event_producer" not in r["file_path"]


# ---------------------------------------------------------------------------
# retrieve_batch
# ---------------------------------------------------------------------------

class TestRetrieveBatch:
    def test_calls_retrieve_for_each_query(self, retriever, mock_vector_store):
        mock_vector_store.search.return_value = _sample_results()
        queries = ["error one", "error two", "error three"]
        batch_results = retriever.retrieve_batch(queries, top_k=2)

        assert len(batch_results) == 3
        for results in batch_results:
            assert isinstance(results, list)
