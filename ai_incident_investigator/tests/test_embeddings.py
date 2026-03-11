"""Tests for the EmbeddingModel module."""

import os
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure sentence_transformers can be imported
sys.modules.setdefault("sentence_transformers", MagicMock())

EMBEDDING_DIM = 384


def _make_mock_sentence_transformer(*args, **kwargs):
    """Factory that produces a mock SentenceTransformer."""
    mock_model = MagicMock()

    def _encode(text_or_texts, **kw):
        if isinstance(text_or_texts, str):
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)
        return np.random.randn(len(text_or_texts), EMBEDDING_DIM).astype(np.float32)

    mock_model.encode = MagicMock(side_effect=_encode)
    mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
    return mock_model


from src.embeddings import EmbeddingModel


@pytest.fixture(autouse=True)
def _patch_sentence_transformer():
    """Patch SentenceTransformer on every test so _load_model uses our mock."""
    with patch("src.embeddings.SentenceTransformer", side_effect=_make_mock_sentence_transformer):
        yield


class TestEmbedText:
    def test_returns_1d_array(self):
        model = EmbeddingModel()
        result = model.embed_text("hello world")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] == EMBEDDING_DIM

    def test_l2_normalized(self):
        model = EmbeddingModel()
        result = model.embed_text("normalize me")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


class TestEmbedBatch:
    def test_returns_2d_array(self):
        model = EmbeddingModel()
        texts = ["first", "second", "third"]
        result = model.embed_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, EMBEDDING_DIM)

    def test_rows_are_l2_normalized(self):
        model = EmbeddingModel()
        texts = ["a", "b"]
        result = model.embed_batch(texts)
        for i in range(result.shape[0]):
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 1e-5, f"Row {i} norm = {norm}"


class TestLazyLoading:
    def test_model_not_loaded_until_first_call(self):
        model = EmbeddingModel()
        assert model._model is None
        model.embed_text("trigger load")
        assert model._model is not None
