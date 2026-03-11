"""Embedding model wrapper for generating text embeddings."""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating and normalizing text embeddings.

    Uses lazy loading to defer model initialization until the first embedding
    request, keeping startup fast when the model isn't immediately needed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model configuration.

        Args:
            model_name: HuggingFace model identifier for the sentence transformer.
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> None:
        """Load the SentenceTransformer model on first use."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded (dimension=%d)", self.get_dimension())

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a unit-length vector.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D numpy array (the L2-normalized embedding vector).
        """
        self._load_model()
        embedding: np.ndarray = self._model.encode(text)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into unit-length vectors.

        Uses the model's built-in batch encoding for efficiency.

        Args:
            texts: List of input texts to embed.

        Returns:
            A 2-D numpy array of shape ``(len(texts), dim)`` with each row
            L2-normalized.
        """
        self._load_model()
        embeddings: np.ndarray = self._model.encode(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings = embeddings / norms
        return embeddings

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension size (e.g. 384 for all-MiniLM-L6-v2).
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
