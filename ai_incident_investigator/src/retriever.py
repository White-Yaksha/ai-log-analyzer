"""Retriever module for querying the vector store with optional stack-trace priority boosting."""

import logging
import os
from copy import deepcopy

import numpy as np

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant code snippets by embedding a query and searching a vector store.

    Supports stack-trace priority boosting: results whose file paths match
    files extracted from a parsed stack trace receive a score multiplier,
    surfacing the most operationally-relevant code first.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        boost_factor: float = 1.5,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedding_model: Model used to produce query embeddings.
            vector_store: Store searched for nearest-neighbour code snippets.
            boost_factor: Multiplier applied to scores of results that match
                stack-trace priority files.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.boost_factor = boost_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        priority_files: list[dict] | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant code snippets for a natural-language query.

        Args:
            query: Natural-language search string.
            top_k: Number of results to return.
            priority_files: Optional list of ``{"file": str, "line": int}``
                dicts (e.g. from ``LogParser``).  Results whose ``file_path``
                basename matches any of these files receive a boosted score.

        Returns:
            A list of result dicts, each containing:
                - ``file_path`` (str)
                - ``function_name`` (str)
                - ``snippet`` (str)
                - ``score`` (float) — potentially boosted
                - ``line_numbers`` (dict with ``start`` and ``end``)
                - ``boosted`` (bool)
        """
        logger.debug("Embedding query: %s", query[:120])
        query_embedding: np.ndarray = self.embedding_model.embed_text(query)

        candidate_count = top_k * 3
        logger.debug(
            "Searching vector store for %d candidates (top_k=%d)",
            candidate_count,
            top_k,
        )
        raw_results: list[dict] = self.vector_store.search(
            query_embedding, top_k=candidate_count
        )

        priority_basenames: set[str] = self._extract_priority_basenames(
            priority_files
        )

        ranked_results = self._boost_and_rank(raw_results, priority_basenames)

        final = ranked_results[:top_k]
        logger.info(
            "Returning %d results (%d boosted)",
            len(final),
            sum(1 for r in final if r["boosted"]),
        )
        return final

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        priority_files: list[dict] | None = None,
    ) -> list[list[dict]]:
        """Run :meth:`retrieve` for each query and return all result lists.

        Args:
            queries: List of natural-language search strings.
            top_k: Number of results per query.
            priority_files: Optional stack-trace priority files applied to
                every query.

        Returns:
            A list of result lists, one per query.
        """
        logger.info("Batch retrieval for %d queries", len(queries))
        return [
            self.retrieve(query, top_k=top_k, priority_files=priority_files)
            for query in queries
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_priority_basenames(
        priority_files: list[dict] | None,
    ) -> set[str]:
        """Return a set of lowered basenames from priority file entries."""
        if not priority_files:
            return set()
        basenames: set[str] = set()
        for entry in priority_files:
            filepath = entry.get("file", "")
            if filepath:
                basenames.add(os.path.basename(filepath).lower())
        return basenames

    def _boost_and_rank(
        self,
        results: list[dict],
        priority_basenames: set[str],
    ) -> list[dict]:
        """Apply score boosting for priority files and re-sort descending."""
        ranked: list[dict] = []
        for result in results:
            entry = self._normalise_result(result)
            basename = os.path.basename(entry["file_path"]).lower()
            if priority_basenames and basename in priority_basenames:
                entry["score"] *= self.boost_factor
                entry["boosted"] = True
            else:
                entry["boosted"] = False
            ranked.append(entry)

        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked

    @staticmethod
    def _normalise_result(result: dict) -> dict:
        """Map raw vector-store result keys into the canonical output schema."""
        return {
            "file_path": result.get("file_path", ""),
            "function_name": result.get("function_name", ""),
            "snippet": result.get("snippet", ""),
            "score": float(result.get("score", 0.0)),
            "line_numbers": {
                "start": result.get("start_line"),
                "end": result.get("end_line"),
            },
        }
