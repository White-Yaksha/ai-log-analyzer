"""Code indexing module for scanning, chunking, embedding, and storing repository code."""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.embedding_cache import EmbeddingCache
from src.embeddings import EmbeddingModel
from src.github_repo_manager import GitHubRepoManager
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Regex patterns for detecting function/class boundaries across languages.
_BOUNDARY_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:def |class |public |private |protected )",
    re.MULTILINE,
)

_FUNC_NAME_RE = re.compile(r"(?:def|class)\s+(\w+)")


@dataclass
class CodeChunk:
    """A contiguous chunk of source code with location metadata."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str
    function_name: str = field(default="")


class CodeIndexer:
    """Indexes a code repository by chunking source files, computing embeddings,
    and storing them in a vector store for later similarity search.

    Supports an optional embedding cache to avoid recomputation for unchanged code.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        embedding_cache: Optional[EmbeddingCache] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialise the indexer.

        Args:
            embedding_model: Model used to produce embeddings for code chunks.
            vector_store: Store where embeddings and metadata are persisted.
            embedding_cache: Optional cache to skip re-embedding unchanged code.
            chunk_size: Approximate maximum chunk size measured in whitespace-split tokens.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_repository(
        self,
        repo_dir: str,
        extensions: Optional[list[str]] = None,
    ) -> int:
        """Run the full indexing pipeline: scan → chunk → embed → store.

        Args:
            repo_dir: Root directory of the repository to index.
            extensions: File extensions to include (e.g. ``['.py', '.java']``).
                        Passed through to ``GitHubRepoManager.scan_files``.

        Returns:
            Total number of code chunks indexed.
        """
        repo_manager = GitHubRepoManager()
        file_paths = repo_manager.scan_files(repo_dir, extensions)
        logger.info("Scanned %d files in %s", len(file_paths), repo_dir)

        all_chunks: list[CodeChunk] = []
        for fpath in file_paths:
            try:
                content = repo_manager.read_file(fpath)
            except Exception:
                logger.warning("Failed to read %s – skipping", fpath, exc_info=True)
                continue
            if not content.strip():
                continue
            chunks = self.chunk_code(fpath, content)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced – nothing to index")
            return 0

        logger.info("Produced %d chunks from %d files", len(all_chunks), len(file_paths))

        embeddings = self._embed_chunks(all_chunks)

        metadata_list = [
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "snippet": c.snippet,
                "function_name": c.function_name,
            }
            for c in all_chunks
        ]

        self.vector_store.add(embeddings, metadata_list)

        # Persist vector store next to the repo.
        store_path = os.path.join(repo_dir, ".code_index")
        os.makedirs(store_path, exist_ok=True)
        self.vector_store.save(store_path)
        logger.info("Vector store saved to %s", store_path)

        if self.embedding_cache is not None:
            self.embedding_cache.save()
            logger.info("Embedding cache saved")

        return len(all_chunks)

    def chunk_code(self, file_path: str, content: str) -> list[CodeChunk]:
        """Split source code into chunks respecting function/class boundaries.

        Strategy:
            1. Attempt to split on function / class boundaries.
            2. If any resulting block exceeds ``chunk_size`` tokens, sub-split by lines.
            3. Fallback to a simple line-based split with overlap when no boundaries
               are detected.

        Args:
            file_path: Path to the source file (used in metadata).
            content: Full text content of the file.

        Returns:
            List of ``CodeChunk`` instances.
        """
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        boundaries = list(_BOUNDARY_RE.finditer(content))

        if boundaries:
            return self._chunk_by_boundaries(file_path, lines, content, boundaries)

        return self._chunk_by_lines(file_path, lines)

    def reindex(
        self,
        repo_dir: str,
        extensions: Optional[list[str]] = None,
    ) -> int:
        """Clear all stored data and rebuild the index from scratch.

        Args:
            repo_dir: Root directory of the repository to index.
            extensions: File extensions to include.

        Returns:
            Total number of code chunks indexed after rebuilding.
        """
        logger.info("Clearing vector store and cache for full reindex")
        self.vector_store.clear()
        if self.embedding_cache is not None:
            self.embedding_cache.save()
        return self.index_repository(repo_dir, extensions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """Compute embeddings for a list of chunks, using the cache when possible.

        Returns:
            A 2-D numpy array of shape ``(len(chunks), dim)``.
        """
        snippets = [c.snippet for c in chunks]
        embeddings: list[Optional[np.ndarray]] = [None] * len(snippets)

        # Indices of snippets that need fresh computation.
        miss_indices: list[int] = []

        if self.embedding_cache is not None:
            for idx, snippet in enumerate(snippets):
                content_hash = EmbeddingCache.compute_hash(snippet)
                cached = self.embedding_cache.get(content_hash)
                if cached is not None:
                    embeddings[idx] = cached
                else:
                    miss_indices.append(idx)
            logger.info(
                "Embedding cache: %d hits, %d misses",
                len(snippets) - len(miss_indices),
                len(miss_indices),
            )
        else:
            miss_indices = list(range(len(snippets)))

        if miss_indices:
            miss_texts = [snippets[i] for i in miss_indices]
            computed = self.embedding_model.embed_batch(miss_texts)
            for pos, idx in enumerate(miss_indices):
                vec = computed[pos]
                embeddings[idx] = vec
                if self.embedding_cache is not None:
                    content_hash = EmbeddingCache.compute_hash(snippets[idx])
                    self.embedding_cache.put(content_hash, vec)

        return np.vstack(embeddings)

    # -- Boundary-based chunking -----------------------------------------

    def _chunk_by_boundaries(
        self,
        file_path: str,
        lines: list[str],
        content: str,
        boundaries: list[re.Match],
    ) -> list[CodeChunk]:
        """Split code along function/class definition boundaries.

        Blocks that exceed ``chunk_size`` tokens are further sub-split by lines.
        """
        # Convert character offsets of boundary matches to 0-based line numbers.
        boundary_line_nos = [
            content[:m.start()].count("\n") for m in boundaries
        ]

        # Build (start_line_0based, end_line_0based_exclusive) segments.
        segments: list[tuple[int, int]] = []
        for i, bl in enumerate(boundary_line_nos):
            # Include any preamble lines before the first boundary.
            if i == 0 and bl > 0:
                segments.append((0, bl))
            end = boundary_line_nos[i + 1] if i + 1 < len(boundary_line_nos) else len(lines)
            segments.append((bl, end))

        chunks: list[CodeChunk] = []
        for seg_start, seg_end in segments:
            seg_lines = lines[seg_start:seg_end]
            seg_text = "".join(seg_lines)
            token_count = len(seg_text.split())

            if token_count > self.chunk_size:
                chunks.extend(
                    self._subsplit_lines(file_path, seg_lines, seg_start)
                )
            else:
                func_name = self._extract_function_name(seg_text)
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=seg_start + 1,
                        end_line=seg_end,
                        snippet=seg_text,
                        function_name=func_name,
                    )
                )

        return chunks

    # -- Line-based chunking (fallback) ----------------------------------

    def _chunk_by_lines(
        self,
        file_path: str,
        lines: list[str],
    ) -> list[CodeChunk]:
        """Split code purely by line count with overlap."""
        return self._subsplit_lines(file_path, lines, offset=0)

    def _subsplit_lines(
        self,
        file_path: str,
        lines: list[str],
        offset: int,
    ) -> list[CodeChunk]:
        """Split *lines* into chunks of roughly ``chunk_size`` tokens with overlap.

        Args:
            file_path: Source file path.
            lines: The block of lines to split.
            offset: 0-based line offset of *lines[0]* within the original file.

        Returns:
            List of ``CodeChunk`` instances.
        """
        chunks: list[CodeChunk] = []
        total = len(lines)
        start = 0

        while start < total:
            token_count = 0
            end = start
            while end < total and token_count < self.chunk_size:
                token_count += len(lines[end].split())
                end += 1

            snippet = "".join(lines[start:end])
            func_name = self._extract_function_name(snippet)

            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=offset + start + 1,
                    end_line=offset + end,
                    snippet=snippet,
                    function_name=func_name,
                )
            )

            advance = max(1, end - start - self._overlap_in_lines(lines, start, end))
            start += advance

            # Prevent an overlap-only trailing chunk that duplicates the last one.
            if start >= total:
                break

        return chunks

    def _overlap_in_lines(
        self,
        lines: list[str],
        start: int,
        end: int,
    ) -> int:
        """Determine how many lines from the end of the current window correspond
        to ``chunk_overlap`` tokens so the next window overlaps properly."""
        overlap_tokens = 0
        overlap_lines = 0
        for i in range(end - 1, start - 1, -1):
            line_tokens = len(lines[i].split())
            if overlap_tokens + line_tokens > self.chunk_overlap:
                break
            overlap_tokens += line_tokens
            overlap_lines += 1
        return overlap_lines

    # -- Utilities -------------------------------------------------------

    @staticmethod
    def _extract_function_name(text: str) -> str:
        """Return the first ``def`` or ``class`` name found in *text*, or ``""``."""
        match = _FUNC_NAME_RE.search(text)
        return match.group(1) if match else ""
