"""
Incident Analyzer — orchestrates the full incident investigation pipeline.

Combines log ingestion (Airflow or local file), log parsing, code retrieval,
and LLM-based root-cause analysis into a single, cohesive workflow.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from src.airflow_client import AirflowClient
from src.code_indexer import CodeIndexer
from src.context_builder import ContextBuilder
from src.embedding_cache import EmbeddingCache
from src.embeddings import EmbeddingModel
from src.github_repo_manager import GitHubRepoManager
from src.llm_engine import LLMEngine
from src.log_parser import LogParser
from src.retriever import Retriever
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Section headers the LLM is expected to produce in its output.
_LLM_SECTION_HEADERS = [
    "Incident Summary",
    "Root Cause",
    "Impacted Module",
    "Failure Path",
    "Suggested Fix",
]

_SECTION_KEY_MAP: dict[str, str] = {
    "Incident Summary": "incident_summary",
    "Root Cause": "root_cause",
    "Impacted Module": "impacted_module",
    "Failure Path": "failure_path",
    "Suggested Fix": "suggested_fix",
}


class IncidentAnalyzer:
    """End-to-end incident investigation engine.

    Orchestrates log fetching, parsing, code retrieval, and LLM analysis to
    produce a structured incident report with root-cause insights and
    suggested fixes.
    """

    def __init__(
        self,
        airflow_url: str = None,
        airflow_user: str = None,
        airflow_pass: str = None,
        repo_url: str = None,
        github_token: str = None,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        top_k: int = 5,
        index_path: str = "data/vector_index",
        cache_path: str = "data/embedding_cache",
    ) -> None:
        """Initialise all pipeline components.

        Args:
            airflow_url: Base URL of the Airflow REST API.
            airflow_user: Airflow username for authentication.
            airflow_pass: Airflow password for authentication.
            repo_url: Default repository URL to clone / index.
            github_token: GitHub personal-access token for private repos.
            model_name: HuggingFace model identifier for the LLM engine.
            top_k: Number of code snippets to retrieve per query.
            index_path: Path for persisting the FAISS vector index.
            cache_path: Path for persisting the embedding cache.
        """
        self.repo_url = repo_url
        self.top_k = top_k
        self.index_path = index_path
        self.cache_path = cache_path
        self._repo_dir: Optional[str] = None

        # --- component wiring ---
        self.airflow_client = AirflowClient(
            base_url=airflow_url,
            username=airflow_user,
            password=airflow_pass,
        )
        self.log_parser = LogParser()
        self.github_manager = GitHubRepoManager(token=github_token)

        self.embedding_model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_cache = EmbeddingCache(cache_dir=cache_path)
        self.vector_store = VectorStore(dimension=self.embedding_model.get_dimension())

        self.code_indexer = CodeIndexer(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            embedding_cache=self.embedding_cache,
        )
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )
        self.context_builder = ContextBuilder()
        self.llm_engine = LLMEngine(model_name=model_name)

        # Load persisted state when available.
        self._load_persisted_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_persisted_state(self) -> None:
        """Load vector index and embedding cache from disk if they exist."""
        if os.path.exists(self.index_path):
            try:
                self.vector_store.load(self.index_path)
                logger.info("Loaded vector index from %s (%d vectors)", self.index_path, len(self.vector_store))
            except Exception:
                logger.warning("Failed to load vector index from %s", self.index_path, exc_info=True)

        if os.path.exists(self.cache_path):
            try:
                self.embedding_cache.load()
                logger.info("Loaded embedding cache from %s", self.cache_path)
            except Exception:
                logger.warning("Failed to load embedding cache from %s", self.cache_path, exc_info=True)

    # ------------------------------------------------------------------
    # Repository indexing
    # ------------------------------------------------------------------

    def index_repo(self, repo_url: str = None, force: bool = False) -> int:
        """Clone (or pull) a repository and index its source code.

        Args:
            repo_url: Git URL to clone.  Falls back to ``self.repo_url``.
            force: When *True*, clear the existing index and reindex from
                scratch instead of performing an incremental update.

        Returns:
            The total number of indexed code chunks.
        """
        url = repo_url or self.repo_url
        if not url:
            raise ValueError("No repository URL provided — pass repo_url or set it in the constructor.")

        target_dir = os.path.join("data", "repo")

        if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, ".git")):
            logger.info("Repository already cloned — pulling latest changes.")
            self.github_manager.pull_latest(target_dir)
        else:
            logger.info("Cloning repository from %s", url)
            target_dir = self.github_manager.clone_repo(url, target_dir)

        self._repo_dir = target_dir

        if force:
            logger.info("Force flag set — reindexing repository.")
            chunk_count = self.code_indexer.reindex(target_dir)
        else:
            chunk_count = self.code_indexer.index_repository(target_dir)

        # Persist index and cache after indexing.
        self.vector_store.save(self.index_path)
        self.embedding_cache.save()
        logger.info("Indexed %d chunks. Index saved to %s", chunk_count, self.index_path)
        return chunk_count

    # ------------------------------------------------------------------
    # Analysis pipelines
    # ------------------------------------------------------------------

    def analyze_from_airflow(self, dag_id: str, run_id: str, task_id: str) -> dict:
        """Run the full investigation pipeline using Airflow logs.

        Args:
            dag_id: Airflow DAG identifier.
            run_id: DAG-run identifier.
            task_id: Task identifier within the run.

        Returns:
            A structured incident report dictionary.
        """
        logger.info("Fetching Airflow logs for dag=%s run=%s task=%s", dag_id, run_id, task_id)
        raw_log = self.airflow_client.fetch_dag_logs(dag_id, run_id, task_id)
        return self._run_pipeline(raw_log)

    def analyze_from_file(self, log_file_path: str) -> dict:
        """Run the full investigation pipeline using a local log file.

        Args:
            log_file_path: Path to a plain-text log file on disk.

        Returns:
            A structured incident report dictionary.
        """
        logger.info("Loading log file from %s", log_file_path)
        raw_log = self.airflow_client.load_local_log(log_file_path)
        return self._run_pipeline(raw_log)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, raw_log: str) -> dict:
        """Execute the shared analysis pipeline on a raw log string.

        Steps:
            1. Parse the log.
            2. Build a retrieval query from error_type + keywords.
            3. Retrieve relevant code snippets.
            4. Compute a confidence score from retrieval similarities.
            5. Build a prompt and generate the LLM analysis.
            6. Format the structured report.

        Args:
            raw_log: The raw log text to analyse.

        Returns:
            A structured incident report dictionary.
        """
        # 1 — Parse
        parsed_log: dict[str, Any] = self.log_parser.parse(raw_log)
        logger.info(
            "Parsed log: error_type=%s severity=%s keywords=%s",
            parsed_log.get("error_type"),
            parsed_log.get("severity"),
            parsed_log.get("keywords"),
        )

        # 2 — Build retrieval query
        error_type = parsed_log.get("error_type", "")
        keywords = parsed_log.get("keywords", [])
        query_parts = [error_type] + (keywords if isinstance(keywords, list) else [keywords])
        query = " ".join(str(p) for p in query_parts if p).strip()
        if not query:
            query = parsed_log.get("summary", "error")
        logger.info("Retrieval query: %s", query)

        # 3 — Retrieve code
        priority_files = parsed_log.get("referenced_files", [])
        retrieval_results: list[dict] = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            priority_files=priority_files,
        )
        logger.info("Retrieved %d code snippets", len(retrieval_results))

        # 4 — Confidence
        confidence = self._compute_confidence(retrieval_results)

        # 5 — Prompt + LLM
        prompt = self.context_builder.build_prompt(parsed_log, retrieval_results)
        llm_output: str = self.llm_engine.generate(prompt)
        logger.info("LLM generation complete (%d chars)", len(llm_output))

        # 6 — Format
        report = self._format_report(parsed_log, llm_output, confidence, retrieval_results)
        return report

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, retrieval_results: list[dict]) -> float:
        """Compute a confidence score from retrieval similarity scores.

        The score is the mean of the ``score`` values in *retrieval_results*,
        clamped to [0.0, 1.0] and rounded to two decimal places.

        Args:
            retrieval_results: List of retrieval dicts, each containing a
                ``score`` key with a float value.

        Returns:
            A float in [0.0, 1.0].
        """
        if not retrieval_results:
            return 0.0

        scores = [r.get("score", 0.0) for r in retrieval_results]
        avg = sum(scores) / len(scores)
        clamped = max(0.0, min(1.0, avg))
        return round(clamped, 2)

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_sections(llm_output: str) -> dict[str, str]:
        """Extract named sections from the LLM output.

        Looks for headers matching the expected section names (case-insensitive)
        and captures everything between consecutive headers.

        Args:
            llm_output: Raw text from the LLM.

        Returns:
            A mapping of section-key → extracted text.
        """
        sections: dict[str, str] = {}
        # Build a regex that matches any known header.
        header_pattern = "|".join(re.escape(h) for h in _LLM_SECTION_HEADERS)
        # Match lines like "## Root Cause", "Root Cause:", "**Root Cause**", etc.
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:[#*\-]*\s*)?({header_pattern})\s*[:\-#*]*\s*\n",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(llm_output))

        for i, match in enumerate(matches):
            header_text = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(llm_output)
            content = llm_output[start:end].strip()
            # Map the matched header to its canonical key.
            for canonical, key in _SECTION_KEY_MAP.items():
                if canonical.lower() == header_text.lower():
                    sections[key] = content
                    break

        return sections

    def _format_report(
        self,
        parsed_log: dict,
        llm_output: str,
        confidence: float,
        retrieval_results: list[dict],
    ) -> dict:
        """Build the structured incident report dictionary.

        Attempts to parse the LLM output into discrete sections.  If parsing
        fails for any section, a sensible default is used and the full LLM
        response is preserved in ``raw_llm_output``.

        Args:
            parsed_log: Parsed log dictionary from :class:`LogParser`.
            llm_output: Raw text generated by the LLM.
            confidence: Confidence score (0.0–1.0).
            retrieval_results: Code snippets returned by the retriever.

        Returns:
            A structured report dictionary.
        """
        sections = self._parse_llm_sections(llm_output)

        retrieved_code = [
            {
                "file_path": r.get("file_path", "unknown"),
                "function_name": r.get("function_name", "unknown"),
            }
            for r in retrieval_results
        ]

        return {
            "incident_summary": sections.get("incident_summary", llm_output.strip()),
            "root_cause": sections.get("root_cause", "Unable to determine root cause."),
            "impacted_module": sections.get(
                "impacted_module",
                parsed_log.get("source_module", "Unknown"),
            ),
            "failure_path": sections.get("failure_path", "Not determined."),
            "suggested_fix": sections.get("suggested_fix", "Review the logs and code for further investigation."),
            "severity": parsed_log.get("severity", "Unknown"),
            "confidence": confidence,
            "timeline": parsed_log.get("timeline", []),
            "retrieved_code": retrieved_code,
            "raw_llm_output": llm_output,
        }

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def format_text_report(self, report: dict) -> str:
        """Format a report dictionary as a human-readable text report.

        Args:
            report: A structured report dict as returned by
                :meth:`analyze_from_airflow` or :meth:`analyze_from_file`.

        Returns:
            A multi-line string suitable for printing to the console.
        """
        separator = "====================================="
        lines: list[str] = [
            separator,
            "INCIDENT ANALYSIS REPORT",
            separator,
            "",
            "Incident Summary",
            "----------------",
            report.get("incident_summary", "N/A"),
            "",
            "Root Cause",
            "----------",
            report.get("root_cause", "N/A"),
            "",
            "Impacted Module",
            "---------------",
            report.get("impacted_module", "N/A"),
            "",
            "Failure Path",
            "------------",
            report.get("failure_path", "N/A"),
            "",
            "Suggested Fix",
            "-------------",
            report.get("suggested_fix", "N/A"),
            "",
            f"Severity: {report.get('severity', 'N/A')}",
            f"Confidence: {report.get('confidence', 'N/A')}",
            "",
            "Incident Timeline",
            "-----------------",
        ]

        timeline = report.get("timeline", [])
        if timeline:
            for entry in timeline:
                if isinstance(entry, dict):
                    ts = entry.get("timestamp", "")
                    event = entry.get("event", "")
                    lines.append(f"{ts}  {event}")
                else:
                    lines.append(str(entry))
        else:
            lines.append("No timeline data available.")

        lines.append("")
        lines.append("Retrieved Code References")
        lines.append("-------------------------")

        retrieved_code = report.get("retrieved_code", [])
        if retrieved_code:
            for ref in retrieved_code:
                file_path = ref.get("file_path", "unknown")
                func_name = ref.get("function_name", "unknown")
                lines.append(f"- {file_path} :: {func_name}")
        else:
            lines.append("No code references retrieved.")

        return "\n".join(lines)

    def format_json_report(self, report: dict) -> str:
        """Serialise a report dictionary to a JSON string.

        Args:
            report: A structured report dict as returned by
                :meth:`analyze_from_airflow` or :meth:`analyze_from_file`.

        Returns:
            A pretty-printed JSON string.
        """
        return json.dumps(report, indent=2, default=str)
