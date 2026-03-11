"""Tests for the IncidentAnalyzer module."""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Mock heavy third-party imports before importing IncidentAnalyzer
# ---------------------------------------------------------------------------

# faiss
_mock_faiss = MagicMock()

class _FakeIndexFlatIP:
    def __init__(self, dim):
        self.dimension = dim
        self.ntotal = 0
    def add(self, v):
        self.ntotal += v.shape[0]
    def search(self, q, k):
        import numpy as np
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

_mock_faiss.IndexFlatIP = _FakeIndexFlatIP
_mock_faiss.write_index = MagicMock()
_mock_faiss.read_index = MagicMock(return_value=_FakeIndexFlatIP(384))
sys.modules["faiss"] = _mock_faiss

# torch
sys.modules.setdefault("torch", MagicMock())

# transformers
sys.modules.setdefault("transformers", MagicMock())

# sentence_transformers
_mock_st = MagicMock()

class _FakeSentenceTransformer:
    def __init__(self, *a, **kw):
        pass
    def encode(self, text, **kw):
        import numpy as np
        if isinstance(text, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(text), 384).astype(np.float32)
    def get_sentence_embedding_dimension(self):
        return 384

_mock_st.SentenceTransformer = _FakeSentenceTransformer
sys.modules["sentence_transformers"] = _mock_st

# git
sys.modules.setdefault("git", MagicMock())

from src.incident_analyzer import IncidentAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_report():
    return {
        "incident_summary": "Kafka producer timed out.",
        "root_cause": "Broker unreachable.",
        "impacted_module": "event_producer",
        "failure_path": "DAG → task → producer.send → timeout",
        "suggested_fix": "Check broker connectivity.",
        "severity": "High",
        "confidence": 0.6,
        "timeline": [
            {"timestamp": "2026-03-08 12:03:21", "event": "DAG started"},
            {"timestamp": "2026-03-08 12:04:00", "event": "Timeout"},
        ],
        "retrieved_code": [
            {"file_path": "producer.py", "function_name": "send_event"},
        ],
        "raw_llm_output": "Incident Summary\nKafka producer timed out.",
    }


def _make_analyzer():
    """Build an IncidentAnalyzer with all components mocked."""
    with patch("src.incident_analyzer.EmbeddingModel") as MockEmbed, \
         patch("src.incident_analyzer.VectorStore") as MockVS, \
         patch("src.incident_analyzer.EmbeddingCache") as MockCache, \
         patch("src.incident_analyzer.CodeIndexer") as MockCI, \
         patch("src.incident_analyzer.Retriever") as MockRet, \
         patch("src.incident_analyzer.ContextBuilder") as MockCB, \
         patch("src.incident_analyzer.LLMEngine") as MockLLM, \
         patch("src.incident_analyzer.AirflowClient") as MockAC, \
         patch("src.incident_analyzer.GitHubRepoManager") as MockGH, \
         patch("src.incident_analyzer.LogParser") as MockLP:

        # EmbeddingModel
        mock_embed = MagicMock()
        mock_embed.get_dimension.return_value = 384
        MockEmbed.return_value = mock_embed

        # VectorStore
        mock_vs = MagicMock()
        MockVS.return_value = mock_vs

        # EmbeddingCache
        mock_cache = MagicMock()
        MockCache.return_value = mock_cache

        # CodeIndexer
        MockCI.return_value = MagicMock()

        # Retriever — returns sample results
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "file_path": "producer.py",
                "function_name": "send_event",
                "snippet": "def send_event(): ...",
                "score": 0.8,
                "line_numbers": {"start": 10, "end": 20},
                "boosted": False,
            },
            {
                "file_path": "utils.py",
                "function_name": "validate",
                "snippet": "def validate(): ...",
                "score": 0.6,
                "line_numbers": {"start": 1, "end": 5},
                "boosted": False,
            },
            {
                "file_path": "config.py",
                "function_name": "load",
                "snippet": "def load(): ...",
                "score": 0.4,
                "line_numbers": {"start": 1, "end": 3},
                "boosted": False,
            },
        ]
        MockRet.return_value = mock_retriever

        # ContextBuilder
        mock_cb = MagicMock()
        mock_cb.build_prompt.return_value = "Prompt text"
        MockCB.return_value = mock_cb

        # LLMEngine
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Incident Summary\nKafka timeout\n\n"
            "Root Cause\nBroker down\n\n"
            "Impacted Module\nevent_producer\n\n"
            "Failure Path\nDAG → task → timeout\n\n"
            "Suggested Fix\nRestart broker\n"
        )
        MockLLM.return_value = mock_llm

        # AirflowClient
        mock_ac = MagicMock()
        mock_ac.fetch_dag_logs.return_value = "KafkaTimeoutException: timed out"
        mock_ac.load_local_log.return_value = "KafkaTimeoutException: timed out"
        MockAC.return_value = mock_ac

        # LogParser
        mock_lp = MagicMock()
        mock_lp.parse.return_value = {
            "error_type": "KafkaTimeoutException",
            "severity": "High",
            "keywords": ["kafka", "timeout"],
            "source_module": "event_producer",
            "summary": "KafkaTimeoutException: timed out",
            "stack_trace": "Traceback...",
            "referenced_files": [{"file": "producer.py", "line": 42}],
            "timeline": [
                {"timestamp": "2026-03-08 12:03:21", "event": "DAG started"},
            ],
        }
        MockLP.return_value = mock_lp

        # GitHubRepoManager
        MockGH.return_value = MagicMock()

        analyzer = IncidentAnalyzer.__new__(IncidentAnalyzer)
        analyzer.airflow_client = mock_ac
        analyzer.log_parser = mock_lp
        analyzer.embedding_model = mock_embed
        analyzer.embedding_cache = mock_cache
        analyzer.vector_store = mock_vs
        analyzer.code_indexer = MockCI.return_value
        analyzer.retriever = mock_retriever
        analyzer.context_builder = mock_cb
        analyzer.llm_engine = mock_llm
        analyzer.github_manager = MockGH.return_value
        analyzer.repo_url = None
        analyzer.top_k = 5
        analyzer.index_path = "data/vector_index"
        analyzer.cache_path = "data/embedding_cache"
        analyzer._repo_dir = None

        return analyzer


# ---------------------------------------------------------------------------
# analyze_from_file
# ---------------------------------------------------------------------------

class TestAnalyzeFromFile:
    def test_returns_report_dict(self):
        analyzer = _make_analyzer()
        report = analyzer.analyze_from_file("fake_log.txt")
        assert isinstance(report, dict)
        assert "incident_summary" in report
        assert "root_cause" in report
        assert "severity" in report
        assert "confidence" in report
        assert "timeline" in report

    def test_pipeline_calls(self):
        analyzer = _make_analyzer()
        analyzer.analyze_from_file("fake_log.txt")
        analyzer.airflow_client.load_local_log.assert_called_once_with("fake_log.txt")
        analyzer.log_parser.parse.assert_called_once()
        analyzer.retriever.retrieve.assert_called_once()
        analyzer.context_builder.build_prompt.assert_called_once()
        analyzer.llm_engine.generate.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_from_airflow
# ---------------------------------------------------------------------------

class TestAnalyzeFromAirflow:
    def test_returns_report_dict(self):
        analyzer = _make_analyzer()
        report = analyzer.analyze_from_airflow("dag_id", "run_id", "task_id")
        assert isinstance(report, dict)
        assert "incident_summary" in report
        analyzer.airflow_client.fetch_dag_logs.assert_called_once_with(
            "dag_id", "run_id", "task_id"
        )


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_average_of_scores(self):
        analyzer = _make_analyzer()
        results = [{"score": 0.8}, {"score": 0.6}, {"score": 0.4}]
        confidence = analyzer._compute_confidence(results)
        assert confidence == 0.6

    def test_empty_results(self):
        analyzer = _make_analyzer()
        assert analyzer._compute_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestFormatTextReport:
    def test_contains_section_headers(self, sample_report):
        analyzer = _make_analyzer()
        text = analyzer.format_text_report(sample_report)
        assert "Incident Summary" in text
        assert "Root Cause" in text
        assert "Impacted Module" in text
        assert "Failure Path" in text
        assert "Suggested Fix" in text

    def test_confidence_in_report(self, sample_report):
        analyzer = _make_analyzer()
        text = analyzer.format_text_report(sample_report)
        assert "0.6" in text

    def test_timeline_in_report(self, sample_report):
        analyzer = _make_analyzer()
        text = analyzer.format_text_report(sample_report)
        assert "2026-03-08 12:03:21" in text
        assert "DAG started" in text


class TestFormatJsonReport:
    def test_valid_json(self, sample_report):
        analyzer = _make_analyzer()
        json_str = analyzer.format_json_report(sample_report)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["incident_summary"] == "Kafka producer timed out."

    def test_confidence_in_json(self, sample_report):
        analyzer = _make_analyzer()
        json_str = analyzer.format_json_report(sample_report)
        parsed = json.loads(json_str)
        assert parsed["confidence"] == 0.6
