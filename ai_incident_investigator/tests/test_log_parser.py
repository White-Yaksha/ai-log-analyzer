"""Tests for the LogParser module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.log_parser import LogParser


@pytest.fixture
def parser():
    return LogParser()


# ---------------------------------------------------------------------------
# Error type extraction
# ---------------------------------------------------------------------------

class TestErrorTypeExtraction:
    def test_kafka_timeout_exception(self, parser):
        log = (
            "2026-03-08 12:04:00 ERROR\n"
            "KafkaTimeoutException: Failed to send message to topic 'events'\n"
        )
        result = parser.parse(log)
        assert result["error_type"] == "KafkaTimeoutException"

    def test_no_exception_returns_unknown(self, parser):
        log = "INFO Starting up the application\nDEBUG Config loaded"
        result = parser.parse(log)
        assert result["error_type"] == "Unknown"


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_timeout_is_high(self, parser):
        log = "Connection timed out after 30s — timeout exhausted"
        result = parser.parse(log)
        assert result["severity"] == "High"

    def test_validation_error_is_medium(self, parser):
        log = "validation error: field 'email' is required"
        result = parser.parse(log)
        assert result["severity"] == "Medium"

    def test_warning_is_low(self, parser):
        log = "WARNING: disk usage above 80%"
        result = parser.parse(log)
        assert result["severity"] == "Low"


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_extracts_relevant_keywords(self, parser):
        log = "KafkaTimeoutException: Failed to send message to topic 'events'"
        result = parser.parse(log)
        keywords = result["keywords"]
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        lower_keywords = [k.lower() for k in keywords]
        assert any("kafka" in k for k in lower_keywords)


# ---------------------------------------------------------------------------
# Source module extraction
# ---------------------------------------------------------------------------

class TestSourceModule:
    def test_python_file_reference(self, parser):
        log = (
            'Traceback (most recent call last):\n'
            '  File "event_producer.py", line 42, in send_event\n'
            '    producer.send(topic, payload)\n'
            'KafkaTimeoutException: timed out\n'
        )
        result = parser.parse(log)
        assert result["source_module"] == "event_producer"


# ---------------------------------------------------------------------------
# Referenced files extraction
# ---------------------------------------------------------------------------

class TestReferencedFiles:
    def test_python_file_line(self, parser):
        log = (
            'Traceback (most recent call last):\n'
            '  File "handler.py", line 10, in process\n'
            '  File "utils.py", line 55, in validate\n'
            'ValueError: invalid\n'
        )
        result = parser.parse(log)
        refs = result["referenced_files"]
        assert any(r["file"] == "handler.py" and r["line"] == 10 for r in refs)
        assert any(r["file"] == "utils.py" and r["line"] == 55 for r in refs)

    def test_java_style_pattern(self, parser):
        log = (
            "Exception in thread \"main\"\n"
            "    at com.example.service.Handler.process(Handler.java:42)\n"
            "    at com.example.service.App.main(App.java:10)\n"
        )
        result = parser.parse(log)
        refs = result["referenced_files"]
        assert any(r["file"] == "Handler.java" and r["line"] == 42 for r in refs)
        assert any(r["file"] == "App.java" and r["line"] == 10 for r in refs)


# ---------------------------------------------------------------------------
# Timeline extraction
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_chronological_events(self, parser):
        log = (
            "2026-03-08 12:03:21 DAG started\n"
            "2026-03-08 12:03:25 Task initialised\n"
            "2026-03-08 12:04:00 KafkaTimeoutException: timed out\n"
        )
        result = parser.parse(log)
        timeline = result["timeline"]
        assert len(timeline) >= 3
        assert timeline[0]["timestamp"] == "2026-03-08 12:03:21"
        assert "DAG started" in timeline[0]["event"]
        # Verify chronological order
        timestamps = [e["timestamp"] for e in timeline]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_log(self, parser):
        result = parser.parse("")
        assert result["error_type"] == "Unknown"
        assert result["severity"] in ("High", "Medium", "Low")
        assert isinstance(result["keywords"], list)
        assert isinstance(result["referenced_files"], list)
        assert isinstance(result["timeline"], list)

    def test_log_no_exceptions(self, parser):
        log = "INFO Application started successfully\nDEBUG Connection pool ready"
        result = parser.parse(log)
        assert result["error_type"] == "Unknown"
