"""Tests for the ContextBuilder module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.context_builder import ContextBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder():
    return ContextBuilder(max_context_length=5000)


@pytest.fixture
def parsed_log():
    return {
        "error_type": "KafkaTimeoutException",
        "severity": "High",
        "keywords": ["kafka", "timeout", "producer"],
        "source_module": "event_producer",
        "summary": "KafkaTimeoutException: Failed to send message",
        "stack_trace": (
            "Traceback (most recent call last):\n"
            '  File "event_producer.py", line 42\n'
            "KafkaTimeoutException: timed out"
        ),
        "timeline": [
            {"timestamp": "2026-03-08 12:03:21", "event": "DAG started"},
            {"timestamp": "2026-03-08 12:03:25", "event": "Task initialised"},
            {"timestamp": "2026-03-08 12:04:00", "event": "Exception raised"},
        ],
        "referenced_files": [
            {"file": "event_producer.py", "line": 42},
        ],
    }


@pytest.fixture
def code_snippets():
    return [
        {
            "file_path": "event_producer.py",
            "function_name": "send_event",
            "line_numbers": {"start": 40, "end": 50},
            "snippet": "def send_event():\n    producer.send(topic, payload)",
            "score": 0.85,
            "boosted": True,
        },
        {
            "file_path": "utils.py",
            "function_name": "validate",
            "line_numbers": {"start": 1, "end": 10},
            "snippet": "def validate(data):\n    pass",
            "score": 0.5,
            "boosted": False,
        },
    ]


# ---------------------------------------------------------------------------
# build_prompt — full prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_all_sections(self, builder, parsed_log, code_snippets):
        prompt = builder.build_prompt(parsed_log, code_snippets)
        assert "Error Information" in prompt
        assert "KafkaTimeoutException" in prompt
        assert "High" in prompt
        assert "Relevant Code" in prompt
        assert "Instructions" in prompt

    def test_contains_stack_trace(self, builder, parsed_log, code_snippets):
        prompt = builder.build_prompt(parsed_log, code_snippets)
        assert "Stack Trace" in prompt
        assert "event_producer.py" in prompt


# ---------------------------------------------------------------------------
# Timeline formatting
# ---------------------------------------------------------------------------

class TestTimelineFormatting:
    def test_timestamps_appear(self, builder, parsed_log, code_snippets):
        prompt = builder.build_prompt(parsed_log, code_snippets)
        assert "2026-03-08 12:03:21" in prompt
        assert "DAG started" in prompt
        assert "Incident Timeline" in prompt

    def test_empty_timeline_omitted(self, builder, code_snippets):
        log = {
            "error_type": "SomeError",
            "severity": "Low",
            "keywords": [],
            "source_module": "",
            "summary": "SomeError occurred",
            "stack_trace": "",
            "timeline": [],
        }
        prompt = builder.build_prompt(log, code_snippets)
        assert "Incident Timeline" not in prompt


# ---------------------------------------------------------------------------
# Boosted snippet marking
# ---------------------------------------------------------------------------

class TestBoostedSnippetMarking:
    def test_high_priority_tag(self, builder, parsed_log, code_snippets):
        prompt = builder.build_prompt(parsed_log, code_snippets)
        assert "[HIGH PRIORITY" in prompt


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_output_respects_max_length(self, parsed_log, code_snippets):
        small_builder = ContextBuilder(max_context_length=300)
        prompt = small_builder.build_prompt(parsed_log, code_snippets)
        assert len(prompt) <= 300


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_snippets_produces_valid_prompt(self, builder, parsed_log):
        prompt = builder.build_prompt(parsed_log, [])
        assert "Error Information" in prompt
        assert "no relevant code snippets found" in prompt

    def test_empty_timeline_and_stack_trace(self, builder):
        log = {
            "error_type": "Unknown",
            "severity": "Low",
            "keywords": [],
            "source_module": "",
            "summary": "N/A",
            "stack_trace": "",
            "timeline": [],
        }
        prompt = builder.build_prompt(log, [])
        assert "Error Information" in prompt
        assert len(prompt) > 0
