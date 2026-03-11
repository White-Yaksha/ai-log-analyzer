"""Tests for the AirflowClient module."""

import os
import sys
import tempfile

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.airflow_client import AirflowClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Return an AirflowClient configured for testing."""
    return AirflowClient(
        base_url="http://airflow.example.com",
        username="admin",
        password="secret",
    )


# ---------------------------------------------------------------------------
# fetch_dag_logs
# ---------------------------------------------------------------------------

class TestFetchDagLogs:
    def test_url_construction_and_auth(self, client):
        """Verify the correct URL is built and auth header is present."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.ok = True
        fake_response.text = "log line 1\nlog line 2"

        with patch.object(client._session, "request", return_value=fake_response) as mock_req:
            result = client.fetch_dag_logs("my_dag", "run_123", "task_abc", try_number=2)

        expected_url = (
            "http://airflow.example.com/api/v1/dags/my_dag/dagRuns/run_123"
            "/taskInstances/task_abc/logs/2"
        )
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == expected_url
        assert result == "log line 1\nlog line 2"

    def test_auth_is_set_on_session(self, client):
        """Verify Basic Auth tuple is set on the internal session."""
        assert client._session.auth == ("admin", "secret")


# ---------------------------------------------------------------------------
# fetch_task_instances
# ---------------------------------------------------------------------------

class TestFetchTaskInstances:
    def test_parses_json_response(self, client):
        """Mock a JSON response and verify parsing."""
        payload = {
            "task_instances": [
                {"task_id": "t1", "state": "success"},
                {"task_id": "t2", "state": "failed"},
            ]
        }
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.ok = True
        fake_response.json.return_value = payload

        with patch.object(client._session, "request", return_value=fake_response):
            result = client.fetch_task_instances("dag_x", "run_y")

        assert len(result) == 2
        assert result[0]["task_id"] == "t1"
        assert result[1]["state"] == "failed"


# ---------------------------------------------------------------------------
# load_local_log
# ---------------------------------------------------------------------------

class TestLoadLocalLog:
    def test_reads_temp_file(self, client):
        """Create a temp file, read it via load_local_log, verify content."""
        content = "ERROR: something broke\nTraceback ..."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            result = client.load_local_log(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_file_not_found(self, client):
        """Verify FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            client.load_local_log("/nonexistent/path/log.txt")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_401_raises_value_error(self, client):
        """HTTP 401 should raise ValueError."""
        fake_response = MagicMock()
        fake_response.status_code = 401
        fake_response.ok = False
        fake_response.text = "Unauthorized"

        with patch.object(client._session, "request", return_value=fake_response):
            with pytest.raises(ValueError, match="Authentication failed"):
                client.fetch_dag_logs("d", "r", "t")

    def test_connection_error_raises(self, client):
        """Network-level ConnectionError should propagate."""
        import requests

        with patch.object(
            client._session, "request", side_effect=requests.ConnectionError("refused")
        ):
            with pytest.raises(ConnectionError):
                client.fetch_dag_logs("d", "r", "t")

    def test_500_raises_runtime_error(self, client):
        """HTTP 500 should raise RuntimeError."""
        fake_response = MagicMock()
        fake_response.status_code = 500
        fake_response.ok = False
        fake_response.text = "Internal Server Error"

        with patch.object(client._session, "request", return_value=fake_response):
            with pytest.raises(RuntimeError, match="Airflow API error"):
                client.fetch_dag_logs("d", "r", "t")
