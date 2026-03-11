"""Airflow REST API client for fetching DAG logs and task instance metadata."""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class AirflowClient:
    """Client for interacting with the Apache Airflow REST API.

    Supports fetching DAG run logs, task instance metadata, and reading
    local log files. Authentication is handled via HTTP Basic Auth.

    Args:
        base_url: Base URL of the Airflow webserver (e.g. ``http://localhost:8080``).
            Falls back to the ``AIRFLOW_URL`` environment variable if not provided.
        username: Airflow username for basic auth.
            Falls back to the ``AIRFLOW_USER`` environment variable if not provided.
        password: Airflow password for basic auth.
            Falls back to the ``AIRFLOW_PASSWORD`` environment variable if not provided.
    """

    def __init__(
        self,
        base_url: str = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AIRFLOW_URL", "")).rstrip("/")
        self.username = username or os.environ.get("AIRFLOW_USER")
        self.password = password or os.environ.get("AIRFLOW_PASSWORD")

        if not self.base_url:
            logger.warning(
                "No Airflow base URL provided and AIRFLOW_URL env var is not set."
            )

        self._session = requests.Session()
        if self.username and self.password:
            self._session.auth = (self.username, self.password)
        self._session.headers.update({"Accept": "application/json"})

    # -- public API -----------------------------------------------------------

    def fetch_dag_logs(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        try_number: int = 1,
    ) -> str:
        """Fetch the log content for a specific task instance try.

        Args:
            dag_id: The DAG identifier.
            run_id: The DAG run identifier.
            task_id: The task identifier within the DAG.
            try_number: The attempt number (default ``1``).

        Returns:
            The raw log text for the requested task instance try.

        Raises:
            ConnectionError: If the Airflow webserver is unreachable.
            ValueError: If authentication fails (HTTP 401/403).
            RuntimeError: For any other HTTP error response.
        """
        url = (
            f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
            f"/taskInstances/{task_id}/logs/{try_number}"
        )
        response = self._request("GET", url, headers={"Accept": "text/plain"})
        return response.text

    def fetch_task_instances(self, dag_id: str, run_id: str) -> list[dict]:
        """Fetch all task instances for a given DAG run.

        Args:
            dag_id: The DAG identifier.
            run_id: The DAG run identifier.

        Returns:
            A list of task instance dictionaries as returned by the Airflow API.

        Raises:
            ConnectionError: If the Airflow webserver is unreachable.
            ValueError: If authentication fails (HTTP 401/403).
            RuntimeError: For any other HTTP error response.
        """
        url = (
            f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        )
        response = self._request("GET", url)
        data = response.json()
        return data.get("task_instances", [])

    def load_local_log(self, file_path: str) -> str:
        """Read a local log file and return its contents.

        Args:
            file_path: Path to the log file on disk.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read for other I/O reasons.
        """
        logger.debug("Reading local log file: %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except FileNotFoundError:
            logger.error("Log file not found: %s", file_path)
            raise
        except OSError as exc:
            logger.error("Failed to read log file %s: %s", file_path, exc)
            raise

    # -- internals ------------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Execute an HTTP request with unified error handling.

        Args:
            method: HTTP method (e.g. ``GET``).
            url: Fully-qualified request URL.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`requests.Session.request`.

        Returns:
            The :class:`requests.Response` object on success.

        Raises:
            ConnectionError: If a network-level error occurs.
            ValueError: If the server returns 401 or 403.
            RuntimeError: For any other non-2xx status code.
        """
        logger.debug("%s %s", method, url)
        try:
            response = self._session.request(method, url, **kwargs)
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Unable to connect to Airflow at {url}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Request to Airflow failed ({url}): {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise ValueError(
                f"Authentication failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

        if not response.ok:
            raise RuntimeError(
                f"Airflow API error (HTTP {response.status_code}) "
                f"for {method} {url}: {response.text}"
            )

        return response
