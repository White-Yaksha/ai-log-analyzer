"""Tests for the GitHubRepoManager module."""

import os
import sys
import tempfile

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.github_repo_manager import GitHubRepoManager


@pytest.fixture
def manager():
    return GitHubRepoManager(token="ghp_test_token_123")


@pytest.fixture
def manager_no_token():
    return GitHubRepoManager(token=None)


# ---------------------------------------------------------------------------
# scan_files
# ---------------------------------------------------------------------------

class TestScanFiles:
    def test_filters_by_extension(self, manager):
        """Verify only matching extensions are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for name in ["app.py", "util.java", "readme.txt", "config.yaml", "data.csv"]:
                open(os.path.join(tmpdir, name), "w").close()

            results = manager.scan_files(tmpdir, extensions=[".py", ".java", ".yaml"])
            basenames = [os.path.basename(p) for p in results]
            assert "app.py" in basenames
            assert "util.java" in basenames
            assert "config.yaml" in basenames
            assert "readme.txt" not in basenames
            assert "data.csv" not in basenames

    def test_skips_hidden_dirs_and_pycache(self, manager):
        """Verify hidden directories and __pycache__ are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a hidden directory
            hidden_dir = os.path.join(tmpdir, ".hidden")
            os.makedirs(hidden_dir)
            open(os.path.join(hidden_dir, "secret.py"), "w").close()

            # Create __pycache__
            pycache_dir = os.path.join(tmpdir, "__pycache__")
            os.makedirs(pycache_dir)
            open(os.path.join(pycache_dir, "cached.py"), "w").close()

            # Create a normal file
            open(os.path.join(tmpdir, "main.py"), "w").close()

            results = manager.scan_files(tmpdir, extensions=[".py"])
            basenames = [os.path.basename(p) for p in results]
            assert "main.py" in basenames
            assert "secret.py" not in basenames
            assert "cached.py" not in basenames


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_content(self, manager):
        """Create temp file, read it, verify content."""
        content = "def hello():\n    print('world')\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = manager.read_file(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_file_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.read_file("/nonexistent/file.py")


# ---------------------------------------------------------------------------
# clone_repo
# ---------------------------------------------------------------------------

class TestCloneRepo:
    @patch("src.github_repo_manager.Repo")
    def test_clone_from_called(self, MockRepo, manager):
        """Verify Repo.clone_from is called with correct args."""
        target = os.path.join(tempfile.gettempdir(), "test_clone_xyz_nonexistent")
        # Ensure target does not exist
        if os.path.exists(target):
            os.rmdir(target)

        url = "https://github.com/org/repo.git"
        expected_auth_url = f"https://ghp_test_token_123@github.com/org/repo.git"

        manager.clone_repo(url, target)

        MockRepo.clone_from.assert_called_once_with(
            expected_auth_url, str(os.path.abspath(target))
        )


# ---------------------------------------------------------------------------
# pull_latest
# ---------------------------------------------------------------------------

class TestPullLatest:
    @patch("src.github_repo_manager.Repo")
    def test_pull_is_called(self, MockRepo, manager):
        """Verify pull is called on origin remote."""
        mock_repo = MagicMock()
        mock_repo.head.is_detached = False
        mock_repo.active_branch.name = "main"
        mock_origin = MagicMock()
        mock_repo.remotes.origin = mock_origin
        MockRepo.return_value = mock_repo

        manager.pull_latest("/some/repo/dir")

        mock_origin.pull.assert_called_once()


# ---------------------------------------------------------------------------
# Token injection
# ---------------------------------------------------------------------------

class TestTokenInjection:
    def test_injects_token_into_https_url(self, manager):
        url = "https://github.com/org/repo.git"
        result = manager._inject_token(url)
        assert result == "https://ghp_test_token_123@github.com/org/repo.git"

    def test_no_token_returns_original(self, manager_no_token):
        url = "https://github.com/org/repo.git"
        result = manager_no_token._inject_token(url)
        assert result == url
