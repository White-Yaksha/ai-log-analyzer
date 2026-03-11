"""GitHub repository manager for cloning, updating, and scanning repositories."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from git import GitCommandError, InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = [".py", ".java", ".sql", ".yaml", ".yml"]
SKIP_DIRS = {"__pycache__", "node_modules", ".git", "venv", ".venv"}


class GitHubRepoManager:
    """Manage local clones of GitHub repositories.

    Handles cloning with optional token-based authentication, pulling
    latest changes, scanning for source files, and reading file content
    with encoding fallback.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the repo manager.

        Args:
            token: GitHub personal access token. Falls back to the
                ``GITHUB_TOKEN`` environment variable when not provided.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            logger.warning(
                "No GitHub token provided. Private repositories will not be accessible."
            )

    def _inject_token(self, repo_url: str) -> str:
        """Inject the authentication token into an HTTPS repository URL.

        Args:
            repo_url: The original repository URL.

        Returns:
            The URL with the token embedded for authentication, or the
            original URL if it is not HTTPS or no token is available.
        """
        if not self.token:
            return repo_url
        # Only inject into HTTPS URLs
        match = re.match(r"^(https?://)(github\.com/.+)$", repo_url)
        if match:
            return f"{match.group(1)}{self.token}@{match.group(2)}"
        return repo_url

    def clone_repo(self, repo_url: str, target_dir: str) -> str:
        """Clone a repository into *target_dir*.

        If *target_dir* already exists and contains a valid Git repository,
        :meth:`pull_latest` is called instead of cloning again.

        Args:
            repo_url: HTTPS URL of the GitHub repository.
            target_dir: Local directory path for the clone.

        Returns:
            Absolute path to the cloned (or existing) repository.

        Raises:
            GitCommandError: If the clone command fails (e.g. network error,
                invalid URL, or authentication failure).
        """
        target = Path(target_dir).resolve()

        if target.exists():
            try:
                Repo(str(target))
                logger.info("Repository already exists at %s — pulling latest.", target)
                self.pull_latest(str(target))
                return str(target)
            except InvalidGitRepositoryError:
                logger.error(
                    "Directory %s exists but is not a valid Git repository.", target
                )
                raise

        auth_url = self._inject_token(repo_url)
        logger.info("Cloning %s into %s", repo_url, target)
        try:
            Repo.clone_from(auth_url, str(target))
        except GitCommandError:
            logger.error("Failed to clone repository from %s", repo_url)
            raise
        logger.info("Repository cloned successfully to %s", target)
        return str(target)

    def pull_latest(self, repo_dir: str) -> None:
        """Pull the latest changes on the current branch.

        Merge conflicts are logged as a warning and do not raise an
        exception so that callers can continue working with the local
        copy.

        Args:
            repo_dir: Path to a local Git repository.

        Raises:
            InvalidGitRepositoryError: If *repo_dir* is not a valid Git
                repository.
        """
        repo_path = Path(repo_dir).resolve()
        try:
            repo = Repo(str(repo_path))
        except InvalidGitRepositoryError:
            logger.error("%s is not a valid Git repository.", repo_path)
            raise

        if repo.head.is_detached:
            logger.warning("HEAD is detached at %s — skipping pull.", repo_path)
            return

        branch = repo.active_branch.name
        logger.info("Pulling latest changes on branch '%s' in %s", branch, repo_path)
        try:
            origin = repo.remotes.origin
            origin.pull()
            logger.info("Pull completed successfully.")
        except GitCommandError as exc:
            if "conflict" in str(exc).lower() or "merge" in str(exc).lower():
                logger.warning(
                    "Merge conflict encountered while pulling in %s. "
                    "Continuing with the current local state.",
                    repo_path,
                )
            else:
                logger.error("Git pull failed in %s: %s", repo_path, exc)
                raise

    def scan_files(
        self,
        repo_dir: str,
        extensions: Optional[list[str]] = None,
    ) -> list[str]:
        """Recursively scan a directory for source files.

        Hidden directories (names starting with ``.``) and common
        non-code directories (``__pycache__``, ``node_modules``, etc.)
        are automatically skipped.

        Args:
            repo_dir: Root directory to scan.
            extensions: File extensions to include (e.g. ``[".py", ".java"]``).
                Defaults to ``.py``, ``.java``, ``.sql``, ``.yaml``, and
                ``.yml``.

        Returns:
            Sorted list of absolute file paths matching the given
            extensions.
        """
        if extensions is None:
            extensions = DEFAULT_EXTENSIONS
        ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}

        root = Path(repo_dir).resolve()
        if not root.is_dir():
            logger.error("Scan path does not exist or is not a directory: %s", root)
            return []

        matched: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune hidden and non-code directories in-place
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and d not in SKIP_DIRS
            ]
            for filename in filenames:
                if Path(filename).suffix in ext_set:
                    matched.append(str(Path(dirpath, filename).resolve()))

        matched.sort()
        return matched

    def read_file(self, file_path: str) -> str:
        """Read a file's content as a string.

        Tries UTF-8 first; falls back to latin-1 on decode errors so
        that binary-ish files can still be returned as text.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            The file content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read for other OS-level reasons.
        """
        path = Path(file_path)
        if not path.is_file():
            logger.error("File not found: %s", path)
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for %s — falling back to latin-1.", path
            )
            content = path.read_text(encoding="latin-1")
        return content
