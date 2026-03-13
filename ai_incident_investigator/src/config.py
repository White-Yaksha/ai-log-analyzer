"""Configuration loader for the AI Incident Investigator.

Loads settings from a YAML config file located at
``~/.ai-incident-investigator/config.yaml``.  Values in the config file
serve as defaults that can still be overridden by environment variables
or explicit CLI arguments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".ai-incident-investigator"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"

# Cached config dict (loaded once per process).
_config: Optional[dict[str, Any]] = None


def _load_config() -> dict[str, Any]:
    """Read and cache the YAML config file.

    Returns an empty dict when the file does not exist or cannot be parsed.
    """
    global _config
    if _config is not None:
        return _config

    if not _CONFIG_FILE.is_file():
        _bootstrap_config()
        _config = {}
        return _config

    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as fh:
            _config = yaml.safe_load(fh) or {}
            logger.debug("Loaded config from %s", _CONFIG_FILE)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to parse config file %s – using defaults.", _CONFIG_FILE, exc_info=True)
        _config = {}

    return _config


def get(key: str, *, section: Optional[str] = None, env_var: Optional[str] = None) -> Optional[str]:
    """Retrieve a configuration value.

    Resolution order (first non-``None`` wins):
        1. Environment variable (``env_var``)
        2. Config file value (``section.key`` or top-level ``key``)

    Args:
        key: The config key to look up.
        section: Optional YAML section (e.g. ``"github"``, ``"airflow"``).
        env_var: Environment variable name to check first.

    Returns:
        The resolved value as a string, or ``None`` if not set anywhere.
    """
    # 1. Environment variable takes precedence.
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    # 2. Config file.
    cfg = _load_config()
    if section:
        value = cfg.get(section, {}).get(key)
    else:
        value = cfg.get(key)

    return str(value) if value else None


def _bootstrap_config() -> None:
    """Create the config directory and a sample config file on first run."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(generate_sample_config(), encoding="utf-8")
        logger.info("Created sample config at %s – edit it to add your credentials.", _CONFIG_FILE)
    except OSError:
        logger.debug("Could not create config file at %s – skipping.", _CONFIG_FILE, exc_info=True)


def get_config_path() -> Path:
    """Return the path to the config file (for user-facing messages)."""
    return _CONFIG_FILE


def generate_sample_config() -> str:
    """Return a sample YAML config string for documentation / bootstrapping."""
    return """\
# AI Incident Investigator – configuration
# Place this file at: ~/.ai-incident-investigator/config.yaml

github:
  # Generate a token at: https://github.com/settings/tokens
  # Required scopes: repo (Full control of private repositories)
  token: ""

airflow:
  url: ""
  user: ""
  password: ""
"""
