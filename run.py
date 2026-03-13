#!/usr/bin/env python3
"""Root-level entry point — delegates to the CLI module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ai_incident_investigator"))

from cli.analyze_incident import main  # noqa: E402

if __name__ == "__main__":
    main()
