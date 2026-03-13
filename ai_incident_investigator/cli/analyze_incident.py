#!/usr/bin/env python3
"""CLI entry point for the AI Incident Investigator.

Provides a command-line interface to analyze Airflow task failures or local
log files using RAG-based incident analysis powered by an LLM and a code
vector index.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Make sure the project root (one level above cli/) is on sys.path so that
# ``from src.incident_analyzer import IncidentAnalyzer`` resolves correctly.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all
        supported options and validation rules described in the help text.
    """
    parser = argparse.ArgumentParser(
        description="AI Incident Investigator – analyse Airflow task failures "
        "or local log files with RAG-based root-cause analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python analyze_incident.py --dag shipment_pipeline --run scheduled__2026-03-08 --task process_events
              python analyze_incident.py --log-file logs/failure.log --repo https://github.com/org/repo
              python analyze_incident.py --repo https://github.com/org/repo --reindex
            """
        ),
    )

    # -- Airflow source --
    airflow_group = parser.add_argument_group("Airflow log source")
    airflow_group.add_argument("--dag", type=str, default=None, help="DAG ID (for Airflow mode)")
    airflow_group.add_argument("--run", type=str, default=None, help="DAG run ID (for Airflow mode)")
    airflow_group.add_argument("--task", type=str, default=None, help="Task ID (for Airflow mode)")

    # -- Local file source --
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a local log file (alternative to Airflow mode)",
    )

    # -- Repository / indexing --
    repo_group = parser.add_argument_group("Repository indexing")
    repo_group.add_argument("--repo", type=str, default=None, help="GitHub repo URL to index")
    repo_group.add_argument(
        "--reindex",
        action="store_true",
        default=False,
        help="Force re-index the repository (requires --repo)",
    )

    # -- Model / retrieval --
    model_group = parser.add_argument_group("Model and retrieval")
    model_group.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of code snippets to retrieve (default: 5)",
    )
    model_group.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help='LLM model name (default: "microsoft/Phi-3-mini-4k-instruct")',
    )

    # -- Output --
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help='Output format: "text" or "json" (default: "text")',
    )

    # -- Airflow connection --
    airflow_conn = parser.add_argument_group("Airflow connection")
    airflow_conn.add_argument("--airflow-url", type=str, default=None, help="Airflow base URL")
    airflow_conn.add_argument("--airflow-user", type=str, default=None, help="Airflow username")
    airflow_conn.add_argument("--airflow-pass", type=str, default=None, help="Airflow password")

    # -- Paths --
    paths_group = parser.add_argument_group("Storage paths")
    paths_group.add_argument(
        "--index-path",
        type=str,
        default="data/vector_index",
        help='Path to vector index directory (default: "data/vector_index")',
    )
    paths_group.add_argument(
        "--cache-path",
        type=str,
        default="data/embedding_cache",
        help='Path to embedding cache directory (default: "data/embedding_cache")',
    )

    # -- Verbosity --
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Set logging level to DEBUG",
    )

    # -- Config bootstrapping --
    parser.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Create a sample config file at ~/.ai-incident-investigator/config.yaml and exit",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate mutually-exclusive and co-dependent argument constraints.

    Args:
        args: Parsed CLI arguments.

    Raises:
        SystemExit: If validation fails, prints an error and exits with
            code 1.
    """
    airflow_args = (args.dag, args.run, args.task)
    has_airflow = any(a is not None for a in airflow_args)
    has_all_airflow = all(a is not None for a in airflow_args)
    has_log_file = args.log_file is not None

    if has_airflow and has_log_file:
        print(
            "Error: --dag/--run/--task and --log-file are mutually exclusive. "
            "Provide either Airflow coordinates or a local log file, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    if has_airflow and not has_all_airflow:
        print(
            "Error: Airflow mode requires all three arguments: --dag, --run, and --task.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not has_airflow and not has_log_file and not args.reindex:
        print(
            "Error: You must provide either (--dag, --run, --task) or --log-file "
            "to run an analysis, or use --reindex with --repo to re-index a repository.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.reindex and args.repo is None:
        print(
            "Error: --reindex requires --repo to specify which repository to index.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Parse arguments, run incident analysis, and print the report."""
    parser = build_parser()
    args = parser.parse_args()

    # -- Configure logging --
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- Bootstrap config file --
    if args.init_config:
        from src.config import get_config_path, _bootstrap_config

        config_path = get_config_path()
        if config_path.exists():
            print(f"Config file already exists at {config_path}")
        else:
            _bootstrap_config()
            print(f"Sample config created at {config_path}")
            print("Edit the file and fill in your credentials.")
        return

    validate_args(args)

    try:
        # Defer heavy imports (torch, transformers, etc.) to avoid slow startup
        # for lightweight commands like --init-config.
        from src.incident_analyzer import IncidentAnalyzer

        # -- Instantiate the analyser --
        analyzer = IncidentAnalyzer(
            airflow_url=args.airflow_url,
            airflow_user=args.airflow_user,
            airflow_pass=args.airflow_pass,
            repo_url=args.repo,
            github_token=os.environ.get("GITHUB_TOKEN"),  # also resolved via config file in GitHubRepoManager
            model_name=args.model,
            top_k=args.top_k,
            index_path=args.index_path,
            cache_path=args.cache_path,
        )

        # -- Index repository if requested --
        if args.repo or args.reindex:
            logger.info("Indexing repository: %s (force=%s)", args.repo, args.reindex)
            num_indexed = analyzer.index_repo(args.repo, force=args.reindex)
            logger.info("Indexed %d files", num_indexed)

            # If the user only wanted to re-index, stop here.
            if args.log_file is None and args.dag is None:
                print(f"Repository indexed successfully ({num_indexed} files).")
                return

        # -- Run analysis --
        if args.dag is not None:
            logger.info(
                "Analysing Airflow task: dag=%s, run=%s, task=%s",
                args.dag,
                args.run,
                args.task,
            )
            report = analyzer.analyze_from_airflow(args.dag, args.run, args.task)
        else:
            logger.info("Analysing log file: %s", args.log_file)
            report = analyzer.analyze_from_file(args.log_file)

        # -- Format and print --
        if args.output_format == "json":
            output = analyzer.format_json_report(report)
        else:
            output = analyzer.format_text_report(report)

        print(output)

    except FileNotFoundError as exc:
        logger.debug("FileNotFoundError: %s", exc, exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        logger.debug("ConnectionError: %s", exc, exc_info=True)
        print(f"Error: Could not connect – {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unexpected error: %s", exc, exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
