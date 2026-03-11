# AI Incident Investigator

## Overview

AI Incident Investigator is an AI-powered debugging assistant for distributed systems that analyzes Airflow DAG execution logs, retrieves relevant source code from GitHub repositories using Retrieval Augmented Generation (RAG), and leverages a local LLM to produce structured root cause analysis reports. The system ingests logs from Airflow's REST API or local files, parses them to extract errors, stack traces, and timelines, indexes the associated codebase into a FAISS vector store, and semantically retrieves the most relevant code snippets to provide as context to the language model. Key features include a hash-based embedding cache to avoid recomputation on unchanged files, stack trace priority boosting that elevates code files referenced in tracebacks, an incident timeline reconstructed from log timestamps, and confidence scores derived from retrieval similarity to quantify the reliability of each analysis.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Airflow Logs   │────▶│  Log Parser  │────▶│ Structured Query │
│  (REST API /    │     │              │     │ + Timeline       │
│   local file)   │     └──────────────┘     │ + Referenced     │
└─────────────────┘                          │   Files          │
                                             └────────┬─────────┘
                                                      │
┌─────────────────┐     ┌──────────────┐              ▼
│ GitHub Repo     │────▶│ Code Indexer │────▶┌──────────────────┐
│ (clone/pull)    │     │ (chunking +  │     │  FAISS Vector    │
└─────────────────┘     │  embedding)  │     │  Store           │
                        └──────────────┘     └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │    Retriever     │
                                             │ (semantic search │
                                             │ + priority boost)│
                                             └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │ Context Builder  │
                                             │ (prompt assembly)│
                                             └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │   LLM Engine     │
                                             │ (Phi-3/Llama/    │
                                             │  Mistral)        │
                                             └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │ Incident Report  │
                                             │ (text/JSON +     │
                                             │  confidence)     │
                                             └──────────────────┘
```

## RAG Pipeline

AI Incident Investigator uses a pure **Retrieval Augmented Generation (RAG)** approach — no model fine-tuning is required. The system works by dynamically retrieving relevant code context at query time and injecting it into the LLM prompt.

The pipeline consists of six stages:

1. **Log Ingestion and Parsing** — Raw logs are fetched from the Airflow REST API or read from a local file. The log parser extracts error types, stack traces, timestamps, referenced file paths, and keywords to form a structured query.

2. **Code Repository Indexing** — The target GitHub repository is cloned (or pulled if already present). Source files (`.py`, `.java`, `.sql`, `.yaml`, `.yml`) are scanned and chunked at function/class boundaries into segments of approximately 400 tokens with 50-token overlap. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS vector index. A SHA-256 hash-based embedding cache skips recomputation for unchanged files.

3. **Semantic Retrieval with Priority Boosting** — The structured error query is embedded and used to search the FAISS index via cosine similarity. Results referencing files that appear in the stack trace receive a **1.5× score boost**, ensuring the most directly relevant code surfaces first.

4. **Context Assembly with Timeline** — Retrieved code snippets and the parsed incident timeline are assembled into a structured prompt. The context budget allocates approximately 70% to code snippets and 30% to stack trace information, with a default maximum context length of 3,000 characters.

5. **LLM Reasoning with Confidence Scoring** — The assembled prompt is sent to the configured LLM (Phi-3, Llama-3, or Mistral). The model generates a root cause analysis. A confidence score (0.0–1.0) is computed as the mean of retrieval similarity scores, clamped and rounded to two decimal places.

6. **Structured Report Generation** — The LLM output is parsed into a structured report containing the incident summary, root cause, impacted module, failure path, suggested fix, severity, confidence score, timeline, and referenced code files. Output is available in both text and JSON formats.

## System Workflow

1. The user invokes the CLI with either Airflow coordinates (`--dag`, `--run`, `--task`) or a local log file (`--log-file`), along with the target repository URL.
2. If using Airflow mode, the **Airflow Client** authenticates and fetches the task instance log via the REST API.
3. The **Log Parser** processes the raw log text, extracting error messages, stack traces, timestamps, and referenced file names.
4. The **GitHub Repo Manager** clones (or pulls) the specified repository to a local working directory.
5. The **Code Indexer** scans the repository for supported source files, chunks them respecting function/class boundaries, computes embeddings (using the cache where possible), and upserts them into the **FAISS Vector Store**.
6. The **Retriever** takes the parsed error information, embeds it as a query, and performs a cosine similarity search against the vector store. Files matching the stack trace receive a 1.5× priority boost.
7. The **Context Builder** assembles the LLM prompt from the error information, incident timeline, stack trace, and the top-k retrieved code snippets.
8. The **LLM Engine** loads the configured model (with optional 4-bit quantization via bitsandbytes) and generates the analysis.
9. The **Incident Analyzer** orchestrator parses the LLM output, computes the confidence score, and produces the final structured report.

## Airflow Integration

The system connects to Apache Airflow via its REST API to fetch task execution logs programmatically.

**Endpoint format:**

```
GET {AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs
```

**Authentication:** HTTP Basic Auth using credentials supplied via environment variables (`AIRFLOW_USER` and `AIRFLOW_PASSWORD`) or CLI flags (`--airflow-user`, `--airflow-pass`).

**Local file fallback:** When Airflow is not available or the logs are already on disk, use the `--log-file` flag to point directly at a log file. The log parser handles both sources identically.

## GitHub Repository Indexing

- **Clone/Pull:** Managed via [GitPython](https://gitpython.readthedocs.io/). Repositories are cloned on first use and pulled on subsequent runs to stay up to date.
- **Supported file types:** `.py`, `.java`, `.sql`, `.yaml`, `.yml`
- **Chunking strategy:** Code is first split at function and class boundaries. Blocks exceeding the chunk size are sub-split by lines. A fallback line-based splitter with overlap handles edge cases.
  - **Chunk size:** ~400 tokens (whitespace-split)
  - **Chunk overlap:** 50 tokens
- **Embedding model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — a lightweight, fast model producing 384-dimensional normalized vectors.
- **Embedding cache:** Each file's content is hashed with SHA-256. If the hash matches a cached entry, the stored embedding is reused, avoiding expensive recomputation during re-indexing.
- **Vector storage:** [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity).

## Installation

```bash
git clone <repo-url>
cd ai_incident_investigator
pip install -r requirements.txt
```

> **Note:** PyTorch (`torch`) is required. Install the appropriate version for your hardware (CPU or CUDA) from [pytorch.org](https://pytorch.org/get-started/locally/) if the default installation does not match your environment.

## Configuration

The following environment variables can be used to configure the system:

| Variable           | Description                            | Required          |
|--------------------|----------------------------------------|-------------------|
| `AIRFLOW_URL`      | Airflow base URL (e.g., `http://localhost:8080`) | For Airflow mode  |
| `AIRFLOW_USER`     | Airflow username for Basic Auth        | For Airflow mode  |
| `AIRFLOW_PASSWORD`  | Airflow password for Basic Auth        | For Airflow mode  |
| `GITHUB_TOKEN`     | GitHub personal access token           | For private repos |

All values can alternatively be passed as CLI arguments (see [Usage](#usage)).

## Usage

### Analyze from Airflow

```bash
python cli/analyze_incident.py \
  --dag shipment_pipeline \
  --run scheduled__2026-03-08 \
  --task process_events \
  --repo https://github.com/org/repo
```

### Analyze from a local log file

```bash
python cli/analyze_incident.py \
  --log-file logs/failure.log \
  --repo https://github.com/org/repo
```

### Force re-index a repository

```bash
python cli/analyze_incident.py \
  --repo https://github.com/org/repo \
  --reindex
```

### JSON output format

```bash
python cli/analyze_incident.py \
  --log-file logs/failure.log \
  --repo https://github.com/org/repo \
  --output-format json
```

### Custom model and top-k retrieval

```bash
python cli/analyze_incident.py \
  --log-file logs/failure.log \
  --repo https://github.com/org/repo \
  --model meta-llama/Llama-3-8B \
  --top-k 10
```

### Sample Output Report

```
================================================================================
                         INCIDENT ANALYSIS REPORT
================================================================================

INCIDENT SUMMARY:
  The scheduled DAG 'shipment_pipeline' failed during the 'process_events'
  task due to a NullPointerException when parsing shipment event records
  with missing carrier_code fields.

ROOT CAUSE:
  The EventProcessor.parse_record() method assumes carrier_code is always
  present in the input payload. When upstream system 'logistics-gateway'
  sends events without this field, the code raises a NullPointerException
  at EventProcessor.java:142.

IMPACTED MODULE:
  com.org.pipeline.EventProcessor

FAILURE PATH:
  1. Upstream 'logistics-gateway' sent shipment event with null carrier_code
  2. Kafka consumer ingested the malformed event
  3. EventProcessor.parse_record() called carrier_code.strip() on null value
  4. NullPointerException propagated up, failing the Airflow task

SUGGESTED FIX:
  Add a null check for carrier_code in EventProcessor.parse_record() before
  calling string methods. Consider adding a default value or logging a
  warning for missing fields rather than failing the entire task.

SEVERITY: HIGH

CONFIDENCE: 0.82

TIMELINE:
  [2026-03-08 06:00:01] DAG 'shipment_pipeline' run started
  [2026-03-08 06:12:34] Task 'process_events' started
  [2026-03-08 06:12:47] ERROR NullPointerException at EventProcessor.java:142
  [2026-03-08 06:12:47] Task 'process_events' failed

RETRIEVED CODE REFERENCES:
  - EventProcessor.java :: parse_record
  - ShipmentSchema.py :: validate_event
  - pipeline_config.yaml :: carrier_settings

================================================================================
```

## Project Structure

```
ai_incident_investigator/
├── cli/
│   ├── __init__.py
│   └── analyze_incident.py       # CLI entry point (argparse interface)
├── src/
│   ├── __init__.py
│   ├── incident_analyzer.py      # Main orchestrator — coordinates full pipeline
│   ├── airflow_client.py         # Airflow REST API client (log fetching)
│   ├── log_parser.py             # Log parsing: errors, stack traces, timelines
│   ├── code_indexer.py           # Code chunking + embedding + FAISS indexing
│   ├── github_repo_manager.py    # Git clone/pull and source file discovery
│   ├── retriever.py              # Semantic search with stack trace priority boosting
│   ├── llm_engine.py             # HuggingFace LLM inference (4-bit quantization)
│   ├── embeddings.py             # SentenceTransformer embedding model wrapper
│   ├── embedding_cache.py        # SHA-256 hash-keyed embedding cache
│   ├── context_builder.py        # Prompt assembly from logs + retrieved code
│   └── vector_store.py           # FAISS IndexFlatIP vector store
├── data/
│   ├── embedding_cache/          # Persisted embedding cache files
│   └── vector_index/             # Persisted FAISS index files
├── tests/
│   └── __init__.py               # Test package (pytest)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Features

- **Embedding Cache** — SHA-256 hash-based cache that detects unchanged files and reuses their embeddings, dramatically speeding up re-indexing of large repositories.
- **Stack Trace Priority Boosting** — Code files explicitly referenced in stack traces receive a 1.5× retrieval score boost, ensuring the most directly relevant source code is surfaced first.
- **Incident Timeline** — Chronological reconstruction of events from log timestamps, providing temporal context to the LLM for more accurate root cause analysis.
- **Confidence Score** — A 0.0–1.0 score derived from the mean cosine similarity of retrieved code chunks, giving users a quantitative measure of how well the retrieved context supports the analysis.
- **Configurable LLM** — Supports multiple HuggingFace models out of the box, including Microsoft Phi-3-mini-4k-instruct (default), Meta Llama-3-8B, and Mistral-7B. Optional 4-bit quantization via bitsandbytes for reduced memory usage.
- **JSON and Text Output** — Reports can be generated in human-readable text or structured JSON for integration with alerting and ticketing systems.

## Dependencies

| Package                | Version    | Description                                              |
|------------------------|------------|----------------------------------------------------------|
| `requests`             | ≥ 2.31.0   | HTTP client for Airflow REST API communication           |
| `sentence-transformers`| ≥ 2.2.0    | Embedding model for encoding code and log text           |
| `faiss-cpu`            | ≥ 1.7.4    | Facebook AI Similarity Search for vector storage         |
| `transformers`         | ≥ 4.36.0   | HuggingFace model loading and tokenization               |
| `torch`                | ≥ 2.1.0    | PyTorch deep learning framework (model inference)        |
| `gitpython`            | ≥ 3.1.40   | Git repository clone/pull operations                     |
| `pyyaml`               | ≥ 6.0      | YAML configuration file parsing                          |
| `bitsandbytes`         | ≥ 0.41.0   | 4-bit model quantization for reduced memory usage        |
| `pytest`               | ≥ 7.4.0    | Test framework                                           |

## Running Tests

```bash
pytest tests/ -v
```

## Design Principles

- **Modular architecture** — Each component (log parsing, code indexing, retrieval, LLM inference) is a standalone module that can be developed, tested, and replaced independently.
- **Clear separation of responsibilities** — The orchestrator (`incident_analyzer.py`) coordinates the pipeline without implementing business logic; each module owns its domain.
- **Production-style project structure** — Organized into `cli/`, `src/`, `data/`, and `tests/` packages following Python best practices.
- **Well-documented components** — Modules include docstrings and type hints for maintainability.
- **Minimal external dependencies** — Only essential, well-maintained libraries are used; no heavy frameworks beyond PyTorch/HuggingFace.
- **Local-first** — The entire system runs on a developer's machine with no cloud services required; models run locally via HuggingFace Transformers.
- **No model fine-tuning** — Pure RAG approach means any compatible HuggingFace causal language model can be swapped in without retraining, keeping the system flexible and easy to maintain.
