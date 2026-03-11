# AI Incident Investigator

An AI-powered debugging assistant for distributed systems. It analyzes Airflow DAG execution logs, retrieves relevant source code from GitHub repositories using **Retrieval Augmented Generation (RAG)**, and leverages a local LLM to produce structured root cause analysis reports — all running on your local machine with no cloud dependencies.

**Key Highlights:**
- 🔍 Automatic log parsing with error type detection and severity classification
- 🧠 RAG-based code retrieval using FAISS vector search (no fine-tuning needed)
- ⚡ Embedding cache to skip recomputation on unchanged files
- 📊 Stack trace priority boosting for more accurate results
- 📅 Incident timeline extraction from log timestamps
- 🎯 Confidence scoring (0.0–1.0) for every analysis
- 📄 Text and JSON output formats

---

## Table of Contents

- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Step-by-Step: Using with Your Organization Project](#step-by-step-using-with-your-organization-project)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Usage Examples](#usage-examples)
- [Sample Output](#sample-output)
- [Architecture](#architecture)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [System Workflow](#system-workflow)
- [Airflow Integration](#airflow-integration)
- [GitHub Repository Indexing](#github-repository-indexing)
- [Key Features Explained](#key-features-explained)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Design Principles](#design-principles)

---

## Quick Start (5 Minutes)

```bash
# 1. Clone this repo
git clone https://github.com/white-yaksha/ai-log-analyzer.git
cd ai-log-analyzer/ai_incident_investigator

# 2. Create a virtual environment (recommended)
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
# source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Index your org's GitHub repository
python cli/analyze_incident.py \
  --repo https://github.com/your-org/your-project \
  --reindex

# 5. Analyze a failure log
python cli/analyze_incident.py \
  --log-file path/to/your/failure.log \
  --repo https://github.com/your-org/your-project
```

That's it! You'll get a root cause analysis report in your terminal.

---

## Step-by-Step: Using with Your Organization Project

This section walks through a complete real-world scenario — investigating a failed Airflow DAG in your org's codebase.

### Step 1: Install the Tool

```bash
# Clone the AI Incident Investigator
git clone https://github.com/white-yaksha/ai-log-analyzer.git
cd ai-log-analyzer/ai_incident_investigator

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt
```

**What gets installed:**

```bash
pip install requests              # HTTP client for Airflow API
pip install sentence-transformers # Embedding model (downloads ~80MB model on first run)
pip install faiss-cpu             # Vector search engine
pip install transformers          # LLM model loading
pip install torch                 # PyTorch (ML backend)
pip install gitpython             # Git repo cloning
pip install pyyaml                # YAML parsing
pip install bitsandbytes          # 4-bit model quantization (optional, saves RAM)
pip install pytest                # Testing framework
```

> **Note:** The `pip install -r requirements.txt` command installs all of these at once. You don't need to run them individually.

> **PyTorch:** If you have a CUDA-capable GPU and want faster LLM inference, install the GPU version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) instead of the default CPU version.

### Step 2: Configure Credentials

Set environment variables for your org's services:

```bash
# ── For private GitHub repos ──
# Generate a token at: https://github.com/settings/tokens
# Required scopes: repo (Full control of private repositories)

# Windows (PowerShell):
$env:GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# macOS/Linux:
# export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ── For Airflow log fetching (skip if using local log files) ──
# Windows (PowerShell):
$env:AIRFLOW_URL="https://airflow.your-company.com"
$env:AIRFLOW_USER="your_username"
$env:AIRFLOW_PASSWORD="your_password"

# macOS/Linux:
# export AIRFLOW_URL="https://airflow.your-company.com"
# export AIRFLOW_USER="your_username"
# export AIRFLOW_PASSWORD="your_password"
```

> **Tip:** Add these to your shell profile (`.bashrc`, `.zshrc`, or PowerShell `$PROFILE`) so they persist across sessions.

### Step 3: Index Your Organization's Codebase

Before analyzing any failures, the tool needs to build a searchable index of your codebase. This is a **one-time setup** (re-run only when code changes significantly).

```bash
# Index a public repo
python cli/analyze_incident.py \
  --repo https://github.com/your-org/your-project \
  --reindex

# Index a private repo (uses GITHUB_TOKEN from environment)
python cli/analyze_incident.py \
  --repo https://github.com/your-org/private-service \
  --reindex
```

**What happens during indexing:**
1. Clones (or pulls) the repo to a local `data/repos/` directory
2. Scans all `.py`, `.java`, `.sql`, `.yaml`, `.yml` files
3. Splits code into chunks at function/class boundaries (~400 tokens each)
4. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2` (downloads ~80MB on first run)
5. Stores vectors in FAISS index at `data/vector_index/`
6. Caches embeddings at `data/embedding_cache/` for faster future re-indexing

```
Output:
  Repository indexed successfully (1,247 chunks).
```

> **Re-indexing:** Run with `--reindex` after significant code changes. The embedding cache means unchanged files won't be re-processed, so re-indexing is fast.

### Step 4: Analyze a Failure

You have **two ways** to feed logs into the tool:

#### Option A: From a Local Log File

Copy/save the failure log from your monitoring system, Airflow UI, or log aggregator to a local file:

```bash
# Save your failure log to a file first, then:
python cli/analyze_incident.py \
  --log-file C:\logs\dag_failure_2026-03-08.log \
  --repo https://github.com/your-org/your-project
```

**Example log file content** (`dag_failure_2026-03-08.log`):
```
2026-03-08 12:03:21 INFO  Starting task process_events
2026-03-08 12:03:22 INFO  Connecting to Kafka broker kafka-prod-01:9092
2026-03-08 12:03:25 WARN  Connection attempt 1 failed, retrying...
2026-03-08 12:03:28 WARN  Connection attempt 2 failed, retrying...
2026-03-08 12:03:31 ERROR KafkaTimeoutException: Failed to send message
Traceback (most recent call last):
  File "event_producer.py", line 42, in send_event
    producer.send(topic, event)
  File "kafka/producer.py", line 128, in send
    raise KafkaTimeoutException("Broker unreachable")
KafkaTimeoutException: Failed to send message after 3 retries
2026-03-08 12:03:31 ERROR Task process_events failed
```

#### Option B: Directly from Airflow

If your Airflow instance has the REST API enabled:

```bash
python cli/analyze_incident.py \
  --dag shipment_pipeline \
  --run scheduled__2026-03-08T06:00:00+00:00 \
  --task process_events \
  --repo https://github.com/your-org/your-project \
  --airflow-url https://airflow.your-company.com \
  --airflow-user admin \
  --airflow-pass secret
```

> **Finding the DAG run ID:** In the Airflow UI, go to your DAG → click the failed run → the run ID is shown in the URL and the header (e.g., `scheduled__2026-03-08T06:00:00+00:00` or `manual__2026-03-08T12:00:00+00:00`).

### Step 5: Read the Report

The tool outputs a structured incident report:

```
================================================================================
                         INCIDENT ANALYSIS REPORT
================================================================================

INCIDENT SUMMARY:
  KafkaTimeoutException during task 'process_events' — Kafka producer
  failed to reach broker after 3 retry attempts.

ROOT CAUSE:
  The send_event() function in event_producer.py attempts to send messages
  to Kafka without validating broker connectivity first. The hardcoded
  retry count (3) is insufficient for transient network issues.

IMPACTED MODULE:
  event_producer.py

FAILURE PATH:
  1. Task 'process_events' started and initialized Kafka producer
  2. Producer attempted to connect to kafka-prod-01:9092
  3. Connection failed on all 3 retry attempts
  4. KafkaTimeoutException raised at event_producer.py:42
  5. Task marked as failed

SUGGESTED FIX:
  1. Increase retry count and add exponential backoff in producer config
  2. Add broker health check before sending messages
  3. Verify kafka-prod-01:9092 is reachable from the Airflow worker

SEVERITY: HIGH

CONFIDENCE: 0.82

TIMELINE:
  [12:03:21] Starting task process_events
  [12:03:22] Connecting to Kafka broker kafka-prod-01:9092
  [12:03:25] Connection attempt 1 failed, retrying...
  [12:03:28] Connection attempt 2 failed, retrying...
  [12:03:31] KafkaTimeoutException: Failed to send message
  [12:03:31] Task process_events failed

RETRIEVED CODE REFERENCES:
  - event_producer.py :: send_event()
  - kafka_config.py :: get_producer_config()
================================================================================
```

### Step 6: Get JSON Output (for Automation)

For integration with ticketing systems (Jira, ServiceNow, PagerDuty):

```bash
python cli/analyze_incident.py \
  --log-file C:\logs\dag_failure.log \
  --repo https://github.com/your-org/your-project \
  --output-format json
```

Output:
```json
{
  "incident_summary": "KafkaTimeoutException during task 'process_events'...",
  "root_cause": "The send_event() function in event_producer.py...",
  "impacted_module": "event_producer.py",
  "failure_path": "1. Task started... 2. Connection failed...",
  "suggested_fix": "Increase retry count and add exponential backoff...",
  "severity": "High",
  "confidence": 0.82,
  "timeline": [
    {"timestamp": "12:03:21", "event": "Starting task process_events"},
    {"timestamp": "12:03:31", "event": "KafkaTimeoutException: Failed to send message"}
  ],
  "retrieved_code": [
    {"file": "event_producer.py", "function": "send_event()"},
    {"file": "kafka_config.py", "function": "get_producer_config()"}
  ]
}
```

> **Tip:** Pipe JSON output to a file: `... --output-format json > report.json`

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python**  | 3.9+    | Check with `python --version` |
| **pip**     | 21.0+   | Check with `pip --version` |
| **Git**     | 2.30+   | Required for cloning repos. Check with `git --version` |
| **RAM**     | 8 GB+   | 16 GB recommended if using 7B LLM models |
| **Disk**    | ~5 GB   | For model downloads + vector index |

---

## Installation

### Standard Installation

```bash
git clone https://github.com/white-yaksha/ai-log-analyzer.git
cd ai-log-analyzer/ai_incident_investigator

python -m venv .venv

# Activate virtual environment:
# Windows PowerShell:
.venv\Scripts\activate
# Windows CMD:
# .venv\Scripts\activate.bat
# macOS/Linux:
# source .venv/bin/activate

# Install all dependencies at once:
pip install -r requirements.txt
```

### Install Dependencies Individually (if needed)

If you prefer to install packages one by one or troubleshoot specific installs:

```bash
# Core dependencies
pip install requests>=2.31.0
pip install sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.4
pip install transformers>=4.36.0
pip install gitpython>=3.1.40
pip install pyyaml>=6.0

# PyTorch — pick ONE based on your hardware:
pip install torch>=2.1.0                          # CPU only (default)
# pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU (CUDA 12.1)
# pip install torch --index-url https://download.pytorch.org/whl/cu118   # NVIDIA GPU (CUDA 11.8)

# Optional: 4-bit quantization (reduces LLM memory from ~14GB to ~4GB)
pip install bitsandbytes>=0.41.0

# Testing
pip install pytest>=7.4.0
```

### Verify Installation

```bash
# Should print version without errors
python -c "import sentence_transformers; print('sentence-transformers:', sentence_transformers.__version__)"
python -c "import faiss; print('faiss-cpu:', faiss.__version__)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Run the test suite (all 82 tests should pass)
cd ai_incident_investigator
pytest tests/ -v
```

---

## Configuration

### Environment Variables

| Variable           | Description                                       | When Required             |
|--------------------|---------------------------------------------------|---------------------------|
| `GITHUB_TOKEN`     | GitHub personal access token ([create one](https://github.com/settings/tokens)) | For **private** repos only |
| `AIRFLOW_URL`      | Airflow base URL (e.g., `http://airflow.company.com:8080`) | For Airflow mode only     |
| `AIRFLOW_USER`     | Airflow username for HTTP Basic Auth              | For Airflow mode only     |
| `AIRFLOW_PASSWORD`  | Airflow password for HTTP Basic Auth              | For Airflow mode only     |

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:AIRFLOW_URL="https://airflow.your-company.com"
$env:AIRFLOW_USER="your_username"
$env:AIRFLOW_PASSWORD="your_password"
```

**macOS/Linux (bash/zsh):**
```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export AIRFLOW_URL="https://airflow.your-company.com"
export AIRFLOW_USER="your_username"
export AIRFLOW_PASSWORD="your_password"
```

> **Note:** All of these can also be passed as CLI arguments instead (see [CLI Reference](#cli-reference)).

---

## CLI Reference

```
python cli/analyze_incident.py [OPTIONS]
```

### All Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **Log Source (pick one):** | | | |
| `--dag` | string | — | Airflow DAG ID |
| `--run` | string | — | Airflow DAG run ID |
| `--task` | string | — | Airflow task ID |
| `--log-file` | path | — | Path to a local log file |
| **Repository:** | | | |
| `--repo` | URL | — | GitHub repository URL to index and search |
| `--reindex` | flag | false | Force re-index the repository |
| **Model & Retrieval:** | | | |
| `--model` | string | `microsoft/Phi-3-mini-4k-instruct` | HuggingFace model name |
| `--top-k` | int | 5 | Number of code snippets to retrieve |
| **Output:** | | | |
| `--output-format` | text/json | text | Report format |
| **Airflow Connection:** | | | |
| `--airflow-url` | URL | `$AIRFLOW_URL` | Airflow base URL |
| `--airflow-user` | string | `$AIRFLOW_USER` | Airflow username |
| `--airflow-pass` | string | `$AIRFLOW_PASSWORD` | Airflow password |
| **Storage:** | | | |
| `--index-path` | path | `data/vector_index` | Where to store the FAISS index |
| `--cache-path` | path | `data/embedding_cache` | Where to store the embedding cache |
| **Debug:** | | | |
| `-v`, `--verbose` | flag | false | Enable DEBUG-level logging |

### Rules

- **Airflow mode** requires all three: `--dag`, `--run`, and `--task`
- **Local file mode** uses `--log-file` (cannot combine with `--dag/--run/--task`)
- **`--reindex`** requires `--repo`
- If only `--repo --reindex` is provided (no log source), the tool indexes the repo and exits

---

## Usage Examples

### Example 1: Analyze a local log file against your org repo

```bash
python cli/analyze_incident.py \
  --log-file /var/log/airflow/dag_failures/shipment_pipeline_2026-03-08.log \
  --repo https://github.com/your-org/shipment-service
```

### Example 2: Pull logs directly from Airflow and analyze

```bash
python cli/analyze_incident.py \
  --dag shipment_pipeline \
  --run scheduled__2026-03-08T06:00:00+00:00 \
  --task process_events \
  --repo https://github.com/your-org/shipment-service
```

### Example 3: Re-index your repo after a code deploy

```bash
python cli/analyze_incident.py \
  --repo https://github.com/your-org/shipment-service \
  --reindex
```

### Example 4: Use a different LLM model with more code snippets

```bash
python cli/analyze_incident.py \
  --log-file failure.log \
  --repo https://github.com/your-org/shipment-service \
  --model meta-llama/Llama-3-8B-Instruct \
  --top-k 10
```

### Example 5: JSON output for scripting / automation

```bash
python cli/analyze_incident.py \
  --log-file failure.log \
  --repo https://github.com/your-org/shipment-service \
  --output-format json > incident_report.json
```

### Example 6: Private repo with verbose logging

```bash
# Set token first
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

python cli/analyze_incident.py \
  --log-file failure.log \
  --repo https://github.com/your-org/private-service \
  --verbose
```

### Example 7: Full Airflow mode with all connection details inline

```bash
python cli/analyze_incident.py \
  --dag etl_daily_ingest \
  --run scheduled__2026-03-10T00:00:00+00:00 \
  --task load_to_warehouse \
  --repo https://github.com/your-org/etl-pipelines \
  --airflow-url https://airflow.your-company.com \
  --airflow-user admin \
  --airflow-pass secret123 \
  --output-format json \
  --top-k 8
```

---

## Sample Output

### Text Report

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

---

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

---

## How the RAG Pipeline Works

AI Incident Investigator uses a pure **Retrieval Augmented Generation (RAG)** approach — **no model fine-tuning is required**. The system dynamically retrieves relevant code at query time and injects it into the LLM prompt.

### The 6 Stages:

1. **Log Ingestion & Parsing** — Raw logs are fetched from the Airflow REST API or read from a local file. The parser extracts error types, stack traces, timestamps, referenced file paths, and keywords.

2. **Code Repository Indexing** — The GitHub repo is cloned. Source files (`.py`, `.java`, `.sql`, `.yaml`, `.yml`) are chunked at function/class boundaries (~400 tokens with 50-token overlap), embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored in a FAISS vector index. A SHA-256 embedding cache skips unchanged files.

3. **Semantic Retrieval + Priority Boosting** — The error query is embedded and searched via cosine similarity. Files appearing in the stack trace get a **1.5× score boost**.

4. **Context Assembly** — Code snippets and the incident timeline are assembled into a structured LLM prompt (~3,000 chars budget).

5. **LLM Reasoning + Confidence** — The prompt is sent to the LLM. A confidence score (0.0–1.0) is computed from retrieval similarity.

6. **Report Generation** — Output is formatted as text or JSON with: summary, root cause, impacted module, failure path, fix, severity, confidence, timeline, and code references.

---

## System Workflow

1. User runs the CLI with Airflow coordinates (`--dag`, `--run`, `--task`) or a local log file (`--log-file`)
2. **Airflow Client** fetches the task log via REST API (or reads the local file)
3. **Log Parser** extracts errors, stack traces, timestamps, and referenced files
4. **GitHub Repo Manager** clones (or pulls) the target repository
5. **Code Indexer** scans → chunks → embeds → stores in FAISS (using embedding cache)
6. **Retriever** embeds the error query → cosine search → priority-boosts stack trace files
7. **Context Builder** assembles the LLM prompt from errors + timeline + code snippets
8. **LLM Engine** generates the analysis (with optional 4-bit quantization)
9. **Incident Analyzer** parses output, computes confidence, produces the final report

---

## Airflow Integration

The system connects to Apache Airflow via its [REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html).

**Endpoint format:**
```
GET {AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs
```

**Authentication:** HTTP Basic Auth via environment variables or CLI flags.

**Local file fallback:** When Airflow is unavailable or logs are already on disk, use `--log-file`. The parser handles both sources identically.

---

## GitHub Repository Indexing

| Setting | Value |
|---------|-------|
| **Supported files** | `.py`, `.java`, `.sql`, `.yaml`, `.yml` |
| **Chunk size** | ~400 tokens (whitespace-split) |
| **Chunk overlap** | 50 tokens |
| **Splitting strategy** | Function/class boundaries first, then line-based fallback |
| **Embedding model** | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim vectors) |
| **Vector store** | FAISS `IndexFlatIP` (cosine similarity via inner product on normalized vectors) |
| **Embedding cache** | SHA-256 content hash → cached embedding (avoids recomputation) |

---

## Key Features Explained

### Embedding Cache
When you re-index a repository, most files haven't changed. The embedding cache stores a `SHA-256(file content) → embedding` mapping, so unchanged files skip the expensive embedding step. For a 1,000-file repo, re-indexing typically processes only the changed files.

### Stack Trace Priority Boosting
When the log contains `File "event_producer.py", line 42`, the retriever automatically boosts code chunks from `event_producer.py` by **1.5×** in the similarity ranking. This dramatically improves accuracy because the most relevant file is right there in the error.

### Incident Timeline
The parser extracts timestamps from log lines and constructs a chronological timeline:
```
[12:03:21] DAG started
[12:03:25] Kafka connection attempt
[12:03:26] Timeout occurred
```
This gives the LLM temporal context about the failure progression.

### Confidence Score
After retrieving code snippets, the system computes `mean(similarity_scores)` clamped to `[0.0, 1.0]`. Higher scores mean the retrieved code closely matches the error — the LLM has better context to reason about.

| Score | Interpretation |
|-------|---------------|
| **0.8–1.0** | High confidence — retrieved code very relevant |
| **0.5–0.8** | Moderate — some relevant code found |
| **0.0–0.5** | Low — code context may not match the error well |

---

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
│   ├── __init__.py
│   ├── test_airflow_client.py
│   ├── test_log_parser.py
│   ├── test_github_repo_manager.py
│   ├── test_code_indexer.py
│   ├── test_embeddings.py
│   ├── test_embedding_cache.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_context_builder.py
│   ├── test_llm_engine.py
│   └── test_incident_analyzer.py
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Dependencies

| Package                | Version    | Purpose                                                  | Install Command |
|------------------------|------------|----------------------------------------------------------|-----------------|
| `requests`             | ≥ 2.31.0   | HTTP client for Airflow REST API                        | `pip install requests` |
| `sentence-transformers`| ≥ 2.2.0    | Code and log text embedding model                       | `pip install sentence-transformers` |
| `faiss-cpu`            | ≥ 1.7.4    | Vector similarity search engine                         | `pip install faiss-cpu` |
| `transformers`         | ≥ 4.36.0   | HuggingFace model loading and tokenization              | `pip install transformers` |
| `torch`                | ≥ 2.1.0    | PyTorch deep learning framework                         | `pip install torch` |
| `gitpython`            | ≥ 3.1.40   | Git repository clone/pull operations                    | `pip install gitpython` |
| `pyyaml`               | ≥ 6.0      | YAML configuration file parsing                         | `pip install pyyaml` |
| `bitsandbytes`         | ≥ 0.41.0   | 4-bit model quantization (saves ~10GB RAM)              | `pip install bitsandbytes` |
| `pytest`               | ≥ 7.4.0    | Test framework                                          | `pip install pytest` |

---

## Running Tests

```bash
# Run all 82 tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_log_parser.py -v

# Run with output capture disabled (see print statements)
pytest tests/ -v -s
```

All tests are self-contained — they mock external dependencies (Airflow API, GitHub, LLM models) and require no network access, GPU, or model downloads.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
```

### First run is slow (downloading models)
On the first run, `sentence-transformers/all-MiniLM-L6-v2` (~80MB) is downloaded automatically. The LLM model (e.g., Phi-3 at ~2.4GB quantized) is also downloaded on first use. Subsequent runs use cached models.

### Out of memory when loading the LLM
Try a smaller model or ensure quantization is enabled:
```bash
# Use the smaller Phi-3 mini (default, ~2.4GB with 4-bit quantization)
python cli/analyze_incident.py --model microsoft/Phi-3-mini-4k-instruct ...

# Or disable quantization if bitsandbytes causes issues (needs more RAM)
```

### "ConnectionError" when connecting to Airflow
- Verify `AIRFLOW_URL` is correct and reachable: `curl $AIRFLOW_URL/api/v1/health`
- Check credentials: try logging into the Airflow UI with the same username/password
- Use `--log-file` as a fallback: copy the log from the Airflow UI manually

### Private GitHub repo access denied
- Create a personal access token at https://github.com/settings/tokens
- Required scope: `repo` (Full control of private repositories)
- Set it: `export GITHUB_TOKEN="ghp_xxx..."` or `$env:GITHUB_TOKEN="ghp_xxx..."`

---

## Design Principles

- **Modular architecture** — Each component is a standalone module that can be developed, tested, and replaced independently
- **Clear separation of responsibilities** — The orchestrator coordinates the pipeline; each module owns its domain
- **Production-style project structure** — `cli/`, `src/`, `data/`, `tests/` following Python best practices
- **Well-documented components** — All modules include docstrings and type hints
- **Minimal external dependencies** — Only essential, well-maintained libraries
- **Local-first** — Runs entirely on a developer machine; no cloud services required
- **No model fine-tuning** — Pure RAG approach; swap in any compatible HuggingFace model
