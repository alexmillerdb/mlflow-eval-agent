# MLflow Evaluation Agent - Codebase Reference

> **Purpose**: Autonomous agent for analyzing MLflow traces and building complete evaluation suites using Claude Agent SDK with a single-agent initializer/worker loop pattern.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Autonomous Loop                          │
├─────────────────────────────────────────────────────────────────┤
│  Session 1 (Initializer)                                        │
│  ├─ Analyze traces → Understand agent behavior                  │
│  ├─ Plan evaluation strategy                                    │
│  └─ Create eval_tasks.json                                      │
├─────────────────────────────────────────────────────────────────┤
│  Sessions 2-N (Worker)                                          │
│  ├─ Read task list, find pending task                           │
│  ├─ Execute ONE task (dataset/scorer/script/validate)           │
│  ├─ Save artifact to evaluation/                                │
│  └─ Update task status                                          │
├─────────────────────────────────────────────────────────────────┤
│  Termination: all_tasks_complete() or max_iterations            │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
- Fresh context per session (no memory accumulation)
- State persistence via JSON files
- 3 consolidated MCP tools (reduced from 11)
- External prompts in markdown files

---

## Directory Structure

```
mlflow-eval-agent/
├── src/                      # Core agent implementation
│   ├── agent.py              # Main agent with async streaming
│   ├── cli.py                # CLI: autonomous, interactive, test modes
│   ├── config.py             # Environment configuration
│   ├── runtime.py            # Databricks/local runtime detection
│   ├── tools.py              # 3 MCP tools (query, annotate, save)
│   ├── mlflow_ops.py         # MLflow operations + context monitoring
│   ├── databricks_auth.py    # Auth configuration
│   └── test_harness.py       # Component testing framework
│
├── prompts/                  # External prompt files
│   ├── initializer.md        # Phase 1: Trace analysis + task planning
│   └── worker.md             # Phase 2: Task execution
│
├── .claude/
│   ├── skills/
│   │   └── mlflow-evaluation/  # Critical skill with 60+ patterns
│   │       ├── GOTCHAS.md
│   │       ├── CRITICAL-interfaces.md
│   │       ├── patterns-datasets.md
│   │       ├── patterns-scorers.md
│   │       ├── patterns-evaluation.md
│   │       └── patterns-trace-analysis.md
│   ├── commands/             # Custom slash commands
│   └── settings.local.json   # MCP server configuration
│
├── evaluation/               # Generated evaluation suite
│   ├── eval_dataset.py       # Evaluation dataset
│   ├── scorers.py            # Custom scorers
│   ├── skill_scorers.py      # Reusable code quality scorers
│   └── run_eval.py           # Evaluation orchestrator
│
├── notebooks/
│   └── run_eval_agent.py     # Databricks notebook entry point
│
└── scripts/
    ├── analyze_trace.py       # Single trace analysis
    ├── analyze_traces_batch.py # Batch trace analysis
    ├── skill_benchmark.py     # Skill evaluation CLI
    └── skill_eval_pipeline.py # Evaluation pipeline
```

---

## Core Components

### 1. Agent (`src/agent.py`)

**MLflowAgent class**: Single agent handling both autonomous and interactive modes

```python
# Key methods
agent.query(prompt, session_id)  # Async generator yielding streaming results
run_autonomous(experiment_id, max_iterations)  # Main loop

# Data flow
CLI → env vars → setup_mlflow() → Config.from_env() →
  Session 1 (initializer) → eval_tasks.json →
  Sessions 2-N (worker) → evaluation suite complete
```

### 2. MCP Tools (`src/tools.py`)

| Tool | Operations | Purpose |
|------|------------|---------|
| `mlflow_query` | search, get, assessment, search_runs, get_run | Read traces and runs |
| `mlflow_annotate` | tag, feedback, expectation | Write annotations |
| `save_findings` | — | Persist state to JSON files |

### 3. Prompts

**Initializer** (`prompts/initializer.md`):
- Analyze traces to understand agent (type, tools, I/O formats, errors)
- Map characteristics to evaluation dimensions (Safety, Correctness, Relevance, Groundedness)
- Choose dataset strategy (traces, manual, hybrid)
- Create ordered task list in `eval_tasks.json`

**Worker** (`prompts/worker.md`):
- Load task list, find next pending task
- Execute ONE task type:
  - `dataset`: Create `evaluation/eval_dataset.py`
  - `scorer`: Create `evaluation/scorers.py`
  - `script`: Create `evaluation/run_eval.py`
  - `validate`: Run and verify results
- Update task status, end session

### 4. Configuration (`src/config.py`)

```python
# Environment variables
DATABRICKS_HOST          # Workspace URL
DATABRICKS_TOKEN         # Auth token
MLFLOW_EXPERIMENT_ID     # Target experiment
MLFLOW_TRACKING_URI      # "databricks" for remote
ANTHROPIC_MODEL          # Default: "databricks-claude-opus-4-5"
```

### 5. Runtime Detection (`src/runtime.py`)

| Context | Detection | Session Path |
|---------|-----------|--------------|
| `DATABRICKS_JOB` | DB_IS_JOB, runtime version | Unity Catalog Volume |
| `LOCAL` | Default | `./sessions` |

---

## Evaluation Framework

### Scorers (`evaluation/scorers.py`)

**Built-in (from MLflow)**:
- `Safety` - No harmful content
- `RelevanceToQuery` - Addresses user query

**Guidelines-Based (LLM Judges)**:
```python
from mlflow.genai.scorers import Guidelines

correct_routing_scorer = Guidelines(
    name="correct_routing",
    guidelines="Verify the agent routed to appropriate domain specialist..."
)
```

**Custom Code-Based**:
```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def genie_call_count(outputs) -> Feedback:
    response = outputs.get("response", "")
    count = response.count("[Agent:")
    return Feedback(name="genie_call_count", value=count)

@scorer(aggregations=["mean", "min", "max", "p90"])
def response_length_words(outputs) -> int:
    return len(outputs.get("response", "").split())
```

### Dataset Schema

```python
# REQUIRED structure for MLflow GenAI
record = {
    "inputs": {"query": "..."},          # REQUIRED - passed to predict_fn
    "outputs": {"response": "..."},      # OPTIONAL - pre-computed outputs
    "expectations": {"expected": "..."}  # OPTIONAL - ground truth for Correctness
}
```

### Running Evaluation

```python
import mlflow
from mlflow.genai import evaluate

# CRITICAL: Order matters!
mlflow.set_tracking_uri("databricks")    # MUST be first
mlflow.set_experiment(experiment_id=123)

results = evaluate(
    data=eval_dataset,
    predict_fn=None,  # Use pre-computed outputs
    scorers=[safety_scorer, relevance_scorer, custom_scorer]
)
```

---

## Critical Gotchas

| # | Mistake | Correct |
|---|---------|---------|
| 1 | `mlflow.evaluate()` | `mlflow.genai.evaluate()` |
| 2 | `from mlflow.metrics import ...` | `from mlflow.genai.scorers import ...` |
| 3 | `{"query": "..."}` | `{"inputs": {"query": "..."}}` |
| 4 | `predict_fn(inputs)` | `predict_fn(**inputs)` (unpacked kwargs) |
| 5 | `set_experiment()` first | `set_tracking_uri()` BEFORE `set_experiment()` |
| 6 | `Guidelines(guidelines=...)` | BOTH `name` AND `guidelines` required |
| 7 | Return dict from scorer | Return `Feedback` object or primitive |
| 8 | Correctness without expectations | Requires `expected_facts` or `expected_response` |
| 9 | RetrievalGroundedness anywhere | Only works with `span_type="RETRIEVER"` spans |
| 10 | `aggregations=["p50", "p99"]` | Valid: `min, max, mean, median, variance, p90` |

---

## CLI Commands

### Execution Modes

```bash
# Autonomous mode - full evaluation suite generation
uv run python -m src.cli -a -e <experiment_id>

# Interactive mode - free-form queries
uv run python -m src.cli -i

# Single query
uv run python -m src.cli "Analyze trace xyz"
```

### Testing

```bash
# Test initializer (trace analysis + task planning)
uv run python -m src.cli test initializer -e <experiment_id>

# Test worker (task execution)
uv run python -m src.cli test worker -e <experiment_id> --session-dir <path>

# Test specific tool
uv run python -m src.cli test tools mlflow_query --operation search -e <experiment_id>

# Full integration
uv run python -m src.cli test integration -e <experiment_id> --max-iterations 2

# Mock testing (no MLflow connection)
uv run python -m src.cli test initializer -e 123 --mock
```

### Analysis Scripts

```bash
# Single trace deep analysis
python scripts/analyze_trace.py <trace-id>
python scripts/analyze_trace.py <trace-id> --tokens  # Token usage
python scripts/analyze_trace.py <trace-id> --full    # All details

# Batch analysis
python scripts/analyze_traces_batch.py <experiment-id>

# Smoke test MLflow tools
python scripts/smoke_test_mlflow_tools.py
```

---

## Session State

```
sessions/{session_id}/
├── eval_tasks.json           # Task list with status
├── state/
│   ├── analysis.json         # Initializer findings
│   ├── recommendations.json  # Evaluation strategy
│   └── validation_results.json
└── evaluation/
    ├── eval_dataset.py
    ├── scorers.py
    └── run_eval.py
```

### Task Status Flow

```
pending → in_progress → completed
                    └→ failed (after 5 attempts)
```

---

## Context Monitoring

The agent tracks token usage per session:

```python
# Warnings at thresholds
40KB  → Warning: Consider summarizing
80KB  → Critical: Risk of context overflow

# Tool call budget
Target: 5-8 calls per task
Max: 12 calls before declaring blocked

# Trace detail levels
summary  → ~2KB  (exploration)
analysis → ~10KB (debugging)
full     → ~50KB (deep dive)
```

---

## Databricks Deployment

```bash
# Deploy bundle
databricks bundle deploy -t dev

# Run notebook job
databricks bundle run eval_agent_notebook -t dev

# Notebook sets up:
# - Widget parameters (experiment_id, volume_path, model)
# - Secret scope authentication
# - Calls run_autonomous() with max 20 iterations
```

---

## MLflow Evaluation Skill

Located at `.claude/skills/mlflow-evaluation/`, contains:

| File | Content |
|------|---------|
| `GOTCHAS.md` | 19+ common mistakes that break code |
| `CRITICAL-interfaces.md` | Exact API signatures |
| `patterns-datasets.md` | 16 dataset creation patterns |
| `patterns-scorers.md` | 16 scorer implementation patterns |
| `patterns-evaluation.md` | 7 evaluation workflow patterns |
| `patterns-trace-analysis.md` | 9 trace analysis patterns |

**Always read these before generating evaluation code.**

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Python LOC | ~4,000 |
| Core modules | 7 files |
| MCP Tools | 3 |
| Pattern examples | 60+ |
| Test types | 4 (initializer, worker, tools, integration) |
| Context warning | 40KB |
| Context critical | 80KB |

---

## Quick Reference: Creating Scorers

### Guidelines-Based (LLM Judge)
```python
from mlflow.genai.scorers import Guidelines

my_scorer = Guidelines(
    name="my_guidelines",
    guidelines="Evaluate if the response properly addresses..."
)
```

### Code-Based (Fast, Deterministic)
```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def my_metric(outputs) -> Feedback:
    value = compute_something(outputs["response"])
    return Feedback(name="my_metric", value=value)
```

### With Aggregations
```python
@scorer(aggregations=["mean", "min", "max", "median", "p90"])
def word_count(outputs) -> int:
    return len(outputs.get("response", "").split())
```

---

## Quick Reference: Dataset Creation

### From Production Traces
```python
import mlflow

traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'OK'"
)

eval_data = [
    {
        "inputs": trace.data.request,
        "outputs": {"response": trace.data.response}
    }
    for trace in traces
]
```

### Manual with Expectations
```python
eval_data = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"}
    }
]
```

---

## Workflow Summary

1. **Set experiment**: `MLFLOW_EXPERIMENT_ID=<id>`
2. **Run autonomous**: `uv run python -m src.cli -a -e <id>`
3. **Initializer**: Analyzes traces → creates `eval_tasks.json`
4. **Worker(s)**: Execute tasks → generate `evaluation/*.py`
5. **Validate**: Run evaluation → verify results
6. **Output**: Complete evaluation suite in `sessions/<id>/evaluation/`
