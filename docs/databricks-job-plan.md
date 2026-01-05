# Databricks Job Adaptation Plan

## Goal
Adapt the MLflow Evaluation Agent to run as a Databricks Job using DABs (Databricks Asset Bundles) with:
- Python wheel with prompts included
- Unity Catalog Volume for session storage
- Single long-running job execution

---

## Implementation Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DABs Deployment Flow                        │
├─────────────────────────────────────────────────────────────────┤
│  1. `databricks bundle deploy`                                  │
│     ├── Builds wheel (includes prompts via package_data)        │
│     └── Creates job with wheel task                             │
│                                                                 │
│  2. Job Execution                                               │
│     ├── Installs wheel + dependencies                           │
│     ├── Runs: mlflow-eval --autonomous -e {experiment_id}       │
│     └── Writes state to Volume path                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Include Prompts in Wheel Package

**File: `pyproject.toml`**

Add prompts as package data so they're included in the wheel:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src", "prompts"]

[tool.hatch.build.targets.wheel.force-include]
"prompts" = "prompts"
```

**File: `src/agent.py` (line 59)**

Update to use `importlib.resources` for package-relative prompts:

```python
# Before
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# After
import importlib.resources

def get_prompts_dir() -> Path:
    """Get prompts directory, works both installed and in dev."""
    # Try package resources first (installed wheel)
    try:
        with importlib.resources.files("prompts") as prompts_path:
            if prompts_path.is_dir():
                return Path(prompts_path)
    except (TypeError, FileNotFoundError):
        pass

    # Fallback to relative path (development)
    return Path(__file__).parent.parent / "prompts"
```

---

## Step 2: Create Runtime Detection Module

**New file: `src/runtime.py`** (~50 lines)

```python
"""Runtime environment detection for Databricks Jobs."""
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class RuntimeContext(Enum):
    LOCAL = "local"
    DATABRICKS_JOB = "databricks_job"

@dataclass
class RuntimeInfo:
    context: RuntimeContext
    job_id: str | None = None
    job_run_id: str | None = None
    volume_path: str | None = None

def detect_runtime() -> RuntimeInfo:
    """Detect execution context from environment."""
    if os.getenv("DB_IS_JOB", "").upper() == "TRUE":
        return RuntimeInfo(
            context=RuntimeContext.DATABRICKS_JOB,
            job_id=os.getenv("DB_JOB_ID"),
            job_run_id=os.getenv("DB_JOB_RUN_ID"),
            volume_path=os.getenv("MLFLOW_AGENT_VOLUME_PATH"),
        )
    return RuntimeInfo(context=RuntimeContext.LOCAL)

def get_sessions_base_path() -> Path:
    """Get base path for session storage."""
    volume_path = os.getenv("MLFLOW_AGENT_VOLUME_PATH")
    if volume_path:
        return Path(volume_path) / "sessions"
    return Path("sessions")
```

---

## Step 3: Update Config for Job Context

**File: `src/config.py`**

Add job-aware session ID generation:

```python
# After dotenv loading, update session_id generation:
from .runtime import detect_runtime

runtime = detect_runtime()

session_id = os.getenv("SESSION_ID", "")
if not session_id:
    if runtime.job_id and runtime.job_run_id:
        session_id = f"job-{runtime.job_id}-run-{runtime.job_run_id}"
    else:
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
```

---

## Step 4: Update Agent for Volume Storage

**File: `src/agent.py`**

Replace hardcoded path (line 241):
```python
# Before
SESSIONS_DIR = Path("sessions")

# After
from .runtime import get_sessions_base_path
```

Update `run_autonomous()` (line 267):
```python
session_dir = get_sessions_base_path() / config.session_id
```

---

## Step 5: Migrate print() to logging

**Files: `src/agent.py`, `src/mlflow_ops.py`**

Replace `print()` with `logging.info()` for job log compatibility.

Key functions to update:
- `agent.py`: `run_autonomous()` (~14 print statements)
- `mlflow_ops.py`: `print_progress_summary()`, `print_final_summary()` (~20 statements)

---

## Step 6: Expand DABs Configuration

**File: `databricks.yml`**

```yaml
bundle:
  name: mlflow-eval-agent

artifacts:
  mlflow_eval_agent:
    type: whl
    path: .
    build: uv build

resources:
  jobs:
    eval_agent_job:
      name: "MLflow Eval Agent"
      tasks:
        - task_key: run_autonomous
          python_wheel_task:
            package_name: mlflow_eval_agent
            entry_point: mlflow-eval
            parameters:
              - "--autonomous"
              - "--experiment-id"
              - "${var.experiment_id}"
          libraries:
            - whl: ./dist/*.whl
          # Use Serverless Compute
          environment_key: default
          env_vars:
            # Session storage
            MLFLOW_AGENT_VOLUME_PATH: "${var.volume_path}"
            MLFLOW_EXPERIMENT_ID: "${var.experiment_id}"
            # Databricks FM API configuration
            ANTHROPIC_BASE_URL: "https://${workspace.host}/serving-endpoints/anthropic"
            ANTHROPIC_AUTH_TOKEN: "{{secrets/mlflow-eval/databricks-token}}"
            ANTHROPIC_API_KEY: ""
            DABS_MODEL: "databricks-claude-sonnet-4"
            CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: "1"

# Serverless environment
environments:
  - environment_key: default
    spec:
      client: "1"  # Serverless compute

variables:
  experiment_id:
    description: "MLflow Experiment ID to analyze"
  volume_path:
    description: "Unity Catalog Volume path for session storage"
    default: "/Volumes/main/default/mlflow-eval-agent"

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://e2-demo-field-eng.cloud.databricks.com
    variables:
      experiment_id: "123456789"
```

**Notes on Databricks FM API:**
- `ANTHROPIC_BASE_URL`: Points to workspace serving endpoint
- `ANTHROPIC_AUTH_TOKEN`: Databricks PAT stored in secret scope
- `ANTHROPIC_API_KEY`: Set empty to disable direct Anthropic API
- `DABS_MODEL`: Use Databricks-hosted Claude model
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS`: Required for FM API compatibility

---

## Step 7: Guard Interactive Mode

**File: `src/cli.py`**

```python
elif args.interactive:
    from .runtime import detect_runtime, RuntimeContext

    if detect_runtime().context == RuntimeContext.DATABRICKS_JOB:
        logging.error("Interactive mode not supported in Databricks Jobs")
        return
    # ... existing interactive code
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add prompts to wheel package_data |
| `src/runtime.py` | **NEW** - Runtime detection (~50 lines) |
| `src/agent.py` | Volume paths, logging, prompts loading |
| `src/config.py` | Job-aware session ID |
| `src/mlflow_ops.py` | Migrate print→logging |
| `src/cli.py` | Guard interactive mode |
| `databricks.yml` | Add wheel artifact + job definition |

---

## Deployment Commands

```bash
# Build and deploy bundle
databricks bundle deploy -t dev

# Run job manually
databricks bundle run eval_agent_job -t dev \
  --param experiment_id=123456789

# View job logs
databricks jobs get-run <run_id>
```

---

## Testing Checklist

1. **Local dev unchanged**: `python -m src.cli -a -e 123` uses `sessions/`
2. **With Volume env**: Set `MLFLOW_AGENT_VOLUME_PATH=/tmp/test`, verify path used
3. **Simulated job**: Set `DB_IS_JOB=TRUE`, verify session ID format
4. **DABs deploy**: `databricks bundle deploy` succeeds
5. **Job execution**: Run job, verify:
   - Wheel installs correctly
   - Prompts load from package
   - State writes to Volume
   - Logs visible in job output
