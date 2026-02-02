---
description: Run validation tests for a specific implementation phase
argument-hint: "<phase-number> [--mock] [--experiment-id <id>]"
---

Run the testing ladder for a specific implementation phase. Phase 0 requires all 6 levels of validation due to high import breakage risk.

## Usage

```bash
# Phase 0 validation (comprehensive - all 6 levels)
/test-phase 0 --experiment-id $EXPERIMENT_ID

# Quick mock-only validation (Levels 1-3)
/test-phase 0 --mock

# Other phases
/test-phase 1  # Streamlit MVP
/test-phase 2  # Files SDK Integration
/test-phase 3  # Runtime & Auth
/test-phase 4  # Enhanced UI
/test-phase 5 --experiment-id $EXPERIMENT_ID  # Deployment
```

## Testing Ladder

| Level | Description | Required For |
|-------|-------------|--------------|
| 1 | pytest unit tests | All phases |
| 2 | Import verification | Phase 0 |
| 3 | Mock integration (`--mock`) | Phase 0, 2 |
| 4 | Local integration (real MLflow) | Phase 0, 1, 2 |
| 5 | Autonomous run | Phase 0 |
| 6 | Databricks deployment | Phase 0, 5 |

## Phase-Specific Tests

### Phase 0: Codebase Restructure (HIGH RISK)

**Level 1: Unit Tests**
```bash
uv run pytest tests/ -v
```

**Level 2: Import Verification**
```bash
uv run python -c "
from src.core.config import Config
from src.core.runtime import detect_runtime, RuntimeContext
from src.agent.agent import MLflowAgent, setup_mlflow
from src.agent.autonomous import run_autonomous
from src.agent.tools import create_tools
print('All imports OK')
print(f'Tools: {len(create_tools())}')
"
```

**Level 3: Mock Integration**
```bash
uv run python -m src.cli test tools mlflow_query --operation search -e 123 --mock
uv run python -m src.cli test initializer -e 123 --mock
```

**Level 4: Local Integration** (requires `--experiment-id`)
```bash
# Run initializer
uv run python -m src.cli test initializer -e <experiment-id>
# Note session_dir from output, then:
uv run python -m src.cli test worker -e <experiment-id> --session-dir <path>
```

**Level 5: Local Autonomous Run**
```bash
uv run python -m src.cli -a -e <experiment-id> --max-iterations 2
```

**Level 6: Databricks Deployment**
```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run eval_agent_job -t dev
```

### Phase 1: Streamlit MVP

**Level 1: Unit Tests**
```bash
uv run pytest tests/test_app/test_streaming.py -v
```

**Level 2: Local App Smoke Test**
```bash
uv run streamlit run src/app/main.py --server.port 8501
# Open http://localhost:8501
# Verify: Page loads, sidebar renders, chat input works
```

**Level 3: Integration with Agent**
1. Enter real experiment ID in sidebar
2. Submit: "List the most recent 5 traces"
3. Verify tool calls appear (🔧 indicators)
4. Verify streaming text response
5. Check MLflow for agent trace

### Phase 2: Files SDK Integration

**Level 1: Unit Tests**
```bash
uv run pytest tests/test_core/test_files.py -v
```

**Level 2: SDK Connectivity** (requires Databricks auth)
```python
from src.core.files import uc_volume_write, uc_volume_read, uc_volume_list
uc_volume_write("/Volumes/catalog/schema/vol/test.txt", "hello")
assert uc_volume_read("/Volumes/catalog/schema/vol/test.txt") == "hello"
print(uc_volume_list("/Volumes/catalog/schema/vol/"))
```

**Level 3: Tool Integration**
```bash
uv run python -m src.cli test tools uc_volume_list --path /Volumes/catalog/schema/vol/
```

### Phase 3: Runtime & Auth

**Level 1: Unit Tests**
```bash
uv run pytest tests/test_core/test_runtime.py -v
```

**Level 2: Environment Mocking**
```python
import os
from unittest.mock import patch
from src.core.runtime import detect_runtime, RuntimeContext

# Test APP detection
with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "test-app"}):
    runtime = detect_runtime()
    assert runtime.context == RuntimeContext.DATABRICKS_APP

# Test APP precedence over JOB
with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "test-app", "DB_IS_JOB": "TRUE"}):
    runtime = detect_runtime()
    assert runtime.context == RuntimeContext.DATABRICKS_APP  # APP wins
```

### Phase 4: Enhanced UI

**Level 1: Component Tests**
```bash
uv run python -c "
from src.app.components.progress import render_task_progress, render_task_list
from src.app.components.sidebar import render_sidebar
print('Components import OK')
"
```

**Level 2: Manual UI Testing**
1. `uv run streamlit run src/app/main.py`
2. Verify tabs: Interactive | Autonomous
3. Verify sidebar shows runtime info
4. Start autonomous run with real experiment
5. Verify progress bar updates

### Phase 5: Deployment

**Level 1: Bundle Validation**
```bash
databricks bundle validate -t dev
```

**Level 2: Deploy & Verify**
```bash
databricks bundle deploy -t dev
databricks apps list
databricks apps get mlflow-eval-agent
```

**Level 3: App Functional Testing**
1. Open app URL from `databricks apps get` output
2. Verify OAuth login works
3. Submit test query
4. Verify streaming response
5. Check UC Volume for session files

## Success Criteria by Phase

| Phase | Criteria |
|-------|----------|
| 0 | All imports resolve, pytest passes, initializer creates `eval_tasks.json`, autonomous completes 1+ tasks |
| 1 | Streaming shows incremental text, tool use indicators appear, no event loop errors |
| 2 | Unit tests pass with mocks, real SDK operations succeed, tool returns correct format |
| 3 | RuntimeContext.DATABRICKS_APP detection works, APP takes precedence over JOB |
| 4 | Components render without errors, tabs switch correctly, progress updates in real-time |
| 5 | Bundle validates, app deploys, OAuth works, full agent functionality works |

## Follow-up Analysis

After any test run that produces a trace:
```bash
/analyze-trace <trace-id>
/analyze-tokens <trace-id>
```
