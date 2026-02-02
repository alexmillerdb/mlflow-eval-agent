# Phase 0: Codebase Restructure

> **Status**: Complete
> **Completed**: 2026-01-30
> **Commit**: `a6c945c`

**Goal**: Reorganize flat `src/` into three-tier `core/` + `agent/` + `app/`
**Estimated Effort**: Medium
**Dependencies**: None
**Risk**: Import breakage - requires careful testing

---

## Feature 0.1: Create Directory Structure

**Status**: ✅ Complete

**Task**: Create the new directory structure with `__init__.py` files.

**Files to Create**:
- [x] `src/core/__init__.py`
- [x] `src/agent/__init__.py`
- [x] `src/app/__init__.py`
- [x] `src/app/components/__init__.py`

**Commands**:
```bash
mkdir -p src/core src/agent src/app/components
touch src/core/__init__.py src/agent/__init__.py src/app/__init__.py
touch src/app/components/__init__.py
```

**Verification**:
```bash
ls -la src/core src/agent src/app
```

---

## Feature 0.2: Move Core Infrastructure

**Status**: ✅ Complete

**Task**: Move shared infrastructure files to `core/` package.

**File Migrations**:
| Source | Destination | Notes |
|--------|-------------|-------|
| `src/config.py` | `src/core/config.py` | No changes needed |
| `src/runtime.py` | `src/core/runtime.py` | No changes needed |
| `src/databricks_auth.py` | `src/core/auth.py` | Rename file |

**Update `src/core/__init__.py`**:
```python
"""Core infrastructure shared across agent and app."""
from .config import Config
from .runtime import RuntimeContext, RuntimeInfo, detect_runtime, get_sessions_base_path
from .auth import configure_env
```

**Verification**:
```bash
uv run python -c "from src.core.config import Config; print('OK')"
uv run python -c "from src.core.runtime import detect_runtime; print(detect_runtime())"
```

---

## Feature 0.3: Move Agent Logic

**Status**: ✅ Complete

**Task**: Move agent-related files to `agent/` package.

**File Migrations**:
| Source | Destination | Notes |
|--------|-------------|-------|
| `src/agent.py` | `src/agent/agent.py` | Extract autonomous.py and prompts.py |
| `src/tools.py` | `src/agent/tools.py` | Update imports |
| `src/mlflow_ops.py` | `src/agent/mlflow_ops.py` | Update imports |
| `src/skills.py` | `src/agent/skills.py` | No changes needed |

**Update `src/agent/__init__.py`**:
```python
"""Autonomous MLflow evaluation agent."""
from .agent import MLflowAgent, AgentResult, setup_mlflow
from .autonomous import run_autonomous
from .prompts import load_prompt
from .tools import create_tools, MCPTools, BuiltinTools
```

**Verification**:
```bash
uv run python -c "from src.agent.agent import MLflowAgent; print('OK')"
uv run python -c "from src.agent.tools import create_tools; print(len(create_tools()), 'tools')"
```

---

## Feature 0.4: Extract Autonomous Module

**Status**: ✅ Complete

**Task**: Extract `run_autonomous()` function from `agent.py` to dedicated `autonomous.py`.

**Create `src/agent/autonomous.py`**:
- Move `run_autonomous()` function (lines ~321-470 of current agent.py)
- Move `AUTO_CONTINUE_DELAY_SECONDS` constant
- Update imports to use relative paths

**Key Imports for autonomous.py**:
```python
import asyncio
import logging
import mlflow
from ..core.config import Config
from ..core.runtime import get_sessions_base_path, detect_runtime, RuntimeContext
from .agent import MLflowAgent
from .prompts import load_prompt
from .mlflow_ops import (
    all_tasks_complete,
    print_progress_summary,
    print_final_summary,
    set_session_dir,
    get_tasks_file,
    start_context_monitoring,
)
```

**Verification**:
```bash
uv run python -c "from src.agent.autonomous import run_autonomous; print('OK')"
```

---

## Feature 0.5: Extract Prompts Module

**Status**: ✅ Complete

**Task**: Extract prompt loading functions from `agent.py` to dedicated `prompts.py`.

**Create `src/agent/prompts.py`**:
- Move `_get_prompts_dir()` function
- Move `load_prompt()` function

**Contents**:
```python
"""Prompt loading utilities."""
import importlib.resources
import logging
from pathlib import Path
import mlflow

logger = logging.getLogger(__name__)

def _get_prompts_dir() -> Path:
    """Get prompts directory, works both installed and in dev."""
    try:
        files = importlib.resources.files("prompts")
        with importlib.resources.as_file(files) as prompts_path:
            if prompts_path.is_dir():
                return prompts_path
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    return Path(__file__).parent.parent.parent / "prompts"

@mlflow.trace
def load_prompt(name: str = "system") -> str:
    """Load external prompt from prompts/ directory."""
    try:
        files = importlib.resources.files("prompts")
        prompt_file = files.joinpath(f"{name}.md")
        return prompt_file.read_text()
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    path = Path(__file__).parent.parent.parent / "prompts" / f"{name}.md"
    if path.exists():
        return path.read_text()

    logger.warning(f"Prompt file not found: {name}.md")
    return ""
```

**Verification**:
```bash
uv run python -c "from src.agent.prompts import load_prompt; print(len(load_prompt('initializer')), 'chars')"
```

---

## Feature 0.6: Update CLI Imports

**Status**: ✅ Complete

**Task**: Update `src/cli.py` to use new import paths.

**Import Changes**:
```python
# BEFORE
from .config import Config
from .agent import MLflowAgent, run_autonomous, setup_mlflow
from .runtime import detect_runtime, RuntimeContext

# AFTER
from .core.config import Config
from .core.runtime import detect_runtime, RuntimeContext
from .agent.agent import MLflowAgent, setup_mlflow
from .agent.autonomous import run_autonomous
```

**Verification**:
```bash
uv run python -m src.cli --help
uv run python -m src.cli test tools --help
```

---

## Feature 0.7: Update Internal Agent Imports

**Status**: ✅ Complete

**Task**: Update all relative imports within agent/ module.

**Files to Update**:

1. **`src/agent/agent.py`**:
```python
# BEFORE
from .config import Config
from .runtime import detect_runtime
from .tools import create_tools, MCPTools, BuiltinTools
from .mlflow_ops import ...

# AFTER
from ..core.config import Config
from ..core.runtime import detect_runtime
from .tools import create_tools, MCPTools, BuiltinTools
from .mlflow_ops import ...
```

2. **`src/agent/tools.py`**:
```python
# Add at top if needed
from . import mlflow_ops
```

3. **`src/agent/mlflow_ops.py`**:
```python
# BEFORE
from .runtime import get_sessions_base_path

# AFTER
from ..core.runtime import get_sessions_base_path
```

**Verification**:
```bash
uv run python -c "from src.agent import MLflowAgent, run_autonomous; print('OK')"
```

---

## Feature 0.8: Update pyproject.toml

**Status**: ✅ Complete

**Task**: Update wheel build configuration for new structure.

**Changes to `pyproject.toml`**:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src", "prompts"]

# Ensure subpackages are included
[tool.hatch.build.targets.wheel.sources]
"src" = "src"
"prompts" = "prompts"
```

**Verification**:
```bash
uv build
unzip -l dist/*.whl | grep -E "(core|agent|app)"
```

---

## Feature 0.9: Run Full Test Suite

**Status**: ✅ Complete

**Task**: Verify all existing functionality works after restructure.

**Test Commands**:
```bash
# Unit tests
uv run pytest tests/ -v

# CLI commands
uv run python -m src.cli --help
uv run python -m src.cli test tools mlflow_query --operation search -e 123 --mock

# Import verification
uv run python -c "
from src.core.config import Config
from src.core.runtime import detect_runtime
from src.agent.agent import MLflowAgent
from src.agent.autonomous import run_autonomous
from src.agent.tools import create_tools
print('All imports OK')
"
```

**Success Criteria**:
- [x] All unit tests pass
- [x] CLI --help works
- [x] All major imports resolve
- [x] No circular import errors

---

## Feature 0.10: Move Test Harness

**Status**: ✅ Complete

**Task**: Move `test_harness.py` to tests directory.

**Migration**:
| Source | Destination |
|--------|-------------|
| `src/test_harness.py` | `tests/harness.py` |

**Update imports in harness.py**:
```python
# Update path references to work from tests/
```

**Verification**:
```bash
uv run python -c "from tests.harness import TestHarness; print('OK')"
```

---

## Feature 0.11: Integration Testing Gate

**Status**: ✅ Complete

**Task**: Validate restructure with full integration test suite before proceeding.

**Risk**: HIGH (import breakage) - requires comprehensive validation at all 6 levels.

**Level 1: Unit Tests**:
```bash
uv run pytest tests/ -v
```

**Level 2: Import Verification**:
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

**Level 3: Mock Integration**:
```bash
uv run python -m src.cli test tools mlflow_query --operation search -e 123 --mock
uv run python -m src.cli test initializer -e 123 --mock
```

**Level 4: Local Integration** (requires $EXPERIMENT_ID):
```bash
# Run initializer
uv run python -m src.cli test initializer -e $EXPERIMENT_ID
# Capture session_dir from output, then:
uv run python -m src.cli test worker -e $EXPERIMENT_ID --session-dir <path>
# Analyze results
/analyze-trace <trace-id>
```

**Level 5: Local Autonomous Run**:
```bash
uv run python -m src.cli -a -e $EXPERIMENT_ID --max-iterations 2
```

**Level 6: Databricks Deployment** (final gate):
```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run eval_agent_job -t dev
databricks bundle run eval_agent_notebook -t dev
```

**Success Criteria**:
- [x] All pytest tests pass
- [x] All imports resolve (no circular imports)
- [x] Mock integration tests pass
- [x] Local integration creates valid eval_tasks.json
- [x] Autonomous run completes at least 1 task
- [x] Databricks job completes successfully
