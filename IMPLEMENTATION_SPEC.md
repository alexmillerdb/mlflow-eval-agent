# MLflow Eval Agent - Databricks Apps Implementation Spec

> **Purpose**: Phased implementation plan for restructuring codebase and adding Streamlit-based Databricks Apps deployment. Designed for modular execution by Claude Code sub-agents.

**Created**: 2024-01-30
**Status**: Phase 0 Complete
**Current Phase**: Ready for Phase 1

---

## Quick Reference

| Phase | Name | Status | Features |
|-------|------|--------|----------|
| 0 | Codebase Restructure | ✅ Complete | 0.1-0.11 |
| 1 | Streamlit MVP | ⬜ Not Started | 1.1-1.6 |
| 2 | Files SDK Integration | ⬜ Not Started | 2.1-2.6 |
| 3 | Runtime & Auth | ⬜ Not Started | 3.1-3.4 |
| 4 | Enhanced UI | ⬜ Not Started | 4.1-4.4 |
| 5 | Deployment | ⬜ Not Started | 5.1-5.4 |

---

## Architecture Overview

### Target Structure

```
src/
├── __init__.py
├── cli.py                        # Root CLI (wheel entry point)
│
├── core/                         # SHARED INFRASTRUCTURE
│   ├── __init__.py              # Exports: Config, RuntimeContext, etc.
│   ├── config.py                # Environment configuration
│   ├── runtime.py               # Runtime detection (LOCAL/JOB/APP)
│   ├── auth.py                  # Unified Databricks auth
│   └── files.py                 # Databricks Files SDK wrapper
│
├── agent/                        # AUTONOMOUS AGENT LOGIC
│   ├── __init__.py              # Exports: MLflowAgent, run_autonomous
│   ├── agent.py                 # MLflowAgent class
│   ├── tools.py                 # MCP tools (MLflow + Files)
│   ├── mlflow_ops.py            # MLflow operations + state
│   ├── prompts.py               # Prompt loading
│   ├── autonomous.py            # run_autonomous() loop
│   └── skills.py                # Skill definitions
│
└── app/                          # STREAMLIT UI
    ├── __init__.py
    ├── main.py                   # Streamlit entry point
    ├── streaming.py              # Async→sync bridge
    └── components/
        ├── __init__.py
        ├── chat.py               # Chat interface
        ├── sidebar.py            # Config sidebar
        └── progress.py           # Task progress
```

### Dependency Flow

```
cli.py (wheel) ──► agent/ ──► core/
                              ▲
app/ (Streamlit) ─────────────┘
```

### Deployment Modes

| Mode | Package | Entry Point | Use Case |
|------|---------|-------------|----------|
| Jobs | Wheel | `mlflow-eval` CLI | Scheduled/batch evaluation |
| Apps | Source | `streamlit run src/app/main.py` | Interactive UI |

---

## Phase 0: Codebase Restructure

**Goal**: Reorganize flat `src/` into three-tier `core/` + `agent/` + `app/`
**Estimated Effort**: Medium
**Dependencies**: None
**Risk**: Import breakage - requires careful testing

### Feature 0.1: Create Directory Structure

**Status**: ✅ Complete

**Task**: Create the new directory structure with `__init__.py` files.

**Files to Create**:
- [ ] `src/core/__init__.py`
- [ ] `src/agent/__init__.py`
- [ ] `src/app/__init__.py`
- [ ] `src/app/components/__init__.py`

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

### Feature 0.2: Move Core Infrastructure

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

### Feature 0.3: Move Agent Logic

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

### Feature 0.4: Extract Autonomous Module

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

### Feature 0.5: Extract Prompts Module

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

### Feature 0.6: Update CLI Imports

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

### Feature 0.7: Update Internal Agent Imports

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

### Feature 0.8: Update pyproject.toml

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

### Feature 0.9: Run Full Test Suite

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
- [ ] All unit tests pass
- [ ] CLI --help works
- [ ] All major imports resolve
- [ ] No circular import errors

---

### Feature 0.10: Move Test Harness

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

### Feature 0.11: Integration Testing Gate

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

---

## Phase 1: Streamlit MVP

**Goal**: Basic Streamlit chat with Claude Agent SDK streaming
**Estimated Effort**: Medium
**Dependencies**: Phase 0 complete
**Risk**: Async/sync bridging complexity

### Feature 1.1: Add Streamlit Dependency

**Status**: ⬜ Not Started

**Task**: Add Streamlit to project dependencies.

**Changes to `pyproject.toml`**:
```toml
dependencies = [
    # existing...
    "streamlit>=1.38.0",
]
```

**Commands**:
```bash
uv add streamlit
uv sync
```

**Verification**:
```bash
uv run streamlit --version
```

---

### Feature 1.2: Create Async-to-Sync Bridge

**Status**: ⬜ Not Started

**Task**: Create streaming adapter for Claude Agent SDK.

**Create `src/app/streaming.py`**:
```python
"""Async-to-sync bridge for Claude Agent SDK streaming."""
import asyncio
import threading
from queue import Queue
from typing import Generator, Callable, Any

def async_to_sync_generator(async_gen_func: Callable, *args, **kwargs) -> Generator:
    """
    Bridge async Claude Agent SDK streaming to Streamlit's sync write_stream.

    Uses a dedicated thread with its own event loop to avoid
    "Event loop is closed" errors that plague async/Streamlit integrations.

    Args:
        async_gen_func: Async generator function (e.g., agent.query)
        *args, **kwargs: Arguments to pass to the async function

    Yields:
        Items from the async generator
    """
    queue: Queue = Queue()

    async def run_async():
        try:
            async for message in async_gen_func(*args, **kwargs):
                queue.put(message)
        except Exception as e:
            queue.put(("__ERROR__", e))
        finally:
            queue.put(None)  # Sentinel to signal completion

    def thread_target():
        # Create a NEW event loop for this thread (critical!)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_async())
        finally:
            loop.close()

    # Start async execution in background thread
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    # Yield items as they arrive (sync generator)
    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__ERROR__":
            raise item[1]
        yield item
```

**Verification**:
```bash
uv run python -c "from src.app.streaming import async_to_sync_generator; print('OK')"
```

---

### Feature 1.3: Create Streamlit Main App

**Status**: ⬜ Not Started

**Task**: Create basic Streamlit chat interface.

**Create `src/app/main.py`**:
```python
"""Streamlit app for MLflow Eval Agent."""
import streamlit as st
from ..core.config import Config
from ..agent.agent import MLflowAgent, setup_mlflow
from .streaming import async_to_sync_generator

# Page config
st.set_page_config(
    page_title="MLflow Eval Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "initialized" not in st.session_state:
    setup_mlflow()
    st.session_state.initialized = True

# Sidebar
with st.sidebar:
    st.header("Configuration")
    experiment_id = st.text_input(
        "Experiment ID",
        value=st.session_state.get("experiment_id", ""),
        help="MLflow experiment ID to analyze"
    )
    st.session_state.experiment_id = experiment_id

    st.divider()

    if st.button("🔄 New Session"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    if st.button("🗑️ Clear State"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Main title
st.title("🔍 MLflow Eval Agent")
st.caption("Analyze traces and build evaluation suites")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your MLflow traces..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        def stream_response():
            config = Config.from_env()
            if experiment_id:
                config.experiment_id = experiment_id
            agent = MLflowAgent(config)

            current_text = ""
            for result in async_to_sync_generator(
                agent.query,
                prompt,
                st.session_state.session_id
            ):
                if result.event_type == "text":
                    # Yield only the new text delta
                    new_text = result.response[len(current_text):]
                    current_text = result.response
                    if new_text:
                        yield new_text
                elif result.event_type == "tool_use":
                    yield f"\n\n🔧 *Using {result.tool_name}...*\n\n"
                elif result.event_type == "result":
                    st.session_state.session_id = result.session_id
                    if result.cost_usd:
                        yield f"\n\n---\n*Cost: ${result.cost_usd:.4f} | Duration: {result.duration_ms}ms*"

        response = st.write_stream(stream_response())
        st.session_state.messages.append({"role": "assistant", "content": response})
```

**Verification**:
```bash
uv run streamlit run src/app/main.py --server.port 8501
# Open http://localhost:8501
```

---

### Feature 1.4: Create App Package Init

**Status**: ⬜ Not Started

**Task**: Set up app package exports.

**Update `src/app/__init__.py`**:
```python
"""Streamlit web application for MLflow Eval Agent."""
from .streaming import async_to_sync_generator

__all__ = ["async_to_sync_generator"]
```

**Verification**:
```bash
uv run python -c "from src.app import async_to_sync_generator; print('OK')"
```

---

### Feature 1.5: Add Streaming Unit Tests

**Status**: ⬜ Not Started

**Task**: Create unit tests for async-to-sync bridge.

**Create `tests/test_app/test_streaming.py`**:
```python
"""Tests for async-to-sync streaming bridge."""
import pytest
from src.app.streaming import async_to_sync_generator

async def mock_async_generator():
    """Simple async generator for testing."""
    for i in range(3):
        yield f"item_{i}"

async def mock_async_generator_with_error():
    """Async generator that raises an error."""
    yield "item_0"
    raise ValueError("Test error")

def test_async_to_sync_basic():
    """Test basic async-to-sync conversion."""
    items = list(async_to_sync_generator(mock_async_generator))
    assert items == ["item_0", "item_1", "item_2"]

def test_async_to_sync_error_propagation():
    """Test that errors are propagated correctly."""
    gen = async_to_sync_generator(mock_async_generator_with_error)
    assert next(gen) == "item_0"
    with pytest.raises(ValueError, match="Test error"):
        next(gen)
```

**Verification**:
```bash
uv run pytest tests/test_app/test_streaming.py -v
```

---

### Feature 1.6: Streamlit Integration Testing

**Status**: ⬜ Not Started

**Task**: Validate Streamlit app with agent integration.

**Level 1: Unit Tests**:
```bash
uv run pytest tests/test_app/test_streaming.py -v
```

**Level 2: Local App Smoke Test**:
```bash
uv run streamlit run src/app/main.py --server.port 8501
# Open http://localhost:8501
# Verify: Page loads, sidebar renders, chat input works
```

**Level 3: Integration with Agent**:
1. Enter real experiment ID in sidebar
2. Submit: "List the most recent 5 traces"
3. Verify tool calls appear (🔧 indicators)
4. Verify streaming text response
5. Check MLflow for agent trace

**Success Criteria**:
- [ ] No "Event loop is closed" errors
- [ ] Streaming shows incremental text
- [ ] Tool use indicators appear
- [ ] Session continuity works (follow-up questions reference prior context)

---

## Phase 2: Files SDK Integration

**Goal**: Add MCP tools for UC Volumes and Workspace Files
**Estimated Effort**: Medium
**Dependencies**: Phase 0 complete
**Risk**: SDK authentication complexity

### Feature 2.1: Create Files SDK Wrapper

**Status**: ⬜ Not Started

**Task**: Create Databricks Files SDK wrapper in core module.

**Create `src/core/files.py`**:
```python
"""Databricks Files SDK wrapper for UC Volumes and Workspace Files."""
import io
import logging
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, ObjectType

logger = logging.getLogger(__name__)

def get_workspace_client(user_token: Optional[str] = None) -> WorkspaceClient:
    """Get WorkspaceClient with optional user-level token for OBO."""
    if user_token:
        return WorkspaceClient(token=user_token, auth_type="pat")
    return WorkspaceClient()  # Uses environment auth

# ============ UC VOLUMES ============

def uc_volume_read(path: str, user_token: Optional[str] = None) -> str:
    """Read text file from UC Volume."""
    w = get_workspace_client(user_token)
    with w.files.download(path) as f:
        return f.read().decode('utf-8')

def uc_volume_write(
    path: str,
    content: str,
    user_token: Optional[str] = None,
    overwrite: bool = True
) -> None:
    """Write text file to UC Volume."""
    w = get_workspace_client(user_token)
    w.files.upload(
        file_path=path,
        contents=io.BytesIO(content.encode('utf-8')),
        overwrite=overwrite
    )

def uc_volume_list(path: str, user_token: Optional[str] = None) -> list[dict]:
    """List UC Volume directory contents."""
    w = get_workspace_client(user_token)
    return [
        {
            "name": entry.name,
            "is_directory": entry.is_directory,
            "size": entry.file_size if not entry.is_directory else None
        }
        for entry in w.files.list_directory_contents(path)
    ]

def uc_volume_exists(path: str, user_token: Optional[str] = None) -> bool:
    """Check if file/directory exists in UC Volume."""
    w = get_workspace_client(user_token)
    try:
        w.files.get_status(path)
        return True
    except Exception:
        return False

def uc_volume_delete(path: str, user_token: Optional[str] = None) -> None:
    """Delete file from UC Volume."""
    w = get_workspace_client(user_token)
    w.files.delete(path)

# ============ WORKSPACE FILES ============

def workspace_read(path: str, user_token: Optional[str] = None) -> str:
    """Read file from Databricks Workspace."""
    w = get_workspace_client(user_token)
    with w.workspace.download(path) as f:
        return f.read().decode('utf-8')

def workspace_write(
    path: str,
    content: str,
    user_token: Optional[str] = None,
    overwrite: bool = True
) -> None:
    """Write file to Databricks Workspace."""
    w = get_workspace_client(user_token)
    w.workspace.upload(
        path=path,
        content=io.BytesIO(content.encode('utf-8')),
        overwrite=overwrite,
        format=ImportFormat.AUTO
    )

def workspace_list(path: str, user_token: Optional[str] = None) -> list[dict]:
    """List Workspace directory contents."""
    w = get_workspace_client(user_token)
    return [
        {
            "path": obj.path,
            "type": obj.object_type.value if obj.object_type else "UNKNOWN"
        }
        for obj in w.workspace.list(path)
    ]

def workspace_mkdir(path: str, user_token: Optional[str] = None) -> None:
    """Create directory in Workspace."""
    w = get_workspace_client(user_token)
    w.workspace.mkdirs(path)
```

**Verification**:
```bash
uv run python -c "from src.core.files import uc_volume_read, workspace_read; print('OK')"
```

---

### Feature 2.2: Add File Tools to Agent

**Status**: ⬜ Not Started

**Task**: Add MCP tools for file operations.

**Update `src/agent/tools.py`** - Add after existing tools:
```python
# ============ FILE TOOLS ============

@mlflow.trace(name="tool_uc_volume_read", span_type="TOOL")
@tool("uc_volume_read", "Read file from Unity Catalog Volume", {"path": str})
async def uc_volume_read_tool(args: dict) -> dict:
    """Read text file from UC Volume."""
    from ..core.files import uc_volume_read
    path = args.get("path", "")
    if not path:
        return mlflow_ops.text_result("[Files] Error: path required")
    try:
        content = uc_volume_read(path)
        result = mlflow_ops.text_result(content)
        record_tool_call("uc_volume_read", len(str(args)), len(content))
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Files] Error: {e}")

@mlflow.trace(name="tool_uc_volume_write", span_type="TOOL")
@tool("uc_volume_write", "Write file to Unity Catalog Volume", {
    "path": str, "content": str, "overwrite": bool
})
async def uc_volume_write_tool(args: dict) -> dict:
    """Write text file to UC Volume."""
    from ..core.files import uc_volume_write
    path = args.get("path", "")
    content = args.get("content", "")
    overwrite = args.get("overwrite", True)
    if not path:
        return mlflow_ops.text_result("[Files] Error: path required")
    try:
        uc_volume_write(path, content, overwrite=overwrite)
        result = mlflow_ops.text_result(f"✅ Written to {path}")
        record_tool_call("uc_volume_write", len(content), 50)
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Files] Error: {e}")

@mlflow.trace(name="tool_uc_volume_list", span_type="TOOL")
@tool("uc_volume_list", "List Unity Catalog Volume directory", {"path": str})
async def uc_volume_list_tool(args: dict) -> dict:
    """List contents of UC Volume directory."""
    from ..core.files import uc_volume_list
    import json
    path = args.get("path", "")
    if not path:
        return mlflow_ops.text_result("[Files] Error: path required")
    try:
        entries = uc_volume_list(path)
        result = mlflow_ops.text_result(json.dumps(entries, indent=2))
        record_tool_call("uc_volume_list", len(str(args)), len(str(entries)))
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Files] Error: {e}")

@mlflow.trace(name="tool_workspace_read", span_type="TOOL")
@tool("workspace_read", "Read file from Databricks Workspace", {"path": str})
async def workspace_read_tool(args: dict) -> dict:
    """Read file from Databricks Workspace."""
    from ..core.files import workspace_read
    path = args.get("path", "")
    if not path:
        return mlflow_ops.text_result("[Workspace] Error: path required")
    try:
        content = workspace_read(path)
        result = mlflow_ops.text_result(content)
        record_tool_call("workspace_read", len(str(args)), len(content))
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Workspace] Error: {e}")

@mlflow.trace(name="tool_workspace_write", span_type="TOOL")
@tool("workspace_write", "Write file to Databricks Workspace", {
    "path": str, "content": str, "overwrite": bool
})
async def workspace_write_tool(args: dict) -> dict:
    """Write file to Databricks Workspace."""
    from ..core.files import workspace_write
    path = args.get("path", "")
    content = args.get("content", "")
    overwrite = args.get("overwrite", True)
    if not path:
        return mlflow_ops.text_result("[Workspace] Error: path required")
    try:
        workspace_write(path, content, overwrite=overwrite)
        result = mlflow_ops.text_result(f"✅ Written to {path}")
        record_tool_call("workspace_write", len(content), 50)
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Workspace] Error: {e}")

@mlflow.trace(name="tool_workspace_list", span_type="TOOL")
@tool("workspace_list", "List Databricks Workspace directory", {"path": str})
async def workspace_list_tool(args: dict) -> dict:
    """List contents of Workspace directory."""
    from ..core.files import workspace_list
    import json
    path = args.get("path", "")
    if not path:
        return mlflow_ops.text_result("[Workspace] Error: path required")
    try:
        entries = workspace_list(path)
        result = mlflow_ops.text_result(json.dumps(entries, indent=2))
        record_tool_call("workspace_list", len(str(args)), len(str(entries)))
        return result
    except Exception as e:
        return mlflow_ops.text_result(f"[Workspace] Error: {e}")
```

**Update `create_tools()` return**:
```python
return [
    mlflow_query_tool,
    mlflow_annotate_tool,
    save_findings_tool,
    # File tools
    uc_volume_read_tool,
    uc_volume_write_tool,
    uc_volume_list_tool,
    workspace_read_tool,
    workspace_write_tool,
    workspace_list_tool,
]
```

---

### Feature 2.3: Update MCPTools Constants

**Status**: ⬜ Not Started

**Task**: Add tool name constants for file tools.

**Update `src/agent/tools.py`**:
```python
class MCPTools:
    """Tool names for the MCP server."""
    # MLflow tools
    MLFLOW_QUERY = f"mcp__{MCP_SERVER_NAME}__mlflow_query"
    MLFLOW_ANNOTATE = f"mcp__{MCP_SERVER_NAME}__mlflow_annotate"
    SAVE_FINDINGS = f"mcp__{MCP_SERVER_NAME}__save_findings"

    # File tools
    UC_VOLUME_READ = f"mcp__{MCP_SERVER_NAME}__uc_volume_read"
    UC_VOLUME_WRITE = f"mcp__{MCP_SERVER_NAME}__uc_volume_write"
    UC_VOLUME_LIST = f"mcp__{MCP_SERVER_NAME}__uc_volume_list"
    WORKSPACE_READ = f"mcp__{MCP_SERVER_NAME}__workspace_read"
    WORKSPACE_WRITE = f"mcp__{MCP_SERVER_NAME}__workspace_write"
    WORKSPACE_LIST = f"mcp__{MCP_SERVER_NAME}__workspace_list"
```

---

### Feature 2.4: Update Agent Allowed Tools

**Status**: ⬜ Not Started

**Task**: Add file tools to agent's allowed_tools list.

**Update `src/agent/agent.py`** in `_build_options()`:
```python
allowed_tools=[
    # Built-in Claude tools
    BuiltinTools.READ,
    BuiltinTools.BASH,
    BuiltinTools.GLOB,
    BuiltinTools.GREP,
    BuiltinTools.SKILL,
    # MLflow tools
    MCPTools.MLFLOW_QUERY,
    MCPTools.MLFLOW_ANNOTATE,
    MCPTools.SAVE_FINDINGS,
    # File tools
    MCPTools.UC_VOLUME_READ,
    MCPTools.UC_VOLUME_WRITE,
    MCPTools.UC_VOLUME_LIST,
    MCPTools.WORKSPACE_READ,
    MCPTools.WORKSPACE_WRITE,
    MCPTools.WORKSPACE_LIST,
],
```

**Verification**:
```bash
uv run python -c "from src.agent.tools import create_tools; print(len(create_tools()), 'tools')"
# Should print "9 tools"
```

---

### Feature 2.5: Add Files SDK Tests

**Status**: ⬜ Not Started

**Task**: Create unit tests for files SDK with mocks.

**Create `tests/test_core/test_files.py`**:
```python
"""Tests for Databricks Files SDK wrapper."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.files import (
    uc_volume_read, uc_volume_write, uc_volume_list,
    workspace_read, workspace_write, workspace_list
)

@pytest.fixture
def mock_workspace_client():
    """Create mock WorkspaceClient."""
    with patch('src.core.files.WorkspaceClient') as mock:
        client = Mock()
        mock.return_value = client
        yield client

class TestUCVolume:
    def test_uc_volume_read(self, mock_workspace_client):
        """Test reading from UC Volume."""
        mock_file = MagicMock()
        mock_file.read.return_value = b"test content"
        mock_workspace_client.files.download.return_value.__enter__ = Mock(return_value=mock_file)
        mock_workspace_client.files.download.return_value.__exit__ = Mock(return_value=False)

        result = uc_volume_read("/Volumes/cat/schema/vol/test.txt")
        assert result == "test content"

    def test_uc_volume_list(self, mock_workspace_client):
        """Test listing UC Volume directory."""
        mock_entry = Mock()
        mock_entry.name = "file.txt"
        mock_entry.is_directory = False
        mock_entry.file_size = 100
        mock_workspace_client.files.list_directory_contents.return_value = [mock_entry]

        result = uc_volume_list("/Volumes/cat/schema/vol/")
        assert len(result) == 1
        assert result[0]["name"] == "file.txt"

class TestWorkspace:
    def test_workspace_read(self, mock_workspace_client):
        """Test reading from Workspace."""
        mock_file = MagicMock()
        mock_file.read.return_value = b"workspace content"
        mock_workspace_client.workspace.download.return_value.__enter__ = Mock(return_value=mock_file)
        mock_workspace_client.workspace.download.return_value.__exit__ = Mock(return_value=False)

        result = workspace_read("/Workspace/Users/test/file.py")
        assert result == "workspace content"
```

**Verification**:
```bash
uv run pytest tests/test_core/test_files.py -v
```

---

### Feature 2.6: Files SDK Integration Testing

**Status**: ⬜ Not Started

**Task**: Validate Files SDK with real Databricks connectivity.

**Level 1: Unit Tests**:
```bash
uv run pytest tests/test_core/test_files.py -v
```

**Level 2: SDK Connectivity** (requires Databricks auth):
```python
from src.core.files import uc_volume_write, uc_volume_read, uc_volume_list
uc_volume_write("/Volumes/catalog/schema/vol/test.txt", "hello")
assert uc_volume_read("/Volumes/catalog/schema/vol/test.txt") == "hello"
print(uc_volume_list("/Volumes/catalog/schema/vol/"))
```

**Level 3: Tool Integration**:
```bash
uv run python -m src.cli test tools uc_volume_list --path /Volumes/catalog/schema/vol/
```

**Success Criteria**:
- [ ] Unit tests pass with mocked WorkspaceClient
- [ ] Real SDK operations succeed (write/read/list)
- [ ] Tool wrapper returns correct format

---

## Phase 3: Runtime & Auth

**Goal**: Add DATABRICKS_APP runtime context and OAuth support
**Estimated Effort**: Small
**Dependencies**: Phase 0 complete
**Risk**: Low

### Feature 3.1: Extend Runtime Detection

**Status**: ⬜ Not Started

**Task**: Add DATABRICKS_APP to RuntimeContext enum.

**Update `src/core/runtime.py`**:
```python
class RuntimeContext(Enum):
    """Runtime execution context."""
    LOCAL = "local"
    DATABRICKS_JOB = "databricks_job"
    DATABRICKS_APP = "databricks_app"  # NEW

def detect_runtime() -> RuntimeInfo:
    """Detect the current runtime environment."""
    volume_path = os.getenv("MLFLOW_AGENT_VOLUME_PATH")

    # Check for Databricks App FIRST (most specific)
    if os.getenv("DATABRICKS_APP_NAME"):
        logger.info(f"Detected Databricks App: {os.getenv('DATABRICKS_APP_NAME')}")
        return RuntimeInfo(
            context=RuntimeContext.DATABRICKS_APP,
            volume_path=volume_path,
        )

    # Check Databricks job context
    is_job = (
        os.getenv("DB_IS_JOB", "").upper() == "TRUE" or
        os.getenv("DATABRICKS_RUNTIME_VERSION") is not None or
        Path("/databricks").exists()
    )

    if is_job:
        # ... existing job detection ...

    # Local development
    return RuntimeInfo(context=RuntimeContext.LOCAL, volume_path=volume_path)
```

**Update `RuntimeInfo.is_databricks` property**:
```python
@property
def is_databricks(self) -> bool:
    """Check if running in Databricks."""
    return self.context in (RuntimeContext.DATABRICKS_JOB, RuntimeContext.DATABRICKS_APP)
```

---

### Feature 3.2: Create App Auth Module

**Status**: ⬜ Not Started

**Task**: Create authentication helper for Streamlit app.

**Create `src/app/auth.py`**:
```python
"""Authentication helpers for Streamlit app."""
import logging
import streamlit as st
from typing import Optional
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

def get_current_user() -> Optional[dict]:
    """Get current user info from Databricks.

    Returns:
        Dict with user_name and display_name, or None if not authenticated.
    """
    try:
        w = WorkspaceClient()
        me = w.current_user.me()
        return {
            "user_name": me.user_name,
            "display_name": me.display_name or me.user_name,
            "id": me.id
        }
    except Exception as e:
        logger.warning(f"Could not get current user: {e}")
        return None

def configure_user_context() -> None:
    """Configure auth context and store user info in session state."""
    if "user" not in st.session_state:
        user = get_current_user()
        if user:
            st.session_state.user = user
            logger.info(f"Authenticated as: {user['user_name']}")
        else:
            st.session_state.user = None

def get_user_volume_path(base_volume: str) -> str:
    """Get user-specific volume path for session storage.

    Args:
        base_volume: Base volume path (e.g., /Volumes/catalog/schema/volume)

    Returns:
        User-specific path like /Volumes/.../users/{user_name}/
    """
    user = st.session_state.get("user")
    if user:
        return f"{base_volume}/users/{user['user_name']}"
    return f"{base_volume}/users/anonymous"

def require_auth():
    """Decorator/guard to require authentication."""
    configure_user_context()
    if not st.session_state.get("user"):
        st.error("Authentication required. Please ensure you're logged into Databricks.")
        st.stop()
```

---

### Feature 3.3: Add Runtime Tests

**Status**: ⬜ Not Started

**Task**: Create tests for new runtime detection.

**Update `tests/test_core/test_runtime.py`**:
```python
"""Tests for runtime detection."""
import pytest
import os
from unittest.mock import patch
from src.core.runtime import detect_runtime, RuntimeContext

class TestRuntimeDetection:
    def test_local_default(self):
        """Test local runtime is default."""
        with patch.dict(os.environ, {}, clear=True):
            runtime = detect_runtime()
            assert runtime.context == RuntimeContext.LOCAL

    def test_databricks_app_detection(self):
        """Test Databricks App detection."""
        with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "my-app"}):
            runtime = detect_runtime()
            assert runtime.context == RuntimeContext.DATABRICKS_APP

    def test_app_takes_precedence_over_job(self):
        """Test App context takes precedence over Job context."""
        with patch.dict(os.environ, {
            "DATABRICKS_APP_NAME": "my-app",
            "DB_IS_JOB": "TRUE"
        }):
            runtime = detect_runtime()
            assert runtime.context == RuntimeContext.DATABRICKS_APP

    def test_is_databricks_property(self):
        """Test is_databricks property includes both Job and App."""
        with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "my-app"}):
            runtime = detect_runtime()
            assert runtime.is_databricks is True
```

**Verification**:
```bash
uv run pytest tests/test_core/test_runtime.py -v
```

---

### Feature 3.4: Runtime Detection Testing

**Status**: ⬜ Not Started

**Task**: Validate runtime detection including new DATABRICKS_APP context.

**Level 1: Unit Tests**:
```bash
uv run pytest tests/test_core/test_runtime.py -v
```

**Level 2: Environment Mocking**:
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

**Success Criteria**:
- [ ] RuntimeContext.DATABRICKS_APP detection works
- [ ] APP takes precedence over JOB
- [ ] is_databricks property returns True for both APP and JOB

---

## Phase 4: Enhanced UI

**Goal**: Full-featured Streamlit UI with autonomous mode
**Estimated Effort**: Medium
**Dependencies**: Phases 1, 2, 3 complete
**Risk**: UI complexity

### Feature 4.1: Create Progress Component

**Status**: ⬜ Not Started

**Task**: Create reusable progress display component.

**Create `src/app/components/progress.py`**:
```python
"""Task progress display component."""
import streamlit as st
from ...agent.mlflow_ops import get_task_status

def render_task_progress():
    """Render task progress in a compact format."""
    status = get_task_status()
    if not status or status.get("total", 0) == 0:
        return

    total = status["total"]
    completed = status["completed"]
    failed = status.get("failed", 0)
    pending = status.get("pending", 0)

    # Progress bar
    progress = completed / total if total > 0 else 0
    st.progress(progress)

    # Status text
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed", f"{completed}/{total}")
    with col2:
        st.metric("Pending", pending)
    with col3:
        if failed > 0:
            st.metric("Failed", failed, delta_color="inverse")
        else:
            st.metric("Failed", 0)

def render_task_list():
    """Render full task list with details."""
    from ...agent.mlflow_ops import load_tasks
    tasks = load_tasks()
    if not tasks:
        st.info("No tasks found")
        return

    for task in tasks:
        status_icon = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌"
        }.get(task.get("status", "pending"), "❓")

        with st.expander(f"{status_icon} {task.get('type', 'unknown')} - {task.get('status', 'unknown')}"):
            st.json(task)
```

---

### Feature 4.2: Create Sidebar Component

**Status**: ⬜ Not Started

**Task**: Create reusable sidebar component.

**Create `src/app/components/sidebar.py`**:
```python
"""Sidebar configuration component."""
import streamlit as st
from ...core.runtime import detect_runtime
from ..auth import configure_user_context, get_current_user

def render_sidebar():
    """Render the configuration sidebar."""
    with st.sidebar:
        # User info
        user = get_current_user()
        if user:
            st.caption(f"👤 {user['display_name']}")

        st.header("Configuration")

        # Experiment ID
        experiment_id = st.text_input(
            "Experiment ID",
            value=st.session_state.get("experiment_id", ""),
            help="MLflow experiment ID to analyze"
        )
        st.session_state.experiment_id = experiment_id

        # Runtime info
        runtime = detect_runtime()
        st.caption(f"Runtime: {runtime.context.value}")

        st.divider()

        # Session controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()

        # Session info
        if st.session_state.get("session_id"):
            st.caption(f"Session: {st.session_state.session_id[:8]}...")
```

---

### Feature 4.3: Add Autonomous Mode Tab

**Status**: ⬜ Not Started

**Task**: Add autonomous mode functionality to main app.

**Update `src/app/main.py`** with tabs:
```python
# After imports, add:
from .components.sidebar import render_sidebar
from .components.progress import render_task_progress, render_task_list
from ..agent.autonomous import run_autonomous

# Replace sidebar code with:
render_sidebar()

# Main content with tabs
tab1, tab2 = st.tabs(["💬 Interactive", "🤖 Autonomous"])

with tab1:
    # ... existing chat interface ...

with tab2:
    st.header("Autonomous Evaluation")

    exp_id = st.text_input(
        "Experiment ID",
        value=st.session_state.get("experiment_id", ""),
        key="auto_exp_id"
    )

    col1, col2 = st.columns(2)
    with col1:
        max_iter = st.number_input("Max Iterations", min_value=1, max_value=50, value=10)
    with col2:
        st.empty()  # Placeholder for future options

    if st.button("▶️ Start Autonomous Run", type="primary"):
        if not exp_id:
            st.error("Experiment ID is required")
        else:
            st.session_state.auto_running = True
            # Run in background with progress updates
            with st.spinner("Running autonomous evaluation..."):
                # Note: This needs async handling - see Feature 4.5
                st.info("Autonomous mode running...")

    st.divider()

    # Progress display
    st.subheader("Progress")
    render_task_progress()

    with st.expander("Task Details"):
        render_task_list()
```

---

### Feature 4.4: Enhanced UI Testing

**Status**: ⬜ Not Started

**Task**: Validate enhanced UI components and autonomous mode tab.

**Level 1: Component Tests**:
```bash
# Test components render without error
uv run python -c "
from src.app.components.progress import render_task_progress, render_task_list
from src.app.components.sidebar import render_sidebar
print('Components import OK')
"
```

**Level 2: Manual UI Testing**:
1. `uv run streamlit run src/app/main.py`
2. Verify tabs: Interactive | Autonomous
3. Verify sidebar shows runtime info
4. Start autonomous run with real experiment
5. Verify progress bar updates

**Success Criteria**:
- [ ] Components render without errors
- [ ] Tabs switch correctly
- [ ] Progress updates in real-time
- [ ] Autonomous mode doesn't block UI

---

## Phase 5: Deployment

**Goal**: Deploy to Databricks Apps via DABs
**Estimated Effort**: Small
**Dependencies**: Phases 0-4 complete
**Risk**: Deployment configuration

### Feature 5.1: Create App Configuration

**Status**: ⬜ Not Started

**Task**: Create Databricks App configuration file.

**Create `app.yaml`**:
```yaml
# Databricks App Configuration
# https://docs.databricks.com/dev-tools/databricks-apps/

name: mlflow-eval-agent
description: Interactive MLflow trace analysis and evaluation agent

# Entry point command
command:
  - streamlit
  - run
  - src/app/main.py
  - --server.port=8080
  - --server.address=0.0.0.0
  - --server.headless=true

# Environment variables
env:
  - name: MLFLOW_TRACKING_URI
    value: databricks
  - name: MLFLOW_AGENT_VOLUME_PATH
    valueFrom:
      config: volume_path
  - name: ANTHROPIC_MODEL
    valueFrom:
      config: model

# App-level configuration (set at deploy time)
config:
  - name: volume_path
    description: Unity Catalog Volume path for session storage
    default: /Volumes/users/default/mlflow-eval-agent
  - name: experiment_id
    description: Default MLflow experiment ID
    default: ""
  - name: model
    description: Claude model to use
    default: databricks-claude-sonnet-4

# Resource requirements
resources:
  cpu: 2
  memory: 4Gi
```

---

### Feature 5.2: Extend DABs Configuration

**Status**: ⬜ Not Started

**Task**: Add apps section to databricks.yml.

**Update `databricks.yml`** - Add under resources:
```yaml
resources:
  # Existing jobs section...

  apps:
    eval_agent_app:
      name: "[${bundle.target}] MLflow Eval Agent App"
      description: "Streamlit UI for MLflow trace analysis and evaluation"
      source_code_path: .
      config:
        - name: volume_path
          value: "${var.volume_path}"
        - name: experiment_id
          value: "${var.experiment_id}"
        - name: model
          value: "${var.anthropic_model}"
```

---

### Feature 5.3: Test Deployment

**Status**: ⬜ Not Started

**Task**: Validate and deploy to Databricks.

**Commands**:
```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy (builds wheel and creates app)
databricks bundle deploy -t dev

# List apps to verify
databricks apps list

# Get app URL
databricks apps get mlflow-eval-agent
```

**Verification Checklist**:
- [ ] Bundle validates without errors
- [ ] App deploys successfully
- [ ] App URL is accessible
- [ ] OAuth login works
- [ ] Can query MLflow experiments
- [ ] Streaming responses work

---

### Feature 5.4: Full Deployment Validation

**Status**: ⬜ Not Started

**Task**: Comprehensive deployment validation before production release.

**Level 1: Bundle Validation**:
```bash
databricks bundle validate -t dev
```

**Level 2: Deploy & Verify**:
```bash
databricks bundle deploy -t dev
databricks apps list
databricks apps get mlflow-eval-agent
```

**Level 3: App Functional Testing**:
1. Open app URL from `databricks apps get` output
2. Verify OAuth login works
3. Submit test query
4. Verify streaming response
5. Check UC Volume for session files

**Level 4: End-to-End**:
1. Run autonomous mode from deployed app
2. Verify tasks complete
3. Run generated eval script
4. Check MLflow for all traces

**Success Criteria**:
- [ ] App accessible via Databricks URL
- [ ] OAuth authenticates correctly
- [ ] Full agent functionality works
- [ ] Session state persists to UC Volume

---

## Execution Guide

### Sub-Agent Strategy

| Phase | Recommended Approach |
|-------|---------------------|
| 0 (Restructure) | Single agent, sequential features |
| 1 (Streamlit) | Single agent |
| 2 (Files SDK) | Parallel: 2.1-2.3 can run together |
| 3 (Runtime) | Single agent, small scope |
| 4 (Enhanced UI) | Sequential, depends on prior phases |
| 5 (Deployment) | Single agent |

### Testing Strategy

**Testing Ladder** (Applied to Each Phase):

```
Level 1: Unit Tests (pytest)        → Fast, no external deps
Level 2: Import Verification        → Catches circular imports
Level 3: Mock Integration (--mock)  → Tests logic without MLflow
Level 4: Local Integration          → Real MLflow, local environment
Level 5: Autonomous Run             → Full agent loop locally
Level 6: Databricks Deployment      → Production validation (Phase 0 + Phase 5 only)
```

**After each feature**:
1. Run specific unit tests
2. Run import verification
3. Run full test suite for regressions

**After each phase**:
1. Run `/test-phase <phase-number>` for comprehensive validation
2. Manual verification where required
3. Update status in this file

**Before shipping**:
1. Run `/test-full-cycle <experiment-id>` for complete validation
2. Run `/validate-deployment --target dev` for deployment testing

### Error Debugging Workflow

When tests fail, use this pattern:

1. **Capture trace ID** from test output
2. **Analyze trace**: `/analyze-trace <trace-id>`
   - Check span errors
   - Identify bottlenecks
   - Review tool call sequence
3. **Analyze tokens**: `/analyze-tokens <trace-id>`
   - Check cache efficiency (>50% = good)
   - Identify context growth issues
4. **Fix issue** based on analysis
5. **Re-run same test** to verify fix
6. **Proceed** to next feature/phase

**Key Metrics to Watch**:

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Cache efficiency | >50% | 30-50% | <30% |
| Initializer cost | <$1.50 | $1.50-$2.50 | >$2.50 |
| Worker cost | <$0.75 | $0.75-$1.25 | >$1.25 |
| Context growth | Stable | Slow growth | Runaway |

### Progress Tracking

Update feature status using:
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- ❌ Blocked

---

## Appendix: File Reference

### Files by Phase

**Phase 0 (Restructure)**:
- `src/core/__init__.py`
- `src/core/config.py` (moved)
- `src/core/runtime.py` (moved)
- `src/core/auth.py` (moved + renamed)
- `src/agent/__init__.py`
- `src/agent/agent.py` (moved + modified)
- `src/agent/autonomous.py` (new)
- `src/agent/prompts.py` (new)
- `src/agent/tools.py` (moved)
- `src/agent/mlflow_ops.py` (moved)
- `src/cli.py` (modified imports)

**Phase 1 (Streamlit)**:
- `src/app/__init__.py`
- `src/app/main.py`
- `src/app/streaming.py`
- `pyproject.toml` (add streamlit)

**Phase 2 (Files SDK)**:
- `src/core/files.py` (new)
- `src/agent/tools.py` (add file tools)

**Phase 3 (Runtime)**:
- `src/core/runtime.py` (add APP context)
- `src/app/auth.py` (new)

**Phase 4 (Enhanced UI)**:
- `src/app/components/progress.py`
- `src/app/components/sidebar.py`
- `src/app/main.py` (add tabs)

**Phase 5 (Deployment)**:
- `app.yaml` (new)
- `databricks.yml` (add apps section)
