# MLflow Eval Agent - Databricks Apps Implementation Spec

> **Purpose**: Phased implementation plan for restructuring codebase and adding Streamlit-based Databricks Apps deployment. Designed for modular execution by Claude Code sub-agents.

**Created**: 2024-01-30
**Status**: Phase 2 Complete
**Current Phase**: Ready for Phase 3

---

## Quick Reference

| Phase | Name | Status | Features |
|-------|------|--------|----------|
| 0 | Codebase Restructure | ✅ Complete | 0.1-0.11 |
| 1 | Streamlit MVP | ✅ Complete | 1.1-1.6 |
| 2 | Files SDK Integration | ✅ Complete | 2.1-2.6 |
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

## Phase 0: Codebase Restructure ✅

**Status**: Complete (2026-01-30)
**Commit**: `a6c945c`

Reorganized flat `src/` into `core/` + `agent/` + `app/` hierarchy.

[Full details → docs/archive/phase-0-restructure.md](docs/archive/phase-0-restructure.md)

---

## Phase 1: Streamlit MVP ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Basic Streamlit chat interface with Claude Agent SDK streaming via async-to-sync bridge.

[Full details → docs/archive/phase-1-streamlit-mvp.md](docs/archive/phase-1-streamlit-mvp.md)

---

## Phase 2: Files SDK Integration ✅

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

Added MCP tools for UC Volumes and Workspace Files (6 new tools, 9 total). Includes unit tests (29) and integration tests (7).

[Full details → docs/archive/phase-2-files-sdk.md](docs/archive/phase-2-files-sdk.md)

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

### Phase Archiving

When a phase is complete, archive it to keep this spec focused on remaining work:

1. **Create archive file**: `docs/archive/phase-N-name.md`
   - Copy full phase content (goal, features, verification steps)
   - Mark success criteria as checked
   - Add completion date and commit hash

2. **Update this spec**: Replace phase details with summary:
   ```markdown
   ## Phase N: Name ✅

   **Status**: Complete (YYYY-MM-DD)
   **Commit**: `abc1234`

   One-line description of what was accomplished.

   [Full details → docs/archive/phase-N-name.md](docs/archive/phase-N-name.md)
   ```

3. **Verify**: Ensure relative link works and main spec is under ~1,600 lines

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
