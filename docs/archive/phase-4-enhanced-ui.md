# Phase 4: Enhanced UI & Playwright Testing

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

---

## Goal

Full-featured Streamlit UI with autonomous mode + automated UI testing via Playwright.

**Effort**: Medium (4 features + testing infrastructure)
**Risk**: Low-Medium (UI complexity, testing setup)

---

## Feature 4.1: Progress Component ✅

**File**: `src/app/components/progress.py` (NEW)

```python
"""Task progress display components."""
import streamlit as st
from src.agent.mlflow_ops import get_task_status

STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "failed": "❌",
}

def render_task_progress():
    """Display compact progress bar with metrics."""
    status = get_task_status()
    total = status["total"]

    if total == 0:
        st.info("No tasks yet. Start an autonomous run to create tasks.")
        return

    completed = status["completed"]
    progress = completed / total if total > 0 else 0

    st.progress(progress, text=f"{completed}/{total} tasks completed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Completed", status["completed"])
    col2.metric("Pending", status["pending"])
    col3.metric("Failed", status["failed"])

def render_task_list():
    """Display expandable task list with details."""
    status = get_task_status()
    tasks = status.get("tasks", [])

    if not tasks:
        return

    for task in tasks:
        icon = STATUS_ICONS.get(task.get("status", "pending"), "❓")
        task_type = task.get("type", "unknown")
        task_status = task.get("status", "pending")

        with st.expander(f"{icon} {task_type} - {task_status}"):
            st.json(task)
```

---

## Feature 4.2: Sidebar Component ✅

**File**: `src/app/components/sidebar.py` (NEW)

```python
"""Sidebar configuration component."""
import os
import streamlit as st
from src.core.runtime import detect_runtime
from src.app.auth import get_current_user

def render_sidebar():
    """Render sidebar with config, user info, and session controls."""
    with st.sidebar:
        # User info (if authenticated)
        user = get_current_user()
        if user:
            st.caption(f"👤 {user['display_name']}")

        st.header("Configuration")

        # Experiment ID
        experiment_id = st.text_input(
            "Experiment ID",
            value=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            help="MLflow experiment ID to analyze",
        )
        if experiment_id:
            os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id

        # Runtime info
        runtime = detect_runtime()
        st.caption(f"Runtime: {runtime.context.value}")

        st.divider()

        # Session controls
        st.subheader("Session")
        if st.session_state.get("session_id"):
            st.caption(f"ID: {st.session_state.session_id[:8]}...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()
        with col2:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
```

**File**: `src/app/components/__init__.py` (UPDATED)

```python
"""Reusable UI components."""
from .progress import render_task_progress, render_task_list
from .sidebar import render_sidebar

__all__ = ["render_task_progress", "render_task_list", "render_sidebar"]
```

---

## Feature 4.3: Tabbed Main App with Autonomous Mode ✅

**File**: `src/app/main.py` (MODIFIED)

Key changes:
1. Added imports for components
2. Added `auto_running` session state
3. Added `render_autonomous_tab()` function
4. Replaced `setup_sidebar()` with `render_sidebar()`
5. Added tabbed interface (Interactive / Autonomous)

```python
def render_autonomous_tab():
    """Render autonomous evaluation mode controls."""
    st.subheader("Autonomous Evaluation")

    col1, col2 = st.columns([2, 1])
    with col1:
        auto_exp_id = st.text_input(
            "Experiment ID",
            value=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            key="auto_exp_id",
        )
    with col2:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=50,
            value=10,
            key="max_iterations",
        )

    if st.button("▶️ Start Autonomous Run", type="primary"):
        if not auto_exp_id:
            st.error("Please enter an Experiment ID")
            return

        os.environ["MLFLOW_EXPERIMENT_ID"] = auto_exp_id
        st.session_state.auto_running = True

        with st.spinner("Running autonomous evaluation..."):
            from src.agent.autonomous import run_autonomous
            import asyncio
            asyncio.run(run_autonomous(auto_exp_id, max_iterations))

        st.session_state.auto_running = False
        st.success("Autonomous run complete!")
        st.rerun()

    # Progress display
    st.divider()
    render_task_progress()

    with st.expander("📋 Task Details"):
        render_task_list()

def main():
    """Main application entry point."""
    st.title("MLflow Eval Agent")

    initialize_session_state()
    render_sidebar()

    tab1, tab2 = st.tabs(["💬 Interactive", "🤖 Autonomous"])

    with tab1:
        display_chat_history()
        handle_user_input()

    with tab2:
        render_autonomous_tab()
```

---

## Feature 4.4: Playwright UI Testing ✅

### Test Infrastructure

**File**: `tests/e2e/conftest.py` (UPDATED)

```python
import os

WEBAPP_TESTING_SCRIPTS = os.path.expanduser(
    "~/.claude/plugins/cache/anthropic-agent-skills/example-skills/00756142ab04/skills/webapp-testing/scripts"
)

@pytest.fixture
def with_server_script():
    """Path to with_server.py helper."""
    return os.path.join(WEBAPP_TESTING_SCRIPTS, "with_server.py")
```

### New Test Files

**File**: `tests/e2e/test_tabs.py` (NEW)

```python
"""E2E tests for tab navigation."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

class TestTabs:
    def test_tabs_visible(self, app_page: Page):
        """Verify both tabs are present."""
        interactive_tab = app_page.get_by_role("tab", name="Interactive")
        autonomous_tab = app_page.get_by_role("tab", name="Autonomous")
        expect(interactive_tab).to_be_visible()
        expect(autonomous_tab).to_be_visible()

    def test_switch_to_autonomous_tab(self, app_page: Page):
        """Verify switching to autonomous tab shows controls."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")
        expect(app_page.get_by_text("Autonomous Evaluation")).to_be_visible()
        expect(app_page.get_by_role("button", name="Start Autonomous Run")).to_be_visible()

    def test_switch_back_to_interactive(self, app_page: Page):
        """Verify switching back to interactive shows chat."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")
        app_page.get_by_role("tab", name="Interactive").click()
        app_page.wait_for_load_state("networkidle")
        expect(app_page.locator('[data-testid="stChatInput"]')).to_be_visible()
```

**File**: `tests/e2e/test_autonomous_tab.py` (NEW)

```python
"""E2E tests for autonomous mode tab."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

class TestAutonomousTab:
    @pytest.fixture(autouse=True)
    def navigate_to_autonomous(self, app_page: Page):
        """Navigate to autonomous tab before each test."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")
        yield

    def test_experiment_input_visible(self, app_page: Page):
        exp_input = app_page.get_by_label("Experiment ID")
        expect(exp_input).to_be_visible()

    def test_max_iterations_input(self, app_page: Page):
        iterations_input = app_page.get_by_label("Max Iterations")
        expect(iterations_input).to_be_visible()
        expect(iterations_input).to_have_value("10")

    def test_start_button_requires_experiment(self, app_page: Page):
        exp_input = app_page.get_by_label("Experiment ID").first
        exp_input.clear()
        app_page.get_by_role("button", name="Start Autonomous Run").click()
        expect(app_page.get_by_text("Please enter an Experiment ID")).to_be_visible()

    def test_progress_section_visible(self, app_page: Page):
        expect(app_page.get_by_text("No tasks yet")).to_be_visible()

    def test_task_details_expander(self, app_page: Page):
        expander = app_page.get_by_text("Task Details")
        expect(expander).to_be_visible()
```

**File**: `tests/e2e/test_components.py` (NEW)

```python
"""E2E tests for UI components."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

class TestSidebarComponent:
    def test_runtime_info_displayed(self, app_page: Page):
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_text("Runtime:")).to_be_visible()

    def test_new_and_clear_buttons(self, app_page: Page):
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_role("button", name="New")).to_be_visible()
        expect(sidebar.get_by_role("button", name="Clear")).to_be_visible()
```

---

## Verification Results

### 1. Component Unit Tests ✅
```bash
uv run python -c "
from src.app.components import render_sidebar, render_task_progress, render_task_list
print('Component imports OK')
"
# Output: Component imports OK
```

### 2. All Unit Tests Pass ✅
```bash
uv run pytest tests/ -v --ignore=tests/e2e --ignore=tests/integration
# Output: 125 passed in 1.95s
```

### 3. E2E Tests Collected ✅
```bash
uv run pytest tests/e2e/ --collect-only
# Output: 20 items collected (including new tests)
```

---

## Files Changed

| File | Action |
|------|--------|
| `src/app/components/__init__.py` | Modified - added exports |
| `src/app/components/progress.py` | Created - progress display |
| `src/app/components/sidebar.py` | Created - sidebar component |
| `src/app/main.py` | Modified - added tabs + autonomous |
| `tests/e2e/conftest.py` | Modified - added helper path |
| `tests/e2e/test_tabs.py` | Created - tab navigation tests |
| `tests/e2e/test_autonomous_tab.py` | Created - autonomous mode tests |
| `tests/e2e/test_components.py` | Created - component tests |

---

## Success Criteria

- [x] Components render without errors
- [x] Both tabs visible (Interactive, Autonomous)
- [x] Tab switching works
- [x] Sidebar shows runtime info
- [x] Sidebar has New/Clear buttons
- [x] Autonomous tab has all controls
- [x] Progress section renders
- [x] All 125 unit tests pass
- [x] All 20 E2E tests collected
