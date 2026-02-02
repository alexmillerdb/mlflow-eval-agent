# Phase 3: Runtime & Auth

**Status**: Complete (2026-02-02)
**Branch**: `fix-env-vars`

---

## Goal

Add DATABRICKS_APP runtime context and OAuth support for Streamlit apps.

**Effort**: Small
**Risk**: Low

---

## Feature 3.1: Extend Runtime Detection ✅

**File**: `src/core/runtime.py` (MODIFIED)

Added `DATABRICKS_APP` to RuntimeContext enum and detection logic:

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
    app_name = os.getenv("DATABRICKS_APP_NAME")
    if app_name:
        logger.info(f"Detected Databricks App: {app_name}")
        return RuntimeInfo(
            context=RuntimeContext.DATABRICKS_APP,
            volume_path=volume_path,
            app_name=app_name,
        )

    # Check Databricks job context
    is_job = (
        os.getenv("DB_IS_JOB", "").upper() == "TRUE" or
        os.getenv("DATABRICKS_RUNTIME_VERSION") is not None or
        Path("/databricks").exists()
    )

    if is_job:
        # ... job detection ...

    # Local development
    return RuntimeInfo(context=RuntimeContext.LOCAL, volume_path=volume_path)
```

Updated `RuntimeInfo.is_databricks` property:

```python
@property
def is_databricks(self) -> bool:
    """Check if running in Databricks."""
    return self.context in (RuntimeContext.DATABRICKS_JOB, RuntimeContext.DATABRICKS_APP)

@property
def session_prefix(self) -> str:
    """Generate session ID prefix based on context."""
    if self.context == RuntimeContext.DATABRICKS_JOB and self.job_id and self.job_run_id:
        return f"job-{self.job_id}-run-{self.job_run_id}"
    if self.context == RuntimeContext.DATABRICKS_APP and self.app_name:
        return f"app-{self.app_name}"
    return ""
```

---

## Feature 3.2: Create App Auth Module ✅

**File**: `src/app/auth.py` (NEW)

```python
"""Authentication helpers for Streamlit app."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_current_user() -> Optional[dict]:
    """Get current user info from Databricks.

    Uses the WorkspaceClient to fetch the current authenticated user's info.

    Returns:
        dict with user_name, display_name, id or None on failure.
    """
    try:
        from src.core.files import get_workspace_client

        client = get_workspace_client()
        me = client.current_user.me()
        return {
            "user_name": me.user_name,
            "display_name": me.display_name or me.user_name,
            "id": me.id,
        }
    except Exception as e:
        logger.warning(f"Could not get current user: {e}")
        return None

def configure_user_context() -> None:
    """Configure auth context and store user info in session state."""
    import streamlit as st

    if "user" not in st.session_state:
        user = get_current_user()
        if user:
            st.session_state.user = user
            logger.info(f"Authenticated as: {user['user_name']}")
        else:
            st.session_state.user = None

def get_user_volume_path(base_volume: str) -> str:
    """Get user-specific volume path for session storage."""
    import streamlit as st

    user = st.session_state.get("user")
    if user:
        return f"{base_volume}/users/{user['user_name']}"
    return f"{base_volume}/users/anonymous"

def require_auth():
    """Guard function to require authentication."""
    import streamlit as st

    configure_user_context()
    if not st.session_state.get("user"):
        st.error("Authentication required. Please ensure you're logged into Databricks.")
        st.stop()
```

**File**: `src/app/__init__.py` (UPDATED)

```python
"""Application layer for MLflow Evaluation Agent."""
from .auth import (
    configure_user_context,
    get_current_user,
    get_user_volume_path,
    require_auth,
)
from .streaming import async_to_sync_generator

__all__ = [
    "async_to_sync_generator",
    "configure_user_context",
    "get_current_user",
    "get_user_volume_path",
    "require_auth",
]
```

---

## Feature 3.3: Add Runtime Tests ✅

**File**: `tests/test_core/test_runtime.py` (NEW)

```python
"""Tests for runtime detection."""
import pytest
import os
from unittest.mock import patch
from src.core.runtime import detect_runtime, RuntimeContext, RuntimeInfo

class TestRuntimeContext:
    def test_enum_values(self):
        assert RuntimeContext.LOCAL.value == "local"
        assert RuntimeContext.DATABRICKS_JOB.value == "databricks_job"
        assert RuntimeContext.DATABRICKS_APP.value == "databricks_app"

class TestRuntimeInfo:
    def test_is_databricks_local(self):
        info = RuntimeInfo(context=RuntimeContext.LOCAL)
        assert info.is_databricks is False

    def test_is_databricks_job(self):
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_JOB)
        assert info.is_databricks is True

    def test_is_databricks_app(self):
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_APP)
        assert info.is_databricks is True

class TestDetectRuntime:
    def test_local_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.core.runtime.Path.exists", return_value=False):
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.LOCAL

    def test_databricks_app_detection(self):
        with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "my-app"}, clear=True):
            runtime = detect_runtime()
            assert runtime.context == RuntimeContext.DATABRICKS_APP

    def test_app_takes_precedence_over_job(self):
        with patch.dict(os.environ, {
            "DATABRICKS_APP_NAME": "my-app",
            "DB_IS_JOB": "TRUE"
        }, clear=True):
            runtime = detect_runtime()
            assert runtime.context == RuntimeContext.DATABRICKS_APP
```

**File**: `tests/test_app/test_auth.py` (NEW)

```python
"""Tests for app auth module."""
import pytest
from unittest.mock import patch, MagicMock

class TestGetCurrentUser:
    def test_get_current_user_success(self):
        # Tests user info retrieval from WorkspaceClient

    def test_get_current_user_failure(self):
        # Tests graceful failure when not authenticated

class TestConfigureUserContext:
    def test_configure_user_context_stores_user(self):
        # Tests session state storage

class TestGetUserVolumePath:
    def test_get_user_volume_path_authenticated(self):
        # Tests user-specific path generation

    def test_get_user_volume_path_anonymous(self):
        # Tests fallback to anonymous path

class TestRequireAuth:
    def test_require_auth_passes_when_authenticated(self):
        # Tests guard passes with user

    def test_require_auth_stops_when_not_authenticated(self):
        # Tests guard stops without user
```

---

## Verification Results

### Unit Tests ✅
```bash
uv run pytest tests/test_core/test_runtime.py -v
# All runtime tests pass

uv run pytest tests/test_app/test_auth.py -v
# All auth tests pass (11 tests)
```

---

## Files Changed

| File | Action |
|------|--------|
| `src/core/runtime.py` | Modified - added DATABRICKS_APP context |
| `src/app/auth.py` | Created - auth helpers |
| `src/app/__init__.py` | Modified - added auth exports |
| `tests/test_core/test_runtime.py` | Created - runtime tests |
| `tests/test_app/test_auth.py` | Created - auth tests |

---

## Success Criteria

- [x] RuntimeContext.DATABRICKS_APP detection works
- [x] APP takes precedence over JOB
- [x] is_databricks property returns True for both APP and JOB
- [x] session_prefix works for APP context
- [x] Auth helpers work with WorkspaceClient
- [x] All tests pass
