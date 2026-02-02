"""Tests for runtime detection."""
import os
from unittest.mock import patch

import pytest

from src.core.runtime import RuntimeContext, RuntimeInfo, detect_runtime


class TestRuntimeContext:
    """Tests for RuntimeContext enum."""

    def test_enum_values(self):
        """All expected context values exist."""
        assert RuntimeContext.LOCAL.value == "local"
        assert RuntimeContext.DATABRICKS_JOB.value == "databricks_job"
        assert RuntimeContext.DATABRICKS_APP.value == "databricks_app"


class TestRuntimeInfo:
    """Tests for RuntimeInfo dataclass."""

    def test_is_databricks_local(self):
        """is_databricks returns False for LOCAL."""
        info = RuntimeInfo(context=RuntimeContext.LOCAL)
        assert info.is_databricks is False

    def test_is_databricks_job(self):
        """is_databricks returns True for JOB."""
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_JOB)
        assert info.is_databricks is True

    def test_is_databricks_app(self):
        """is_databricks returns True for APP."""
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_APP)
        assert info.is_databricks is True

    def test_session_prefix_local(self):
        """Local context has empty session prefix."""
        info = RuntimeInfo(context=RuntimeContext.LOCAL)
        assert info.session_prefix == ""

    def test_session_prefix_job(self):
        """Job session prefix format is job-{id}-run-{id}."""
        info = RuntimeInfo(
            context=RuntimeContext.DATABRICKS_JOB,
            job_id="123",
            job_run_id="456",
        )
        assert info.session_prefix == "job-123-run-456"

    def test_session_prefix_job_missing_ids(self):
        """Job without IDs has empty session prefix."""
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_JOB)
        assert info.session_prefix == ""

    def test_session_prefix_app(self):
        """App session prefix format is app-{name}."""
        info = RuntimeInfo(
            context=RuntimeContext.DATABRICKS_APP,
            app_name="test-app",
        )
        assert info.session_prefix == "app-test-app"

    def test_session_prefix_app_missing_name(self):
        """App without name has empty session prefix."""
        info = RuntimeInfo(context=RuntimeContext.DATABRICKS_APP)
        assert info.session_prefix == ""


class TestDetectRuntime:
    """Tests for detect_runtime function."""

    def test_local_default(self):
        """LOCAL is default with clean environment."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.LOCAL

    def test_databricks_app_detection(self):
        """DATABRICKS_APP detected with DATABRICKS_APP_NAME."""
        with patch.dict(os.environ, {"DATABRICKS_APP_NAME": "my-app"}, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_APP
                assert runtime.app_name == "my-app"

    def test_app_takes_precedence_over_job(self):
        """APP context takes precedence over JOB context."""
        env = {
            "DATABRICKS_APP_NAME": "my-app",
            "DB_IS_JOB": "TRUE",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_APP

    def test_databricks_job_detection_db_is_job(self):
        """DATABRICKS_JOB detected with DB_IS_JOB."""
        env = {
            "DB_IS_JOB": "TRUE",
            "DB_JOB_ID": "123",
            "DB_JOB_RUN_ID": "456",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_JOB
                assert runtime.job_id == "123"
                assert runtime.job_run_id == "456"

    def test_databricks_job_detection_runtime_version(self):
        """DATABRICKS_JOB detected with DATABRICKS_RUNTIME_VERSION."""
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.0"}, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_JOB

    def test_databricks_job_detection_path_exists(self):
        """DATABRICKS_JOB detected when /databricks exists."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_JOB

    def test_volume_path_propagated(self):
        """MLFLOW_AGENT_VOLUME_PATH is captured in all contexts."""
        env = {"MLFLOW_AGENT_VOLUME_PATH": "/Volumes/test/path"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.volume_path == "/Volumes/test/path"

    def test_volume_path_with_app(self):
        """Volume path is captured for app context."""
        env = {
            "DATABRICKS_APP_NAME": "my-app",
            "MLFLOW_AGENT_VOLUME_PATH": "/Volumes/test/path",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.core.runtime.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                runtime = detect_runtime()
                assert runtime.context == RuntimeContext.DATABRICKS_APP
                assert runtime.volume_path == "/Volumes/test/path"
