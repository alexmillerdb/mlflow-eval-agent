"""Tests for context optimization features.

TDD tests for:
- Task retry limit functions (get_task_attempts, increment_task_attempts)
- ContextMetrics class for monitoring
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# =============================================================================
# TASK RETRY LIMIT TESTS
# =============================================================================

class TestMaxTaskAttemptsConstant:
    """Tests for MAX_TASK_ATTEMPTS constant."""

    def test_max_attempts_constant_exists(self):
        """MAX_TASK_ATTEMPTS should be defined."""
        from src.mlflow_ops import MAX_TASK_ATTEMPTS
        assert MAX_TASK_ATTEMPTS is not None

    def test_max_attempts_constant_is_5(self):
        """MAX_TASK_ATTEMPTS should be 5."""
        from src.mlflow_ops import MAX_TASK_ATTEMPTS
        assert MAX_TASK_ATTEMPTS == 5


class TestGetTaskAttempts:
    """Tests for get_task_attempts function."""

    def test_get_attempts_no_file(self, tmp_path):
        """Returns 0 when tasks file doesn't exist."""
        from src.mlflow_ops import get_task_attempts, set_session_dir

        set_session_dir(tmp_path)
        attempts = get_task_attempts(1)
        assert attempts == 0

    def test_get_attempts_task_without_attempts_field(self, tmp_path):
        """Returns 0 when task exists but has no attempts field."""
        from src.mlflow_ops import get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending"}
            ]
        }))

        attempts = get_task_attempts(1)
        assert attempts == 0

    def test_get_attempts_returns_stored_value(self, tmp_path):
        """Returns stored attempts count when present."""
        from src.mlflow_ops import get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 3}
            ]
        }))

        attempts = get_task_attempts(1)
        assert attempts == 3

    def test_get_attempts_task_not_found(self, tmp_path):
        """Returns 0 when task ID doesn't exist."""
        from src.mlflow_ops import get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 3}
            ]
        }))

        attempts = get_task_attempts(999)
        assert attempts == 0

    def test_get_attempts_handles_list_format(self, tmp_path):
        """Handles tasks file with list format (no wrapper dict)."""
        from src.mlflow_ops import get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps([
            {"id": 1, "name": "Test task", "status": "pending", "attempts": 2}
        ]))

        attempts = get_task_attempts(1)
        assert attempts == 2


class TestIncrementTaskAttempts:
    """Tests for increment_task_attempts function."""

    def test_increment_first_attempt(self, tmp_path):
        """First attempt sets count to 1, returns True."""
        from src.mlflow_ops import increment_task_attempts, get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending"}
            ]
        }))

        result = increment_task_attempts(1)
        assert result is True
        assert get_task_attempts(1) == 1

    def test_increment_within_limit(self, tmp_path):
        """Incrementing within limit returns True."""
        from src.mlflow_ops import increment_task_attempts, get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 3}
            ]
        }))

        result = increment_task_attempts(1)
        assert result is True
        assert get_task_attempts(1) == 4

    def test_increment_at_limit_returns_true(self, tmp_path):
        """Incrementing to exactly max (5) still returns True."""
        from src.mlflow_ops import increment_task_attempts, get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 4}
            ]
        }))

        result = increment_task_attempts(1)
        assert result is True
        assert get_task_attempts(1) == 5

    def test_increment_exceeds_limit_returns_false(self, tmp_path):
        """Incrementing beyond limit returns False."""
        from src.mlflow_ops import increment_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 5}
            ]
        }))

        result = increment_task_attempts(1)
        assert result is False

    def test_increment_exceeds_limit_marks_failed(self, tmp_path):
        """Task is marked as failed when exceeding limit."""
        from src.mlflow_ops import increment_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending", "attempts": 5}
            ]
        }))

        increment_task_attempts(1)

        # Re-read and check status
        data = json.loads(tasks_file.read_text())
        task = data["tasks"][0]
        assert task["status"] == "failed"
        assert "max attempts" in task.get("failure_reason", "").lower()

    def test_increment_handles_list_format(self, tmp_path):
        """Handles tasks file with list format (no wrapper dict)."""
        from src.mlflow_ops import increment_task_attempts, get_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps([
            {"id": 1, "name": "Test task", "status": "pending", "attempts": 2}
        ]))

        result = increment_task_attempts(1)
        assert result is True
        assert get_task_attempts(1) == 3

    def test_increment_task_not_found(self, tmp_path):
        """Returns False when task ID doesn't exist."""
        from src.mlflow_ops import increment_task_attempts, set_session_dir, get_tasks_file

        set_session_dir(tmp_path)
        tasks_file = get_tasks_file()
        tasks_file.write_text(json.dumps({
            "tasks": [
                {"id": 1, "name": "Test task", "status": "pending"}
            ]
        }))

        result = increment_task_attempts(999)
        assert result is False


# =============================================================================
# CONTEXT METRICS TESTS
# =============================================================================

class TestContextMetrics:
    """Tests for ContextMetrics class."""

    def test_context_metrics_creation(self):
        """ContextMetrics can be created with session_id."""
        from src.agent import ContextMetrics

        metrics = ContextMetrics(session_id="test_session")
        assert metrics.session_id == "test_session"
        assert metrics.tool_calls == 0
        assert metrics.estimated_messages == 2
        assert metrics.estimated_context_kb == 0.0

    def test_record_tool_call(self):
        """Recording a tool call updates metrics."""
        from src.agent import ContextMetrics

        metrics = ContextMetrics(session_id="test_session")
        metrics.record_tool_call("mlflow_query", input_size=100, output_size=500)

        assert metrics.tool_calls == 1
        assert metrics.estimated_messages == 4  # +2 for tool request/response
        assert metrics.estimated_context_kb > 0

    def test_multiple_tool_calls(self):
        """Multiple tool calls accumulate correctly."""
        from src.agent import ContextMetrics

        metrics = ContextMetrics(session_id="test_session")
        metrics.record_tool_call("mlflow_query", input_size=100, output_size=500)
        metrics.record_tool_call("save_findings", input_size=200, output_size=100)

        assert metrics.tool_calls == 2
        assert metrics.estimated_messages == 6

    def test_to_dict(self):
        """to_dict returns proper dictionary representation."""
        from src.agent import ContextMetrics

        metrics = ContextMetrics(session_id="test_session")
        metrics.record_tool_call("mlflow_query", input_size=100, output_size=500)

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["session_id"] == "test_session"
        assert result["tool_calls"] == 1
        assert "estimated_messages" in result
        assert "estimated_context_kb" in result


class TestContextMonitoringFunctions:
    """Tests for global context monitoring functions."""

    def test_start_context_monitoring(self):
        """start_context_monitoring initializes global metrics."""
        from src.agent import start_context_monitoring, get_context_metrics

        metrics = start_context_monitoring("session_1", "Test prompt")

        assert metrics is not None
        assert metrics.session_id == "session_1"

        # Should be retrievable via get_context_metrics
        retrieved = get_context_metrics()
        assert retrieved is metrics

    def test_get_context_metrics_returns_none_before_start(self):
        """get_context_metrics returns None if not started."""
        from src.agent import get_context_metrics, _reset_context_metrics

        _reset_context_metrics()  # Clear any existing state
        result = get_context_metrics()
        assert result is None

    def test_record_tool_call_global(self):
        """Global record_tool_call updates current metrics."""
        from src.agent import start_context_monitoring, record_tool_call, get_context_metrics

        start_context_monitoring("session_2", "Test prompt")
        record_tool_call("test_tool", input_size=50, output_size=200)

        metrics = get_context_metrics()
        assert metrics.tool_calls == 1
