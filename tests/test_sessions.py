"""Integration tests for initializer and worker sessions.

Tests session behavior without running the full agent.
Uses mock data to test state management and file output.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock


# =============================================================================
# INITIALIZER SESSION TESTS
# =============================================================================


class TestInitializerOutputVerification:
    """Tests for verifying initializer session outputs."""

    def test_verify_with_valid_outputs(self, session_dir):
        """verify_initializer_outputs succeeds with valid files."""
        from src.test_harness import verify_initializer_outputs
        from src.mlflow_ops import get_tasks_file, get_state_dir

        # Create valid tasks file
        tasks = {
            "tasks": [
                {"id": 1, "name": "Test task", "type": "dataset", "status": "pending", "details": "Test"},
            ]
        }
        get_tasks_file().write_text(json.dumps(tasks))

        # Create valid analysis file
        analysis = {
            "experiment_id": "123",
            "dataset_strategy": "traces",
            "recommended_scorers": [{"name": "Safety", "type": "builtin"}],
        }
        state_dir = get_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "analysis.json").write_text(json.dumps(analysis))

        result = verify_initializer_outputs(session_dir)

        assert result["valid"] is True
        assert result["tasks_file_exists"] is True
        assert result["tasks_valid"] is True
        assert result["analysis_file_exists"] is True
        assert result["analysis_valid"] is True
        assert len(result["errors"]) == 0

    def test_verify_without_tasks_file(self, session_dir):
        """verify_initializer_outputs fails without tasks file."""
        from src.test_harness import verify_initializer_outputs

        result = verify_initializer_outputs(session_dir)

        assert result["valid"] is False
        assert result["tasks_file_exists"] is False
        assert "Tasks file not found" in str(result["errors"])

    def test_verify_without_analysis_file(self, session_dir):
        """verify_initializer_outputs fails without analysis file."""
        from src.test_harness import verify_initializer_outputs
        from src.mlflow_ops import get_tasks_file

        # Create valid tasks file only
        tasks = {"tasks": [{"id": 1, "name": "Test", "type": "dataset", "status": "pending"}]}
        get_tasks_file().write_text(json.dumps(tasks))

        result = verify_initializer_outputs(session_dir)

        assert result["valid"] is False
        assert result["tasks_file_exists"] is True
        assert result["analysis_file_exists"] is False

    def test_verify_with_missing_task_fields(self, session_dir):
        """verify_initializer_outputs fails with incomplete tasks."""
        from src.test_harness import verify_initializer_outputs
        from src.mlflow_ops import get_tasks_file, get_state_dir

        # Create tasks missing required fields
        tasks = {"tasks": [{"id": 1, "name": "Test"}]}  # Missing type, status
        get_tasks_file().write_text(json.dumps(tasks))

        # Create valid analysis
        analysis = {
            "experiment_id": "123",
            "dataset_strategy": "traces",
            "recommended_scorers": [],
        }
        state_dir = get_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "analysis.json").write_text(json.dumps(analysis))

        result = verify_initializer_outputs(session_dir)

        assert result["valid"] is False
        assert "missing required fields" in str(result["errors"]).lower()


# =============================================================================
# WORKER SESSION TESTS
# =============================================================================


class TestWorkerOutputVerification:
    """Tests for verifying worker session outputs."""

    def test_verify_task_found(self, session_with_tasks):
        """verify_worker_outputs finds existing task."""
        from src.test_harness import verify_worker_outputs

        result = verify_worker_outputs(session_with_tasks, task_id=1)

        assert result["task_found"] is True

    def test_verify_task_not_found(self, session_with_tasks):
        """verify_worker_outputs handles missing task."""
        from src.test_harness import verify_worker_outputs

        result = verify_worker_outputs(session_with_tasks, task_id=999)

        assert result["task_found"] is False
        assert "not found" in str(result["errors"]).lower()

    def test_verify_completed_task(self, session_dir):
        """verify_worker_outputs detects completed task."""
        from src.test_harness import verify_worker_outputs
        from src.mlflow_ops import get_tasks_file

        tasks = {"tasks": [{"id": 1, "name": "Test", "type": "dataset", "status": "completed"}]}
        get_tasks_file().write_text(json.dumps(tasks))

        result = verify_worker_outputs(session_dir, task_id=1)

        assert result["task_found"] is True
        assert result["task_completed"] is True
        assert result["task_status"] == "completed"

    def test_verify_pending_task(self, session_with_tasks):
        """verify_worker_outputs detects pending task."""
        from src.test_harness import verify_worker_outputs

        result = verify_worker_outputs(session_with_tasks, task_id=1)

        assert result["task_found"] is True
        assert result["task_completed"] is False
        assert result["task_status"] == "pending"

    def test_verify_artifacts_created(self, session_dir):
        """verify_worker_outputs detects created artifacts."""
        from src.test_harness import verify_worker_outputs
        from src.mlflow_ops import get_tasks_file

        # Create tasks file
        tasks = {"tasks": [{"id": 1, "name": "Test", "type": "dataset", "status": "completed"}]}
        get_tasks_file().write_text(json.dumps(tasks))

        # Create evaluation directory with artifacts
        eval_dir = session_dir / "evaluation"
        eval_dir.mkdir()
        (eval_dir / "eval_dataset.py").write_text("# test")
        (eval_dir / "scorers.py").write_text("# test")

        result = verify_worker_outputs(session_dir, task_id=1)

        assert "eval_dataset.py" in result["artifacts_created"]
        assert "scorers.py" in result["artifacts_created"]
        assert "run_eval.py" not in result["artifacts_created"]  # Not created


# =============================================================================
# MOCK TASKS CREATION TESTS
# =============================================================================


class TestMockTasksCreation:
    """Tests for creating mock tasks for testing."""

    def test_create_mock_tasks_default(self, session_dir):
        """create_mock_tasks creates default task types."""
        from src.test_harness import create_mock_tasks
        from src.mlflow_ops import get_tasks_file

        tasks_file = create_mock_tasks(session_dir)

        assert tasks_file.exists()
        data = json.loads(tasks_file.read_text())
        tasks = data.get("tasks", [])

        assert len(tasks) == 4
        task_types = [t["type"] for t in tasks]
        assert "dataset" in task_types
        assert "scorer" in task_types
        assert "script" in task_types
        assert "validate" in task_types

    def test_create_mock_tasks_custom_types(self, session_dir):
        """create_mock_tasks with custom task types."""
        from src.test_harness import create_mock_tasks

        tasks_file = create_mock_tasks(session_dir, task_types=["dataset", "scorer"])

        data = json.loads(tasks_file.read_text())
        tasks = data.get("tasks", [])

        assert len(tasks) == 2

    def test_create_mock_analysis(self, session_dir):
        """create_mock_analysis creates valid analysis file."""
        from src.test_harness import create_mock_analysis
        from src.mlflow_ops import get_state_dir

        analysis_file = create_mock_analysis(session_dir, "test-exp-123")

        assert analysis_file.exists()
        data = json.loads(analysis_file.read_text())

        assert data["experiment_id"] == "test-exp-123"
        assert "dataset_strategy" in data
        assert "recommended_scorers" in data
        assert isinstance(data["recommended_scorers"], list)


# =============================================================================
# TASK PROGRESS TESTS
# =============================================================================


class TestTaskProgress:
    """Tests for task progress tracking."""

    def test_picks_first_pending_task(self, session_with_tasks):
        """Worker should pick first pending task."""
        from src.mlflow_ops import get_tasks_file

        data = json.loads(get_tasks_file().read_text())
        tasks = data.get("tasks", [])

        # Find first pending
        pending = [t for t in tasks if t["status"] == "pending"]
        assert len(pending) > 0
        assert pending[0]["id"] == 1

    def test_task_status_update(self, session_with_tasks):
        """Task status can be updated."""
        from src.mlflow_ops import get_tasks_file

        tasks_file = get_tasks_file()
        data = json.loads(tasks_file.read_text())

        # Update first task
        data["tasks"][0]["status"] = "completed"
        tasks_file.write_text(json.dumps(data))

        # Verify
        updated = json.loads(tasks_file.read_text())
        assert updated["tasks"][0]["status"] == "completed"

    def test_respects_attempt_limit(self, session_dir):
        """Task with max attempts is marked failed."""
        from src.mlflow_ops import get_tasks_file, increment_task_attempts, MAX_TASK_ATTEMPTS

        # Create task at max attempts
        tasks = {"tasks": [{"id": 1, "name": "Test", "status": "pending", "attempts": MAX_TASK_ATTEMPTS}]}
        get_tasks_file().write_text(json.dumps(tasks))

        # Increment should fail
        result = increment_task_attempts(1)
        assert result is False

        # Task should be marked failed
        data = json.loads(get_tasks_file().read_text())
        assert data["tasks"][0]["status"] == "failed"


# =============================================================================
# SESSION STATE TESTS
# =============================================================================


class TestSessionState:
    """Tests for session state management."""

    def test_state_isolation_between_sessions(self, tmp_path):
        """Each session should have isolated state."""
        from src.mlflow_ops import set_session_dir, save_state, load_state

        # Session 1
        session1 = tmp_path / "session1"
        set_session_dir(session1)
        save_state("test", {"session": 1})

        # Session 2
        session2 = tmp_path / "session2"
        set_session_dir(session2)
        save_state("test", {"session": 2})

        # Verify isolation
        set_session_dir(session1)
        assert load_state("test")["session"] == 1

        set_session_dir(session2)
        assert load_state("test")["session"] == 2

    def test_state_persists_across_calls(self, session_dir):
        """State should persist between load calls."""
        from src.mlflow_ops import save_state, load_state

        save_state("persist_test", {"count": 42, "name": "test"})

        # Load multiple times
        data1 = load_state("persist_test")
        data2 = load_state("persist_test")

        assert data1 == data2
        assert data1["count"] == 42

    def test_load_missing_state_returns_none(self, session_dir):
        """Loading non-existent state returns None."""
        from src.mlflow_ops import load_state

        result = load_state("nonexistent_key")
        assert result is None


# =============================================================================
# HARNESS FUNCTION TESTS (Unit level)
# =============================================================================


class TestTestHarnessFunctions:
    """Unit tests for test harness helper functions."""

    def test_print_test_result_pass(self, capsys):
        """print_test_result shows PASS for success."""
        from src.test_harness import TestResult, print_test_result

        result = TestResult(
            success=True,
            trace_id="tr-test-001",
            duration_ms=1500,
        )

        print_test_result(result)
        captured = capsys.readouterr()

        assert "PASS" in captured.out
        assert "tr-test-001" in captured.out

    def test_print_test_result_fail(self, capsys):
        """print_test_result shows FAIL for failure."""
        from src.test_harness import TestResult, print_test_result

        result = TestResult(
            success=False,
            error="Test error message",
        )

        print_test_result(result)
        captured = capsys.readouterr()

        assert "FAIL" in captured.out
        assert "Test error message" in captured.out

    def test_print_test_result_verbose(self, capsys):
        """print_test_result shows outputs when verbose."""
        from src.test_harness import TestResult, print_test_result

        result = TestResult(
            success=True,
            outputs={"key": "value", "count": 42},
        )

        print_test_result(result, verbose=True)
        captured = capsys.readouterr()

        assert "Outputs:" in captured.out
        assert "key" in captured.out
        assert "42" in captured.out
