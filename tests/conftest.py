"""Shared pytest fixtures for MLflow Evaluation Agent tests.

Provides fixtures for:
- Session directories
- Mock MLflow client
- Sample tasks and analysis data
- MLflow tracking setup
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# SESSION DIRECTORY FIXTURES
# =============================================================================


@pytest.fixture
def session_dir(tmp_path):
    """Create a temporary session directory.

    Sets up the session directory in mlflow_ops for the test.
    """
    from src.mlflow_ops import set_session_dir

    set_session_dir(tmp_path)
    return tmp_path


@pytest.fixture
def session_with_tasks(session_dir):
    """Create a session directory with sample tasks file."""
    from src.mlflow_ops import get_tasks_file

    tasks = {
        "tasks": [
            {"id": 1, "name": "Build evaluation dataset", "type": "dataset", "status": "pending", "details": "Extract from traces"},
            {"id": 2, "name": "Create scorers", "type": "scorer", "status": "pending", "details": "Safety and relevance"},
            {"id": 3, "name": "Generate eval script", "type": "script", "status": "pending", "details": "Simple script"},
            {"id": 4, "name": "Run and validate", "type": "validate", "status": "pending", "details": "Run eval"},
        ]
    }

    tasks_file = get_tasks_file()
    tasks_file.write_text(json.dumps(tasks, indent=2))

    return session_dir


@pytest.fixture
def session_with_analysis(session_dir):
    """Create a session directory with analysis file."""
    from src.mlflow_ops import get_state_dir

    analysis = {
        "experiment_id": "test-experiment-123",
        "agent_type": "Test agent for unit testing",
        "dataset_strategy": "traces",
        "has_predict_fn": False,
        "trace_summary": {
            "total_analyzed": 10,
            "success_count": 8,
            "error_count": 2,
            "avg_latency_ms": 2500,
        },
        "sample_trace_ids": ["tr-test-001", "tr-test-002", "tr-test-003"],
        "recommended_scorers": [
            {"name": "Safety", "type": "builtin", "rationale": "Required for all agents"},
            {"name": "RelevanceToQuery", "type": "builtin", "rationale": "Agent responds to queries"},
        ],
        "error_patterns": [],
    }

    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = state_dir / "analysis.json"
    analysis_file.write_text(json.dumps(analysis, indent=2))

    return session_dir


@pytest.fixture
def session_with_all_state(session_with_tasks, session_with_analysis):
    """Create a session directory with both tasks and analysis."""
    return session_with_tasks  # They share the same session_dir via fixture chain


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_tasks():
    """Return sample task list for testing."""
    return [
        {"id": 1, "name": "Build evaluation dataset", "type": "dataset", "status": "pending", "details": "Extract from traces"},
        {"id": 2, "name": "Create scorers", "type": "scorer", "status": "pending", "details": "Safety and relevance"},
        {"id": 3, "name": "Generate eval script", "type": "script", "status": "pending", "details": "Simple script"},
        {"id": 4, "name": "Run and validate", "type": "validate", "status": "pending", "details": "Run eval"},
    ]


@pytest.fixture
def sample_trace_data():
    """Return sample trace data for testing."""
    return {
        "info": {
            "trace_id": "tr-test-001",
            "status": "OK",
            "execution_time_ms": 2500,
        },
        "spans": [
            {
                "span_id": "span-001",
                "name": "agent_query",
                "span_type": "AGENT",
                "duration_ms": 2400,
                "parent_id": None,
            },
            {
                "span_id": "span-002",
                "name": "llm_call",
                "span_type": "LLM",
                "duration_ms": 1500,
                "parent_id": "span-001",
                "tokens": {"input": 500, "output": 200},
                "model": "claude-3-sonnet",
            },
        ],
        "assessments": [],
    }


@pytest.fixture
def sample_trace_list():
    """Return sample list of trace summaries for testing."""
    return [
        {
            "trace_id": "tr-test-001",
            "status": "OK",
            "execution_time_ms": 2500,
            "timestamp_ms": 1700000000000,
        },
        {
            "trace_id": "tr-test-002",
            "status": "OK",
            "execution_time_ms": 3200,
            "timestamp_ms": 1700000001000,
        },
        {
            "trace_id": "tr-test-003",
            "status": "ERROR",
            "execution_time_ms": 500,
            "timestamp_ms": 1700000002000,
        },
    ]


# =============================================================================
# MOCK MLFLOW FIXTURES
# =============================================================================


@pytest.fixture
def mock_mlflow_client(sample_trace_list, sample_trace_data):
    """Create a mock MLflow client for offline testing."""
    from tests.integration.mock_mlflow import MockMLflowClient

    client = MockMLflowClient()
    # Pre-populate with sample data
    for trace in sample_trace_list:
        client.add_trace(trace["trace_id"], trace)

    # Add full trace data for the first trace
    client.set_full_trace("tr-test-001", sample_trace_data)

    return client


@pytest.fixture
def mock_mlflow_ops(mock_mlflow_client):
    """Patch mlflow_ops.get_client to return mock client."""
    with patch("src.mlflow_ops.get_client", return_value=mock_mlflow_client):
        yield mock_mlflow_client


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def local_mlflow_env(monkeypatch):
    """Set up environment for local MLflow testing."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    # Clear cached client
    from src.mlflow_ops import clear_client_cache
    clear_client_cache()

    yield

    clear_client_cache()


@pytest.fixture
def databricks_env(monkeypatch):
    """Set up environment for Databricks testing."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "test")

    # Clear cached client
    from src.mlflow_ops import clear_client_cache
    clear_client_cache()

    yield

    clear_client_cache()


@pytest.fixture
def clean_mlflow_state():
    """Clean up MLflow-related global state before and after tests."""
    from src.mlflow_ops import (
        clear_client_cache,
        clear_trace_cache,
        _reset_context_metrics,
    )

    # Clean before
    clear_client_cache()
    clear_trace_cache()
    _reset_context_metrics()

    yield

    # Clean after
    clear_client_cache()
    clear_trace_cache()
    _reset_context_metrics()


# =============================================================================
# ASYNC TEST SUPPORT
# =============================================================================


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
