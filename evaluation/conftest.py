"""Pytest configuration and fixtures for evaluation tests."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Markers
# -----------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires MLflow/Databricks)"
    )


# -----------------------------------------------------------------------------
# Environment Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_experiment_path():
    """Experiment path for evaluation runs."""
    return "/Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations"


@pytest.fixture(scope="session")
def mlflow_config():
    """MLflow configuration for integration tests."""
    return {
        "tracking_uri": "databricks",
        "experiment_path": "/Users/alex.miller@databricks.com/mlflow-eval-agent-evaluations",
    }


@pytest.fixture
def setup_mlflow(mlflow_config, eval_experiment_path):
    """Set up MLflow for integration tests."""
    import mlflow

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(eval_experiment_path)
    return mlflow


# -----------------------------------------------------------------------------
# Mock Fixtures for Unit Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_workspace():
    """Create a mock SharedWorkspace for unit tests."""
    from src.workspace import SharedWorkspace

    workspace = SharedWorkspace()
    return workspace


@pytest.fixture
def mock_workspace_with_trace_data(mock_workspace):
    """Workspace pre-populated with trace_analyst outputs."""
    mock_workspace.write(
        "trace_analysis_summary",
        {
            "error_rate": 0.1,
            "success_rate": 0.9,
            "trace_count": 100,
            "avg_latency_ms": 250.5,
            "experiment_id": "123",
            "time_range": "last 24 hours",
        },
        agent="trace_analyst",
    )
    mock_workspace.write(
        "error_patterns",
        [
            {
                "error_type": "TimeoutError",
                "count": 5,
                "example_trace_ids": ["trace_1", "trace_2"],
            }
        ],
        agent="trace_analyst",
    )
    return mock_workspace


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for testing routing logic."""
    from src.subagents.registry import AGENT_REGISTRY

    return AGENT_REGISTRY.copy()


@pytest.fixture
def mock_trace():
    """Create a mock MLflow Trace object for scorer testing."""
    mock = MagicMock()
    mock.search_spans = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_trace_with_agent_spans():
    """Trace with agent invocation spans for routing tests."""
    mock = MagicMock()

    # Create mock spans representing agent invocations
    trace_analyst_span = MagicMock()
    trace_analyst_span.name = "trace_analyst"
    trace_analyst_span.start_time_ns = 1000000000
    trace_analyst_span.end_time_ns = 2000000000

    context_engineer_span = MagicMock()
    context_engineer_span.name = "context_engineer"
    context_engineer_span.start_time_ns = 2000000000
    context_engineer_span.end_time_ns = 3000000000

    def search_spans_mock(span_type=None, name=None):
        spans = [trace_analyst_span, context_engineer_span]
        if name:
            return [s for s in spans if name.lower() in s.name.lower()]
        return spans

    mock.search_spans = search_spans_mock
    return mock


@pytest.fixture
def mock_trace_with_tool_calls():
    """Trace with tool call spans for tool usage tests."""
    from mlflow.entities import SpanType

    mock = MagicMock()

    # Mock tool spans
    search_traces_span = MagicMock()
    search_traces_span.name = "search_traces"
    search_traces_span.span_type = SpanType.TOOL

    get_trace_span = MagicMock()
    get_trace_span.name = "get_trace"
    get_trace_span.span_type = SpanType.TOOL

    write_workspace_span = MagicMock()
    write_workspace_span.name = "write_to_workspace"
    write_workspace_span.span_type = SpanType.TOOL

    all_spans = [search_traces_span, get_trace_span, write_workspace_span]

    def search_spans_mock(span_type=None, name=None):
        if span_type == SpanType.TOOL:
            return all_spans
        return all_spans

    mock.search_spans = search_spans_mock
    return mock


# -----------------------------------------------------------------------------
# Config Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def eval_agent_config():
    """Load EvalAgentConfig from environment (with validation disabled for tests)."""
    from src.config import EvalAgentConfig

    try:
        return EvalAgentConfig.from_env(validate=False)
    except Exception:
        # Return minimal config for unit tests
        return EvalAgentConfig(
            databricks_host="https://test.cloud.databricks.com",
            experiment_id="123",
        )
