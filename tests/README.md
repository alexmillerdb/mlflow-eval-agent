# Testing Framework

Unit tests and mock infrastructure for the MLflow Evaluation Agent.

## Quick Start

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sessions.py

# Run with verbose output
uv run pytest -v

# Run only tests matching pattern
uv run pytest -k "test_verify"
```

## Test Organization

| File | Tests | Coverage |
|------|-------|----------|
| `test_sessions.py` | 22 | Session state management, output verification, task progress tracking |
| `test_tools_unit.py` | 19 | MCP tools (`mlflow_query`, `mlflow_annotate`, `save_findings`) |
| `test_context_optimization.py` | 21 | Task retry limits, context metrics, attempt tracking |

**Total: 62 unit tests**

## Key Test Classes

### test_sessions.py

- `TestInitializerOutputVerification` - Validates `eval_tasks.json` and `analysis.json` creation
- `TestWorkerOutputVerification` - Validates task status updates and artifact creation
- `TestMockTasksCreation` - Mock task/analysis file generation
- `TestTaskProgress` - Task status updates and pending task selection
- `TestSessionState` - State isolation between sessions, persistence
- `TestTestHarnessFunctions` - `TestResult` printing and formatting

### test_tools_unit.py

- `TestMlflowQueryTool` - Search, get, and assessment operations
- `TestMlflowAnnotateTool` - Tag, feedback, and expectation operations
- `TestSaveFindingsTool` - State file persistence
- `TestToolCreation` - Tool registration and naming

### test_context_optimization.py

- `TestMaxTaskAttemptsConstant` - `MAX_TASK_ATTEMPTS` constant (default: 5)
- `TestGetTaskAttempts` - Attempt count retrieval
- `TestIncrementTaskAttempts` - Attempt incrementing and failure marking
- `TestContextMetrics` - Context size tracking
- `TestContextMonitoringFunctions` - Global context monitoring

## Fixtures (conftest.py)

### Session Directory Fixtures

| Fixture | Description |
|---------|-------------|
| `session_dir` | Creates temporary session directory |
| `session_with_tasks` | Session with sample `eval_tasks.json` |
| `session_with_analysis` | Session with sample `state/analysis.json` |
| `session_with_all_state` | Session with both tasks and analysis |

### Sample Data Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_tasks` | List of 4 sample tasks (dataset, scorer, script, validate) |
| `sample_trace_data` | Single trace with spans and assessments |
| `sample_trace_list` | List of 3 trace summaries (2 OK, 1 ERROR) |

### Mock MLflow Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_mlflow_client` | `MockMLflowClient` with pre-populated traces |
| `mock_mlflow_ops` | Patches `mlflow_ops.get_client` with mock |

### Environment Fixtures

| Fixture | Description |
|---------|-------------|
| `local_mlflow_env` | Sets up local MLflow tracking (`http://localhost:5000`) |
| `databricks_env` | Sets up Databricks MLflow tracking |
| `clean_mlflow_state` | Clears client/trace caches and context metrics |

## Mock Infrastructure

### MockMLflowClient (integration/mock_mlflow.py)

Enables offline testing without Databricks connection:

```python
from tests.integration.mock_mlflow import MockMLflowClient, create_sample_traces

# Create client with sample data
client = create_sample_traces(count=10)  # 10 traces, 20% error rate

# Or manually populate
client = MockMLflowClient()
client.add_trace("tr-001", {"status": "OK", "execution_time_ms": 2500})
client.set_full_trace("tr-001", {"info": {...}, "spans": [...]})
```

### Mock Data Classes

- `MockTrace` - Full trace object with `info` and `data`
- `MockTraceInfo` - Trace metadata (trace_id, status, execution_time_ms)
- `MockTraceData` - Trace spans
- `MockSpan` - Individual span with timing and attributes
- `MockSpanStatus` - Span status code

## CLI Test Harness

The test harness in `src/test_harness.py` provides functions for testing components:

| Function | Purpose |
|----------|---------|
| `run_initializer_session()` | Test initializer in isolation |
| `run_worker_session()` | Test worker on existing session |
| `test_tool_direct()` | Call MCP tools without agent |
| `run_integration_test()` | Full autonomous loop with limits |
| `verify_initializer_outputs()` | Validate initializer output files |
| `verify_worker_outputs()` | Validate worker task completion |
| `create_mock_tasks()` | Generate mock `eval_tasks.json` |
| `create_mock_analysis()` | Generate mock `analysis.json` |
| `print_test_result()` | Format test results for console |

## Writing New Tests

### Basic Test Structure

```python
import pytest
from pathlib import Path

class TestMyFeature:
    """Tests for my feature."""

    def test_basic_case(self, session_dir):
        """Test description."""
        from src.mlflow_ops import my_function

        result = my_function(session_dir)

        assert result is not None
        assert result["key"] == "expected"

    @pytest.mark.asyncio
    async def test_async_case(self, session_dir, mock_mlflow_ops):
        """Test async operations with mocked MLflow."""
        from src.tools import create_tools

        tools = create_tools()
        result = await tools[0]({"operation": "search", "experiment_id": "123"})

        assert "content" in result
```

### Using Mock MLflow

```python
from unittest.mock import patch

def test_with_mock(self, mock_mlflow_client, session_dir):
    """Test with mocked MLflow client."""
    with patch("src.mlflow_ops.get_client", return_value=mock_mlflow_client):
        # Your test code - all MLflow calls use mock
        pass
```

### Testing Tool Handlers

```python
def get_tool_by_name(tools: list, name: str):
    """Get tool handler by name from tools list."""
    for tool in tools:
        wrapped = getattr(tool, "__wrapped__", None)
        if wrapped and hasattr(wrapped, "name") and wrapped.name == name:
            if hasattr(wrapped, "handler"):
                return wrapped.handler
    return None
```

## Running with Mock Mode

For development without MLflow connection:

```bash
# Test initializer with mock data
uv run python -m src.cli test initializer -e 123 --mock

# Test worker with mock tasks
uv run python -m src.cli test worker -e 123 --session-dir /tmp/test --mock
```
