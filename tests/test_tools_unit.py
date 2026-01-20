"""Unit tests for MCP tools.

Tests the three simplified MCP tools:
- mlflow_query: search, get, assessment operations
- mlflow_annotate: tag, feedback, expectation operations
- save_findings: state persistence
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from tests.integration.mock_mlflow import MockMLflowClient, create_sample_traces


def get_tool_by_name(tools: list, name: str):
    """Get a tool handler by its registered name from the tools list.

    The tools are wrapped by decorators and mlflow.trace, so we need to
    extract the actual handler function from the SdkMcpTool for direct testing.
    """
    for tool in tools:
        wrapped = getattr(tool, "__wrapped__", None)
        if wrapped and hasattr(wrapped, "name") and wrapped.name == name:
            # Return the handler function for direct testing
            # This bypasses the mlflow.trace decorator which causes issues in tests
            if hasattr(wrapped, "handler"):
                return wrapped.handler
            return tool
        # Also check handler name as fallback
        if wrapped and hasattr(wrapped, "handler"):
            handler = wrapped.handler
            if name in handler.__name__:
                return handler
    return None


# =============================================================================
# MLFLOW_QUERY TOOL TESTS
# =============================================================================


class TestMlflowQueryTool:
    """Tests for the mlflow_query tool."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client with sample traces."""
        return create_sample_traces(5)

    @pytest.mark.asyncio
    async def test_search_operation(self, mock_client, session_dir):
        """Test search operation returns trace list."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        with patch("src.mlflow_ops.get_client", return_value=mock_client):
            result = await query_tool({
                "operation": "search",
                "experiment_id": "123",
                "max_results": 10,
            })

        # Check result structure
        assert "content" in result
        assert len(result["content"]) > 0
        text = result["content"][0]["text"]
        assert "Trace ID" in text or "trace_id" in text.lower()

    @pytest.mark.asyncio
    async def test_search_with_filter(self, mock_client, session_dir):
        """Test search with filter string."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        with patch("src.mlflow_ops.get_client", return_value=mock_client):
            result = await query_tool({
                "operation": "search",
                "experiment_id": "123",
                "filter_string": "status = 'ERROR'",
                "max_results": 10,
            })

        assert "content" in result
        # Should only return error traces

    @pytest.mark.asyncio
    async def test_get_operation(self, mock_client, session_dir):
        """Test get operation returns trace details."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        with patch("src.mlflow_ops.get_client", return_value=mock_client):
            result = await query_tool({
                "operation": "get",
                "trace_id": "tr-sample-001",
            })

        assert "content" in result
        text = result["content"][0]["text"]
        # Should contain trace info
        assert "trace_id" in text or "status" in text

    @pytest.mark.asyncio
    async def test_get_with_detail_level(self, mock_client, session_dir):
        """Test get with different detail levels."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        for detail_level in ["summary", "analysis", "full"]:
            with patch("src.mlflow_ops.get_client", return_value=mock_client):
                result = await query_tool({
                    "operation": "get",
                    "trace_id": "tr-sample-001",
                    "detail_level": detail_level,
                })

            assert "content" in result
            assert "Error" not in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_search_without_experiment_id(self, session_dir):
        """Test search without experiment_id returns error."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        result = await query_tool({
            "operation": "search",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "experiment_id" in text.lower()

    @pytest.mark.asyncio
    async def test_get_without_trace_id(self, session_dir):
        """Test get without trace_id returns error."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        result = await query_tool({
            "operation": "get",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "trace_id" in text.lower()

    @pytest.mark.asyncio
    async def test_invalid_operation(self, session_dir):
        """Test invalid operation returns error."""
        from src.tools import create_tools

        tools = create_tools()
        query_tool = get_tool_by_name(tools, "mlflow_query")

        result = await query_tool({
            "operation": "invalid_op",
            "experiment_id": "123",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Unknown operation" in text or "invalid_op" in text


# =============================================================================
# MLFLOW_ANNOTATE TOOL TESTS
# =============================================================================


class TestMlflowAnnotateTool:
    """Tests for the mlflow_annotate tool."""

    @pytest.mark.asyncio
    async def test_tag_operation(self, session_dir):
        """Test tag operation sets tag on trace."""
        from src.tools import create_tools

        tools = create_tools()
        annotate_tool = get_tool_by_name(tools, "mlflow_annotate")

        with patch("src.mlflow_ops.set_tag") as mock_set_tag:
            result = await annotate_tool({
                "operation": "tag",
                "trace_id": "tr-test-001",
                "key": "test_key",
                "value": "test_value",
            })

        mock_set_tag.assert_called_once_with("tr-test-001", "test_key", "test_value")
        assert "content" in result
        assert "test_key" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_feedback_operation(self, session_dir):
        """Test feedback operation logs feedback."""
        from src.tools import create_tools

        tools = create_tools()
        annotate_tool = get_tool_by_name(tools, "mlflow_annotate")

        with patch("src.mlflow_ops.log_feedback") as mock_log_feedback:
            result = await annotate_tool({
                "operation": "feedback",
                "trace_id": "tr-test-001",
                "name": "quality",
                "value": "good",
                "rationale": "Test rationale",
            })

        mock_log_feedback.assert_called_once()
        call_args = mock_log_feedback.call_args
        assert call_args.kwargs["trace_id"] == "tr-test-001"
        assert call_args.kwargs["name"] == "quality"
        assert call_args.kwargs["value"] == "good"

    @pytest.mark.asyncio
    async def test_expectation_operation(self, session_dir):
        """Test expectation operation logs expectation."""
        from src.tools import create_tools

        tools = create_tools()
        annotate_tool = get_tool_by_name(tools, "mlflow_annotate")

        with patch("src.mlflow_ops.log_expectation") as mock_log_expectation:
            result = await annotate_tool({
                "operation": "expectation",
                "trace_id": "tr-test-001",
                "name": "expected_answer",
                "value": "Paris",
            })

        mock_log_expectation.assert_called_once_with(
            trace_id="tr-test-001",
            name="expected_answer",
            value="Paris",
        )

    @pytest.mark.asyncio
    async def test_annotate_without_trace_id(self, session_dir):
        """Test annotate without trace_id returns error."""
        from src.tools import create_tools

        tools = create_tools()
        annotate_tool = get_tool_by_name(tools, "mlflow_annotate")

        result = await annotate_tool({
            "operation": "tag",
            "key": "test",
            "value": "test",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "trace_id" in text.lower()

    @pytest.mark.asyncio
    async def test_tag_without_key(self, session_dir):
        """Test tag without key returns error."""
        from src.tools import create_tools

        tools = create_tools()
        annotate_tool = get_tool_by_name(tools, "mlflow_annotate")

        result = await annotate_tool({
            "operation": "tag",
            "trace_id": "tr-test-001",
            "value": "test",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "key" in text.lower()


# =============================================================================
# SAVE_FINDINGS TOOL TESTS
# =============================================================================


class TestSaveFindingsTool:
    """Tests for the save_findings tool."""

    @pytest.mark.asyncio
    async def test_saves_to_state_file(self, session_dir):
        """Test save_findings creates state file."""
        from src.tools import create_tools
        from src.mlflow_ops import get_state_dir

        tools = create_tools()
        save_tool = get_tool_by_name(tools, "save_findings")

        result = await save_tool({
            "key": "test_findings",
            "data": {"test": "data", "count": 42},
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Saved" in text or "test_findings" in text

        # Verify file was created
        state_file = get_state_dir() / "test_findings.json"
        assert state_file.exists()

        # Verify content
        content = json.loads(state_file.read_text())
        assert content["test"] == "data"
        assert content["count"] == 42

    @pytest.mark.asyncio
    async def test_returns_confirmation(self, session_dir):
        """Test save_findings returns confirmation."""
        from src.tools import create_tools

        tools = create_tools()
        save_tool = get_tool_by_name(tools, "save_findings")

        result = await save_tool({
            "key": "confirmation_test",
            "data": {"status": "ok"},
        })

        assert "content" in result
        text = result["content"][0]["text"]
        # Should confirm the save
        assert "Saved" in text or "confirmation_test" in text

    @pytest.mark.asyncio
    async def test_without_key(self, session_dir):
        """Test save_findings without key returns error."""
        from src.tools import create_tools

        tools = create_tools()
        save_tool = get_tool_by_name(tools, "save_findings")

        result = await save_tool({
            "data": {"test": "data"},
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "key" in text.lower()

    @pytest.mark.asyncio
    async def test_without_data(self, session_dir):
        """Test save_findings without data returns error."""
        from src.tools import create_tools

        tools = create_tools()
        save_tool = get_tool_by_name(tools, "save_findings")

        result = await save_tool({
            "key": "test_key",
        })

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "data" in text.lower()


# =============================================================================
# TOOL CREATION TESTS
# =============================================================================


class TestToolCreation:
    """Tests for tool creation and registration."""

    def test_create_tools_returns_three_tools(self):
        """create_tools should return exactly 3 tools."""
        from src.tools import create_tools

        tools = create_tools()
        assert len(tools) == 3

    def test_all_tools_are_callable(self):
        """All tools should be callable."""
        from src.tools import create_tools

        tools = create_tools()
        for tool in tools:
            assert callable(tool)

    def test_tool_names_match_constants(self):
        """Tool names should match MCPTools constants."""
        from src.tools import create_tools, MCPTools, MCP_SERVER_NAME

        tools = create_tools()

        # Get tool names from __wrapped__ attribute
        tool_names = set()
        for tool in tools:
            wrapped = getattr(tool, "__wrapped__", None)
            if wrapped and hasattr(wrapped, "name"):
                tool_names.add(wrapped.name)

        # Check that expected tools exist
        expected_names = {"mlflow_query", "mlflow_annotate", "save_findings"}
        assert expected_names == tool_names
