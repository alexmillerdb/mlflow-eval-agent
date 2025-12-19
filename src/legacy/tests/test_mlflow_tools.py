"""Unit tests for MLflow custom tools.

Tests the in-process MLflow tools that replace the external mlflow-mcp server.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.mlflow_client import (
    format_traces_output,
    format_trace_detail,
    text_result,
    clear_client_cache,
)
from src.tools import (
    create_mlflow_tools,
    MCPTools,
    MCP_SERVER_NAME,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_trace():
    """Create a mock trace object."""
    trace = MagicMock()
    trace.info.trace_id = "tr-test123"
    trace.info.status = "OK"
    trace.info.execution_time_ms = 1500
    trace.info.timestamp_ms = 1700000000000
    trace.info.assessments = []

    # Create mock spans
    llm_span = MagicMock()
    llm_span.span_id = "span-llm-1"
    llm_span.name = "llm_call"
    llm_span.span_type = "CHAT_MODEL"
    llm_span.start_time_ns = 0
    llm_span.end_time_ns = 800_000_000  # 800ms
    llm_span.parent_id = None
    llm_span.status = None
    llm_span.events = []
    llm_span.inputs = {"messages": [{"role": "user", "content": "Hello"}]}
    llm_span.outputs = {"content": "Hi there!"}
    llm_span.attributes = {
        "mlflow.chat_model.input_tokens": 10,
        "mlflow.chat_model.output_tokens": 5,
        "mlflow.chat_model.total_tokens": 15,
        "mlflow.chat_model.model": "claude-3-sonnet",
    }

    retriever_span = MagicMock()
    retriever_span.span_id = "span-retriever-1"
    retriever_span.name = "retriever"
    retriever_span.span_type = "RETRIEVER"
    retriever_span.start_time_ns = 0
    retriever_span.end_time_ns = 500_000_000  # 500ms
    retriever_span.parent_id = "span-llm-1"
    retriever_span.status = None
    retriever_span.events = []
    retriever_span.inputs = {"query": "test query"}
    retriever_span.outputs = {"documents": ["doc1", "doc2"]}
    retriever_span.attributes = {}

    trace.data.spans = [llm_span, retriever_span]

    return trace


@pytest.fixture
def mock_error_trace():
    """Create a mock trace with errors."""
    trace = MagicMock()
    trace.info.trace_id = "tr-error123"
    trace.info.status = "ERROR"
    trace.info.execution_time_ms = 500
    trace.info.timestamp_ms = 1700000000000
    trace.info.assessments = []

    error_span = MagicMock()
    error_span.span_id = "span-error-1"
    error_span.name = "failed_call"
    error_span.span_type = "TOOL"
    error_span.start_time_ns = 0
    error_span.end_time_ns = 100_000_000  # 100ms
    error_span.parent_id = None
    error_span.inputs = {"arg": "value"}
    error_span.outputs = None
    error_span.attributes = {}

    # Mock error status
    error_span.status = MagicMock()
    error_span.status.status_code = "ERROR"
    error_span.status.description = "Connection timeout after 30s"

    # Mock exception event
    exception_event = MagicMock()
    exception_event.name = "exception"
    exception_event.attributes = {"type": "TimeoutError", "message": "Connection timeout"}
    error_span.events = [exception_event]

    trace.data.spans = [error_span]

    return trace


@pytest.fixture
def mock_trace_with_assessment():
    """Create a mock trace with assessments."""
    trace = MagicMock()
    trace.info.trace_id = "tr-assessed123"
    trace.info.status = "OK"
    trace.info.execution_time_ms = 1000
    trace.info.timestamp_ms = 1700000000000

    assessment = MagicMock()
    assessment.name = "quality_score"
    assessment.value = "0.85"
    assessment.source_type = "LLM_JUDGE"
    assessment.rationale = "Good response quality"
    assessment.assessment_id = "asmt-123"

    trace.info.assessments = [assessment]
    trace.data.spans = []

    return trace


@pytest.fixture
def mock_mlflow_client(mock_trace):
    """Create a mock MlflowClient."""
    client = MagicMock()
    client.search_traces.return_value = [mock_trace]
    client.get_trace.return_value = mock_trace
    return client


# =============================================================================
# FORMAT FUNCTIONS TESTS
# =============================================================================

class TestFormatFunctions:
    """Test trace formatting functions."""

    def test_format_traces_output_table(self, mock_trace):
        """Test table format output."""
        result = format_traces_output([mock_trace], "table")

        assert "tr-test123" in result
        assert "OK" in result
        assert "1500" in result
        assert "|" in result  # Table format

    def test_format_traces_output_json(self, mock_trace):
        """Test JSON format output."""
        result = format_traces_output([mock_trace], "json")

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["trace_id"] == "tr-test123"
        assert data[0]["status"] == "OK"
        assert data[0]["execution_time_ms"] == 1500

    def test_format_traces_output_empty(self):
        """Test empty results."""
        result = format_traces_output([], "table")
        assert result == "No traces found."

    def test_format_trace_detail_basic(self, mock_trace):
        """Test trace detail formatting."""
        result = format_trace_detail(mock_trace)
        data = json.loads(result)

        assert data["info"]["trace_id"] == "tr-test123"
        assert data["info"]["status"] == "OK"
        assert len(data["spans"]) == 2

    def test_format_trace_detail_llm_tokens(self, mock_trace):
        """Test LLM span includes token info."""
        result = format_trace_detail(mock_trace)
        data = json.loads(result)

        llm_span = next(s for s in data["spans"] if s["span_type"] == "CHAT_MODEL")
        assert "tokens" in llm_span
        assert llm_span["tokens"]["input"] == 10
        assert llm_span["tokens"]["output"] == 5
        assert llm_span["tokens"]["total"] == 15
        assert llm_span["model"] == "claude-3-sonnet"

    def test_format_trace_detail_error_info(self, mock_error_trace):
        """Test error span includes error details."""
        result = format_trace_detail(mock_error_trace)
        data = json.loads(result)

        error_span = data["spans"][0]
        assert "error" in error_span
        assert error_span["error"]["message"] == "Connection timeout after 30s"
        assert "exception" in error_span["error"]

    def test_format_trace_detail_io_preview(self, mock_trace):
        """Test spans include inputs/outputs preview."""
        result = format_trace_detail(mock_trace)
        data = json.loads(result)

        llm_span = next(s for s in data["spans"] if s["span_type"] == "CHAT_MODEL")
        assert "inputs_preview" in llm_span
        assert "outputs_preview" in llm_span

    def test_format_trace_detail_assessments(self, mock_trace_with_assessment):
        """Test assessments are included."""
        result = format_trace_detail(mock_trace_with_assessment)
        data = json.loads(result)

        assert len(data["assessments"]) == 1
        assert data["assessments"][0]["name"] == "quality_score"
        assert data["assessments"][0]["value"] == "0.85"

    def test_text_result_format(self):
        """Test text_result returns correct MCP format."""
        result = text_result("Hello world")

        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello world"


# =============================================================================
# TOOL CONSTANTS TESTS
# =============================================================================

class TestToolConstants:
    """Test tool name constants."""

    def test_mcp_server_name(self):
        """Test MCP server name is correct."""
        assert MCP_SERVER_NAME == "mlflow-eval"

    def test_mcp_tools_format(self):
        """Test tool names follow MCP format."""
        assert MCPTools.SEARCH_TRACES == "mcp__mlflow-eval__search_traces"
        assert MCPTools.GET_TRACE == "mcp__mlflow-eval__get_trace"
        assert MCPTools.SET_TRACE_TAG == "mcp__mlflow-eval__set_trace_tag"
        assert MCPTools.DELETE_TRACE_TAG == "mcp__mlflow-eval__delete_trace_tag"
        assert MCPTools.LOG_FEEDBACK == "mcp__mlflow-eval__log_feedback"
        assert MCPTools.LOG_EXPECTATION == "mcp__mlflow-eval__log_expectation"
        assert MCPTools.GET_ASSESSMENT == "mcp__mlflow-eval__get_assessment"
        assert MCPTools.UPDATE_ASSESSMENT == "mcp__mlflow-eval__update_assessment"


# =============================================================================
# TOOL FUNCTION TESTS
# =============================================================================

def get_tool_handler(tool):
    """Get the underlying async handler function from an SdkMcpTool."""
    # The @tool decorator creates an SdkMcpTool which wraps the original function
    # Access the handler through the 'handler' attribute
    if hasattr(tool, 'handler'):
        return tool.handler
    # Fallback: try to call it directly (some versions may be callable)
    return tool


class TestMLflowTools:
    """Test MLflow tool functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear client cache before each test."""
        clear_client_cache()

    def test_create_mlflow_tools_count(self):
        """Test correct number of tools created."""
        tools = create_mlflow_tools()
        assert len(tools) == 8  # search, get, set_tag, delete_tag, log_feedback, log_expectation, get_assessment, update_assessment

    @pytest.mark.asyncio
    async def test_search_traces_tool(self, mock_mlflow_client, mock_trace):
        """Test search_traces tool."""
        with patch('src.tools.get_mlflow_client', return_value=mock_mlflow_client):
            tools = create_mlflow_tools()
            search_handler = get_tool_handler(tools[0])

            result = await search_handler({
                "experiment_id": "123",
                "filter_string": "status = 'OK'",
                "max_results": 10,
            })

            assert "content" in result
            text = result["content"][0]["text"]
            assert "tr-test123" in text

            mock_mlflow_client.search_traces.assert_called_once_with(
                experiment_ids=["123"],
                filter_string="status = 'OK'",
                max_results=10
            )

    @pytest.mark.asyncio
    async def test_search_traces_empty_experiment(self):
        """Test search_traces with missing experiment_id and experiment_name."""
        tools = create_mlflow_tools()
        search_handler = get_tool_handler(tools[0])

        result = await search_handler({})

        text = result["content"][0]["text"]
        assert "Error" in text
        assert "experiment_id" in text or "experiment_name" in text

    @pytest.mark.asyncio
    async def test_get_trace_tool(self, mock_mlflow_client, mock_trace):
        """Test get_trace tool."""
        with patch('src.tools.get_mlflow_client', return_value=mock_mlflow_client):
            tools = create_mlflow_tools()
            get_handler = get_tool_handler(tools[1])

            result = await get_handler({"trace_id": "tr-test123"})

            assert "content" in result
            text = result["content"][0]["text"]

            # Should be valid JSON with trace data
            data = json.loads(text)
            assert data["info"]["trace_id"] == "tr-test123"
            assert len(data["spans"]) == 2

            mock_mlflow_client.get_trace.assert_called_once_with(trace_id="tr-test123")

    @pytest.mark.asyncio
    async def test_get_trace_missing_id(self):
        """Test get_trace with missing trace_id."""
        tools = create_mlflow_tools()
        get_handler = get_tool_handler(tools[1])

        result = await get_handler({})

        text = result["content"][0]["text"]
        assert "Error" in text
        assert "trace_id" in text

    @pytest.mark.asyncio
    async def test_set_trace_tag_tool(self):
        """Test set_trace_tag tool."""
        with patch('src.tools.mlflow') as mock_mlflow:
            tools = create_mlflow_tools()
            set_tag_handler = get_tool_handler(tools[2])

            result = await set_tag_handler({
                "trace_id": "tr-test123",
                "key": "eval_candidate",
                "value": "error_case"
            })

            text = result["content"][0]["text"]
            assert "eval_candidate=error_case" in text
            assert "tr-test123" in text

            mock_mlflow.set_trace_tag.assert_called_once_with(
                "tr-test123", "eval_candidate", "error_case"
            )

    @pytest.mark.asyncio
    async def test_delete_trace_tag_tool(self):
        """Test delete_trace_tag tool."""
        with patch('src.tools.mlflow') as mock_mlflow:
            tools = create_mlflow_tools()
            delete_tag_handler = get_tool_handler(tools[3])

            result = await delete_tag_handler({
                "trace_id": "tr-test123",
                "key": "eval_candidate"
            })

            text = result["content"][0]["text"]
            assert "deleted" in text.lower()

            mock_mlflow.delete_trace_tag.assert_called_once_with(
                "tr-test123", "eval_candidate"
            )

    @pytest.mark.asyncio
    async def test_log_feedback_tool(self, mock_mlflow_client):
        """Test log_feedback tool."""
        # log_feedback uses module-level mlflow.log_feedback, not MlflowClient method
        with patch('src.tools.mlflow') as mock_mlflow:
            tools = create_mlflow_tools()
            log_feedback_handler = get_tool_handler(tools[4])

            result = await log_feedback_handler({
                "trace_id": "tr-test123",
                "name": "bottleneck_detected",
                "value": "retriever",
                "source_type": "CODE",
                "rationale": "Retriever is slow"
            })

            text = result["content"][0]["text"]
            assert "bottleneck_detected" in text
            assert "tr-test123" in text

            mock_mlflow.log_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_expectation_tool(self, mock_mlflow_client):
        """Test log_expectation tool."""
        # log_expectation uses module-level mlflow.log_expectation, not MlflowClient method
        with patch('src.tools.mlflow') as mock_mlflow:
            tools = create_mlflow_tools()
            log_expectation_handler = get_tool_handler(tools[5])

            result = await log_expectation_handler({
                "trace_id": "tr-test123",
                "name": "expected_output",
                "value": '{"answer": "42"}'
            })

            text = result["content"][0]["text"]
            assert "expected_output" in text

            mock_mlflow.log_expectation.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_assessment_tool(self, mock_trace_with_assessment):
        """Test get_assessment tool."""
        mock_client = MagicMock()
        mock_client.get_trace.return_value = mock_trace_with_assessment

        with patch('src.tools.get_mlflow_client', return_value=mock_client):
            tools = create_mlflow_tools()
            get_assessment_handler = get_tool_handler(tools[6])

            result = await get_assessment_handler({
                "trace_id": "tr-assessed123",
                "assessment_name": "quality_score"
            })

            text = result["content"][0]["text"]
            data = json.loads(text)

            assert data["name"] == "quality_score"
            assert data["value"] == "0.85"
            assert data["rationale"] == "Good response quality"

    @pytest.mark.asyncio
    async def test_get_assessment_not_found(self, mock_trace):
        """Test get_assessment when assessment doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_trace.return_value = mock_trace

        with patch('src.tools.get_mlflow_client', return_value=mock_client):
            tools = create_mlflow_tools()
            get_assessment_handler = get_tool_handler(tools[6])

            result = await get_assessment_handler({
                "trace_id": "tr-test123",
                "assessment_name": "nonexistent"
            })

            text = result["content"][0]["text"]
            # "No assessments found" or "not found" are both valid responses
            assert "no assessments found" in text.lower() or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_update_assessment_tool(self, mock_trace_with_assessment):
        """Test update_assessment tool."""
        mock_client = MagicMock()
        mock_client.get_trace.return_value = mock_trace_with_assessment

        with patch('src.tools.get_mlflow_client', return_value=mock_client):
            tools = create_mlflow_tools()
            update_assessment_handler = get_tool_handler(tools[7])

            result = await update_assessment_handler({
                "trace_id": "tr-assessed123",
                "assessment_name": "quality_score",
                "rationale": "Updated rationale"
            })

            text = result["content"][0]["text"]
            assert "updated" in text.lower()

            mock_client.update_assessment.assert_called_once()


# =============================================================================
# INTEGRATION TESTS (require mocking full workflow)
# =============================================================================

class TestToolWorkflow:
    """Test tool workflow scenarios."""

    @pytest.mark.asyncio
    async def test_trace_analyst_workflow(self, mock_mlflow_client, mock_trace, mock_error_trace):
        """Test complete trace_analyst workflow."""
        # Scenario: Search for errors, get details, tag, log finding

        mock_mlflow_client.search_traces.return_value = [mock_error_trace]
        mock_mlflow_client.get_trace.return_value = mock_error_trace

        with patch('src.tools.get_mlflow_client', return_value=mock_mlflow_client):
            with patch('src.tools.mlflow') as mock_mlflow:
                tools = create_mlflow_tools()

                # Step 1: Search for errors
                search_handler = get_tool_handler(tools[0])
                search_result = await search_handler({
                    "experiment_id": "123",
                    "filter_string": "status = 'ERROR'",
                    "max_results": 20
                })
                assert "tr-error123" in search_result["content"][0]["text"]

                # Step 2: Get trace details
                get_handler = get_tool_handler(tools[1])
                get_result = await get_handler({"trace_id": "tr-error123"})
                data = json.loads(get_result["content"][0]["text"])
                assert data["info"]["status"] == "ERROR"
                assert "error" in data["spans"][0]

                # Step 3: Tag for evaluation
                tag_handler = get_tool_handler(tools[2])
                await tag_handler({
                    "trace_id": "tr-error123",
                    "key": "eval_candidate",
                    "value": "error_case"
                })
                mock_mlflow.set_trace_tag.assert_called()

                # Step 4: Log finding (uses module-level mlflow.log_feedback)
                feedback_handler = get_tool_handler(tools[4])
                await feedback_handler({
                    "trace_id": "tr-error123",
                    "name": "error_category",
                    "value": "timeout",
                    "source_type": "CODE",
                    "rationale": "Connection timeout detected"
                })
                mock_mlflow.log_feedback.assert_called()


# =============================================================================
# OUTPUT VALIDATION TESTS
# =============================================================================

class TestOutputValidation:
    """Test that tool outputs are usable by sub-agents."""

    def test_search_traces_output_parseable(self, mock_trace):
        """Test search_traces output can be parsed."""
        result = format_traces_output([mock_trace], "json")
        data = json.loads(result)

        # Sub-agent should be able to extract trace IDs
        trace_ids = [t["trace_id"] for t in data]
        assert "tr-test123" in trace_ids

    def test_get_trace_output_has_required_fields(self, mock_trace):
        """Test get_trace output has fields needed by sub-agents."""
        result = format_trace_detail(mock_trace)
        data = json.loads(result)

        # Required fields for trace_analyst
        assert "info" in data
        assert "spans" in data

        for span in data["spans"]:
            assert "span_id" in span
            assert "name" in span
            assert "span_type" in span
            assert "duration_ms" in span
            assert "parent_id" in span

    def test_llm_span_has_token_data(self, mock_trace):
        """Test LLM spans include token usage for context_engineer."""
        result = format_trace_detail(mock_trace)
        data = json.loads(result)

        llm_spans = [s for s in data["spans"] if s["span_type"] == "CHAT_MODEL"]
        assert len(llm_spans) > 0

        for span in llm_spans:
            assert "tokens" in span
            # At least one token count should be present
            assert (
                span["tokens"]["input"] is not None or
                span["tokens"]["output"] is not None or
                span["tokens"]["total"] is not None
            )

    def test_error_span_has_error_details(self, mock_error_trace):
        """Test error spans include error info for trace_analyst."""
        result = format_trace_detail(mock_error_trace)
        data = json.loads(result)

        error_spans = [s for s in data["spans"] if "error" in s]
        assert len(error_spans) > 0

        for span in error_spans:
            assert "message" in span["error"]
            assert span["error"]["message"]  # Not empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
