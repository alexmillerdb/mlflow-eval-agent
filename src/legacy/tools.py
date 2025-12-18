"""MCP tools for the MLflow Evaluation Agent.

Includes:
- Workspace tools for inter-agent communication
- MLflow tools for trace analysis (replaces external MCP server)
- Tool result compression for large responses
"""

import json
import logging
from typing import Any, Optional

import mlflow
from claude_agent_sdk import tool

from .workspace import SharedWorkspace
from .mlflow_client import (
    get_mlflow_client,
    format_traces_output,
    format_trace_detail,
    text_result,
)
from .tool_compression import compress_tool_result, OutputMode, get_output_mode

logger = logging.getLogger(__name__)


def create_workspace_tools(workspace: SharedWorkspace) -> list:
    """Create workspace tools for inter-agent communication."""

    @tool(
        "write_to_workspace",
        "Write analysis findings to shared workspace. Data must be valid JSON matching the schema for the key.",
        {"key": str, "data": Any, "agent_name": str}
    )
    async def write_to_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Write to shared workspace with Pydantic validation."""
        key = args.get("key", "")
        data = args.get("data")
        agent_name = args.get("agent_name", "unknown")

        if data is None:
            return text_result(f"[Workspace] Error: 'data' parameter is required for key '{key}'")

        success, msg = workspace.write(key, data, agent_name)

        if success:
            return text_result(f"[Workspace] {msg}")
        else:
            return text_result(f"[Workspace] Validation error: {msg}")

    @tool("read_from_workspace", "Read analysis findings from shared workspace", {"key": str})
    async def read_from_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Read from shared workspace."""
        key = args.get("key", "")
        data = workspace.read(key)

        if data is None:
            available = workspace.list_keys()
            return text_result(
                f"[Workspace] No data found for key: '{key}'\n"
                f"Available keys: {available if available else '(none)'}"
            )

        return text_result(json.dumps(data, indent=2, default=str))

    @tool("check_workspace_dependencies", "Check if required workspace entries exist", {"required_keys": list})
    async def check_workspace_dependencies_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Validate workspace dependencies before proceeding."""
        required = args.get("required_keys", [])
        all_present, missing = workspace.has_required_dependencies(required)

        if all_present:
            return text_result(f"[Workspace] All required dependencies present: {required}")
        return text_result(
            f"[Workspace] Missing dependencies: {missing}\n"
            "Run trace_analyst first to populate these entries."
        )

    return [write_to_workspace_tool, read_from_workspace_tool, check_workspace_dependencies_tool]


def create_mlflow_tools() -> list:
    """Create MLflow tools for trace analysis.

    These tools replace the external mlflow-mcp server with in-process
    tools using the MLflow Python client.
    """

    @tool(
        "search_traces",
        "Search for traces in an experiment with filters. Returns trace IDs, status, and timing. Provide either experiment_id (numeric) OR experiment_name (string), at least one is required.",
        {
            "experiment_id": str,
            "experiment_name": str,
            "filter_string": str,
            "max_results": int,
            "output_format": str,
        }
    )
    async def search_traces_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Search traces in an experiment.

        Args:
            experiment_id: MLflow experiment ID (numeric string, e.g. '159502977489049')
            experiment_name: MLflow experiment name (string, e.g. '/Users/user@example.com/my-experiment')
            filter_string: Filter like "status = 'OK'" or "timestamp_ms > 1700000000"
            max_results: Maximum traces to return (default 100)
            output_format: "table" or "json" (default "table")

        Note: Provide either experiment_id OR experiment_name, at least one is required.
        """
        try:
            client = get_mlflow_client()
            experiment_id = args.get("experiment_id", "")
            experiment_name = args.get("experiment_name", "")
            filter_string = args.get("filter_string")
            max_results = args.get("max_results", 100)
            output_format = args.get("output_format", "table")

            # Require at least one of experiment_id or experiment_name
            if not experiment_id and not experiment_name:
                return text_result(
                    "[MLflow] Error: Either experiment_id or experiment_name is required. "
                    "Use the experiment ID from the system prompt, or provide the experiment name."
                )

            # If experiment_name provided, look up the ID
            if experiment_name and not experiment_id:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    return text_result(
                        f"[MLflow] Error: No experiment found with name '{experiment_name}'. "
                        "Check the experiment name or use experiment_id instead."
                    )
                experiment_id = experiment.experiment_id
                logger.info(f"Resolved experiment name '{experiment_name}' to ID '{experiment_id}'")

            # Validate experiment_id is numeric
            if experiment_id and not experiment_id.isdigit():
                return text_result(
                    f"[MLflow] Error: experiment_id must be numeric, got '{experiment_id}'. "
                    "Use experiment_name parameter for string names."
                )

            traces = client.search_traces(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )

            return text_result(format_traces_output(traces, output_format))

        except Exception as e:
            logger.exception("Error searching traces")
            return text_result(f"[MLflow] Error searching traces: {str(e)}")

    @tool(
        "get_trace",
        "Get detailed trace by ID including all spans with token usage, errors, and I/O. "
        "Use output_mode='full' for complete data, 'summary' (default) for adaptive truncation, "
        "or 'preview' for minimal 500-char truncation.",
        {"trace_id": str, "output_mode": str}
    )
    async def get_trace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Get detailed trace by ID.

        Returns summary with file reference for large traces (unless output_mode='full').
        Full data is saved to .claude/cache/tool_results/<trace_id>.json
        and can be accessed via the Read tool when needed.

        Args:
            trace_id: The trace ID to fetch
            output_mode: "preview" (500 chars), "summary" (adaptive, default), or "full" (no limit)
                         Can also be set via TRACE_OUTPUT_MODE env var.

        Summary includes:
        - info: trace_id, status, execution_time_ms
        - span count and error count
        - LLM calls and token usage
        - Top bottlenecks
        """
        try:
            client = get_mlflow_client()
            trace_id = args.get("trace_id", "")
            output_mode_str = args.get("output_mode", "").lower()

            if not trace_id:
                return text_result("[MLflow] Error: trace_id is required")

            # Determine output mode
            if output_mode_str:
                try:
                    mode = OutputMode(output_mode_str)
                except ValueError:
                    return text_result(
                        f"[MLflow] Error: Invalid output_mode '{output_mode_str}'. "
                        "Use 'preview', 'summary', or 'full'."
                    )
            else:
                mode = get_output_mode()  # From env var or default

            trace = client.get_trace(trace_id=trace_id)
            result = text_result(format_trace_detail(trace, output_mode=mode.value))

            # Compress large results: write to file, return summary (respects mode)
            return compress_tool_result("get_trace", result, trace_id=trace_id, mode=mode)

        except Exception as e:
            logger.exception(f"Error getting trace {args.get('trace_id')}")
            return text_result(f"[MLflow] Error getting trace: {str(e)}")

    @tool(
        "set_trace_tag",
        "Set a tag on a trace for later filtering (e.g., tag error traces for eval dataset).",
        {"trace_id": str, "key": str, "value": str}
    )
    async def set_trace_tag_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Set a tag on a trace."""
        try:
            trace_id = args.get("trace_id", "")
            key = args.get("key", "")
            value = args.get("value", "")

            if not trace_id or not key:
                return text_result("[MLflow] Error: trace_id and key are required")

            mlflow.set_trace_tag(trace_id, key, value)
            return text_result(f"[MLflow] Tag '{key}={value}' set on trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error setting trace tag")
            return text_result(f"[MLflow] Error setting trace tag: {str(e)}")

    @tool(
        "delete_trace_tag",
        "Delete a tag from a trace.",
        {"trace_id": str, "key": str}
    )
    async def delete_trace_tag_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Delete a tag from a trace."""
        try:
            trace_id = args.get("trace_id", "")
            key = args.get("key", "")

            if not trace_id or not key:
                return text_result("[MLflow] Error: trace_id and key are required")

            mlflow.delete_trace_tag(trace_id, key)
            return text_result(f"[MLflow] Tag '{key}' deleted from trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error deleting trace tag")
            return text_result(f"[MLflow] Error deleting trace tag: {str(e)}")

    @tool(
        "log_feedback",
        "Log feedback/assessment to a trace (e.g., bottleneck_detected, quality_score).",
        {
            "trace_id": str,
            "name": str,
            "value": str,
            "source_type": str,
            "rationale": str,
        }
    )
    async def log_feedback_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Log feedback assessment to a trace.

        Args:
            trace_id: Trace to attach feedback to
            name: Feedback name (e.g., "bottleneck_detected", "quality_score")
            value: Feedback value (string, number, or JSON)
            source_type: "HUMAN", "LLM_JUDGE", or "CODE" (default "CODE")
            rationale: Explanation for the feedback
        """
        from mlflow.entities.assessment_source import AssessmentSource

        try:
            trace_id = args.get("trace_id", "")
            name = args.get("name", "")
            value = args.get("value")
            source_type = args.get("source_type", "CODE")
            rationale = args.get("rationale")

            if not trace_id or not name:
                return text_result("[MLflow] Error: trace_id and name are required")

            # Use module-level mlflow.log_feedback (not MlflowClient method)
            source = AssessmentSource(source_type=source_type)
            mlflow.log_feedback(
                trace_id=trace_id,
                name=name,
                value=value,
                source=source,
                rationale=rationale
            )
            return text_result(f"[MLflow] Feedback '{name}' logged to trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error logging feedback")
            return text_result(f"[MLflow] Error logging feedback: {str(e)}")

    @tool(
        "log_expectation",
        "Log expected output/ground truth to a trace for evaluation.",
        {"trace_id": str, "name": str, "value": str}
    )
    async def log_expectation_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Log expectation (ground truth) to a trace.

        Args:
            trace_id: Trace to attach expectation to
            name: Expectation name (e.g., "expected_output", "expected_facts")
            value: Expected value (string or JSON)
        """
        try:
            trace_id = args.get("trace_id", "")
            name = args.get("name", "")
            value = args.get("value")

            if not trace_id or not name:
                return text_result("[MLflow] Error: trace_id and name are required")

            # Use module-level mlflow.log_expectation (not MlflowClient method)
            mlflow.log_expectation(
                trace_id=trace_id,
                name=name,
                value=value
            )
            return text_result(f"[MLflow] Expectation '{name}' logged to trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error logging expectation")
            return text_result(f"[MLflow] Error logging expectation: {str(e)}")

    @tool(
        "get_assessment",
        "Get an assessment (feedback or expectation) from a trace.",
        {"trace_id": str, "assessment_name": str}
    )
    async def get_assessment_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Get an assessment from a trace.

        Args:
            trace_id: Trace containing the assessment
            assessment_name: Name of the assessment to retrieve
        """
        try:
            client = get_mlflow_client()
            trace_id = args.get("trace_id", "")
            assessment_name = args.get("assessment_name", "")

            if not trace_id or not assessment_name:
                return text_result("[MLflow] Error: trace_id and assessment_name are required")

            # Get the trace and find the assessment
            trace = client.get_trace(trace_id=trace_id)

            if not hasattr(trace.info, 'assessments') or not trace.info.assessments:
                return text_result(f"[MLflow] No assessments found on trace {trace_id}")

            for assessment in trace.info.assessments:
                if assessment.name == assessment_name:
                    # source is an AssessmentSource object with source_type attribute
                    source_type = None
                    if hasattr(assessment, 'source') and assessment.source:
                        source_type = str(assessment.source.source_type) if hasattr(assessment.source, 'source_type') else None
                    result = {
                        "name": assessment.name,
                        "value": assessment.value,
                        "source_type": source_type,
                        "rationale": assessment.rationale,
                    }
                    return text_result(json.dumps(result, indent=2, default=str))

            return text_result(f"[MLflow] Assessment '{assessment_name}' not found on trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error getting assessment")
            return text_result(f"[MLflow] Error getting assessment: {str(e)}")

    @tool(
        "update_assessment",
        "Update an existing assessment's value or rationale.",
        {"trace_id": str, "assessment_name": str, "value": str, "rationale": str}
    )
    async def update_assessment_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Update an existing assessment.

        Args:
            trace_id: Trace containing the assessment
            assessment_name: Name of the assessment to update
            value: New value (optional)
            rationale: New rationale (optional)
        """
        try:
            client = get_mlflow_client()
            trace_id = args.get("trace_id", "")
            assessment_name = args.get("assessment_name", "")
            value = args.get("value")
            rationale = args.get("rationale")

            if not trace_id or not assessment_name:
                return text_result("[MLflow] Error: trace_id and assessment_name are required")

            # Find assessment ID first
            trace = client.get_trace(trace_id=trace_id)

            if not hasattr(trace.info, 'assessments') or not trace.info.assessments:
                return text_result(f"[MLflow] No assessments found on trace {trace_id}")

            assessment_id = None
            for assessment in trace.info.assessments:
                if assessment.name == assessment_name:
                    assessment_id = assessment.assessment_id
                    break

            if not assessment_id:
                return text_result(f"[MLflow] Assessment '{assessment_name}' not found on trace {trace_id}")

            # Build update kwargs
            update_kwargs = {"trace_id": trace_id, "assessment_id": assessment_id}
            if value is not None:
                update_kwargs["value"] = value
            if rationale is not None:
                update_kwargs["rationale"] = rationale

            client.update_assessment(**update_kwargs)
            return text_result(f"[MLflow] Assessment '{assessment_name}' updated on trace {trace_id}")

        except Exception as e:
            logger.exception(f"Error updating assessment")
            return text_result(f"[MLflow] Error updating assessment: {str(e)}")

    return [
        search_traces_tool,
        get_trace_tool,
        set_trace_tag_tool,
        delete_trace_tag_tool,
        log_feedback_tool,
        log_expectation_tool,
        get_assessment_tool,
        update_assessment_tool,
    ]


def create_tools(workspace: SharedWorkspace) -> list:
    """Create all MCP tools (workspace + MLflow).

    Args:
        workspace: SharedWorkspace instance for inter-agent communication

    Returns:
        List of all tool functions for the MCP server
    """
    workspace_tools = create_workspace_tools(workspace)
    mlflow_tools = create_mlflow_tools()
    return workspace_tools + mlflow_tools


# =============================================================================
# TOOL NAME CONSTANTS
# =============================================================================

# MCP server name for all custom tools
MCP_SERVER_NAME = "mlflow-eval"


class MCPTools:
    """MLflow tool names (in-process, replaces external mlflow-mcp server)."""
    # Trace operations
    SEARCH_TRACES = f"mcp__{MCP_SERVER_NAME}__search_traces"
    GET_TRACE = f"mcp__{MCP_SERVER_NAME}__get_trace"
    # Tagging
    SET_TRACE_TAG = f"mcp__{MCP_SERVER_NAME}__set_trace_tag"
    DELETE_TRACE_TAG = f"mcp__{MCP_SERVER_NAME}__delete_trace_tag"
    # Assessments
    LOG_FEEDBACK = f"mcp__{MCP_SERVER_NAME}__log_feedback"
    LOG_EXPECTATION = f"mcp__{MCP_SERVER_NAME}__log_expectation"
    GET_ASSESSMENT = f"mcp__{MCP_SERVER_NAME}__get_assessment"
    UPDATE_ASSESSMENT = f"mcp__{MCP_SERVER_NAME}__update_assessment"


class InternalTools:
    """Internal workspace tool names."""
    WRITE_TO_WORKSPACE = f"mcp__{MCP_SERVER_NAME}__write_to_workspace"
    READ_FROM_WORKSPACE = f"mcp__{MCP_SERVER_NAME}__read_from_workspace"
    CHECK_DEPENDENCIES = f"mcp__{MCP_SERVER_NAME}__check_workspace_dependencies"


class BuiltinTools:
    """Claude SDK built-in tools."""
    READ = "Read"
    BASH = "Bash"
    GLOB = "Glob"
    GREP = "Grep"
    SKILL = "Skill"
