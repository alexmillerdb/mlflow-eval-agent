"""MCP tools for the MLflow Evaluation Agent."""

import json
from typing import Any

from claude_agent_sdk import tool

from .workspace import SharedWorkspace


def text_result(msg: str) -> dict:
    """Create a text content result for MCP tools."""
    return {"content": [{"type": "text", "text": msg}]}


def create_tools(workspace: SharedWorkspace) -> list:
    """Create MCP tools for inter-agent communication."""

    @tool("write_to_workspace", "Write analysis findings to shared workspace", {"key": str, "data": dict, "agent_name": str})
    async def write_to_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Write to shared workspace with validation."""
        _, msg = workspace.write(
            args.get("key", ""),
            args.get("data", {}),
            args.get("agent_name", "unknown")
        )
        return text_result(f"[Workspace] {msg}")

    @tool("read_from_workspace", "Read analysis findings from shared workspace", {"key": str})
    async def read_from_workspace_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Read from shared workspace."""
        key = args.get("key", "")
        data = workspace.read(key)

        if data is None:
            available = workspace.list_keys()
            return text_result(f"[Workspace] No data found for key: '{key}'\nAvailable keys: {available if available else '(none)'}")

        return text_result(json.dumps(data, indent=2, default=str))

    @tool("check_workspace_dependencies", "Check if required workspace entries exist", {"required_keys": list})
    async def check_workspace_dependencies_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Validate workspace dependencies before proceeding."""
        required = args.get("required_keys", [])
        all_present, missing = workspace.has_required_dependencies(required)

        if all_present:
            return text_result(f"[Workspace] All required dependencies present: {required}")
        return text_result(f"[Workspace] Missing dependencies: {missing}\nRun trace_analyst first to populate these entries.")

    return [write_to_workspace_tool, read_from_workspace_tool, check_workspace_dependencies_tool]


# =============================================================================
# TOOL NAME CONSTANTS
# =============================================================================

class MCPTools:
    """MLflow MCP Server tool names."""
    # Trace operations
    SEARCH_TRACES = "mcp__mlflow-mcp__search_traces"
    GET_TRACE = "mcp__mlflow-mcp__get_trace"
    # Tagging
    SET_TRACE_TAG = "mcp__mlflow-mcp__set_trace_tag"
    DELETE_TRACE_TAG = "mcp__mlflow-mcp__delete_trace_tag"
    # Assessments
    LOG_FEEDBACK = "mcp__mlflow-mcp__log_feedback"
    LOG_EXPECTATION = "mcp__mlflow-mcp__log_expectation"
    GET_ASSESSMENT = "mcp__mlflow-mcp__get_assessment"
    UPDATE_ASSESSMENT = "mcp__mlflow-mcp__update_assessment"
    # Experiments
    GET_EXPERIMENT = "mcp__mlflow-mcp__get_experiment"
    SEARCH_EXPERIMENTS = "mcp__mlflow-mcp__search_experiments"


class InternalTools:
    """Internal workspace tool names."""
    WRITE_TO_WORKSPACE = "mcp__mlflow-eval__write_to_workspace"
    READ_FROM_WORKSPACE = "mcp__mlflow-eval__read_from_workspace"
    CHECK_DEPENDENCIES = "mcp__mlflow-eval__check_workspace_dependencies"


class BuiltinTools:
    """Claude SDK built-in tools."""
    READ = "Read"
    BASH = "Bash"
    GLOB = "Glob"
    GREP = "Grep"
    SKILL = "Skill"
