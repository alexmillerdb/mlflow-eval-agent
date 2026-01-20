"""Simplified MCP tools for the MLflow Evaluation Agent.

Reduced from 11 tools to 3 following "one-agent-one-tool" principle.
Each tool handles a category of operations rather than individual actions.
"""

import json
import logging
from typing import Any

import mlflow
from claude_agent_sdk import tool

from . import mlflow_ops
from .mlflow_ops import record_tool_call

logger = logging.getLogger(__name__)


def create_tools() -> list:
    """Create the 3 essential MCP tools.

    Returns:
        List of tool functions for the MCP server
    """

    @mlflow.trace(name="tool_mlflow_query", span_type="TOOL")
    @tool(
        "mlflow_query",
        "Query MLflow data. Operations: 'search' (find traces), 'get' (trace details), 'assessment' (get feedback), 'search_runs' (find eval/training runs), 'get_run' (run details). Use detail_level for trace 'get'.",
        {
            "operation": str,
            "experiment_id": str,
            "trace_id": str,
            "run_id": str,
            "filter_string": str,
            "max_results": int,
            "assessment_name": str,
            "detail_level": str,
        }
    )
    async def mlflow_query_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Query MLflow for trace and run data.

        Args:
            operation: "search", "get", "assessment", "search_runs", or "get_run"
            experiment_id: Required for search/search_runs
            trace_id: Required for get/assessment (trace operations)
            run_id: Required for get_run (run operations)
            filter_string: Optional filter for search/search_runs
                - For traces: "status = 'OK'", "name", "timestamp_ms", etc.
                - For runs: "attributes.run_name = '...'", "metrics.x > 0.9", etc.
            max_results: Max results for search operations (default 100)
            assessment_name: Required for assessment operation
            detail_level: For trace 'get' - "summary", "analysis", or "full"
        """
        try:
            operation = args.get("operation", "").lower()

            if operation == "search":
                experiment_id = args.get("experiment_id", "")
                if not experiment_id:
                    return mlflow_ops.text_result("[MLflow] Error: experiment_id required for search")

                traces = mlflow_ops.search_traces(
                    experiment_id=experiment_id,
                    filter_string=args.get("filter_string"),
                    max_results=args.get("max_results", 100)
                )
                result = mlflow_ops.text_result(mlflow_ops.format_traces_table(traces))
                record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                return result

            elif operation == "search_runs":
                experiment_id = args.get("experiment_id", "")
                if not experiment_id:
                    return mlflow_ops.text_result("[MLflow] Error: experiment_id required for search_runs")

                runs = mlflow_ops.search_runs(
                    experiment_id=experiment_id,
                    filter_string=args.get("filter_string"),
                    max_results=args.get("max_results", 100)
                )
                result = mlflow_ops.text_result(mlflow_ops.format_runs_table(runs))
                record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                return result

            elif operation == "get":
                trace_id = args.get("trace_id", "")
                if not trace_id:
                    return mlflow_ops.text_result("[MLflow] Error: trace_id required for get")

                detail_level = args.get("detail_level", "summary")
                trace = mlflow_ops.get_trace(trace_id, detail_level=detail_level)
                result = mlflow_ops.text_result(json.dumps(trace, indent=2, default=str))
                record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                return result

            elif operation == "get_run":
                run_id = args.get("run_id", "")
                if not run_id:
                    return mlflow_ops.text_result("[MLflow] Error: run_id required for get_run")

                run = mlflow_ops.get_run(run_id)
                result = mlflow_ops.text_result(json.dumps(run, indent=2, default=str))
                record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                return result

            elif operation == "assessment":
                trace_id = args.get("trace_id", "")
                assessment_name = args.get("assessment_name", "")
                if not trace_id or not assessment_name:
                    return mlflow_ops.text_result("[MLflow] Error: trace_id and assessment_name required")

                trace = mlflow_ops.get_trace(trace_id)
                for a in trace.get("assessments", []):
                    if a["name"] == assessment_name:
                        result = mlflow_ops.text_result(json.dumps(a, indent=2))
                        record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                        return result
                result = mlflow_ops.text_result(f"[MLflow] Assessment '{assessment_name}' not found")
                record_tool_call("mlflow_query", len(str(args)), len(str(result)))
                return result

            else:
                return mlflow_ops.text_result(
                    f"[MLflow] Unknown operation '{operation}'. Use 'search', 'get', 'assessment', 'search_runs', or 'get_run'."
                )

        except Exception as e:
            logger.exception(f"Error in mlflow_query: {operation}")
            return mlflow_ops.text_result(f"[MLflow] Error: {str(e)}")

    @mlflow.trace(name="tool_mlflow_annotate", span_type="TOOL")
    @tool(
        "mlflow_annotate",
        "Annotate traces. Operations: 'tag' (set tag), 'feedback' (log assessment), 'expectation' (log ground truth).",
        {
            "operation": str,
            "trace_id": str,
            "key": str,
            "value": str,
            "name": str,
            "rationale": str,
        }
    )
    async def mlflow_annotate_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Annotate MLflow traces with metadata.

        Args:
            operation: "tag", "feedback", or "expectation"
            trace_id: Required - trace to annotate
            key: Tag key (for tag operation)
            value: Tag/feedback/expectation value
            name: Feedback/expectation name
            rationale: Optional explanation for feedback
        """
        try:
            operation = args.get("operation", "").lower()
            trace_id = args.get("trace_id", "")

            if not trace_id:
                return mlflow_ops.text_result("[MLflow] Error: trace_id required")

            if operation == "tag":
                key = args.get("key", "")
                value = args.get("value", "")
                if not key:
                    return mlflow_ops.text_result("[MLflow] Error: key required for tag")
                mlflow_ops.set_tag(trace_id, key, value)
                result = mlflow_ops.text_result(f"[MLflow] Tag '{key}={value}' set on {trace_id}")
                record_tool_call("mlflow_annotate", len(str(args)), len(str(result)))
                return result

            elif operation == "feedback":
                name = args.get("name", "")
                value = args.get("value")
                if not name:
                    return mlflow_ops.text_result("[MLflow] Error: name required for feedback")
                mlflow_ops.log_feedback(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    rationale=args.get("rationale")
                )
                result = mlflow_ops.text_result(f"[MLflow] Feedback '{name}' logged to {trace_id}")
                record_tool_call("mlflow_annotate", len(str(args)), len(str(result)))
                return result

            elif operation == "expectation":
                name = args.get("name", "")
                value = args.get("value")
                if not name:
                    return mlflow_ops.text_result("[MLflow] Error: name required for expectation")
                mlflow_ops.log_expectation(trace_id=trace_id, name=name, value=value)
                result = mlflow_ops.text_result(f"[MLflow] Expectation '{name}' logged to {trace_id}")
                record_tool_call("mlflow_annotate", len(str(args)), len(str(result)))
                return result

            else:
                return mlflow_ops.text_result(
                    f"[MLflow] Unknown operation '{operation}'. Use 'tag', 'feedback', or 'expectation'."
                )

        except Exception as e:
            logger.exception(f"Error in mlflow_annotate: {operation}")
            return mlflow_ops.text_result(f"[MLflow] Error: {str(e)}")

    @mlflow.trace(name="tool_save_findings", span_type="TOOL")
    @tool(
        "save_findings",
        "Save analysis findings to persistent JSON file. Use keys like 'analysis', 'recommendations', 'eval_cases'.",
        {"key": str, "data": Any}
    )
    async def save_findings_tool(args: dict[str, Any]) -> dict[str, Any]:
        """Save findings to file-based state.

        Args:
            key: State key (e.g., "analysis", "recommendations")
            data: Data to save (will be JSON serialized)

        State files are saved to .claude/state/<key>.json
        and can be read later with the Read tool.
        """
        try:
            key = args.get("key", "")
            data = args.get("data")

            if not key:
                return mlflow_ops.text_result("[State] Error: key required")
            if data is None:
                return mlflow_ops.text_result("[State] Error: data required")

            # Handle case where data is passed as JSON string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON

            path = mlflow_ops.save_state(key, data)
            result = mlflow_ops.text_result(f"[State] Saved to {path}")
            record_tool_call("save_findings", len(str(args)), len(str(result)))
            return result

        except Exception as e:
            logger.exception("Error saving findings")
            return mlflow_ops.text_result(f"[State] Error: {str(e)}")

    return [mlflow_query_tool, mlflow_annotate_tool, save_findings_tool]


# =============================================================================
# TOOL NAME CONSTANTS
# =============================================================================

MCP_SERVER_NAME = "mlflow-eval"


class MCPTools:
    """Tool names for the simplified MCP server."""
    MLFLOW_QUERY = f"mcp__{MCP_SERVER_NAME}__mlflow_query"
    MLFLOW_ANNOTATE = f"mcp__{MCP_SERVER_NAME}__mlflow_annotate"
    SAVE_FINDINGS = f"mcp__{MCP_SERVER_NAME}__save_findings"


class BuiltinTools:
    """Claude SDK built-in tools."""
    READ = "Read"
    BASH = "Bash"
    GLOB = "Glob"
    GREP = "Grep"
    SKILL = "Skill"
