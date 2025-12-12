"""MLflow client wrapper for custom tools.

Provides authenticated MlflowClient access using Databricks Config Profile
and formatting utilities for agent-friendly trace output.
"""

import json
import logging
from functools import lru_cache
from typing import Any, Optional

import mlflow
from mlflow.client import MlflowClient

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_mlflow_client() -> MlflowClient:
    """Get authenticated MLflow client using Databricks config profile.

    Authentication order (via Databricks SDK):
    1. DATABRICKS_CONFIG_PROFILE env var -> reads ~/.databrickscfg
    2. Default profile in ~/.databrickscfg
    3. Falls back to DATABRICKS_HOST + DATABRICKS_TOKEN if set

    Returns:
        Authenticated MlflowClient instance
    """
    mlflow.set_tracking_uri("databricks")
    return MlflowClient()


def clear_client_cache():
    """Clear the cached MLflow client (useful for testing)."""
    get_mlflow_client.cache_clear()


def format_traces_output(traces: list, output_format: str = "table") -> str:
    """Format trace list for agent-friendly output.

    Args:
        traces: List of Trace objects from MlflowClient.search_traces()
        output_format: "table" for markdown table, "json" for JSON

    Returns:
        Formatted string suitable for agent consumption
    """
    if not traces:
        return "No traces found."

    if output_format == "json":
        return json.dumps([{
            "trace_id": t.info.trace_id,
            "status": str(t.info.status),
            "execution_time_ms": t.info.execution_time_ms,
            "timestamp_ms": t.info.timestamp_ms,
        } for t in traces], indent=2)

    # Table format (default)
    lines = [
        "| Trace ID | Status | Duration (ms) | Timestamp |",
        "|----------|--------|---------------|-----------|"
    ]
    for t in traces[:50]:  # Limit to 50 for readability
        lines.append(
            f"| {t.info.trace_id} | {t.info.status} | "
            f"{t.info.execution_time_ms} | {t.info.timestamp_ms} |"
        )
    if len(traces) > 50:
        lines.append(f"\n... and {len(traces) - 50} more traces")

    return "\n".join(lines)


def format_trace_detail(trace) -> str:
    """Format single trace with spans for agent analysis.

    CRITICAL: Includes all data needed for agent improvement:
    - Token usage from LLM spans (for context_engineer)
    - Error messages from failed spans (for trace_analyst)
    - Inputs/outputs preview (for evaluation dataset building)

    Args:
        trace: Trace object from MlflowClient.get_trace()

    Returns:
        JSON string with trace info, assessments, and spans
    """
    info = {
        "trace_id": trace.info.trace_id,
        "status": str(trace.info.status),
        "execution_time_ms": trace.info.execution_time_ms,
        "timestamp_ms": trace.info.timestamp_ms,
    }

    # Include assessments if present (feedback/expectations)
    assessments = []
    if hasattr(trace.info, 'assessments') and trace.info.assessments:
        for a in trace.info.assessments:
            # source is an AssessmentSource object with source_type attribute
            source_type = None
            if hasattr(a, 'source') and a.source:
                source_type = str(a.source.source_type) if hasattr(a.source, 'source_type') else None
            assessments.append({
                "name": a.name,
                "value": a.value,
                "source_type": source_type,
                "rationale": a.rationale,
            })

    spans = []
    for span in trace.data.spans:
        span_data = _format_span(span)
        spans.append(span_data)

    return json.dumps({
        "info": info,
        "assessments": assessments,
        "spans": spans
    }, indent=2, default=str)


def _format_span(span) -> dict[str, Any]:
    """Format a single span with all relevant data for agent analysis.

    Args:
        span: Span object from trace.data.spans

    Returns:
        Dictionary with span data including tokens, errors, and previews
    """
    # Calculate duration
    duration_ms = 0.0
    if span.end_time_ns and span.start_time_ns:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6

    attrs = span.attributes or {}

    span_data = {
        "span_id": span.span_id,
        "name": span.name,
        "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
        "duration_ms": round(duration_ms, 2),
        "parent_id": span.parent_id,
    }

    # Extract token usage for LLM/CHAT_MODEL spans (needed by context_engineer)
    span_type_str = str(span.span_type) if span.span_type else ""
    if "LLM" in span_type_str or "CHAT_MODEL" in span_type_str:
        span_data["tokens"] = {
            "input": attrs.get("mlflow.chat_model.input_tokens"),
            "output": attrs.get("mlflow.chat_model.output_tokens"),
            "total": attrs.get("mlflow.chat_model.total_tokens"),
        }
        span_data["model"] = (
            attrs.get("mlflow.chat_model.model") or
            attrs.get("llm.model_name")
        )

    # Extract error info for failed spans (needed by trace_analyst)
    if span.status and hasattr(span.status, 'status_code'):
        status_code_str = str(span.status.status_code)
        if "ERROR" in status_code_str:
            error_info = {
                "message": span.status.description if span.status.description else "Unknown error",
            }
            # Check for exception events
            if span.events:
                for event in span.events:
                    if "exception" in event.name.lower():
                        error_info["exception"] = dict(event.attributes) if event.attributes else None
            span_data["error"] = error_info

    # Include inputs/outputs preview for dataset building
    if span.inputs:
        span_data["inputs_preview"] = _truncate_str(str(span.inputs), 500)
    if span.outputs:
        span_data["outputs_preview"] = _truncate_str(str(span.outputs), 500)

    return span_data


def _truncate_str(s: str, max_len: int) -> str:
    """Truncate string to max length with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def text_result(msg: str) -> dict[str, Any]:
    """Create a text content result for MCP tools.

    Args:
        msg: Message text to return

    Returns:
        MCP tool result format with text content
    """
    return {"content": [{"type": "text", "text": msg}]}
