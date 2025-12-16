#!/usr/bin/env python
"""Single trace analysis using MLflow 3 SDK.

Usage:
    python scripts/analyze_trace.py <trace-id>
    python scripts/analyze_trace.py <trace-id> --full    # Include full span data
    python scripts/analyze_trace.py <trace-id> --raw     # Raw trace JSON only

Environment variables (loaded from .env):
    DATABRICKS_HOST: Databricks workspace URL
    DATABRICKS_TOKEN: Databricks personal access token
    - OR -
    DATABRICKS_CONFIG_PROFILE: Named profile from ~/.databrickscfg

Outputs JSON report with:
- Summary (status, execution time, span count)
- Architecture detection (DSPy, LangGraph, RAG, Tool-Calling)
- Latency breakdown by span type
- Bottleneck detection
- Error analysis
- Tool/LLM usage patterns
- Recommendations

With --full flag, also includes:
- Full span hierarchy with inputs/outputs
- LLM messages and responses
- Tool call details

With --raw flag:
- Outputs raw trace data for direct analysis
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanStatusCode, SpanType, Trace


def setup_databricks_auth():
    """Configure Databricks authentication from .env or environment variables."""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    # Set MLflow tracking URI to Databricks
    mlflow.set_tracking_uri("databricks")

    # Check for authentication
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")

    if profile:
        # Use named profile from ~/.databrickscfg
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    elif token and host:
        # Use direct token authentication
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
    else:
        # Will use default authentication (databricks-cli config, etc.)
        pass


def get_trace(trace_id: str) -> Trace:
    """Fetch trace by ID using MLflow client."""
    client = MlflowClient()
    return client.get_trace(trace_id)


def analyze_span_hierarchy(trace: Trace) -> dict[str, Any]:
    """Analyze span hierarchy and structure."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    span_by_id = {s.span_id: s for s in spans}
    children = defaultdict(list)
    root_spans = []

    for span in spans:
        if span.parent_id is None:
            root_spans.append(span)
        else:
            children[span.parent_id].append(span)

    def build_tree(span, depth=0):
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        node = {
            "name": span.name,
            "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
            "duration_ms": round(duration_ms, 2),
            "depth": depth,
            "children": [],
        }
        for child in children.get(span.span_id, []):
            node["children"].append(build_tree(child, depth + 1))
        return node

    return {
        "root_count": len(root_spans),
        "total_spans": len(spans),
        "max_depth": max((build_tree(r)["depth"] for r in root_spans), default=0),
        "hierarchy": [build_tree(root) for root in root_spans],
    }


def latency_by_span_type(trace: Trace) -> dict[str, dict]:
    """Break down latency by span type (LLM, TOOL, RETRIEVER, etc.)."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    type_latencies = defaultdict(list)

    for span in spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        span_type = str(span.span_type) if span.span_type else "UNKNOWN"
        type_latencies[span_type].append({"name": span.name, "duration_ms": duration_ms})

    results = {}
    for span_type, items in type_latencies.items():
        durations = [i["duration_ms"] for i in items]
        results[span_type] = {
            "count": len(items),
            "total_ms": round(sum(durations), 2),
            "avg_ms": round(sum(durations) / len(durations), 2),
            "max_ms": round(max(durations), 2),
            "min_ms": round(min(durations), 2),
        }

    return results


def latency_by_component(trace: Trace) -> dict[str, dict]:
    """Break down latency by component name."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    component_latencies = defaultdict(list)

    for span in spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        component_latencies[span.name].append({"duration_ms": duration_ms})

    results = {}
    for component, items in component_latencies.items():
        durations = [i["duration_ms"] for i in items]
        results[component] = {
            "count": len(items),
            "total_ms": round(sum(durations), 2),
            "avg_ms": round(sum(durations) / len(durations), 2) if durations else 0,
        }

    return results


def find_bottlenecks(trace: Trace, top_n: int = 5) -> list[dict]:
    """Find the slowest spans in a trace."""
    spans = trace.data.spans if hasattr(trace, "data") else []
    exclude_patterns = ["forward", "predict", "root", "__init__"]

    span_timings = []
    for span in spans:
        span_name_lower = span.name.lower()
        if any(p in span_name_lower for p in exclude_patterns):
            continue

        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        span_timings.append(
            {
                "name": span.name,
                "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
                "duration_ms": round(duration_ms, 2),
                "span_id": span.span_id,
            }
        )

    span_timings.sort(key=lambda x: -x["duration_ms"])
    return span_timings[:top_n]


def detect_errors(trace: Trace) -> dict[str, list]:
    """Detect error patterns in a trace."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    errors = {"failed_spans": [], "exceptions": [], "empty_outputs": []}

    for span in spans:
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            errors["failed_spans"].append(
                {
                    "name": span.name,
                    "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
                    "error_message": span.status.description if span.status.description else "Unknown error",
                }
            )

        if hasattr(span, "events") and span.events:
            for event in span.events:
                if "exception" in event.name.lower():
                    errors["exceptions"].append(
                        {
                            "span_name": span.name,
                            "event": event.name,
                            "attributes": dict(event.attributes) if event.attributes else {},
                        }
                    )

        if span.outputs is None or span.outputs == {} or span.outputs == []:
            errors["empty_outputs"].append({"name": span.name, "span_type": str(span.span_type) if span.span_type else "UNKNOWN"})

    return errors


def analyze_tool_calls(trace: Trace) -> dict[str, Any]:
    """Analyze tool calls in a trace."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    tool_calls = []
    for span in tool_spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0

        tool_name = span.name
        if "." in tool_name:
            tool_name_short = tool_name.split(".")[-1]
        else:
            tool_name_short = tool_name

        success = True
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            success = False

        tool_calls.append(
            {
                "tool_name": tool_name_short,
                "full_name": span.name,
                "duration_ms": round(duration_ms, 2),
                "success": success,
            }
        )

    tool_stats = {}
    for tc in tool_calls:
        name = tc["tool_name"]
        if name not in tool_stats:
            tool_stats[name] = {"count": 0, "total_ms": 0, "successes": 0}
        tool_stats[name]["count"] += 1
        tool_stats[name]["total_ms"] += tc["duration_ms"]
        if tc["success"]:
            tool_stats[name]["successes"] += 1

    return {
        "total_tool_calls": len(tool_calls),
        "unique_tools": len(tool_stats),
        "calls": tool_calls,
        "stats": tool_stats,
    }


def analyze_llm_calls(trace: Trace) -> dict[str, Any]:
    """Analyze LLM calls in a trace."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    llm_spans = [s for s in spans if s.span_type in [SpanType.LLM, SpanType.CHAT_MODEL]]

    llm_calls = []
    for span in llm_spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0

        attributes = span.attributes or {}

        llm_calls.append(
            {
                "name": span.name,
                "duration_ms": round(duration_ms, 2),
                "model": attributes.get("mlflow.chat_model.model") or attributes.get("llm.model_name"),
                "input_tokens": attributes.get("mlflow.chat_model.input_tokens"),
                "output_tokens": attributes.get("mlflow.chat_model.output_tokens"),
                "total_tokens": attributes.get("mlflow.chat_model.total_tokens"),
            }
        )

    total_input = sum(c["input_tokens"] or 0 for c in llm_calls)
    total_output = sum(c["output_tokens"] or 0 for c in llm_calls)
    total_latency = sum(c["duration_ms"] for c in llm_calls)

    return {
        "total_llm_calls": len(llm_calls),
        "total_latency_ms": round(total_latency, 2),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "calls": llm_calls,
    }


def detect_architecture(trace: Trace) -> dict[str, Any]:
    """Detect agent architecture from trace patterns."""
    spans = trace.data.spans if hasattr(trace, "data") else []
    span_names = [s.name.lower() for s in spans]
    span_types = [s.span_type for s in spans]

    indicators = {
        "dspy_multi_agent": any(p in " ".join(span_names) for p in ["classifier", "rewriter", "gatherer", "executor"]),
        "langgraph": any(p in " ".join(span_names) for p in ["langgraph", "graph", "node", "state"]),
        "rag": SpanType.RETRIEVER in span_types,
        "tool_calling": SpanType.TOOL in span_types,
        "simple_chat": len(set(span_types)) <= 2 and SpanType.CHAT_MODEL in span_types,
    }

    if indicators["dspy_multi_agent"]:
        arch_type = "dspy_multi_agent"
    elif indicators["langgraph"]:
        arch_type = "langgraph"
    elif indicators["rag"] and indicators["tool_calling"]:
        arch_type = "rag_with_tools"
    elif indicators["rag"]:
        arch_type = "rag"
    elif indicators["tool_calling"]:
        arch_type = "tool_calling"
    else:
        arch_type = "simple_chat"

    span_type_distribution = {}
    for st in set(span_types):
        span_type_distribution[str(st)] = sum(1 for s in spans if s.span_type == st)

    return {
        "architecture": arch_type,
        "indicators": indicators,
        "span_type_distribution": span_type_distribution,
    }


def extract_span_details(trace: Trace) -> list[dict]:
    """Extract detailed span data including inputs/outputs."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    span_details = []
    for span in spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0

        detail = {
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "name": span.name,
            "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
            "duration_ms": round(duration_ms, 2),
            "status": str(span.status.status_code) if span.status else "UNKNOWN",
            "inputs": span.inputs,
            "outputs": span.outputs,
            "attributes": dict(span.attributes) if span.attributes else {},
        }

        # Add error info if present
        if span.status and span.status.description:
            detail["error_message"] = span.status.description

        span_details.append(detail)

    return span_details


def extract_llm_messages(trace: Trace) -> list[dict]:
    """Extract LLM call messages and responses."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    llm_calls = []
    for span in spans:
        if span.span_type not in [SpanType.LLM, SpanType.CHAT_MODEL]:
            continue

        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0
        attributes = span.attributes or {}

        call = {
            "span_id": span.span_id,
            "name": span.name,
            "duration_ms": round(duration_ms, 2),
            "model": attributes.get("mlflow.chat_model.model") or attributes.get("llm.model_name"),
            "input_tokens": attributes.get("mlflow.chat_model.input_tokens"),
            "output_tokens": attributes.get("mlflow.chat_model.output_tokens"),
        }

        # Extract messages from inputs
        if span.inputs:
            if isinstance(span.inputs, dict):
                call["messages"] = span.inputs.get("messages", span.inputs)
            else:
                call["messages"] = span.inputs

        # Extract response from outputs
        if span.outputs:
            call["response"] = span.outputs

        llm_calls.append(call)

    return llm_calls


def extract_tool_details(trace: Trace) -> list[dict]:
    """Extract tool call details with inputs/outputs."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    tool_calls = []
    for span in spans:
        if span.span_type != SpanType.TOOL:
            continue

        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6 if span.end_time_ns else 0

        success = True
        error_msg = None
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            success = False
            error_msg = span.status.description

        tool_calls.append(
            {
                "span_id": span.span_id,
                "name": span.name,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "inputs": span.inputs,
                "outputs": span.outputs,
                "error": error_msg,
            }
        )

    return tool_calls


def extract_raw_trace(trace: Trace) -> dict:
    """Extract raw trace data for direct analysis."""
    spans = trace.data.spans if hasattr(trace, "data") else []

    return {
        "trace_id": trace.info.trace_id,
        "status": str(trace.info.status),
        "execution_time_ms": trace.info.execution_time_ms,
        "timestamp_ms": trace.info.timestamp_ms,
        "tags": dict(trace.info.tags) if trace.info.tags else {},
        "spans": [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "span_type": str(s.span_type) if s.span_type else None,
                "start_time_ns": s.start_time_ns,
                "end_time_ns": s.end_time_ns,
                "duration_ms": round((s.end_time_ns - s.start_time_ns) / 1e6, 2) if s.end_time_ns else 0,
                "status": str(s.status.status_code) if s.status else None,
                "status_message": s.status.description if s.status else None,
                "inputs": s.inputs,
                "outputs": s.outputs,
                "attributes": dict(s.attributes) if s.attributes else {},
                "events": [{"name": e.name, "attributes": dict(e.attributes) if e.attributes else {}} for e in (s.events or [])],
            }
            for s in spans
        ],
    }


def generate_recommendations(bottlenecks: list, errors: dict, llm_analysis: dict, total_ms: float) -> list[str]:
    """Generate actionable recommendations from analysis."""
    recommendations = []

    if bottlenecks and total_ms > 0 and bottlenecks[0]["duration_ms"] > total_ms * 0.5:
        b = bottlenecks[0]
        pct = bottlenecks[0]["duration_ms"] / total_ms * 100
        recommendations.append(f"BOTTLENECK: '{b['name']}' takes {pct:.0f}% of total time. Consider optimizing this component.")

    if llm_analysis["total_llm_calls"] > 5:
        recommendations.append(
            f"HIGH LLM CALLS: {llm_analysis['total_llm_calls']} LLM calls detected. Consider batching or reducing calls."
        )

    if errors["failed_spans"]:
        failed_names = [e["name"] for e in errors["failed_spans"][:3]]
        recommendations.append(f"ERRORS: {len(errors['failed_spans'])} failed spans detected. Review: {failed_names}")

    if not recommendations:
        recommendations.append("No major issues detected. Trace looks healthy.")

    return recommendations


def generate_report(trace: Trace, include_full: bool = False) -> dict[str, Any]:
    """Generate comprehensive trace analysis report.

    Args:
        trace: MLflow trace to analyze
        include_full: If True, include full span/LLM/tool details with inputs/outputs
    """
    hierarchy = analyze_span_hierarchy(trace)
    latency_types = latency_by_span_type(trace)
    bottlenecks = find_bottlenecks(trace, top_n=5)
    errors = detect_errors(trace)
    tool_analysis = analyze_tool_calls(trace)
    llm_analysis = analyze_llm_calls(trace)
    architecture = detect_architecture(trace)

    spans = trace.data.spans if hasattr(trace, "data") else []
    root_spans = [s for s in spans if s.parent_id is None]
    total_ms = 0
    if root_spans:
        root = root_spans[0]
        total_ms = (root.end_time_ns - root.start_time_ns) / 1e6 if root.end_time_ns else 0

    recommendations = generate_recommendations(bottlenecks, errors, llm_analysis, total_ms)

    report = {
        "summary": {
            "trace_id": trace.info.trace_id,
            "status": str(trace.info.status),
            "total_duration_ms": round(total_ms, 2),
            "total_spans": len(spans),
        },
        "architecture": architecture,
        "hierarchy": {
            "root_count": hierarchy["root_count"],
            "total_spans": hierarchy["total_spans"],
            "max_depth": hierarchy["max_depth"],
        },
        "latency_by_type": latency_types,
        "bottlenecks": bottlenecks,
        "errors": {
            "failed_spans_count": len(errors["failed_spans"]),
            "exceptions_count": len(errors["exceptions"]),
            "failed_spans": errors["failed_spans"],
        },
        "tool_calls": {
            "total": tool_analysis["total_tool_calls"],
            "unique_tools": tool_analysis["unique_tools"],
            "stats": tool_analysis["stats"],
        },
        "llm_calls": {
            "total": llm_analysis["total_llm_calls"],
            "total_latency_ms": llm_analysis["total_latency_ms"],
            "total_input_tokens": llm_analysis["total_input_tokens"],
            "total_output_tokens": llm_analysis["total_output_tokens"],
        },
        "recommendations": recommendations,
    }

    # Include full details if requested
    if include_full:
        report["full_data"] = {
            "spans": extract_span_details(trace),
            "llm_messages": extract_llm_messages(trace),
            "tool_details": extract_tool_details(trace),
        }

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace-id> [--full | --raw]", file=sys.stderr)
        print("  --full: Include full span data with inputs/outputs", file=sys.stderr)
        print("  --raw:  Output raw trace JSON only", file=sys.stderr)
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    trace_id = None
    include_full = False
    raw_mode = False

    for arg in args:
        if arg == "--full":
            include_full = True
        elif arg == "--raw":
            raw_mode = True
        elif not arg.startswith("-"):
            trace_id = arg.strip()

    if not trace_id:
        print("Error: trace-id is required", file=sys.stderr)
        sys.exit(1)

    try:
        # Setup Databricks authentication
        setup_databricks_auth()

        trace = get_trace(trace_id)

        if raw_mode:
            # Output raw trace data
            report = extract_raw_trace(trace)
        else:
            # Output analysis report
            report = generate_report(trace, include_full=include_full)

        print(json.dumps(report, indent=2))
    except Exception as e:
        error_report = {"error": str(e), "trace_id": trace_id}
        print(json.dumps(error_report, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
